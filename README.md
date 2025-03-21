# prime-gaps-cuda

A highly optimized GPU-accelerated program for finding large prime gaps.

# Setup
This code is linux-only. However, if you have Windows, it works great on WSL. However, since the code is GPU-based, you will need a GPU. Specifically, you'll need a CUDA-capable NVIDIA GPU.
The code in this repo is designed to be easy to plug-and-go. There are only two things you have to install before you are ready to run the code: CUDA and python.

## CUDA setup
To install CUDA, follow the instructions [here](https://developer.nvidia.com/cuda-downloads). If you want more technical details about the installation, you can go to [this page](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#package-manager-installation), specifically section 3.

To verify that the CUDA installation worked, run `nvcc --version` in the terminal. If the terminal cannot find `nvcc` even after installing CUDA, then you can add it to the PATH by doing these steps:
- Find the folder in your filesystem that CUDA was installed to: it usually looks something like `/usr/lib/cuda/` or `/usr/local/cuda-12.0/`.
- Run these commands (replacing the folders in them with your actual CUDA folder):
  1. `echo 'export PATH=/usr/lib/cuda/bin:$PATH' >> ~/.bashrc`
  2. `echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc`
  3. `source ~/.bashrc​`

Note: Although most varieties of Linux come with a C compiler by default, you might need to install it manually. If `gcc --version` gives an error, then you can install `gcc` manually by running `sudo apt update` followed by `sudo apt install build-essential`.

## Python setup
There are a few ways you can get a Python environment set up, but here is a simple one. Any relatively recent Python 3 version will work fine.
- `sudo apt update` (update the package lists)
- `sudo apt install python3` (install python 3)
- `python3 --version` (verify that the install worked)


# Tutorial
Once you have everything set up, go ahead and run main.py using `python3 main.py` or `python main.py`. It should work with no additional input using the default settings!

However, there are two files you will have to edit before you start running actual work units: worktodo.txt and settings.json. 
## worktodo.txt
worktodo.txt lists all the work that will be done by the program.
worktodo.txt contains lines of the following format:
`startNum,endNum,minGap,[deviceNum]`
`startNum` and `endNum` are multiplied by 10^12. This means, for example, a work unit of `23456789,23457000,1200` will find all gaps between `23456789000000000000` and `23457000000000000000` of length >=1200. In the case of a large gap that straddles the border of a work unit, what matters is whether the *smaller* prime is inside the work unit boundaries.
## settings.json
This is where you can configure several variables that can greatly affect the program's performance. If you set them to -1, the code will automatically try to estimate a good value, but I would recommend tweaking the values until you find the fastest possible parameters for your GPU. 
### Configurable settings
- `WORD_LENGTH` represents the amount of numbers represented by a single 32-bit word. It must be either 120 or 240, no other values are currently acceptable. If you use a minGap < 960 (or if you have a fast GPU and minGap < 1200) then you should use `WORD_LENGTH`=120. Otherwise, use `WORD_LENGTH`=240 for a ~15-20% performance improvement. Also, minGap must be a multiple of `WORD_LENGTH`/4.
- `GPU_BLOCKS` represents how many GPU thread blocks to invoke the kernel with. It is recommended to keep this at -1 for the program to choose automatically.
- `BLOCK_SIZE` represents the size of the range of numbers we store in memory for sieving. This should be as high as possible while not getting too close to the limits of the GPU's memory. It is recommended to keep this at -1, in which case it will estimate a value such that about half of the GPUs memory will be in use at any one time. (From testing, I found that using an amount of memory close to 100% can cause performance loss)
- `PROPORTION_OF_BLOCKS_FOR_SIEVING` This parameter represents the proportion of the total `GPU_BLOCKS` to use for sieving, while the rest are used for prime gap checks. The code sieves a block of length `BLOCK_SIZE` while *simultaneously* searching for gaps on the previous sieved block. Recommended to keep this at -1.
- `SHARED_SIZE_WORDS` is the length of the cached sieve segments in words. GPUs have "shared memory" which is essentially a user-managed cache, which is ultra-fast for memory accesses and writes. We do nearly all the sieving here. Recommended to keep this at -1.
- `SMALL_PRIME_WHEELS`is the number of small prime wheels to use. (See explanation in a later section)
- `NUM_MEDIUM_PRIMES_BASE` is (approximately) the number of primes to sieve with, after applying the small prime wheels. It must be a multiple of 512, but it is recommended to keep this at -1 to automatically choose a good value.

There are two more options that only affect the gap report file, they are:
- `SORT_OUTPUT_BY_GAPSIZE` can be either true or false. If true, the gaps in the output file will be sorted by their gap length, with the largest gap first. If false, it will sort the gaps "chronologically", with the smallest primes first.
- `NAME`Your name. This will get included in the gap report. It is recommended that this is exactly 8 characters long. Please do not use a pseudonym.
## Using multiple GPUs
To use multiple GPUs, go to worktodo.txt and set the `deviceNum` on each row to be the index of the device that you want to do that work unit. When you invoke the python code, you can pass a command line argument for the device index that  you'd like to use.
For example, consider these work units:
`22000000,22010000,1200,0`
`22010000,22020000,1200,1`
`22020000,22030000,1200,0`
`22030000,22040000,1200,1`
`22040000,22050000,1200`
Running `prime_gaps.py` or `prime_gaps.py 0` will only run the 1st, 3rd, and 5th of those. Running `prime_gaps.py 1` will only run the 2nd, 4th, and 5th. Therefore, if you have multiple GPUs on a system, you just need 1 running instance per GPU, and you can use a single worktodo file for all of them.
## Continuing failed runs
If you are running a long work unit and something unexpected happens (e.g. a power outage) which kills the process, that is completely OK! Just run `python3 main.py <deviceIdx> continue`. This is in contrast to the default mode `run`. In `continue` mode, the program will not look in worktodo.txt for stuff to do, but instead look for runs that have not finished, and finish those.

Just like with worktodo.txt, the program will only finish runs whose deviceIdx matches the one in the invocation to `main.py`. This means that you must use the same deviceIdx to finish a run as you did to start it! 

# How it Works
The program is divided into several steps:
## Sieving
1. We begin with a bunch of memory for a sieve, initialized to 0. Within each byte, the 8 bits represent numbers 1,7,11,13,17,19,23,29 more than a multiple of 30, respectively. In older versions of the code, every block of 30 is represented in the sieve as a byte, but a significant performance improvement was achieved when I started only sieving *half* of all blocks of 30, and leaving out the rest. For example, byte #0 might sieve x+(1,7,11,13,17,19,23,29), and byte #1 sieves x+(61,67,71,73,77,79,83,89), completely skipping values from x+30 to x+60. With this alternating strategy, we only have to sieve half as many values at once, and the later post-sieving parts of the code have to do negligibly more work. This is represented by the `WORD_LENGTH` parameter, which is either 120 (sieving every byte) or 240 (sieving every 2nd byte).
2. The actual sieving begins with small prime wheels. These are precomputed blocks that can sieve for many small primes at a time. There can be a maximum of 4, with the following values:
  - Wheel 1 uses primes 7-29, for a total of 215,656,441 words
  - Wheel 2 uses primes 31-47, for a total of 95,041,567 words
  - Wheel 3 uses primes 53-67, for a total of 12,780,049 words
  - Wheel 4 uses primes 71-83, for a total of 33,984,931 words
 I tried making a 5th one, but I found it was more efficient to stop at 4. For primes larger than 83, it is better to sieve them using the next method.
3. For medium-size primes, we utilize the GPU's shared memory to make super-fast memory accesses. We divide the whole sieve into smaller blocks, and we copy them into shared memory. Starting from the smallest prime not covered by the previous step, the next 8 primes will be sieved with 8 threads each, for a total of 64. The next 16 primes get 4 threads each, then the next 32 primes get 2 threads each, then all primes past this point get 1 thread. The kernel is always initialized with 512 threads.
4. The next step is to deal with pseudoprimes. We don't have to worry about PSPs that are divisible by a prime that we would have sieved out already, and below 2^65, we don't have to deal with any PSP with no prime factors below 5,000,000 (since they have all been pre-checked for prime gaps, with the largest being <900). **NOTE: This means that the generated list of gaps with minGap < 900 is not guaranteed to be 100% exhaustive, as it may miss gaps that contain a pseudoprime.** However, you will not miss gaps of size >=900 up to 2^65. PSPs are sieved using a very useful property of them: If a 2-PSP `N` is divisible by a prime p, then we have `N = 1 (mod p*ord(p))`, where `ord(p)` is the order of 2 mod p. The proof of this is left as an exercise for the reader. Since `ord(p)` is usually similar to p in magnitude, `p*ord(p)` is close to p^2, which means we only have to sieve around 1 number per every p^2. Since we only do this with large primes (usually p>30000, so p^2>900000000), this step is extremely fast.
## Gap finding
5. The final step is to scan through the sieved data to detect large gaps. This step is done *concurrently* with the previous 4 steps, so for example, a block [X,X+4.8e10] would be sieved while at the same time, the previously sieved block [X-4.8e10,X] is checked for gaps. This usually gives at least some improvement, although for weaker GPUs this improvement is very small. We use an extremely efficient implementation of a 65-bit Fermat test, written by Perig of the Mersenne Forums. Despite its very high speed, it is still one of the main performance bottlenecks we have, along with sieving medium primes. The gap finding algorithm works by jumping up by minGap whenever we find a prime using the Fermat test, then searching *backward* until we find another prime, then repeating the process. Whenever we search backward so far that we hit the prime we started at, we know there is a large gap, and we can deal with it accordingly. It will also never miss any large gaps, since each confirmed prime is less than minGap more than the next one, so there is no room for any extra gaps. Despite the erratic pattern of memory accesses for this step, I found that the memory accesses seem to have a negligible impact on performance here.
6. Whenever we find a large gap, it is recorded on a list and submitted to the CPU. If `WORD_LENGTH == 240`, each submitted gap will not have any primes congruent to 1,7,11,13,17,19,23, or 29 mod 60 (because those are the ones that we actually sieved), but it might still have primes congruent to 31,37,41,43,47,49,53, or 59 mod 60 (we entirely skipped these ones in the sieve). We just double check these all on the CPU to see if it is a real large gap, since the rate that the GPU produces results is low enough that the CPU will not have to do much. The CPU uses a strong PRP test to bases 2,3,5,7,11,13,17,19,23,29,31, and 37, which is deterministic up to well past 2^65, which is all that we care about for now. We don't care about the speed of this test because we don't get GPU results often enough for the speed to be an issue.


# Restrictions
- The code currently only works between 2^64 and 2^65.
- For minGap<900, the resulting list of gaps may not be exhaustive.

# Selected Benchmarks
Note: real-world values may deviate slightly due to many factors.
| GPU | Billion/second |
| -------- | ------- |
| RTX 4090 | 1750 |
| RTX 4070 | 700 |
| RTX 3080 | 650 |
| Titan V | 570  |
| GTX 1660 Mobile | 190