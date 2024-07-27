#include <array>
#include <getopt.h>
#include <iostream>
#include <math.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <thread>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <unistd.h>

using namespace std;

#define count_set_bits_64 __builtin_popcountll

#define uint128_t unsigned __int128

#define WIPE_LINE "\r\033[K"
#define END_OF_RANGE ~0

#define RESULT_LIST_SIZE 65536

#ifndef PROGRESS_UPDATE_BLOCKS
#define PROGRESS_UPDATE_BLOCKS 1
#endif

#ifndef MIN_GAP_SIZE
#define MIN_GAP_SIZE 900 // low enough that it will remind people to set it properly
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 46080000000 // we don't actually need to add the UL as far as I know
#endif

#ifndef USE_SPECIFIC_SIEVING_THREAD_BREAKDOWN
#define USE_SPECIFIC_SIEVING_THREAD_BREAKDOWN 0
#endif

#if USE_SPECIFIC_SIEVING_THREAD_BREAKDOWN
#define SIEVING_DUPLICATED_PRIMES 189 // this number is based on the test1, test2, test3 arrays
#else
#define SIEVING_DUPLICATED_PRIMES 80
#endif

#define WORD_LENGTH 120
#define WORD_SIEVING_LENGTH 120

// TODO: THESE ARRAYS HAVE TO CHANGE BASED ON WORD_LENGTH!!

// lower numbers are the least significant digit
__constant__ uint8_t SIEVE_POS_TO_VALUE[32] = {
    1,7,11,13,17,19,23,29,
    31,37,41,43,47,49,53,59,
    61,67,71,73,77,79,83,89,
    91,97,101,103,107,109,113,119,
};

__constant__ uint8_t SIEVE_VALUE_TO_POS[60] = { // WE ARE ONLY TAKING ODD NUMBERS HERE
    0,0,0,1,0,2,3,0,4,5,0,6,0,0,7,
    8,0,0,9,0,10,11,0,12,13,0,14,0,0,15,
    16,0,0,17,0,18,19,0,20,21,0,22,0,0,23,
    24,0,0,25,0,26,27,0,28,29,0,30,0,0,31,
};

__constant__ uint8_t SIEVE_INCREMENTS[8] = {
    6,4,2,4,2,4,6,2,
};

__constant__ uint8_t NEXT_SIEVE_HIT[30] = {
    1,0,5,4,3,2,1,0,3,2,
    1,0,1,0,3,2,1,0,1,0,
    3,2,1,0,5,4,3,2,1,0,
};

__constant__ bool IS_COPRIME_30[15] = {
    1,0,0,1,0,1,1,0,1,1,0,1,0,0,1,
};


__constant__ const uint32_t SHARED_SIZE_WORDS = 12288; // SET THIS TO BE THE TOTAL SIZE OF SHARED MEMORY
__constant__ const uint32_t NUM_SMALL_PRIMES = 23;
__constant__ const uint32_t NUM_MEDIUM_PRIMES = 512 - SIEVING_DUPLICATED_PRIMES;

struct PrimeGap {
    uint128_t startPrime;
    uint32_t gap;
};
bool compareByPrime(const PrimeGap &a, const PrimeGap &b) {
    return a.startPrime < b.startPrime;
}


__device__ uint32_t getSmallMask(uint32_t prime, uint64_t wordOffset) {
    uint32_t word = 0;
    for (int idx=0; idx<32; idx++) {
        if ((prime - (SIEVE_POS_TO_VALUE[idx] % prime)) % prime == wordOffset % prime) {
            word |= 1 << idx;
        }
    }
    if (wordOffset == 0) {
        // For some reason, if we don't have any print statements in this function,
        // it & the makeSmallPrimeWheels function get completely optimized out, and it doesn't modify the
        // 4 wheels at all. 
        // TODO: What variable do I have to mark as volatile so I don't need this hacky code?
        printf("%d ", word % 1000);
    }
    return word;
}


__global__ void makeSmallPrimeWheels(uint32_t* wheel1, uint32_t* wheel2, uint32_t* wheel3, uint32_t* wheel4) {
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    if (tidx == 0) {
        printf("Don't optimize everything out lol ");
    }
    for (uint64_t i=tidx; i<7*11*13*17*19*23*29; i+=stride) {
        uint64_t wordStart = i * WORD_LENGTH;
        wheel1[i] |= getSmallMask(7, wordStart);
        wheel1[i] |= getSmallMask(11, wordStart);
        wheel1[i] |= getSmallMask(13, wordStart);
        wheel1[i] |= getSmallMask(17, wordStart);
        wheel1[i] |= getSmallMask(19, wordStart);
        wheel1[i] |= getSmallMask(23, wordStart);
        wheel1[i] |= getSmallMask(29, wordStart);
    }
    
    for (uint64_t i=tidx; i<31*37*41*43*47; i+=stride) {
        uint64_t wordStart = i * WORD_LENGTH;
        wheel2[i] |= getSmallMask(31, wordStart);
        wheel2[i] |= getSmallMask(37, wordStart);
        wheel2[i] |= getSmallMask(41, wordStart);
        wheel2[i] |= getSmallMask(43, wordStart);
        wheel2[i] |= getSmallMask(47, wordStart);
    }
    
    for (uint64_t i=tidx; i<53*59*61*67; i+=stride) {
        uint64_t wordStart = i * WORD_LENGTH;
        wheel3[i] |= getSmallMask(53, wordStart);
        wheel3[i] |= getSmallMask(59, wordStart);
        wheel3[i] |= getSmallMask(61, wordStart);
        wheel3[i] |= getSmallMask(67, wordStart);
    }
    
    for (uint64_t i=tidx; i<71*73*79*83; i+=stride) {
        uint64_t wordStart = i * WORD_LENGTH;
        wheel4[i] |= getSmallMask(71, wordStart);
        wheel4[i] |= getSmallMask(73, wordStart);
        wheel4[i] |= getSmallMask(79, wordStart);
        wheel4[i] |= getSmallMask(83, wordStart);
    }
    if (tidx == 0) {
        printf("\n");
    }
}


__device__ int getBigNumStr(uint128_t result, char* digits) {
    // buffer must be at size 40
    uint32_t digNum = 1;
    for (; (digNum<=40) && result; digNum++) {
        digits[40-digNum] = (char) ('0' + (result % ((uint128_t) 10)));
        result /= 10;
    }
    return 41-digNum;
}


__device__ void clearSieve(uint32_t* sieve, uint32_t sieveLengthWords, uint32_t tidx, uint32_t stride) {
    for (int idx=tidx; idx<sieveLengthWords; idx+=stride) {
        sieve[idx] = 0;
    }
}




__device__ uint64_t lo19(uint128_t n) {
    return (uint64_t) (n % 10000000000000000000UL);
}
__device__ uint64_t hi19(uint128_t n) {
    return (uint64_t) (n / 10000000000000000000UL) % 10000000000000000000UL;
}
uint64_t lo19c(uint128_t n) {
    return (uint64_t) (n % 10000000000000000000UL);
}
uint64_t hi19c(uint128_t n) {
    return (uint64_t) (n / 10000000000000000000UL) % 10000000000000000000UL;
}

__device__ void printBigNum(uint128_t num) {
    // works up to 10^38
    printf("%lu%019lu\n", hi19(num), lo19(num));
}

__device__ uint64_t getMagic(uint128_t mod) {
    // !! THIS ONLY WORKS IF mod > 2^64, otherwise we would get overflow!
    return (uint64_t) ((((uint128_t) 0) - 1) / mod);
}

__device__ uint64_t mul_128_64_hi64_inexact(uint128_t a128, uint64_t b64) {
    // Gets the highest 64 bits of a product of 128-bit and 64-bit integers
    // We are ignoring the lower 64 bits of a128, since that will affect the result by at most 1.
    // We will deal with the +1 later.
    return (uint64_t) (((a128 >> 64) * b64) >> 64);
}

__device__ uint128_t fastMod(uint128_t n, uint128_t mod, uint64_t magic) {
    // !!! THIS ONLY WORKS IF mod > 2^64
    // (this is because "magic" needs to fit in a 64-bit int for efficiency)
    uint128_t result = n - mod * mul_128_64_hi64_inexact(n, magic);

    // potential +1 from losing the carry in the mul128*64, and another +1 from the inexactness of the magic num

    return result - ((result >= mod) + (result >= mod*2)) * mod;
}

__device__ bool fermatTest64(uint128_t n) {
    uint128_t result = 1;
    for (int bit=0; bit<64; bit++) {
        if (n & (1 << bit)) {
            result = (result * 2) % n;
        }
        result = (result * result) % n;
    }
    return (result == 1);
}

__device__ uint128_t squareMod84(uint128_t a, uint128_t mod, uint128_t magic) {
    uint128_t ahi = a>>42;
	uint128_t alo = a & 0x3ffffffffffL;
	//return ((((a*ahi) % m) << 42) + a*alo) % m
    return fastMod((fastMod(a*ahi, mod, magic) << 42) + a*alo, mod, magic);
}

__device__ bool fermatTest645_BRIAN(uint128_t n) {
    // n must be < 2^64.5 ~ 26087635650665564424 = 2.6087e19
    uint128_t result = 1;
    uint128_t mod128 = ((uint128_t) -1) % n + 1;
    uint64_t magic = getMagic(n);
    for (int bit=64; bit>=1; bit--) {
        if ((n >> bit) & 1) {
            result *= 2;
            result -= n * (result >= n); // using an if statement here miiiiight be faster?
        }
        bool overflow = result >= (((uint128_t) 1) << 64);
        result = fastMod(result * result, n, magic);
        if (overflow) {
            result += mod128; // we know that mod128 <= n
            result -= n * (result >= n);
        }
    }
    return result == 1;
}

__device__ bool fermatTest84(uint128_t n) {
    // n must be < 2^64.5 ~ 26087635650665564424 = 2.6087e19
    uint128_t result = 1;
    //uint128_t mod128 = ((uint128_t) -1) % n + 1;
    uint64_t magic = getMagic(n);
    for (int bit=84; bit>=1; bit--) {
        if ((n >> bit) & 1) {
            result *= 2;
            result -= n * (result >= n); // using an if statement here miiiiight be faster?
        }
        result = squareMod84(result, n, magic);
    }
    return result == 1;
}

__device__ int64_t mulmod52(uint64_t a, uint64_t b, uint64_t n, double one_over_n) {
   int64_t tmp, ret;
   double x, y;
   x = (int64_t)a;
   y = (int64_t)b;
   tmp = (int64_t)n * (int64_t)(x * y * one_over_n);
   ret = a * b - tmp;
   if (ret < 0) ret += n;
   else if (ret >= (int64_t)n) ret -= n;
   return ret;
}

__device__ bool fermatTest52(uint64_t n) {
    int64_t result = 1;
    double one_over_n = 1.0 / n;
    for (int bit=51; bit>=0; bit--) {
        result = mulmod52(result, result, n, one_over_n);
        if ((n-1) & (1L << bit)) {
            result *= 2;
        }
    }
    return (result == 1);
}


__device__ uint64_t my_getMagic1(uint128_t mod)
{
	// precomputes (2^128 - 1)/ mod    (a 64 bits number)
	// !! THIS ONLY WORKS IF mod > 2^64, otherwise we would get overflow!
	return (uint64_t) ((((uint128_t) 0) - 1) / mod);
}

__device__ uint128_t my_getMagic2(uint128_t mod, uint64_t magic1)
{
	// precomputes 2^96 % mod    (a 65 bits number)
	//
	// this magic helps later to reduce a 128-bit number r to less than 97 bits
	// magic2 = (1 << 96) % mod
	// r =  (r & ((1 << 96) -1)) + (r >> 96) * magic2
	// 
#if 0
	// slow way
	uint128_t t = 1;
	t <<= 96;
	t %= mod;
#else
	// faster way : barrett reduction
	uint128_t t = (uint128_t) 1 << 96;
	uint128_t e = (uint128_t) magic1 << 32;
	uint64_t e_hi = (uint64_t) (e >> 64);
	uint64_t mod_lo = (uint64_t) mod;
	t -= ((uint128_t) mod_lo * e_hi) + ((uint128_t) e_hi << 64);
	t -= t >= mod ? mod : 0;
#endif
	if (t >> 64)
		t += ((uint128_t) 0xfffffffffffffffeull) << 64;
	return t;
}

// input n : a number up to 64 + 8 = 72 bits
// input mod : the modulus withouts top bit (bit 64 is always 1)
// output result, a number less than 68 bits
//
// This code assumes the compiler knows how to optimize in 1 multiplication
// res_128 = (uint128_t)op1_64 * op2_64
// This code assumes the compiler knows how to optimize the 64 bits shift 
// and the constructions of 128 bits.

__device__ uint128_t my_fastModSqr(uint128_t n, uint64_t mod_lo, uint64_t magic1, uint128_t magic2)
{
	uint64_t n_lo = (uint64_t) n;
	uint64_t n_hi = (uint64_t) (n >> 64);	// let assume n_hi is less than 8 bits

	// step 1
	// do the squaring r = n_lo^2 + n_hi^2 + 2 * n_lo * n_hi;
	uint128_t lo = (uint128_t) n_lo * n_lo;	// lo is less than 64+64 = 128 bits
	uint64_t hi = n_hi * n_hi;	// hi is less than 8 + 8 = 16 bits
	uint128_t mid = (uint128_t) n_lo * (n_hi * 2);	// mid is less than 64 + 16 + 1 = 81 bits
	mid += (uint64_t) (lo >> 64);	// mid is less than 81 -> 82 bits
	lo = (uint64_t) lo;	// lo is less than 64 bits 

	// reduce (r & ((1 << 98) -1)) + (r >> 98) * magic2
	// by doing
	// lo += (hi << (98 - 64) + mid >> (98 - 64)) * magic2
	hi = (hi << 32) + (uint64_t) (mid >> 32);	// hi is less than (16 + 32) (81 - 32) -> 50   bits
	mid = (uint32_t) mid;	// mid is less than 32 bits
	uint64_t magic2_lo = (uint64_t) magic2;	// a 64 bit number
	uint64_t magic2_hi = (uint64_t) (magic2 >> 64);	// a bit mask
	lo += (uint128_t) magic2_lo *hi;	// lo is less than 50+64 = 114 bits
	lo += ((uint128_t) (magic2_hi & hi)) << 64;	// lo is less than 64 + 50 = 114 bits
	// 
	mid += (lo >> 64);	// mid is less than (114 - 64) (32) -> 51 bits
	lo = (uint64_t) lo;	// lo is less than 64 bits
	uint128_t res = (mid << 64) + lo;	// res is less than (51 + 64) (64) -> 116 bits

	// barrett approximate reduction, less than 4 extra bits left
	// magic1 is 64 bits, mid is 52 bits
	uint128_t e = (uint128_t) magic1 * mid;	// e is less than  (51 + 64) = 115 bits
	uint64_t e_hi = (uint64_t) (e >> 64);	// e_hi is less than 51 bits
	res -= ((uint128_t) mod_lo * e_hi) + (((uint128_t) e_hi) << 64);
	// - barrett reduction : 1 extra subtraction sometimes needed  (about less 50 %)
	// - barrett magic1 is underestimated by up to 1 : 1 extra subtraction (rarely needed)
	// -> res is less than 3 times the modulus, i.e about 0x6xxx...xxx (67 bit number)

	return res;

}


#if !defined(EULER_CRITERION)
#define EULER_CRITERION 1
#endif
__device__ bool fermatTest65(uint128_t n)
{

	// - iterate on 64 bits before squaring would overflow 
	// and until modular reduction becomes necessary
	// - hardcode 2 most significant bits
	// - Advance 2 bits at a time with window size 2
	uint64_t n_lo = (uint64_t) n;
	uint64_t result64 = ((n_lo >> 63) & 1) ? 8 : 4;	// 2 most significant bits of the modulus 
	int bit = 63;
	while (bit && result64 <= 38967) {
		bit -= 2;
		result64 *= result64;
		result64 *= result64;
		result64 *= 1 << ((n_lo >> bit) & 3);
	}

	// - iterate on 128 bits with modular reduction
	// - at each step and intermediate numbers are only
	//   a few bits larger than the modulus
	// - the last iteration is optimized out the loop
	uint128_t result = result64;
	uint64_t magic1 = my_getMagic1(n);
	uint128_t magic2 = my_getMagic2(n, magic1);
	while (bit > 1) {
		// advance 2 bits at a time
		bit -= 2;
		// square and reduce
		result = my_fastModSqr(result, n, magic1, magic2);
		result = my_fastModSqr(result, n, magic1, magic2);
		// - multiply with window size 2     (values are 1, 2, 4 or 8)
		// and let the number overflow a little bit
		// - result is less than 64 + 3 + 4 = 71 bits   (quite over-rounded)
		result *= 1 << ((n_lo >> bit) & 3);
	}

#if EULER_CRITERION
	// - last round with 1 bit to process
	// - Euler's criterion 2^(n>>1) == legendre_symbol(2,n) (https://en.wikipedia.org/wiki/Euler%27s_criterion)
	// - This skips the modexp last round. Thanks Mr Leonhard.
	// - shortcut calculation of legendre symbol (https://en.wikipedia.org/wiki/Legendre_symbol)
	// legendre_symbol(2,n) = 1 if n = 1 or 7 mod 8     (when bits 1 and 2 are same)
	// legendre_symbol(2,n) = -1 if n = 3 or 5 mod 8    (when bits 1 and 2 are different)
	uint64_t legendre = ((n_lo >> 1) & 1) ^ ((n_lo >> 2) & 1);	// shortcut calculation of legendre symbol
	uint128_t expected = legendre ? n - 1 : 1;	// shortcut calculation of legendre symbol

	// - final reductions :  result %= n;
	// - the last operation was a multiplication by 1,2,4, or 8, without reduction. 
	//   therefore, there are many extra bits to shave. worst case is 7 bits.
	// - use repeated subtractions within a log2 algorithm
	uint128_t t = n;
#if 1
	while (t < result)
		t *= 2;
#else
	t <<= 63 - __builtin_clzll((uint64_t)(result >> 64));
#endif
	while (t >= n) {
		if (result >= t) {
			result -= t;
		}
		t >>= 1;
	}

#else
	// - last round with 1 bit to process , and this bit from n-1 is always 0. No multiplication by 2 is needed
	result = my_fastModSqr(result, n, magic1, magic2);
	uint128_t expected = 1;

	// - final reductions :  result %= n;
	// - only a few bits to shave
	while (result >= n) {
		result -= n;
	}
#endif

	return result == expected;
}


__device__ void sieveSmallPrimes(uint32_t* sieve, uint32_t sieveLengthWords, uint128_t start,
                                 uint32_t* smallPrimeWheel1, uint32_t* smallPrimeWheel2,
                                 uint32_t* smallPrimeWheel3, uint32_t* smallPrimeWheel4) {

    // sieve should be in SHARED MEMORY for this function to work properly
    for (uint32_t i = threadIdx.x; i < sieveLengthWords; i += blockDim.x) {
        // TODO: Check if these modulos are getting optimized, especially for int128

        uint128_t wordStart = start/WORD_LENGTH + i;
        // We cannot replace the atomicOr with a non-atomic operation, because that might skip sieving out some values
        // and we can't miss any because of pseudoprimes
        // making this non-atomic actually doesn't increase performance at all, so the bottleneck is elsewhere on my laptop
        atomicOr(&sieve[i], smallPrimeWheel1[wordStart % (7*11*13*17*19*23*29)] | 
                            smallPrimeWheel2[wordStart % (31*37*41*43*47)] | 
                            smallPrimeWheel3[wordStart % (53*59*61*67)] | 
                            smallPrimeWheel4[wordStart % (71*73*79*83)]);
        

    }
    __syncthreads();
}

__device__ void sieveMediumLargePrimesInner(uint32_t* sieve, uint32_t sieveLengthWords, uint128_t start, uint32_t p) {
    // this function could do with some optimization overall
    uint128_t x = start / p + 1;
    x = x + NEXT_SIEVE_HIT[x % 30]; // the next number k after start/p that's k for gcd(k,30) = 1
    uint32_t pMultSieveIndex = SIEVE_VALUE_TO_POS[(x % 30) / 2];
    x *= p; // the next number p*k after start that has gcd(k,30) = 1
    uint32_t currentWord = (uint32_t) ((x - start) / WORD_LENGTH);
    uint32_t currentPosInWord = x % WORD_LENGTH;
    while (currentWord < sieveLengthWords) {
        // Update the sieve
        if (currentPosInWord < WORD_SIEVING_LENGTH) { // TODO: THIS CHECK LOSES A BUNCH OF TIME
            atomicOr(&sieve[currentWord], 1 << SIEVE_VALUE_TO_POS[currentPosInWord / 2]);
        }

        // Find the next position
        currentPosInWord += p * SIEVE_INCREMENTS[pMultSieveIndex];
        pMultSieveIndex = (pMultSieveIndex + 1) % 8;
        currentWord += (currentPosInWord) / WORD_LENGTH;
        currentPosInWord %= WORD_LENGTH;
    }
}

__device__ uint32_t test1[256] = {11,11,11,11,11,11,11,11,11,11,11,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};
__device__ uint32_t test2[256] = {0,1,2,3,4,5,6,7,8,9,10,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,0,1,2,3,4,5,6,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1};
__device__ uint32_t test3[256] = {0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,8,8,8,8,8,8,8,9,9,9,9,9,9,10,10,10,10,10,10,11,11,11,11,11,11,12,12,12,12,12,12,13,13,13,13,13,14,14,14,14,14,15,15,15,15,15,16,16,16,16,16,17,17,17,17,17,18,18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,22,22,22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,27,27,27,28,28,28,29,29,29,30,30,30,31,31,31,32,32,32,33,33,33,34,34,34,35,35,35,36,36,36,37,37,37,38,38,38,39,39,39,40,40,41,41,42,42,43,43,44,44,45,45,46,46,47,47,48,48,49,49,50,50,51,51,52,52,53,53,54,54,55,55,56,56,57,57,58,58,59,59,60,60,61,61,62,62,63,63,64,64,65,65,66,66};
__device__ void sieveMediumPrimes(uint32_t* sieve, uint32_t sieveLengthWords, uint128_t start,
                                  uint32_t* primeList, uint32_t primeCount) {
    // sieve should be in SHARED MEMORY for this function to work properly
    // from the perspective of this function, the first 23 primes DON'T EXIST!!!!!

    /*
    (1): array that replaces the individiual numbers (8,8,8,8....4,4,4,4.....1,1,1,1)
    (2): array that replaces threadIdx.x % 8 (0,1,2,3,4,5,6,7,0,1,2,3.....0,1,0,1,0,1....0,0,0,0)
    (3): array that has the prime indexes: (0,0,0,0,0,0,0,0,1,1,1.....237,238,239)
    starting pos in the sieve: depends on (2)
    length of the search space in sieve: depends on (1)
    starting N: depends on (2)
    prime: depends on (3)
    */
#if USE_SPECIFIC_SIEVING_THREAD_BREAKDOWN
    // This code works better on my laptop GPU
    if (threadIdx.x < 256) {
        uint32_t numBlocks = test1[threadIdx.x];
        uint32_t blockIdx = test2[threadIdx.x];
        uint32_t primeIdx = test3[threadIdx.x];
        uint32_t wordStart = (uint32_t) (((double) sieveLengthWords) * blockIdx / numBlocks);
        uint32_t wordEnd   = (uint32_t) (((double) sieveLengthWords) * (blockIdx+1) / numBlocks);
        // Note: If numBlocks is not a factor of sieveLengthWords, we might end up skipping the last word!!
        sieveMediumLargePrimesInner(
            sieve + wordStart,
            wordEnd - wordStart,
            start + wordStart*WORD_LENGTH,
            primeList[primeIdx]
        );
    }
    for (uint32_t pidx = threadIdx.x+blockDim.x-189; pidx < primeCount; pidx += blockDim.x) {
        sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, primeList[pidx]);
    }
#else
    // This code works better on an RTX 4090
    if (threadIdx.x < 64) {
        sieveMediumLargePrimesInner(sieve + (sieveLengthWords/4 * (threadIdx.x % 4)), sieveLengthWords/4,
            start + (sieveLengthWords * WORD_LENGTH/4 * (threadIdx.x % 4)), primeList[threadIdx.x/4]);
    } else if (threadIdx.x < 128) {
        sieveMediumLargePrimesInner(sieve + (sieveLengthWords/2 * (threadIdx.x % 2)), sieveLengthWords/2,
            start + (sieveLengthWords * WORD_LENGTH/2 * (threadIdx.x % 2)), primeList[(threadIdx.x-64)/2 + 16]);
    } else {
        sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, primeList[threadIdx.x-80]);
    }
    for (uint32_t pidx = threadIdx.x+blockDim.x-80; pidx < primeCount; pidx += blockDim.x) {
        sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, primeList[pidx]);
    }
#endif
    /*
    if (threadIdx.x < 256) {
        if (threadIdx.x < 128) {
            sieveMediumLargePrimesInner(sieve + (sieveLengthWords/8 * (threadIdx.x % 8)), sieveLengthWords/8,
                start + (sieveLengthWords * WORD_LENGTH/8 * (threadIdx.x % 8)), primeList[threadIdx.x/8]);
        } else {
            sieveMediumLargePrimesInner(sieve + (sieveLengthWords/4 * (threadIdx.x % 4)), sieveLengthWords/4,
                start + (sieveLengthWords * WORD_LENGTH/4 * (threadIdx.x % 4)), primeList[(threadIdx.x-128)/2 + 16]);
        }
    } else {
        if (threadIdx.x < 384) {
            sieveMediumLargePrimesInner(sieve + (sieveLengthWords/2 * (threadIdx.x % 2)), sieveLengthWords/2,
                start + (sieveLengthWords * WORD_LENGTH/2 * (threadIdx.x % 2)), primeList[(threadIdx.x-256)/2 + 16+32]);
        } else {
            sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, primeList[threadIdx.x-272]);
        }
    }
    for (uint32_t pidx = threadIdx.x+blockDim.x-272; pidx < primeCount; pidx += blockDim.x) {
        sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, primeList[pidx]);
    }
    */
    
    __syncthreads();
}

__device__ void sieveLargePrimes(uint32_t* sieve, uint32_t sieveLengthWords, uint128_t start,
                                 uint32_t* primeList, uint32_t primeCount, uint32_t numBlocks) {
    // sieve should be in GLOBAL MEMORY for this function to work properly
    
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * numBlocks;

    for (uint32_t pidx = tidx; pidx < primeCount; pidx += stride) {
        sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, primeList[pidx]);
    }
    __syncthreads();
}

__device__ void sievePseudoprimes(uint32_t* sieve, uint32_t sieveLengthWords, uint128_t start,
                                  uint32_t* primeList, uint32_t* rhoList, uint32_t primeCount,
                                  uint32_t numBlocks) {
    // sieve should be in GLOBAL MEMORY for this function to work properly

    // We are sieving for entries that are congruent to p mod p*rho(p), because this is guaranteed
    // to remove all 2-PSPs that have p as a factor.
    // If we do this up to p=5000000, then all remaining 2-PSPs (i.e. with no prime factors below 5M)
    //   have been checked up to 2^65, with no gaps of size >900.
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * numBlocks;

    for (uint32_t pidx = tidx; pidx < primeCount; pidx += stride) {
        uint32_t p = primeList[pidx];
        uint32_t rho = rhoList[pidx];
        uint64_t pTimesRho = ((uint64_t) p) * rho;
        pTimesRho <<= pTimesRho % 2; // if it's odd, multiply it by 2
        uint64_t position = pTimesRho + p - (start % pTimesRho);
        position -= pTimesRho * (position > pTimesRho);
        
        uint64_t currentWord = position / WORD_LENGTH; // this needs to be 64 bit
        uint32_t currentPosInWord = position % WORD_LENGTH;

        uint32_t pTimesRhoMod120 = pTimesRho % WORD_LENGTH;
        uint64_t pTimesRhoDiv120 = pTimesRho / WORD_LENGTH;
        while (currentWord < sieveLengthWords) {
            // Update the sieve
            if (currentPosInWord < WORD_SIEVING_LENGTH && IS_COPRIME_30[(currentPosInWord % 30) / 2]) {
                /*if ((~sieve[currentWord]) & (1 << SIEVE_VALUE_TO_POS[currentPosInWord / 2])) {
                    uint128_t num = start + ((uint128_t) currentWord)*WORD_LENGTH + currentPosInWord;
                    if (fermatTest645(num)) {
                        printf("Pseudoprime p=%d mod 1e19=%lu %lu %u %u\n", p, (uint64_t) (num % 10000000000000000000UL),
                        currentWord, currentPosInWord, (uint32_t) (num%p));
                        //printf("20000%lu\n", (uint64_t) (num % 10000000000000000000UL));
                    }
                }*/
                atomicOr(&sieve[currentWord], 1 << SIEVE_VALUE_TO_POS[currentPosInWord / 2]);
            }

            // Find the next position
            currentPosInWord += pTimesRhoMod120;
            currentWord += pTimesRhoDiv120 + (currentPosInWord >= WORD_LENGTH);
            currentPosInWord -= WORD_LENGTH * (currentPosInWord >= WORD_LENGTH);
        }
    }
    __syncthreads();
}

__device__ void sieveAll(uint32_t* globalSieve, uint128_t sieveStart, uint32_t sieveLengthWords,
                         uint32_t* primeList, uint32_t* rhoList, uint32_t primeCount,
                         uint32_t* smallPrimeWheel1, uint32_t* smallPrimeWheel2,
                         uint32_t* smallPrimeWheel3, uint32_t* smallPrimeWheel4,
                         uint32_t numBlocks) {
    // the actual sieve length is WORD_LENGTH * sieveLengthWords

    uint32_t tidx = blockIdx.x * numBlocks + threadIdx.x;
    
    if (sieveLengthWords % SHARED_SIZE_WORDS != 0) {
        if (tidx == 0) {
            printf("ERROR: Length of the block (%lu) is not a multiple of %d times the shared size (%d)\n",
                   ((uint64_t) sieveLengthWords)*WORD_LENGTH, WORD_LENGTH, SHARED_SIZE_WORDS);
        }
        return;
    }

    __shared__ uint32_t sharedSieve[SHARED_SIZE_WORDS];
    uint32_t numSharedBlocks = sieveLengthWords / SHARED_SIZE_WORDS;
    
    for (uint64_t sharedBlockIdx = blockIdx.x; sharedBlockIdx < numSharedBlocks; sharedBlockIdx += numBlocks) {
        // Reset the shared memory to 0, since it doesn't necessarily start out that way
        for (int idx=threadIdx.x; idx<SHARED_SIZE_WORDS; idx+=blockDim.x) {
            sharedSieve[idx] = 0;
        }

        sieveSmallPrimes(sharedSieve, SHARED_SIZE_WORDS, sieveStart + sharedBlockIdx*SHARED_SIZE_WORDS*WORD_LENGTH,
                         smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4);

        if (NUM_MEDIUM_PRIMES > 0) {
            sieveMediumPrimes(sharedSieve, SHARED_SIZE_WORDS, sieveStart + sharedBlockIdx*SHARED_SIZE_WORDS*WORD_LENGTH,
                            primeList+NUM_SMALL_PRIMES, NUM_MEDIUM_PRIMES);
        }

        for (int sharedIdx=threadIdx.x; sharedIdx<SHARED_SIZE_WORDS; sharedIdx += numBlocks) {
            atomicOr(&globalSieve[sharedBlockIdx*SHARED_SIZE_WORDS + sharedIdx], sharedSieve[sharedIdx]);
        }
    }

#if 0
        sieveLargePrimes(globalSieve, sieveLengthWords, sieveStart,
                         primeList+NUM_SMALL_PRIMES+NUM_MEDIUM_PRIMES, primeCount-NUM_SMALL_PRIMES-NUM_MEDIUM_PRIMES,
                         numBlocks);

#endif
    sievePseudoprimes(globalSieve, sieveLengthWords, sieveStart,
                      primeList+NUM_SMALL_PRIMES+NUM_MEDIUM_PRIMES,
                      rhoList+NUM_SMALL_PRIMES+NUM_MEDIUM_PRIMES,
                      primeCount-NUM_SMALL_PRIMES-NUM_MEDIUM_PRIMES, numBlocks);
    
}

__global__ void kernel(uint32_t* globalSieve, uint128_t sieveStart, uint32_t sieveLengthWords,
                       uint32_t* primeList, uint32_t* rhoList, uint32_t primeCount,
                       uint32_t* smallPrimeWheel1, uint32_t* smallPrimeWheel2,
                       uint32_t* smallPrimeWheel3, uint32_t* smallPrimeWheel4) {
    sieveAll(globalSieve, sieveStart, sieveLengthWords, primeList, rhoList, primeCount,
        smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4, gridDim.x);
}



// ========== THIS IS THE START OF THE 2ND PART OF THE CODE, PARSING THE SIEVE TO FIND PRIME GAPS ========== //


__device__ uint64_t findNextUnsieved(uint32_t* sieve, uint64_t sieveLengthWords, uint64_t bitPosition) {
    // bitPosition treats the entire sieve as a bitSet
    // Gets the next unsieved number from a given position,
    //   starting with AND INCLUDING bitPosition
    
    if (bitPosition >= sieveLengthWords*32) return END_OF_RANGE;
    int64_t wordIdx = bitPosition/32;
    while (wordIdx < sieveLengthWords) { // total number of bits in shared memory
        uint32_t word = ~(sieve[wordIdx]) & ((~0U) << (bitPosition%32));
        if (word) {
            return wordIdx*32 + __ffs(word) - 1;
        }
        wordIdx++;
        bitPosition = 0; // maybe can optimize this more
    }
    return END_OF_RANGE; // return 0xFFFFFFFF if no result
}

__device__ uint64_t findPrevUnsieved(uint32_t* sieve, int64_t bitPosition) {
    // bitPosition treats the entire sieve as a bitSet
    // Gets the previous unsieved number from a given position,
    //   starting with AND INCLUDING bitPosition
    
    if (bitPosition < 0) return END_OF_RANGE;
    int64_t wordIdx = bitPosition/32; // signed int, so we can compare it with 0 properly
    while (wordIdx >= 0) { // total number of bits in shared memory
        uint32_t word = (~sieve[wordIdx]) & ((~0U) >> (31 - bitPosition%32));
        if (word) {
            return wordIdx*32 + 31 - __clz(word);
        }
        wordIdx--;
        bitPosition = 31; // maybe can optimize this more
    }
    return END_OF_RANGE; // return 0xFFFFFFFF if no result
}

__device__ uint128_t getNumberFromSieve(uint128_t start, int64_t bitPosition) {
    return start + bitPosition/32*WORD_LENGTH + SIEVE_POS_TO_VALUE[bitPosition%32];
}

#define FERMAT_TEST fermatTest65
__device__ void findGaps(uint32_t* sieve, uint128_t start, uint64_t sieveLengthWords, uint32_t startBlock, PrimeGap* resultList) {
    // sieve should be in GLOBAL MEMORY for this function to work properly
    uint32_t gridDimNew = gridDim.x - startBlock;
    uint32_t blockIdxNew = blockIdx.x - startBlock;

    const int MIN_GAP_SIZE_30 = MIN_GAP_SIZE / 30;
    // one group of 30 numbers corresponds to 8 bits

    int64_t bitPosition;
    int64_t limitBitPosition;
    uint32_t tidx = blockIdxNew * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDimNew;
    bitPosition = sieveLengthWords*32 / stride * tidx;
    limitBitPosition = sieveLengthWords*32 / stride * (tidx + 1);
    
    bitPosition -= bitPosition % 32;
    limitBitPosition -= limitBitPosition % 32;

    bool hitEndOfRange = false;
    bitPosition = findNextUnsieved(sieve, sieveLengthWords, bitPosition);

    // Calculate the first prime in the range
    uint128_t lastPrime = getNumberFromSieve(start, bitPosition);
    while (!FERMAT_TEST(lastPrime)) {
        if (bitPosition == END_OF_RANGE) {
            // this will only happen if we find a single gap of size sieveLengthWords*WORD_LENGTH/threadsPerBlock
            // which almost certainly will never happen
            // if we ever hit the end of the shared memory range while finding a gap, we can ignore it
            // because we will find that gap anyway while searching on the borders of shared memory blocks
            hitEndOfRange = true;
            break;
        }
        bitPosition = findNextUnsieved(sieve, sieveLengthWords, ++bitPosition);
        lastPrime = getNumberFromSieve(start, bitPosition);
    }
    
    __syncthreads();
    
    bool isPrime = false;
    if (hitEndOfRange) goto endLabel;

    bitPosition += 8 * MIN_GAP_SIZE_30;

    while (true) {
        bitPosition = findPrevUnsieved(sieve, --bitPosition);
        uint128_t toTest = start + bitPosition/32*WORD_LENGTH + SIEVE_POS_TO_VALUE[bitPosition%32];
        if (toTest == lastPrime) {
            // found a large gap! this part will get entered rarely so I don't really have to optimize it
            bitPosition += 8 * MIN_GAP_SIZE_30;
            bitPosition = findNextUnsieved(sieve, sieveLengthWords, bitPosition);
            uint128_t upperPrime = getNumberFromSieve(start, bitPosition);
            while (!FERMAT_TEST(upperPrime)) {
                if (bitPosition == END_OF_RANGE) goto endLabel;
                bitPosition = findNextUnsieved(sieve, sieveLengthWords, ++bitPosition);
                upperPrime = getNumberFromSieve(start, bitPosition);
            }
            uint32_t gap = (uint32_t) (upperPrime - lastPrime);
            
            int resultIdx = atomicAdd(&resultList[0].gap, 1) + 1; // index 0 of the list keeps track of the length
            resultList[resultIdx].startPrime = lastPrime;
            resultList[resultIdx].gap = gap;

            lastPrime = upperPrime;
            bitPosition += 8 * MIN_GAP_SIZE_30;
        } else {
            isPrime = FERMAT_TEST(toTest);
            if (isPrime) lastPrime = toTest;
        }

        if (bitPosition >= limitBitPosition && isPrime) break;
        bitPosition += 8 * MIN_GAP_SIZE_30 * isPrime;
        // for some reason, it doesn't work if I just put this in the while loop condition
    }
    endLabel:
    __syncthreads();
}


__global__ void kernel2(uint32_t* globalSieve, uint128_t sieveStart, uint64_t sieveLengthWords, PrimeGap* resultList) {
    findGaps(globalSieve, sieveStart, sieveLengthWords, 0, resultList);
}


__global__ void kernelBoth(uint32_t* globalSieve1, uint32_t* globalSieve2, uint128_t sieveStart, uint32_t sieveLengthWords,
                           uint32_t* primeList, uint32_t* rhoList, uint32_t primeCount,
                           uint32_t* smallPrimeWheel1, uint32_t* smallPrimeWheel2,
                           uint32_t* smallPrimeWheel3, uint32_t* smallPrimeWheel4,
                           uint32_t numSieveBlocks, PrimeGap* resultList) {
    if (blockIdx.x < numSieveBlocks) {
        sieveAll(globalSieve1, sieveStart, sieveLengthWords, primeList, rhoList, primeCount,
            smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4, numSieveBlocks);
    } else {
        findGaps(globalSieve2, sieveStart - ((uint128_t) sieveLengthWords)*WORD_LENGTH, sieveLengthWords, numSieveBlocks, resultList);
    }
}


void printBigNumCPU(uint128_t result) {
    // THIS WORKS!!!
    if ((uint128_t) ((uint64_t) result) == result) {
        printf("%lu", (uint64_t) result);
    } else {
        char digits[40];
        uint32_t digNum = 1;
        for (; (digNum<=40) && result; digNum++) {
            digits[40-digNum] = (char) ('0' + (result % 10));
            result /= 10;
        }
        printf("%s", digits + (41-digNum));
    }
}

uint128_t squareMod84CPU(uint128_t a, uint128_t mod) {
    uint128_t ahi = a>>42;
	uint128_t alo = a & 0x3ffffffffffL;
    return ((((a*ahi) % mod) << 42) + a*alo) % mod;
}

bool fermatTest84CPU(uint128_t n) {
    uint128_t result = 1;
    for (int bit=84; bit>=1; bit--) {
        if ((n >> bit) & 1) {
            result *= 2;
            result -= n * (result >= n);
        }
        result = squareMod84CPU(result, n);
    }
    return result == 1;
}

uint32_t modExp32CPU(uint32_t base, uint64_t n, uint32_t mod) {
    // THIS WORKS!!!
    uint64_t result = 1;
    for (int bit=32; bit>=0; bit--) {
        result = (result * result) % mod;
        if ((n >> bit) & 1) {
            result = (result * base) % mod;
        }
    }
    return (uint32_t) result;
}

void deviceInfo() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    
    printf("Number of devices: %d\n", nDevices);
    
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (MHz): %d\n",
                prop.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n",
                prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
                2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
        printf("  Number of multiprocessors: %d\n",prop.multiProcessorCount);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
    }
}

uint32_t* sieveInitialSmallPrimes(uint32_t limit) {
    // THIS WORKS!!!

    // here, if we just do bool[...] sieve then we can get a segfault (OOM error) for large sizes
    uint32_t* sieve = new uint32_t[limit/2];
    sieve[0] = 1;
    for (int i=1; i<limit/2; i++) {
        sieve[i] = 0;
    }
    uint32_t pr = 3;
    while (pr*pr <= limit) {
        for (int hit=pr*pr/2; hit<limit/2; hit += pr) {
            // sieve[hit] will store the lowest prime factor of hit*2+1, or 0 if there it's prime
            if (sieve[hit] == 0) sieve[hit] = pr;
        }
        do {
            pr += 2;
        } while (sieve[pr/2]);
    }
    return sieve;
}

vector<uint32_t> generateSmallPrimesList(uint32_t limit, uint32_t* sieve) {
    vector<uint32_t> primes;
    primes.push_back(2);
    for (int p=3; p<limit; p+=2) {
        if (sieve[p/2] == 0) {
            primes.push_back(p);
        }
    }
    return primes;
}

vector<uint32_t> generateRhoList(uint32_t limit, uint32_t* sieve, vector<uint32_t> primes) {
    vector<uint32_t> rhos;
    rhos.push_back(0);
    for (auto &p : primes) {
        if (p == 2) continue;
        uint32_t rho = p-1;
        while (rho%2 == 0 && modExp32CPU(2, rho/2, p) == 1) {
            rho /= 2;
        }
        uint32_t remaining = rho;
        while (remaining%2 == 0) {
            remaining /= 2;
        }
        int idx = remaining / 2;
        while (sieve[idx] > 1) {
            if (modExp32CPU(2, rho/sieve[idx], p) == 1) {
                rho /= sieve[idx];
            }
            remaining /= sieve[idx];
            idx /= sieve[idx];
        }
        if (modExp32CPU(2, rho/remaining, p) == 1) {
            rho /= remaining;
        }
        rhos.push_back(rho);
    }
    return rhos;
}

void cpuFindGapAround(uint128_t n, uint32_t minGap) {
    uint128_t p1 = n;
    p1 -= 1 - (p1 % 2);
    while (!fermatTest84CPU(p1)) {p1 -= 2;}

    uint128_t p2 = n;
    p2 += 1 - (p2 % 2);
    while (!fermatTest84CPU(p2)) {p2 += 2;}

    int gap = (int) (p2-p1);
    if (gap >= minGap) {
        printf("%lu%019lu %lu%019lu %d\n",
               hi19c(p1), lo19c(p1),
               hi19c(p2), lo19c(p2), gap);
    }
}










__global__ void sievePseudoprimesSeparate(uint128_t start, uint64_t sieveLengthWords,
                                          uint32_t* primeList, uint32_t* rhoList, uint32_t primeCount) {
    uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t pidx = tidx; pidx < primeCount; pidx += stride) {
        uint32_t p = primeList[pidx];
        uint32_t rho = rhoList[pidx];
        uint64_t pTimesRho = ((uint64_t) p) * rho;
        pTimesRho <<= pTimesRho % 2; // if it's odd, multiply it by 2 - TODO: PRECOMPUTE THIS IN RHO??
        //x = ((start - p) / pTimesRho + 1) * pTimesRho - start;
        uint64_t position = pTimesRho + p - (start % pTimesRho);
        position -= pTimesRho * (position > pTimesRho);
        
        uint64_t currentWord = position / WORD_LENGTH; // this needs to be 64 bit
        uint32_t currentPosInWord = position % WORD_LENGTH;
        while (currentWord < sieveLengthWords) {
            if (IS_COPRIME_30[(currentPosInWord % 30) / 2]) {
                uint128_t num = start + ((uint128_t) currentWord)*WORD_LENGTH + currentPosInWord;
                if (num%7 && num%11 && num%13 && num%17 && num%19 && num%23 && num%29 && num%31) {
                    if (FERMAT_TEST(num)) {
                        printBigNum(num);
                        //printf("Pseudoprime p=%d mod 1e19=%lu %lu %u %u\n", p, (uint64_t) (num % 10000000000000000000UL),
                        //currentWord, currentPosInWord, (uint32_t) (num%p));
                        //printf("20000%lu\n", (uint64_t) (num % 10000000000000000000UL));
                    }
                }
            }
            

            // Find the next position
            currentPosInWord += pTimesRho % WORD_LENGTH;
            currentWord += (uint64_t) (pTimesRho / WORD_LENGTH) + (currentPosInWord >= WORD_LENGTH);
            currentPosInWord -= WORD_LENGTH * (currentPosInWord >= WORD_LENGTH);
        }
    }
}


uint128_t atouint128_t(const char *s) {
    // https://stackoverflow.com/questions/45608424/atoi-for-int128-t-type
    const char *p = s;
    uint128_t val = 0;

    while (*p >= '0' && *p <= '9') {
        val = (10 * val) + (*p - '0');
        p++;
    }
    return val;
}

void displayResultsAndClear(PrimeGap* resultList) {
    sort(resultList+1, resultList+resultList[0].gap+1, compareByPrime);
    for (int i=1; i<=resultList[0].gap; i++) {
        uint128_t endPrime = resultList[i].startPrime + resultList[i].gap;
        printf("%lu%019lu %lu%019lu %u\n",
            hi19c(resultList[i].startPrime), lo19c(resultList[i].startPrime),
            hi19c(endPrime), lo19c(endPrime),
            resultList[i].gap);
        resultList[i].startPrime = 0;
        resultList[i].gap = 0;
    }
    resultList[0].gap = 0;
}


int main(int argc, char* argv[]) {
    setbuf(stdout, NULL);
    if (argc < 2) {
        printf("Incorrect amount of command line arguments (got %d, expected 2)\n", argc);
        exit(1);
    }
    int DEVICE_NUM = 0;
    if (argc > 2) DEVICE_NUM = atoi(argv[2]);
    cudaSetDevice(DEVICE_NUM);

    //deviceInfo();

    printf("Starting\n");

    int SMALL_PRIME_LIMIT = 5000000; // don't change this

    printf("Generating primes below %u\n", SMALL_PRIME_LIMIT);
    uint32_t* smallSieve = sieveInitialSmallPrimes(SMALL_PRIME_LIMIT);
    vector<uint32_t> primeList = generateSmallPrimesList(SMALL_PRIME_LIMIT, smallSieve);
    vector<uint32_t> rhoList = generateRhoList(SMALL_PRIME_LIMIT, smallSieve, primeList);
    printf("Done generating primes below %u\n", SMALL_PRIME_LIMIT);
    delete smallSieve;

    uint32_t* primeListCuda;
    auto err = cudaMallocManaged(&primeListCuda, primeList.size() * sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate managed memory: %s\n", cudaGetErrorString(err));
        return 1;
    }
    uint32_t* rhoListCuda;
    err = cudaMallocManaged(&rhoListCuda, rhoList.size() * sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate managed memory: %s\n", cudaGetErrorString(err));
        return 1;
    }
    for (int i=0; i<primeList.size(); i++) {
        primeListCuda[i] = primeList[i];
        rhoListCuda[i] = rhoList[i];
    }


    // pseudoprime is 21693774589725076147 (67 mod 120), with start 21693774589725076080
    // first kilogap above that is 21693776423220625951-6953, or 1833495550873=1.8e12 larger
    uint128_t sieveStart = atouint128_t(argv[1]);
    
    uint64_t sieveLength = BLOCK_SIZE;
    if (sieveLength >= 4294967296L * WORD_LENGTH) {
        printf("ERROR: Sieve length too large: %ld (maximum is 2^32 * %d)\n", sieveLength, WORD_LENGTH);
        exit(1);
    }
    if (sieveStart % WORD_LENGTH) {
        printf("ERROR: Sieve start must be a multiple of %d\n", WORD_LENGTH);
        exit(1);
    }
    uint64_t sieveLengthWords = sieveLength / WORD_LENGTH;

    /*int offset=78498; // doing primes from 1M to 1.1M
    printf("using asdf %u\n", (primeList.size()-offset));
    
    auto startp = chrono::high_resolution_clock::now();
    sievePseudoprimesSeparate<<<384,64>>>(sieveStart, 100000000000000UL,
                                          primeListCuda+offset, rhoListCuda+offset, 7216); //primeList.size()-offset);
    cudaDeviceSynchronize();
    auto finishp = chrono::high_resolution_clock::now();
    cout << "Done in " << chrono::duration_cast<chrono::nanoseconds>(finishp-startp).count()/1e9 << " seconds\n";
    return 0;*/
    
    uint128_t* endpoints;
    cudaMalloc((void **) &endpoints, sieveLengthWords/12288 * 2 * sizeof(uint128_t));




    uint32_t* smallPrimeWheel1;
    uint32_t* smallPrimeWheel2;
    uint32_t* smallPrimeWheel3;
    uint32_t* smallPrimeWheel4;
    cudaMalloc((void **) &smallPrimeWheel1, (7*11*13*17*19*23*29) * sizeof(uint32_t));
    cudaMalloc((void **) &smallPrimeWheel2, (31*37*41*43*47) * sizeof(uint32_t));
    cudaMalloc((void **) &smallPrimeWheel3, (53*59*61*67) * sizeof(uint32_t));
    cudaMalloc((void **) &smallPrimeWheel4, (71*73*79*83) * sizeof(uint32_t));
    printf("Making small prime sieve\n");
    
    auto start1 = chrono::high_resolution_clock::now();
    makeSmallPrimeWheels<<<96,512>>>(smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4);
    auto finish1 = chrono::high_resolution_clock::now();
    cout << "Done in " << chrono::duration_cast<chrono::nanoseconds>(finish1-start1).count()/1e9 << " seconds\n";
   
    PrimeGap* resultList;
    err = cudaMallocManaged((void **) &resultList, sizeof(PrimeGap) * RESULT_LIST_SIZE);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate managed memory: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    uint32_t* globalSieve1;
    uint32_t* globalSieve2;
    cudaMalloc((void **) &globalSieve1, sieveLengthWords * sizeof(uint32_t));
    cudaMalloc((void **) &globalSieve2, sieveLengthWords * sizeof(uint32_t));


    cudaMemset(globalSieve1, 0, sieveLengthWords * sizeof(uint32_t));
    kernel<<<96,512>>>(
        globalSieve1, sieveStart, (uint32_t) sieveLengthWords,
        primeListCuda, rhoListCuda, primeList.size(),
        smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4
    );
    cudaDeviceSynchronize();

    for (int i=0; i<RESULT_LIST_SIZE; i++) {
        resultList[i].startPrime = 0;
        resultList[i].gap = 0;
    }

    if (MIN_GAP_SIZE % 30) {
        printf("Searching for gaps of size >= %d... (modified from %d)\n", MIN_GAP_SIZE - (MIN_GAP_SIZE % 30), MIN_GAP_SIZE);
    } else {
        printf("Searching for gaps of size >= %d...\n", MIN_GAP_SIZE);
    }

    int blocksToTest = 100;
    auto start = chrono::high_resolution_clock::now();
    auto finish = start;
    for (int i=0; i<blocksToTest-1; i++) {
        if (i%PROGRESS_UPDATE_BLOCKS == 0) {
            finish = chrono::high_resolution_clock::now();
            printf("Done %d blocks (limit=%lu%019lu time=%f seconds)\n", i, hi19c(sieveStart), lo19c(sieveStart),
                    chrono::duration_cast<chrono::nanoseconds>(finish-start).count()/1e9);
        }
        cudaMemset(globalSieve2, 0, sieveLengthWords * sizeof(uint32_t));
        
        kernelBoth<<<192,512>>>(
            globalSieve2, globalSieve1, sieveStart+sieveLength, (uint32_t) sieveLengthWords,
            primeListCuda, rhoListCuda, primeList.size(),
            smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4, 120,
            resultList
        );
        cpuFindGapAround(sieveStart, MIN_GAP_SIZE - (MIN_GAP_SIZE % 30)); // TODO: ADD THIS TO THE LIST IF WE FIND ONE!!!
        cudaDeviceSynchronize();
        displayResultsAndClear(resultList);
        sieveStart += sieveLength;
        swap(globalSieve1, globalSieve2);
    }

    if ((blocksToTest-1)%PROGRESS_UPDATE_BLOCKS == 0) {
        finish = chrono::high_resolution_clock::now();
        printf("Done %d blocks (limit=%lu%019lu time=%f seconds)\n", blocksToTest-1, hi19c(sieveStart), lo19c(sieveStart),
            chrono::duration_cast<chrono::nanoseconds>(finish-start).count()/1e9);
    }

    for (int i=0; i<RESULT_LIST_SIZE; i++) {
        resultList[i].startPrime = 0;
        resultList[i].gap = 0;
    }
    kernel2<<<96,256>>>(globalSieve1, sieveStart, sieveLengthWords, resultList);
    cpuFindGapAround(sieveStart, MIN_GAP_SIZE - (MIN_GAP_SIZE % 30));
    cudaDeviceSynchronize();
    displayResultsAndClear(resultList);
    sieveStart += sieveLength;

    finish = chrono::high_resolution_clock::now();
    printf("Done %d blocks (limit=%lu%019lu time=%f seconds)\n", blocksToTest, hi19c(sieveStart), lo19c(sieveStart),
        chrono::duration_cast<chrono::nanoseconds>(finish-start).count()/1e9);

    return 0;
}

/*
- General tips:
    - Don't be afraid to UNROLL LOOPS!
    - Remember to __syncthreads() between each step!
    - We need to use an AtomicOr function on ALL STEPS!!! Because some blocks might be in a different step than others!
        - This includes the shared memory step, because we cannot miss any bits with sieving small primes due to PSPs
    - Maximum amount of shared memory per block is 49152 bytes, or 12288 ints
    - Maximum amount of constant memory is 65536 bytes, or 16384 ints

- General structure of the code:
    - We have 2 kernels, one for setting up the global memory sieve and the other for iterating over it
    - The CPU code is essentially (for each block of 36 billion) (do kernel 1 then kernel 2)
    - POTENTIAL PROBLEM WITH 2 KERNELS: If we separate the global memory accesses from the fermat tests, then
        it could be less efficient than if we did them separately (we have to parallelize fermat tests and global memory)

- We have a global(?) variable that signifies what multiple of 2^64 we are using
    - we have to precompute (2^64*that) % p for primes p

- The sieve is a pointer to ints with another int that signifies its size
- Each int has 32 bits, which covers 120 numbers because we are skipping multiples of 2,3,5 (the tiny primes)
    - We will probably need constant arrays to help with this format
- We have constant memory that has wheels for small primes starting from 7 up to let's say 47.

- Next step is medium primes, we use shared memory for this.
    - Parallelize over the list of primes
    - Can't use buckets here, too much memory.
    - So we will have to calculate modulos. But we can do this with fastmod by precomputing inverses for all primes.
    - When we calculate the offset, we need to worry about the 120/32 format of the bits

- Next is big primes, we use global memory for this.
    https://github.com/kimwalisch/primesieve/blob/master/doc/ALGORITHMS.md
    - Use a bucket sieve:
    - List of lists [L1, L2, L3, L4...] where the nth list contains all primes whose next multiple is in that subinterval
    - we use wraparound for larger subintervals while we sieve
    - the reason for this is we don't want to kill our memory usage
    - if the shared memory list length is 1024 ints, that can hold 1024*120 = 122,880 numbers

- Next step is PSP sieving.
    - For any remaining primes up to 5 million or whatever our precomputed PSP limit is,
    - Precompute all rho(p) values, parallelize over the list of primes
    - Remove values that are p (mod p*rho(p))

- The final step is fermat tests.
    - Global memory access is negligible here, since the actual fermat tests take longer
    - If we find a prime, skip forward by the minimum gap size, and search backward for a prime
    - If we get back to where we started without finding one, go back but search forward instead, then print that gap
    - Be smart with this code so that we can fully parallelize the fermat tests


THINGS TO ADD:

settings/worktodo files: (should put this in the readme at some point)

=== settings.txt file format: ===
# comment: there should be a script to automatically find the optimal parameters to set
GPU_BLOCKS=192
NUM_BLOCKS_FOR_SIEVING=120
GPU_THREADS=512
BLOCK_SIZE=46080000000
SORT_OUTPUT_BY_GAPSIZE=1 # If 0, sorts by the prime (increasing). If 1, sorts by the gap size (decreasing)
NAME=B.Kehrig

=== worktodo.txt file format: ===
# format: start(e12), end(e12), minGap, username
18470057,18571674,1200,B.Kehrig  #will find the 1552 and 1572 gaps, but also will take a while (smaller tasks recommended)
# worktodo will have an ETA

=== output file format: === (location: output/gaps_<start>e12_<end>e12_min<mingap>_<name>.txt)
===== PRIME GAP REPORT =====
Minimum gap size: <mingap>
Gaps >=1200: <x> (or whatever hundred is at least as large as mingap)
Gaps >=1300: <x>
... keep going until there are none left
Largest gap: <size> <prime>

All gaps >= <mingap>: # format: <gapsize> <startprime> <merit> <name>
1572 18571673432051830099 35.430806 B.Kehrig
1552 18470057946260698231 34.984359 B.Kehrig # (these would be in the opposite order if SORT_OUTPUPT_BY_GAPSIZE=0)


How do I write TESTS???
Use a compile-time variable RUN_TESTS
if set, it will do a run with some specified settings (start num, block size, etc...) to stay consistent
it will look at certain values from sieving and pseudoprime sieving and compare that to the correct values
For testing the primality tests, ???

I'm pretty sure that for the 4090, the bottleneck is MEMORY ACCESSES, so it cannot benefit from any more fermat test upgrades.
For my laptop GPU, speeding up the Fermat can still help.
*/