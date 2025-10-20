#include <array>
#include <getopt.h>
#include <iostream>
#include <math.h>
#include <cuda_profiler_api.h>
#include <cassert>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <unistd.h>

// #include "fermat/optimized.h"
#define FERMAT_TEST fermatTestPerig

#define count_set_bits_64 __builtin_popcountll

typedef unsigned __int128 uint128_t;

#define WIPE_LINE "\r\033[K"
#define END_OF_RANGE ~0

#define RESULT_LIST_SIZE 131072

#ifndef RUN_TESTS
#define RUN_TESTS 0
#endif

#ifndef GPU_BLOCKS
#define GPU_BLOCKS 192
#endif

#ifndef GPU_THREADS
#define GPU_THREADS 512
#endif

#if RUN_TESTS
    #define BLOCK_SIZE 46080000000
    #define WORD_LENGTH 120
    #define WORD_SIEVING_LENGTH 120
    #define MIN_GAP_SIZE 720
    #define HIGH_64 1
#else

#ifndef MIN_GAP_SIZE
#define MIN_GAP_SIZE 960 // low enough that it will remind people to set it properly
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 92160000000 // we don't actually need to add the UL as far as I know
#endif

#ifndef WORD_LENGTH
#define WORD_LENGTH 240
#endif

#ifndef WORD_SIEVING_LENGTH
#define WORD_SIEVING_LENGTH 120
#endif

#endif // RUN_TESTS block

#ifndef PROGRESS_UPDATE_BLOCKS
#define PROGRESS_UPDATE_BLOCKS 1
#endif

#if (WORD_LENGTH != WORD_SIEVING_LENGTH)
    #define SIEVE_BY_30 1
#else
    #ifndef SIEVE_BY_30
    #define SIEVE_BY_30 1
    #endif
#endif

#if SIEVE_BY_30 // don't change these values!!!
#define SIEVING_DUPLICATED_PRIMES 136
#else
#define SIEVING_DUPLICATED_PRIMES 392
#endif

__constant__ uint8_t SIEVE_VALUE_TO_POS[WORD_LENGTH/2];

__constant__ uint8_t SIEVE_POS_TO_VALUE[32];

__constant__ bool IS_COPRIME_30[15] = {
    1,0,0,1,0,1,1,0,1,1,0,1,0,0,1,
};

__constant__ uint32_t WORD_INVERSES[WORD_LENGTH/2];
__constant__ uint32_t INVERSES_30[15] = {
    1,0,0,13,0,11,7,0,23,19,0,17,0,0,29,
};
__constant__ uint8_t NEXT_SIEVE_HIT[32][WORD_LENGTH/2][3];


#if RUN_TESTS
#define SHARED_SIZE_WORDS 12288 // DON'T CHANGE THESE VALUES!!!!!
#define NUM_MEDIUM_PRIMES (1536 - SIEVING_DUPLICATED_PRIMES)
#define NUM_SMALL_PRIME_WHEELS 2
#else
#ifndef SHARED_SIZE_WORDS
#define SHARED_SIZE_WORDS 8192 // for some reason it can be slightly faster to not use all 12288 bytes
#endif
#ifndef NUM_MEDIUM_PRIMES_BASE
#define NUM_MEDIUM_PRIMES_BASE 4096
#endif

#ifndef PROPORTION_OF_BLOCKS_FOR_SIEVING
#define PROPORTION_OF_BLOCKS_FOR_SIEVING 0.5
#endif

#define NUM_MEDIUM_PRIMES (NUM_MEDIUM_PRIMES_BASE - SIEVING_DUPLICATED_PRIMES)

#ifndef NUM_SMALL_PRIME_WHEELS
#define NUM_SMALL_PRIME_WHEELS 4 // this can't be higher than 4
#endif
#endif

// The algorithm to sieve large primes is currently too slow to be useful, this value should be kept at 0
#define NUM_LARGE_PRIMES 0 // MAKING THIS HIGHER THAN 1024 BREAKS THE ASSERTION FOR SOME REASON

#if (NUM_SMALL_PRIME_WHEELS == 1)
#define NUM_SMALL_PRIMES 10
#elif (NUM_SMALL_PRIME_WHEELS == 2)
#define NUM_SMALL_PRIMES 15
#elif (NUM_SMALL_PRIME_WHEELS == 3)
#define NUM_SMALL_PRIMES 19
#elif (NUM_SMALL_PRIME_WHEELS == 4)
#define NUM_SMALL_PRIMES 23
#endif

void handleError(cudaError_t err, const char *file, int line) {
    // from "CUDA By Example"
    if (err != cudaSuccess) {
        printf("ERROR: '%s' in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(1);
    }
}
#define HANDLE_ERROR(err) (handleError(err, __FILE__, __LINE__))

struct PrimeGap {
    uint128_t startPrime;
    uint32_t gap;
};
bool compareByPrime(const PrimeGap &a, const PrimeGap &b) {
    return a.startPrime < b.startPrime;
}

struct WordPosition {
    uint32_t wordIdx;
    uint8_t posInWord;
};

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
        // wheels at all. 
        // TODO: What variable do I have to mark as volatile so I don't need this hacky code?
        printf("%d ", word % 1000);
    }
    return word;
}


__global__ void makeSmallPrimeWheels(uint32_t* wheel1, uint32_t* wheel2, uint32_t* wheel3, uint32_t* wheel4) {
    // We still make all 4 wheels, even if we don't end up using all of them (NUM_SMALL_PRIME_WHEELS)
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




__host__ __device__ uint64_t lo19(uint128_t n) {
    return (uint64_t) (n % 10000000000000000000UL);
}
__host__ __device__ uint64_t hi19(uint128_t n) {
    return (uint64_t) (n / 10000000000000000000UL) % 10000000000000000000UL;
}

__device__ void printBigNum(uint128_t num) {
    // works with numbers between 10^19 and 10^38
    printf("%lu%019lu\n", hi19(num), lo19(num));
}




// ===== Thanks to Perig for this code ===== 
//       vvvvvvvvvvvvvvvvvvvvvvvvvvvvv

// (a::b) <<= c
__host__ __device__ static inline __attribute__((always_inline))
void my_shld64(uint64_t * a, uint64_t * b, uint64_t c)
{
#if PARANOID
	assert(c < 64);
#endif
	if (c == 0) {
	} else {
		(*a) = ((*a) << c) | ((*b) >> (64 - c));
		(*b) <<= c;
	}
}

// borrow::diff = a - b - borrow_in
static inline __attribute__((always_inline))
__host__ __device__ uint8_t my_sbb64(uint8_t borrow_in, uint64_t a, uint64_t b, uint64_t * diff)
{
    uint64_t tmp1 = a - borrow_in;
    uint8_t borrow = (tmp1 > a);

    uint64_t tmp2 = tmp1 - b;
    borrow |= (tmp2 > tmp1);
    *diff = tmp2;
    return borrow;
}

// count leading zeroes in binary representation
static inline __attribute__((always_inline))
__host__ __device__ uint64_t my_clz64(uint64_t n)
{
#ifdef __CUDA_ARCH__
    return __clzll(n);
#else
	if (n == 0)
		return 64;
	uint64_t r = 0;
	if ((n & (0xFFFFFFFFull << 32)) == 0)
		r += 32, n <<= 32;
	if ((n & (0xFFFFull << 48)) == 0)
		r += 16, n <<= 16;
	if ((n & (0xFFull << 56)) == 0)
		r += 8, n <<= 8;
	if ((n & (0xFull << 60)) == 0)
		r += 4, n <<= 4;
	if ((n & (0x3ull << 62)) == 0)
		r += 2, n <<= 2;
	if ((n & (0x1ull << 63)) == 0)
		r += 1;
	return r;
#endif
}

static inline __attribute__((always_inline))
__host__ __device__ uint64_t montgomeryInverse64(uint64_t mod_lo)
{
	uint64_t x = (3ull * mod_lo) ^ 2ull;	// 5 bits acurate
	uint64_t t = 1ull - mod_lo * x;
	x *= 1 + t;		// 10 bits accurate
	t *= t;
	x *= 1 + t;		// 20 bits accurate
	t *= t;
	x *= 1 + t;		// 40 bits accurate
	t *= t;
	x *= 1 + t;		// 80 bits accurate , i.e. > 64 bits
	return 0 - x;
}

// subtract the modulus 'mod' multiple times from the input number 'res', if needed
static inline __attribute__((always_inline))
__host__ __device__ void ciosSubtract128(uint64_t * res_lo, uint64_t * res_hi, uint64_t mod_lo, uint64_t mod_hi)
{
	uint64_t n_lo, n_hi;
	uint64_t t_lo, t_hi;
	uint8_t b;
	n_lo = *res_lo;
	n_hi = *res_hi;
	// save, subtract the modulus until a borrows occurs
	do {
		t_lo = n_lo;
		t_hi = n_hi;
		b = my_sbb64(0, n_lo, mod_lo, &n_lo);
		b = my_sbb64(b, n_hi, mod_hi, &n_hi);
	}
	while (b == 0);
	// get the saved values
	*res_lo = t_lo;
	*res_hi = t_hi;
}

static inline __attribute__((always_inline))
__host__ __device__ void ciosConstants128(uint64_t mod_lo, uint64_t mod_hi, uint64_t * magic_lo, uint64_t * magic_hi)
{
	// computes 2^128 % mod
	uint128_t m = ((uint128_t) mod_hi << 64) + mod_lo;
	uint128_t t = -m;	// 2^128-m
	t %= m;			// (2^128-m) % m
	*magic_lo = (uint64_t) t;
	*magic_hi = (uint64_t) (t >> 64);
#if PARANOID
	assert(*magic_hi <= mod_hi);
#endif
}

static inline __attribute__((always_inline))
__host__ __device__ void ciosModSquare128(uint64_t * res_lo, uint64_t * res_hi, uint64_t mod_lo, uint64_t mod_hi, uint64_t mmagic)
{
	uint64_t n_lo = *res_lo, n_hi = *res_hi;
	uint128_t cs, cc;
	uint64_t t0, t1, t2, m;

	cc = (uint128_t) n_lo *n_lo;	// #1
	t0 = (uint64_t) cc;
	cc = cc >> 64;
	cc += (uint128_t) n_lo *(n_hi + n_hi);	// #2
	t1 = (uint64_t) cc;
	cc = cc >> 64;
	t2 = (uint64_t) cc;
#if PARANOID
	assert(cc >> 64 == 0);
#endif

	m = t0 * mmagic;	// #3
	cs = (uint128_t) m *mod_lo;	// #4
	cs += t0;
	cs = cs >> 64;

    cs += (uint128_t) m *mod_hi; // CAN OPTIMIZE
	cs += t1;
    
	t0 = (uint64_t) cs;
	cs = cs >> 64;
	cs += t2;
	t1 = (uint64_t) cs;
	cs = cs >> 64;
	t2 = (uint64_t) cs;
#if PARANOID
	assert(cs >> 64 == 0);
#endif

	cc = (uint128_t) n_hi *n_hi;	// #6
	cc += t1;
	t1 = (uint64_t) cc;
	cc = cc >> 64;
	cc += t2;
	t2 = (uint64_t) cc;
#if 0
	// not necessary with 2-bits guard
	cc = cc >> 64;
	uint64_t t3 = (uint64_t) cc;
	assert(t3 == 0);
#endif
#if PARANOID
	assert(cc >> 64 == 0);
#endif

	m = t0 * mmagic;	// #3
	cs = (uint128_t) m *mod_lo;	// #8
	cs += t0;
	cs = cs >> 64;
    
    cs += (uint128_t) m *mod_hi; // CAN OPTIMIZE
    

	cs += t1;
	t0 = (uint64_t) cs;
	cs = cs >> 64;
	cs += t2;
	t1 = (uint64_t) cs;
#if 0
	// not necessary with 2-bits guard
	cs = cs >> 64;
	cs += t3;
	t2 = (uint64_t) cs;
	assert(t2 == 0);
#endif
#if PARANOID
	assert(cs >> 64 == 0);
#endif

	*res_lo = t0;
	*res_hi = t1;

}

__host__ __device__ inline void ciosModSquare3_128(uint64_t * res_lo, uint64_t * res_hi, uint64_t mod_lo,
                                                   uint64_t mod_hi, uint64_t mmagic)
{
	ciosModSquare128(res_lo, res_hi, mod_lo, mod_hi, mmagic);
	ciosModSquare128(res_lo, res_hi, mod_lo, mod_hi, mmagic);
	ciosModSquare128(res_lo, res_hi, mod_lo, mod_hi, mmagic);
}

#ifdef HIGH_64
__host__ __device__ bool ciosFermatTest128(uint64_t n_lo) {
#else
__host__ __device__ bool ciosFermatTest128(uint64_t n_lo, uint64_t HIGH_64) {
#endif
#if PARANOID
	assert((n_lo & 1) == 1);
#endif
    //const uint64_t n_hi = <stuff here>;

	uint64_t res_lo, res_hi;
	uint64_t one_lo, one_hi;
	int bit;
	// constant -1/m mod 2^64
	uint64_t mmagic = montgomeryInverse64(n_lo);

	// enter montgomery domain
	// constant 2^128 mod m
	ciosConstants128(n_lo, HIGH_64, &one_lo, &one_hi);
	res_hi = one_hi;
	res_lo = one_lo;

    if (HIGH_64 == 0) {
        bit = 64 - my_clz64(n_lo);
        uint64_t msb_bits = bit < 5 ? bit - 1 : 3;
        uint64_t msb_mask = (1 << msb_bits) - 1;
        bit -= msb_bits;
        my_shld64(&res_hi, &res_lo, (n_lo >> bit) & msb_mask);

    } else {
        bit = 64 - my_clz64(HIGH_64);
        uint64_t msb_bits = bit < 4 ? bit : 3;
        uint64_t msb_mask = (1 << msb_bits) - 1;
        bit -= msb_bits;
        my_shld64(&res_hi, &res_lo, (HIGH_64 >> bit) & msb_mask);

        while (bit >= 3) {
            bit -= 3;
            // square and reduce
            ciosModSquare3_128(&res_lo, &res_hi, n_lo, HIGH_64, mmagic);
            // shift
            my_shld64(&res_hi, &res_lo, ((HIGH_64 >> bit) & 7));
        }

        while (bit) {
            bit -= 1;
            // square and reduce
            ciosModSquare128(&res_lo, &res_hi, n_lo, HIGH_64, mmagic);
            // shift
            my_shld64(&res_hi, &res_lo, ((HIGH_64 >> bit) & 1));
        }

        bit = 64;
    }
	//}
	while (bit >= 5) {
		bit -= 3;
		// square and reduce
		ciosModSquare3_128(&res_lo, &res_hi, n_lo, HIGH_64, mmagic);
		// shift
		my_shld64(&res_hi, &res_lo, ((n_lo >> bit) & 7));
	}

	while (bit > 1) {
		bit -= 1;
		// square and reduce
		ciosModSquare128(&res_lo, &res_hi, n_lo, HIGH_64, mmagic);
		// shift
		my_shld64(&res_hi, &res_lo, ((n_lo >> bit) & 1));
	}

	// make sure result is strictly less than the modulus
	ciosSubtract128(&res_lo, &res_hi, n_lo, HIGH_64);

	uint64_t legendre = ((n_lo >> 1) ^ (n_lo >> 2)) & 1;	// shortcut calculation of legendre symbol

	uint64_t m1_lo;
	uint64_t m1_hi;
	uint8_t c;
	c = my_sbb64(0, n_lo, one_lo, &m1_lo);
	my_sbb64(c, HIGH_64, one_hi, &m1_hi);

	return ((res_lo == (legendre ? m1_lo : one_lo)) && (res_hi == (legendre ? m1_hi : one_hi)));
}


// ciosFermatTest128 should be used for up to 119 bits
// montgomeryFermatTest128 should be used fro 120-127 bits



//       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// ===== Thanks to Perig for this code ===== (end)

// POTENTIAL OPTIMIZATION: REUSE MAGIC CALCULATIONS FOR MULTIPLE FERMAT TESTS


/*
__host__ __device__ bool fermatTest65(uint128_t n, uint32_t delta, uint64_t orig_magic, double orig_derivative) {
    // estimate of magic1(n) is:  magicp - 2^128*d//p^2
    // orig_magic should be precomputed as get_magic1(start)
    // orig_derivative should be 2^128 / start^2
    uint64_t magic = orig_magic - (uint64_t) (orig_derivative*delta); // this might be 1 more or 1 less than correct
    uint128_t test = ((uint128_t) (magic+2)) * n;
    if (test >= n+n+n) {
        //printf("BAD FERMAT TEST: n=%lu%019lu, delta=%u orig_magic=%lu orig_derivative=%f magic=%lu test=%lu%019lu\n",
        //    hi19(n), lo19(n), delta, orig_magic, orig_derivative, magic, hi19(test), lo19(test));
        //assert(false);
    }
    if (test >= n+n) magic--;// TODO: CAN WE JUST GET RID OF THESE?
    if (test < n) magic++;
    // where d = n-p, aand 2^128/p^2 is precalculated as a double
    return ciosFermatTest128(n, magic);
}
*/

__host__ __device__ bool fermatTestPerig(uint128_t n, uint32_t delta, uint64_t orig_magic, double orig_derivative) {
    // all arguments other than n are unused - we just keep it the same format as fermatTest65
    //uint64_t magic1 = my_getMagic1(n);
#ifdef HIGH_64
    return ciosFermatTest128((uint64_t) n);
#else
    return ciosFermatTest128((uint64_t) n, (uint64_t) (n>>64));
#endif
}



__device__ void sieveSmallPrimes(uint32_t* sieve, uint32_t sieveLengthWords, uint128_t start,
                                 uint32_t* smallPrimeWheel1, uint32_t* smallPrimeWheel2,
                                 uint32_t* smallPrimeWheel3, uint32_t* smallPrimeWheel4) {

    // sieve should be in SHARED MEMORY for this function to work properly
    int wheel1Idx = (start/WORD_LENGTH + threadIdx.x) % (7*11*13*17*19*23*29);
#if (NUM_SMALL_PRIME_WHEELS >= 2)
    int wheel2Idx = (start/WORD_LENGTH + threadIdx.x) % (31*37*41*43*47);
#endif
#if (NUM_SMALL_PRIME_WHEELS >= 3)
    int wheel3Idx = (start/WORD_LENGTH + threadIdx.x) % (53*59*61*67);
#endif
#if (NUM_SMALL_PRIME_WHEELS >= 4)
    int wheel4Idx = (start/WORD_LENGTH + threadIdx.x) % (71*73*79*83);
#endif
    for (uint32_t i = threadIdx.x; i < sieveLengthWords; i += blockDim.x) {
        // We cannot replace the atomicOr with a non-atomic operation, because that might skip sieving out some values
        // and we must not miss any (because of pseudoprimes)
        uint32_t mask = smallPrimeWheel1[wheel1Idx];
        wheel1Idx += blockDim.x;
        wheel1Idx = (wheel1Idx >= 7*11*13*17*19*23*29) ? (wheel1Idx - 7*11*13*17*19*23*29) : wheel1Idx;
#if (NUM_SMALL_PRIME_WHEELS >= 2)
        mask |= smallPrimeWheel2[wheel2Idx];
        wheel2Idx += blockDim.x;
        wheel2Idx = (wheel2Idx >= 31*37*41*43*47) ? (wheel2Idx - 31*37*41*43*47) : wheel2Idx;
#endif
#if (NUM_SMALL_PRIME_WHEELS >= 3)
        mask |= smallPrimeWheel3[wheel3Idx];
        wheel3Idx += blockDim.x;
        wheel3Idx = (wheel3Idx >= 53*59*61*67) ? (wheel3Idx - 53*59*61*67) : wheel3Idx;
#endif
#if (NUM_SMALL_PRIME_WHEELS >= 4)
        mask |= smallPrimeWheel4[wheel4Idx];
        wheel4Idx += blockDim.x;
        wheel4Idx = (wheel4Idx >= 71*73*79*83) ? (wheel4Idx - 71*73*79*83) : wheel4Idx;
#endif

        atomicOr(&sieve[i], mask);
    }
    __syncthreads();
}

__device__ void sieveMediumLargePrimesInner(uint32_t* sieve, uint32_t sieveLengthWords, uint128_t start,
                                            uint32_t p, uint32_t startBit, uint32_t numBits) {
    // This function is by far the most performance-sensitive part of the sieving step
#if SIEVE_BY_30
    uint32_t pInv = INVERSES_30[(p%30)/2];
    // if start/30 is under 2^64, then we can convert to uint64 before the "% p" for a ~0.5% speedup
    uint32_t precalculated = p - (start/30) % p;
    for (uint32_t bit=startBit; bit<startBit+numBits; bit++) {
        // TODO: I should be able to precalculate this entire thing
        // only need to precalculate (8 bits) * (# of sieving primes) words
        // we should put this in global memory (???), ordered like this:
        // [prime0bit0, prime1bit0, prime2bit0 .... prime0bit1 ...]

        /*
        SEGMENTED SIEVE FOR LARGE PRIMES: (not implemented yet, first try was slower than it was before)
        Suppose we are doing a segmented sieve with 32768 primes.
        The interval of 92,160,000,000 is split into 192 parts for the 192 blocks, each gets 480,000,000.
        The width of one segment is 8192 * 120 = 983,040 which is about 488.28125 (488 or 489) segments per block.
        32768 primes / 256 threads = 128 primes/thread.
        We will have thread 0 do primes 0,256,512,768...32512 to even out the work.
        We have a local list of size 128, containing the next "hit" for each of the 128 primes.
        For each segment, we just have to iterate over the list, and whenever its "hit" is in the current segment,
            sieve it out and increment it (and repeat until it's no longer in the segment)
        */
        uint32_t startByte = (p * ((SIEVE_POS_TO_VALUE[bit] * pInv)%30)) / 30 + precalculated;
#if (WORD_LENGTH == 240)
        if (startByte % 2) startByte += p;
        startByte /= 2;
#endif
        if (startByte >= p) startByte -= p;
        uint8_t mask = 1 << bit;
        for (int32_t byte=startByte; byte<sieveLengthWords*4; byte += p) {
            atomicOr(&sieve[byte/4], mask << ((byte%4)*8));
        }
    }
#else
    uint32_t pInv = WORD_INVERSES[(p%WORD_LENGTH)/2];
    uint32_t precalculated = p - (start/WORD_LENGTH) % p;
    for (uint32_t bit=startBit; bit<startBit+numBits; bit++) {
        uint32_t startByte = (p * ((SIEVE_POS_TO_VALUE[bit] * pInv)%WORD_LENGTH)) / WORD_LENGTH + precalculated;
        // the formula is actually startByte <- startByte + x*p for some x, such that the new startByte is 0 mod N
        // where N is WORD_LENGTH / WORD_SIEVING_LENGTH
        // startByte + x*p = 0 (mod N)
        // x = -startByte * pInv (mod N)
        // For all N <= 4, we have p^-1 mod N = p mod N
        // so startByte = startByte + (((N-startByte) * p) % N)*p
        // then, we have to divide by N
        // (6 + (((n-6)*p)%n)*p)//n
        startByte = startByte - (startByte >= p)*p;
        //startByte = (startByte + (((N - startByte + p) * p - 1) % N) * 89) / N;
        uint32_t mask = 1 << bit;
        for (int32_t byte=startByte; byte<sieveLengthWords; byte += p) {
            atomicOr(&sieve[byte], mask);
        }
    }
#endif
}


__device__ void sieveMediumPrimes(uint32_t* sieve, uint32_t sieveLengthWords, uint128_t start,
                                  uint32_t* primeList, uint32_t primeCount) {
#if SIEVE_BY_30
    uint32_t bitIdx, numBits, pIdx;
    if (threadIdx.x < 64) { // the first 64 threads will split primes 71,73,79,83,89,97,101,103 into 8 threads/prime
        bitIdx = (threadIdx.x % 8);
        numBits = 1;
        pIdx = threadIdx.x/8;
    } else if (threadIdx.x < 128) { // the next 64 threads will split the 16 primes 107-191 into 4 threads/prime
        bitIdx = (threadIdx.x % 4)*2;
        numBits = 2;
        pIdx = (threadIdx.x-64)/4 + 8;
    } else if (threadIdx.x < 192) { // 32 primes 193-379 will get 2 threads/prime
        bitIdx = (threadIdx.x % 2)*4;
        numBits = 4;
        pIdx = (threadIdx.x-128)/2 + 8+16;
    } else { // the remaining threads each handle 1 prime by themselves
        bitIdx = 0;
        numBits = 8;
        pIdx = threadIdx.x-SIEVING_DUPLICATED_PRIMES;
    }
    sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, primeList[pIdx], bitIdx, numBits);
    
    // If we are sieving more than 512 primes, we do more iterations as needed
    for (uint32_t pidx = threadIdx.x+blockDim.x-SIEVING_DUPLICATED_PRIMES; pidx < primeCount; pidx += blockDim.x) {
        sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, primeList[pidx], 0, 8);
    }
#else
    if (threadIdx.x < 128) {
        uint32_t bitIdx = (threadIdx.x % 16)*2;
        uint32_t p = primeList[threadIdx.x/16];
        sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, p, bitIdx, 2);
    } else if (threadIdx.x < 256) {
        uint32_t bitIdx = (threadIdx.x % 8)*4;
        uint32_t p = primeList[(threadIdx.x-128)/8 + 8];
        sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, p, bitIdx, 4);
    } else if (threadIdx.x < 384) {
        uint32_t bitIdx = (threadIdx.x % 4)*8;
        uint32_t p = primeList[(threadIdx.x-256)/4 + 24];
        sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, p, bitIdx, 8);
    } else {
        uint32_t bitIdx = (threadIdx.x % 2)*16;
        uint32_t p = primeList[(threadIdx.x-384)/2 + 56];
        sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, p, bitIdx, 16);
    }
    for (uint32_t pidx = threadIdx.x+blockDim.x-SIEVING_DUPLICATED_PRIMES; pidx < primeCount; pidx += blockDim.x) {
        uint32_t p = primeList[pidx];
        sieveMediumLargePrimesInner(sieve, sieveLengthWords, start, p, 0, 32);
    }
#endif
    __syncthreads();
}


__device__ void sieveLargePrimes(uint32_t* sieve, uint32_t sieveLengthWords, uint128_t sieveStart,
                                 uint32_t* primeList, uint32_t primeCount,
                                 WordPosition* nextLargePrimeSieveHits, uint32_t* primesModWordLength) {
    /*
    with ./prime_gaps.out 20733746418401443920 where mingap=1680 and we don't check with cpu:
    We SHOULD see: nothing in first block, then 1696, then 1920,1850,1740, then 1740,1858, then 1686
    */
    for (int pidx=0; pidx<NUM_LARGE_PRIMES/256; pidx++) {
        int idx = pidx*256+threadIdx.x;
        uint32_t p = primeList[idx];
        uint32_t increaseIndex = primesModWordLength[idx]/2;

        while (nextLargePrimeSieveHits[pidx].wordIdx < sieveLengthWords) {
            /*
            int posInWord = SIEVE_VALUE_TO_POS[(nextLargePrimeSieveHits[pidx] % WORD_LENGTH) / 2];
            atomicOr(&sieve[nextLargePrimeSieveHits[pidx] / WORD_LENGTH], 1 << posInWord);
            nextLargePrimeSieveHits[pidx] += p * NEXT_SIEVE_HIT[posInWord][increaseIndex];
            */
            atomicOr(&sieve[nextLargePrimeSieveHits[pidx].wordIdx], 1 << nextLargePrimeSieveHits[pidx].posInWord);

            uint8_t* data = NEXT_SIEVE_HIT[nextLargePrimeSieveHits[pidx].posInWord][increaseIndex];

            nextLargePrimeSieveHits[pidx].posInWord = data[1];
            nextLargePrimeSieveHits[pidx].wordIdx += data[2] + p * data[0];
        }
        nextLargePrimeSieveHits[pidx].wordIdx -= sieveLengthWords;
    }
    __syncthreads();
}




__device__ void sievePseudoprimes(uint32_t* sieve, uint32_t sieveLengthWords, uint128_t start,
                                  uint32_t* primeList, uint32_t* rhoList, uint32_t primeCount,
                                  uint32_t numBlocks) {
    // sieve should be in GLOBAL MEMORY for this function to work properly

    // We are sieving for entries that are congruent to p mod p*rho(p), because this is guaranteed
    // to remove all 2-PSPs that have p as a factor.

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

        uint32_t pTimesRhoModWordLen = pTimesRho % WORD_LENGTH;
        uint64_t pTimesRhoDivWordLen = pTimesRho / WORD_LENGTH;
        while (currentWord < sieveLengthWords) {
            // Update the sieve
            if (currentPosInWord < WORD_SIEVING_LENGTH && IS_COPRIME_30[(currentPosInWord % 30) / 2]) {
                uint8_t wordPos = SIEVE_VALUE_TO_POS[currentPosInWord / 2];
                if (wordPos || (currentPosInWord==1)) {
                    atomicOr(&sieve[currentWord], 1 << wordPos);
                }
            }

            // Find the next position
            currentPosInWord += pTimesRhoModWordLen;
            currentWord += pTimesRhoDivWordLen + (currentPosInWord >= WORD_LENGTH);
            currentPosInWord -= WORD_LENGTH * (currentPosInWord >= WORD_LENGTH);
        }
    }
    __syncthreads();
}

__device__ void sieveAll(uint32_t* globalSieve, uint128_t sieveStart, uint32_t sieveLengthWords,
                         uint32_t* primeList, uint32_t* rhoList, uint32_t* primeMods, uint32_t primeCount,
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
    // We have to have each thread block run through consecutive shared memory blocks
    // so that we can use the result from the previous shared block to cut down on computation for the next
    // (THE ABOVE IS NOT IMPLEMENTED YET)
    uint32_t numSharedBlocks = sieveLengthWords / SHARED_SIZE_WORDS;
    uint32_t firstSharedBlockIdx = (uint32_t) (((double) numSharedBlocks) * blockIdx.x / numBlocks);
    uint32_t lastSharedBlockIdx = (uint32_t) (((double) numSharedBlocks) * (blockIdx.x+1) / numBlocks);

    // the extra stuff here is only here to prevent crashing when we have 0 large primes
    WordPosition nextLargePrimeSieveHits[NUM_LARGE_PRIMES/256 + (NUM_LARGE_PRIMES < 256)];

    /*
    // This for loop initializes the segmented sieve for large primes
    for (int i=0; i<NUM_LARGE_PRIMES/blockIdx.x; i++) {
        // thread N will deal with primes N, N+256, N+512, N+768... if there are 256 threads per block
        int pidx = i*blockIdx.x + threadIdx.x; // can optimize, increment pidx each time instead of multiplying
        int p = primeList[NUM_SMALL_PRIMES+NUM_MEDIUM_PRIMES+pidx];
        
        int testOffset = p - ((sieveStart + ((uint64_t) firstSharedBlockIdx)*SHARED_SIZE_WORDS*WORD_LENGTH) % p);
        if (testOffset % 2 == 0) testOffset += p;
        while (true) {
            int mod = testOffset % WORD_LENGTH;
            if (SIEVE_VALUE_TO_POS[mod/2] || (mod == 1)) {
                break;
            }
            testOffset += p*2;
        }
        nextLargePrimeSieveHits[i].wordIdx = testOffset / WORD_LENGTH;
        nextLargePrimeSieveHits[i].posInWord = testOffset % WORD_LENGTH;
    }
    */

    __syncthreads();
    
    for (uint64_t sharedBlockIdx = firstSharedBlockIdx; sharedBlockIdx < lastSharedBlockIdx; sharedBlockIdx++) {
        // Reset the shared memory to 0, since it doesn't necessarily start out that way
        for (int idx=threadIdx.x; idx<SHARED_SIZE_WORDS; idx+=blockDim.x) {
            sharedSieve[idx] = 0;
        }
        
    // the pseudoprime will be in the 2nd block
    // idx 38685076147
    // word idx 322375634
    // shared block number 26234 (start=21693774589723607040)
        sieveSmallPrimes(sharedSieve, SHARED_SIZE_WORDS, sieveStart + sharedBlockIdx*SHARED_SIZE_WORDS*WORD_LENGTH,
                         smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4);
#if RUN_TESTS
        bool good = true;
        if (threadIdx.x == 97 && (sieveStart + sharedBlockIdx*SHARED_SIZE_WORDS*WORD_LENGTH)%1000000000 == 723607040) {
            if (sharedSieve[12242] != 1683790270) {
                //good = false;
                //printf("KERNEL 1 TEST 1 FAILED! (expected=%u actual=%u)\n", 1683790270, sharedSieve[12242]);
            }
        }
#endif

        if (NUM_MEDIUM_PRIMES > 0) {
            sieveMediumPrimes(sharedSieve, SHARED_SIZE_WORDS, sieveStart + sharedBlockIdx*SHARED_SIZE_WORDS*WORD_LENGTH,
                            primeList+NUM_SMALL_PRIMES, NUM_MEDIUM_PRIMES);
        }

        if (NUM_LARGE_PRIMES > 0) {
            assert(false);
            sieveLargePrimes(sharedSieve, SHARED_SIZE_WORDS, sieveStart + sharedBlockIdx*SHARED_SIZE_WORDS*WORD_LENGTH,
                             primeList+NUM_SMALL_PRIMES+NUM_MEDIUM_PRIMES, NUM_LARGE_PRIMES, nextLargePrimeSieveHits,
                             primeMods+NUM_SMALL_PRIMES+NUM_MEDIUM_PRIMES);
        }
#if RUN_TESTS
        if (threadIdx.x == 97 && (sieveStart + sharedBlockIdx*SHARED_SIZE_WORDS*WORD_LENGTH)%1000000000 == 723607040) {
            if (sharedSieve[12242] != 1834868158) {
                good = false;
                printf("KERNEL 1 TEST 2 FAILED! (expected=%u actual=%u)\n", 1834868670, sharedSieve[12242]);
            }
            if (good) {printf("Passed kernel 1 tests.\n");}
        }
#endif

        for (int sharedIdx=threadIdx.x; sharedIdx<SHARED_SIZE_WORDS; sharedIdx += blockDim.x) {
            atomicOr(&globalSieve[sharedBlockIdx*SHARED_SIZE_WORDS + sharedIdx], sharedSieve[sharedIdx]);
        }
    
        if (threadIdx.x%32==0 && blockIdx.x%32==0 && sharedBlockIdx == lastSharedBlockIdx-1) {
            //printf("e %u %d %d\n", globalSieve[56349497], threadIdx.x, blockIdx.x);
        }
    }
    
    sievePseudoprimes(globalSieve, sieveLengthWords, sieveStart,
                      primeList+NUM_SMALL_PRIMES+NUM_MEDIUM_PRIMES+NUM_LARGE_PRIMES,
                      rhoList+NUM_SMALL_PRIMES+NUM_MEDIUM_PRIMES+NUM_LARGE_PRIMES,
                      primeCount-NUM_SMALL_PRIMES-NUM_MEDIUM_PRIMES-NUM_LARGE_PRIMES, numBlocks);
    
}

__global__ void kernel(uint32_t* globalSieve, uint128_t sieveStart, uint32_t sieveLengthWords,
                       uint32_t* primeList, uint32_t* rhoList, uint32_t* primeMods, uint32_t primeCount,
                       uint32_t* smallPrimeWheel1, uint32_t* smallPrimeWheel2,
                       uint32_t* smallPrimeWheel3, uint32_t* smallPrimeWheel4) {
    sieveAll(globalSieve, sieveStart, sieveLengthWords, primeList, rhoList, primeMods, primeCount,
        smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4, gridDim.x);
}



// ========== THIS IS THE START OF THE 2ND PART OF THE CODE, PARSING THE SIEVE TO FIND PRIME GAPS ========== //


__device__ uint64_t findNextUnsieved(uint32_t* sieve, uint64_t sieveLengthWords, int64_t bitPosition) {
    // bitPosition treats the entire sieve as a bitSet
    // Gets the next unsieved number from a given position,
    //   starting with AND INCLUDING bitPosition
    
    if ((bitPosition < 0) || (bitPosition >= sieveLengthWords*32)) return END_OF_RANGE;
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

__device__ uint64_t findPrevUnsieved(uint32_t* sieve, uint64_t sieveLengthWords, int64_t bitPosition) {
    // bitPosition treats the entire sieve as a bitSet
    // Gets the previous unsieved number from a given position,
    //   starting with AND INCLUDING bitPosition
    
    if ((bitPosition < 0) || (bitPosition >= sieveLengthWords*32)) return END_OF_RANGE;
    int64_t wordIdx = bitPosition/32; // signed int, so we can compare it with 0 properly
    while (wordIdx >= 0) {
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

__device__ void findGaps(uint32_t* sieve, uint128_t start, uint64_t sieveLengthWords, uint32_t startBlock, uint32_t minGapSize,
                         PrimeGap* resultList, uint128_t* totalFirstPrime, uint128_t* totalLastPrime) {
    // sieve should be in GLOBAL MEMORY for this function to work properly
    uint32_t gridDimNew = gridDim.x - startBlock;
    uint32_t blockIdxNew = blockIdx.x - startBlock;

#if (WORD_LENGTH == 120) || (WORD_LENGTH == 240)
    const int MIN_GAP_SIZE_BITS = (minGapSize / (WORD_LENGTH/4)) * 8;
#else
    const int MIN_GAP_SIZE_BITS = (minGapSize / WORD_LENGTH) * 32;
#endif

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
    const uint128_t firstPrime = lastPrime;
    
    double orig_derivative = 1.0 / firstPrime / firstPrime * 4294967296.0 * 4294967296.0 * 4294967296.0 * 4294967296.0;
    uint64_t orig_magic = 0; //my_getMagic(firstPrime);


    while (!FERMAT_TEST(lastPrime, (uint32_t) (lastPrime-firstPrime), orig_magic, orig_derivative)) {

        if (bitPosition == END_OF_RANGE) {
            // this will only happen if the last thread doesn't have a single prime in it
            // which will almost certainly never happen (would require a gap of >1M)
            hitEndOfRange = true;
            printf("FOUND INSANELY LARGE PRIME GAP??? AROUND %lu%019lu\n", hi19(lastPrime), lo19(lastPrime));
            break;
        }
        bitPosition = findNextUnsieved(sieve, sieveLengthWords, ++bitPosition);
        lastPrime = getNumberFromSieve(start, bitPosition);
    }
    if (tidx == 0) *totalFirstPrime = lastPrime;
    
    __syncthreads();
    
    bool isPrime = false;
    if (hitEndOfRange) goto endLabel;

    bitPosition += MIN_GAP_SIZE_BITS;

    while (true) {
        bitPosition = findPrevUnsieved(sieve, sieveLengthWords, --bitPosition);
        if (bitPosition == END_OF_RANGE) {
            // we have to find the LAST prime in the range, to compare to the first prime in the next range
            bitPosition = sieveLengthWords*32;
            do {
                bitPosition = findPrevUnsieved(sieve, sieveLengthWords, --bitPosition);
                lastPrime = getNumberFromSieve(start, bitPosition);
            } while (!FERMAT_TEST(lastPrime, (uint32_t) (lastPrime-firstPrime), orig_magic, orig_derivative));
            *totalLastPrime = lastPrime;
            break;
        }
        uint128_t toTest = start + bitPosition/32*WORD_LENGTH + SIEVE_POS_TO_VALUE[bitPosition%32];
        if (toTest == lastPrime) {
            // found a large gap! but how large?
            bitPosition += MIN_GAP_SIZE_BITS;
            bitPosition = findNextUnsieved(sieve, sieveLengthWords, bitPosition);
            uint128_t upperPrime = getNumberFromSieve(start, bitPosition);
            while (!FERMAT_TEST(upperPrime, (uint32_t) (upperPrime-firstPrime), orig_magic, orig_derivative)) {
                if (bitPosition == END_OF_RANGE) {
                    *totalLastPrime = lastPrime;
                    goto endLabel;
                }
                bitPosition = findNextUnsieved(sieve, sieveLengthWords, ++bitPosition);
                upperPrime = getNumberFromSieve(start, bitPosition);
            }
            uint32_t gap = (uint32_t) (upperPrime - lastPrime);
            
            int resultIdx = atomicAdd(&resultList[0].gap, 1) + 1; // index 0 of the list keeps track of the length
            //assert(resultIdx != RESULT_LIST_SIZE-1); // will raise an error if we overflow the result list capacity
            if (resultIdx < RESULT_LIST_SIZE-1) {
                resultList[resultIdx].startPrime = lastPrime;
                resultList[resultIdx].gap = gap;
            }

            lastPrime = upperPrime;
            bitPosition += MIN_GAP_SIZE_BITS;
        } else {
            isPrime = FERMAT_TEST(toTest, (uint32_t) (toTest-firstPrime), orig_magic, orig_derivative);
            if (isPrime) lastPrime = toTest;
        }

        if (bitPosition >= limitBitPosition && isPrime) break;

        bitPosition += MIN_GAP_SIZE_BITS * isPrime;
        // for some reason, it doesn't work if I just put this in the while loop condition
    }

    endLabel:
    
    __syncthreads();
}


__global__ void kernel2(uint32_t* globalSieve, uint128_t sieveStart, uint64_t sieveLengthWords, uint32_t minGapSize, 
                        PrimeGap* resultList, uint128_t* firstPrimeInBlock, uint128_t* lastPrimeInBlock) {
    findGaps(globalSieve, sieveStart, sieveLengthWords, 0, minGapSize, resultList, firstPrimeInBlock, lastPrimeInBlock);
}


__global__ void kernelBoth(uint32_t* globalSieve1, uint32_t* globalSieve2, uint128_t sieveStart, uint32_t sieveLengthWords,
                           uint32_t minGapSize, uint32_t* primeList, uint32_t* rhoList, uint32_t* primeMods, uint32_t primeCount,
                           uint32_t* smallPrimeWheel1, uint32_t* smallPrimeWheel2,
                           uint32_t* smallPrimeWheel3, uint32_t* smallPrimeWheel4,
                           uint32_t numSieveBlocks, PrimeGap* resultList,
                           uint128_t* firstPrimeInBlock, uint128_t* lastPrimeInBlock) {
    if (blockIdx.x < numSieveBlocks) {
        sieveAll(globalSieve1, sieveStart, sieveLengthWords, primeList, rhoList, primeMods, primeCount,
            smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4, numSieveBlocks);
    } else {
        findGaps(globalSieve2, sieveStart - ((uint128_t) sieveLengthWords)*WORD_LENGTH, sieveLengthWords, numSieveBlocks,
                 minGapSize, resultList, firstPrimeInBlock, lastPrimeInBlock);
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

bool fermatTest84CPU(uint128_t n, uint32_t base) {
    uint128_t result = 1;
    for (int bit=84; bit>=1; bit--) {
        if ((n >> bit) & 1) {
            result = (result * base) % n; // with base=2 this can be simplified but we might use base>2
        }
        result = squareMod84CPU(result, n);
    }
    return result == 1;
}

bool fermatTest84CPUStrong(uint128_t n, uint32_t base) {
    uint128_t m = n-1;
    int zeros = 0;
    while (m % 2 == 0) {
        m /= 2;
        zeros++;
    }
    uint128_t result = 1;
    for (int bit=84; bit>=1; bit--) {
        if ((n >> bit) & 1) {
            result = (result * base) % n; // with base=2 this can be simplified but we might use base>2
        }
        if (bit == zeros && result == 1) return true;
        if (bit <= zeros && result == n-1) return true;
        result = squareMod84CPU(result, n);
    }
    return false;
}

bool isPrime84(uint128_t n) { // deterministic up to at least 2^78
    return (
        n%3 && n%5 && n%7 && n&11 && n&13 && n&17 && n&19 && n&23 &&
        fermatTest84CPUStrong(n, 2) && 
        fermatTest84CPUStrong(n, 3) && 
        fermatTest84CPUStrong(n, 5) && 
        fermatTest84CPUStrong(n, 7) && 
        fermatTest84CPUStrong(n, 11) && 
        fermatTest84CPUStrong(n, 13) && 
        fermatTest84CPUStrong(n, 17) && 
        fermatTest84CPUStrong(n, 19) && 
        fermatTest84CPUStrong(n, 23) && 
        fermatTest84CPUStrong(n, 29) && 
        fermatTest84CPUStrong(n, 31) && 
        fermatTest84CPUStrong(n, 37)
    );
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
        //printf("  Memory Clock Rate (MHz): %d\n",
        //        prop.memoryClockRate/1024);
        printf("  Memory Bus Width (bits): %d\n",
                prop.memoryBusWidth);
        //printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
        //        2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
        printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
        printf("  Number of multiprocessors: %d\n",prop.multiProcessorCount);
        printf("  minor-major: %d-%d\n", prop.minor, prop.major);
        printf("  Warp-size: %d\n", prop.warpSize);
        printf("  L2 cache size: %d\n", prop.l2CacheSize);
        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        //printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
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

std::vector<uint32_t> generateSmallPrimesList(uint32_t limit, uint32_t* sieve) {
    std::vector<uint32_t> primes;
    primes.push_back(2);
    for (int p=3; p<limit; p+=2) {
        if (sieve[p/2] == 0) {
            primes.push_back(p);
        }
    }
    return primes;
}

std::vector<uint32_t> generateRhoList(uint32_t limit, uint32_t* sieve, std::vector<uint32_t> primes) {
    std::vector<uint32_t> rhos;
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

int gcd(int a, int b) {
    // https://www.geeksforgeeks.org/gcd-in-cpp/
    // std::gcd might not be available in older c++ versions, so this is mainly for compatibility
    // this is not a good algorithm in the general case but it is fine when 1 of the numbers is small
    int result = min(a, b);
    while (result > 0) {
        if (a % result == 0 && b % result == 0) {
            break;
        }
        result--;
    }
    return result;
}
void initConstantArrays() {
    uint8_t valueToPos[WORD_LENGTH/2] = { // WE ARE ONLY TAKING ODD NUMBERS HERE
#if (WORD_LENGTH == 120)
        0,0,0,1,0,2,3,0,4,5,0,6,0,0,7,
        8,0,0,9,0,10,11,0,12,13,0,14,0,0,15,
        16,0,0,17,0,18,19,0,20,21,0,22,0,0,23,
        24,0,0,25,0,26,27,0,28,29,0,30,0,0,31,
#elif (WORD_LENGTH == 240)
        0,0,0,1,0,2,3,0,4,5,0,6,0,0,7,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        8,0,0,9,0,10,11,0,12,13,0,14,0,0,15,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        16,0,0,17,0,18,19,0,20,21,0,22,0,0,23,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        24,0,0,25,0,26,27,0,28,29,0,30,0,0,31,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
#endif
    };
    HANDLE_ERROR(cudaMemcpyToSymbol(SIEVE_VALUE_TO_POS, valueToPos, sizeof(uint8_t)*(WORD_LENGTH/2), 0, cudaMemcpyHostToDevice));

    uint32_t inverses[WORD_LENGTH/2];
    for (int p=1; p<WORD_LENGTH; p+=2) {
        if (gcd(WORD_LENGTH, p) > 1) {
            inverses[p/2] = 0;
        } else {
            for (int inv=1; inv<WORD_LENGTH; inv+=2) {
                if ((p*inv) % WORD_LENGTH == 1) {
                    inverses[p/2] = inv;
                    break;
                }
            }
        }
    }
    HANDLE_ERROR(cudaMemcpyToSymbol(WORD_INVERSES, inverses, sizeof(uint32_t)*(WORD_LENGTH/2), 0, cudaMemcpyHostToDevice));

    uint8_t posToValue[32] = {
#if WORD_LENGTH == 240
        1,7,11,13,17,19,23,29,
        61,67,71,73,77,79,83,89,
        121,127,131,133,137,139,143,149,
        181,187,191,193,197,199,203,209,
#elif WORD_LENGTH == 120
        1,7,11,13,17,19,23,29,
        31,37,41,43,47,49,53,59,
        61,67,71,73,77,79,83,89,
        91,97,101,103,107,109,113,119,
#endif
    };
    HANDLE_ERROR(cudaMemcpyToSymbol(SIEVE_POS_TO_VALUE, posToValue, sizeof(uint8_t)*32, 0, cudaMemcpyHostToDevice));

    uint8_t array[32][WORD_LENGTH/2][3];
    for (int x=0; x<32; x++) {
        int startMod = posToValue[x];
        for (int y=0; y<WORD_LENGTH/2; y++) {
            int increase = y*2+1;
            uint8_t add = 2;
            int mod = (startMod + increase*add) % WORD_LENGTH;
            while ((valueToPos[mod/2] == 0) && (mod != 1)) {
                add += 2;
                mod = (startMod + increase*add) % WORD_LENGTH;
            }
            array[x][y][0] = add;
            array[x][y][1] = (startMod + increase * add) % WORD_LENGTH;
            array[x][y][2] = (startMod + increase * add) / WORD_LENGTH;
        }
    }
    HANDLE_ERROR(cudaMemcpyToSymbol(NEXT_SIEVE_HIT, array, sizeof(uint8_t)*32*WORD_LENGTH/2*3, 0, cudaMemcpyHostToDevice));
}

void printGap(uint128_t startPrime, uint128_t endPrime) {
#if RUN_TESTS
    return;
#endif
    uint32_t gap = endPrime-startPrime;
    std::string suffix = "";
    int missing[] = {1504,1514,1532,1538,1546,1558,1560,1562,1570,
                     1574,1580,1582,1584,1586,1588,1590,1594,1596,1598,1602,1604,1606,
                     1608,1610,1612,1614,1616,1618,1620,1622,1624,1626,1630,1632,1634,1638,1640,
                     1642,1646,1648,1650,1652,1654,1656,1658,1660,1662,1664,1666,1668,1670,1672,1674};
    if (gap > 1676) {
        suffix = " MAXIMAL";
    } else {
        for (int testGap : missing) {
            if (testGap == gap) {
                suffix = " FIRST OCCURRENCE";
                break;
            }
        }
    }
    if (startPrime >> 64) {
        printf(": %lu%019lu %u %.6f%s\n", hi19(startPrime), lo19(startPrime), gap, gap/log(startPrime), suffix.c_str());  
    } else {
        printf(": %lu %u %.6f%s\n", (uint64_t) startPrime, gap, gap/log(startPrime), suffix.c_str());
    }
}

void cpuFindGapAround(uint128_t n, uint32_t minGap) {
    uint128_t p1 = n;
    p1 -= 1 - (p1 % 2);
    while (!isPrime84(p1)) {p1 -= 2;}

    uint128_t p2 = n;
    p2 += 1 - (p2 % 2);
    while (!isPrime84(p2)) {p2 += 2;}

    int gap = (int) (p2-p1);
    if (gap >= minGap) {
        printGap(p1, p2);
    }
}

void checkGapAndPrint(uint128_t startPrime, uint128_t endPrime, uint32_t minGapSize) {
    uint128_t lastPrime = startPrime;
    uint128_t test = startPrime + minGapSize - 2;
    while (lastPrime <= endPrime - minGapSize) {
        if (isPrime84(test)) {
            //assert(test % 60 > 30); // if this fails then we sieved out too much (doesn't affect correctness of results)
            lastPrime = test;
            test += minGapSize-2;
        } else {
            assert(test != startPrime && test != endPrime); // if this fails then the pseudoprime sieving is incorrect
            test -= 2;
            if (test == lastPrime) {
                test += minGapSize;
                while (!isPrime84(test)) {
                    test += 2;
                }
                printGap(lastPrime, test);
                lastPrime = test;
            }
        }
    }
}

void displayResultsAndClear(PrimeGap* resultList, uint32_t minGapSize) {
    //printf("%d results in block\n", resultList[0].gap);
    if (resultList[0].gap > RESULT_LIST_SIZE-1) {
        resultList[0].gap = RESULT_LIST_SIZE-1;
    }
    std::sort(resultList+1, resultList+resultList[0].gap+1, compareByPrime);
    for (int i=1; i<=resultList[0].gap; i++) {
        uint128_t endPrime = resultList[i].startPrime + resultList[i].gap;
        if (WORD_SIEVING_LENGTH < WORD_LENGTH) {
            //printGap(resultList[i].startPrime, endPrime);
            checkGapAndPrint(resultList[i].startPrime, endPrime, minGapSize);
        } else {
            printGap(resultList[i].startPrime, endPrime);
        }
        resultList[i].startPrime = 0;
        resultList[i].gap = 0;
    }
    resultList[0].gap = 0;
}

std::chrono::_V2::steady_clock::time_point printProgress(
    std::chrono::_V2::steady_clock::time_point start,
    std::chrono::_V2::steady_clock::time_point lastFinish,
    uint128_t sieveStart, int i
) {
    std::chrono::steady_clock::time_point finish = std::chrono::steady_clock::now();
    double totalSecs = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count()/1e9;
    double lastSecs = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-lastFinish).count()/1e9;
    double speed = BLOCK_SIZE * PROGRESS_UPDATE_BLOCKS / lastSecs;
    int speedLog10 = log10(speed);
#ifndef RUN_FROM_PYTHON
    if (i) {
        printf("Done %d blocks (limit=%lu%019lu, time=%f seconds, speed=%.3fe%d/sec)\n",
            i, hi19(sieveStart), lo19(sieveStart), totalSecs, speed/pow(10,speedLog10), speedLog10);
    } else {
        printf("Done %d blocks (limit=%lu%019lu, time=%f seconds)\n", i, hi19(sieveStart), lo19(sieveStart), totalSecs);
    }
#else
    if (i) {
        printf("Progress %d %lu%019lu %.3f\n", i, hi19(sieveStart), lo19(sieveStart), speed/1e9);
    } else {
        printf("Progress %d %lu%019lu\n", i, hi19(sieveStart), lo19(sieveStart));
    }
#endif
    return finish;
}

/*void tests() {
    uint128_t firstPrime = 9523372036854775808UL;
    firstPrime *= 2;
    double orig_derivative = 1.0 / firstPrime / firstPrime * 4294967296.0 * 4294967296.0 * 4294967296.0 * 4294967296.0 - 0.00000000001;
    uint64_t orig_magic = my_getMagic1(firstPrime);
    for (uint128_t p = firstPrime+2; p<firstPrime+4294967296; p+=20000) {
        if (fermatTest65Full(p, (uint32_t)(p-firstPrime), orig_magic, orig_derivative) !=
            fermatTest65(p, (uint32_t)(p-firstPrime), orig_magic, orig_derivative)) {
            printf("Bad %lu\n", (uint64_t) (p%10000000000));
        }
    }
    exit(1);
}*/



int main(int argc, char* argv[]) {
    setbuf(stdout, NULL);

    /*for (long lo=3106524393915332021; lo<3106524393915333021; lo+=2) {
        bool result = ciosFermatTest128(lo, 2);
        if (result) {
            printf("Prime at lo=%lu\n", lo);
        }
    }
    return 0;*/

    if (argc < 4 || argc > 5) {
        printf("Usage: ./prime_gaps.out <minGap> <start> [numBlocks] [deviceNum]\n");
        exit(1);
    }
    int DEVICE_NUM = 0;
    if (argc > 4) DEVICE_NUM = atoi(argv[4]);
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (DEVICE_NUM >= deviceCount) {
        printf("ERROR: deviceNum (%d) must be less than the number of devices (%d)\n", DEVICE_NUM, deviceCount);
        exit(1);
    }

    cudaSetDevice(DEVICE_NUM);

    //deviceInfo();

    printf("Starting\n");
    initConstantArrays();

    int SMALL_PRIME_LIMIT = 10000000; // don't change this

    printf("Generating primes below %u\n", SMALL_PRIME_LIMIT);
    uint32_t* smallSieve = sieveInitialSmallPrimes(SMALL_PRIME_LIMIT);
    std::vector<uint32_t> primeList = generateSmallPrimesList(SMALL_PRIME_LIMIT, smallSieve);
    std::vector<uint32_t> rhoList = generateRhoList(SMALL_PRIME_LIMIT, smallSieve, primeList);

    std::vector<uint32_t> primeMods(primeList.size());
    std::transform(primeList.begin(), primeList.end(), primeMods.begin(), [](uint32_t x) { return x % 240; });

    printf("Done generating primes below %u\n", SMALL_PRIME_LIMIT);
    if (primeList.size() != 664579) {
        printf("WRONG SIZE! Got %lu, expected %d\n", primeList.size(), 664579);
    }
    delete smallSieve;

    uint32_t* primeListCuda;
    HANDLE_ERROR(cudaMallocManaged(&primeListCuda, primeList.size() * sizeof(uint32_t)));
    uint32_t* rhoListCuda;
    HANDLE_ERROR(cudaMallocManaged(&rhoListCuda, rhoList.size() * sizeof(uint32_t)));
    uint32_t* primeModsCuda;
    HANDLE_ERROR(cudaMallocManaged(&primeModsCuda, primeMods.size() * sizeof(uint32_t)));

    for (int i=0; i<primeList.size(); i++) {
        primeListCuda[i] = primeList[i];
        rhoListCuda[i] = rhoList[i];
        primeModsCuda[i] = primeMods[i];
    }

    // pseudoprime is 21693774589725076147 (67 mod 120), with start 21693774589725076080
    // first kilogap above that is 21693776423220625951-6953, or 1833495550873=1.8e12 larger
#if RUN_TESTS
    uint128_t sieveStart = ((uint128_t) 10000000000000000000UL) + 11693774504960000000UL; // 21693774504960000000
#else
    uint128_t sieveStart = atouint128_t(argv[2]);
#endif
    
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
    if (sieveLengthWords % SHARED_SIZE_WORDS != 0) {
        printf("ERROR: Block length (%lu) must be a multiple of the word length (%d) times the shared size (%d)\n",
               ((uint64_t) sieveLengthWords)*WORD_LENGTH, WORD_LENGTH, SHARED_SIZE_WORDS);
        exit(1);
    }

    /*int offset=78498; // doing primes from 1M to 1.1M
    printf("using asdf %u\n", (primeList.size()-offset));
    
    std::chrono::steady_clock::time_point startp = chrono::steady_clock::now();
    sievePseudoprimesSeparate<<<384,64>>>(sieveStart, 100000000000000UL,
                                          primeListCuda+offset, rhoListCuda+offset, 7216); //primeList.size()-offset);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point finishp = chrono::steady_clock::now();
    cout << "Done in " << chrono::duration_cast<chrono::nanoseconds>(finishp-startp).count()/1e9 << " seconds\n";
    return 0;*/


    uint32_t* smallPrimeWheel1;
    uint32_t* smallPrimeWheel2;
    uint32_t* smallPrimeWheel3;
    uint32_t* smallPrimeWheel4;
    HANDLE_ERROR(cudaMalloc((void **) &smallPrimeWheel1, (7*11*13*17*19*23*29) * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void **) &smallPrimeWheel2, (31*37*41*43*47) * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void **) &smallPrimeWheel3, (53*59*61*67) * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void **) &smallPrimeWheel4, (71*73*79*83) * sizeof(uint32_t)));
    printf("Making small prime sieve\n");
    
    std::chrono::steady_clock::time_point start1 = std::chrono::steady_clock::now();
    makeSmallPrimeWheels<<<96,512>>>(smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4);
    HANDLE_ERROR(cudaDeviceSynchronize());
    std::chrono::steady_clock::time_point finish1 = std::chrono::steady_clock::now();
    std::cout << "Done in " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish1-start1).count()/1e9 << " seconds\n";
   
    // here is where we could run tests for small prime wheels

    PrimeGap* resultList;
    PrimeGap resultListHost[RESULT_LIST_SIZE];
    HANDLE_ERROR(cudaMalloc((void **) &resultList, sizeof(PrimeGap) * RESULT_LIST_SIZE));
    HANDLE_ERROR(cudaMemset(resultList, 0, RESULT_LIST_SIZE * sizeof(PrimeGap)));
    
    uint32_t* globalSieve1;
    uint32_t* globalSieve2;
    HANDLE_ERROR(cudaMalloc((void **) &globalSieve1, sieveLengthWords * sizeof(uint32_t)));
    HANDLE_ERROR(cudaMalloc((void **) &globalSieve2, sieveLengthWords * sizeof(uint32_t)));

    uint128_t* firstPrimeInBlock;
    uint128_t* lastPrimeInBlock;
    HANDLE_ERROR(cudaMallocManaged((void **) &firstPrimeInBlock, sizeof(uint128_t)));
    HANDLE_ERROR(cudaMallocManaged((void **) &lastPrimeInBlock, sizeof(uint128_t)));

    HANDLE_ERROR(cudaMemset(globalSieve1, 0, sieveLengthWords * sizeof(uint32_t)));
    kernel<<<GPU_BLOCKS,GPU_THREADS>>>(
        globalSieve1, sieveStart, (uint32_t) sieveLengthWords,
        primeListCuda, rhoListCuda, primeModsCuda, primeList.size(),
        smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4
    );
    HANDLE_ERROR(cudaDeviceSynchronize());

    uint128_t lastPrimeInLastBlock = 0;

#if RUN_TESTS
    int blocksToTest = 4;
    uint32_t minGapSize = 720;
#else
    int blocksToTest = atoi(argv[3]);
    uint32_t minGapSize = atoi(argv[1]);
#endif

#ifdef HIGH_64
    assert(sieveStart >> 64 == HIGH_64);
    assert((sieveStart + BLOCK_SIZE*blocksToTest) >> 64 == HIGH_64);
#endif

#if (WORD_LENGTH == 120) || (WORD_LENGTH == 240)
    if (minGapSize % (WORD_LENGTH / 4) != 0) {
        printf("ERROR: minGapSize (%d) must be a multiple of WORD_LENGTH/4 (%d)\n", minGapSize, WORD_LENGTH/4);
        exit(1);
    }
#else
    // currently unused
    if (minGapSize % WORD_LENGTH != 0) {
        printf("ERROR: minGapSize (%d) must be a multiple of WORD_LENGTH (%d)\n", minGapSize, WORD_LENGTH);
        exit(1);
    }
#endif
    printf("Searching for gaps of size >= %d...\n", minGapSize);
    
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point finish = start;
    for (int i=0; i<blocksToTest-1; i++) {
        /* In this loop, we are: (0-indexed)
        Sieving block i+1,
        Gap-finding block i, and
        (if i >= 1) Double-checking and printing the gaps of block i-1
        If WORD_LENGTH == WORD_SIEVING_LENGTH, then we don't double check and just print all of them
        */
        HANDLE_ERROR(cudaMemset(globalSieve2, 0, sieveLengthWords * sizeof(uint32_t)));
        
        if (i>0) {
            HANDLE_ERROR(cudaMemcpy(resultListHost, resultList, sizeof(PrimeGap)*RESULT_LIST_SIZE, cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemset(resultList, 0, RESULT_LIST_SIZE * sizeof(PrimeGap)));
        }
#if RUN_TESTS
        if (i==1) assert(resultListHost[0].gap == 27);
        if (i==2) assert(resultListHost[0].gap == 25);
#endif

        kernelBoth<<<GPU_BLOCKS,GPU_THREADS>>>(
            globalSieve2, globalSieve1, sieveStart+sieveLength, (uint32_t) sieveLengthWords,
            minGapSize, primeListCuda, rhoListCuda, primeModsCuda, primeList.size(),
            smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4,
            (int) (PROPORTION_OF_BLOCKS_FOR_SIEVING * GPU_BLOCKS),
            resultList, firstPrimeInBlock, lastPrimeInBlock
        );
        
#if RUN_TESTS
        if (i == 0) {
            /*
            With wordsize=240, the following changes need to be made to the testing:
            Start 46.08e9 earlier
            The global sieve index changes from 322375634 to 322375634/2 + 46080000000/240
            The target bitmask after PSP sieving changes from 1834999230 to 2801754046
            Before PSP, it changes from 1834868158 to 2801753534
            Number of resulting gaps chaanges to 52,37,49 for the 3 blocks
            */
            uint32_t val;
            HANDLE_ERROR(cudaMemcpy(&val, globalSieve2+322375634, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            if (val != 1834999230) {printf("FAILED PSP TEST IN KERNEL 2 %u\n", val);}
            else {printf("Passed kernel 2 tests.\n");}
        }
#endif
        if (i>0) {
            displayResultsAndClear(resultListHost, minGapSize);
        }
        HANDLE_ERROR(cudaDeviceSynchronize());
        
        if (i%PROGRESS_UPDATE_BLOCKS == 0) {
            finish = printProgress(start, finish, sieveStart, i);
        }
        //printf("    aksjdf this %d %d last %d\n", (*firstPrimeInBlock % 1000000000), (*lastPrimeInBlock % 1000000000),
        //    (lastPrimeInLastBlock % 1000000000));

        if (lastPrimeInLastBlock && (*firstPrimeInBlock - lastPrimeInLastBlock >= minGapSize)) {
            checkGapAndPrint(lastPrimeInLastBlock, *firstPrimeInBlock, minGapSize);
        }
        //printf("    after \n");
        lastPrimeInLastBlock = *lastPrimeInBlock;

        sieveStart += sieveLength;
        std::swap(globalSieve1, globalSieve2);
    }

    /*if ((blocksToTest-2)%PROGRESS_UPDATE_BLOCKS == 0) {
        finish = printProgress(start, finish, sieveStart, blocksToTest-2);
    }*/
    HANDLE_ERROR(cudaMemcpy(resultListHost, resultList, sizeof(PrimeGap)*RESULT_LIST_SIZE, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemset(resultList, 0, RESULT_LIST_SIZE * sizeof(PrimeGap)));
#if RUN_TESTS
    assert(resultListHost[0].gap == 12); // only 12 gaps of size >=720 in this block, a lot less than expected
#endif

    kernel2<<<GPU_BLOCKS,GPU_THREADS>>>(globalSieve1, sieveStart, sieveLengthWords, minGapSize,
                                resultList, firstPrimeInBlock, lastPrimeInBlock);
    displayResultsAndClear(resultListHost, minGapSize);
    HANDLE_ERROR(cudaDeviceSynchronize());

    if (lastPrimeInLastBlock && (*firstPrimeInBlock - lastPrimeInLastBlock >= minGapSize)) {
        checkGapAndPrint(lastPrimeInLastBlock, *firstPrimeInBlock, minGapSize);
    }

    if ((blocksToTest-1)%PROGRESS_UPDATE_BLOCKS == 0) {
        finish = printProgress(start, finish, sieveStart, blocksToTest-1);
    }

    HANDLE_ERROR(cudaMemcpy(resultListHost, resultList, sizeof(PrimeGap)*RESULT_LIST_SIZE, cudaMemcpyDeviceToHost));
#if RUN_TESTS
    assert(resultListHost[0].gap == 22);
#endif
    displayResultsAndClear(resultListHost, minGapSize);
    sieveStart += sieveLength;
    cpuFindGapAround(sieveStart, minGapSize);

    finish = printProgress(start, finish, sieveStart, blocksToTest);

    return 0;
}

/*
THINGS TO ADD:

handle constants in a better way than macros
find a way to do "proofs" that you searched a range
    some sort of combination of all the primes you searched?
    i.e. for one block, the proof would look like <block start> <all the params> N
        where N is something that can only be found if you have actually done the work
        that way, if you only do 90% of the work, a randomly chosen block will find that out 10% of the time

*/