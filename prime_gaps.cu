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
#define FERMAT_TEST fermatTest65Full

#define count_set_bits_64 __builtin_popcountll

#define uint128_t unsigned __int128

#define WIPE_LINE "\r\033[K"
#define END_OF_RANGE ~0

#define RESULT_LIST_SIZE 131072

#ifndef RUN_TESTS
#define RUN_TESTS 0
#endif

#if RUN_TESTS
    #define BLOCK_SIZE 46080000000
    #define WORD_LENGTH 120
    #define WORD_SIEVING_LENGTH 120
    #define MIN_GAP_SIZE 720
#else

#ifndef MIN_GAP_SIZE
#define MIN_GAP_SIZE 900 // low enough that it will remind people to set it properly
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 46080000000 // we don't actually need to add the UL as far as I know
#endif

#ifndef WORD_LENGTH
#define WORD_LENGTH 120
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

// TODO: THESE ARRAYS HAVE TO CHANGE BASED ON WORD_LENGTH!!

#if (WORD_LENGTH == 120)
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
#elif (WORD_LENGTH == 240)
__constant__ uint8_t SIEVE_POS_TO_VALUE[32] = {
    1,7,11,13,17,19,23,29,
    61,67,71,73,77,79,83,89,
    121,127,131,133,137,139,143,149,
    181,187,191,193,197,199,203,209,
};
__constant__ uint8_t SIEVE_VALUE_TO_POS[120] = { // WE ARE ONLY TAKING ODD NUMBERS HERE
    0,0,0,1,0,2,3,0,4,5,0,6,0,0,7,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    8,0,0,9,0,10,11,0,12,13,0,14,0,0,15,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    16,0,0,17,0,18,19,0,20,21,0,22,0,0,23,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    24,0,0,25,0,26,27,0,28,29,0,30,0,0,31,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
};
#endif

__constant__ bool IS_COPRIME_30[15] = {
    1,0,0,1,0,1,1,0,1,1,0,1,0,0,1,
};

__constant__ uint32_t WORD_INVERSES[WORD_LENGTH/2];
__constant__ uint32_t INVERSES_30[15] = {
    1,0,0,13,0,11,7,0,23,19,0,17,0,0,29,
};


#if RUN_TESTS
#define SHARED_SIZE_WORDS 12288 // DON'T CHANGE THESE VALUES!!!!!
#define NUM_MEDIUM_PRIMES (1536 - SIEVING_DUPLICATED_PRIMES)
#define NUM_SMALL_PRIME_WHEELS 2
#else
#define SHARED_SIZE_WORDS 8192
#define NUM_MEDIUM_PRIMES (3584 - SIEVING_DUPLICATED_PRIMES)
#define NUM_SMALL_PRIME_WHEELS 3 // this shouldn't be higher than 4
#endif

#if (NUM_SMALL_PRIME_WHEELS == 1)
#define NUM_SMALL_PRIMES 10
#elif (NUM_SMALL_PRIME_WHEELS == 2)
#define NUM_SMALL_PRIMES 15
#elif (NUM_SMALL_PRIME_WHEELS == 3)
#define NUM_SMALL_PRIMES 19
#elif (NUM_SMALL_PRIME_WHEELS == 4)
#define NUM_SMALL_PRIMES 23
#endif

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
        // wheels at all. 
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




__host__ __device__ uint64_t lo19(uint128_t n) {
    return (uint64_t) (n % 10000000000000000000UL);
}
__host__ __device__ uint64_t hi19(uint128_t n) {
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

__device__ inline uint128_t fastMod(uint128_t n, uint128_t mod, uint64_t magic) {
    // !!! THIS ONLY WORKS IF mod > 2^64
    // (this is because "magic" needs to fit in a 64-bit int for efficiency)
    uint128_t result = n - mod * mul_128_64_hi64_inexact(n, magic);

    // potential +1 from losing the carry in the mul128*64, and another +1 from the inexactness of the magic num

    return result - ((result >= mod) + (result >= mod*2)) * mod;
}

__device__ inline uint128_t superfastMod(uint128_t n, uint128_t mod, uint64_t magic) {
    // same as fastMod but it might give a result up to 2*mod higher (still correct mod "mod")
    return n - mod * mul_128_64_hi64_inexact(n, magic);
}

__device__ inline uint128_t squareMod84(uint128_t a, uint128_t mod, uint128_t magic) {
    uint128_t ahi = a>>42;
	uint128_t alo = a & 0x3ffffffffffL;
	//return ((((a*ahi) % m) << 42) + a*alo) % m
    return fastMod((fastMod(a*ahi, mod, magic) << 42) + a*alo, mod, magic);
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

__host__ __device__ inline uint64_t my_getMagic1(uint128_t mod)
{
	// precomputes (2^128 - 1)/ mod    (a 64 bits number)
	// !! THIS ONLY WORKS IF mod > 2^64, otherwise we would get overflow!
	return (uint64_t) ((((uint128_t) 0) - 1) / mod);
}

__host__ __device__ inline uint128_t my_getMagic2(uint128_t mod, uint64_t magic1)
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

__host__ __device__ uint128_t my_fastModSqr(uint128_t n, uint64_t mod_lo, uint64_t magic1, uint128_t magic2)
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
 __device__ inline bool fermatTest65Inner(uint128_t n, uint64_t magic1)
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
	//uint64_t magic1 = my_getMagic1(n);
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

// POTENTIAL OPTIMIZATION: REUSE MAGIC CALCULATIONS FOR MULTIPLE FERMAT TESTS

__device__ bool fermatTest65(uint128_t n, uint32_t delta, uint64_t orig_magic, double orig_derivative) {
    // estimate of magic1(n) is:  magicp - 2^128*d//p^2
    // orig_magic should be precomputed as get_magic1(start)
    // orig_derivative should be 2^128 / start^2
    uint64_t magic1 = orig_magic - (uint64_t) (orig_derivative*delta); // this might be 1 more or 1 less than correct
    uint128_t test = ((uint128_t) (magic1+1)) * n;
    if (test >= n+n) magic1--;
    if (test < n) magic1++;
    // where d = n-p, aand 2^128/p^2 is precalculated as a double
    return fermatTest65Inner(n, magic1);
}

__device__ bool fermatTest65Full(uint128_t n, uint32_t delta, uint64_t orig_magic, double orig_derivative) {
    // all arguments other than n are unused - we just keep it the same format as fermatTest65
    uint64_t magic1 = my_getMagic1(n);
    return fermatTest65Inner(n, magic1);
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
        //uint128_t wordStart = start/WORD_LENGTH + i;
        // We cannot replace the atomicOr with a non-atomic operation, because that might skip sieving out some values
        // and we can't miss any because of pseudoprimes
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
#if SIEVE_BY_30
    uint32_t pInv = INVERSES_30[(p%30)/2];
    uint32_t precalculated = p - (start/30) % p;
    for (uint32_t bit=startBit; bit<startBit+numBits; bit++) {
        // TODO: I should be able to precalculate this entire thing
        // only need to precalculate (8 bits) * (# of sieving primes) words
        uint32_t startByte = (p * ((SIEVE_POS_TO_VALUE[bit] * pInv)%30)) / 30 + precalculated;
#if (WORD_LENGTH == 240)
        if (startByte % 2) startByte += p;
        startByte /= 2;
#endif
        if (startByte >= p) startByte -= p;
        uint32_t mask = 1 << bit;
        // TODO: THIS IS NOT SET UP FOR WORD_SIEVING_LENGTH > WORD_LENGTH
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
    if (threadIdx.x < 64) { // the first 64 threads will split primes 53,59,61,67,71,73,79,83 into 8 threads/prime
        bitIdx = (threadIdx.x % 8);
        numBits = 1;
        pIdx = threadIdx.x/8;
    } else if (threadIdx.x < 128) { // the next 64 threads will split the 16 primes 67-139 into 4 threads/prime
        bitIdx = (threadIdx.x % 4)*2;
        numBits = 2;
        pIdx = (threadIdx.x-64)/4 + 8;
    } else if (threadIdx.x < 192) { // 32 primes 149-317 will get 2 threads/prime
        bitIdx = (threadIdx.x % 2)*4;
        numBits = 4;
        pIdx = (threadIdx.x-128)/2 + 8+16;
    } else { // the remaining 320 threads each handle 1 prime by themselves
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

        uint32_t pTimesRhoModWordLen = pTimesRho % WORD_LENGTH;
        uint64_t pTimesRhoDivWordLen = pTimesRho / WORD_LENGTH;
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
            currentPosInWord += pTimesRhoModWordLen;
            currentWord += pTimesRhoDivWordLen + (currentPosInWord >= WORD_LENGTH);
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
    // We have to have each thread block run through consecutive shared memory blocks
    // so that we can use the result from the previous shared block to cut down on computation for the next
    // (THE ABOVE IS NOT IMPLEMENTED YET)
    uint32_t numSharedBlocks = sieveLengthWords / SHARED_SIZE_WORDS;
    uint32_t firstSharedBlockIdx = (uint32_t) (((double) numSharedBlocks) * blockIdx.x / numBlocks);
    uint32_t lastSharedBlockIdx = (uint32_t) (((double) numSharedBlocks) * (blockIdx.x+1) / numBlocks);
    
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
#if RUN_TESTS
        if (threadIdx.x == 97 && (sieveStart + sharedBlockIdx*SHARED_SIZE_WORDS*WORD_LENGTH)%1000000000 == 723607040) {
            if (sharedSieve[12242] != 1834868158) {
                good = false;
                printf("KERNEL 1 TEST 2 FAILED! (expected=%u actual=%u)\n", 1834868670, sharedSieve[12242]);
            }
            if (good) {printf("Passed kernel 1 tests.\n");}
        }
#endif

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

__device__ void findGaps(uint32_t* sieve, uint128_t start, uint64_t sieveLengthWords, uint32_t startBlock, PrimeGap* resultList,
                         uint128_t* totalFirstPrime, uint128_t* totalLastPrime) {
    // sieve should be in GLOBAL MEMORY for this function to work properly
    uint32_t gridDimNew = gridDim.x - startBlock;
    uint32_t blockIdxNew = blockIdx.x - startBlock;

    const int MIN_GAP_SIZE_BITS = (MIN_GAP_SIZE / WORD_LENGTH) * 32;

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
    uint64_t orig_magic = my_getMagic1(firstPrime);


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
            resultList[resultIdx].startPrime = lastPrime;
            resultList[resultIdx].gap = gap;

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


__global__ void kernel2(uint32_t* globalSieve, uint128_t sieveStart, uint64_t sieveLengthWords, PrimeGap* resultList,
                        uint128_t* firstPrimeInBlock, uint128_t* lastPrimeInBlock) {
    findGaps(globalSieve, sieveStart, sieveLengthWords, 0, resultList, firstPrimeInBlock, lastPrimeInBlock);
}


__global__ void kernelBoth(uint32_t* globalSieve1, uint32_t* globalSieve2, uint128_t sieveStart, uint32_t sieveLengthWords,
                           uint32_t* primeList, uint32_t* rhoList, uint32_t primeCount,
                           uint32_t* smallPrimeWheel1, uint32_t* smallPrimeWheel2,
                           uint32_t* smallPrimeWheel3, uint32_t* smallPrimeWheel4,
                           uint32_t numSieveBlocks, PrimeGap* resultList,
                           uint128_t* firstPrimeInBlock, uint128_t* lastPrimeInBlock) {
    if (blockIdx.x < numSieveBlocks) {
        sieveAll(globalSieve1, sieveStart, sieveLengthWords, primeList, rhoList, primeCount,
            smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4, numSieveBlocks);
    } else {
        findGaps(globalSieve2, sieveStart - ((uint128_t) sieveLengthWords)*WORD_LENGTH, sieveLengthWords, numSieveBlocks,
                 resultList, firstPrimeInBlock, lastPrimeInBlock);
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
    int result = min(a, b);
    while (result > 0) {
        if (a % result == 0 && b % result == 0) {
            break;
        }
        result--;
    }
    return result;
} 

void initInverseArray() {
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
    auto err = cudaMemcpyToSymbol(WORD_INVERSES, inverses, sizeof(uint32_t)*(WORD_LENGTH/2), 0, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for inverses: %s\n", cudaGetErrorString(err));
        exit(1);
    }
}

void printGap(uint128_t startPrime, uint128_t endPrime) {
#if RUN_TESTS
    return;
#endif
    uint32_t gap = endPrime-startPrime;
    std::string suffix = "";
    int missing[] = {1432,1444,1458,1472,1474,1478,1484,1492,1496,1498,1500,1504,1508,1512,1514,1516,1518,1520,1522,
                     1524,1528,1532,1534,1536,1538,1540,1542,1544,1546,1548,1554,1556,1558,1560,1562,1564,1566,1568,1570};
    if (gap > 1572) {
        suffix = " MAXIMAL";
    } else {
        for (int testGap : missing) {
            if (testGap == gap) {
                suffix = " FIRST OCCURRENCE";
                break;
            }
        }
    }

    printf("%lu%019lu %u %.6f%s\n", hi19(startPrime), lo19(startPrime), gap, gap/log(startPrime), suffix.c_str());
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

void checkGapAndPrint(uint128_t startPrime, uint128_t endPrime) {
    uint128_t lastPrime = startPrime;
    uint128_t test = startPrime + MIN_GAP_SIZE - 2;
    while (lastPrime <= endPrime - MIN_GAP_SIZE) {
        if (isPrime84(test)) {
            lastPrime = test;
            test += MIN_GAP_SIZE-2;
        } else {
            test -= 2;
            if (test == lastPrime) {
                test += MIN_GAP_SIZE;
                while (!isPrime84(test)) {
                    test += 2;
                }
                printGap(lastPrime, test);
                lastPrime = test;
            }
        }
    }
}

void displayResultsAndClear(PrimeGap* resultList) {
    std::sort(resultList+1, resultList+resultList[0].gap+1, compareByPrime);
    //printf("%d results in block\n", resultList[0].gap);
    for (int i=1; i<=resultList[0].gap; i++) {
        uint128_t endPrime = resultList[i].startPrime + resultList[i].gap;
        if (WORD_SIEVING_LENGTH < WORD_LENGTH) {
            //printGap(resultList[i].startPrime, endPrime);
            checkGapAndPrint(resultList[i].startPrime, endPrime);
        } else {
            printGap(resultList[i].startPrime, endPrime);
        }
        resultList[i].startPrime = 0;
        resultList[i].gap = 0;
    }
    resultList[0].gap = 0;
}

std::chrono::_V2::system_clock::time_point printProgress(
    std::chrono::_V2::system_clock::time_point start,
    std::chrono::_V2::system_clock::time_point lastFinish,
    uint128_t sieveStart, int i
) {
    auto finish = std::chrono::high_resolution_clock::now();
    double totalSecs = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count()/1e9;
    double lastSecs = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-lastFinish).count()/1e9;
    double speed = BLOCK_SIZE * PROGRESS_UPDATE_BLOCKS / lastSecs;
    int speedLog10 = log10(speed);
    if (i) {
        printf("Done %d blocks (limit=%lu%019lu, time=%f seconds, speed=%.3fe%d/sec)\n",
            i, hi19(sieveStart), lo19(sieveStart), totalSecs, speed/pow(10,speedLog10), speedLog10);
    } else {
        printf("Done %d blocks (limit=%lu%019lu, time=%f seconds)\n", i, hi19(sieveStart), lo19(sieveStart), totalSecs);
    }
    return finish;
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
    initInverseArray();

    int SMALL_PRIME_LIMIT = 5000000; // don't change this

    printf("Generating primes below %u\n", SMALL_PRIME_LIMIT);
    uint32_t* smallSieve = sieveInitialSmallPrimes(SMALL_PRIME_LIMIT);
    std::vector<uint32_t> primeList = generateSmallPrimesList(SMALL_PRIME_LIMIT, smallSieve);
    std::vector<uint32_t> rhoList = generateRhoList(SMALL_PRIME_LIMIT, smallSieve, primeList);
    printf("Done generating primes below %u\n", SMALL_PRIME_LIMIT);
    delete smallSieve;

    uint32_t* primeListCuda;
    auto err = cudaMallocManaged(&primeListCuda, primeList.size() * sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate managed memory: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    uint32_t* rhoListCuda;
    err = cudaMallocManaged(&rhoListCuda, rhoList.size() * sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate managed memory: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    for (int i=0; i<primeList.size(); i++) {
        primeListCuda[i] = primeList[i];
        rhoListCuda[i] = rhoList[i];
    }


    // pseudoprime is 21693774589725076147 (67 mod 120), with start 21693774589725076080
    // first kilogap above that is 21693776423220625951-6953, or 1833495550873=1.8e12 larger
#if RUN_TESTS
    uint128_t sieveStart = ((uint128_t) 10000000000000000000UL) + 11693774504960000000UL; // 21693774504960000000
#else
    uint128_t sieveStart = atouint128_t(argv[1]);
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
    
    auto startp = chrono::high_resolution_clock::now();
    sievePseudoprimesSeparate<<<384,64>>>(sieveStart, 100000000000000UL,
                                          primeListCuda+offset, rhoListCuda+offset, 7216); //primeList.size()-offset);
    cudaDeviceSynchronize();
    auto finishp = chrono::high_resolution_clock::now();
    cout << "Done in " << chrono::duration_cast<chrono::nanoseconds>(finishp-startp).count()/1e9 << " seconds\n";
    return 0;*/


    uint32_t* smallPrimeWheel1;
    uint32_t* smallPrimeWheel2;
    uint32_t* smallPrimeWheel3;
    uint32_t* smallPrimeWheel4;
    cudaMalloc((void **) &smallPrimeWheel1, (7*11*13*17*19*23*29) * sizeof(uint32_t));
    cudaMalloc((void **) &smallPrimeWheel2, (31*37*41*43*47) * sizeof(uint32_t));
    cudaMalloc((void **) &smallPrimeWheel3, (53*59*61*67) * sizeof(uint32_t));
    cudaMalloc((void **) &smallPrimeWheel4, (71*73*79*83) * sizeof(uint32_t));
    printf("Making small prime sieve\n");
    
    auto start1 = std::chrono::high_resolution_clock::now();
    makeSmallPrimeWheels<<<96,512>>>(smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4);
    auto finish1 = std::chrono::high_resolution_clock::now();
    std::cout << "Done in " << std::chrono::duration_cast<std::chrono::nanoseconds>(finish1-start1).count()/1e9 << " seconds\n";
   
    PrimeGap* resultList;
    PrimeGap resultListHost[RESULT_LIST_SIZE];
    err = cudaMalloc((void **) &resultList, sizeof(PrimeGap) * RESULT_LIST_SIZE);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for the result list: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaMemset(resultList, 0, RESULT_LIST_SIZE * sizeof(PrimeGap));
    
    uint32_t* globalSieve1;
    uint32_t* globalSieve2;
    err = cudaMalloc((void **) &globalSieve1, sieveLengthWords * sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for the sieve: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void **) &globalSieve2, sieveLengthWords * sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for the sieve: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    uint128_t* firstPrimeInBlock;
    uint128_t* lastPrimeInBlock;
    err = cudaMallocManaged((void **) &firstPrimeInBlock, sizeof(uint128_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for the first/last primes: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMallocManaged((void **) &lastPrimeInBlock, sizeof(uint128_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate memory for the first/last primes: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaMemset(globalSieve1, 0, sieveLengthWords * sizeof(uint32_t));
    kernel<<<192,512>>>(
        globalSieve1, sieveStart, (uint32_t) sieveLengthWords,
        primeListCuda, rhoListCuda, primeList.size(),
        smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4
    );
    cudaDeviceSynchronize();

    uint128_t lastPrimeInLastBlock = 0; // TODO: CALCULATE ON CPU THE LAST PRIME BEFORE THE STARTING POINT OF BLOCK 0

    if (MIN_GAP_SIZE % WORD_LENGTH) {
        printf("Searching for gaps of size >= %d... (modified from %d)\n", MIN_GAP_SIZE - (MIN_GAP_SIZE % WORD_LENGTH), MIN_GAP_SIZE);
    } else {
        printf("Searching for gaps of size >= %d...\n", MIN_GAP_SIZE);
    }
#if RUN_TESTS
    int blocksToTest = 4;
#else
    int blocksToTest = 1000000000;
#endif
    auto start = std::chrono::high_resolution_clock::now();
    auto finish = start;
    cpuFindGapAround(sieveStart, MIN_GAP_SIZE);
    for (int i=0; i<blocksToTest-1; i++) {
        /* In this loop, we are: (0-indexed)
        Sieving block i+1,
        Gap-finding block i, and
        (if i >= 1) Double-checking and printing the gaps of block i-1
        If WORD_LENGTH == WORD_SIEVING_LENGTH, then we don't double check and just print all of them
        */
        if (i>0 && (i-1)%PROGRESS_UPDATE_BLOCKS == 0) {
            finish = printProgress(start, finish, sieveStart, i-1);
        }
        cudaMemset(globalSieve2, 0, sieveLengthWords * sizeof(uint32_t));
        
        if (i>0) {
            cudaMemcpy(resultListHost, resultList, sizeof(PrimeGap)*RESULT_LIST_SIZE, cudaMemcpyDeviceToHost);
            cudaMemset(resultList, 0, RESULT_LIST_SIZE * sizeof(PrimeGap));
        }
#if RUN_TESTS
        if (i==1) assert(resultListHost[0].gap == 27);
        if (i==2) assert(resultListHost[0].gap == 25);
#endif

        kernelBoth<<<192,512>>>(
            globalSieve2, globalSieve1, sieveStart+sieveLength, (uint32_t) sieveLengthWords,
            primeListCuda, rhoListCuda, primeList.size(),
            smallPrimeWheel1, smallPrimeWheel2, smallPrimeWheel3, smallPrimeWheel4, 144,
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
            cudaMemcpy(&val, globalSieve2+322375634, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            if (val != 1834999230) {printf("FAILED PSP TEST IN KERNEL 2 %u\n", val);}
            else {printf("Passed kernel 2 tests.\n");}
        }
#endif
        if (i>0) {
            displayResultsAndClear(resultListHost);
        }
        cudaDeviceSynchronize();
        
        if (lastPrimeInLastBlock && (*firstPrimeInBlock - lastPrimeInLastBlock >= MIN_GAP_SIZE)) {
            checkGapAndPrint(lastPrimeInLastBlock, *firstPrimeInBlock);
        }
        lastPrimeInLastBlock = *lastPrimeInBlock;

        sieveStart += sieveLength;
        std::swap(globalSieve1, globalSieve2);
    }

    if ((blocksToTest-2)%PROGRESS_UPDATE_BLOCKS == 0) {
        finish = printProgress(start, finish, sieveStart, blocksToTest-2);
    }
    cudaMemcpy(resultListHost, resultList, sizeof(PrimeGap)*RESULT_LIST_SIZE, cudaMemcpyDeviceToHost);
    cudaMemset(resultList, 0, RESULT_LIST_SIZE * sizeof(PrimeGap));
#if RUN_TESTS
    assert(resultListHost[0].gap == 12); // only 12 gaps of size >=720 in this block, a lot less than expected
#endif

    kernel2<<<192,512>>>(globalSieve1, sieveStart, sieveLengthWords, resultList, firstPrimeInBlock, lastPrimeInBlock);
    displayResultsAndClear(resultListHost);
    cudaDeviceSynchronize();

    if (lastPrimeInLastBlock && (*firstPrimeInBlock - lastPrimeInLastBlock >= MIN_GAP_SIZE)) {
        printGap(lastPrimeInLastBlock, *firstPrimeInBlock);
    }

    if ((blocksToTest-1)%PROGRESS_UPDATE_BLOCKS == 0) {
        finish = printProgress(start, finish, sieveStart, blocksToTest-1);
    }

    cudaMemcpy(resultListHost, resultList, sizeof(PrimeGap)*RESULT_LIST_SIZE, cudaMemcpyDeviceToHost);
#if RUN_TESTS
    assert(resultListHost[0].gap == 22); // only 12 gaps of size >=720 in this block, a lot less than expected
#endif
    displayResultsAndClear(resultListHost);
    sieveStart += sieveLength;

    finish = printProgress(start, finish, sieveStart, blocksToTest);

    return 0;
}

/*
- General tips:
    - Don't be afraid to UNROLL LOOPS!
    - Inline functions?
    - We need to use an AtomicOr function on ALL STEPS!!! Because some blocks might be in a different step than others!
        - This includes the shared memory step, because we cannot miss any bits with sieving small primes due to PSPs

THINGS TO ADD:

settings/worktodo files: (should put this in the readme at some point)

=== settings.txt file format: ===
# comment: there should be a script to automatically find the optimal parameters to set (far future)
GPU_BLOCKS=192
NUM_BLOCKS_FOR_SIEVING=120
GPU_THREADS=512
BLOCK_SIZE=46080000000
SORT_OUTPUT_BY_GAPSIZE=1 # If 0, sorts by the prime (increasing). If 1, sorts by the gap size (decreasing)
NAME=B.Kehrig

=== worktodo.txt file format: ===
# format: start(e12), end(e12), minGap, username
18470057,18571674,1200,B.Kehrig  #will find the 1552 and 1572 gaps, but also will take a while (smaller tasks recommended)

=== output file format: === (location: output/gaps_<start>e12_<end>e12_min<mingap>_<name>.txt)
===== PRIME GAP REPORT =====
Target gap size: <mingap>
Gaps >=1200: <x> (or whatever hundred is at least as large as mingap)
Gaps >=1300: <x>
... keep going until there are none left
Largest gap: <size> <prime> <merit> <name>

All gaps >= <mingap>: # format: <gapsize> <startprime> <merit> <name>
1572 18571673432051830099 35.430806 B.Kehrig
1552 18470057946260698231 34.984359 B.Kehrig # (these would be in the opposite order if SORT_OUTPUPT_BY_GAPSIZE=0)


How do I write TESTS???
Use a compile-time variable RUN_TESTS
if set, it will do a run with some specified settings (start num, block size, etc...) to stay consistent
it will look at certain values from sieving and pseudoprime sieving and compare that to the correct values
For testing the primality tests, ???

For RTX 4090, ideal settings seems to be:
256 thread blocks
Use 11520 words in shared memory, leaving 768
Can get 1.23e12/s ideally

*/