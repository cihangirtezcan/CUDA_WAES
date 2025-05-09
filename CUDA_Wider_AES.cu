#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <ctime>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CUDA_Wider_AES.h"
__global__ void AES128_Exhaustive_Search(u32* pt, u32* ct, u32* rk, u32* t0G, u32* t4G, u32* rconG, u64* range, u8* SAES) {
	u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	//	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;
		// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 rconS[RCON_SIZE];
	__shared__ u32 ctS[U32_SIZE];
	//	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u8 Sbox[64][32][4];

	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
			Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x];
		}
		//		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];		}
		//		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) { Sbox[threadIdx.x/4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x]; }
		if (threadIdx.x < U32_SIZE) { ctS[threadIdx.x] = ct[threadIdx.x]; }
		if (threadIdx.x < RCON_SIZE) { rconS[threadIdx.x] = rconG[threadIdx.x]; }

	}	// </SHARED MEMORY>
	__syncthreads(); // Wait until every thread is ready
	u32 rk0Init, rk1Init, rk2Init, rk3Init;
	rk0Init = rk[0];	rk1Init = rk[1];	rk2Init = rk[2];	rk3Init = rk[3];
	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	pt0Init = pt[0];	pt1Init = pt[1];	pt2Init = pt[2];	pt3Init = pt[3];
	u64 threadRange = *range;
	u64 threadRangeStart = threadIndex * threadRange;
	rk2Init = rk2Init + threadRangeStart / (u64)MAX_U32;
	rk3Init = rk3Init + threadRangeStart % (u64)MAX_U32;
	for (u64 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		u32 rk0, rk1, rk2, rk3;
		rk0 = rk0Init;		rk1 = rk1Init;		rk2 = rk2Init;		rk3 = rk3Init;
		// Create plaintext as 32 bit unsigned integers
		u32 s0, s1, s2, s3;
		s0 = pt0Init;		s1 = pt1Init;		s2 = pt2Init;		s3 = pt3Init;
		// First round just XORs input with key.
		s0 = s0 ^ rk0;		s1 = s1 ^ rk1;		s2 = s2 ^ rk2;		s3 = s3 ^ rk3;
		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
			// Calculate round key
			u32 temp = rk3;
			rk0 = rk0 ^
				arithmeticRightShift((u32)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16)) % 4], 8) ^
				arithmeticRightShift((u32)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8)) % 4], 16) ^
				arithmeticRightShift((u32)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp)) % 4], 24) ^
				((u32)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^
				rconS[roundCount];
			rk1 = rk1 ^ rk0;			rk2 = rk2 ^ rk1;			rk3 = rk2 ^ rk3;
			// Table based round function
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s3 & 0xFF][warpThreadIndex], 24) ^ rk0;
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s0 & 0xFF][warpThreadIndex], 24) ^ rk1;
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s1 & 0xFF][warpThreadIndex], 24) ^ rk2;
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s2 & 0xFF][warpThreadIndex], 24) ^ rk3;
			s0 = t0;			s1 = t1;			s2 = t2;			s3 = t3;
		}
		// Calculate the last round key
		u32 temp = rk3;
		rk0 = rk0 ^
			arithmeticRightShift((u32)Sbox[((temp >> 16) & 0xff) / 4][warpThreadIndex][((temp >> 16)) % 4], 8) ^
			arithmeticRightShift((u32)Sbox[((temp >> 8) & 0xff) / 4][warpThreadIndex][((temp >> 8)) % 4], 16) ^
			arithmeticRightShift((u32)Sbox[((temp) & 0xff) / 4][warpThreadIndex][((temp)) % 4], 24) ^
			((u32)Sbox[((temp >> 24) / 4)][warpThreadIndex][((temp >> 24) % 4)]) ^
			rconS[ROUND_COUNT_MIN_1];
		// Last round uses s-box directly and XORs to produce output.
		s0 = arithmeticRightShift((u32)Sbox[((t0 >> 24)) / 4][warpThreadIndex][((t0 >> 24)) % 4], 8) ^ arithmeticRightShift((u32)Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex][((t1 >> 16)) % 4], 16) ^ arithmeticRightShift((u32)Sbox[((t2 >> 8) & 0xFF) / 4][warpThreadIndex][((t2 >> 8)) % 4], 24) ^ ((u32)Sbox[((t3 & 0xFF) / 4)][warpThreadIndex][((t3 & 0xFF) % 4)]) ^ rk0;
		if (s0 == ctS[0]) {
			rk1 = rk1 ^ rk0;
			s1 = arithmeticRightShift((u32)Sbox[((t1 >> 24)) / 4][warpThreadIndex][((t1 >> 24)) % 4], 8) ^ arithmeticRightShift((u32)Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex][((t2 >> 16)) % 4], 16) ^ arithmeticRightShift((u32)Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex][((t3 >> 8)) % 4], 24) ^ ((u32)Sbox[((t0 & 0xFF) / 4)][warpThreadIndex][((t0 & 0xFF) % 4)]) ^ rk1;
			if (s1 == ctS[1]) {
				rk2 = rk2 ^ rk1;
				s2 = arithmeticRightShift((u32)Sbox[((t2 >> 24)) / 4][warpThreadIndex][((t2 >> 24)) % 4], 8) ^ arithmeticRightShift((u32)Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex][((t3 >> 16)) % 4], 16) ^ arithmeticRightShift((u32)Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex][((t0 >> 8)) % 4], 24) ^ ((u32)Sbox[((t1 & 0xFF) / 4)][warpThreadIndex][((t1 & 0xFF) % 4)]) ^ rk2;
				if (s2 == ctS[2]) {
					rk3 = rk2 ^ rk3;
					s3 = arithmeticRightShift((u32)Sbox[((t3 >> 24)) / 4][warpThreadIndex][((t3 >> 24)) % 4], 8) ^ arithmeticRightShift((u32)Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex][((t0 >> 16)) % 4], 16) ^ arithmeticRightShift((u32)Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex][((t1 >> 8)) % 4], 24) ^ ((u32)Sbox[((t2 & 0xFF) / 4)][warpThreadIndex][((t2 & 0xFF) % 4)]) ^ rk3;
					if (s3 == ctS[3]) {
						printf("! Found key : %08x %08x %08x %08x\n", rk0Init, rk1Init, rk2Init, rk3Init);
						printf("-------------------------------\n");
					}
				}
			}
		}
		// Overflow
		if (rk3Init == MAX_U32) { rk2Init++; }
		rk3Init++;		// Create key as 32 bit unsigned integers
	}
}
__global__ void AES128_CTR(u32* pt, u32* rk, u32* t0G, u32* t4G, u64* range, u8* SAES) {
	u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u8 Sbox[64][32][4];
	//	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rkS[AES_128_KEY_SIZE_INT];
	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
			Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x];
		}
		//		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {	t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];	}
		//		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) { Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x]; }
		if (threadIdx.x < AES_128_KEY_SIZE_INT) { rkS[threadIdx.x] = rk[threadIdx.x]; }
	}
	__syncthreads();
	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	u32 s0, s1, s2, s3;
	pt0Init = pt[0];	pt1Init = pt[1];	pt2Init = pt[2];	pt3Init = pt[3];
	u64 threadRange = *range;
	u64 threadRangeStart = pt2Init;
	threadRangeStart = threadRangeStart << 32;
	threadRangeStart ^= pt3Init;
	threadRangeStart += threadIndex * threadRange;
	pt2Init = threadRangeStart >> 32;
	pt3Init = threadRangeStart & 0xFFFFFFFF;
	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		// Create plaintext as 32 bit unsigned integers
		s0 = pt0Init;		s1 = pt1Init;		s2 = pt2Init;		s3 = pt3Init;
		// First round just XORs input with key.
		s0 = s0 ^ rkS[0];		s1 = s1 ^ rkS[1];		s2 = s2 ^ rkS[2];		s3 = s3 ^ rkS[3];
		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1; roundCount++) {
			// Table based round function
			u32 rkStart = roundCount * 4 + 4;
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s3 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart];
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s0 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 1];
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s1 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 2];
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s2 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 3];
			s0 = t0;			s1 = t1;			s2 = t2;			s3 = t3;
		}
		// Calculate the last round key
		// Last round uses s-box directly and XORs to produce output.
/*		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[40];
		s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[41];
		s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[42];
		s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[43];*/
		s0 = arithmeticRightShift((u64)Sbox[((t0 >> 24)) / 4][warpThreadIndex][((t0 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex][((t1 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t2 >> 8) & 0xFF) / 4][warpThreadIndex][((t2 >> 8)) % 4], 24) ^ ((u64)Sbox[((t3 & 0xFF) / 4)][warpThreadIndex][((t3 & 0xFF) % 4)]) ^ rkS[40];
		s1 = arithmeticRightShift((u64)Sbox[((t1 >> 24)) / 4][warpThreadIndex][((t1 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex][((t2 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex][((t3 >> 8)) % 4], 24) ^ ((u64)Sbox[((t0 & 0xFF) / 4)][warpThreadIndex][((t0 & 0xFF) % 4)]) ^ rkS[41];
		s2 = arithmeticRightShift((u64)Sbox[((t2 >> 24)) / 4][warpThreadIndex][((t2 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex][((t3 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex][((t0 >> 8)) % 4], 24) ^ ((u64)Sbox[((t1 & 0xFF) / 4)][warpThreadIndex][((t1 & 0xFF) % 4)]) ^ rkS[42];
		s3 = arithmeticRightShift((u64)Sbox[((t3 >> 24)) / 4][warpThreadIndex][((t3 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex][((t0 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex][((t1 >> 8)) % 4], 24) ^ ((u64)Sbox[((t2 & 0xFF) / 4)][warpThreadIndex][((t2 & 0xFF) % 4)]) ^ rkS[43];
		// Overflow
		if (pt3Init == MAX_U32) { pt2Init++; }
		pt3Init++;
	}
	if (threadIndex == 1048575) {
		printf("threadIndex : %lld\n", threadIndex);
		printf("Plaintext   : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		printf("Ciphertext  : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		printf("-------------------------------\n");
	}
}
__global__ void AES256_CTR(u32* pt, u32* rk, u32* t0G, u32* t4G, u64* range) {

	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;

	// <SHARED MEMORY>
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rkS[AES_256_KEY_SIZE_INT];

	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
		}

		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {
			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];
		}

		if (threadIdx.x < AES_256_KEY_SIZE_INT) {
			rkS[threadIdx.x] = rk[threadIdx.x];
		}

	}
	// </SHARED MEMORY>

	// Wait until every thread is ready
	__syncthreads();

	u32 pt0Init, pt1Init, pt2Init, pt3Init;
	u32 s0, s1, s2, s3;
	pt0Init = pt[0];
	pt1Init = pt[1];
	pt2Init = pt[2];
	pt3Init = pt[3];

	u32 threadRange = *range;
	u64 threadRangeStart = pt2Init;
	threadRangeStart = threadRangeStart << 32;
	threadRangeStart ^= pt3Init;
	threadRangeStart += (u64)threadIndex * threadRange;
	pt2Init = threadRangeStart >> 32;
	pt3Init = threadRangeStart & 0xFFFFFFFF;

	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {

		// Create plaintext as 32 bit unsigned integers
		s0 = pt0Init;
		s1 = pt1Init;
		s2 = pt2Init;
		s3 = pt3Init;

		// First round just XORs input with key.
		s0 = s0 ^ rkS[0];
		s1 = s1 ^ rkS[1];
		s2 = s2 ^ rkS[2];
		s3 = s3 ^ rkS[3];

		u32 t0, t1, t2, t3;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1_256; roundCount++) {

			// Table based round function
			u32 rkStart = roundCount * 4 + 4;
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s3 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart];
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s0 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 1];
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s1 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 2];
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShiftBytePerm(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], SHIFT_1_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], SHIFT_2_RIGHT) ^ arithmeticRightShiftBytePerm(t0S[s2 & 0xFF][warpThreadIndex], SHIFT_3_RIGHT) ^ rkS[rkStart + 3];

			s0 = t0;
			s1 = t1;
			s2 = t2;
			s3 = t3;

		}

		// Calculate the last round key
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[56];
		s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[57];
		s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[58];
		s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[59];

		//if (threadIndex == 0 && rangeCount == 0) {
		//printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		//}

		// Overflow
		if (pt3Init == MAX_U32) {
			pt2Init++;
		}

		// Create key as 32 bit unsigned integers
		pt3Init++;
	}

	if (threadIndex == 1048575) {
		printf("Plaintext : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		printf("-------------------------------\n");
	}

}
__global__ void Wider_AES256_CTR(u32* pt, u32* rk, u32* t0G, u32* t4G, u64* range) {
	int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	int warpThreadIndexSBox = warpThreadIndex % S_BOX_BANK_SIZE;
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u32 t4S[TABLE_SIZE][S_BOX_BANK_SIZE];
	__shared__ u32 rkS[120];
	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];		}
		for (u8 bankIndex = 0; bankIndex < S_BOX_BANK_SIZE; bankIndex++) {			t4S[threadIdx.x][bankIndex] = t4G[threadIdx.x];		}
		if (threadIdx.x < 120) {			rkS[threadIdx.x] = rk[threadIdx.x];		}
	}
	__syncthreads();
	u32 pt0Init, pt1Init, pt2Init, pt3Init, pt4Init, pt5Init, pt6Init, pt7Init;
	u32 s0, s1, s2, s3, s4, s5, s6, s7;
	pt0Init = pt[0];	pt1Init = pt[1];	pt2Init = pt[2];	pt3Init = pt[3]; pt4Init = pt[4];	pt5Init = pt[5];	pt6Init = pt[6];	pt7Init = pt[7];
	u32 threadRange = *range;
	u64 threadRangeStart = pt2Init;
	threadRangeStart = threadRangeStart << 32;
	threadRangeStart ^= pt3Init;
	threadRangeStart += (u64)threadIndex * threadRange;
	pt2Init = threadRangeStart >> 32;
	pt3Init = threadRangeStart & 0xFFFFFFFF;
	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		s0 = pt0Init;		s1 = pt1Init;		s2 = pt2Init;		s3 = pt3Init; s4 = pt4Init;		s5 = pt5Init;		s6 = pt6Init;		s7 = pt7Init;
		s0 = s0 ^ rkS[0];		s1 = s1 ^ rkS[1];		s2 = s2 ^ rkS[2];		s3 = s3 ^ rkS[3]; s4 = s4 ^ rkS[4];		s5 = s5 ^ rkS[5];		s6 = s6 ^ rkS[6];		s7 = s7 ^ rkS[7];
		u32 t0, t1, t2, t3, t4, t5, t6, t7;
		for (u8 roundCount = 0; roundCount < ROUND_COUNT_MIN_1_256; roundCount++) {
			// Table based round function
			u32 rkStart = roundCount * 8 + 8;
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s4 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart];
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s4 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s5 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 1];
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s5 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s6 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 2];
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s4 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s6 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s7 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 3];
			t4 = t0S[s4 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s5 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s7 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s0 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 4];
			t5 = t0S[s5 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s6 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s1 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 5];
			t6 = t0S[s6 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s7 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s2 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 6];
			t7 = t0S[s7 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s3 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 7];
			s0 = t0;			s1 = t1;			s2 = t2;			s3 = t3; s4 = t4;			s5 = t5;			s6 = t6;			s7 = t7;
		}
		// Calculate the last round key
		// Last round uses s-box directly and XORs to produce output.
		s0 = (t4S[t0 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t1 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t3 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t4) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[112];
		s1 = (t4S[t1 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t2 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t4 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t5) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[113];
		s2 = (t4S[t2 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t3 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t5 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t6) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[114];
		s3 = (t4S[t3 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t4 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t6 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t7) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[115];
		s4 = (t4S[t4 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t5 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t7 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t0) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[116];
		s5 = (t4S[t5 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t7 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t0 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t1) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[117];
		s6 = (t4S[t6 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t7 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t1 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t2) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[118];
		s7 = (t4S[t7 >> 24][warpThreadIndexSBox] & 0xFF000000) ^ (t4S[(t0 >> 16) & 0xff][warpThreadIndexSBox] & 0x00FF0000) ^ (t4S[(t2 >> 8) & 0xff][warpThreadIndexSBox] & 0x0000FF00) ^ (t4S[(t3) & 0xFF][warpThreadIndexSBox] & 0x000000FF) ^ rkS[119];

		// Overflow
		if (pt3Init == MAX_U32) {			pt2Init++;		}
		// Create key as 32 bit unsigned integers
		pt3Init++;
	}
	if (threadIndex == 1048575) {
		printf("Plaintext : %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init);
		printf("Ciphertext : %08x %08x %08x %08x\n", s0, s1, s2, s3);
		printf("-------------------------------\n");
	}

}
__global__ void Wider_AES128_CTR(u32* pt, u32* rk, u32* t0G, u32* t4G, u64* range, u8* SAES) {
	u64 threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int warpThreadIndex = threadIdx.x & 31;
	__shared__ u32 t0S[TABLE_SIZE][SHARED_MEM_BANK_SIZE];
	__shared__ u8 Sbox[64][32][4];
	__shared__ u32 rkS[WAES_256_KEY_SIZE_INT];
	if (threadIdx.x < TABLE_SIZE) {
		for (u8 bankIndex = 0; bankIndex < SHARED_MEM_BANK_SIZE; bankIndex++) {
			t0S[threadIdx.x][bankIndex] = t0G[threadIdx.x];
			Sbox[threadIdx.x / 4][bankIndex][threadIdx.x % 4] = SAES[threadIdx.x];
		}
		if (threadIdx.x < WAES_256_KEY_SIZE_INT) { rkS[threadIdx.x] = rk[threadIdx.x]; }
	}
	__syncthreads();
	u32 pt0Init, pt1Init, pt2Init, pt3Init, pt4Init, pt5Init, pt6Init, pt7Init;
	u32 s0, s1, s2, s3, s4, s5, s6, s7;
	pt0Init = pt[0];	pt1Init = pt[1];	pt2Init = pt[2];	pt3Init = pt[3]; pt4Init = pt[4];	pt5Init = pt[5];	pt6Init = pt[6];	pt7Init = pt[7];
	u64 threadRange = *range;
	u64 threadRangeStart = pt2Init;
	threadRangeStart = threadRangeStart << 32;
	threadRangeStart ^= pt3Init;
	threadRangeStart += threadIndex * threadRange;
	pt2Init = threadRangeStart >> 32;
	pt3Init = threadRangeStart & 0xFFFFFFFF;
	for (u32 rangeCount = 0; rangeCount < threadRange; rangeCount++) {
		// Create plaintext as 32 bit unsigned integers
		s0 = pt0Init;		s1 = pt1Init;		s2 = pt2Init;		s3 = pt3Init; s4 = pt4Init;		s5 = pt5Init;		s6 = pt6Init;		s7 = pt7Init;
		// First round just XORs input with key.
		s0 = s0 ^ rkS[0];		s1 = s1 ^ rkS[1];		s2 = s2 ^ rkS[2];		s3 = s3 ^ rkS[3]; s4 = s4 ^ rkS[4];		s5 = s5 ^ rkS[5];		s6 = s6 ^ rkS[6];		s7 = s7 ^ rkS[7];
		u32 t0, t1, t2, t3, t4, t5, t6, t7;
		for (u8 roundCount = 0; roundCount < 13; roundCount++) {
			// Table based round function
			u32 rkStart = roundCount * 8 + 8;
			t0 = t0S[s0 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s1 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s3 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s4 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart];
			t1 = t0S[s1 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s2 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s4 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s5 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 1];
			t2 = t0S[s2 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s3 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s5 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s6 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 2];
			t3 = t0S[s3 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s4 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s6 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s7 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 3];
			t4 = t0S[s4 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s5 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s7 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s0 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 4];
			t5 = t0S[s5 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s6 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s0 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s1 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 5];
			t6 = t0S[s6 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s7 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s1 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s2 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 6];
			t7 = t0S[s7 >> 24][warpThreadIndex] ^ arithmeticRightShift(t0S[(s0 >> 16) & 0xFF][warpThreadIndex], 8) ^ arithmeticRightShift(t0S[(s2 >> 8) & 0xFF][warpThreadIndex], 16) ^ arithmeticRightShift(t0S[s3 & 0xFF][warpThreadIndex], 24) ^ rkS[rkStart + 7];
			s0 = t0;			s1 = t1;			s2 = t2;			s3 = t3; s4 = t4;			s5 = t5;			s6 = t6;			s7 = t7;
		}
		// Calculate the last round key
		// Last round uses s-box directly and XORs to produce output.
		s0 = arithmeticRightShift((u64)Sbox[((t0 >> 24)) / 4][warpThreadIndex][((t0 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t1 >> 16) & 0xff) / 4][warpThreadIndex][((t1 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t3 >> 8) & 0xFF) / 4][warpThreadIndex][((t3 >> 8)) % 4], 24) ^ ((u64)Sbox[((t4 & 0xFF) / 4)][warpThreadIndex][((t4 & 0xFF) % 4)]) ^ rkS[112];
		s1 = arithmeticRightShift((u64)Sbox[((t1 >> 24)) / 4][warpThreadIndex][((t1 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t2 >> 16) & 0xff) / 4][warpThreadIndex][((t2 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t4 >> 8) & 0xFF) / 4][warpThreadIndex][((t4 >> 8)) % 4], 24) ^ ((u64)Sbox[((t5 & 0xFF) / 4)][warpThreadIndex][((t5 & 0xFF) % 4)]) ^ rkS[113];
		s2 = arithmeticRightShift((u64)Sbox[((t2 >> 24)) / 4][warpThreadIndex][((t2 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t3 >> 16) & 0xff) / 4][warpThreadIndex][((t3 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t5 >> 8) & 0xFF) / 4][warpThreadIndex][((t5 >> 8)) % 4], 24) ^ ((u64)Sbox[((t6 & 0xFF) / 4)][warpThreadIndex][((t6 & 0xFF) % 4)]) ^ rkS[114];
		s3 = arithmeticRightShift((u64)Sbox[((t3 >> 24)) / 4][warpThreadIndex][((t3 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t4 >> 16) & 0xff) / 4][warpThreadIndex][((t4 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t6 >> 8) & 0xFF) / 4][warpThreadIndex][((t6 >> 8)) % 4], 24) ^ ((u64)Sbox[((t7 & 0xFF) / 4)][warpThreadIndex][((t7 & 0xFF) % 4)]) ^ rkS[115];
		s4 = arithmeticRightShift((u64)Sbox[((t4 >> 24)) / 4][warpThreadIndex][((t4 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t5 >> 16) & 0xff) / 4][warpThreadIndex][((t5 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t7 >> 8) & 0xFF) / 4][warpThreadIndex][((t7 >> 8)) % 4], 24) ^ ((u64)Sbox[((t0 & 0xFF) / 4)][warpThreadIndex][((t0 & 0xFF) % 4)]) ^ rkS[116];
		s5 = arithmeticRightShift((u64)Sbox[((t5 >> 24)) / 4][warpThreadIndex][((t5 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t6 >> 16) & 0xff) / 4][warpThreadIndex][((t6 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t0 >> 8) & 0xFF) / 4][warpThreadIndex][((t0 >> 8)) % 4], 24) ^ ((u64)Sbox[((t1 & 0xFF) / 4)][warpThreadIndex][((t1 & 0xFF) % 4)]) ^ rkS[117];
		s6 = arithmeticRightShift((u64)Sbox[((t6 >> 24)) / 4][warpThreadIndex][((t6 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t7 >> 16) & 0xff) / 4][warpThreadIndex][((t7 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t1 >> 8) & 0xFF) / 4][warpThreadIndex][((t1 >> 8)) % 4], 24) ^ ((u64)Sbox[((t2 & 0xFF) / 4)][warpThreadIndex][((t2 & 0xFF) % 4)]) ^ rkS[118];
		s7 = arithmeticRightShift((u64)Sbox[((t7 >> 24)) / 4][warpThreadIndex][((t7 >> 24)) % 4], 8) ^ arithmeticRightShift((u64)Sbox[((t0 >> 16) & 0xff) / 4][warpThreadIndex][((t0 >> 16)) % 4], 16) ^ arithmeticRightShift((u64)Sbox[((t2 >> 8) & 0xFF) / 4][warpThreadIndex][((t2 >> 8)) % 4], 24) ^ ((u64)Sbox[((t3 & 0xFF) / 4)][warpThreadIndex][((t3 & 0xFF) % 4)]) ^ rkS[119];
		// Overflow
		if (pt3Init == MAX_U32) { pt2Init++; }
		pt3Init++;
	}
	if (threadIndex == 1048575) {
		printf("threadIndex : %lld\n", threadIndex);
		printf("Plaintext   : %08x %08x %08x %08x %08x %08x %08x %08x\n", pt0Init, pt1Init, pt2Init, pt3Init, pt4Init, pt5Init, pt6Init, pt7Init);
		printf("Ciphertext  : %08x %08x %08x %08x %08x %08x %08x %08x\n", s0, s1, s2, s3, s4, s5, s6, s7);
		printf("-------------------------------\n");
	}
}
__host__ void keyExpansion(u32* key, u32* rk) {
	u32 rk0, rk1, rk2, rk3;
	rk0 = key[0];
	rk1 = key[1];
	rk2 = key[2];
	rk3 = key[3];
	rk[0] = rk0;
	rk[1] = rk1;
	rk[2] = rk2;
	rk[3] = rk3;
	for (u8 roundCount = 0; roundCount < ROUND_COUNT; roundCount++) {
		u32 temp = rk3;
		rk0 = rk0 ^ T4_3[(temp >> 16) & 0xff] ^ T4_2[(temp >> 8) & 0xff] ^ T4_1[(temp) & 0xff] ^ T4_0[(temp >> 24)] ^ RCON32[roundCount];
		rk1 = rk1 ^ rk0;
		rk2 = rk2 ^ rk1;
		rk3 = rk2 ^ rk3;
		rk[roundCount * 4 + 4] = rk0;
		rk[roundCount * 4 + 5] = rk1;
		rk[roundCount * 4 + 6] = rk2;
		rk[roundCount * 4 + 7] = rk3;
	}
}
__host__ void keyExpansion256(u32* key, u32* rk) {

	u32 rk0, rk1, rk2, rk3, rk4, rk5, rk6, rk7;
	rk0 = key[0];
	rk1 = key[1];
	rk2 = key[2];
	rk3 = key[3];
	rk4 = key[4];
	rk5 = key[5];
	rk6 = key[6];
	rk7 = key[7];

	rk[0] = rk0;
	rk[1] = rk1;
	rk[2] = rk2;
	rk[3] = rk3;
	rk[4] = rk4;
	rk[5] = rk5;
	rk[6] = rk6;
	rk[7] = rk7;

	for (u8 roundCount = 0; roundCount < ROUND_COUNT_256; roundCount++) {
		u32 temp = rk7;
		rk0 = rk0 ^ T4_3[(temp >> 16) & 0xff] ^ T4_2[(temp >> 8) & 0xff] ^ T4_1[(temp) & 0xff] ^ T4_0[(temp >> 24)] ^ RCON32[roundCount];
		rk1 = rk1 ^ rk0;
		rk2 = rk2 ^ rk1;
		rk3 = rk3 ^ rk2;
		rk4 = rk4 ^ T4_3[(rk3 >> 24) & 0xff] ^ T4_2[(rk3 >> 16) & 0xff] ^ T4_1[(rk3 >> 8) & 0xff] ^ T4_0[rk3 & 0xff];
		rk5 = rk5 ^ rk4;
		rk6 = rk6 ^ rk5;
		rk7 = rk7 ^ rk6;

		rk[roundCount * 8 + 8] = rk0;
		rk[roundCount * 8 + 9] = rk1;
		rk[roundCount * 8 + 10] = rk2;
		rk[roundCount * 8 + 11] = rk3;
		if (roundCount == 6) {
			break;
		}
		rk[roundCount * 8 + 12] = rk4;
		rk[roundCount * 8 + 13] = rk5;
		rk[roundCount * 8 + 14] = rk6;
		rk[roundCount * 8 + 15] = rk7;

	}

	//for (int i = 0; i < 60; i++) {
	//	printf("%08x ", rk[i]);
	//	if ((i + 1) % 4 == 0) {
	//		printf("Round: %d\n", i / 4);
	//	}
	//}
}
__host__ void keyExpansionW256(u32* key, u32* rk) {
	u32 rk0, rk1, rk2, rk3, rk4, rk5, rk6, rk7;
	rk0 = key[0];
	rk1 = key[1];
	rk2 = key[2];
	rk3 = key[3];
	rk4 = key[4];
	rk5 = key[5];
	rk6 = key[6];
	rk7 = key[7];

	rk[0] = rk0;
	rk[1] = rk1;
	rk[2] = rk2;
	rk[3] = rk3;
	rk[4] = rk4;
	rk[5] = rk5;
	rk[6] = rk6;
	rk[7] = rk7;

	for (u8 roundCount = 0; roundCount < ROUND_COUNT_256; roundCount++) {
		u32 temp = rk7;
		rk0 = rk0 ^ T4_3[(temp >> 16) & 0xff] ^ T4_2[(temp >> 8) & 0xff] ^ T4_1[(temp) & 0xff] ^ T4_0[(temp >> 24)] ^ RCON32[roundCount];
		rk1 = rk1 ^ rk0;
		rk2 = rk2 ^ rk1;
		rk3 = rk3 ^ rk2;
		rk4 = rk4 ^ T4_3[(rk3 >> 24) & 0xff] ^ T4_2[(rk3 >> 16) & 0xff] ^ T4_1[(rk3 >> 8) & 0xff] ^ T4_0[rk3 & 0xff];
		rk5 = rk5 ^ rk4;
		rk6 = rk6 ^ rk5;
		rk7 = rk7 ^ rk6;

		rk[roundCount * 8 + 8] = rk0;
		rk[roundCount * 8 + 9] = rk1;
		rk[roundCount * 8 + 10] = rk2;
		rk[roundCount * 8 + 11] = rk3;
		if (roundCount == 6) {
			break;
		}
		rk[roundCount * 8 + 12] = rk4;
		rk[roundCount * 8 + 13] = rk5;
		rk[roundCount * 8 + 14] = rk6;
		rk[roundCount * 8 + 15] = rk7;

	}
}
__host__ int AES128ExhaustiveSearch(int choice) {
	printf("\n");	printf("########## AES-128 Exhaustive Search Implementation ##########\n");	printf("\n");
	// Allocate plaintext, ciphertext and initial round key
	u32* pt, * ct, * rk;
	gpuErrorCheck(cudaMallocManaged(&pt, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&ct, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&rk, 4 * sizeof(u32)));
	pt[0] = 0x3243F6A8U;	pt[1] = 0x885A308DU;	pt[2] = 0x313198A2U;	pt[3] = 0xE0370734U;
	//	pt[0] = 0;	pt[1] = 0;	pt[2] = 0;	pt[3] = 0;
	ct[0] = 0x3925841DU;	ct[1] = 0x02DC09FBU;	ct[2] = 0xDC118597U;	ct[3] = 0x196A0B32U;
	// aes-cipher-internals.xlsx
	rk[0] = 0x2B7E1516U;	rk[1] = 0x28AED2A6U;	rk[2] = 0xABF71588U;	rk[3] = 0x09CF0000U;
	// Allocate RCON values
	u32* rcon;
	gpuErrorCheck(cudaMallocManaged(&rcon, RCON_SIZE * sizeof(u32)));
	for (int i = 0; i < RCON_SIZE; i++) { rcon[i] = RCON32[i]; }
	// Allocate Tables
	u32* t0, * t1, * t2, * t3, * t4, * t4_0, * t4_1, * t4_2, * t4_3;
	u8* SAES_d; // Cihangir
	gpuErrorCheck(cudaMallocManaged(&t0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&SAES_d, 256 * sizeof(u8))); // Cihangir
	for (int i = 0; i < TABLE_SIZE; i++) {
		t0[i] = T0[i];		t1[i] = T1[i];		t2[i] = T2[i];		t3[i] = T3[i];		t4[i] = T4[i];
		t4_0[i] = T4_0[i];		t4_1[i] = T4_1[i];		t4_2[i] = T4_2[i];		t4_3[i] = T4_3[i];
	}
	for (int i = 0; i < 256; i++) SAES_d[i] = SAES[i]; // Cihangir
	printf("-------------------------------\n");
	u64* range = calculateRange();
	/*	printf("Plaintext                     : %08x %08x %08x %08x\n", pt[0], pt[1], pt[2], pt[3]);
		printf("Ciphertext                    : %08x %08x %08x %08x\n", ct[0], ct[1], ct[2], ct[3]);
		printf("Initial Key                   : %08x %08x %08x %08x\n", rk[0], rk[1], rk[2], rk[3]);
		printf("-------------------------------\n");*/

	clock_t beginTime = clock();
	if (choice == 1) AES128_Exhaustive_Search << <BLOCKS, THREADS >> > (pt, ct, rk, t0, t4, rcon, range, SAES_d);
	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);

	printf("-------------------------------\n");
	printLastCUDAError();
	// Free alocated arrays
	cudaFree(range); cudaFree(pt);	cudaFree(ct);	cudaFree(rk);	cudaFree(t0);	cudaFree(t1);	cudaFree(t2);	cudaFree(t3);	cudaFree(t4);
	cudaFree(t4_0);	cudaFree(t4_1);	cudaFree(t4_2);	cudaFree(t4_3);	cudaFree(rcon); cudaFree(SAES_d);
	return 0;
}
__host__ int AES128Ctr() {
	printf("\n");
	printf("########## AES-128 Counter Mode Implementation ##########\n");
	printf("\n");

	// Allocate plaintext and every round key
	u32* pt, * rk, * roundKeys;
	gpuErrorCheck(cudaMallocManaged(&pt, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&rk, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&roundKeys, AES_128_KEY_SIZE_INT * sizeof(u32)));

	pt[0] = 0x3243F6A8U;
	pt[1] = 0x885A308DU;
	pt[2] = 0x313198A2U;
	pt[3] = 0x00000000U;

	rk[0] = 0x2B7E1516U;
	rk[1] = 0x28AED2A6U;
	rk[2] = 0xABF71588U;
	rk[3] = 0x09CF4F3CU;

	// Allocate RCON values
	u32* rcon;
	gpuErrorCheck(cudaMallocManaged(&rcon, RCON_SIZE * sizeof(u32)));
	for (int i = 0; i < RCON_SIZE; i++) {
		rcon[i] = RCON32[i];
	}

	// Allocate Tables
	u32* t0, * t1, * t2, * t3, * t4, * t4_0, * t4_1, * t4_2, * t4_3;
	u8* SAES_d; // Cihangir
	gpuErrorCheck(cudaMallocManaged(&t0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&SAES_d, 256 * sizeof(u8))); // Cihangir
	for (int i = 0; i < TABLE_SIZE; i++) {
		t0[i] = T0[i];
		t1[i] = T1[i];
		t2[i] = T2[i];
		t3[i] = T3[i];
		t4[i] = T4[i];
		t4_0[i] = T4_0[i];
		t4_1[i] = T4_1[i];
		t4_2[i] = T4_2[i];
		t4_3[i] = T4_3[i];
	}
	for (int i = 0; i < 256; i++) SAES_d[i] = SAES[i]; // Cihangir
	printf("-------------------------------\n");
	u64* range = calculateRange();
	/*	printf("Initial Plaintext              : %08x %08x %08x %08x\n", pt[0], pt[1], pt[2], pt[3]);
		printf("Initial Key                    : %08x %08x %08x %08x\n", rk[0], rk[1], rk[2], rk[3]);
		printf("-------------------------------\n");*/

		// Key expansion
	keyExpansion(rk, roundKeys);

	clock_t beginTime = clock();
	// Kernels
//	counterWithOneTableExtendedSharedMemoryBytePermPartlyExtendedSBox<<<BLOCKS, THREADS>>>(pt, roundKeys, t0, t4, range);
	AES128_CTR << <BLOCKS, THREADS >> > (pt, roundKeys, t0, t4, range, SAES_d);
	//	counterWithOneTableExtendedSharedMemoryBytePerm4ShiftedSbox<<<BLOCKS, THREADS>>>(pt, roundKeys, t0, t4_0, t4_1, t4_2, t4_3, range);
	//	cudaMemcpy(rk, pt, 4*sizeof(u32), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
	printf("-------------------------------\n");
	printLastCUDAError();
	printf("plaintext: %x %x %x %x\n", rk[0], rk[1], rk[2], rk[3]);

	// Free alocated arrays
	cudaFree(range);
	cudaFree(pt);
	cudaFree(rk);
	cudaFree(roundKeys);
	cudaFree(t0);
	cudaFree(t1);
	cudaFree(t2);
	cudaFree(t3);
	cudaFree(t4);
	cudaFree(t4_0);
	cudaFree(t4_1);
	cudaFree(t4_2);
	cudaFree(t4_3);
	cudaFree(rcon);

	return 0;
}
__host__ int WAES128Ctr() {
	printf("\n");	printf("########## WAES-128 Counter Mode Implementation ##########\n");	printf("\n");
	// Allocate plaintext and every round key
	u32* pt, * rk, * roundKeys;
	gpuErrorCheck(cudaMallocManaged(&pt, 8 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&rk, 8 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&roundKeys, WAES_256_KEY_SIZE_INT * sizeof(u32)));
	pt[0] = 0x3243F6A8U;	pt[1] = 0x885A308DU;	pt[2] = 0x313198A2U;	pt[3] = 0x00000000U;	pt[4] = 0x3243F6A8U;	pt[5] = 0x885A308DU;	pt[6] = 0x313198A2U;	pt[7] = 0x00000000U;
	rk[0] = 0x2B7E1516U;	rk[1] = 0x28AED2A6U;	rk[2] = 0xABF71588U;	rk[3] = 0x09CF4F3CU;	rk[4] = 0x2B7E1516U;	rk[5] = 0x28AED2A6U;	rk[6] = 0xABF71588U;	rk[7] = 0x09CF4F3CU;
	// Allocate RCON values
	u32* rcon;
	gpuErrorCheck(cudaMallocManaged(&rcon, RCON_SIZE * 2* sizeof(u32)));
	for (int i = 0; i < RCON_SIZE*2; i++) {
		rcon[i] = RCON32[i];
	}
	// Allocate Tables
	u32* t0, * t1, * t2, * t3, * t4, * t4_0, * t4_1, * t4_2, * t4_3;
	u8* SAES_d; // Cihangir
	gpuErrorCheck(cudaMallocManaged(&t0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&SAES_d, 256 * sizeof(u8))); // Cihangir
	for (int i = 0; i < TABLE_SIZE; i++) {
		t0[i] = T0[i];
		t1[i] = T1[i];
		t2[i] = T2[i];
		t3[i] = T3[i];
		t4[i] = T4[i];
		t4_0[i] = T4_0[i];
		t4_1[i] = T4_1[i];
		t4_2[i] = T4_2[i];
		t4_3[i] = T4_3[i];
	}
	for (int i = 0; i < 256; i++) SAES_d[i] = SAES[i]; // Cihangir
	printf("-------------------------------\n");
	u64* range = calculateRange();
	keyExpansionW256(rk, roundKeys);	
	clock_t beginTime = clock();
	Wider_AES128_CTR << <BLOCKS, THREADS >> > (pt, roundKeys, t0, t4, range, SAES_d);
	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
	printf("-------------------------------\n");
	printLastCUDAError();
	printf("plaintext: %x %x %x %x\n", rk[0], rk[1], rk[2], rk[3]);

	cudaFree(range);	cudaFree(pt);	cudaFree(rk);	cudaFree(roundKeys);	cudaFree(t0);	cudaFree(t1);	cudaFree(t2);	cudaFree(t3);	cudaFree(t4);	cudaFree(t4_0);	cudaFree(t4_1);	cudaFree(t4_2);	cudaFree(t4_3);	cudaFree(rcon);
	return 0;
}
__host__ int AES256Ctr() {
	printf("\n");
	printf("########## AES-256 Counter Mode Implementation ##########\n");
	printf("\n");

	// Allocate plaintext and every round key
	u32* pt, * ct, * rk256, * roundKeys256;
	gpuErrorCheck(cudaMallocManaged(&pt, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&ct, 4 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&rk256, 8 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&roundKeys256, AES_256_KEY_SIZE_INT * sizeof(u32)));

	pt[0] = 0x6bc1bee2U;
	pt[1] = 0x2e409f96U;
	pt[2] = 0xe93d7e11U;
	pt[3] = 0x7393172aU;

	ct[0] = 0xF3EED1BDU;
	ct[1] = 0xB5D2A03CU;
	ct[2] = 0x064B5A7EU;
	ct[3] = 0x3DB181F8U;

	rk256[0] = 0x603deb10U;
	rk256[1] = 0x15ca71beU;
	rk256[2] = 0x2b73aef0U;
	rk256[3] = 0x857d7781U;
	rk256[4] = 0x1f352c07U;
	rk256[5] = 0x3b6108d7U;
	rk256[6] = 0x2d9810a3U;
	rk256[7] = 0x0914dff4U;

	// Allocate RCON values
	u32* rcon;
	gpuErrorCheck(cudaMallocManaged(&rcon, RCON_SIZE * sizeof(u32)));
	for (int i = 0; i < RCON_SIZE; i++) {
		rcon[i] = RCON32[i];
	}

	// Allocate Tables
	u32* t0, * t1, * t2, * t3, * t4, * t4_0, * t4_1, * t4_2, * t4_3;
	gpuErrorCheck(cudaMallocManaged(&t0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_3, TABLE_SIZE * sizeof(u32)));
	for (int i = 0; i < TABLE_SIZE; i++) {
		t0[i] = T0[i];
		t1[i] = T1[i];
		t2[i] = T2[i];
		t3[i] = T3[i];
		t4[i] = T4[i];
		t4_0[i] = T4_0[i];
		t4_1[i] = T4_1[i];
		t4_2[i] = T4_2[i];
		t4_3[i] = T4_3[i];
	}

	printf("-------------------------------\n");
	u64* range = calculateRange();
	/*	printf("Initial Plaintext              : %08x %08x %08x %08x\n", pt[0], pt[1], pt[2], pt[3]);
		printf("Initial Key                    : %08x %08x %08x %08x %08x %08x %08x %08x\n", rk256[0], rk256[1], rk256[2], rk256[3], rk256[4], rk256[5], rk256[6], rk256[7]);
		printf("-------------------------------\n");*/

	keyExpansion256(rk256, roundKeys256);
	clock_t beginTime = clock();
	// Kernels
	AES256_CTR << <BLOCKS, THREADS >> > (pt, roundKeys256, t0, t4, range);

	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
	printf("-------------------------------\n");
	printLastCUDAError();

	// Free alocated arrays
	cudaFree(range);
	cudaFree(pt);
	cudaFree(ct);
	cudaFree(rk256);
	cudaFree(roundKeys256);
	cudaFree(t0);
	cudaFree(t1);
	cudaFree(t2);
	cudaFree(t3);
	cudaFree(t4);
	cudaFree(t4_0);
	cudaFree(t4_1);
	cudaFree(t4_2);
	cudaFree(t4_3);
	cudaFree(rcon);


	return 0;
}
__host__ int WAES256Ctr() {
	printf("\n");	printf("########## WAES-256 Counter Mode Implementation ##########\n");	printf("\n");
	// Allocate plaintext and every round key
	u32* pt, * ct, * rk256, * roundKeys256;
	gpuErrorCheck(cudaMallocManaged(&pt, 8 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&ct, 8 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&rk256, 8 * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&roundKeys256, WAES_256_KEY_SIZE_INT * sizeof(u32)));
	pt[0] = 0x3243F6A8U;	pt[1] = 0x885A308DU;	pt[2] = 0x313198A2U;	pt[3] = 0x00000000U;	pt[4] = 0x3243F6A8U;	pt[5] = 0x885A308DU;	pt[6] = 0x313198A2U;	pt[7] = 0x00000000U;
	ct[0] = 0xF3EED1BDU;	ct[1] = 0xB5D2A03CU;	ct[2] = 0x064B5A7EU;	ct[3] = 0x3DB181F8U;
	rk256[0] = 0x603deb10U;	rk256[1] = 0x15ca71beU;	rk256[2] = 0x2b73aef0U;	rk256[3] = 0x857d7781U;	rk256[4] = 0x1f352c07U;	rk256[5] = 0x3b6108d7U;	rk256[6] = 0x2d9810a3U;	rk256[7] = 0x0914dff4U;
	// Allocate RCON values
	u32* rcon;
	gpuErrorCheck(cudaMallocManaged(&rcon, RCON_SIZE *2* sizeof(u32)));
	for (int i = 0; i < RCON_SIZE; i++) {		rcon[i] = RCON32[i];	}
	// Allocate Tables
	u32* t0, * t1, * t2, * t3, * t4, * t4_0, * t4_1, * t4_2, * t4_3;
	gpuErrorCheck(cudaMallocManaged(&t0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t3, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_0, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_1, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_2, TABLE_SIZE * sizeof(u32)));
	gpuErrorCheck(cudaMallocManaged(&t4_3, TABLE_SIZE * sizeof(u32)));
	for (int i = 0; i < TABLE_SIZE; i++) {
		t0[i] = T0[i];
		t1[i] = T1[i];
		t2[i] = T2[i];
		t3[i] = T3[i];
		t4[i] = T4[i];
		t4_0[i] = T4_0[i];
		t4_1[i] = T4_1[i];
		t4_2[i] = T4_2[i];
		t4_3[i] = T4_3[i];
	}
	printf("-------------------------------\n");
	u64* range = calculateRange();
	keyExpansionW256(rk256, roundKeys256);
	clock_t beginTime = clock();
	Wider_AES256_CTR << <BLOCKS, THREADS >> > (pt, roundKeys256, t0, t4, range);
	cudaDeviceSynchronize();
	printf("Time elapsed: %f sec\n", float(clock() - beginTime) / CLOCKS_PER_SEC);
	printf("-------------------------------\n");
	printLastCUDAError();
	cudaFree(range);	cudaFree(pt);	cudaFree(ct);	cudaFree(rk256);	cudaFree(roundKeys256);	cudaFree(t0);	cudaFree(t1);	cudaFree(t2);	cudaFree(t3);	cudaFree(t4);	cudaFree(t4_0);	cudaFree(t4_1);	cudaFree(t4_2);	cudaFree(t4_3);	cudaFree(rcon);
	return 0;
}
int main() {
	cudaSetDevice(0);
	int choice;
	printf(
		"(1)  AES-128 Exhaustive Search\n"
		"(2)  AES-128 CTR \n"
		"(3)  Wider AES-128 Exhaustive Search\n"
		"(4)  Wider AES-128/192/256 CTR \n"
		"(5)  AES-256 CTR \n"
		"(6)  WAES-256 CTR \n"
		"Choice: ");
	scanf_s("%d", &choice);
	if (choice == 1) AES128ExhaustiveSearch(1);
	else if (choice == 2) AES128Ctr();
	else if (choice == 4) WAES128Ctr();
	else if (choice == 5) AES256Ctr();
	else if (choice == 6) WAES256Ctr();
	else printf("Wrong selection\n");
	return 0;
}