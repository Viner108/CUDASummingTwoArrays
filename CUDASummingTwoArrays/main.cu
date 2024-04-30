#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel.h"

int main(int argc, char* argv[])
{
	//variables to compare function execution time on GPU and CPU
	float timerValueGPU, timerValueCPU;
	//creating events for the start and end of time reporting
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float *hA;
	float *dA;
	float *hB;
	float *dB;
	float *hC;
	float * dC;

	// amount of data
	int size = 512 * 50000;
	// number of threads
	int  N_thread = 512;
	// number of blocks
	int N_blocks;
	int i;
	//amount of allocated memory
	unsigned int mem_size = sizeof(float) * size;

	hA = (float*)malloc(mem_size);
	hB = (float*)malloc(mem_size);
	hC = (float*)malloc(mem_size);
	//hA, hB, hC are in the CPU RAM

	cudaError_t err;

	err = cudaMalloc((void**)&dA, mem_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot allocate GPU memory: %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc((void**)&dB, mem_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot allocate GPU memory: %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMalloc((void**)&dC, mem_size);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot allocate GPU memory: %s\n", cudaGetErrorString(err));
		return 1;
	}
	//dA, dB, dC in GPU global memory

	for ( i = 0; i < size; i++)
	{
		hA[i] = 1.0f / ((i + 1.0F) * (i + 1.0f));
		hB[i] = expf(1.0f / (1 + 1.0f));
		hC[i] = 0.0f;
	}

	if ((size % N_thread) == 0) {
		N_blocks = size / N_thread;
	}
	else {
		N_blocks = (int)(size / N_thread) + 1;
	}
	dim3 blocks(N_blocks);

	//time reference point
	cudaEventRecord(start, 0);
	
	err = cudaMemcpy(dA, hA, mem_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot copy data host/device : %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMemcpy(dB, hB, mem_size, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot copy data host/device : %s\n", cudaGetErrorString(err));
		return 1;
	}
	//hA, hB are copied to dA, dB from RAM to GPU global memory

	function << < N_blocks,N_thread >> > (dA,dB,dC,size);
	
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot launch CUDA kernel: %s\n", cudaGetErrorString(err));
		return 1;
	}

	err = cudaMemcpy(hC, dC, mem_size, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {
		fprintf(stderr, "Cannot copy data device/host : %s\n", cudaGetErrorString(err));
		return 1;
	}
	//dC is copied to hC from GPU global memory to RAM
	 
	//end of timing
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// How many milliseconds does the calculation take on the GPU
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time: %f ms\n", timerValueGPU);
	//time reference point
	cudaEventRecord(start, 0);

	for ( i = 0; i < size; i++)
	{
		hC[i] = hA[i] + hB[i];
	}
	// hA, hB, hC are in the CPU RAM
	
	//end of timing
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// How many milliseconds does the calculation take on the CPU
	cudaEventElapsedTime(&timerValueCPU, start, stop);
	printf("\n CPU calculation time: %f ms\n", timerValueCPU);

	// How many times faster does the GPU process data than the CPU
	printf("\n Rate: %f x\n", timerValueCPU / timerValueGPU);

	free(hA);
	free(hB);
	free(hC);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	return 0;
}
