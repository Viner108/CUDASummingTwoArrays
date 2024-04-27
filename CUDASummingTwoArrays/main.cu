#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel.h"

int main(int argc, char* argv[])
{
	//переменные для сравнения времени выполнения функции на GPU и на CPU
	float timerValueGPU, timerValueCPU;
	// создание ивентов для начала и конца отчета времени
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float *hA;
	float *dA;
	float *hB;
	float *dB;
	float *hC;
	float *dC;

	// колическо данных
	int size = 512 * 50000;
	// число нитей
	int  N_thread = 512;
	// число блоков
	int N_blocks;
	int i;
	// количество выделяемой памяти
	unsigned int mem_size = sizeof(float) * size;

	hA = (float*)malloc(mem_size);
	hB = (float*)malloc(mem_size);
	hC = (float*)malloc(mem_size);
	// hA, hB, hC лежат в оперативной памяти CPU

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
	//dA, dB, dC в глобальной памяти GPU

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

	//точка отсчета времени
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
	//hA, hB копируются в dA, dB из оперативной памяти в глобальную

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
	//dC копируется в hC с глобальной памяти в оперативную
	 
	//конец отсчета времени
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// сколько в милисекундах занимает расчет на CPU
	cudaEventElapsedTime(&timerValueGPU, start, stop);
	printf("\n GPU calculation time: %f ms\n", timerValueGPU);
	//точка отсчета времени
	cudaEventRecord(start, 0);

	for ( i = 0; i < size; i++)
	{
		hC[i] = hA[i] + hB[i];
	}
	// hA, hB, hC лежат в оперативной памяти CPU
	
	//конец отсчета времени
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// сколько в милисекундах занимает расчет на CPU
	cudaEventElapsedTime(&timerValueCPU, start, stop);
	printf("\n CPU calculation time: %f ms\n", timerValueCPU);

	// во сколько раз быстрее GPU обрабатывает данные чем CPU
	printf("\n Rate: %f x\n", timerValueCPU / timerValueGPU);

	free(hA);
	free(hB);
	free(hC);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	return 0;
}
