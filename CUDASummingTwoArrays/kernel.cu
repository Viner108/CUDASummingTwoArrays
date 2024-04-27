#include "kernel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

__global__ void function(float* dA, float *dB, float *dC,int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	//если бы число нитей было произвольное то чтобы нити последнего блока не смогли обратиться в недопустимой ячейке памяти
	//добавлено условие i < size
	if (i < size) dC[i] = dA[i] + dB[i];
}