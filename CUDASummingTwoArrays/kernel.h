#include "cuda_runtime.h"
#ifndef KERNEL_H
#define KERNEL_H

//Declaring a function for the GPU kernel
__global__ void function(float* dA, float* dB, float* dC, int size);

#endif // KERNEL_H