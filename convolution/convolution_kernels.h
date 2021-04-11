#ifndef __CONVOLUTION__H
#define __CONVOLUTION__H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <helper_timer.h>
#include "helper_cuda.h"
#include <device_launch_parameters.h>
#include "miscellaneous.h"

#define BLOCK_DIM   16
#define MAX_FILTER_LENGTH 128

__constant__ float c_filter[MAX_FILTER_LENGTH * MAX_FILTER_LENGTH];

void copyFilterValues(float* host_filter, int filterSize); 

__global__ void convolution_kernel_v1(float* device_outputMatrix, float* device_inputMatrix, float* device_filter,
    int imageRows, int imageColumns, int filterSize);

__global__ void convolution_kernel_v2(float* device_outputMatrix, float* device_inputMatrix, float* device_filter,
    int imageRows, int imageColumns, int filterSize);

__global__ void convolution_kernel_v3(float* device_outputMatrix, float* device_inputMatrix, float* device_filter,
    int imageRows, int imageColumns, int filterSize);

void convolutionGPU(int version, float* device_outputMatrix, float* device_inputMatrix, float* device_filter,
    int imageRows, int imageColumns, int filterSize);

void convolutionCPU(float* host_outputMatrix, float* host_inputMatrix, float* host_filter,
    int imageRows, int imageColumns, int filterSize);

#endif
