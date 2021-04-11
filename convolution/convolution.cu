#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <helper_timer.h>
#include "helper_cuda.h"
#include <assert.h>
#include <device_launch_parameters.h>

#define BLOCK_DIM   16
#define MAX_FILTER_LENGTH 128
#define RESULT_VERIFICATION 1   // change 1 if you want to verify the convolvedValue

__global__ void
convolution_kernel_v1(float *device_outputMatrix, float *device_inputMatrix, float *device_filter, 
                      int numberOfRows, int numberOfColumns, int filterSize)
{
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;

    float convolvedValue = 0.f;
    for (int eachRowOfFilter = -filterSize / 2; eachRowOfFilter <= filterSize / 2; ++eachRowOfFilter)
    {
        for (int eachColumnOfFilter = -filterSize / 2; eachColumnOfFilter <= filterSize / 2; ++eachColumnOfFilter)
        {
            // Find the global position to apply the given filter
            int imageRow = index_y + eachRowOfFilter;
            int imageColumn = index_x + eachColumnOfFilter;

            float pixelValue = (imageRow >= 0 && imageRow < numberOfRows && imageColumn >= 0 && imageColumn < numberOfColumns) ?
                                            device_inputMatrix[imageRow * numberOfColumns + imageColumn] : 0.f;
            float filterValue = device_filter[(eachRowOfFilter + filterSize / 2) * filterSize + eachColumnOfFilter + filterSize / 2];

            convolvedValue += pixelValue * filterValue;
        }
    }

    device_outputMatrix[index_y * numberOfColumns + index_x] = convolvedValue;
}

__constant__ float c_filter[MAX_FILTER_LENGTH * MAX_FILTER_LENGTH];

__global__ void convolution_kernel_v2(float *device_outputMatrix, float *device_inputMatrix, float *device_filter, 
                                      int numberOfRows, int numberOfColumns, int filterSize)
{
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;

    float convolvedValue = 0.f;
    for (int eachRowOfFilter = -filterSize / 2; eachRowOfFilter <= filterSize / 2; ++eachRowOfFilter)
    {
        for (int eachColumnOfFilter = -filterSize / 2; eachColumnOfFilter <= filterSize / 2; ++eachColumnOfFilter)
        {
            int imageRow = index_y + eachRowOfFilter;
            int imageColumn = index_x + eachColumnOfFilter;

            float pixelValue = (imageRow >= 0 && imageRow < numberOfRows && imageColumn >= 0 && imageColumn < numberOfColumns) ?
                                            device_inputMatrix[imageRow * numberOfColumns + imageColumn] : 0.f;
            float filterValue = c_filter[(eachRowOfFilter + filterSize / 2) * filterSize + eachColumnOfFilter + filterSize / 2];

            convolvedValue += pixelValue * filterValue;
        }
    }

    device_outputMatrix[index_y * numberOfColumns + index_x] = convolvedValue;
}

__global__ void convolution_kernel_v3(float *device_outputMatrix, float *device_inputMatrix, float *device_filter, 
                      int numberOfRows, int numberOfColumns, int filterSize)
{
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    int paddingSize = filterSize / 2;
    int tileSize = BLOCK_DIM + 2 * paddingSize;

    extern __shared__ float s_input[];

    for (int row = 0; row <= tileSize / BLOCK_DIM; row++)
    {
        for (int col = 0; col <= tileSize / BLOCK_DIM; col++)
        {
            int idx_row = index_y + BLOCK_DIM * row - paddingSize;   // input data index row
            int idx_col = index_x + BLOCK_DIM * col - paddingSize;   // input data index column
            int fid_row = threadIdx.y + BLOCK_DIM * row; // filter index row
            int fid_col = threadIdx.x + BLOCK_DIM * col; // filter index column
            
            if (fid_row >= tileSize || fid_col >= tileSize)   continue;

            s_input[tileSize * fid_row + fid_col] = \
                (idx_row >= 0 && idx_row < numberOfRows && idx_col >= 0 && idx_col < numberOfColumns) ? 
                    device_inputMatrix[numberOfColumns * idx_row + idx_col] : 0.f;
        }
    }

    __syncthreads();

    /* Tile Debugging */
    // if (index_x == BLOCK_DIM*1 && index_y == BLOCK_DIM*1) 
    // {
    //     for (int row = 0; row < 2*paddingSize + BLOCK_DIM; row++)
    //     {
    //         for (int col = 0; col < 2*paddingSize + BLOCK_DIM; col++)
    //         {
    //             printf("%.0f ", s_input[tileSize * row + col]);
    //         }
    //         printf("\n");
    //     }
    // }

    float convolvedValue = 0.f;
    for (int eachRowOfFilter = -filterSize / 2; eachRowOfFilter <= filterSize / 2; ++eachRowOfFilter)
    {
        for (int eachColumnOfFilter = -filterSize / 2; eachColumnOfFilter <= filterSize / 2; ++eachColumnOfFilter)
        {
            // Find the global position to apply the given filter            
            int imageRow = threadIdx.y + paddingSize + eachRowOfFilter;
            int imageColumn = threadIdx.x + paddingSize + eachColumnOfFilter;

            float pixelValue  = s_input[tileSize * imageRow + imageColumn];            
            float filterValue = c_filter[(eachRowOfFilter + filterSize / 2) * filterSize + eachColumnOfFilter + filterSize / 2];

            convolvedValue += pixelValue * filterValue;
        }
    }

    device_outputMatrix[index_y * numberOfColumns + index_x] = convolvedValue;
}

void convolution_gpu(int version, float *device_outputMatrix, float *device_inputMatrix, float *device_filter, 
                     int numberOfRows, int numberOfColumns, int filterSize)
{
    dim3 blockDimension(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDimension((numberOfColumns + BLOCK_DIM - 1) / BLOCK_DIM, (numberOfRows + BLOCK_DIM - 1) / BLOCK_DIM);
    if (version == 1)
        convolution_kernel_v1<<<gridDimension, blockDimension>>>(device_outputMatrix, device_inputMatrix, device_filter, numberOfRows, numberOfColumns, filterSize);
    else if (version == 2) 
        convolution_kernel_v2<<<gridDimension, blockDimension>>>(device_outputMatrix, device_inputMatrix, device_filter, numberOfRows, numberOfColumns, filterSize);
    else // version == 3
    {
        int sharedMemorySize = (2*filterSize+BLOCK_DIM) * (2*filterSize+BLOCK_DIM) * sizeof(float);
        convolution_kernel_v3<<<gridDimension, blockDimension, sharedMemorySize, 0 >>>(device_outputMatrix, device_inputMatrix, device_filter, numberOfRows, numberOfColumns, filterSize);
    }
    
    checkCudaErrors(cudaGetLastError());
}

void convolution_host(float *host_outputMatrix, float *host_inputMatrix, float *host_filter, 
                      int numberOfRows, int numberOfColumns, int filterSize)
{
    //For every pixel in the image
    #pragma omp parallel 
    for (int eachRowOfImage = 0; eachRowOfImage < (int)numberOfRows; ++eachRowOfImage)
    {
        for (int eachColumnOfImage = 0; eachColumnOfImage < (int)numberOfColumns; ++eachColumnOfImage)
        {
            float convolvedValue = 0.f;
            //For every value in the filter around the pixel (c, r)
            for (int eachRowOfFilter = -filterSize / 2; eachRowOfFilter <= filterSize / 2; ++eachRowOfFilter)
            {
                for (int eachColumnOfFilter = -filterSize / 2; eachColumnOfFilter <= filterSize / 2; ++eachColumnOfFilter)
                {
                    // Find the global image position for this filter position
                    int imageRow = eachRowOfImage + eachRowOfFilter;
                    int imageColumn = eachColumnOfImage + eachColumnOfFilter;

                    float pixelValue = (imageRow >= 0 && imageRow < numberOfRows && imageColumn >= 0 && imageColumn < numberOfColumns) ?
                                            host_inputMatrix[imageRow * numberOfColumns + imageColumn] : 0.f;
                    float filterValue = host_filter[(eachRowOfFilter + filterSize / 2) * filterSize + eachColumnOfFilter + filterSize / 2];

                    convolvedValue += pixelValue * filterValue;
                }
            }

            host_outputMatrix[eachRowOfImage * numberOfColumns + eachColumnOfImage] = convolvedValue;
        }
    }
}


/* Generates Bi-symetric Gaussian Filter */
void generate_filter(float *host_filter, int filterSize)
{
    float blurFilterSigma = 2.;

    float filterCummulativeSum = 0.f; //for normalization
    for (int eachFilterRow = -filterSize / 2; eachFilterRow <= filterSize / 2; eachFilterRow++)
    {
        for (int eachFilterColumn = -filterSize / 2; eachFilterColumn <= filterSize / 2; eachFilterColumn++)
        {
            float filterValue = expf(-(float)(eachFilterColumn * eachFilterColumn + eachFilterRow * eachFilterRow) / (2.f * blurFilterSigma * blurFilterSigma));
            host_filter[(eachFilterRow + filterSize / 2) * filterSize + eachFilterColumn + filterSize / 2] = filterValue;
            filterCummulativeSum += filterValue;
        }
    }

    // normalization
    float normalizationFactor = 1.f / filterCummulativeSum;
    for (int eachFilterRow = -filterSize / 2; eachFilterRow <= filterSize / 2; eachFilterRow++)
        for (int eachFilterColumn = -filterSize / 2; eachFilterColumn <= filterSize / 2; eachFilterColumn++)
            host_filter[(eachFilterRow + filterSize / 2) * filterSize + eachFilterColumn + filterSize / 2] *= normalizationFactor;
}

void generate_data(float *host_inputMatrix, int num_row, int num_col)
{
    for (int eachFilterRow = 0; eachFilterRow < num_row; eachFilterRow++) {
        for (int eachFilterColumn = 0; eachFilterColumn < num_col; eachFilterColumn++) {
            // host_inputMatrix[eachFilterRow * numberOfColumns + eachFilterColumn] = float(rand() & 0xFFFFFF) / RAND_MAX;
            host_inputMatrix[eachFilterRow * num_col + eachFilterColumn] = 1.f;
        }
    }
}

bool value_test(float *host_outputMatrixCPU, float *host_outputMatrixGPU, int length)
{
    float errorMargin = 0.000001;
    for (int i = 0; i < length; i++)
        if (abs(host_outputMatrixCPU[i] - host_outputMatrixGPU[i]) >= errorMargin)
            return false;
    return true;
}

int main()
{
    int numberOfRows = 2048;
    int numberOfColumns = 2048;
    int filterSize = 9;
    int bufferSize = numberOfRows * numberOfColumns * sizeof(float);

    float *host_inputMatrix, *device_inputMatrix;
    float *host_outputMatrix, *host_outputMatrixGPU, *device_outputMatrix;
    float *host_filter, *device_filter;

    float elapsedTime;

    // initialize timer
    StopWatchInterface *host_timer, *device_timer;
    sdkCreateTimer(&host_timer);
    sdkCreateTimer(&device_timer);

    srand(2019);

    // allocate host memories
    host_inputMatrix = (float *)malloc(bufferSize);
    host_outputMatrix = (float *)malloc(bufferSize);
    host_outputMatrixGPU = (float *)malloc(bufferSize);
    host_filter = (float *)malloc(filterSize * filterSize * sizeof(float));

    // allocate gpu memories
    checkCudaErrors(cudaMalloc((void **)&device_inputMatrix, bufferSize));
    checkCudaErrors(cudaMalloc((void **)&device_outputMatrix, bufferSize));
    checkCudaErrors(cudaMalloc((void **)&device_filter, filterSize * filterSize * sizeof(float)));

    // generate data
    generate_data(host_inputMatrix, numberOfRows, numberOfColumns);
    generate_filter(host_filter, filterSize);

    // copy input date to gpu
    checkCudaErrors(cudaMemcpy(device_inputMatrix, host_inputMatrix, bufferSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_filter, host_filter, filterSize * filterSize * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(c_filter, host_filter, filterSize * filterSize * sizeof(float)));

    // processing in GPU
    sdkStartTimer(&device_timer);
    cudaProfilerStart();
    convolution_gpu(1, device_outputMatrix, device_inputMatrix, device_filter, numberOfRows, numberOfColumns, filterSize);
    cudaDeviceSynchronize();
    sdkStopTimer(&device_timer);
    elapsedTime = sdkGetTimerValue(&device_timer);
    printf("Processing Time (1) -> GPU: %.2f ms\n", elapsedTime);

    // processing in GPU
    sdkResetTimer(&device_timer);
    sdkStartTimer(&device_timer);
    convolution_gpu(2, device_outputMatrix, device_inputMatrix, device_filter, numberOfRows, numberOfColumns, filterSize);
    cudaDeviceSynchronize();
    sdkStopTimer(&device_timer);
    elapsedTime = sdkGetTimerValue(&device_timer);
    printf("Processing Time (2) -> GPU: %.2f ms\n", elapsedTime);

    // processing in GPU (3)
    sdkResetTimer(&device_timer);
    sdkStartTimer(&device_timer);
    convolution_gpu(3, device_outputMatrix, device_inputMatrix, device_filter, numberOfRows, numberOfColumns, filterSize);
    cudaDeviceSynchronize();
    sdkStopTimer(&device_timer);
    cudaProfilerStop();
    elapsedTime = sdkGetTimerValue(&device_timer);
    printf("Processing Time (3) -> GPU: %.2f ms\n", elapsedTime);

#if (RESULT_VERIFICATION)
    // processing in CPU
    sdkStartTimer(&host_timer);
    convolution_host(host_outputMatrix, host_inputMatrix, host_filter, numberOfRows, numberOfColumns, filterSize);
    sdkStopTimer(&host_timer);

    float elapsed_time_host = sdkGetTimerValue(&host_timer);
    printf("Processing Time -> Host: %.2f ms\n", elapsed_time_host);

    // compare the convolvedValue
    checkCudaErrors(cudaMemcpy(host_outputMatrixGPU, device_outputMatrix, bufferSize, cudaMemcpyDeviceToHost));
    if (value_test(host_outputMatrix, host_outputMatrixGPU, numberOfRows * numberOfColumns))
        printf("SUCCESS!!\n");
    else
        printf("Error\n");
#endif

    // finalize
    free(host_inputMatrix);
    free(host_outputMatrix);
    free(host_outputMatrixGPU);
    free(host_filter);

    return 0;
}

