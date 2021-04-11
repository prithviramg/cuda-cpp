#include <stdio.h>
#include <helper_timer.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

#define RESULT_VERIFICATION 0   // change 1 if you want to verify the result
#define BLOCK_DIM 8   

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on GPU
//! host_inputArray3 = alpha * host_inputArray1 * host_inputArray2 + beta * host_inputArray3
//! @param host_inputArray1          matrix host_inputArray1 as provided to device (M x K)
//! @param host_inputArray2          matrix host_inputArray2 as provided to device (K x N)
//! @param host_inputArray3          matrix host_inputArray3 as provided to device (M x N)
//! @param N          height of matrix host_inputArray1 and matrix host_inputArray3
//! @param M          width of matrix host_inputArray2 and matrix host_inputArray3
//! @param K          width of matrix host_inputArray1 and height of matrix host_inputArray3
//! @param alpha      scala value for matrix multiplication
//! @param beta       scala value for matrix summation with host_inputArray3
////////////////////////////////////////////////////////////////////////////////
__global__ void sgemm_kernelGPU(const float *host_inputArray1, const float *host_inputArray2, float *host_inputArray3, 
                                int M, int N, int K, float alpha, float beta)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float element_c = 0.f;
    for (int eachElement = 0; eachElement < K; eachElement++)
        element_c += host_inputArray1[row * K + eachElement] * host_inputArray2[eachElement * K + column];

    host_inputArray3[row * N + column] = alpha * element_c + beta * host_inputArray3[row * N + column];
}

__global__ void sgemm_kernelGPU_tiling(const float *host_inputArray1, const float *host_inputArray2, float *host_inputArray3, 
                                       int M, int N, int K, float alpha, float beta)
{
    int blockIndex_x = blockIdx.x * blockDim.x;
    int blockIndex_y = blockIdx.y * blockDim.y;
    int threadIndex_x = threadIdx.x;
    int threadIndex_y = threadIdx.y;

    float element_c = 0.f;
    __shared__ float shared_tileInputArray1[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_tileInputArray2[BLOCK_DIM][BLOCK_DIM];

    // forward tile with tile size in matrix host_inputArray1
    for (int eachTileElement = 0; eachTileElement < K; eachTileElement += BLOCK_DIM)
    {
        shared_tileInputArray1[threadIndex_y][threadIndex_x] = \
            host_inputArray1[ (blockIndex_y + threadIndex_y) * K + threadIndex_x + eachTileElement ]; // Get sub-matrix from host_inputArray1
        shared_tileInputArray2[threadIndex_y][threadIndex_x] = \
            host_inputArray2[ (eachTileElement*BLOCK_DIM + threadIndex_y) * N + blockIndex_x + threadIndex_x ]; // Get sub-matrix from host_inputArray2

        __syncthreads();

        // compute gemm operation with tiles
        for (int eachElement = 0; eachElement < BLOCK_DIM; eachElement++)
            element_c += shared_tileInputArray1[threadIndex_y][eachElement] * shared_tileInputArray2[eachElement][threadIndex_x];
	    
	__syncthreads();
    }

    host_inputArray3[(blockIndex_y + threadIndex_y) * N + (blockIndex_x + threadIndex_x)] = \
        alpha * element_c + beta * host_inputArray3[(blockIndex_y + threadIndex_y) * N + (blockIndex_x + threadIndex_x)];
}

void sgemm_kernelCPU(const float *host_inputArray1, const float *host_inputArray2, float *host_inputArray3, 
                     int M, int N, int K, float alpha, float beta)
{
    for (int row = 0; row < M; row++) {
        for (int column = 0; column < N; column++) {
	    float element_c = 0.f;
            for (int e = 0; e < K; e++) {
                element_c += host_inputArray1[row * K + e] * host_inputArray2[e * N + column];
	        }
            host_inputArray3[row * N + column] = alpha * element_c + beta * host_inputArray3[row * N + column];
        }
    }
}

void randomInitialise(float *arrayData, int length)
{
    for (int iterator = 0; iterator < length; iterator++) {
        arrayData[iterator] = (rand() & 0xFFFF) / (float)RAND_MAX;
    }
}

bool checkErrors(float *arrayFromCPU, float *arrayFromGPU, int length)
{
    float errorMargin = 0.1;
    for (int iterator = 0; iterator < length; iterator++)
        if (abs(arrayFromCPU[iterator] - arrayFromGPU[iterator]) >= errorMargin)
            return false;
    return true;
}

int main(int c, char *argv[])
{

    float* host_inputArray1, * host_inputArray2, * host_inputArray3, * host_copyArray;
    float* device_inputArray1, * device_inputArray2, * device_inputArray3;
    int M, N, K;
    float alpha = 2.f;
    float beta = 1.f;
    printf("enter values of M, K, N: ");
    scanf("%d%d%d", &M, &K, &N);
    printf("values of M = %d, K = %d, N = %d\n", M, K, N);

    // initialize timer
    StopWatchInterface* timer;
    sdkCreateTimer(&timer);

    // allocation of linear memory space
    host_inputArray1 = (float*)malloc(M * K * sizeof(float));
    host_inputArray2 = (float*)malloc(K * N * sizeof(float));
    host_inputArray3 = (float*)malloc(M * N * sizeof(float));
    host_copyArray = (float*)malloc(M * N * sizeof(float));


    // allocation of gpu linear memory space
    checkCudaErrors(cudaMalloc((void**)&device_inputArray1, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&device_inputArray2, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&device_inputArray3, M * N * sizeof(float)));

    // initialize randomized values for memory space
    randomInitialise(host_inputArray1, M * K);
    randomInitialise(host_inputArray2, K * N);

    // profiler will focus from this point
    sdkStartTimer(&timer);

    // copy initial value for gpu memory
    checkCudaErrors(cudaMemcpy(device_inputArray1, host_inputArray1, M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_inputArray2, host_inputArray1, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // do operation
    dim3 blockDimension(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDimension((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
    printf("block Dimensions : %d , %d\n", BLOCK_DIM, BLOCK_DIM);
    printf("grid Dimensions : %d , %d\n", (N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
    cudaProfilerStart();

    sgemm_kernelGPU << <gridDimension, blockDimension >> > (device_inputArray1, device_inputArray2, device_inputArray3, M, N, K, alpha, beta);
    sgemm_kernelGPU_tiling << <gridDimension, blockDimension >> > (device_inputArray1, device_inputArray2, device_inputArray3, M, N, K, alpha, beta);

    // profiler will stop its focus
    cudaProfilerStop();

    // measuring the performance
    cudaDeviceSynchronize();
    sdkStopTimer(&timer); // this profiler should be behined of device synchronization
    printf("completed GPU computation!!!\n");
    printf("Time= %.3f msec\n", sdkGetTimerValue(&timer));

#if (RESULT_VERIFICATION)
    // copy data from the gpu
    checkCudaErrors(cudaMemcpy(host_copyArray, device_inputArray3, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // compare the result
    sgemm_kernelCPU(host_inputArray1, host_inputArray2, host_inputArray3, M, N, K, alpha, beta);

    if (checkErrors(host_inputArray3, host_copyArray, M * N))
        printf("SUCCESS!!\n");
    else
        printf("Error\n");
#endif

    // terminates allocated gpu memory space
    cudaFree(device_inputArray1);
    cudaFree(device_inputArray2);
    cudaFree(device_inputArray3);

    // terminates allocated memory space
    free(host_inputArray1);
    free(host_inputArray2);
    free(host_inputArray3);
    free(host_copyArray);

    return 0;
}