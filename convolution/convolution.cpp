#include <stdio.h>
#include "convolution_kernels.h"
#include "miscellaneous.h"


int main()
{
    int imageRows = 2048;
    int imageColumns = 2048;
    int filterSize = 9;
    int bufferSize = imageRows * imageColumns * sizeof(float);

    float* host_inputMatrix, * device_inputMatrix;
    float* host_outputMatrix, * host_outputMatrixGPU, * device_outputMatrix;
    float* host_filter, * device_filter;

    float elapsedTime;

    // initialize timer
    StopWatchInterface* host_timer, * device_timer;
    sdkCreateTimer(&host_timer);
    sdkCreateTimer(&device_timer);

    srand(2019);

    // allocate host memories
    host_inputMatrix = (float*)malloc(bufferSize);
    host_outputMatrix = (float*)malloc(bufferSize);
    host_outputMatrixGPU = (float*)malloc(bufferSize);
    host_filter = (float*)malloc(filterSize * filterSize * sizeof(float));

    // allocate gpu memories
    checkCudaErrors(cudaMalloc((void**)&device_inputMatrix, bufferSize));
    checkCudaErrors(cudaMalloc((void**)&device_outputMatrix, bufferSize));
    checkCudaErrors(cudaMalloc((void**)&device_filter, filterSize * filterSize * sizeof(float)));

    // generate data
    populateImageData(host_inputMatrix, imageRows, imageColumns);
    populateFilterData(host_filter, filterSize);

    // copy input date to gpu
    checkCudaErrors(cudaMemcpy(device_inputMatrix, host_inputMatrix, bufferSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_filter, host_filter, filterSize * filterSize * sizeof(float), cudaMemcpyHostToDevice));
    copyFilterValues(host_filter, filterSize);

    // processing in GPU
    sdkStartTimer(&device_timer);
    cudaProfilerStart();
    convolutionGPU(1, device_outputMatrix, device_inputMatrix, device_filter, imageRows, imageColumns, filterSize);
    cudaDeviceSynchronize();
    sdkStopTimer(&device_timer);
    elapsedTime = sdkGetTimerValue(&device_timer);
    printf("Processing Time (1) -> GPU: %.2f ms\n", elapsedTime);

    // processing in GPU
    sdkResetTimer(&device_timer);
    sdkStartTimer(&device_timer);
    convolutionGPU(2, device_outputMatrix, device_inputMatrix, device_filter, imageRows, imageColumns, filterSize);
    cudaDeviceSynchronize();
    sdkStopTimer(&device_timer);
    elapsedTime = sdkGetTimerValue(&device_timer);
    printf("Processing Time (2) -> GPU: %.2f ms\n", elapsedTime);

    // processing in GPU (3)
    sdkResetTimer(&device_timer);
    sdkStartTimer(&device_timer);
    convolutionGPU(3, device_outputMatrix, device_inputMatrix, device_filter, imageRows, imageColumns, filterSize);
    cudaDeviceSynchronize();
    sdkStopTimer(&device_timer);
    cudaProfilerStop();
    elapsedTime = sdkGetTimerValue(&device_timer);
    printf("Processing Time (3) -> GPU: %.2f ms\n", elapsedTime);

#if (RESULT_VERIFICATION)
    // processing in CPU
    sdkStartTimer(&host_timer);
    convolutionCPU(host_outputMatrix, host_inputMatrix, host_filter, imageRows, imageColumns, filterSize);
    sdkStopTimer(&host_timer);

    elapsedTime = sdkGetTimerValue(&host_timer);
    printf("Processing Time -> Host: %.2f ms\n", elapsedTime);

    // compare the convolvedValue
    checkCudaErrors(cudaMemcpy(host_outputMatrixGPU, device_outputMatrix, bufferSize, cudaMemcpyDeviceToHost));
    if (value_test(host_outputMatrix, host_outputMatrixGPU, imageRows * imageColumns))
        printf("SUCCESS!!\n");
    else
        printf("Error\n");
#endif

    // finalize
    free(host_inputMatrix);
    free(host_outputMatrix);
    free(host_outputMatrixGPU);
    free(host_filter);

    cudaFree(device_filter);
    cudaFree(device_inputMatrix);
    cudaFree(device_outputMatrix);

    return 0;
}