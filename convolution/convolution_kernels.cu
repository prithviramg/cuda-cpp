#include "convolution_kernels.h"
#include "miscellaneous.h"

void copyFilterValues(float* host_filter, int filterSize) {
    checkCudaErrors(cudaMemcpyToSymbol(c_filter, host_filter, filterSize * filterSize * sizeof(float)));
}

__global__ void convolution_kernel_v1(float* device_outputMatrix, float* device_inputMatrix, float* device_filter,
    int imageRows, int imageColumns, int filterSize)
{
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;

    float convolvedValue = 0.f;
    for (int eachFilterRow = -filterSize / 2; eachFilterRow <= filterSize / 2; ++eachFilterRow)
    {
        for (int eachFilterColumn = -filterSize / 2; eachFilterColumn <= filterSize / 2; ++eachFilterColumn)
        {
            // Find the global position to apply the given filter
            int imageRow = index_y + eachFilterRow;
            int imageColumn = index_x + eachFilterColumn;

            float pixelValue = (imageRow >= 0 && imageRow < imageRows&& imageColumn >= 0 && imageColumn < imageColumns) ?
                device_inputMatrix[imageRow * imageColumns + imageColumn] : 0.f;
            float filterValue = device_filter[(eachFilterRow + filterSize / 2) * filterSize + eachFilterColumn + filterSize / 2];

            convolvedValue += pixelValue * filterValue;
        }
    }

    device_outputMatrix[index_y * imageColumns + index_x] = convolvedValue;
}

__global__ void convolution_kernel_v2(float* device_outputMatrix, float* device_inputMatrix, float* device_filter,
    int imageRows, int imageColumns, int filterSize)
{
    int index_x = blockDim.x * blockIdx.x + threadIdx.x;
    int index_y = blockDim.y * blockIdx.y + threadIdx.y;

    float convolvedValue = 0.f;
    for (int eachFilterRow = -filterSize / 2; eachFilterRow <= filterSize / 2; ++eachFilterRow)
    {
        for (int eachFilterColumn = -filterSize / 2; eachFilterColumn <= filterSize / 2; ++eachFilterColumn)
        {
            int imageRow = index_y + eachFilterRow;
            int imageColumn = index_x + eachFilterColumn;

            float pixelValue = (imageRow >= 0 && imageRow < imageRows&& imageColumn >= 0 && imageColumn < imageColumns) ?
                device_inputMatrix[imageRow * imageColumns + imageColumn] : 0.f;
            float filterValue = c_filter[(eachFilterRow + filterSize / 2) * filterSize + eachFilterColumn + filterSize / 2];

            convolvedValue += pixelValue * filterValue;
        }
    }

    device_outputMatrix[index_y * imageColumns + index_x] = convolvedValue;
}

__global__ void convolution_kernel_v3(float* device_outputMatrix, float* device_inputMatrix, float* device_filter,
    int imageRows, int imageColumns, int filterSize)
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
                (idx_row >= 0 && idx_row < imageRows&& idx_col >= 0 && idx_col < imageColumns) ?
                device_inputMatrix[imageColumns * idx_row + idx_col] : 0.f;
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
    for (int eachFilterRow = -filterSize / 2; eachFilterRow <= filterSize / 2; ++eachFilterRow)
    {
        for (int eachColumnFilter = -filterSize / 2; eachColumnFilter <= filterSize / 2; ++eachColumnFilter)
        {
            // Find the global position to apply the given filter            
            int imageRow = threadIdx.y + paddingSize + eachFilterRow;
            int imageColumn = threadIdx.x + paddingSize + eachColumnFilter;

            float pixelValue = s_input[tileSize * imageRow + imageColumn];
            float filterValue = c_filter[(eachFilterRow + filterSize / 2) * filterSize + eachColumnFilter + filterSize / 2];

            convolvedValue += pixelValue * filterValue;
        }
    }

    device_outputMatrix[index_y * imageColumns + index_x] = convolvedValue;
}

void convolutionGPU(int version, float* device_outputMatrix, float* device_inputMatrix, float* device_filter,
    int imageRows, int imageColumns, int filterSize)
{
    dim3 blockDimension(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDimension((imageColumns + BLOCK_DIM - 1) / BLOCK_DIM, (imageRows + BLOCK_DIM - 1) / BLOCK_DIM);
    if (version == 1)
        convolution_kernel_v1 << <gridDimension, blockDimension >> > (device_outputMatrix, device_inputMatrix, device_filter, imageRows, imageColumns, filterSize);
    else if (version == 2)
        convolution_kernel_v2 << <gridDimension, blockDimension >> > (device_outputMatrix, device_inputMatrix, device_filter, imageRows, imageColumns, filterSize);
    else // version == 3
    {
        int sharedMemorySize = (2 * filterSize + BLOCK_DIM) * (2 * filterSize + BLOCK_DIM) * sizeof(float);
        convolution_kernel_v3 << <gridDimension, blockDimension, sharedMemorySize, 0 >> > (device_outputMatrix, device_inputMatrix, device_filter, imageRows, imageColumns, filterSize);
    }

    checkCudaErrors(cudaGetLastError());
}

void convolutionCPU(float* host_outputMatrix, float* host_inputMatrix, float* host_filter,
    int imageRows, int imageColumns, int filterSize)
{
    //For every pixel in the image
#pragma omp parallel 
    for (int eachRowOfImage = 0; eachRowOfImage < (int)imageRows; ++eachRowOfImage)
    {
        for (int eachColumnOfImage = 0; eachColumnOfImage < (int)imageColumns; ++eachColumnOfImage)
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

                    float pixelValue = (imageRow >= 0 && imageRow < imageRows&& imageColumn >= 0 && imageColumn < imageColumns) ?
                        host_inputMatrix[imageRow * imageColumns + imageColumn] : 0.f;
                    float filterValue = host_filter[(eachRowOfFilter + filterSize / 2) * filterSize + eachColumnOfFilter + filterSize / 2];

                    convolvedValue += pixelValue * filterValue;
                }
            }

            host_outputMatrix[eachRowOfImage * imageColumns + eachColumnOfImage] = convolvedValue;
        }
    }
}