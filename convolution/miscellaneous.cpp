#include "convolution_kernels.h"
#include "miscellaneous.h"

/* Generates Bi-symetric Gaussian Filter */
void populateFilterData(float* host_filter, int filterSize)
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

void populateImageData(float* host_inputMatrix, int num_row, int num_col)
{
    for (int eachFilterRow = 0; eachFilterRow < num_row; eachFilterRow++) {
        for (int eachFilterColumn = 0; eachFilterColumn < num_col; eachFilterColumn++) {
            // host_inputMatrix[eachFilterRow * numberOfColumns + eachFilterColumn] = float(rand() & 0xFFFFFF) / RAND_MAX;
            host_inputMatrix[eachFilterRow * num_col + eachFilterColumn] = 1.f;
        }
    }
}

#if (RESULT_VERIFICATION)
bool value_test(float* host_outputMatrixCPU, float* host_outputMatrixGPU, int length)
{
    float errorMargin = 0.01;
    for (int iterator = 0; iterator < length; iterator++)
        if (abs(host_outputMatrixCPU[iterator] - host_outputMatrixGPU[iterator]) >= errorMargin)
            return false;
    return true;
}
#endif