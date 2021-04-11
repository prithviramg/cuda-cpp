#ifndef __MISCELLANEOUS_H__
#define __MISCELLANEOUS_H__

#include <cuda.h>
#include <assert.h>
#include <stdlib.h>
#include <algorithm>

#define RESULT_VERIFICATION 1 //change this to 1 incase want to test the values

void populateFilterData(float* host_filter, int filterSize);
void populateImageData(float* host_inputMatrix, int num_row, int num_col);

#if (RESULT_VERIFICATION)
bool value_test(float* host_outputMatrixCPU, float* host_outputMatrixGPU, int length);
#endif

#endif