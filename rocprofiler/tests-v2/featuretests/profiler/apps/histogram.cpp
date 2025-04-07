/**********************************************************************
Copyright �2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

�   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
�   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include <math.h>
#include "hip/hip_runtime.h"

#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "histogram.hpp"

#define LINEAR_MEM_ACCESS

#define BIN_SIZE 256
#define SDK_SUCCESS 0
#define SDK_FAILURE 1
#define CHECK_ALLOCATION(x, msg) if(!(x)) { std::cout << __FILE__ << ' ' << __LINE__ << ' ' << msg << std::endl; }


/**
 * @brief   Calculates block-histogram bin whose bin size is 256
 * @param   data  input data pointer
 * @param   sharedArray shared array for thread-histogram bins
 * @param   binResult block-histogram array
 */

__global__
void histogram256(
                  unsigned int* data,
                  unsigned int* binResult)
{
    HIP_DYNAMIC_SHARED(unsigned char, sharedArray);
    size_t localId = hipThreadIdx_x;
    size_t globalId = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;
    size_t groupId = hipBlockIdx_x;
    size_t groupSize = hipBlockDim_x;
    int offSet1 = localId & 31;    
    int offSet2 = 4 * offSet1;      //which element to access in one bank.
    int offSet3 = localId >> 5;     //bank number
    /* initialize shared array to zero */
    uchar4 * input = (uchar4*)sharedArray;
    for(int i = 0; i < 64; ++i)
        input[groupSize * i + localId] = make_uchar4(0,0,0,0);

    __syncthreads();


    /* calculate thread-histograms */
	//128 accumulations per thread
	for(int i = 0; i < 128; i++)
    {
#ifdef LINEAR_MEM_ACCESS
       uint value =  data[groupId * (groupSize * (BIN_SIZE/2)) + i * groupSize + localId]; 
#else
       uint  value = data[globalId + i*4096];

#endif // LINEAR_MEM_ACCESS
	   sharedArray[value * 128 + offSet2 + offSet3]++;
    }
    __syncthreads();
    
    /* merge all thread-histograms into block-histogram */

	uint4 binCount;
	uint result;
	uchar4 binVal;	            //Introduced uint4 for summation to avoid overflows
	uint4 binValAsUint;
	for(int i = 0; i < BIN_SIZE / groupSize; ++i)
    {
        int passNumber = BIN_SIZE / 2 * 32 * i +  localId * 32 ;
		binCount = make_uint4(0,0,0,0);
		result= 0;
        for(int j = 0; j < 32; ++j)
		{
			int bankNum = (j + offSet1) & 31;   // this is bank number
            binVal = input[passNumber  +bankNum];

            binValAsUint.x = (unsigned int)binVal.x;
            binValAsUint.y = (unsigned int)binVal.y;
            binValAsUint.z = (unsigned int)binVal.z;
            binValAsUint.w = (unsigned int)binVal.w;

            binCount.x += binValAsUint.x;
            binCount.y += binValAsUint.y;
            binCount.z += binValAsUint.z;
            binCount.w += binValAsUint.w;

		}
        result = binCount.x + binCount.y + binCount.z + binCount.w;
        binResult[groupId * BIN_SIZE + groupSize * i + localId ] = result;
	}
}

int
Histogram::calculateHostBin()
{
    for(int i = 0; i < height; ++i)
    {
        for(int j = 0; j < width; ++j)
        {
            hostBin[data[i * width + j]]++;
        }
    }

    return SDK_SUCCESS;
}


int
Histogram::setupHistogram()
{
    int i = 0;

    data = (unsigned int *)malloc(sizeof(unsigned int) * width * height);

    for(i = 0; i < width * height; i++)
    {
        data[i] = rand() % (unsigned int)(binSize);
    }

    hostBin = (unsigned int*)malloc(binSize * sizeof(unsigned int));
    CHECK_ALLOCATION(hostBin, "Failed to allocate host memory. (hostBin)");

    memset(hostBin, 0, binSize * sizeof(unsigned int));

    deviceBin = (unsigned int*)malloc(binSize * sizeof(unsigned int));
    CHECK_ALLOCATION(deviceBin, "Failed to allocate host memory. (deviceBin)");
    midDeviceBin = (unsigned int*)malloc(sizeof(unsigned int) * binSize * subHistgCnt);

    memset(deviceBin, 0, binSize * sizeof(unsigned int));
    return SDK_SUCCESS;
}

int
Histogram::setupHIP(void)
{
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;

    return SDK_SUCCESS;
}


int
Histogram::runKernels(void)
{
    groupSize = 128;
    globalThreads = (width * height) / (GROUP_ITERATIONS);

    localThreads = groupSize;


    hipHostMalloc((void**)&dataBuf,sizeof(unsigned int) * width * height, hipHostMallocDefault);
    unsigned int *din;
    hipHostGetDevicePointer((void**)&din, dataBuf,0);
    hipMemcpy(din, data,sizeof(unsigned int) * width * height, hipMemcpyHostToDevice);

    subHistgCnt = (width * height) / (groupSize * groupIterations);

    hipHostMalloc((void**)&midDeviceBinBuf,sizeof(unsigned int) * binSize * subHistgCnt, hipHostMallocDefault);

    hipLaunchKernelGGL(histogram256,
                    dim3(globalThreads/localThreads),
                    dim3(localThreads),
                    groupSize * binSize * sizeof(unsigned char), 0,
                    dataBuf ,midDeviceBinBuf);

    hipDeviceSynchronize();

    hipMemcpy(midDeviceBin, midDeviceBinBuf,sizeof(unsigned int) * binSize * subHistgCnt, hipMemcpyDeviceToHost);
        //printArray<unsigned int>("midDeviceBin", midDeviceBin, sizeof(unsigned int) * binSize * subHistgCnt, 1);
    // Clear deviceBin array
    memset(deviceBin, 0, binSize * sizeof(unsigned int));

    // Calculate final histogram bin
    for(int i = 0; i < subHistgCnt; ++i)
    {
        for(int j = 0; j < binSize; ++j)
        {
            deviceBin[j] += midDeviceBin[i * binSize + j];
        }
    }

    return SDK_SUCCESS;
}

int
Histogram::setup()
{
    if(iterations < 1)
    {
        std::cout<<"Error, iterations cannot be 0 or negative. Exiting..\n";
        exit(0);
    }
    int status = 0;

    /* width must be multiples of binSize and
     * height must be multiples of groupSize
     */
    width = (width / binSize ? width / binSize: 1) * binSize;
    height = (height / groupSize ? height / groupSize: 1) * groupSize;

    status = setupHIP();
    if(status != SDK_SUCCESS)
        return status;

    status = setupHistogram();
    if(status != SDK_SUCCESS)
        return status;

    return SDK_SUCCESS;
}


int Histogram::run()
{
    for(int i = 0; i < 2 && iterations != 1; i++)
        if(runKernels() != SDK_SUCCESS)
            return SDK_FAILURE;

    for(int i = 0; i < iterations; i++)
        if(runKernels() != SDK_SUCCESS)
            return SDK_FAILURE;

    return SDK_SUCCESS;
}

int Histogram::cleanup()
{
    hipFree(dataBuf);
    hipFree(midDeviceBinBuf);

    free(hostBin);
    free(deviceBin);

    return SDK_SUCCESS;
}

int
main(int argc, char * argv[])
{
    int status = 0;
    // Create MonteCalroAsian object
    Histogram hipHistogram;

    // Setup
    status = hipHistogram.setup();
    if(status != SDK_SUCCESS)
        return status;

    // Run
    if(hipHistogram.run() != SDK_SUCCESS)
        return SDK_FAILURE;

    // Cleanup resources created
    if(hipHistogram.cleanup() != SDK_SUCCESS)
        return SDK_FAILURE;

    return SDK_SUCCESS;
}
