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


#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_


#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

using namespace std;

#define SAMPLE_VERSION "HIP-Examples-Applications-v1.0"
#define WIDTH 1024
#define HEIGHT 1024
#define BIN_SIZE 256
#define GROUP_SIZE 128
#define GROUP_ITERATIONS (BIN_SIZE / 2)//This is done to avoid overflow in the kernel
#define SUB_HISTOGRAM_COUNT ((WIDTH * HEIGHT) /(GROUP_SIZE * GROUP_ITERATIONS))


#ifndef __global__
#define __global__
#endif

#ifndef HIP_DYNAMIC_SHARED
#define HIP_DYNAMIC_SHARED(t,x) t* x
#endif

#ifndef hipThreadIdx_x
#define hipThreadIdx_x 0
#endif

#ifndef hipThreadIdx_y
#define hipThreadIdx_y 0
#endif

#ifndef hipThreadIdx_z
#define hipThreadIdx_z 0
#endif

#ifndef hipBlockIdx_x
#define hipBlockIdx_x 0
#endif

#ifndef hipBlockDim_x
#define hipBlockDim_x 0
#endif

#ifndef __syncthreads
#define __syncthreads()
#endif

/**
* Histogram
* Class implements 256 Histogram bin implementation

*/

class Histogram
{

        int binSize;             /**< Size of Histogram bin */
        int groupSize;           /**< Number of threads in group */
        int subHistgCnt;         /**< Sub histogram count */
        unsigned int *data;              /**< input data initialized with normalized(0 - binSize) random values */
        int width;               /**< width of the input */
        int height;              /**< height of the input */
        unsigned int *hostBin;           /**< Host result for histogram bin */
        unsigned int *midDeviceBin;      /**< Intermittent sub-histogram bins */
        unsigned int *deviceBin;         /**< Device result for histogram bin */

        unsigned long totalLocalMemory;      /**< Max local memory allowed */
        unsigned long usedLocalMemory;       /**< Used local memory by kernel */

        unsigned int* dataBuf;                 /**< CL memory buffer for data */
        unsigned int* midDeviceBinBuf;         /**< CL memory buffer for intermittent device bin */

        int iterations;                     /**< Number of iterations for kernel execution */
        unsigned int globalThreads;
        unsigned int localThreads ;
        int groupIterations;

    public:

        /**
        * Constructor
        * Initialize member variables
        * @param name name of sample (string)
        */
        Histogram()
            :
            binSize(BIN_SIZE),
            groupSize(GROUP_SIZE),
            subHistgCnt(SUB_HISTOGRAM_COUNT),
            groupIterations(GROUP_ITERATIONS),
            data(NULL),
            hostBin(NULL),
            midDeviceBin(NULL),
            deviceBin(NULL),
            iterations(1)
        {
            /* Set default values for width and height */
            width = WIDTH;
            height = HEIGHT;
        }


        ~Histogram()
        {
        }

        /**
        * Allocate and initialize required host memory with appropriate values
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupHistogram();

        /**
        * HIP related initialisations.
        * Set up Context, Device list, Command Queue, Memory buffers
        * Build HIP kernel program executable
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setupHIP();

        /**
        * Set values for kernels' arguments, enqueue calls to the kernels
        * on to the command queue, wait till end of kernel execution.
        * Get kernel start and end time if timing is enabled
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int runKernels();


        /**
        * Override from SDKSample, adjust width and height
        * of execution domain, perform all sample setup
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int setup();

        /**
        * Override from SDKSample
        * Run HIP Black-Scholes
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int run();

        /**
        * Override from SDKSample
        * Cleanup memory allocations
        * @return SDK_SUCCESS on success and SDK_FAILURE on failure
        */
        int cleanup();

    private:

        /**
        *  Calculate histogram bin on host
        */
        int calculateHostBin();
};
#endif
