/*
* Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

# include <stdio.h>
# include <stdint.h>
# include <hip/hip_runtime.h>

typedef unsigned int ARRAY_TYPE;//array element type
#define K_LEN 1024              //number of first batch accesses to be timed
#define DELTA 0
int CNT = 1 * K_LEN;            //i.e., iterations * K_LEN

__global__ void cache_test_RW (ARRAY_TYPE * my_array, int array_length,
                int iterations, unsigned int *index) {
    unsigned int j = 0;
    int k;

    for (k = 0; k < iterations*K_LEN+DELTA; k++) {
        j = my_array[j];
        index[k] = j;
    }
}

__global__ void cache_test_RO (ARRAY_TYPE * my_array, int array_length,
                int iterations, unsigned int *index) {
    unsigned int j = 0;

    int k;

    for (k = 0; k < iterations*K_LEN+DELTA; k++) {
        j = my_array[j];
        clock();
    }

    index[0] = j;
}

__global__ void cache_test_WO (ARRAY_TYPE * my_array, int array_length,
                int iterations, int stride) {
    uint64_t k;
    uint64_t cnt = (uint64_t)iterations*K_LEN*stride;

    for (k = 0; k < cnt; k+=stride) {
        my_array[k%array_length] = k;
    }
}

void run_test(int N, int iterations, int stride);

int main(int argc, char* argv[]){
    if (argc <= 3) {
        printf("Please input <s, N, iter> ...\n");
        return -1;
    }

    int stride = atoi(argv[1]);
    int N = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    CNT = iterations * K_LEN + DELTA;

    hipSetDevice(1);

    printf("\n=====%10.4f KB (N=%d) array, %d total accesses, "
                    "%d iterations ====\n", sizeof(ARRAY_TYPE)*(float)N/1024,
                    N, CNT, iterations);

    printf("Stride = %d element, %lu byte\n", stride,
                    stride * sizeof(ARRAY_TYPE));

    run_test(N, iterations, stride);
    printf("===============================================\n\n");

    hipDeviceReset();
    return 0;
}

void run_test(int N, int iterations, int stride) {
    hipDeviceReset();

    hipError_t error_id;

    int i;
    ARRAY_TYPE * h_a;
    /* allocate on CPU */
    h_a = (ARRAY_TYPE *)malloc(sizeof(ARRAY_TYPE) * N);
    ARRAY_TYPE * d_a;
    /* allocate on GPU */
    error_id = hipMalloc ((void **) &d_a, sizeof(ARRAY_TYPE) * N);
    if (error_id != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error_id));
    }

    /* pointer-chase: initialize array elements on CPU. */

    for (i = 0; i < N; i++) {
        h_a[i] = (ARRAY_TYPE)((i+stride)%N);
    }

    /* copy array elements from CPU to GPU */
    error_id = hipMemcpy(d_a, h_a, sizeof(ARRAY_TYPE) * N,
                    hipMemcpyHostToDevice);
    if (error_id != hipSuccess) {
        printf("Error: is %s\n", hipGetErrorString(error_id));
    }

    unsigned int *h_index = (unsigned int *)malloc(sizeof(unsigned int)*CNT);

    unsigned int *d_index;
    error_id = hipMalloc( (void **) &d_index, sizeof(unsigned int)*CNT );
    if (error_id != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error_id));
    }

    hipDeviceSynchronize ();
    /* launch kernel: single thread*/
    dim3 Db = dim3(1);          //dimGrid, how many WGs
    dim3 Dg = dim3(1,1,1);      //dimBlock. WG size

    hipLaunchKernelGGL((cache_test_RO), dim3(Dg), dim3(Db), 0, 0, d_a, N,
                    iterations, d_index);
    hipDeviceSynchronize ();

    hipLaunchKernelGGL((cache_test_RW), dim3(Dg), dim3(Db), 0, 0, d_a, N,
                    iterations, d_index);
    hipDeviceSynchronize ();

    hipLaunchKernelGGL((cache_test_WO), dim3(Dg), dim3(Db), 0, 0, d_a, N,
                    iterations, stride);
    hipDeviceSynchronize ();

    error_id = hipGetLastError();
    if (error_id != hipSuccess) {
        printf("Error kernel is %s\n", hipGetErrorString(error_id));
    }

    /* copy results from GPU to CPU */
    hipDeviceSynchronize ();

    if (error_id != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error_id));
    }

    error_id = hipMemcpy((void *)h_index, (void *)d_index,
                    sizeof(unsigned int)*CNT, hipMemcpyDeviceToHost);
    if (error_id != hipSuccess) {
        printf("Error: %s\n", hipGetErrorString(error_id));
    }

    hipDeviceSynchronize ();

    /* free memory on GPU */
    hipFree(d_a);
    hipFree(d_index);

    /*free memory on CPU */
    free(h_a);
    free(h_index);

    hipDeviceReset();
}
