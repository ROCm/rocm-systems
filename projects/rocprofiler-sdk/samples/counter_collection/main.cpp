// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <hip/hip_runtime.h>

#include <libgen.h>
#include "client.hpp"

#define HIP_CALL(call)                                                                             \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = call;                                                                     \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            fprintf(stderr, "%s\n", hipGetErrorString(err));                                       \
            abort();                                                                               \
        }                                                                                          \
    } while(0)

__global__ void
kernelA(int x, int y)
{
    x = x + y;
}

__global__ void
kernelB(int x, int y)
{
    x = x + y;
}

template <typename T>
__global__ void
kernelC(T* C_d, const T* A_d, size_t N)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = offset; i < N; i += stride)
    {
        C_d[i] = A_d[i] * A_d[i];
    }
}

void
launchKernels(const long NUM_LAUNCH, const long SYNC_INTERVAL, const int DEV_ID)
{
    // Normal HIP Calls
    HIP_CALL(hipSetDevice(DEV_ID));
    [[maybe_unused]] hipDeviceProp_t devProp;
    HIP_CALL(hipGetDeviceProperties(&devProp, DEV_ID));

    int* gpuMem = nullptr;
    HIP_CALL(hipMalloc((void**) &gpuMem, 1 * sizeof(int)));

    for(long i = 0; i < NUM_LAUNCH; i++)
    {
        // KernelA and KernelB to be profiled as part of the session
        hipLaunchKernelGGL(kernelA, dim3(1), dim3(1), 0, 0, 1, 2);
        hipLaunchKernelGGL(kernelB, dim3(1), dim3(1), 0, 0, 1, 2);
        if(i % SYNC_INTERVAL == (SYNC_INTERVAL - 1)) HIP_CALL(hipDeviceSynchronize());
    }

    const int NElems = 512 * 512;
    const int Nbytes = NElems * sizeof(int);
    int *     A_d, *C_d;
    int       A_h[NElems], C_h[NElems];

    for(int i = 0; i < NElems; i++)
    {
        A_h[i] = i;
    }

    HIP_CALL(hipDeviceSynchronize());

    HIP_CALL(hipMalloc(&A_d, Nbytes));
    HIP_CALL(hipMalloc(&C_d, Nbytes));
    HIP_CALL(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CALL(hipDeviceSynchronize());
    const unsigned blocks          = 512;
    const unsigned threadsPerBlock = 256;
    for(long i = 0; i < NUM_LAUNCH; i++)
    {
        hipLaunchKernelGGL(kernelC, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, NElems);
        if(i % SYNC_INTERVAL == (SYNC_INTERVAL - 1)) HIP_CALL(hipDeviceSynchronize());
    }
    HIP_CALL(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipFree(gpuMem));
    HIP_CALL(hipFree(A_d));
    HIP_CALL(hipFree(C_d));
    HIP_CALL(hipDeviceReset());
}

int
main(int argc, char** argv)
{
    auto* exe_name = basename(argv[0]);

    int ntotdevice = 0;
    HIP_CALL(hipGetDeviceCount(&ntotdevice));

    long nitr    = 5000;
    long nsync   = 50;
    long ndevice = 0;

    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{argv[i]};
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            fprintf(stderr,
                    "usage: %s [NUM_ITERATION (%li)] [SYNC_EVERY_N_ITERATIONS (%li)] "
                    "[NUMBER_OF_DEVICES (%li)]\n\n\tBy default, 0 for the number of devices means "
                    "use all device available",
                    exe_name,
                    nitr,
                    nsync,
                    ndevice);
            exit(EXIT_SUCCESS);
        }
    }

    if(argc > 1) nitr = atol(argv[1]);
    if(argc > 2) nsync = atoll(argv[2]);
    if(argc > 3) ndevice = atol(argv[3]);

    if(ndevice > ntotdevice) ndevice = ntotdevice;
    if(ndevice < 1) ndevice = ntotdevice;

    printf("[%s] Number of devices used: %li\n", exe_name, ndevice);
    printf("[%s] Number of iterations: %li\n", exe_name, nitr);
    printf("[%s] Syncing every %li iterations\n", exe_name, nsync);
    std::cout << std::flush;

    start();
    for(long devid = 0; devid < ndevice; ++devid)
        launchKernels(nitr, nsync, devid);

    std::cerr << "Run complete\n" << std::flush;
}
