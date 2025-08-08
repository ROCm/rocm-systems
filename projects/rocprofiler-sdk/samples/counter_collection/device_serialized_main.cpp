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
#include <stdlib.h>
#include <time.h>
#include <cassert>

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
kernelA(int devid, volatile int* wait_on, int value, int* no_opt)
{
    printf("[device=%i][begin]  Wait on %i: %i (%i)\n", devid, value, *wait_on, *no_opt);
    while(*wait_on != value)
    {
        (*no_opt)++;
    };
    printf("[device=%i][break]  Wait on %i: %i (%i)\n", devid, value, *wait_on, *no_opt);
    (*wait_on)--;
    printf("[device=%i][return] Wait on %i: %i (%i)\n", devid, value, *wait_on, *no_opt);
}

__global__ void
check_order_kernel(int expected, int* actual)
{
    // Note: We do not use atomics here on purpose to ensure that the barrier
    // being injected has proper fencing set.
    if(*actual != expected)
    {
        printf("[error]  Expected %i but got %i\n", expected, *actual);
    }
    assert(*actual == expected);
    (*actual)++;
}

class DualStreamExecutor
{
private:
    hipStream_t stream1_, stream2_;
    int         device_;

public:
    DualStreamExecutor(int device = 0)
    : device_(device)
    {
        HIP_CALL(hipSetDevice(device_));
        HIP_CALL(hipStreamCreate(&stream1_));
        HIP_CALL(hipStreamCreate(&stream2_));
        std::cout << "Created dual streams on device " << device_ << std::endl;
    }

    ~DualStreamExecutor()
    {
        HIP_CALL(hipStreamDestroy(stream1_));
        HIP_CALL(hipStreamDestroy(stream2_));
    }

    // Function template to launch any kernel on both streams
    template <typename KernelFunc, typename... Args>
    void launch_kernel_on_both_streams(KernelFunc kernel,
                                       dim3       gridSize,
                                       dim3       blockSize,
                                       size_t     sharedMem,
                                       Args... args)
    {
        hipLaunchKernelGGL(kernel, gridSize, blockSize, sharedMem, stream1_, args...);
        hipLaunchKernelGGL(kernel, gridSize, blockSize, sharedMem, stream2_, args...);
    }

    // Synchronize both streams
    void synchronize()
    {
        HIP_CALL(hipStreamSynchronize(stream1_));
        HIP_CALL(hipStreamSynchronize(stream2_));
        std::cout << "Both streams synchronized" << std::endl;
    }

    // Get stream handles
    hipStream_t get_stream1() const { return stream1_; }
    hipStream_t get_stream2() const { return stream2_; }

    // Execute async memory operations on both streams
    void async_memcpy_to_device(void* dst1, void* dst2, const void* src, size_t size)
    {
        HIP_CALL(hipMemcpyAsync(dst1, src, size, hipMemcpyHostToDevice, stream1_));
        HIP_CALL(hipMemcpyAsync(dst2, src, size, hipMemcpyHostToDevice, stream2_));
    }

    void async_memcpy_to_host(void*       dst1,
                              void*       dst2,
                              const void* src1,
                              const void* src2,
                              size_t      size)
    {
        HIP_CALL(hipMemcpyAsync(dst1, src1, size, hipMemcpyDeviceToHost, stream1_));
        HIP_CALL(hipMemcpyAsync(dst2, src2, size, hipMemcpyDeviceToHost, stream2_));
    }
};

int
main(int, char**)
{
    int ntotdevice = 0;
    HIP_CALL(hipGetDeviceCount(&ntotdevice));
    if(ntotdevice < 2) return 0;

    start();
    volatile int* check_value = nullptr;
    int*          no_opt_0    = nullptr;
    int*          no_opt_1    = nullptr;
    HIP_CALL(hipMallocManaged(&check_value, sizeof(*check_value)));
    HIP_CALL(hipMallocManaged(&no_opt_0, sizeof(*no_opt_0)));
    HIP_CALL(hipMallocManaged(&no_opt_1, sizeof(*no_opt_1)));
    *no_opt_0    = 0;
    *no_opt_1    = 0;
    *check_value = 1;

    // Will hang if per-device serialization is not functional
    HIP_CALL(hipSetDevice(0));
    hipLaunchKernelGGL(kernelA, dim3(1), dim3(1), 0, 0, 0, check_value, 0, no_opt_0);

    HIP_CALL(hipSetDevice(1));
    hipLaunchKernelGGL(kernelA, dim3(1), dim3(1), 0, 0, 1, check_value, 1, no_opt_1);

    HIP_CALL(hipSetDevice(0));
    HIP_CALL(hipDeviceSynchronize());

    // Validate that kernels are being processed in order on the same device
    HIP_CALL(hipSetDevice(0));
    DualStreamExecutor executor(0);
    *no_opt_0 = 0;
    srand((unsigned int) time(NULL));

    for(int i = 0; i < 10000; i++)
    {
        if(rand() & 1)
        {
            hipLaunchKernelGGL(
                check_order_kernel, dim3(1), dim3(1), 0, executor.get_stream1(), i, no_opt_0);
        }
        else
        {
            hipLaunchKernelGGL(
                check_order_kernel, dim3(1), dim3(1), 0, executor.get_stream2(), i, no_opt_1);
        }
    }
    executor.synchronize();
    HIP_CALL(hipDeviceSynchronize());
    std::cerr << "Run complete\n";
}
