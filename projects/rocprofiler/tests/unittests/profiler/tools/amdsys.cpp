// TODO(aelwazir): To be checked

#include "hip/hip_runtime.h"
 
#include <cstdio>
#include <unistd.h>
#include <hip/hip_profile.h>
#include <iostream>
 
 
#define N 2560
//change here to run this app longer
#define num_iters 1
 
 
template<int n, int m>
__global__ void kernel(double* x) {
    for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < N; idx += gridDim.x * blockDim.x)
    {
        #pragma unroll
        for (int i = 0; i < n; ++i)
            x[idx] += i * m;
    }
}
 
void cpuWork() {
    // Do some CPU "work".
    usleep(1000);
}
 
inline void hip_assert(hipError_t err, const char *file, int line)
{
    if (err != hipSuccess)
    {
        fprintf(stderr,"HIP error: %s %s %d\n", hipGetErrorString(err), file, line);
        exit(-1);
    }
}
 
#define hipErrorCheck(f) { hip_assert((f), __FILE__, __LINE__); }
#define kernelErrorCheck() { hipErrorCheck(hipPeekAtLastError()); }
 
int main() {
 
    double* x;
    double* x_h;
 
    size_t sz = N * sizeof(double);
    std::cout << "running app....." << std::endl;
    hipErrorCheck(hipHostMalloc(&x_h, sz));
 
    memset(x_h, 0, sz);
    hipErrorCheck(hipMallocManaged(&x, sz));
    hipErrorCheck(hipMemset(x, 0, sz));
 
    hipStream_t stream;
    hipErrorCheck(hipStreamCreate(&stream));
 
    hipFuncAttributes attr;
 
    int blocks = 80;
    int threads = 32;
    int fact = 100;
    for (int j = 0; j < num_iters; ++j) {
        for (int n = 0; n < 25*fact; ++n) {
            hipErrorCheck(hipMemcpyAsync(x, x_h, sz, hipMemcpyHostToDevice));
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,1>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,2>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,3>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,4>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,5>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,6>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,7>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,8>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,9>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,10>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,11>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,12>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,13>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,14>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,15>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,16>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,17>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,18>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,19>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,20>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,20>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,21>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,22>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,23>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,24>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,25>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,26>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,27>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,28>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,29>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,30>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,30>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,31>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,32>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,33>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,34>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,35>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,36>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,37>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,38>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,39>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<1,40>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipErrorCheck(hipMemcpyAsync(x_h, x, sz, hipMemcpyDeviceToHost));
            hipErrorCheck(hipDeviceSynchronize());
        }
 
        hipErrorCheck(hipMemset(x, 0, sz));
        cpuWork();
 
        for (int n = 0; n < 200*fact; ++n) {
            hipErrorCheck(hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel<10,1>)));
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<10,1>), dim3(blocks), dim3(threads), 0, stream, x);
            kernelErrorCheck();
            hipErrorCheck(hipStreamSynchronize(stream));
        }
 
        hipErrorCheck(hipMemset(x, 0, sz));
        cpuWork();
 
        for (int n = 0; n < 30*fact; ++n) {
            for (int k = 0; k < 7; ++k) {
                hipErrorCheck(hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(kernel<8,1>)));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<8,1>), dim3(blocks), dim3(threads), 0, 0, x);
                kernelErrorCheck();
            }
            hipErrorCheck(hipDeviceSynchronize());
        }
 
        hipErrorCheck(hipMemset(x, 0, sz));
        cpuWork();
 
        for (int n = 0; n < 100*fact; ++n) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<7,1>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<7,2>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<7,3>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<7,4>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<7,5>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipErrorCheck(hipDeviceSynchronize());
        }
 
        hipErrorCheck(hipMemset(x, 0, sz));
        cpuWork();
 
        for (int n = 0; n < 100*fact; ++n) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<7,1>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<7,2>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<7,3>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<7,4>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<7,5>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipErrorCheck(hipDeviceSynchronize());
        }
 
        hipErrorCheck(hipMemset(x, 0, sz));
        cpuWork();
 
        for (int n = 0; n < 50*fact; ++n) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<6,1>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<6,2>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<6,3>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<6,4>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<6,5>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<6,6>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<6,7>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<6,8>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipErrorCheck(hipDeviceSynchronize());
        }
 
        hipErrorCheck(hipMemset(x, 0, sz));
        cpuWork();
 
        for (int n = 0; n < 50*fact; ++n) {
            int val;
            hipErrorCheck(hipDeviceGetAttribute(&val, hipDeviceAttributeMaxThreadsPerBlock, 0));
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<4000,1>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipErrorCheck(hipDeviceSynchronize());
        }
 
        hipErrorCheck(hipMemset(x, 0, sz));
        cpuWork();
 
        for (int n = 0; n < 50*fact; ++n) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel<5000,1>), dim3(blocks), dim3(threads), 0, 0, x);
            kernelErrorCheck();
            hipErrorCheck(hipDeviceSynchronize());
        }
 
        hipErrorCheck(hipMemset(x, 0, sz));
        cpuWork();
 
        hipErrorCheck(hipDeviceSynchronize());
 
    }
 
    hipErrorCheck(hipHostFree(x_h));
    hipErrorCheck(hipFree(x));
    hipErrorCheck(hipStreamDestroy(stream));
 
}