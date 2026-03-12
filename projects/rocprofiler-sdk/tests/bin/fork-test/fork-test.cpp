/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hip_runtime.h>
#include <sys/wait.h>
#include <unistd.h>
#include <iostream>

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

// Simple vector addition kernel
__global__ void
vectorAdd(float* out, float* a, float* b, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < n)
    {
        out[idx] = a[idx] + b[idx];
    }
}

void
runKernel(const char* process_name)
{
    const int N    = 1024;
    size_t    size = N * sizeof(float);

    // Allocate device memory
    float* d_a   = nullptr;
    float* d_b   = nullptr;
    float* d_out = nullptr;

    HIP_API_CALL(hipMalloc(&d_a, size));
    HIP_API_CALL(hipMalloc(&d_b, size));
    HIP_API_CALL(hipMalloc(&d_out, size));

    // Initialize host arrays
    float* h_a   = new float[N];
    float* h_b   = new float[N];
    float* h_out = new float[N];

    for(int i = 0; i < N; i++)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Copy to device
    HIP_API_CALL(hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice));
    HIP_API_CALL(hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "[" << process_name << " PID=" << getpid() << "] Launching kernel with "
              << blocksPerGrid << " blocks and " << threadsPerBlock << " threads per block\n";

    hipLaunchKernelGGL(
        vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, d_out, d_a, d_b, N);

    // Copy result back
    HIP_API_CALL(hipMemcpy(h_out, d_out, size, hipMemcpyDeviceToHost));
    HIP_API_CALL(hipDeviceSynchronize());

    // Verify result
    bool success = true;
    for(int i = 0; i < N; i++)
    {
        if(h_out[i] != h_a[i] + h_b[i])
        {
            success = false;
            break;
        }
    }

    std::cout << "[" << process_name << " PID=" << getpid() << "] Kernel execution "
              << (success ? "PASSED" : "FAILED") << "\n";

    // Cleanup
    HIP_API_CALL(hipFree(d_a));
    HIP_API_CALL(hipFree(d_b));
    HIP_API_CALL(hipFree(d_out));
    delete[] h_a;
    delete[] h_b;
    delete[] h_out;
}

int
main()
{
    std::cout << "Fork test starting...\n";
    std::cout << "Parent PID: " << getpid() << "\n";

    pid_t pid = fork();

    if(pid < 0)
    {
        std::cerr << "Fork failed!\n";
        return 1;
    }
    else if(pid == 0)
    {
        // Child process
        std::cout << "\n=== CHILD PROCESS ===\n";
        runKernel("CHILD");
        std::cout << "=== CHILD PROCESS COMPLETE ===\n\n";
        return 0;
    }
    else
    {
        // Parent process
        std::cout << "\n=== PARENT PROCESS ===\n";
        runKernel("PARENT");
        std::cout << "=== PARENT PROCESS COMPLETE ===\n\n";

        // Wait for child to complete
        int status;
        waitpid(pid, &status, 0);

        if(WIFEXITED(status))
        {
            std::cout << "Child process exited with status: " << WEXITSTATUS(status) << "\n";
        }
    }

    std::cout << "Fork test complete!\n";
    return 0;
}
