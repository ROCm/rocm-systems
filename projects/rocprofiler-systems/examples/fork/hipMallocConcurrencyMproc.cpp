/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "hip/hip_runtime.h"
#include <assert.h>
#include <iostream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define REQUIRE assert
#define HIP_CHECK(cmd)                                                                   \
    {                                                                                    \
        hipError_t error = cmd;                                                          \
        if(error != hipSuccess)                                                          \
        {                                                                                \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error),      \
                    error, __FILE__, __LINE__);                                          \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }
#define TOL 0.001

namespace HipTest
{
// Setters and Memory Management

template <typename T>
void
setDefaultData(size_t numElements, T* A_h, T* B_h, T* C_h)
{
    // Initialize the host data:

    for(size_t i = 0; i < numElements; i++)
    {
        if(std::is_same<T, int>::value || std::is_same<T, unsigned int>::value)
        {
            if(A_h) A_h[i] = 3;
            if(B_h) B_h[i] = 4;
            if(C_h) C_h[i] = 5;
        }
        else if(std::is_same<T, char>::value || std::is_same<T, unsigned char>::value)
        {
            if(A_h) A_h[i] = 'a';
            if(B_h) B_h[i] = 'b';
            if(C_h) C_h[i] = 'c';
        }
        else
        {
            if(A_h) A_h[i] = 3.146f + i;
            if(B_h) B_h[i] = 1.618f + i;
            if(C_h) C_h[i] = 1.4f + i;
        }
    }
}

template <typename T>
bool
initArraysForHost(T** A_h, T** B_h, T** C_h, size_t N, bool usePinnedHost = false)
{
    size_t Nbytes = N * sizeof(T);

    if(usePinnedHost)
    {
        if(A_h)
        {
            HIP_CHECK(hipHostMalloc((void**) A_h, Nbytes));
        }
        if(B_h)
        {
            HIP_CHECK(hipHostMalloc((void**) B_h, Nbytes));
        }
        if(C_h)
        {
            HIP_CHECK(hipHostMalloc((void**) C_h, Nbytes));
        }
    }
    else
    {
        if(A_h)
        {
            *A_h = (T*) malloc(Nbytes);
            REQUIRE(*A_h != nullptr);
        }

        if(B_h)
        {
            *B_h = (T*) malloc(Nbytes);
            REQUIRE(*B_h != nullptr);
        }

        if(C_h)
        {
            *C_h = (T*) malloc(Nbytes);
            REQUIRE(*C_h != nullptr);
        }
    }

    setDefaultData(N, A_h ? *A_h : nullptr, B_h ? *B_h : nullptr, C_h ? *C_h : nullptr);
    return true;
}

template <typename T>
bool
initArrays(T** A_d, T** B_d, T** C_d, T** A_h, T** B_h, T** C_h, size_t N,
           bool usePinnedHost = false)
{
    size_t Nbytes = N * sizeof(T);

    if(A_d)
    {
        HIP_CHECK(hipMalloc(A_d, Nbytes));
    }
    if(B_d)
    {
        HIP_CHECK(hipMalloc(B_d, Nbytes));
    }
    if(C_d)
    {
        HIP_CHECK(hipMalloc(C_d, Nbytes));
    }

    return initArraysForHost(A_h, B_h, C_h, N, usePinnedHost);
}

template <typename T>
bool
freeArraysForHost(T* A_h, T* B_h, T* C_h, bool usePinnedHost)
{
    if(usePinnedHost)
    {
        if(A_h)
        {
            HIP_CHECK(hipHostFree(A_h));
        }
        if(B_h)
        {
            HIP_CHECK(hipHostFree(B_h));
        }
        if(C_h)
        {
            HIP_CHECK(hipHostFree(C_h));
        }
    }
    else
    {
        if(A_h)
        {
            free(A_h);
        }
        if(B_h)
        {
            free(B_h);
        }
        if(C_h)
        {
            free(C_h);
        }
    }
    return true;
}

template <typename T>
bool
freeArrays(T* A_d, T* B_d, T* C_d, T* A_h, T* B_h, T* C_h, bool usePinnedHost)
{
    if(A_d)
    {
        HIP_CHECK(hipFree(A_d));
    }
    if(B_d)
    {
        HIP_CHECK(hipFree(B_d));
    }
    if(C_d)
    {
        HIP_CHECK(hipFree(C_d));
    }

    return freeArraysForHost(A_h, B_h, C_h, usePinnedHost);
}

static inline unsigned
setNumBlocks(unsigned blocksPerCU, unsigned threadsPerBlock, size_t N)
{
    int device{ 0 };
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    unsigned blocks = props.multiProcessorCount * blocksPerCU;
    if(blocks * threadsPerBlock < N)
    {
        blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    }

    return blocks;
}

template <typename T>
__global__ void
vectorADD(const T* A_d, const T* B_d, T* C_d, size_t NELEM)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for(size_t i = offset; i < NELEM; i += stride)
    {
        C_d[i] = A_d[i] + B_d[i];
    }
}

template <typename T>
size_t
checkVectors(T* A, T* B, T* Out, size_t N, T (*F)(T a, T b), bool expectMatch = true,
             bool reportMismatch = true)
{
    size_t mismatchCount     = 0;
    size_t firstMismatch     = 0;
    size_t mismatchesToPrint = 10;
    for(size_t i = 0; i < N; i++)
    {
        T expected = F(A[i], B[i]);
        if(std::fabs(Out[i] - expected) > TOL)
        {
            if(mismatchCount == 0)
            {
                firstMismatch = i;
            }
            mismatchCount++;
            if((mismatchCount <= mismatchesToPrint) && expectMatch)
            {
                std::cout << "Mismatch at " << i << " Computed: " << Out[i]
                          << " Expected: " << expected << std::endl;
                REQUIRE(false);
            }
        }
    }

    if(reportMismatch)
    {
        if(expectMatch)
        {
            if(mismatchCount)
            {
                std::cout << mismatchCount
                          << " Mismatches  First Mismatch at index : " << firstMismatch
                          << std::endl;
                REQUIRE(false);
            }
        }
        else
        {
            if(mismatchCount == 0)
            {
                std::cout << "Expected Mismatch but not found any" << std::endl;
                REQUIRE(false);
            }
        }
    }

    return mismatchCount;
}

template <typename T>
size_t
checkVectorADD(T* A_h, T* B_h, T* result_H, size_t N, bool expectMatch = true,
               bool reportMismatch = true)
{
    return checkVectors<T>(
        A_h, B_h, result_H, N, [](T a, T b) { return a + b; }, expectMatch,
        reportMismatch);
}
}  // namespace HipTest

/**
 * Validates data consistency on supplied gpu
 */
static bool
validateMemoryOnGPU(int gpu, bool concurOnOneGPU = false)
{
    // Check if any ROCm-capable GPU is available (for CI without GPU)
    int        deviceCount = 0;
    hipError_t err         = hipGetDeviceCount(&deviceCount);
    if(err != hipSuccess || deviceCount == 0)
    {
        printf("No ROCm-capable device detected. Validation PASSED (skipped)\n");
        return true;  // Return success for CI environments
    }

    int *          A_d, *B_d, *C_d;
    int *          A_h, *B_h, *C_h;
    size_t         prevAvl, prevTot, curAvl, curTot;
    bool           TestPassed      = true;
    constexpr auto N               = 4 * 1024 * 1024;
    constexpr auto blocksPerCU     = 6;  // to hide latency
    constexpr auto threadsPerBlock = 256;
    size_t         Nbytes          = N * sizeof(int);

    HIP_CHECK(hipSetDevice(gpu));
    HIP_CHECK(hipMemGetInfo(&prevAvl, &prevTot));
    HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
    HIP_CHECK(hipMemGetInfo(&curAvl, &curTot));

    if(!concurOnOneGPU && (prevAvl < curAvl || prevTot != curTot))
    {
        // In concurrent calls on one GPU, we cannot verify leaking in this way
        printf("%s : Memory allocation mismatch observed."
               "Possible memory leak.\n",
               __func__);
        TestPassed &= false;
    }

    unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

    HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, 0,
                       static_cast<const int*>(A_d), static_cast<const int*>(B_d), C_d,
                       N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    if(!HipTest::checkVectorADD(A_h, B_h, C_h, N))
    {
        printf("Validation PASSED for gpu %d from pid %d\n", gpu, getpid());
    }
    else
    {
        printf("Validation FAILED for gpu %d from pid %d\n", gpu, getpid());
        TestPassed = false;
    }

    HIP_CHECK(hipMemGetInfo(&prevAvl, &prevTot));
    HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
    HIP_CHECK(hipMemGetInfo(&curAvl, &curTot));

    if(!concurOnOneGPU && (curAvl < prevAvl || prevTot != curTot))
    {
        // In concurrent calls on one GPU, we cannot verify leaking in this way
        std::cout << "validateMemoryOnGPU : Memory allocation mismatch observed."
                  << "Possible memory leak." << std::endl;
        TestPassed = false;
    }

    if(!concurOnOneGPU && (prevAvl != curAvl || prevTot != curTot))
    {
        // In concurrent calls on one GPU, we cannot verify leaking in this way
        printf("%s : Memory allocation mismatch observed."
               "Possible memory leak.\n",
               __func__);
        TestPassed = false;
    }

    return TestPassed;
}

/**
 * Parallel execution of parent and child on gpu0
 */
void
Unit_hipMalloc_ChildConcurrencyDefaultGpu()
{
    int            pid        = 0;
    constexpr auto resSuccess = 0, resFailure = 1;

    if((pid = fork()) < 0)
    {
        std::cout << "Child_Concurrency_DefaultGpu : fork() returned error : " << pid
                  << std::endl;
        REQUIRE(false);
    }
    else if(!pid)
    {  // Child process
        bool TestPassedChild = false;

        // Allocates and validates memory on Gpu0 simultaneously with parent
        TestPassedChild = validateMemoryOnGPU(0, true);

        if(TestPassedChild)
        {
            exit(resSuccess);  // child exit with success status
        }
        else
        {
            exit(resFailure);  // child exit with failure status
        }
    }
    else
    {  // Parent process
        int exitStatus;

        // Allocates and validates memory on Gpu0 simultaneously with child
        bool TestPassed = validateMemoryOnGPU(0, true);

        // Wait and get result from child
        pid = wait(&exitStatus);
        if((WEXITSTATUS(exitStatus) == resFailure) || (pid < 0)) TestPassed = false;

        // Explicitly use the variable to avoid compiler warning
        (void) TestPassed;
        REQUIRE(TestPassed == true);
    }
}

int
main()
{
    Unit_hipMalloc_ChildConcurrencyDefaultGpu();
    std::cout << "Unit_hipMalloc_ChildConcurrencyDefaultGpu PASSED!" << std::endl;
    return 0;
}
