/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <dlfcn.h>

/**
* @addtogroup hipLaunchKernelGGL hipLaunchCooperativeKernel
* @{
* @ingroup DynamicLoading
* `hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
  std::uint32_t sharedMemBytes, hipStream_t stream, Args... args)` -
* launches Kernel with launch parameters and shared memory on stream with arguments passed
* `hipError_t hipLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX,
                                         void** kernelParams, unsigned int sharedMemBytes,
                                         hipStream_t stream))` -
* launches kernel f with launch parameters and shared memory on stream with arguments passed
* to kernelparams or extra, where thread blocks can cooperate and synchronize as they execute
*/

/**
 * Test Description
 * ------------------------
 * - Test case to verify locally loaded kernels and dynamically loaded kernels from the library.

 * Test source
 * ------------------------
 * - catch/unit/dynamicLoading/complex_loading_behavior.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */

__global__ static void vector_add(float* C, float* A, float* B, size_t N) {
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = offset; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

static bool launch_local_kernel() {
  bool testResult = true;
  float *A_d, *B_d, *C_d;
  float *A_h, *B_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(float);
  static int device = 0;

  HIPCHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, device));

  A_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(A_h == nullptr ? hipErrorOutOfMemory : hipSuccess);
  B_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(B_h == nullptr ? hipErrorOutOfMemory : hipSuccess);
  C_h = reinterpret_cast<float*>(malloc(Nbytes));
  HIPCHECK(C_h == nullptr ? hipErrorOutOfMemory : hipSuccess);

  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 1.618f + i;
    B_h[i] = 1.618f + i;
  }

  HIPCHECK(hipMalloc(&A_d, Nbytes));
  HIPCHECK(hipMalloc(&B_d, Nbytes));
  HIPCHECK(hipMalloc(&C_d, Nbytes));
  HIPCHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIPCHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  const unsigned blocks = 512;
  const unsigned threadsPerBlock = 256;
  hipLaunchKernelGGL(vector_add, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, B_d, N);
  HIPCHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != (A_h[i] + B_h[i])) {
      testResult = false;
      break;
    }
  }

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(B_d));
  HIPCHECK(hipFree(C_d));

  free(A_h);
  free(B_h);
  free(C_h);
  return testResult;
}

static bool launch_dynamically_loaded_kernel() {
  bool testResult = true;
  int ret = 1;

  void* handle = dlopen("./libLazyLoad.so", RTLD_LAZY);
  if (!handle) {
    INFO("dlopen Error: " << dlerror() << "\n");
    testResult = false;
    return testResult;
  }
  void* sym = dlsym(handle, "lazyLoad");
  if (!sym) {
    INFO("unable to locate lazyLoad within lazyLoad.so\n");
    dlclose(handle);
    testResult = false;
    return testResult;
  }

  int (*fp)() = reinterpret_cast<int (*)()>(sym);
  ret = fp();

  if (ret == 0) {
    testResult = false;
  }

  dlclose(handle);
  return testResult;
}

TEST_CASE("Unit_dynamic_loading_device_kernels_from_library") {
  bool testResult = true;

  testResult &= launch_local_kernel();
  testResult &= launch_dynamically_loaded_kernel();

  REQUIRE(testResult == true);
}

/**
 * End doxygen group DynamicLoading.
 * @}
 */
