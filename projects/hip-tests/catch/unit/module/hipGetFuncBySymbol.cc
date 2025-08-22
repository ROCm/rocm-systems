/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @addtogroup hipGetFuncBySymbol hipGetFuncBySymbol
 * @{
 * @ingroup KernelTest
 * `hipError_t hipGetFuncBySymbol (hipFunction_t* functionPtr,
 *                                 const void* symbolPtr
 *                                )` -
 * Gets pointer to device entry function that matches entry function symbolPtr.
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip_test_checkers.hh>
#include <hip_test_process.hh>

#define LEN 64
#define SIZE LEN * sizeof(float)

#define ARR_SIZE (32 * 32)
#define SIZE_BYTES (ARR_SIZE * sizeof(int))

extern "C" __global__ void bit_extract_kernel(uint32_t* C_d, const uint32_t* A_d, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  for (size_t i = offset; i < N; i += stride) {
#if HT_AMD
    C_d[i] = __bitextract_u32(A_d[i], 8, 4);
#else /* defined __HIP_PLATFORM_NVIDIA__ or other path */
    C_d[i] = ((A_d[i] & 0xf00) >> 8);
#endif
  }
}

/**
 * Host Function to check for negative case.
 */
__host__ void hostFunction() { printf("hostFunction\n"); }

/**
 * Sample Kernel to be used for functional test cases
 */
__global__ void hipKernel(int* a) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = offset; i < ARR_SIZE; i += stride) {
    a[i] += a[i];
  }
}

/**
 * Local Function to validate the result
 */
bool verifyResult(int* a, int* output_ref, int arrSize) {
  for (int i = 0; i < arrSize; i++) {
    if (a[i] != output_ref[i]) {
      return false;
    }
  }
  return true;
}

/**
 * Test Description
 * ------------------------
 * - Test is to get function ptr (hipFunction_t) using hipGetFuncBySymbol and launch
 *   bit_extract kernel. Verify the output.

 * Test source
 * ------------------------
 * - catch/unit/module/hipGetFuncBySymbol.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.2
 */

TEST_CASE("Unit_hipGetFuncBySymbol_PositiveTest") {
  uint32_t *A_d, *C_d;
  uint32_t *A_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(uint32_t);

  hipDevice_t device;
  HIPCHECK(hipGetDevice(&device));

  hipDeviceProp_t props;
  HIPCHECK(hipGetDeviceProperties(&props, device));
  A_h = reinterpret_cast<uint32_t*>(malloc(Nbytes));
  REQUIRE(A_h != NULL);
  C_h = reinterpret_cast<uint32_t*>(malloc(Nbytes));
  REQUIRE(C_h != NULL);

  for (size_t i = 0; i < N; i++) {
    A_h[i] = i;
  }

  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes));
  HIPCHECK(hipMalloc(reinterpret_cast<void**>(&C_d), Nbytes));

  HIPCHECK(hipMemcpyHtoD((hipDeviceptr_t)(A_d), A_h, Nbytes));

  struct {
    void* _Cd;
    void* _Ad;
    size_t _N;
  } args;
  args._Cd = reinterpret_cast<void**>(C_d);
  args._Ad = reinterpret_cast<void**>(A_d);
  args._N = static_cast<size_t>(N);
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};

  hipFunction_t Function;
  HIPCHECK(hipGetFuncBySymbol(&Function, reinterpret_cast<void*>(bit_extract_kernel)));

  HIPCHECK(hipModuleLaunchKernel(Function, 1, 1, 1, LEN, 1, 1, 0, 0, NULL,
                                 reinterpret_cast<void**>(&config)));

  HIPCHECK(hipMemcpyDtoH(C_h, (hipDeviceptr_t)(C_d), Nbytes));

  for (size_t i = 0; i < N; i++) {
    unsigned Agold = ((A_h[i] & 0xf00) >> 8);
    REQUIRE(C_h[i] == Agold);
  }

  HIPCHECK(hipFree(A_d));
  HIPCHECK(hipFree(C_d));

  free(A_h);
  free(C_h);
}

/**
 * Test Description
 * ------------------------
 *    - Pass the NULL as a symbolPtr it should return
 *      hipErrorInvalidDeviceFunction
 *    - Pass a host function as a symbolPtr it should return
 *      hipErrorInvalidDeviceFunction
 * Test source
 * ------------------------
 *    - catch/unit/module/hipGetFuncBySymbol.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */

TEST_CASE("Unit_hipGetFuncBySymbol_NegativeTests") {
  hipFunction_t funcPointer;

  // Passing NULL as second parameter
  REQUIRE(hipGetFuncBySymbol(&funcPointer, NULL) != hipSuccess);

  // Passing hostFunction as second parameter
  REQUIRE(hipGetFuncBySymbol(&funcPointer, reinterpret_cast<const void*>(hostFunction)));
}

/**
 * Test Description
 * ------------------------
 *    - Create a child process and pass the __global__ function as a symbolPtr
 *      it should return hipSuccess, and kernel launch and execution with
 *      functionPtr should success.
 * Test source
 * ------------------------
 *    - catch/unit/module/hipGetFuncBySymbol.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetFuncBySymbol_InChildProcess") {
  hip::SpawnProc proc("hipGetFuncBySymbol_exe", true);
  REQUIRE(proc.run() == 0);
}

/**
 * Test Description
 * ------------------------
 *    - For all the GPU devices in the system and pass the __global__ function
 *      as a symbolPtr it should return hipSuccess, and kernel launch and
 *      execution with functionPtr should success.
 * Test source
 * ------------------------
 *    - catch/unit/module/hipGetFuncBySymbol.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetFuncBySymbol_MultiDev") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  hipFunction_t funcPointer;

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    HIP_CHECK(hipSetDevice(deviceId));

    REQUIRE(hipGetFuncBySymbol(&funcPointer, reinterpret_cast<const void*>(hipKernel)) ==
            hipSuccess);

    int* h_a = reinterpret_cast<int*>(malloc(SIZE_BYTES));
    REQUIRE(h_a != nullptr);
    int* output_ref = reinterpret_cast<int*>(malloc(SIZE_BYTES));
    REQUIRE(output_ref != nullptr);

    for (int i = 0; i < ARR_SIZE; i++) {
      h_a[i] = 2;
      output_ref[i] = 4;
    }

    int* d_a = nullptr;
    HIP_CHECK(hipMalloc(&d_a, SIZE_BYTES));
    REQUIRE(d_a != nullptr);
    HIP_CHECK(hipMemcpy(d_a, h_a, SIZE_BYTES, hipMemcpyHostToDevice));

    dim3 blocksPerGrid(1, 1, 1);
    dim3 threadsPerBlock(1, 1, 64);

    void* kernelParam[] = {d_a};
    auto size = sizeof(kernelParam);
    void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                                HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

    REQUIRE(hipModuleLaunchKernel(funcPointer, blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z,
                                  threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z, 0, 0,
                                  nullptr, kernel_parameter) == hipSuccess);

    HIP_CHECK(hipMemcpy(h_a, d_a, SIZE_BYTES, hipMemcpyDeviceToHost));

    REQUIRE(verifyResult(h_a, output_ref, ARR_SIZE) == true);

    free(h_a);
    free(output_ref);
    HIP_CHECK(hipFree(d_a));
  }
}

/**
 * Local function useful to create stream and memory copy and launch kernel
 */
void MultiThreadMultiDevFunc(int DevId) {
  HIP_CHECK(hipSetDevice(DevId));

  int* h_a = reinterpret_cast<int*>(malloc(SIZE_BYTES));
  REQUIRE(h_a != nullptr);
  int* output_ref = reinterpret_cast<int*>(malloc(SIZE_BYTES));
  REQUIRE(output_ref != nullptr);

  for (int i = 0; i < ARR_SIZE; i++) {
    h_a[i] = 2;
    output_ref[i] = 4;
  }

  hipStream_t stream;
  HIP_CHECK(hipSetDevice(DevId));
  HIP_CHECK(hipStreamCreate(&stream));

  int* d_a = nullptr;
  HIP_CHECK(hipMalloc(&d_a, SIZE_BYTES));
  REQUIRE(d_a != nullptr);
  HIP_CHECK(hipMemcpyAsync(d_a, h_a, SIZE_BYTES, hipMemcpyHostToDevice, stream));

  dim3 blocksPerGrid(1, 1, 1);
  dim3 threadsPerBlock(1, 1, 64);

  hipFunction_t funcPointer;
  REQUIRE(hipGetFuncBySymbol(&funcPointer, reinterpret_cast<const void*>(hipKernel)) == hipSuccess);

  void* kernelParam[] = {d_a};
  auto size = sizeof(kernelParam);
  void* kernel_parameter[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &kernelParam,
                              HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};

  REQUIRE(hipModuleLaunchKernel(funcPointer, blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z,
                                threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z, 0, stream,
                                nullptr, kernel_parameter) == hipSuccess);

  HIP_CHECK(hipMemcpyAsync(h_a, d_a, SIZE_BYTES, hipMemcpyDeviceToHost, stream));

  REQUIRE(verifyResult(h_a, output_ref, ARR_SIZE) == true);

  free(h_a);
  free(output_ref);
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(d_a));
}

/**
 * Test Description
 * ------------------------
 *    - Create number of threads equals to number of devices and in each devices
 *      pass the __global__ function as a symbolPtr it should return hipSuccess,
 *      and kernel launch and execution with functionPtr should success.
 * Test source
 * ------------------------
 *    - catch/unit/module/hipGetFuncBySymbol.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetFuncBySymbol_MultiDevMultiThread") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  ::std::vector< ::std::thread> threads;

  for (int dev = 0; dev < deviceCount; dev++) {
    threads.push_back(::std::thread(MultiThreadMultiDevFunc, dev));
  }

  for (int dev = 0; (dev < deviceCount) && (dev < threads.size()); dev++) {
    threads[dev].join();
  }
}
