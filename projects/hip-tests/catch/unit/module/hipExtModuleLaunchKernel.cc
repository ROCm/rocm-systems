/*
Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * @addtogroup hipExtModuleLaunchKernel hipExtModuleLaunchKernel
 * @{
 * @ingroup KernelTest
 * `hipError_t hipExtModuleLaunchKernel	(hipFunction_t  f,
 *                                      uint32_t  globalWorkSizeX,
 *                                      uint32_t  globalWorkSizeY,
 *                                      uint32_t  globalWorkSizeZ,
 *                                      uint32_t  localWorkSizeX,
 *                                      uint32_t  localWorkSizeY,
 *                                      uint32_t  localWorkSizeZ,
 *                                      size_t  sharedMemBytes,
 *                                      hipStream_t  hStream,
 *                                      void **  kernelParams,
 *                                      void **  extra,
 *                                      hipEvent_t  startEvent = nullptr,
 *                                      hipEvent_t 	stopEvent = nullptr,
 *                                      uint32_t  flags = 0
 *                                      )` -
 * Launches kernel with parameters and shared memory on stream with arguments
 * passed to kernel params or extra arguments.
 */
#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <math.h>
#include <iostream>
#include <fstream>
#include <regex>  // NOLINT
#include <string>
#include "hip_module_launch_kernel_common.hh"  // NOLINT
#include "hip/hip_ext.h"

constexpr auto fileName = "copyKernel.code";
constexpr auto kernel_name = "copy_ker";
constexpr auto fileNameCompressed = "copyKernelCompressed.code";
constexpr auto fileNameGenericTarget = "copyKernelGenericTarget.code";
constexpr auto fileNameGenericTargetCompressed = "copyKernelGenericTargetCompressed.code";

static constexpr auto totalWorkGroups{1024};
static constexpr auto localWorkSize{512};
static constexpr auto lastWorkSizeEven{256};
static constexpr auto lastWorkSizeOdd{257};

/**
  Local Function to search a string in file.
*/
static bool searchRegExpr(const std::regex& expr, const char* filename) {
  std::ifstream assemblyfile(filename, std::ifstream::binary);
  REQUIRE(true == assemblyfile.is_open());

  // Copy the contents of the file to buffer
  assemblyfile.seekg(0, assemblyfile.end);
  int len = assemblyfile.tellg();
  assemblyfile.seekg(0, assemblyfile.beg);
  char* fbuf = new char[len + 1];
  assemblyfile.read(fbuf, len);
  fbuf[len] = '\0';

  // Search for uniform_work_group_size
  std::smatch pattern;
  std::string assemblyStr = fbuf;
  bool bfound = std::regex_search(assemblyStr, pattern, expr);
  delete[] fbuf;
  assemblyfile.close();
  return bfound;
}
/**
 * Test Description
 * ------------------------
 *    - Parse the Code Object Assembly file copyKernel.s and verify
 * if .uniform_work_group_size metadata is available.
 * ------------------------
 *    - catch\unit\module\hipExtModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipExtModuleLaunchKernel_CheckCodeObjAttr") {
  // Open copyKernel.s and read the file
  const std::regex regexp("uniform_work_group_size\\s*:\\s*[0-1]");
  REQUIRE(true == searchRegExpr(regexp, "copyKernel.s"));
}

/**
 * Test Description
 * ------------------------
 *    - Precondition: .uniform_work_group_size = 1. Which means uniform workgroup
 *      is enforced.
 *    - Create a buffer of size globalWorkSizeX = (non multiple of localWorkSizeX)
 * and launch a kernel to perform some operations on it. If uniform work grouping
 * is enforced then hipExtModuleLaunchKernel returns hipErrorInvalidValue.
 * Test source
 * ------------------------
 *    - catch\unit\module\hipExtModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipExtModuleLaunchKernel_NonUniformWorkGroup") {
  // first check if uniform_work_group_size = 1.
  const std::regex regexp("uniform_work_group_size\\s*:\\s*1");
  if (false == searchRegExpr(regexp, "copyKernel.s")) {
    HipTest::HIP_SKIP_TEST("uniform_work_group_size != 1. Skipping test ...");
    return;
  }
  REQUIRE(true == searchRegExpr(regexp, "copyKernel.s"));
  auto isEven = GENERATE(0, 1);
  // Calculate size
  auto lastWorkSize = isEven ? lastWorkSizeEven : lastWorkSizeOdd;
  size_t arraylength = (totalWorkGroups - 1) * localWorkSize + lastWorkSize;
  size_t sizeBytes{arraylength * sizeof(int)};
  // Get module and function from module
  hipModule_t Module;
  hipFunction_t Function;
  HIP_CHECK(hipModuleLoad(&Module, fileName));
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  // Allocate resources
  int* A = new int[arraylength];
  REQUIRE(A != nullptr);
  int* B = new int[arraylength];
  REQUIRE(B != nullptr);
  // Inititialize data
  for (size_t i = 0; i < arraylength; i++) {
    A[i] = i;
  }
  int *Ad, *Bd;
  HIP_CHECK(hipMalloc(&Ad, sizeBytes));
  HIP_CHECK(hipMalloc(&Bd, sizeBytes));
  struct {
    void* _Ad;
    void* _Bd;
    size_t buffersize;
  } args;

  args._Ad = Ad;
  args._Bd = Bd;
  args.buffersize = arraylength;
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  // Memcpy from A to Ad
  HIP_CHECK(hipMemcpy(Ad, A, sizeBytes, hipMemcpyDefault));
  REQUIRE(hipErrorInvalidValue == hipExtModuleLaunchKernel(Function, arraylength, 1, 1,
                                                           localWorkSize, 1, 1, 0, 0, NULL,
                                                           reinterpret_cast<void**>(&config), 0));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  delete[] A;
  delete[] B;
  HIP_CHECK(hipModuleUnload(Module));
}

/**
 * Test Description
 * ------------------------
 *    - Create a buffer of size globalWorkSizeX = (multiple of localWorkSizeX)
 * and launch a kernel to perform some operations on it. Verify the output. This
 * operation should be allowed for both uniform_work_group_size = true/false.
 *
 * Test source
 * ------------------------
 *    - catch\unit\module\hipExtModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipExtModuleLaunchKernel_UniformWorkGroup") {
  size_t arraylength = totalWorkGroups * localWorkSize;
  size_t sizeBytes{arraylength * sizeof(int)};
  // Get module and function from module
  hipModule_t Module;
  hipFunction_t Function;
  SECTION("regular fatbin") { HIP_CHECK(hipModuleLoad(&Module, fileName)); }
  SECTION("compressed fatbin") { HIP_CHECK(hipModuleLoad(&Module, fileNameCompressed)); }
  SECTION("generic target in regular fatbin") {
    if (!isGenericTargetSupported()) {
      fprintf(stderr, "Generic target test is skipped\n");
      return;
    }
    HIP_CHECK(hipModuleLoad(&Module, fileNameGenericTarget));
  }
  SECTION("generic target in compressed fatbin") {
    if (!isGenericTargetSupported()) {
      fprintf(stderr, "Generic target test is skipped\n");
      return;
    }
    HIP_CHECK(hipModuleLoad(&Module, fileNameGenericTargetCompressed));
  }
  HIP_CHECK(hipModuleGetFunction(&Function, Module, kernel_name));
  // Allocate resources
  int* A = new int[arraylength];
  REQUIRE(A != nullptr);
  int* B = new int[arraylength];
  REQUIRE(B != nullptr);
  // Inititialize data
  for (size_t i = 0; i < arraylength; i++) {
    A[i] = i;
  }
  int *Ad, *Bd;
  HIP_CHECK(hipMalloc(&Ad, sizeBytes));
  HIP_CHECK(hipMalloc(&Bd, sizeBytes));
  struct {
    void* _Ad;
    void* _Bd;
    size_t buffersize;
  } args;

  args._Ad = Ad;
  args._Bd = Bd;
  args.buffersize = arraylength;
  size_t size = sizeof(args);

  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                    HIP_LAUNCH_PARAM_END};
  // Memcpy from A to Ad
  HIP_CHECK(hipMemcpy(Ad, A, sizeBytes, hipMemcpyDefault));
  HIP_CHECK(hipExtModuleLaunchKernel(Function, arraylength, 1, 1, localWorkSize, 1, 1, 0, 0, NULL,
                                     reinterpret_cast<void**>(&config), 0));
  // Memcpy results back to host
  HIP_CHECK(hipMemcpy(B, Bd, sizeBytes, hipMemcpyDefault));
  HIP_CHECK(hipDeviceSynchronize());
  // Verify results
  for (size_t i = 0; i < arraylength; i++) {
    REQUIRE(B[i] == i);
  }
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  delete[] A;
  delete[] B;
  HIP_CHECK(hipModuleUnload(Module));
}

TEST_CASE("Unit_hipExtModuleLaunchKernel_Positive_Parameters") {
  ModuleLaunchKernelPositiveParameters<hipExtModuleLaunchKernel>();
  auto mg = ModuleGuard::InitModule("launch_kernel_module.code");
  SECTION("Pass only start event") {
    hipEvent_t start_event = nullptr;
    HIP_CHECK(hipEventCreate(&start_event));
    const auto kernel = GetKernel(mg.module(), "NOPKernel");
    HIP_CHECK(hipExtModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr,
                                       start_event, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventQuery(start_event));
    HIP_CHECK(hipEventDestroy(start_event));
  }

  SECTION("Pass only stop event") {
    hipEvent_t stop_event = nullptr;
    HIP_CHECK(hipEventCreate(&stop_event));
    const auto kernel = GetKernel(mg.module(), "NOPKernel");
    HIP_CHECK(hipExtModuleLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr,
                                       nullptr, stop_event));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipEventQuery(stop_event));
    HIP_CHECK(hipEventDestroy(stop_event));
  }
}

TEST_CASE("Unit_hipExtModuleLaunchKernel_Negative_Parameters") {
  ModuleLaunchKernelNegativeParameters<hipExtModuleLaunchKernel>(true);
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify Negative tests of hipExtModuleLaunchKernel API.
 * - Test case to verify kernel execution time of the particular kernel by using
 hipExtModuleLaunchKernel.
 * - Test case to verify hipExtModuleLaunchKernel API by disabling time flag in event creation.
 * - Test case to verify hipExtModuleLaunchKernel API's Corner Scenarios for Grid and Block
 dimensions.
 * - Test case to verify different work groups of hipExtModuleLaunchKernel API.

 * Test source
 * ------------------------
 * - catch/unit/module/hipExtModuleLaunchKernel.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */

struct gridblockDim {
  unsigned int gridX;
  unsigned int gridY;
  unsigned int gridZ;
  unsigned int blockX;
  unsigned int blockY;
  unsigned int blockZ;
};
class ModuleLaunchKernel {
  int N = 64;
  int SIZE = N * N;
  int *A, *B, *C;
  hipDeviceptr_t *Ad, *Bd;
  hipStream_t stream1, stream2;
  hipEvent_t start_event1, end_event1, start_event2, end_event2, start_timingDisabled,
      end_timingDisabled;
  hipModule_t Module;
  hipDeviceptr_t deviceGlobal;
  hipFunction_t MultKernel, SixteenSecKernel, FourSecKernel, TwoSecKernel,
      KernelandExtraParamKernel, DummyKernel;
  struct {
    int clockRate;
    void* _Ad;
    void* _Bd;
    void* _Cd;
    int _n;
  } args1, args2;
  struct {
  } args3;
  size_t size1;
  size_t size2;
  size_t size3;
  size_t deviceGlobalSize;

 public:
  void AllocateMemory();
  void DeAllocateMemory();
  void ModuleLoad();
  bool Module_Negative_tests();
  bool ExtModule_Negative_tests();
  bool ExtModule_Corner_tests();
  bool Module_WorkGroup_Test();
  bool ExtModule_KernelExecutionTime();
  bool ExtModule_ConcurencyCheck_GlobalVar(int conc_flag);
  bool ExtModule_ConcurrencyCheck_TimeVer();
  bool ExtModule_Disabled_Timingflag();
};

void ModuleLaunchKernel::AllocateMemory() {
  A = new int[N * N * sizeof(int)];
  B = new int[N * N * sizeof(int)];
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = 1;
      B[i * N + j] = 1;
    }
  }
  HIP_CHECK(hipStreamCreate(&stream1));
  HIP_CHECK(hipStreamCreate(&stream2));
  HIP_CHECK(hipMalloc(&Ad, SIZE * sizeof(int)));
  HIP_CHECK(hipMalloc(&Bd, SIZE * sizeof(int)));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&C), SIZE * sizeof(int)));
  HIP_CHECK(hipMemcpy(Ad, A, SIZE * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(Bd, B, SIZE * sizeof(int), hipMemcpyHostToDevice));
  int clkRate = 0;
#if HT_AMD
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeWallClockRate, 0));
#endif
#if HT_NVIDIA
  HIP_CHECK(hipDeviceGetAttribute(&clkRate, hipDeviceAttributeClockRate, 0));
#endif
  args1._Ad = Ad;
  args1._Bd = Bd;
  args1._Cd = C;
  args1._n = N;
  args1.clockRate = clkRate;
  args2._Ad = NULL;
  args2._Bd = NULL;
  args2._Cd = NULL;
  args2._n = 0;
  args2.clockRate = clkRate;
  size1 = sizeof(args1);
  size2 = sizeof(args2);
  size3 = 0;
  HIP_CHECK(hipEventCreate(&start_event1));
  HIP_CHECK(hipEventCreate(&end_event1));
  HIP_CHECK(hipEventCreate(&start_event2));
  HIP_CHECK(hipEventCreate(&end_event2));
  HIP_CHECK(hipEventCreateWithFlags(&start_timingDisabled, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&end_timingDisabled, hipEventDisableTiming));
}

void ModuleLaunchKernel::ModuleLoad() {
  constexpr auto matmulName = "matmul.code";
  constexpr auto matmulK = "matmulK";
  constexpr auto SixteenSec = "SixteenSecKernel";
  constexpr auto KernelandExtra = "KernelandExtraParams";
  constexpr auto FourSec = "FourSecKernel";
  constexpr auto TwoSec = "TwoSecKernel";
  constexpr auto globalDevVar = "deviceGlobal";
  constexpr auto dummyKernel = "dummyKernel";

  HIP_CHECK(hipModuleLoad(&Module, matmulName));
  HIP_CHECK(hipModuleGetFunction(&MultKernel, Module, matmulK));
  HIP_CHECK(hipModuleGetFunction(&SixteenSecKernel, Module, SixteenSec));
  HIP_CHECK(hipModuleGetFunction(&KernelandExtraParamKernel, Module, KernelandExtra));
  HIP_CHECK(hipModuleGetFunction(&FourSecKernel, Module, FourSec));
  HIP_CHECK(hipModuleGetFunction(&TwoSecKernel, Module, TwoSec));
  HIP_CHECK(hipModuleGetFunction(&DummyKernel, Module, dummyKernel));
  HIP_CHECK(hipModuleGetGlobal(&deviceGlobal, &deviceGlobalSize, Module, globalDevVar));
}

void ModuleLaunchKernel::DeAllocateMemory() {
  HIP_CHECK(hipEventDestroy(start_event1));
  HIP_CHECK(hipEventDestroy(end_event1));
  HIP_CHECK(hipEventDestroy(start_event2));
  HIP_CHECK(hipEventDestroy(end_event2));
  HIP_CHECK(hipEventDestroy(start_timingDisabled));
  HIP_CHECK(hipEventDestroy(end_timingDisabled));
  HIP_CHECK(hipStreamDestroy(stream1));
  HIP_CHECK(hipStreamDestroy(stream2));
  delete[] A;
  delete[] B;
  HIP_CHECK(hipFree(Ad));
  HIP_CHECK(hipFree(Bd));
  HIP_CHECK(hipHostFree(C));
  HIP_CHECK(hipModuleUnload(Module));
}
/*
 * In this scenario,We launch the 4 sec kernel and 2 sec kernel
 * and we fetch the event execution time of each kernel and it
 * should not exceed the execution time of that particular kernel
 */
bool ModuleLaunchKernel::ExtModule_KernelExecutionTime() {
  constexpr auto FOURSEC_KERNEL{4999};
  constexpr auto TWOSEC_KERNEL{2999};
  bool testStatus = true;
  HIP_CHECK(hipSetDevice(0));
  AllocateMemory();
  ModuleLoad();
  float time_4sec, time_2sec;

  void* config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,
                     HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipExtModuleLaunchKernel(FourSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
                                     reinterpret_cast<void**>(&config2), start_event1, end_event1,
                                     0));
  HIP_CHECK(hipExtModuleLaunchKernel(TwoSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
                                     reinterpret_cast<void**>(&config2), start_event2, end_event2,
                                     0));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipEventElapsedTime(&time_4sec, start_event1, end_event1));
  HIP_CHECK(hipEventElapsedTime(&time_2sec, start_event2, end_event2));
  if (time_4sec < FOURSEC_KERNEL && time_2sec < TWOSEC_KERNEL) {
    testStatus = true;
  } else {
    testStatus = false;
  }
  DeAllocateMemory();
  return testStatus;
}
/*
 * In this Scenario, we create events by disabling the timing flag
 * We then Launch the kernel using hipExtModuleLaunchKernel by passing
 * disabled events and try to fetch kernel execution time using
 * hipEventElapsedTime API which would fail as the flag is disabled.
 */
bool ModuleLaunchKernel::ExtModule_Disabled_Timingflag() {
  bool testStatus = true;
  AllocateMemory();
  ModuleLoad();
  hipError_t e;
  float time_2sec;
  void* config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,
                     HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipExtModuleLaunchKernel(TwoSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
                                     reinterpret_cast<void**>(&config2), start_timingDisabled,
                                     end_timingDisabled, 0));
  HIP_CHECK(hipStreamSynchronize(stream1));
  e = hipEventElapsedTime(&time_2sec, start_timingDisabled, end_timingDisabled);
  if (e == hipErrorInvalidHandle) {
    testStatus = true;
  } else {
    INFO("Event elapsed time is success when time flag is disabled \n");
    testStatus = false;
  }
  DeAllocateMemory();
  return testStatus;
}
/*
 * In this scenario , we initially create a global device variable in matmul.cpp
 * with initial value as 1 We then launch the four sec and two sec kernels and
 * try to modify the variable.
 * In case of concurrency,the variable gets updated in four sec kernel to 0x2222
 * and then the two sec kernel would be launched parallely which would again
 * modify the global variable to 0x3333
 * In case of non concurrency,the variale gets updated in four sec kernel
 * and then in two sec kernel and the value of global variable would be 0x5555
 */
bool ModuleLaunchKernel::ExtModule_ConcurencyCheck_GlobalVar(int conc_flag) {
  bool testStatus = true;
  int deviceGlobal_h = 0;
  AllocateMemory();
  ModuleLoad();
  void* config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,
                     HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipExtModuleLaunchKernel(FourSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
                                     reinterpret_cast<void**>(&config2), start_event1, end_event1,
                                     conc_flag));
  HIP_CHECK(hipExtModuleLaunchKernel(TwoSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
                                     reinterpret_cast<void**>(&config2), start_event2, end_event2,
                                     conc_flag));
  HIP_CHECK(hipStreamSynchronize(stream1));
  HIP_CHECK(hipMemcpyDtoH(&deviceGlobal_h, hipDeviceptr_t(deviceGlobal), deviceGlobalSize));
  if (conc_flag && deviceGlobal_h != 0x5555) {
    testStatus = true;
  } else if (!conc_flag && deviceGlobal_h == 0x5555) {
    testStatus = true;
  } else {
    testStatus = false;
  }
  DeAllocateMemory();
  return testStatus;
}
/* In this scenario,we initially launch 2 kernels,one is sixteen sec kernel
 * and other is matrix multiplication with non-concurrency (flag 0)
 * and we launch the same 2 kernels with concurrency flag 1. We then compare
 * the time difference between the concurrency and non currency kernels.
 * The concurrency kernel duration should be less than the non concurrency
 * duration kernels
 */
bool ModuleLaunchKernel::ExtModule_ConcurrencyCheck_TimeVer() {
  bool testStatus = true;
  AllocateMemory();
  ModuleLoad();
  int mismatch = 0;
  void* config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};
  void* config2[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args2, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size2,
                     HIP_LAUNCH_PARAM_END};
  auto start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipExtModuleLaunchKernel(SixteenSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
                                     reinterpret_cast<void**>(&config2), NULL, NULL, 0));
  HIP_CHECK(hipExtModuleLaunchKernel(MultKernel, N, N, 1, 32, 32, 1, 0, stream1, NULL,
                                     reinterpret_cast<void**>(&config1), NULL, NULL, 0));
  HIP_CHECK(hipStreamSynchronize(stream1));
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  start = std::chrono::high_resolution_clock::now();
  HIP_CHECK(hipExtModuleLaunchKernel(SixteenSecKernel, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
                                     reinterpret_cast<void**>(&config2), NULL, NULL, 1));
  HIP_CHECK(hipExtModuleLaunchKernel(MultKernel, N, N, 1, 32, 32, 1, 0, stream1, NULL,
                                     reinterpret_cast<void**>(&config1), NULL, NULL, 1));
  HIP_CHECK(hipStreamSynchronize(stream1));
  stop = std::chrono::high_resolution_clock::now();
  auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  if (!(duration2.count() < duration1.count())) {
    testStatus = false;
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (C[i * N + j] != N) mismatch++;
    }
  }
  if (mismatch) {
    testStatus = false;
  }
  DeAllocateMemory();
  return testStatus;
}
bool ModuleLaunchKernel::ExtModule_Negative_tests() {
  bool testStatus = true;
  HIP_CHECK(hipSetDevice(0));
  hipError_t err;
  AllocateMemory();
  ModuleLoad();
  void* config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args1, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};
  void* params[] = {Ad};
  // Passing nullptr to kernel function in hipExtModuleLaunchKernel API
  err = hipExtModuleLaunchKernel(nullptr, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
                                 reinterpret_cast<void**>(&config1), nullptr, nullptr, 0);
  if (err == hipSuccess) {
    INFO("hipExtModuleLaunchKernel failed nullptr to kernel function");
    testStatus = false;
  }
  // Passing Max int value to block dimensions
  err = hipExtModuleLaunchKernel(MultKernel, 1, 1, 1, std::numeric_limits<uint32_t>::max(),
                                 std::numeric_limits<uint32_t>::max(),
                                 std::numeric_limits<uint32_t>::max(), 0, stream1, NULL,
                                 reinterpret_cast<void**>(&config1), nullptr, nullptr, 0);
  if (err == hipSuccess) {
    INFO("hipExtModuleLaunchKernel failed for max values to block dimension");
    testStatus = false;
  }
  // Passing 0 as value for all dimensions
  err = hipExtModuleLaunchKernel(MultKernel, 0, 0, 0, 0, 0, 0, 0, stream1, NULL,
                                 reinterpret_cast<void**>(&config1), nullptr, nullptr, 0);
  if (err == hipSuccess) {
    INFO("hipExtModuleLaunchKernel failed for 0 as value for all dimensions");
    testStatus = false;
  }
  // Passing 0 as value for x dimension
  err = hipExtModuleLaunchKernel(MultKernel, 0, 1, 1, 0, 1, 1, 0, stream1, NULL,
                                 reinterpret_cast<void**>(&config1), nullptr, nullptr, 0);
  if (err == hipSuccess) {
    INFO("hipExtModuleLaunchKernel failed for 0 as value for x dimension");
    testStatus = false;
  }
  // Passing 0 as value for y dimension
  err = hipExtModuleLaunchKernel(MultKernel, 1, 0, 1, 1, 0, 1, 0, stream1, NULL,
                                 reinterpret_cast<void**>(&config1), nullptr, nullptr, 0);
  if (err == hipSuccess) {
    INFO("hipExtModuleLaunchKernel failed for 0 as value for y dimension");
    testStatus = false;
  }
  // Passing 0 as value for z dimension
  err = hipExtModuleLaunchKernel(MultKernel, 1, 1, 0, 1, 1, 0, 0, stream1, NULL,
                                 reinterpret_cast<void**>(&config1), nullptr, nullptr, 0);
  if (err == hipSuccess) {
    INFO("hipExtModuleLaunchKernel failed for 0 as value for z dimension");
    testStatus = false;
  }
  // Passing both kernel and extra params
  err = hipExtModuleLaunchKernel(KernelandExtraParamKernel, 1, 1, 1, 1, 1, 1, 0, stream1,
                                 reinterpret_cast<void**>(&params),
                                 reinterpret_cast<void**>(&config1), nullptr, nullptr, 0);
  if (err == hipSuccess) {
    INFO("hipExtModuleLaunchKernel fail when we pass both kernel,extra args");
    testStatus = false;
  }
  // Passing more than maxthreadsperblock to block dimensions
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));
  err = hipExtModuleLaunchKernel(MultKernel, 1, 1, 1, deviceProp.maxThreadsPerBlock + 1,
                                 deviceProp.maxThreadsPerBlock + 1,
                                 deviceProp.maxThreadsPerBlock + 1, 0, stream1, NULL,
                                 reinterpret_cast<void**>(&config1), nullptr, nullptr, 0);
  if (err == hipSuccess) {
    INFO("hipExtModuleLaunchKernel failed for max group size");
    testStatus = false;
  }
  // Block dimension X = Max Allowed + 1
  err = hipExtModuleLaunchKernel(MultKernel, 1, 1, 1, deviceProp.maxThreadsDim[0] + 1, 1, 1, 0,
                                 stream1, NULL, reinterpret_cast<void**>(&config1), nullptr,
                                 nullptr, 0);
  if (err == hipSuccess) {
    INFO("hipExtModuleLaunchKernel failed for (MaxBlockDimX + 1)");
    testStatus = false;
  }
  // Block dimension Y = Max Allowed + 1
  err = hipExtModuleLaunchKernel(MultKernel, 1, 1, 1, 1, deviceProp.maxThreadsDim[1] + 1, 1, 0,
                                 stream1, NULL, reinterpret_cast<void**>(&config1), nullptr,
                                 nullptr, 0);
  if (err == hipSuccess) {
    INFO("hipExtModuleLaunchKernel failed for (MaxBlockDimY + 1)");
    testStatus = false;
  }
  // Block dimension Z = Max Allowed + 1
  err = hipExtModuleLaunchKernel(MultKernel, 1, 1, 1, 1, 1, deviceProp.maxThreadsDim[2] + 1, 0,
                                 stream1, NULL, reinterpret_cast<void**>(&config1), nullptr,
                                 nullptr, 0);
  if (err == hipSuccess) {
    INFO("hipExtModuleLaunchKernel failed for (MaxBlockDimZ + 1)");
    testStatus = false;
  }

  // Passing invalid config data in extra params
  void* config3[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size1,
                     HIP_LAUNCH_PARAM_END};
  err = hipExtModuleLaunchKernel(MultKernel, 1, 1, 1, 1, 1, 1, 0, stream1, NULL,
                                 reinterpret_cast<void**>(&config3), nullptr, nullptr, 0);
  if (err == hipSuccess) {
    INFO("hipExtModuleLaunchKernel failed for invalid conf");
    testStatus = false;
  }
  DeAllocateMemory();
  return testStatus;
}

bool ModuleLaunchKernel::ExtModule_Corner_tests() {
  bool testStatus = true;
  HIP_CHECK(hipSetDevice(0));
  hipError_t err;
  AllocateMemory();
  ModuleLoad();
  void* config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args3, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size3,
                     HIP_LAUNCH_PARAM_END};
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));
  unsigned int maxblockX = deviceProp.maxThreadsDim[0];
  unsigned int maxblockY = deviceProp.maxThreadsDim[1];
  unsigned int maxblockZ = deviceProp.maxThreadsDim[2];
  unsigned int maxgridX = deviceProp.maxGridSize[0];
  unsigned int maxgridY = deviceProp.maxGridSize[1];
  unsigned int maxgridZ = deviceProp.maxGridSize[2];
  struct gridblockDim test[6] = {{1, 1, 1, maxblockX, 1, 1}, {1, 1, 1, 1, maxblockY, 1},
                                 {1, 1, 1, 1, 1, maxblockZ}, {maxgridX, 1, 1, 1, 1, 1},
                                 {1, maxgridY, 1, 1, 1, 1},  {1, 1, maxgridZ, 1, 1, 1}};

  for (int i = 0; i < 6; i++) {
    err = hipExtModuleLaunchKernel(DummyKernel, test[i].gridX, test[i].gridY, test[i].gridZ,
                                   test[i].blockX, test[i].blockY, test[i].blockZ, 0, stream1, NULL,
                                   reinterpret_cast<void**>(&config1), nullptr, nullptr, 0);
    if (err != hipSuccess) {
      testStatus = false;
    }
  }
  DeAllocateMemory();
  return testStatus;
}

bool ModuleLaunchKernel::Module_WorkGroup_Test() {
  bool testStatus = true;
  HIP_CHECK(hipSetDevice(0));
  hipError_t err;
  AllocateMemory();
  ModuleLoad();
  void* config1[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args3, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size3,
                     HIP_LAUNCH_PARAM_END};
  hipDeviceProp_t deviceProp;
  HIP_CHECK(hipGetDeviceProperties(&deviceProp, 0));
  double cuberootVal = cbrt(static_cast<double>(deviceProp.maxThreadsPerBlock));
  uint32_t cuberoot_floor = floor(cuberootVal);
  uint32_t cuberoot_ceil = ceil(cuberootVal);
  // Scenario: (block.x * block.y * block.z) <= Work Group Size where
  // block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
  err = hipExtModuleLaunchKernel(DummyKernel, 1, 1, 1, cuberoot_floor, cuberoot_floor,
                                 cuberoot_floor, 0, stream1, NULL,
                                 reinterpret_cast<void**>(&config1), nullptr, nullptr, 0);
  if (err != hipSuccess) {
    testStatus = false;
  }
  // Scenario: (block.x * block.y * block.z) > Work Group Size where
  // block.x < MaxBlockDimX , block.y < MaxBlockDimY and block.z < MaxBlockDimZ
  err = hipExtModuleLaunchKernel(DummyKernel, 1, 1, 1, cuberoot_ceil, cuberoot_ceil,
                                 cuberoot_ceil + 1, 0, stream1, NULL,
                                 reinterpret_cast<void**>(&config1), nullptr, nullptr, 0);
  if (err == hipSuccess) {
    testStatus = false;
  }
  DeAllocateMemory();
  return testStatus;
}

TEST_CASE("Unit_hipExtModuleLaunchKernel_Functional") {
  bool testStatus = true;
  ModuleLaunchKernel kernelLaunch;
  testStatus &= kernelLaunch.ExtModule_Negative_tests();
// Disabled below test cases as firmware currently does not support the
// concurrency in the same stream based on the flag
#if 0
  testStatus &= kernelLaunch.ExtModule_ConcurencyCheck_GlobalVar(1);
  testStatus &= kernelLaunch.ExtModule_ConcurencyCheck_GlobalVar(0);
  testStatus &= kernelLaunch.ExtModule_ConcurrencyCheck_TimeVer();
#endif
  SECTION("Kernel Execution Time") {
    testStatus &= kernelLaunch.ExtModule_KernelExecutionTime();
    REQUIRE(testStatus == true);
  }
  SECTION("Disable Time Flag") {
    testStatus &= kernelLaunch.ExtModule_Disabled_Timingflag();
    REQUIRE(testStatus == true);
  }
  SECTION("Corner Tests") {
    testStatus &= kernelLaunch.ExtModule_Corner_tests();
    REQUIRE(testStatus == true);
  }
  SECTION("WorkGroup Test") {
    testStatus &= kernelLaunch.Module_WorkGroup_Test();
    REQUIRE(testStatus == true);
  }
}
/**
 * End doxygen group KernelTest.
 * @}
 */
