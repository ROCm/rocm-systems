/*
         Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

/*
unsafeAtomicAdd Scenarios with hipRTC:
1. FineGrainMemory with -m-nounsafe-fp-atomics flag
2. FineGrainMemory without compilation flag
3. FineGrainMemory without -munsafe-fp-atomics flag
4. CoarseGrainMemory with -m-nounsafe-fp-atomics flag
5. CoarseGrainMemory without compilation flag
6. CoarseGrainMemory without -munsafe-fp-atomics flag
*/

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_features.hh>
#include <hip/hiprtc.h>
#include <type_traits>
#include <vector>
#define INCREMENT_VAL 10
#define INITIAL_VAL 5

static constexpr auto fkernel{
    R"(
extern "C"
__global__ void AtomicCheck(float* Ad, float *result) {
*result = unsafeAtomicAdd(Ad, 10);
}
)"};

static constexpr auto dkernel{
    R"(
extern "C"
__global__ void AtomicCheck(double* Ad, double *result) {
*result = unsafeAtomicAdd(Ad, 10);
}
)"};

// Structure to hold module and function handles
struct RTCKernel {
  hipModule_t module;
  hipFunction_t function;
};

// Helper function to compile and load RTC kernel
template <typename TestType>
static RTCKernel compileAndLoadKernel(hipDeviceProp_t& props, const char* extraOption = nullptr) {
  hiprtcProgram prog;
  if (std::is_same<TestType, float>::value) {
    hiprtcCreateProgram(&prog, fkernel, "kernel.cu", 0, nullptr, nullptr);
  } else {
    hiprtcCreateProgram(&prog, dkernel, "kernel.cu", 0, nullptr, nullptr);
  }

  std::string sarg = std::string("--gpu-architecture=") + props.gcnArchName;
  std::vector<const char*> options;
  options.push_back(sarg.c_str());
  if (extraOption != nullptr) {
    options.push_back(extraOption);
  }

  hiprtcResult compileResult{hiprtcCompileProgram(prog, options.size(), options.data())};
  size_t logSize;
  HIPRTC_CHECK(hiprtcGetProgramLogSize(prog, &logSize));
  if (logSize) {
    std::string log(logSize, '\0');
    HIPRTC_CHECK(hiprtcGetProgramLog(prog, &log[0]));
    INFO(log);
  }

  REQUIRE(compileResult == HIPRTC_SUCCESS);
  size_t codeSize;
  HIPRTC_CHECK(hiprtcGetCodeSize(prog, &codeSize));

  std::vector<char> code(codeSize);
  HIPRTC_CHECK(hiprtcGetCode(prog, code.data()));
  HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  RTCKernel kernel;
  HIP_CHECK(hipModuleLoadData(&kernel.module, code.data()));
  HIP_CHECK(hipModuleGetFunction(&kernel.function, kernel.module, "AtomicCheck"));
  return kernel;
}

// Helper function to run test with Coherent memory
template <typename TestType>
static void runCoherentTest(hipFunction_t f_kernel, const std::string& gfxName) {
  TestType *A_h, *result;
  TestType *A_d, *result_d;
  HIP_CHECK(
      hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(TestType), hipHostMallocCoherent));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result), sizeof(TestType),
                          hipHostMallocCoherent));
  A_h[0] = INITIAL_VAL;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), A_h, 0));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&result_d), result, 0));
  struct {
    TestType* p;
    TestType* result;
  } args_f{A_d, result_d};
  auto size = sizeof(args_f);
  void* config_d[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_f, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(f_kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, config_d));
  HIP_CHECK(hipDeviceSynchronize());
  if ((gfxName == "gfx90a" || gfxName.find("gfx90a:")) == 0) {
    REQUIRE(A_h[0] == INITIAL_VAL);
    REQUIRE(*result == 0);
  } else {
    REQUIRE(A_h[0] == INITIAL_VAL + INCREMENT_VAL);
    REQUIRE(*result == INITIAL_VAL);
  }
  HIP_CHECK(hipHostFree(A_h));
  HIP_CHECK(hipHostFree(result));
}

// Helper function to run test with NonCoherent memory
template <typename TestType>
static void runNonCoherentTest(hipFunction_t f_kernel) {
  TestType *A_h, *result;
  TestType *A_d, *result_d;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), sizeof(TestType),
                          hipHostMallocNonCoherent));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result), sizeof(TestType)));
  A_h[0] = INITIAL_VAL;
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&A_d), A_h, 0));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&result_d), result, 0));
  struct {
    TestType* p;
    TestType* result;
  } args_f{A_d, result_d};
  auto size = sizeof(args_f);
  void* config_d[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_f, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
                      HIP_LAUNCH_PARAM_END};
  HIP_CHECK(hipModuleLaunchKernel(f_kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, config_d));
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(A_h[0] == INITIAL_VAL + INCREMENT_VAL);
  REQUIRE(*result == INITIAL_VAL);
  HIP_CHECK(hipHostFree(A_h));
  HIP_CHECK(hipHostFree(result));
}

// Helper function to run all unsafeAtomicAdd RTC tests for a specific type
template <typename TestType>
static void runUnsafeAtomicAddRTCTest() {
  int device = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string gfxName(props.gcnArchName);

  if (CheckIfFeatSupported(CTFeatures::CT_FEATURE_FINEGRAIN_HWSUPPORT, gfxName)) {
    if (props.canMapHostMemory != 1) {
      SUCCEED("Does not support HostPinned Memory");
    } else {
      SECTION("Coherent with -mno-unsafe-fp-atomics flag") {
        RTCKernel kernel = compileAndLoadKernel<TestType>(props, "-mno-unsafe-fp-atomics");
        runCoherentTest<TestType>(kernel.function, gfxName);
        HIP_CHECK(hipModuleUnload(kernel.module));
      }

      SECTION("Coherent with -munsafe-fp-atomics flag") {
        RTCKernel kernel = compileAndLoadKernel<TestType>(props, "-munsafe-fp-atomics");
        runCoherentTest<TestType>(kernel.function, gfxName);
        HIP_CHECK(hipModuleUnload(kernel.module));
      }

      SECTION("Coherent without flag") {
        RTCKernel kernel = compileAndLoadKernel<TestType>(props);
        runCoherentTest<TestType>(kernel.function, gfxName);
        HIP_CHECK(hipModuleUnload(kernel.module));
      }

      SECTION("NonCoherent with -mno-unsafe-fp-atomics flag") {
        RTCKernel kernel = compileAndLoadKernel<TestType>(props, "-mno-unsafe-fp-atomics");
        runNonCoherentTest<TestType>(kernel.function);
        HIP_CHECK(hipModuleUnload(kernel.module));
      }

      SECTION("NonCoherent with -munsafe-fp-atomics flag") {
        RTCKernel kernel = compileAndLoadKernel<TestType>(props, "-munsafe-fp-atomics");
        runNonCoherentTest<TestType>(kernel.function);
        HIP_CHECK(hipModuleUnload(kernel.module));
      }

      SECTION("NonCoherent without flag") {
        RTCKernel kernel = compileAndLoadKernel<TestType>(props);
        runNonCoherentTest<TestType>(kernel.function);
        HIP_CHECK(hipModuleUnload(kernel.module));
      }
    }
  } else {
    SUCCEED(
        "Memory model feature is only supported for gfx90a, gfx942, gfx950,"
        "Hence skipping the testcase for this GPU "
        << device);
  }
}

TEST_CASE("Unit_unsafeAtomicAdd_RTC") {
  SECTION("float") { runUnsafeAtomicAddRTCTest<float>(); }
  SECTION("double") { runUnsafeAtomicAddRTCTest<double>(); }
}
