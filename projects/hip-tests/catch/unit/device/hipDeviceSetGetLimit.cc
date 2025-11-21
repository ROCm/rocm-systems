/*
Copyright (c) 2023 ~ 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_process.hh>

/**
 * @addtogroup hipDeviceSetLimit hipDeviceSetLimit
 * @{
 * @ingroup DeviceTest
 * `hipDeviceSetLimit(enum hipLimit_t limit, size_t value)` -
 * Set Resource limits of current device.
 */

void DeviceSetLimitTest(hipLimit_t limit) {
  size_t old_val;
  HIP_CHECK(hipDeviceGetLimit(&old_val, limit));
  REQUIRE(old_val != 0);

  HIP_CHECK(hipDeviceSetLimit(limit, old_val + 8));

  size_t new_val;
  HIP_CHECK(hipDeviceGetLimit(&new_val, limit));
  REQUIRE(new_val >= old_val + 8);
}

/**
 * Test Description
 * ------------------------
 *  - Basic set-get test for `hipLimitStackSize`.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipDeviceSetLimit_Positive_StackSize") { DeviceSetLimitTest(hipLimitStackSize); }

#if HT_NVIDIA

__device__ __managed__ bool stop = false;

/**
 * Test Description
 * ------------------------
 *  - Basic set-get test for `hipLimitPrintfFifoSize`.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipDeviceSetLimit_Positive_PrintfFifoSize") {
  DeviceSetLimitTest(hipLimitPrintfFifoSize);
}

__global__ void PrintfKernel() {
  while (!stop) printf("");
}

/**
 * Test Description
 * ------------------------
 *  - Tests scenario where we try to set `hipLimitPrintfFifoSize` while a kernel that calls
 * `printf()` is running.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipDeviceSetLimit_Negative_PrintfFifoSize") {
  PrintfKernel<<<1, 1>>>();
  HIP_CHECK_ERROR(hipDeviceSetLimit(hipLimitPrintfFifoSize, 1024), hipErrorInvalidValue);
  stop = true;
  HIP_CHECK(hipDeviceSynchronize());
  stop = false;
}

/**
 * Test Description
 * ------------------------
 *  - Basic set-get test for `hipLimitMallocHeapSize`.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipDeviceSetLimit_Positive_MallocHeapSize") {
  DeviceSetLimitTest(hipLimitMallocHeapSize);
}

__global__ void MallocKernel() {
  while (!stop) free(malloc(1));
}

/**
 * Test Description
 * ------------------------
 *  - Tests scenario where we try to set `hipLimitMallocHeapSize` while a kernel that calls
 * `malloc()` and `free()` is running.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipDeviceSetLimit_Negative_MallocHeapSize") {
  MallocKernel<<<1, 1>>>();
  HIP_CHECK_ERROR(hipDeviceSetLimit(hipLimitMallocHeapSize, 1024), hipErrorInvalidValue);
  stop = true;
  HIP_CHECK(hipDeviceSynchronize());
  stop = false;
}

#endif

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipDeviceSetLimit`.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.3
 */
TEST_CASE("Unit_hipDeviceSetLimit_Negative_Parameters") {
#if HT_AMD
  HIP_CHECK_ERROR(hipDeviceSetLimit(static_cast<hipLimit_t>(-1), 1024), hipErrorUnsupportedLimit);
#else
  HIP_CHECK_ERROR(hipDeviceSetLimit(static_cast<hipLimit_t>(-1), 1024), hipErrorInvalidValue);
#endif
}

/**
 * End doxygen group hipDeviceSetLimit.
 * @}
 */

/**
 * @addtogroup hipDeviceGetLimit hipDeviceGetLimit
 * @{
 * @ingroup DeviceTest
 * `hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit)` -
 * Get Resource limits of current device.
 */

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipDeviceGetLimit`.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetLimit_Negative_Parameters") {
  SECTION("nullptr") {
    HIP_CHECK_ERROR(hipDeviceGetLimit(nullptr, hipLimitStackSize), hipErrorInvalidValue);
  }

  SECTION("unsupported limit") {
    size_t val;
#if HT_AMD
    HIP_CHECK_ERROR(hipDeviceGetLimit(&val, static_cast<hipLimit_t>(-1)), hipErrorUnsupportedLimit);
#else
    HIP_CHECK_ERROR(hipDeviceGetLimit(&val, static_cast<hipLimit_t>(-1)), hipErrorInvalidValue);
#endif
  }
}

#if HT_AMD
bool isSetScratchLimitSupported() {
#if __linux__
  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  std::cout << "Device Id = " << deviceId << " props.major = " << props.major
            << " props.minor = " << props.minor << std::endl;
  return ((props.major == 9 && props.minor >= 4) || (props.major == 12 && props.minor >= 5))
             ? true
             : false;
#else
  std::cout << "Only Supported for Linux" << std::endl;
  return false;
#endif
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following Negative scenarios
 *  - 1) Call hipDeviceGetLimit, hipDeviceSetLimit with hipLimitRange
 *  - 2) Call hipDeviceSetLimit with hipExtLimitScratchMin
 *  - 3) Call hipDeviceSetLimit with hipExtLimitScratchMax
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDeviceGetSetLimit_Scratch_Negative") {
  size_t value = 0;
  SECTION("With hipLimitRange") {
    HIP_CHECK_ERROR(hipDeviceGetLimit(&value, hipLimitRange), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipDeviceSetLimit(hipLimitRange, value), hipErrorInvalidValue);
  }

  SECTION("Set the minimum value") {
    HIP_CHECK_ERROR(hipDeviceSetLimit(hipExtLimitScratchMin, value), hipErrorUnsupportedLimit);
  }

  SECTION("Set the Maximum value") {
    HIP_CHECK_ERROR(hipDeviceSetLimit(hipExtLimitScratchMax, value), hipErrorUnsupportedLimit);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenarios
 *  - 1) Set the Current Scratch limit to Minimum(0)
 *  - 2) Get the Maximum Scratch limit and set that as Current Scratch limit
 *  - Set back original Scratch limit value
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDeviceGetSetLimit_Scratch_SetMinAndMaxAsCurrent") {
  if (!isSetScratchLimitSupported()) {
    HipTest::HIP_SKIP_TEST(
        "Set Scratch Limit Not Supported on Current Device."
        " Only Mi300+ and Linux supports");
    return;
  }

  size_t scratchLimitCurrent = 0;
  HIP_CHECK(hipDeviceGetLimit(&scratchLimitCurrent, hipExtLimitScratchCurrent));

  SECTION("Set the hipExtLimitScratchCurrent to hipExtLimitScratchMin") {
    size_t scratchLimitMin = 0;
    HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, scratchLimitMin));
  }

  SECTION("Set the hipExtLimitScratchCurrent to hipExtLimitScratchMax") {
    size_t scratchLimitMax = 0;
    HIP_CHECK(hipDeviceGetLimit(&scratchLimitMax, hipExtLimitScratchMax));
    HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, scratchLimitMax));
  }

  // Set back the original value
  HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, scratchLimitCurrent));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenarios
 *  - 1) Set the Current Scratch limit to 1KB less than Current Scratch limit
 *  - 2) Set the Current Scratch limit to 1KB greater than Current Scratch limit
 *  - Set back original Scratch limit value
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDeviceGetSetLimit_Scratch_DecreaseIncrease") {
  if (!isSetScratchLimitSupported()) {
    HipTest::HIP_SKIP_TEST(
        "Set Scratch Limit Not Supported on Current Device."
        " Only Mi300+ and Linux supports");
    return;
  }

  // Get the current scratch
  size_t orgValue = 0;
  HIP_CHECK(hipDeviceGetLimit(&orgValue, hipExtLimitScratchCurrent));

  SECTION("Decrease from the current value") {
    size_t setValue = orgValue - 1024;
    setValue = (setValue < 0) ? 0 : setValue;

    HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, setValue));

    size_t getValue = 0;
    HIP_CHECK(hipDeviceGetLimit(&getValue, hipExtLimitScratchCurrent));
    REQUIRE(getValue == setValue);
  }

  SECTION("Increase from the current value") {
    size_t setValue = orgValue + 1024;

    size_t max = 0;
    HIP_CHECK(hipDeviceGetLimit(&max, hipExtLimitScratchMax));
    setValue = (setValue > max) ? (max * 0.9) : setValue;

    HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, setValue));

    size_t getValue = 0;
    HIP_CHECK(hipDeviceGetLimit(&getValue, hipExtLimitScratchCurrent));
    REQUIRE(getValue == setValue);
  }

  // Set back original value
  HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, orgValue));
}

static constexpr size_t SIZE = 4 * 1024;
static constexpr size_t SIZE_BYTES = SIZE * sizeof(int);
static constexpr int N_BYTES = sizeof(int);

/*
 * Kernel function uses the scratch memory and fill the value
 */
__global__ void addOneKernelUseScratch(int* arr) {
  int localArr[SIZE];
  for (int i = 0; i < SIZE; i++) {
    localArr[i] = i;
  }

  int sum = 0;
  for (int i = 0; i < SIZE; i += 1) {
    sum += localArr[i];
  }

  *arr = sum;
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks following scenario
 *  - 1) Get the Original Current Scratch limit
 *  - 2) Set the Current Scratch limit to 16KB
 *  - 3) Get the Current Scratch limit and validate it
 *  - 4) Launch the kernel with using scratch
 *  - 5) Validate the kernel output
 *  - 6) Set back original Scratch limit value
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDeviceGetSetLimit_Scratch_SetBeforeKernelLaunch") {
  if (!isSetScratchLimitSupported()) {
    HipTest::HIP_SKIP_TEST(
        "Set Scratch Limit Not Supported on Current Device."
        " Only Mi300+ and Linux supports");
    return;
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, sizeof(int)));
  REQUIRE(devMem != nullptr);

  size_t orgValue = 0;
  HIP_CHECK(hipDeviceGetLimit(&orgValue, hipExtLimitScratchCurrent));

  HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, SIZE_BYTES));

  size_t getValue = 0;
  HIP_CHECK(hipDeviceGetLimit(&getValue, hipExtLimitScratchCurrent));
  REQUIRE(getValue == SIZE_BYTES);

  addOneKernelUseScratch<<<1, 1, 0, stream>>>(devMem);
  HIP_CHECK(hipStreamSynchronize(stream));

  int hostMem = 0, expectedValue = ((SIZE - 1) * (SIZE)) / 2;
  HIP_CHECK(hipMemcpy(&hostMem, devMem, N_BYTES, hipMemcpyDeviceToHost));
  REQUIRE(hostMem == expectedValue);

  // Set back original value
  HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, orgValue));

  HIP_CHECK(hipFree(devMem));
  HIP_CHECK(hipStreamDestroy(stream));
}

/*
 * This helper funtion perform the below operations,
 * 1) Get the Minimum Scratch limit and validate
 * 2) Get the Maximum Scratch limit and validate
 * 3) Get the Current Scratch limit and validate
 * 4) Set the Current Scratch limit to 50% of Maximum Scratch limit
 * 5) Get the Current Scratch limit and validate it
 * 6) Set the Current Scratch limit to original value
 */
void getMinMaxCurrentAndSetCurrent() {
  size_t min = 0, max = 0, orgCurrent = 0;

  HIP_CHECK(hipDeviceGetLimit(&min, hipExtLimitScratchMin));
  REQUIRE(min == 0);

  HIP_CHECK(hipDeviceGetLimit(&max, hipExtLimitScratchMax));
  REQUIRE(max > 0);

  HIP_CHECK(hipDeviceGetLimit(&orgCurrent, hipExtLimitScratchCurrent));
  REQUIRE(orgCurrent >= 0);

  size_t setCurrent = 0.5 * max;
  HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, setCurrent));

  size_t getCurrent = 0;
  HIP_CHECK(hipDeviceGetLimit(&getCurrent, hipExtLimitScratchCurrent));
  REQUIRE(getCurrent == setCurrent);

  HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, orgCurrent));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks the behaviour of
 *  - hipDeviceGetLimit with hipExtLimitScratchMin, hipExtLimitScratchMax,
 *  - hipExtLimitScratchCurrent and hipDeviceSetLimit with
 *  - hipExtLimitScratchCurrent in all the devices.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDeviceGetSetLimit_Scratch_MultiDevice") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  if (!isSetScratchLimitSupported()) {
    HipTest::HIP_SKIP_TEST(
        "Set Scratch Limit Not Supported on Current Device."
        " Only Mi300+ and Linux supports");
    return;
  }

  for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
    HIP_CHECK(hipSetDevice(deviceId));
    getMinMaxCurrentAndSetCurrent();
  }
  HIP_CHECK(hipSetDevice(0));
}

/**
 * Test Description
 * ------------------------
 *  - This testcase checks the behaviour of
 *  - hipDeviceGetLimit with hipExtLimitScratchMin, hipExtLimitScratchMax,
 *  - hipExtLimitScratchCurrent and hipDeviceSetLimit with
 *  - hipExtLimitScratchCurrent in the thread.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDeviceGetSetLimit_Scratch_InThread") {
  if (!isSetScratchLimitSupported()) {
    HipTest::HIP_SKIP_TEST(
        "Set Scratch Limit Not Supported on Current Device."
        " Only Mi300+ and Linux supports");
    return;
  }

  std::thread threadObj(getMinMaxCurrentAndSetCurrent);
  threadObj.join();
}

/**
 * Test Description
 * ------------------------
 *  - This test creates the child process and checks the functionality of
 *  - hipDeviceGetLimit with hipExtLimitScratchMin, hipExtLimitScratchMax,
 *  - hipExtLimitScratchCurrent and hipDeviceSetLimit with
 *  - hipExtLimitScratchCurrent in the child process.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDeviceGetSetLimit_Scratch_InChildProcess") {
  if (!isSetScratchLimitSupported()) {
    HipTest::HIP_SKIP_TEST(
        "Set Scratch Limit Not Supported on Current Device."
        " Only Mi300+ and Linux supports");
    return;
  }

  hip::SpawnProc proc("hipDeviceSetGetScratchExe", true);
  REQUIRE(proc.run() == 0);
}

/*
 * Local function to set the given current scratch limit
 */
void setScratchCurrent(size_t setValue) {
  HIP_CHECK(hipDeviceSetLimit(hipExtLimitScratchCurrent, setValue));
}

/*
 * Local function to get current scratch limit and validate with
 * given reference value
 */
void getScratchCurrent(size_t checkValue) {
  size_t getCurrent = 0;
  HIP_CHECK(hipDeviceGetLimit(&getCurrent, hipExtLimitScratchCurrent));
  REQUIRE(getCurrent == checkValue);
}

/**
 * Test Description
 * ------------------------
 *  - This test tests the below scenario
 *  - 1) Get the Maximum Scratch limit
 *  - 2) Get the Current Scratch limit
 *  - 3) Set the Current Scratch limit to 50% of Maximum Scratch limit in thread
 *  - 4) Get the Current Scratch limit and validate it in thread
 *  - 5) Set the Current Scratch limit to original value in thread
 *  - 6) Get the Current Scratch limit and validate it in thread
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSetGetLimit.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.5
 */
TEST_CASE("Unit_hipDeviceGetSetLimit_Scratch_SetGetThreads") {
  if (!isSetScratchLimitSupported()) {
    HipTest::HIP_SKIP_TEST(
        "Set Scratch Limit Not Supported on Current Device."
        " Only Mi300+ and Linux supports");
    return;
  }

  size_t max = 0, orgCurrent = 0;

  HIP_CHECK(hipDeviceGetLimit(&max, hipExtLimitScratchMax));
  REQUIRE(max > 0);

  HIP_CHECK(hipDeviceGetLimit(&orgCurrent, hipExtLimitScratchCurrent));
  REQUIRE(orgCurrent >= 0);

  std::thread setThread1(setScratchCurrent, max * 0.5);
  setThread1.join();

  std::thread getThread1(getScratchCurrent, max * 0.5);
  getThread1.join();

  std::thread setThread2(setScratchCurrent, orgCurrent);
  setThread2.join();

  std::thread getThread2(getScratchCurrent, orgCurrent);
  getThread2.join();
}

#endif
