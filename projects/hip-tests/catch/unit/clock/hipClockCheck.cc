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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip/hip_ext.h>
#include <dlfcn.h>

/**
 * @addtogroup clock clock
 * @{
 * @ingroup DeviceLanguageTest
 * Contains unit tests for clock, clock64 and wall_clock64 APIs
 */

__global__ void kernel_c64(int clock_rate, uint64_t wait_t) {
  uint64_t start = clock64() / clock_rate, cur = 0;  // in ms
  do {
    cur = clock64() / clock_rate - start;
  } while (cur < wait_t);
}

__global__ void kernel_c(int clock_rate, uint64_t wait_t) {
  uint64_t start = clock() / clock_rate, cur = 0;  // in ms
  do {
    cur = clock() / clock_rate - start;
  } while (cur < wait_t);
}

__global__ void kernel_wc64(int clock_rate, uint64_t wait_t) {
  uint64_t start = wall_clock64() / clock_rate, cur = 0;  // in ms
  do {
    cur = wall_clock64() / clock_rate - start;
  } while (cur < wait_t);
}

bool verify_time_execution(float ratio, float time1, float time2, float expected_time1,
                           float expected_time2) {
  bool test_status = false;

#if (HT_WIN == 1)
  if (time1 > ratio * expected_time1 && time2 > ratio * expected_time2) {
#else
  if (fabs(time1 - expected_time1) < (ratio * expected_time1) &&
      fabs(time2 - expected_time2) < (ratio * expected_time2)) {
#endif
    INFO("Succeeded: Expected Vs Actual: Kernel1 - " << expected_time1 << " Vs " << time1
                                                     << ", Kernel2 - " << expected_time2 << " Vs "
                                                     << time2);
    test_status = true;
  } else {
    INFO("Failed: Expected Vs Actual: Kernel1 -" << expected_time1 << " Vs " << time1
                                                 << ", Kernel2 - " << expected_time2 << " Vs "
                                                 << time2);
    test_status = false;
  }
  return test_status;
}

/*
 * Launching kernel1 and kernel2 and then we try to
 * get the event elapsed time of each kernel using the start and
 * end events.The event elapsed time should return us the kernel
 * execution time for that particular kernel
 */
bool kernel_time_execution(void (*kernel)(int, uint64_t), int clock_rate, uint64_t expected_time1,
                           uint64_t expected_time2) {
  hipStream_t stream;
  hipEvent_t start_event1, end_event1, start_event2, end_event2;
  float time1 = 0, time2 = 0;
  HIP_CHECK(hipEventCreate(&start_event1));
  HIP_CHECK(hipEventCreate(&end_event1));
  HIP_CHECK(hipEventCreate(&start_event2));
  HIP_CHECK(hipEventCreate(&end_event2));
  HIP_CHECK(hipStreamCreate(&stream));
  hipExtLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, stream, start_event1, end_event1, 0,
                        clock_rate, expected_time1);
  hipExtLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, stream, start_event2, end_event2, 0,
                        clock_rate, expected_time2);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipEventElapsedTime(&time1, start_event1, end_event1));
  HIP_CHECK(hipEventElapsedTime(&time2, start_event2, end_event2));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipEventDestroy(start_event1));
  HIP_CHECK(hipEventDestroy(end_event1));
  HIP_CHECK(hipEventDestroy(start_event2));
  HIP_CHECK(hipEventDestroy(end_event2));

#if HT_WIN == 1
  float ratio = 1.0f;
#else
  float ratio = kernel == kernel_wc64 ? 0.01 : 0.5;
#endif

  return verify_time_execution(ratio, time1, time2, expected_time1, expected_time2);
}

 typedef enum {
    AMDSMI_STATUS_SUCCESS = 0,              //!< Call succeeded
    // Library usage errors
    AMDSMI_STATUS_INVAL = 1,                //!< Invalid parameters
    AMDSMI_STATUS_NOT_SUPPORTED = 2,        //!< Command not supported
    AMDSMI_STATUS_NOT_YET_IMPLEMENTED = 3,  //!< Not implemented yet
    AMDSMI_STATUS_FAIL_LOAD_MODULE = 4,     //!< Fail to load lib
    AMDSMI_STATUS_FAIL_LOAD_SYMBOL = 5,     //!< Fail to load symbol
    AMDSMI_STATUS_DRM_ERROR = 6,            //!< Error when call libdrm
    AMDSMI_STATUS_API_FAILED = 7,           //!< API call failed
    AMDSMI_STATUS_TIMEOUT = 8,              //!< Timeout in API call
    AMDSMI_STATUS_RETRY = 9,                //!< Retry operation
    AMDSMI_STATUS_NO_PERM = 10,             //!< Permission Denied
    AMDSMI_STATUS_INTERRUPT = 11,           //!< An interrupt occurred during execution of function
    AMDSMI_STATUS_IO = 12,                  //!< I/O Error
    AMDSMI_STATUS_ADDRESS_FAULT = 13,       //!< Bad address
    AMDSMI_STATUS_FILE_ERROR = 14,          //!< Problem accessing a file
    AMDSMI_STATUS_OUT_OF_RESOURCES = 15,    //!< Not enough memory
    AMDSMI_STATUS_INTERNAL_EXCEPTION = 16,  //!< An internal exception was caught
    AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS = 17, //!< The provided input is out of allowable or safe range
    AMDSMI_STATUS_INIT_ERROR = 18,          //!< An error occurred when initializing internal data structures
    AMDSMI_STATUS_REFCOUNT_OVERFLOW = 19,   //!< An internal reference counter exceeded INT32_MAX
    AMDSMI_STATUS_DIRECTORY_NOT_FOUND = 20, //!< Error when a directory is not found, maps to ENOTDIR
    // Processor related errors
    AMDSMI_STATUS_BUSY = 30,                //!< Processor busy
    AMDSMI_STATUS_NOT_FOUND = 31,           //!< Processor Not found
    AMDSMI_STATUS_NOT_INIT = 32,            //!< Processor not initialized
    AMDSMI_STATUS_NO_SLOT = 33,             //!< No more free slot
    AMDSMI_STATUS_DRIVER_NOT_LOADED = 34,   //!< Processor driver not loaded
    // Data and size errors
    AMDSMI_STATUS_MORE_DATA = 39,           //!< There is more data than the buffer size the user passed
    AMDSMI_STATUS_NO_DATA = 40,             //!< No data was found for a given input
    AMDSMI_STATUS_INSUFFICIENT_SIZE = 41,   //!< Not enough resources were available for the operation
    AMDSMI_STATUS_UNEXPECTED_SIZE = 42,     //!< An unexpected amount of data was read
    AMDSMI_STATUS_UNEXPECTED_DATA = 43,     //!< The data read or provided to function is not what was expected
    //esmi errors
    AMDSMI_STATUS_NON_AMD_CPU = 44,         //!< System has different cpu than AMD
    AMDSMI_STATUS_NO_ENERGY_DRV = 45,       //!< Energy driver not found
    AMDSMI_STATUS_NO_MSR_DRV = 46,          //!< MSR driver not found
    AMDSMI_STATUS_NO_HSMP_DRV = 47,         //!< HSMP driver not found
    AMDSMI_STATUS_NO_HSMP_SUP = 48,         //!< HSMP not supported
    AMDSMI_STATUS_NO_HSMP_MSG_SUP = 49,     //!< HSMP message/feature not supported
    AMDSMI_STATUS_HSMP_TIMEOUT = 50,        //!< HSMP message timed out
    AMDSMI_STATUS_NO_DRV = 51,              //!< No Energy and HSMP driver present
    AMDSMI_STATUS_FILE_NOT_FOUND = 52,      //!< file or directory not found
    AMDSMI_STATUS_ARG_PTR_NULL = 53,        //!< Parsed argument is invalid
    AMDSMI_STATUS_AMDGPU_RESTART_ERR = 54,  //!< AMDGPU restart failed
    AMDSMI_STATUS_SETTING_UNAVAILABLE = 55, //!< Setting is not available
    AMDSMI_STATUS_CORRUPTED_EEPROM = 56,    //!< EEPROM is corrupted
    // General errors
    AMDSMI_STATUS_MAP_ERROR = 0xFFFFFFFE,     //!< The internal library error did not map to a status code
    AMDSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF  //!< An unknown error occurred
} amdsmi_status_t;

int getEngineFreq()
{
  typedef void *amdsmi_processor_handle;
  typedef void *amdsmi_socket_handle;

  void* lib_rocm_smi_hdl;
  amdsmi_status_t status;

  typedef struct {
    uint32_t clk;            //!< In MHz
    uint32_t min_clk;        //!< In MHz
    uint32_t max_clk;        //!< In MHz
    uint8_t clk_locked;      //!< True/False
    uint8_t clk_deep_sleep;  //!< True/False
    uint32_t reserved[4];
  } amdsmi_clk_info_t;

  typedef enum {
    AMDSMI_CLK_TYPE_SYS = 0x0,  //!< System clock
    AMDSMI_CLK_TYPE_FIRST = AMDSMI_CLK_TYPE_SYS,
    AMDSMI_CLK_TYPE_GFX = AMDSMI_CLK_TYPE_SYS,  //!< Graphics clock
    AMDSMI_CLK_TYPE_DF,         /**< Data Fabric clock (for ASICs
                                     running on a separate clock) */
    AMDSMI_CLK_TYPE_DCEF,       /**< Display Controller Engine Front clock,
                                     timing/bandwidth signals to display */
    AMDSMI_CLK_TYPE_SOC,        //!< System On Chip clock, integrated circuit frequency
    AMDSMI_CLK_TYPE_MEM,        //!< Memory clock speed, system operating frequency
    AMDSMI_CLK_TYPE_PCIE,       //!< PCI Express clock, high bandwidth peripherals
    AMDSMI_CLK_TYPE_VCLK0,      //!< Video 0 clock, video processing units
    AMDSMI_CLK_TYPE_VCLK1,      //!< Video 1 clock, video processing units
    AMDSMI_CLK_TYPE_DCLK0,      //!< Display 1 clock, timing signals for display output
    AMDSMI_CLK_TYPE_DCLK1,      //!< Display 2 clock, timing signals for display output
    AMDSMI_CLK_TYPE__MAX = AMDSMI_CLK_TYPE_DCLK1
  } amdsmi_clk_type_t;

  amdsmi_clk_info_t clk_info;
  uint32_t gpu_count;

  lib_rocm_smi_hdl = dlopen("libamd_smi.so", RTLD_LAZY);
  REQUIRE(lib_rocm_smi_hdl);

  void* fnsym = dlsym(lib_rocm_smi_hdl, "amdsmi_init");
  auto fninit = reinterpret_cast<amdsmi_status_t (*)(uint64_t)>(fnsym);
  REQUIRE(fnsym);

  fnsym = dlsym(lib_rocm_smi_hdl, "amdsmi_get_socket_handles");
  auto fnget_socket_handles = reinterpret_cast<amdsmi_status_t (*)(uint32_t*, amdsmi_socket_handle*)>(fnsym);
  REQUIRE(fnsym);

  fnsym = dlsym(lib_rocm_smi_hdl, "amdsmi_get_processor_handles");
  auto fnget_processor_handles = reinterpret_cast<amdsmi_status_t (*)(amdsmi_socket_handle, uint32_t*, amdsmi_processor_handle*)>(fnsym);
  REQUIRE(fnsym);

  fnsym = dlsym(lib_rocm_smi_hdl, "amdsmi_get_clock_info");
  REQUIRE(fnsym);
  auto fnget_clock_info = reinterpret_cast<amdsmi_status_t (*)(amdsmi_processor_handle, amdsmi_clk_type_t, amdsmi_clk_info_t*)>(fnsym);

  fnsym = dlsym(lib_rocm_smi_hdl, "amdsmi_shut_down");
  auto fnshut_down = reinterpret_cast<amdsmi_status_t (*)()>(fnsym);
  REQUIRE(fnsym);

  status = fninit(1 << 1);
  REQUIRE(AMDSMI_STATUS_SUCCESS == status);
  uint32_t socket_count = 0;

  // Get the socket count available in the system.
  status = fnget_socket_handles(&socket_count, nullptr);
  REQUIRE(AMDSMI_STATUS_SUCCESS == status);

  std::vector<amdsmi_socket_handle> sockets(socket_count);
  status = fnget_socket_handles(&socket_count, &sockets[0]);

  // just get number of processors first
  status = fnget_processor_handles(sockets[0], &gpu_count, nullptr);
  REQUIRE(AMDSMI_STATUS_SUCCESS == status);

  std::vector<amdsmi_processor_handle> processors(gpu_count);
  status = fnget_processor_handles(sockets[0], &gpu_count, &processors[0]);
  REQUIRE(AMDSMI_STATUS_SUCCESS == status);
  status = fnget_clock_info(processors[0], AMDSMI_CLK_TYPE_GFX, &clk_info);
  REQUIRE(AMDSMI_STATUS_SUCCESS == status);
  printf("max_clk: %d\n", clk_info.max_clk);
  fnshut_down();
  dlclose(lib_rocm_smi_hdl);
  return 0;
}

/**
 * Test Description
 * ------------------------
 *  - Launches two kernels that run for a specified amount of time passed as a kernel argument by
 * using device function clock64. Kernel execution time is calculated through elapsed time between
 * the start and end event, and calculated time is compared with passed time values.
 * Test source
 * ------------------------
 *  - catch/unit/clock/hipClockCheck.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipClock64_Positive_Basic") {
  HIP_CHECK(hipSetDevice(0));
  int clock_rate = 0;  // in kHz
  HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeClockRate, 0));
  if (clock_rate == 0) {
    HipTest::HIP_SKIP_TEST("hipDeviceAttributeClockRate returns 0");
    return;
  }
  if (IsGfx11()) {
    HipTest::HIP_SKIP_TEST("Issue with clock64() function on gfx11 devices!");
    return;
  }

  getEngineFreq();
  const auto expected_time1 = GENERATE(1000, 1500, 2000);
  const auto expected_time2 = expected_time1 / 2;

  REQUIRE(kernel_time_execution(kernel_c64, clock_rate, expected_time1, expected_time2));
}

/**
 * Test Description
 * ------------------------
 *  - Launches two kernels that run for a specified amount of time passed as a kernel argument by
 * using device function clock. Kernel execution time is calculated through elapsed time between
 * the start and end event, and calculated time is compared with passed time values.
 * Test source
 * ------------------------
 *  - catch/unit/clock/hipClockCheck.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipClock_Positive_Basic") {
  HIP_CHECK(hipSetDevice(0));
  int clock_rate = 0;  // in kHz
  HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeClockRate, 0));
  if (clock_rate == 0) {
    HipTest::HIP_SKIP_TEST("hipDeviceAttributeClockRate returns 0");
    return;
  }
  if (IsGfx11()) {
    HipTest::HIP_SKIP_TEST("Issue with clock() function on gfx11 devices!");
    return;
  }

  const auto expected_time1 = GENERATE(1000, 1500, 2000);
  const auto expected_time2 = expected_time1 / 2;

  REQUIRE(kernel_time_execution(kernel_c, clock_rate, expected_time1, expected_time2));
}

/**
 * Test Description
 * ------------------------
 *  - Launches two kernels that run for a specified amount of time passed as a kernel argument by
 * using device function wall_clock64. Kernel execution time is calculated through elapsed time
 * between the start and end event, and calculated time is compared with passed time values.
 * Test source
 * ------------------------
 *  - catch/unit/clock/hipClockCheck.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipWallClock64_Positive_Basic") {
  HIP_CHECK(hipSetDevice(0));
  int clock_rate = 0;  // in kHz
  HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeWallClockRate, 0));

  if (!clock_rate) {
    HipTest::HIP_SKIP_TEST("hipDeviceAttributeWallClockRate returns 0");
    return;
  }

  const auto expected_time1 = GENERATE(1000, 1500, 2000);
  const auto expected_time2 = expected_time1 / 2;

  REQUIRE(kernel_time_execution(kernel_wc64, clock_rate, expected_time1, expected_time2));
}

/**
 * End doxygen group DeviceLanguageTest.
 * @}
 */
