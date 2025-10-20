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
#include <cstring>
#ifndef _WIN32
#include <dlfcn.h>
#endif

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

template <class T> void loadSym(T& symbol, const char* symbolName, void* handle) {
  using namespace std::string_literals;
  void* fnsym = dlsym(handle, symbolName);

  if (!fnsym)
    throw std::runtime_error("Failure while trying to dynamically load symbol: "s + symbolName);

  symbol = reinterpret_cast<T>(fnsym);
}

void getCurrentDeviceUUID(hipUUID& uuid) {
  hipDeviceProp_t props;
  int deviceId;

  HIP_CHECK(hipGetDevice(&deviceId));
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  std::memcpy(uuid.bytes, props.uuid.bytes, sizeof(hipUUID::bytes));
}

#ifndef _WIN32
// Gets the maximum engine frequency of the GPU by dynamically loading amdsmi
// @uuid the id of the GPU to query the frequency for
// @return the maximum engine frequency of the GPU (MHz) or -1 if error
int getEngineFreq(const hipUUID& uuid) {
  static constexpr unsigned int AMDSMI_MAX_STRING_LENGTH = 256;
  typedef void* amdsmi_processor_handle;
  typedef void* amdsmi_socket_handle;
  typedef enum {
    AMDSMI_STATUS_SUCCESS = 0,  //!< Call succeeded
  } amdsmi_status_t;

  typedef struct {
    uint32_t clk;            //!< In MHz
    uint32_t min_clk;        //!< In MHz
    uint32_t max_clk;        //!< In MHz
    uint8_t clk_locked;      //!< True/False
    uint8_t clk_deep_sleep;  //!< True/False
    uint32_t reserved[4];
  } amdsmi_clk_info_t;

  typedef struct {
    uint32_t drm_render;  //!< the render node under /sys/class/drm/renderD*
    uint32_t drm_card;    //!< the graphic card device under /sys/class/drm/card*
    uint32_t hsa_id;      //!< the HSA enumeration ID
    uint32_t hip_id;      //!< the HIP enumeration ID
    char hip_uuid[AMDSMI_MAX_STRING_LENGTH];  //!< the HIP unique identifer
  } amdsmi_enumeration_info_t;

  typedef enum {
    AMDSMI_CLK_TYPE_GFX = 0x0
  } amdsmi_clk_type_t;

  amdsmi_clk_info_t clk_info;
  uint32_t gpu_count = 0;
  uint32_t num_processor = 0;
  amdsmi_status_t (*fninit)(uint64_t);
  amdsmi_status_t (*fnget_socket_handles)(uint32_t*, amdsmi_socket_handle*);
  amdsmi_status_t (*fnget_processor_handles)(amdsmi_socket_handle, uint32_t*,
                                             amdsmi_processor_handle*);
  amdsmi_status_t (*fnget_gpu_enumeration_info)(amdsmi_processor_handle,
                                                amdsmi_enumeration_info_t*);
  amdsmi_status_t (*fnget_clock_info)(amdsmi_processor_handle, amdsmi_clk_type_t,
                                      amdsmi_clk_info_t*);
  amdsmi_status_t (*fnshut_down)();
  int result = -1;
  bool smi_initialized = false;
  auto cleanUp = [&smi_initialized, &fnshut_down](void* handle) {
    if (smi_initialized)
      fnshut_down();

    if (handle)
      dlclose(handle);
  };
  std::unique_ptr<void, decltype(cleanUp)> lib_hdl(nullptr, cleanUp);

  lib_hdl.reset(dlopen("libamd_smi.so", RTLD_LAZY));

  if (!lib_hdl) {
    return -1;
  }

  try {
    loadSym(fninit, "amdsmi_init", lib_hdl.get());
    loadSym(fnget_socket_handles, "amdsmi_get_socket_handles", lib_hdl.get());
    loadSym(fnget_processor_handles, "amdsmi_get_processor_handles", lib_hdl.get());
    loadSym(fnget_gpu_enumeration_info, "amdsmi_get_gpu_enumeration_info", lib_hdl.get());
    loadSym(fnget_clock_info, "amdsmi_get_clock_info", lib_hdl.get());
    loadSym(fnshut_down, "amdsmi_shut_down", lib_hdl.get());
  } catch (std::runtime_error&) {
    return -1;
  }

  if (fninit(1ul << 1)) {
    return -1;
  } else
    smi_initialized = true;

  uint32_t socket_count = 0;
  uint32_t num_socket = 0;

  // get the socket count available in the system
  if (fnget_socket_handles(&socket_count, nullptr)) {
    return -1;
  }

  std::vector<amdsmi_socket_handle> sockets(socket_count);
  if (fnget_socket_handles(&socket_count, &sockets[0])) {
    return -1;
  }

  while (num_socket < socket_count && result == -1) {
    // just get number of processors first
    if (fnget_processor_handles(sockets[num_socket], &gpu_count, nullptr)) {
      return -1;
    }

    std::vector<amdsmi_processor_handle> processors(gpu_count);
    if (fnget_processor_handles(sockets[num_socket], &gpu_count, &processors[0])) {
      return -1;
    }

    while (num_processor < gpu_count && result == -1) {
      amdsmi_enumeration_info_t info;
      int offset = 0;
      const char* prefix = "GPU-";

      if (fnget_gpu_enumeration_info(processors[num_processor], &info)) {
        return -1;
      }

      if (!std::strncmp(info.hip_uuid, "GPU-", std::strlen(prefix))) {
        // amd-smi adds "GPU-" in front of the hip_uuid; whereas HIP doesn't
        offset = strlen(prefix);
      }

      if (!std::memcmp(uuid.bytes, info.hip_uuid + offset, sizeof(hipUUID::bytes) - offset)) {
        if (fnget_clock_info(processors[num_processor], AMDSMI_CLK_TYPE_GFX, &clk_info)) {
          return -1;
        }

        result = clk_info.max_clk;
      }

      num_processor++;
    }

    num_socket++;
    num_processor = 0;
  }

  return result;
}
#endif

// @max_clock_rate will be set to the maximum clock rate as reported by hipDeviceGetAttribute()
// @return         maximum engine clock rate obtained via amdsmi or -1 if querying via amdsmi fails
int getClockRate(int& max_clock_rate) {
  max_clock_rate = 0;  // in kHz
  HIP_CHECK(hipDeviceGetAttribute(&max_clock_rate, hipDeviceAttributeClockRate, 0));

#ifdef _WIN32
  return -1;
#else
  hipUUID uuid;
  int smi_clock_rate = 0;  // in kHz

  getCurrentDeviceUUID(uuid);
  smi_clock_rate = getEngineFreq(uuid);
  return smi_clock_rate;
#endif
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

  int max_clock_rate;
  int clock_rate = getClockRate(max_clock_rate);

  if (max_clock_rate == 0) {
    HipTest::HIP_SKIP_TEST("hipDeviceAttributeClockRate returns 0");
    return;
  }
  if (IsGfx11()) {
    HipTest::HIP_SKIP_TEST("Issue with clock64() function on gfx11 devices!");
    return;
  }

  if (clock_rate == -1) {
    // libamd_smi.so might not be present depending on some systems, so we load it dynamically
    // and use it if it is, otherwise we use the attribute
    UNSCOPED_INFO(
        "Failed to get clock rate via amdsmi (is libamd_smi.so in the library search path?)");
    clock_rate = max_clock_rate;
  } else {
    clock_rate *= 1000;

    if (clock_rate != max_clock_rate) {
      UNSCOPED_INFO("clock rate: " << clock_rate << "kHz is not set to maximum: " << max_clock_rate
                                   << "kHz");
    } else {
      UNSCOPED_INFO("clock rate: " << clock_rate << "kHz");
    }
  }

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

  int max_clock_rate;
  int clock_rate = getClockRate(max_clock_rate);

  if (max_clock_rate == 0) {
    HipTest::HIP_SKIP_TEST("hipDeviceAttributeClockRate returns 0");
    return;
  }
  if (IsGfx11()) {
    HipTest::HIP_SKIP_TEST("Issue with clock64() function on gfx11 devices!");
    return;
  }

  if (clock_rate == -1) {
    // libamd_smi.so might not be present depending on some systems, so we load it dynamically
    // and use it if it is, otherwise we use the attribute
    UNSCOPED_INFO(
        "Failed to get clock rate via amdsmi (is libamd_smi.so in the library search path?)");
    clock_rate = max_clock_rate;
  } else {
    clock_rate *= 1000;

    if (clock_rate != max_clock_rate) {
      UNSCOPED_INFO("clock rate: " << clock_rate << "kHz is not set to maximum: " << max_clock_rate
                                   << "kHz");
    }
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
