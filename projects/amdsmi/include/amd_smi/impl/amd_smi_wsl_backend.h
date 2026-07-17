/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_BACKEND_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_BACKEND_H_

#include "amd_smi/amdsmi.h"
#include "config/amd_smi_config.h"

namespace amd::smi {

// Single GPU data backend for WSL2 (Windows Subsystem for Linux).
//
// On native Linux, amd-smi answers queries from DRM ioctls and sysfs. Under WSL2
// there is no /sys/class/drm or /dev/kfd; the GPU is reached through the WDDM
// D3DKMT interface via /dev/dxg. Rather than sprinkling `if (wsl) ... else ...`
// through every public function, the dispatcher funnels the affected queries
// through a single backend object. Native builds never see it (the intercept
// macro compiles to nothing); WSL builds select it once at init and every hooked
// function delegates through the same seam.
//
// This class is the one place the WSL/native decision lives. Methods return the
// same public structs the native path fills, so the public API is unchanged.
// Unsupported queries return AMDSMI_STATUS_NOT_SUPPORTED — the WSL capability
// subset is expressed here, not as a separate device type.
//
// The current implementation is a mock that returns synthetic-but-plausible data
// so the seam can be exercised on a native workstation without a WSL host. A real
// implementation replaces the mock body with wsl::thunk D3DKMT calls; nothing
// above this class changes.
class WslBackend {
 public:
  // True only when the backend was compiled in AND activated at runtime.
  //
  // Runtime activation: environment variable AMDSMI_WSL_MODE=1 (an explicit
  // opt-in for testing), or presence of the dxgkrnl module (real WSL2). When the
  // backend is not compiled in, this is a compile-time constant false so the
  // whole intercept path is dead-stripped.
  static bool active();

  static WslBackend& instance();

  // The hooked query surface. Each mirrors the matching amdsmi_* signature but
  // takes the resolved processor handle. Default behavior for anything not
  // overridden by the concrete backend is AMDSMI_STATUS_NOT_SUPPORTED.
  amdsmi_status_t get_gpu_asic_info(amdsmi_processor_handle handle, amdsmi_asic_info_t* info);
  amdsmi_status_t get_gpu_board_info(amdsmi_processor_handle handle, amdsmi_board_info_t* info);
  amdsmi_status_t get_power_info(amdsmi_processor_handle handle, amdsmi_power_info_t* info);
  amdsmi_status_t get_temp_metric(amdsmi_processor_handle handle,
                                  amdsmi_temperature_type_t sensor_type,
                                  amdsmi_temperature_metric_t metric, int64_t* temperature);

  WslBackend(const WslBackend&) = delete;
  WslBackend& operator=(const WslBackend&) = delete;

 private:
  WslBackend() = default;
};

}  // namespace amd::smi

// Single-line intercept placed at the top of a hooked public function, right
// after AMDSMI_CHECK_INIT(). It is the only WSL-vs-native branch in each function
// and centralizes the decision on WslBackend::active().
//
// When ENABLE_WSL_BACKEND is off, this expands to nothing, so the native
// translation unit is identical to upstream.
#if defined(AMDSMI_ENABLE_WSL_BACKEND)
#define AMDSMI_WSL_INTERCEPT(CALL)                  \
  do {                                              \
    if (amd::smi::WslBackend::active()) {           \
      return amd::smi::WslBackend::instance().CALL; \
    }                                               \
  } while (0)
#else
#define AMDSMI_WSL_INTERCEPT(CALL) \
  do {                             \
  } while (0)
#endif

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_BACKEND_H_
