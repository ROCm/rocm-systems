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

// Single GPU data backend for WSL2, where there is no /sys/class/drm or
// /dev/kfd and the GPU is reached through WDDM D3DKMT via /dev/dxg. It keeps the
// WSL/native choice in one place instead of scattered `if (wsl)` branches, and
// lets unsupported queries collapse to NOT_SUPPORTED without a separate device
// type. Bodies currently return synthetic mock data so the seam runs without a
// WSL host; a real port swaps them for D3DKMT calls.
class AMDSmiWslBackend {
 public:
  // True only when compiled in AND activated at runtime; a compile-time false
  // otherwise so the intercept path is dead-stripped.
  static bool active();

  // Uncached activation decision, exposed for tests.
  static bool compute_active();

  static AMDSmiWslBackend& instance();

  amdsmi_status_t get_gpu_asic_info(amdsmi_processor_handle handle, amdsmi_asic_info_t* info);
  amdsmi_status_t get_gpu_board_info(amdsmi_processor_handle handle, amdsmi_board_info_t* info);
  amdsmi_status_t get_power_info(amdsmi_processor_handle handle, amdsmi_power_info_t* info);
  amdsmi_status_t get_temp_metric(amdsmi_processor_handle handle,
                                  amdsmi_temperature_type_t sensor_type,
                                  amdsmi_temperature_metric_t metric, int64_t* temperature);

  AMDSmiWslBackend(const AMDSmiWslBackend&) = delete;
  AMDSmiWslBackend& operator=(const AMDSmiWslBackend&) = delete;

 private:
  AMDSmiWslBackend() = default;
};

}  // namespace amd::smi

// The single WSL-vs-native branch, placed right after AMDSMI_CHECK_INIT() in a
// hooked function. When ENABLE_WSL_BACKEND is off it expands to nothing, so the
// native translation unit is unchanged.
#if defined(AMDSMI_ENABLE_WSL_BACKEND)
#define AMDSMI_WSL_INTERCEPT(CALL)                        \
  do {                                                    \
    if (amd::smi::AMDSmiWslBackend::active()) {           \
      return amd::smi::AMDSmiWslBackend::instance().CALL; \
    }                                                     \
  } while (0)
#else
#define AMDSMI_WSL_INTERCEPT(CALL) \
  do {                             \
  } while (0)
#endif

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_BACKEND_H_
