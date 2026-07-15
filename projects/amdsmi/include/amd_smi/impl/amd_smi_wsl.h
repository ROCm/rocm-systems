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

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"

namespace amd::smi {

/*
 * WSL2 (Windows Subsystem for Linux) support.
 * -----------------------------------------------------------------------------
 * Under WSL2 an AMD GPU is exposed to the Linux guest through Microsoft's
 * /dev/dxg paravirtual device (WDDM/dxgkrnl), NOT through the native amdgpu
 * kernel driver. As a result none of the data sources amdsmi normally relies on
 * are present:
 *   - /dev/kfd is absent
 *   - /sys/class/kfd/kfd/topology/nodes is absent
 *   - the amdgpu sysfs nodes under /sys/class/drm/cardN/device are absent
 *   - /sys/module/amdgpu/initstate is absent
 *
 * Consequently rsmi_init() (the rocm_smi backend that amdsmi_init() drives)
 * fails during KFD node discovery and amdsmi_init() returns
 * AMDSMI_STATUS_DRIVER_NOT_LOADED, leaving callers with no usable device handle.
 *
 * The HIP runtime, however, *does* work under WSL2 (it talks to the Windows
 * driver through /dev/dxg). This module provides a minimal device-enumeration
 * and query fallback that is backed by the HIP runtime instead of the amdgpu
 * driver. It is engaged only when the host is detected to be WSL2 and the
 * native rsmi/KFD discovery has failed, so it has zero effect on bare-metal
 * Linux.
 *
 * The HIP runtime is loaded lazily via dlopen (see amd_smi_lib_loader.h); amdsmi
 * does NOT link against HIP. Only a small set of forward-ABI-stable HIP entry
 * points is used (no hipDeviceProp_t, whose layout is version dependent), so the
 * fallback is robust across ROCm releases.
 *
 * Telemetry that genuinely has no WSL2 data source (power, temperature, fan,
 * clocks, ECC, ...) keeps returning AMDSMI_STATUS_NOT_SUPPORTED.
 */

/// Cached, mostly-static information for a single HIP-visible GPU on WSL2.
struct WslGpuInfo {
  int hip_index = -1;              //!< HIP device ordinal (hipSetDevice index)
  std::string market_name;         //!< e.g. "AMD Radeon(TM) 8060S Graphics"
  std::string bdf_string;          //!< "domain:bus:device.function" from HIP
  amdsmi_bdf_t bdf = {};           //!< parsed BDF (0 fields if unknown)
  uint64_t device_id = 0;          //!< PCI device id if readable, else 0
  uint32_t num_compute_units = 0;  //!< CU count from HIP, 0 if unknown
};

/// Live HIP memory values in bytes, before conversion to the MB-based public
/// amdsmi_vram_usage_t structure.
struct WslVramUsageBytes {
  uint64_t total = 0;
  uint64_t used = 0;
};

/// Returns true if the current process is running inside WSL2.
/// Detection is conservative: it requires a Microsoft/WSL kernel marker AND the
/// /dev/dxg device, AND the absence of /dev/kfd (the native amdgpu path).
bool is_wsl2_environment();

/// Enumerate AMD GPUs visible to the HIP runtime under WSL2 and fill `out` with
/// their cached static info. Returns AMDSMI_STATUS_SUCCESS only if the HIP
/// runtime could be loaded and reported at least one device.
amdsmi_status_t wsl_discover_gpus(std::vector<WslGpuInfo>* out);

/// Fill an amdsmi_asic_info_t for a WSL2 device from cached HIP info.
amdsmi_status_t wsl_fill_asic_info(const WslGpuInfo& gpu, amdsmi_asic_info_t* info);

/// Fill an amdsmi_board_info_t for a WSL2 device from cached HIP info.
amdsmi_status_t wsl_fill_board_info(const WslGpuInfo& gpu, amdsmi_board_info_t* info);

/// Fill an amdsmi_vram_info_t for a WSL2 device. The VRAM size is obtained with
/// a live hipMemGetInfo query so device enumeration does not need to create a
/// HIP context on every GPU up front.
amdsmi_status_t wsl_fill_vram_info(const WslGpuInfo& gpu, amdsmi_vram_info_t* info);

/// Fill an amdsmi_vram_usage_t for a WSL2 device. Performs a live hipMemGetInfo
/// query (total/used change at runtime).
amdsmi_status_t wsl_fill_vram_usage(const WslGpuInfo& gpu, amdsmi_vram_usage_t* info);

/// Query live total/used VRAM in bytes without the precision loss of the
/// MB-based amdsmi_vram_usage_t structure.
amdsmi_status_t wsl_get_vram_usage_bytes(const WslGpuInfo& gpu, WslVramUsageBytes* usage);

/// Generate the deterministic WSL2 UUID from the HIP-visible device identity.
amdsmi_status_t wsl_generate_device_uuid(const WslGpuInfo& gpu, char* uuid);

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_WSL_H_
