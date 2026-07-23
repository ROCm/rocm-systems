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

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_BACKEND_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_BACKEND_H_

#include <vector>

#include "amd_smi/amdsmi.h"

namespace amd::smi {

// Per-GPU backend abstraction. amd_smi.cc dispatches to this instead of
// branching on platform identity. Non-pure virtuals return NOT_SUPPORTED so
// each backend only overrides what it actually implements.
class IGPUBackend {
 public:
  virtual ~IGPUBackend() = default;

  virtual amdsmi_status_t GetAsicInfo(amdsmi_asic_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetBoardInfo(amdsmi_board_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetKfdInfo(amdsmi_kfd_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetVramInfo(amdsmi_vram_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetMemoryTotal(amdsmi_memory_type_t, uint64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetMemoryUsage(amdsmi_memory_type_t, uint64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetTempMetric(amdsmi_temperature_type_t, amdsmi_temperature_metric_t,
                                        int64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetVoltMetric(amdsmi_voltage_type_t, amdsmi_voltage_metric_t, int64_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetPowerInfo(amdsmi_power_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetGpuActivity(amdsmi_engine_usage_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetBusyPercent(uint32_t*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetClockInfo(amdsmi_clk_type_t, amdsmi_clk_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetPcieInfo(amdsmi_pcie_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetDriverInfo(amdsmi_driver_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetVbiosInfo(amdsmi_vbios_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetProcessList(std::vector<amdsmi_proc_info_t>*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetUuid(unsigned int*, char*) { return AMDSMI_STATUS_NOT_SUPPORTED; }
  virtual amdsmi_status_t GetGpuCacheInfo(amdsmi_gpu_cache_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }
  virtual amdsmi_status_t GetFwInfo(amdsmi_fw_info_t*) {
    return AMDSMI_STATUS_NOT_SUPPORTED;
  }

  IGPUBackend(const IGPUBackend&) = delete;
  IGPUBackend& operator=(const IGPUBackend&) = delete;

 protected:
  IGPUBackend() = default;
};

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_BACKEND_H_
