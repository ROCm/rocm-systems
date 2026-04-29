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

#include "amd_smi/impl/production_gpu_backend.h"

#include "amd_smi/impl/amd_smi_common.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "rocm_smi/rocm_smi.h"

namespace amd::smi {

amdsmi_status_t ProductionGPUBackend::get_gpu_metrics_info(amdsmi_gpu_metrics_t* pgpu_metrics) {
  uint32_t total_num_gpu_processors = 0;
  rsmi_num_monitor_devices(&total_num_gpu_processors);
  uint32_t gpu_index = device_.get_gpu_id();
  if ((gpu_index + 1) > total_num_gpu_processors) {
    return AMDSMI_STATUS_NOT_FOUND;
  }

  rsmi_gpu_metrics_t rsmi_metrics{};
  rsmi_status_t rstatus = rsmi_dev_gpu_metrics_info_get(gpu_index, &rsmi_metrics);
  amdsmi_status_t status = amd::smi::rsmi_to_amdsmi_status(rstatus);
  if (status != AMDSMI_STATUS_SUCCESS) return status;

  copy_rsmi_gpu_metrics_to_amdsmi(rsmi_metrics, pgpu_metrics);
  return status;
}

}  // namespace amd::smi
