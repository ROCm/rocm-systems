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

#include "amd_smi/impl/mock_gpu_backend.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <utility>

namespace amd::smi {

MockGPUBackend::MockGPUBackend(const std::string& yaml_path) : data_valid_(false) {
  auto loaded = load_mock_data_yaml(yaml_path);
  if (!loaded.ok) {
    // Defer to runtime: each call returns NOT_SUPPORTED. We still want to
    // surface the load error once on stderr to aid debugging, since this is
    // the dev/test path only.
    std::cerr << "amd-smi MockGPUBackend: failed to load '" << yaml_path << "': " << loaded.error
              << "\n";
    return;
  }
  data_ = std::move(loaded.data);
  data_valid_ = true;
}

amdsmi_status_t MockGPUBackend::get_gpu_metrics_info(amdsmi_gpu_metrics_t* pgpu_metrics) {
  if (pgpu_metrics == nullptr) return AMDSMI_STATUS_INVAL;
  if (!data_valid_) return AMDSMI_STATUS_NOT_SUPPORTED;

  std::memset(pgpu_metrics, 0, sizeof(*pgpu_metrics));

  const MockGPUMetricsData& m = data_.gpu_metrics;
  if (m.temperature_edge) pgpu_metrics->temperature_edge = *m.temperature_edge;
  if (m.temperature_hotspot) pgpu_metrics->temperature_hotspot = *m.temperature_hotspot;
  if (m.temperature_mem) pgpu_metrics->temperature_mem = *m.temperature_mem;
  if (m.average_socket_power) pgpu_metrics->average_socket_power = *m.average_socket_power;
  if (m.average_gfx_activity) pgpu_metrics->average_gfx_activity = *m.average_gfx_activity;
  if (m.average_umc_activity) pgpu_metrics->average_umc_activity = *m.average_umc_activity;
  if (m.current_socket_power) pgpu_metrics->current_socket_power = *m.current_socket_power;
  if (m.current_gfxclk) pgpu_metrics->current_gfxclk = *m.current_gfxclk;
  if (m.energy_accumulator) pgpu_metrics->energy_accumulator = *m.energy_accumulator;
  if (m.system_clock_counter) pgpu_metrics->system_clock_counter = *m.system_clock_counter;
  if (m.firmware_timestamp) pgpu_metrics->firmware_timestamp = *m.firmware_timestamp;

  if (m.temperature_hbm) {
    const auto& src = *m.temperature_hbm;
    const size_t n = std::min<size_t>(src.size(), AMDSMI_NUM_HBM_INSTANCES);
    for (size_t i = 0; i < n; ++i) pgpu_metrics->temperature_hbm[i] = src[i];
  }
  if (m.vcn_activity) {
    const auto& src = *m.vcn_activity;
    const size_t n = std::min<size_t>(src.size(), AMDSMI_MAX_NUM_VCN);
    for (size_t i = 0; i < n; ++i) pgpu_metrics->vcn_activity[i] = src[i];
  }

  return AMDSMI_STATUS_SUCCESS;
}

}  // namespace amd::smi
