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

#ifndef AMD_SMI_INCLUDE_IMPL_MOCK_DATA_YAML_H_
#define AMD_SMI_INCLUDE_IMPL_MOCK_DATA_YAML_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace amd::smi {

// Canned values for the subset of amdsmi_gpu_metrics_t the mock populates.
// Each field is std::optional so the YAML may omit any subset; the mock leaves
// omitted fields zero-initialised in the output struct.
struct MockGPUMetricsData {
  std::optional<uint16_t> temperature_edge;
  std::optional<uint16_t> temperature_hotspot;
  std::optional<uint16_t> temperature_mem;
  std::optional<uint16_t> average_socket_power;
  std::optional<uint16_t> average_gfx_activity;
  std::optional<uint16_t> average_umc_activity;
  std::optional<uint16_t> current_socket_power;
  std::optional<uint16_t> current_gfxclk;
  std::optional<uint64_t> energy_accumulator;
  std::optional<uint64_t> system_clock_counter;
  std::optional<uint64_t> firmware_timestamp;
  std::optional<std::vector<uint16_t>> temperature_hbm;
  std::optional<std::vector<uint16_t>> vcn_activity;
};

// Loaded mock dataset. Currently only one section (gpu_metrics:) is supported,
// but the wrapper struct keeps room for future per-method canned data.
struct MockData {
  MockGPUMetricsData gpu_metrics;
};

// Result of a load attempt.
struct MockDataLoadResult {
  bool ok{false};
  std::string error;
  MockData data;
};

// Parse a YAML mock-data file at @p path. On failure, ok=false and error
// holds a human-readable message; data is value-initialised. On success,
// ok=true and data holds the parsed values.
[[nodiscard]] MockDataLoadResult load_mock_data_yaml(const std::string& path);

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_MOCK_DATA_YAML_H_
