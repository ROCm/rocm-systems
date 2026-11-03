// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <unordered_map>
#include <cstdint>
#include <vector>

namespace pm4_builder {
// SqttBuilder config
struct TraceConfig {
  uint32_t targetCu = 0;
  uint32_t vmIdMask = 0;
  uint32_t simd_sel = 0xF;
  uint32_t occupancy_mode = 0;
  uint32_t deprecated_mask = 0;
  uint32_t deprecated_tokenMask = 0;
  uint32_t deprecated_tokenMask2 = 0;
  // Sampling rate
  uint32_t sampleRate = 625;
  // PERF
  uint32_t perfMASK = ~0u;
  uint32_t perfCTRL = 0;
  uint32_t perfPeriod = 0;
  std::vector<std::pair<size_t, size_t>> perfcounters{};
  // GC configurations used by both TT and SPM
  uint32_t se_number = 0;
  uint32_t sa_number = 0;
  uint32_t xcc_number = 0;
  // SPM mode
  bool spm_sq_32bit_mode = true;
  bool spm_has_core1 = false;
  uint32_t spm_sample_delay_max = 0;

  void* control_buffer_ptr = nullptr;
  uint32_t control_buffer_size = 0;
  void* data_buffer_ptr = nullptr;
  uint64_t data_buffer_size = 0;

  // concurrent kernels mode
  uint32_t concurrent = 0;
  // SE mask for tracing; note -> replicated for all XCCs
  uint64_t se_mask = 0x11;
  // Subset of se_mask whose SEs emit detailed (instruction) tokens. SEs that are captured
  // but not in this mask get a lightweight token mask that still carries wave/event/reg
  // tokens, mirroring PAL's seDetailedMask + SqttTokenConfigNoInst split. Defaults to all
  // bits set so callers that never set it behave exactly as before.
  uint64_t detail_se_mask = ~0ull;

  // Maps shader engine IDs to list of buffers
  std::unordered_map<int, std::vector<void*>> buffer_data{};

  // Unweighted per-SE budget. Still the single source of truth for the gfx9/10/11 paths and
  // for direct struct users; GpuSqttBuilder::Begin additionally fills capacity_per_se_map so
  // a detailed SE can be given a larger slice than a lightweight one (PAL does the same via
  // DetailSqttSeBufferMultiplier).
  uint64_t capacity_per_se = 0x1000;
  uint64_t capacity_per_disabled_se = 0;
  std::unordered_map<int, int> target_cu_per_se{};
  std::unordered_map<int, uint64_t> se_base_addresses{};
  std::unordered_map<int, uint64_t> capacity_per_se_map{};

  bool enable_rt_timestamp{false};

  // Emit the rocprof trace-decoder instrumentation preamble on SQ_THREAD_TRACE_USERDATA_2
  // at SQTT start (the '\0ROC' magic + version + AGENT_INFO packets). This shares the same
  // register channel that RGP parses for its own no-resync marker stream, so it must be
  // suppressed when the capture feeds RGP (see threadtrace.cpp / GpuSqttBuilder::Begin).
  // Defaults to true here to preserve behavior for direct struct users and pm4 unit tests.
  bool emit_decoder_instrument{true};

  int GetTargetCU(int SE) const { return target_cu_per_se.at(SE); };
  uint64_t GetSEmask() const { return se_mask; };
  uint64_t GetSEBaseAddr(int SE) const { return se_base_addresses.at(SE); }
  // True when SE is captured AND selected for detailed (instruction) tokens.
  bool IsDetailSE(int SE) const {
    return (GetTargetCU(SE) >= 0) && (((detail_se_mask >> SE) & 1ull) != 0);
  }
  uint64_t GetCapacity(int SE) const {
    if (GetTargetCU(SE) < 0) return capacity_per_disabled_se;
    const auto it = capacity_per_se_map.find(SE);
    return (it != capacity_per_se_map.end()) ? it->second : capacity_per_se;
  }
};

}  // namespace pm4_builder
