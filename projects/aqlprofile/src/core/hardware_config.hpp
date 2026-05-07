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

#ifndef SRC_CORE_HARDWARE_CONFIG_HPP_
#define SRC_CORE_HARDWARE_CONFIG_HPP_

#include <cstdint>
#include <string>

namespace aql_profile {

/// Hardware configuration data for a specific GPU architecture
/// Consolidates architecture-specific constants that were previously
/// scattered throughout builders and factories
struct HardwareConfig {
  // Architecture identification
  std::string gfxip;          // e.g., "gfx90a", "gfx940", "gfx1100"
  std::string name;            // Human-readable name, e.g., "MI200", "RDNA3"

  // Compute unit topology
  uint32_t se_count;           // Shader Engines count
  uint32_t sa_per_se_count;    // Shader Arrays per SE
  uint32_t cu_count;           // Total Compute Units
  uint32_t wgp_count;          // Work Group Processors (GFX10+)

  // Multi-die configuration (MI300+)
  uint32_t xcc_count;          // XCC (eXtended Compute Complex) count
  uint32_t aid_count;          // AID (Accelerator Integration Die) count

  // Memory hierarchy
  uint32_t l2_cache_count;     // L2 cache instances
  uint32_t memory_channels;    // Memory controller channels

  // Profiling capabilities
  bool supports_pmc;           // Performance Monitor Counters
  bool supports_spm;           // Streaming Performance Monitors
  bool supports_sqtt;          // SQ Thread Trace
  bool supports_concurrent;    // Concurrent profiling mode

  // Architecture flags
  bool has_grbm_perfcounter;          // GRBM perfcounter support
  bool has_aid_aware_counters;        // AID-aware counter routing (MI300+)
  bool has_spm_core1;                 // SPM dual-core support (MI100/MI200)
  bool spm_sq_32bit_mode;             // SPM SQ 32-bit mode
  bool supports_spm_v2;               // SPM v2 support (MI200/MI300/MI350)
  bool has_sqtt_status2_register;     // SQTT uses STATUS2 register (GFX12+)
  bool has_wptr_relative_addressing;  // SQTT wptr is relative to buffer base (GFX11)
  bool needs_sqtt_header_packet;      // SQTT data needs a header packet prepended (GFX9)

  // SPM configuration
  uint32_t spm_sample_delay_max;  // Maximum SPM sample delay

  // Multi-die XCC partitioning
  uint32_t xcc_per_aid;           // XCCs per AID (1 for single-AID, 2 for MI300, 4 for MI450)

  // SQTT configuration
  uint32_t sqtt_buffer_alignment; // SQTT buffer alignment requirement
  uint32_t sqtt_header_version;   // SQTT header version2 field (GFX9 only; 4=generic, 5=MI300, 6=MI350)

  // Default constructor
  HardwareConfig()
      : gfxip("unknown"),
        name("Unknown"),
        se_count(0),
        sa_per_se_count(0),
        cu_count(0),
        wgp_count(0),
        xcc_count(1),
        aid_count(1),
        l2_cache_count(0),
        memory_channels(0),
        supports_pmc(true),
        supports_spm(false),
        supports_sqtt(false),
        supports_concurrent(false),
        has_grbm_perfcounter(true),
        has_aid_aware_counters(false),
        has_spm_core1(false),
        spm_sq_32bit_mode(true),
        supports_spm_v2(false),
        has_sqtt_status2_register(false),
        has_wptr_relative_addressing(false),
        needs_sqtt_header_packet(false),
        xcc_per_aid(1),
        spm_sample_delay_max(0),
        sqtt_buffer_alignment(0x1000),
        sqtt_header_version(0) {}

  // Helper: Is this a multi-XCC architecture?
  bool IsMultiXCC() const { return xcc_count > 1; }

  // Helper: Is this an MI300-series GPU?
  bool IsMI300Series() const { return aid_count > 1; }

  // Helper: Total WGP count (computed)
  uint32_t GetTotalWGPs() const {
    if (wgp_count > 0) return wgp_count;
    // Fallback: approximate from CU count for older architectures
    return cu_count / 2;
  }

  // Helper: SE count per XCC
  uint32_t GetSEPerXCC() const {
    return se_count / xcc_count;
  }
};

}  // namespace aql_profile

#endif  // SRC_CORE_HARDWARE_CONFIG_HPP_
