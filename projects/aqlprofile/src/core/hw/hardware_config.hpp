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

/// Hardware configuration data for a specific GPU architecture instance.
/// Populated from AgentInfo at factory creation time; used by capability
/// query methods on Pm4Factory so callers don't need raw gpu_id_t comparisons.
struct HardwareConfig {
  // Topology (filled from AgentInfo)
  uint32_t se_count;           // Shader Engines count
  uint32_t sa_per_se_count;    // Shader Arrays per SE
  uint32_t cu_count;           // Total Compute Units (after any patching)
  uint32_t wgp_count;          // Work Group Processors (cu_count / 2 for GFX10+)
  uint32_t xcc_count;          // XCC count
  uint32_t xcc_per_aid;        // XCCs per AID (1 normal, 2 MI300, 4 MI450)
  uint32_t aid_count;          // AID (Accelerator Integration Die) count

  // Architecture name strings (for GetGFX() and block table debug)
  std::string name;   // e.g. "gfx1250"
  std::string gfxip;  // same value (name == gfxip on newer APIs)

  // Capability flags — replaces GetGpuId() comparisons in aql_profile.cpp
  bool has_spm_core1;                 // SPM dual-core support (MI100/MI200)
  bool has_wptr_relative_addressing;  // SQTT wptr is relative to buffer base (GFX11)
  bool has_sqtt_status2_register;     // SQTT STATUS2 register present (GFX12+)
  bool needs_sqtt_header_packet;      // Prepend header packet to SQTT data (GFX9)
  bool has_sriov_spm_restriction;     // SPM blocked under SR-IOV (gfx942/MI300)
  bool has_asymmetric_cu_design;      // CUs not uniformly distributed (gfx1250/MI450)
  bool supports_spm_v2;               // SPM v2 ring-buffer mode (MI200/MI300/MI350)

  uint32_t spm_sample_delay_max;      // Max SPM sample delay value (0 = no limit)
  uint32_t sqtt_header_version;       // SQTT header version2 field (GFX9 only)

  // Helpers
  uint32_t GetSEPerXCC() const { return (xcc_count > 0) ? se_count / xcc_count : se_count; }
  bool IsMultiXCC() const { return xcc_count > 1; }

  HardwareConfig()
      : se_count(0), sa_per_se_count(0), cu_count(0), wgp_count(0),
        xcc_count(1), xcc_per_aid(1), aid_count(1),
        has_spm_core1(false), has_wptr_relative_addressing(false),
        has_sqtt_status2_register(false), needs_sqtt_header_packet(false),
        has_sriov_spm_restriction(false), has_asymmetric_cu_design(false),
        supports_spm_v2(false),
        spm_sample_delay_max(0), sqtt_header_version(0) {}
};

}  // namespace aql_profile

#endif  // SRC_CORE_HARDWARE_CONFIG_HPP_
