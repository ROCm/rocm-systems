// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <climits>
#include <iosfwd>

#include "amd_smi/amdsmi.h"

// Library-local test seam for the pp_od_clk_voltage parser behind amd-smi's
// per-domain min/max clock. Not amdsmi_-prefixed, so the linker version script
// keeps it out of libamd_smi.so; tests reach it through the static archive.
// Shared by the definition (src/amd_smi/amd_smi_utils.cc) and the unit tests so
// the signature stays in sync.
//
// Returns true and writes *max_freq/*min_freq when the domain's overdrive
// section has a nonzero max; false when the section is absent (MI45x has no
// OD_FCLK) or all levels read zero, so the caller falls back to pp_dpm_*.
bool smi_amdgpu_parse_od_clk_range(std::istream& od_stream, amdsmi_clk_type_t domain,
                                   unsigned int* max_freq, unsigned int* min_freq);

// Overdrive clock range parsed from pp_od_clk_voltage. When present, the dpm
// parser prefers it over the pp_dpm_* levels for the domain's min/max.
struct SmiAmdgpuOdClkRange {
  bool present = false;
  unsigned int max = 0;
  unsigned int min = UINT_MAX;
};

// Per-domain clock ranges produced by smi_amdgpu_parse_dpm_ranges(). An unset
// field keeps the "unavailable" marker (-1 as int / UINT32_MAX to callers): a
// domain with no minimum dpm level or no sleep state reports it as unavailable.
struct SmiAmdgpuClkRanges {
  int max_freq = 0;
  int min_freq = -1;
  int num_dpm = 0;
  int sleep_state_freq = -1;
};

// Finalizes smi_amdgpu_get_ranges() from an open pp_dpm_* stream (definition in
// src/amd_smi/amd_smi_utils.cc). Exposed as a test seam so the min/max/sleep
// folding and the out-of-bounds guard can be exercised over in-memory streams
// without a GPU. od_range carries the already-parsed pp_od_clk_voltage range;
// the UINT_MAX "unavailable" sentinel is preserved, while other values above
// INT_MAX are rejected.
amdsmi_status_t smi_amdgpu_parse_dpm_ranges(std::istream& dpm_stream,
                                            const SmiAmdgpuOdClkRange& od_range,
                                            SmiAmdgpuClkRanges& ranges);
