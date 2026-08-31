// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

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

// Companion seam for the pp_dpm_* clock table behind the same min/max clock.
// With derive_minmax, min/max and the deep-sleep "S:" floor derive from the
// levels; callers that already have a range from pp_od_clk_voltage pass false.
// Out-params accumulate, so pre-seed: num_dpm = 0, sleep_freq = UINT_MAX, plus
// max = 0 and min = UINT_MAX when derive_minmax.
amdsmi_status_t smi_amdgpu_parse_clk_ranges(std::istream& ranges, bool derive_minmax,
                                            unsigned int* max_freq, unsigned int* min_freq,
                                            unsigned int* num_dpm, unsigned int* sleep_freq);
