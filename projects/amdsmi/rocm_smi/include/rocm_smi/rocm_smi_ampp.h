// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Build-internal helper exposed with real linkage (namespace amd::smi, not
// an anonymous namespace) purely so gtest can exercise the pure-parsing
// logic directly, without any sysfs I/O, root privileges, or mock-sysfs
// overlay -- mirroring the precedent set by
// amd::smi::translate_header_to_flag_version in rocm_smi_gpu_metrics.cc /
// dynamic_metrics_test.cc. Not part of the public amdsmi.h or rocm_smi.h API.

#ifndef ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_AMPP_H_
#define ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_AMPP_H_

#include <cstdint>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi.h"

namespace amd::smi {

// Parses the trimmed content of app_modes/config/writable_slot_mask (e.g.
// "0xe0") into the set of writable slot indices its set bits designate
// (bit N set => slot N writable). Pure string parsing -- no filesystem I/O
// -- so it can be unit-tested directly with arbitrary (including
// malformed/adversarial) input.
//
// The slot count is discovered at runtime, so no slot-count ceiling is
// imposed here: any bit the driver sets within the 64-bit mask is honored,
// letting a future driver publish more than today's 8 slots without an
// amdsmi rebuild. config/writable_slot_mask is still not trusted blindly --
// syntactically malformed content returns RSMI_STATUS_UNEXPECTED_DATA
// instead of being silently truncated, matching the convention used
// elsewhere in this codebase for malformed sysfs numeric content.
//
// @param[in]  trimmed_line Whitespace-trimmed content of
//             config/writable_slot_mask. An empty string yields an empty
//             *out_slots and RSMI_STATUS_SUCCESS (treated as "nothing
//             writable", not a parse error).
// @param[out] out_slots Cleared, then filled in ascending order with the
//             slot indices whose bit is set in the mask on success.
// @retval RSMI_STATUS_SUCCESS on a well-formed (possibly empty) mask.
// @retval RSMI_STATUS_UNEXPECTED_DATA if the content does not parse as a
//         "0x"-prefixed hex value or overflows a 64-bit unsigned value.
rsmi_status_t parse_ampp_writable_slot_mask(const std::string& trimmed_line,
                                            std::vector<uint32_t>* out_slots);

}  // namespace amd::smi

#endif  // ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_AMPP_H_
