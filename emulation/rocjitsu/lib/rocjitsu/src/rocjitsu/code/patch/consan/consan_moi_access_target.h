// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] std::optional<uint16_t>
candidate_lds_byte_offset_vgpr(const ConSanMoiCandidate &candidate,
                               std::vector<std::string> &errors);

[[nodiscard]] bool append_materialize_direct_to_lds_address(std::vector<uint32_t> &words,
                                                            const ConSanMoiCandidate &candidate,
                                                            uint16_t result_vgpr,
                                                            uint16_t exec_save_sgpr,
                                                            rj_code_arch_t arch);

[[nodiscard]] bool candidate_uses_scalar_vector_flat_address(const ConSanMoiCandidate &candidate);
[[nodiscard]] bool
candidate_requires_flat_address_materialization(const ConSanMoiCandidate &candidate);
[[nodiscard]] uint16_t flat_access_address_scratch_count(const ConSanMoiCandidate &candidate);

[[nodiscard]] bool append_materialize_flat_access_address(std::vector<uint32_t> &words,
                                                          const ConSanMoiCandidate &candidate,
                                                          uint16_t input_vgpr, uint16_t result_vgpr,
                                                          rj_code_arch_t arch);

[[nodiscard]] uint16_t candidate_payload_vgpr_count(const ConSanMoiCandidate &candidate);
[[nodiscard]] bool moi_load_clobbers_address(const ConSanMoiCandidate &candidate);
[[nodiscard]] bool
moi_access_requires_high_bank_address_capture(const ConSanMoiCandidate &candidate,
                                              rj_code_arch_t arch);

[[nodiscard]] bool append_compute_effective_lds_byte_offset(std::vector<uint32_t> &words,
                                                            uint16_t destination_vgpr,
                                                            uint16_t address_vgpr,
                                                            uint32_t static_byte_offset,
                                                            rj_code_arch_t arch);

} // namespace rocjitsu::consan_moi_impl
