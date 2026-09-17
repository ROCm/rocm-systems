// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_internal.h"

namespace rocjitsu::consan::detail {

/// Target-normalized and operating-point-projected facts for one access site's
/// resource transaction. ConSan consumes this value without inspecting a
/// concrete architecture or the complete mutable ConSan operating point.
struct AccessResourceFacts {
  uint16_t address_scratch_vgpr_count = 0;
  uint16_t two_address_replay_vgpr_count = 0;
  uint16_t dynamic_stack_reservoir_vgpr_count = 0;
  bool has_exec_save = false;
  bool initialize_owner_epoch = false;
  bool has_persistent_owner_vgpr = false;
  bool uses_private_epoch = false;
  bool has_complete_persistent_sgprs = false;
  bool target_available = false;
  bool supports_native_lds_spill_recovery = false;
  bool supports_clobbered_address_spill_reload = false;
  bool guest_replay_requires_disjoint_address_scratch = false;

  bool operator==(const AccessResourceFacts &) const = default;
};

[[nodiscard]] AccessResourceFacts resolve_access_resource_facts(const OperatingPoint &point,
                                                                const Candidate &candidate,
                                                                rj_code_arch_t arch);

[[nodiscard]] std::optional<uint16_t>
candidate_lds_byte_offset_vgpr(const Candidate &candidate, std::vector<std::string> &errors);

[[nodiscard]] bool
append_materialize_direct_to_lds_address(std::vector<uint32_t> &words, const ProgramSite &site,
                                         uint16_t result_vgpr, uint16_t temporary_vgpr,
                                         uint16_t exec_save_sgpr, rj_code_arch_t arch);

[[nodiscard]] bool candidate_uses_scalar_vector_flat_address(const Candidate &candidate);
[[nodiscard]] bool candidate_requires_flat_address_materialization(const Candidate &candidate);
[[nodiscard]] uint16_t flat_access_address_scratch_count(const Candidate &candidate);

[[nodiscard]] bool append_materialize_flat_access_address(std::vector<uint32_t> &words,
                                                          const Candidate &candidate,
                                                          uint16_t input_vgpr, uint16_t result_vgpr,
                                                          rj_code_arch_t arch);

[[nodiscard]] uint16_t candidate_payload_vgpr_count(const Candidate &candidate);
[[nodiscard]] bool load_clobbers_address(const Candidate &candidate);
[[nodiscard]] bool reject_candidate_scratch_range_overlap(const Candidate &candidate,
                                                          uint16_t scratch_vgpr,
                                                          uint16_t scratch_count,
                                                          std::vector<std::string> &errors);
[[nodiscard]] bool access_requires_high_bank_address_capture(const Candidate &candidate,
                                                             rj_code_arch_t arch);

[[nodiscard]] bool append_compute_effective_lds_byte_offset(std::vector<uint32_t> &words,
                                                            uint16_t destination_vgpr,
                                                            uint16_t address_vgpr,
                                                            uint32_t static_byte_offset,
                                                            rj_code_arch_t arch);

} // namespace rocjitsu::consan::detail
