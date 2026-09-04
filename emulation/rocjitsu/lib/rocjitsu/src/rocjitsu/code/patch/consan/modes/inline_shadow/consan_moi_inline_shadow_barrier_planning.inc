// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// InlineShadow barrier body products and their mode-owned planning.
struct MoiInlineBarrierVisibleEvidencePlan {
  uint64_t counter_address = 0;
  uint16_t result_vgpr = 0;
  uint16_t address_vgpr = 0;
  uint16_t exec_save_sgpr = 0;
};

/// Exact InlineShadow state consumed by one ordinary barrier-epoch body.
/// Candidate planning resolves the persistent representation and scalar ABI;
/// native emission cannot reopen request or operating-point policy.
struct MoiInlineBarrierEpochEmissionPlan {
  std::optional<uint16_t> persistent_epoch_sgpr;
  std::optional<uint16_t> epoch_vgpr;
  std::optional<MoiInlineBarrierVisibleEvidencePlan> visible_evidence;
  std::optional<MoiSpecialStateSgprs> special_state;
};

/// Exact private-epoch state consumed by one InlineShadow barrier body.
struct MoiInlinePrivateEpochBarrierEmissionPlan {
  uint16_t scratch_vgpr = 0;
  uint32_t epoch_offset = 0;
};

[[nodiscard]] std::optional<MoiInlineBarrierEpochEmissionPlan>
plan_inline_shadow_barrier_epoch_emission(const BoundRuntimeResources &bound_resources,
                                          const ConSanMoiOperatingPoint &point,
                                          const MoiScalarAbiPlan &scalar_abi,
                                          std::optional<uint16_t> visible_evidence_scratch_vgpr,
                                          rj_code_arch_t arch, std::vector<std::string> &errors) {
  MoiInlineBarrierEpochEmissionPlan plan{
      .persistent_epoch_sgpr = point.moi_persistent_sgprs.epoch(),
      .epoch_vgpr = point.moi_owner_epoch_vgprs.epoch(),
      .visible_evidence = std::nullopt,
      .special_state = scalar_abi.special_state,
  };
  if (visible_evidence_scratch_vgpr) {
    const ConSanTargetProfile *target = consan_target_profile(arch);
    if (!bound_resources.moi_report_buffer_address || !point.moi_exec_save_sgpr ||
        target == nullptr || *visible_evidence_scratch_vgpr > 253u) {
      errors.emplace_back(
          "ConSan MOI inline barrier visible evidence requires report and scalar state");
      return std::nullopt;
    }
    const uint16_t scratch = *visible_evidence_scratch_vgpr;
    const bool address_first = target->moi_visible_evidence_address_precedes_result;
    plan.visible_evidence = MoiInlineBarrierVisibleEvidencePlan{
        .counter_address = *bound_resources.moi_report_buffer_address +
                           offsetof(ConSanMoiReportHeader, event_counter),
        .result_vgpr = static_cast<uint16_t>(scratch + (address_first ? 2u : 0u)),
        .address_vgpr = static_cast<uint16_t>(scratch + (address_first ? 0u : 1u)),
        .exec_save_sgpr = *point.moi_exec_save_sgpr,
    };
  }
  if ((plan.persistent_epoch_sgpr || plan.visible_evidence) && !plan.special_state) {
    errors.emplace_back("ConSan MOI scalar inline barrier requires scalar special-state storage");
    return std::nullopt;
  }
  return plan;
}
