// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

inline constexpr uint16_t kConSanGfx1250SemanticFormMask =
    kConSanCdnaSemanticFormMask | consan_capability_form_bit(ConSanCapabilityForm::ClusterBarrier) |
    consan_capability_form_bit(ConSanCapabilityForm::OrderedLdsAtomic);

/// Complete gfx1250 target-profile specialization.
inline constexpr ConSanTargetProfile kConSanGfx1250TargetProfile = [] {
  ConSanTargetProfile profile;
  profile.target = ROCJITSU_CODE_TARGET_GFX1250;
  profile.arch = ROCJITSU_CODE_ARCH_CDNA5;
  profile.encoding_family = ConSanEncodingFamily::Gfx12;
  profile.accumulator_model = ConSanAccumulatorModel::SelectableVgprBank;
  profile.dispatch_identity = ConSanDispatchIdentitySource::CodeObjectLiteral;
  profile.workgroup_identity = ConSanWorkgroupIdentitySource::CommandProcessorTtmps;
  profile.wait_counter_family = ConSanWaitCounterFamily::Gfx12;
  profile.direct_call_form = ConSanDirectCallForm::SCallI64;
  profile.code_transport = ConSanCodeTransportModel::PerKernelOwnerTranslation;
  profile.resident_wave_identity = {.hwreg_id = 23, .bit_offset = 0, .bit_width = 10};
  profile.workgroup_shadow_clear = {
      .encoding = ConSanWorkgroupShadowClearEncoding::PackedB128,
      .maximum_lanes = 64,
  };
  profile.supports_wave32 = true;
  profile.vgpr_allocation_granularity_wave32 = 16;
  profile.vgpr_allocation_granularity_wave64 = 8;
  profile.ordinary_sgpr_limit = 106;
  profile.reserved_ordinary_sgpr_base = 102;
  profile.reserved_ordinary_sgpr_count = 4;
  profile.user_sgpr_initialization_limit = 32;
  profile.address_free_private_limit_bytes = 0x800000u;
  profile.max_group_segment_bytes = static_cast<uint32_t>(ROCJITSU_GFX1250_LDS_SIZE_KB) * 1024u;
  profile.has_cluster_facilities = true;
  profile.has_selectable_vgpr_bank = true;
  profile.requires_even_vgpr_tuples = true;
  profile.requires_split_two_address_lds_relocation = true;
  profile.semantic_form_mask = kConSanGfx1250SemanticFormMask;
  profile.supports_sc_branch_only_route = true;
  profile.requires_sc_runtime_flat_group_gate = true;
  profile.sc_dense_route_identity = ConSanScDenseRouteIdentity::ReturnPc;
  profile.requires_raw_memory_order_qualifier = true;
  return profile;
}();
