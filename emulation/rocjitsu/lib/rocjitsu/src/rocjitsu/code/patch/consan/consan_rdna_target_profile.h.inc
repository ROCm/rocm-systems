// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Target-profile facts shared by gfx1100 and gfx1201.
[[nodiscard]] constexpr ConSanTargetProfile consan_rdna_target_profile_defaults() {
  ConSanTargetProfile profile;
  profile.architecture_family = ConSanArchitectureFamily::Rdna;
  profile.dispatch_identity = ConSanDispatchIdentitySource::CodeObjectLiteral;
  profile.resident_wave_identity = {.hwreg_id = 23, .bit_offset = 0, .bit_width = 10};
  profile.moi_dispatch_identity_placement =
      ConSanMoiDispatchIdentityPlacement::PersistentVectorPreferred;
  profile.supports_wave32 = true;
  profile.vgpr_allocation_granularity_wave32 = 8;
  profile.vgpr_allocation_granularity_wave64 = 4;
  profile.ordinary_sgpr_limit = 106;
  profile.user_sgpr_initialization_limit = 16;
  profile.max_group_segment_bytes = 64u * 1024u;
  profile.semantic_form_mask = kConSanCommonSemanticFormMask;
  profile.moi_dense_route_group_capacity = 64;
  profile.requires_raw_memory_order_qualifier = true;
  return profile;
}
