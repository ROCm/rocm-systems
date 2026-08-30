// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Target-profile facts shared by gfx942 and gfx950.
[[nodiscard]] constexpr ConSanTargetProfile consan_gfx9_cdna_target_profile_defaults() {
  ConSanTargetProfile profile;
  profile.accumulator_model = ConSanAccumulatorModel::DescriptorPartitioned;
  profile.resident_wave_identity = {.hwreg_id = 4, .bit_offset = 0, .bit_width = 6};
  profile.vgpr_allocation_granularity_wave64 = 8;
  profile.accumulator_offset_granularity = 4;
  profile.ordinary_sgpr_limit = 102;
  profile.user_sgpr_initialization_limit = 16;
  profile.address_free_private_limit_bytes = 0x1000u;
  profile.private_allocation_granularity_bytes = 16;
  profile.max_group_segment_bytes = 64u * 1024u;
  profile.atomic_address_materialization.flat_and_global = true;
  profile.supports_kernarg_preload_overflow_recovery = true;
  profile.requires_even_vgpr_tuples = true;
  profile.flat_compare_swap_data_pair_alignment = 2;
  profile.semantic_form_mask = kConSanCdnaSemanticFormMask;
  profile.moi_dense_route_group_capacity = 31;
  profile.supports_moi_far_dense_atomic_route = true;
  profile.supports_moi_far_dense_barrier_route = true;
  profile.supports_moi_dense_s_call_b64 = true;
  return profile;
}
