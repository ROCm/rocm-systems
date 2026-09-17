// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Target-profile facts shared by CDNA3 and CDNA4.
[[nodiscard]] constexpr TargetProfile cdna3_cdna4_target_profile_defaults() {
  TargetProfile profile;
  profile.accumulator_model = AccumulatorModel::DescriptorPartitioned;
  profile.scalar_placement_model = ScalarPlacementModel::DescriptorPartitioned;
  profile.program_analysis = &kCdna3Cdna4ProgramAnalysisOperations;
  profile.resident_wave_identity = {.hwreg_id = 4, .bit_offset = 0, .bit_width = 6};
  profile.vgpr_allocation_granularity_wave64 = 8;
  profile.accumulator_offset_granularity = 4;
  profile.ordinary_sgpr_limit = 102;
  profile.user_sgpr_initialization_limit = 16;
  profile.address_free_private_limit_bytes = 0x1000u;
  profile.private_allocation_granularity_bytes = 16;
  profile.max_group_segment_bytes = 64u * 1024u;
  profile.atomic_address_materialization.flat_and_global = true;
  profile.vector_memory.vector_offset_extension = VectorOffsetExtension::Sign;
  profile.native_lds = {
      .mnemonic_dialect = NativeLdsMnemonicDialect::ReadWrite,
      .single_range_atomic_offset_bits = 8,
  };
  profile.access.native_lds_spill_recovery = true;
  profile.access.clobbered_address_spill_reload = true;
  profile.synchronization.workgroup_flat_acquire_wait_fallback = true;
  profile.placement = {
      .scratch_vgpr_alignment = 2,
      .branch_only_spill_embeds_setup_state = true,
      .automatic_dispatch_sgpr_requires_owner_admission = true,
      .full_workgroup_payload_consumption = WorkgroupPayloadConsumption::EntryCapture,
  };
  profile.supports_kernarg_preload_overflow_recovery = true;
  profile.requires_even_vgpr_tuples = true;
  profile.flat_compare_swap_data_pair_alignment = 2;
  profile.semantic_form_mask = kCdnaSemanticFormMask;
  return profile;
}
