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
  profile.accumulator_model = ConSanAccumulatorModel::SelectableVgprBank;
  profile.scalar_placement_model = ConSanScalarPlacementModel::SpillBacked;
  profile.program_analysis = &kConSanCdna5ProgramAnalysisOperations;
  profile.dispatch_identity = ConSanDispatchIdentitySource::CodeObjectLiteral;
  profile.command_processor_workgroup_identity = {
      .grid_x_ttmp = 9u,
      .grid_yz_ttmp = 7u,
      .cluster_workgroup_id_ttmp = 6u,
      .cluster_workgroup_id_low_bit_count = 12u,
  };
  profile.direct_call_form = ConSanDirectCallForm::SCallI64;
  profile.code_transport = ConSanCodeTransportModel::PerKernelOwnerTranslation;
  profile.identity_translation_revision = ProcessorRevision::Gfx1250A0;
  profile.resident_wave_identity = {.hwreg_id = 23, .bit_offset = 0, .bit_width = 10};
  profile.workgroup_shadow_clear = {
      .encoding = ConSanWorkgroupShadowClearEncoding::PackedB128,
      .maximum_lanes = 64,
  };
  profile.vgpr_allocation_granularity_wave64 = 8;
  profile.ordinary_sgpr_limit = 106;
  profile.reserved_ordinary_sgpr_base = 102;
  profile.reserved_ordinary_sgpr_count = 4;
  profile.user_sgpr_initialization_limit = 32;
  profile.address_free_private_limit_bytes = 0x800000u;
  profile.atomic_address_materialization = {
      .flat_and_global = true,
      .buffer_resource = true,
      .lds_byte_offset_token = true,
      .scaled_vglobal = true,
  };
  profile.vector_memory = {
      .instruction_word_count = 3,
      .immediate_offset_bits = 24,
      .flat_vector_only_saddr = 0x7cu,
      .global_vector_only_saddr = 0x7cu,
      .supports_flat_scalar_base = true,
      .vector_offset_extension = ConSanVectorOffsetExtension::Zero,
      .scale_offset = ConSanScaleOffsetCapability::Supported,
  };
  profile.moi_access.dynamic_stack_uses_scalar_reservoir = true;
  profile.moi_access.lazy_workgroup_shadow = true;
  profile.synchronization.workgroup_flat_acquire_wait_fallback = true;
  profile.moi_placement.scratch_vgpr_alignment = 2;
  profile.moi_placement.branch_only_spill_embeds_setup_state = true;
  profile.moi_placement.automatic_dispatch_sgpr_requires_owner_admission = true;
  profile.moi_dispatch_identity_placement =
      ConSanMoiDispatchIdentityPlacement::ScalarThenPersistentVector;
  profile.moi_access_reports_need_explicit_dispatch_identity = false;
  profile.max_group_segment_bytes = static_cast<uint32_t>(ROCJITSU_GFX1250_LDS_SIZE_KB) * 1024u;
  profile.has_cluster_facilities = true;
  profile.has_selectable_vgpr_bank = true;
  profile.requires_even_vgpr_tuples = true;
  profile.requires_split_two_address_lds_relocation = true;
  profile.semantic_form_mask = kConSanGfx1250SemanticFormMask;
  profile.requires_sc_runtime_flat_group_gate = true;
  profile.requires_raw_memory_order_qualifier = true;
  return profile;
}();
