// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Complete gfx1201 target-profile specialization.
inline constexpr ConSanTargetProfile kConSanGfx1201TargetProfile = [] {
  ConSanTargetProfile profile = consan_rdna_target_profile_defaults();
  profile.target = ROCJITSU_CODE_TARGET_GFX1201;
  profile.arch = ROCJITSU_CODE_ARCH_RDNA4;
  profile.program_analysis = &kConSanRdna4ProgramAnalysisOperations;
  profile.command_processor_workgroup_identity = {
      .grid_x_ttmp = 9u,
      .grid_yz_ttmp = 7u,
      .cluster_workgroup_id_ttmp = std::nullopt,
  };
  profile.address_free_private_limit_bytes = 0x800000u;
  profile.atomic_address_materialization.flat_and_global = true;
  profile.atomic_address_materialization.buffer_resource = true;
  profile.vector_memory.instruction_word_count = 3;
  profile.vector_memory.immediate_offset_bits = 24;
  profile.vector_memory.supports_flat_scalar_base = true;
  profile.vector_memory.scale_offset = ConSanScaleOffsetCapability::Disabled;
  profile.moi_access.dynamic_stack_uses_scalar_reservoir = true;
  profile.moi_access.lazy_workgroup_shadow = true;
  profile.moi_placement.scalar_model = ConSanMoiScalarPlacementModel::SpillBacked;
  profile.moi_dispatch_identity_placement = ConSanMoiDispatchIdentityPlacement::ScalarThenLiteral;
  profile.supports_sc_inline_flat_trap_rewrite = true;
  return profile;
}();
