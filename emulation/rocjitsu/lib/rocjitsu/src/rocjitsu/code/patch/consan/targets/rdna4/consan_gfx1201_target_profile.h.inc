// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Complete gfx1201 target-profile specialization.
inline constexpr TargetProfile kGfx1201TargetProfile = [] {
  TargetProfile profile = rdna_target_profile_defaults();
  profile.target = ROCJITSU_CODE_TARGET_GFX1201;
  profile.arch = ROCJITSU_CODE_ARCH_RDNA4;
  profile.scalar_placement_model = ScalarPlacementModel::SpillBacked;
  profile.program_analysis = &kRdna4ProgramAnalysisOperations;
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
  profile.vector_memory.scale_offset = ScaleOffsetCapability::Disabled;
  profile.access.dynamic_stack_uses_scalar_reservoir = true;
  return profile;
}();
