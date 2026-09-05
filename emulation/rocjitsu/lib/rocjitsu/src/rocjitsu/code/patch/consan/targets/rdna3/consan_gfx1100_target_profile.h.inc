// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Complete gfx1100 target-profile specialization.
inline constexpr ConSanTargetProfile kConSanGfx1100TargetProfile = [] {
  ConSanTargetProfile profile = consan_rdna_target_profile_defaults();
  profile.target = ROCJITSU_CODE_TARGET_GFX1100;
  profile.arch = ROCJITSU_CODE_ARCH_RDNA3;
  profile.program_analysis = &kConSanRdna3ProgramAnalysisOperations;
  profile.address_free_private_limit_bytes = 0x1000u;
  profile.private_allocation_granularity_bytes = 16;
  profile.atomic_address_materialization.flat_and_global = true;
  profile.moi_placement.scalar_model = ConSanMoiScalarPlacementModel::LivenessOnly;
  profile.supports_kernarg_preload_overflow_recovery = true;
  return profile;
}();
