// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Complete gfx950 target-profile specialization.
inline constexpr TargetProfile kGfx950TargetProfile = [] {
  TargetProfile profile = cdna3_cdna4_target_profile_defaults();
  profile.target = ROCJITSU_CODE_TARGET_GFX950;
  profile.arch = ROCJITSU_CODE_ARCH_CDNA4;
  profile.device_cache_refresh = DeviceCacheRefreshForm::Cdna4BufferInvSc1;
  return profile;
}();
