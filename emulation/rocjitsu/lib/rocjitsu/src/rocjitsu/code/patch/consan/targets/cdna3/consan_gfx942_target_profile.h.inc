// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Complete gfx942 target-profile specialization.
inline constexpr TargetProfile kGfx942TargetProfile = [] {
  TargetProfile profile = cdna3_cdna4_target_profile_defaults();
  profile.target = ROCJITSU_CODE_TARGET_GFX942;
  profile.arch = ROCJITSU_CODE_ARCH_CDNA3;
  profile.device_cache_refresh = DeviceCacheRefreshForm::Cdna3BufferInvSc1;
  return profile;
}();
