// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Complete gfx950 target-profile specialization.
inline constexpr ConSanTargetProfile kConSanGfx950TargetProfile = [] {
  ConSanTargetProfile profile = consan_cdna3_cdna4_target_profile_defaults();
  profile.target = ROCJITSU_CODE_TARGET_GFX950;
  profile.arch = ROCJITSU_CODE_ARCH_CDNA4;
  profile.device_cache_refresh = ConSanDeviceCacheRefreshForm::Cdna4BufferInvSc1;
  profile.workgroup_shadow_clear.encoding = ConSanWorkgroupShadowClearEncoding::SplitB32Pair;
  return profile;
}();
