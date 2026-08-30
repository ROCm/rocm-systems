// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Complete gfx942 target-profile specialization.
inline constexpr ConSanTargetProfile kConSanGfx942TargetProfile = [] {
  ConSanTargetProfile profile = consan_gfx9_cdna_target_profile_defaults();
  profile.target = ROCJITSU_CODE_TARGET_GFX942;
  profile.arch = ROCJITSU_CODE_ARCH_CDNA3;
  profile.encoding_family = ConSanEncodingFamily::Gfx9Cdna3;
  return profile;
}();
