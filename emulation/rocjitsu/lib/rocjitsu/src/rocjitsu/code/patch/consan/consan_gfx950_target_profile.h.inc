// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Complete gfx950 target-profile specialization.
inline constexpr ConSanTargetProfile kConSanGfx950TargetProfile = [] {
  ConSanTargetProfile profile = consan_gfx9_cdna_target_profile_defaults();
  profile.target = ROCJITSU_CODE_TARGET_GFX950;
  profile.arch = ROCJITSU_CODE_ARCH_CDNA4;
  profile.encoding_family = ConSanEncodingFamily::Gfx9Cdna4;
  profile.workgroup_shadow_clear.encoding = ConSanWorkgroupShadowClearEncoding::SplitB32Pair;
  profile.sc_dense_route_identity = ConSanScDenseRouteIdentity::ExplicitInlineKey;
  return profile;
}();
