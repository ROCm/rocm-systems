// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Complete gfx1201 target-profile specialization.
inline constexpr ConSanTargetProfile kConSanGfx1201TargetProfile = [] {
  ConSanTargetProfile profile = consan_rdna_target_profile_defaults();
  profile.target = ROCJITSU_CODE_TARGET_GFX1201;
  profile.arch = ROCJITSU_CODE_ARCH_RDNA4;
  profile.encoding_family = ConSanEncodingFamily::Gfx12;
  profile.command_processor_workgroup_identity = {
      .grid_x_ttmp = 9u,
      .grid_yz_ttmp = 7u,
      .cluster_workgroup_id_ttmp = std::nullopt,
  };
  profile.address_free_private_limit_bytes = 0x800000u;
  profile.atomic_address_materialization.flat_and_global = true;
  profile.atomic_address_materialization.buffer_resource = true;
  profile.moi_dispatch_identity_placement = ConSanMoiDispatchIdentityPlacement::ScalarThenLiteral;
  profile.supports_moi_far_dense_barrier_route = true;
  profile.supports_moi_dense_s_call_b64 = true;
  profile.supports_sc_branch_only_route = true;
  profile.supports_sc_inline_flat_trap_rewrite = true;
  profile.sc_dense_route_identity = ConSanScDenseRouteIdentity::ExplicitInlineKey;
  return profile;
}();
