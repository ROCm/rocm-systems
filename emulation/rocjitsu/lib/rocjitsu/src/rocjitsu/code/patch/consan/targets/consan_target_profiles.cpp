// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/targets/consan_target_profiles.h"

#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops_internal.h"

namespace rocjitsu::consan {

#include "rocjitsu/code/patch/consan/targets/shared/cdna3_cdna4/consan_cdna3_cdna4_target_profile.h.inc"
#include "rocjitsu/code/patch/consan/targets/shared/rdna3_rdna4/consan_rdna_target_profile.h.inc"

#include "rocjitsu/code/patch/consan/targets/cdna3/consan_gfx942_target_profile.h.inc"
#include "rocjitsu/code/patch/consan/targets/cdna4/consan_gfx950_target_profile.h.inc"
#include "rocjitsu/code/patch/consan/targets/cdna5/consan_gfx1250_target_profile.h.inc"
#include "rocjitsu/code/patch/consan/targets/rdna3/consan_gfx1100_target_profile.h.inc"
#include "rocjitsu/code/patch/consan/targets/rdna4/consan_gfx1201_target_profile.h.inc"

namespace {

constexpr std::array kProfiles = {
    kGfx942TargetProfile,  kGfx950TargetProfile,  kGfx1100TargetProfile,
    kGfx1201TargetProfile, kGfx1250TargetProfile,
};

static_assert(target_profiles_are_valid(kProfiles));
static_assert(!kGfx942TargetProfile.requires_split_two_address_lds_relocation);
static_assert(!kGfx950TargetProfile.requires_split_two_address_lds_relocation);
static_assert(!kGfx1100TargetProfile.requires_split_two_address_lds_relocation);
static_assert(!kGfx1201TargetProfile.requires_split_two_address_lds_relocation);
static_assert(kGfx1250TargetProfile.requires_split_two_address_lds_relocation);
static_assert(!kGfx942TargetProfile.requires_raw_memory_order_qualifier);
static_assert(!kGfx950TargetProfile.requires_raw_memory_order_qualifier);
static_assert(kGfx1100TargetProfile.requires_raw_memory_order_qualifier);
static_assert(kGfx1201TargetProfile.requires_raw_memory_order_qualifier);
static_assert(kGfx1250TargetProfile.requires_raw_memory_order_qualifier);

} // namespace

const std::array<TargetProfile, 5> kTargetProfiles = kProfiles;

const TargetProfile *target_profile(rj_code_target_id_t target) {
  for (const TargetProfile &profile : kTargetProfiles) {
    if (target == profile.target)
      return &profile;
  }
  return nullptr;
}

const TargetProfile *target_profile(rj_code_arch_t arch) {
  for (const TargetProfile &profile : kTargetProfiles) {
    if (arch == profile.arch)
      return &profile;
  }
  return nullptr;
}

bool target_profiles_are_valid() { return target_profiles_are_valid(kTargetProfiles); }

} // namespace rocjitsu::consan
