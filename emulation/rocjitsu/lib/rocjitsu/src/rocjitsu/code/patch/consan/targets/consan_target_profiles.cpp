// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/targets/consan_target_profiles.h"

#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops_internal.h"

namespace rocjitsu {

#include "rocjitsu/code/patch/consan/targets/shared/cdna3_cdna4/consan_cdna3_cdna4_target_profile.h.inc"
#include "rocjitsu/code/patch/consan/targets/shared/rdna3_rdna4/consan_rdna_target_profile.h.inc"

#include "rocjitsu/code/patch/consan/targets/cdna3/consan_gfx942_target_profile.h.inc"
#include "rocjitsu/code/patch/consan/targets/cdna4/consan_gfx950_target_profile.h.inc"
#include "rocjitsu/code/patch/consan/targets/cdna5/consan_gfx1250_target_profile.h.inc"
#include "rocjitsu/code/patch/consan/targets/rdna3/consan_gfx1100_target_profile.h.inc"
#include "rocjitsu/code/patch/consan/targets/rdna4/consan_gfx1201_target_profile.h.inc"

namespace {

constexpr std::array kProfiles = {
    kConSanGfx942TargetProfile,  kConSanGfx950TargetProfile,  kConSanGfx1100TargetProfile,
    kConSanGfx1201TargetProfile, kConSanGfx1250TargetProfile,
};

static_assert(consan_target_profiles_are_valid(kProfiles));
static_assert(!kConSanGfx942TargetProfile.requires_split_two_address_lds_relocation);
static_assert(!kConSanGfx950TargetProfile.requires_split_two_address_lds_relocation);
static_assert(!kConSanGfx1100TargetProfile.requires_split_two_address_lds_relocation);
static_assert(!kConSanGfx1201TargetProfile.requires_split_two_address_lds_relocation);
static_assert(kConSanGfx1250TargetProfile.requires_split_two_address_lds_relocation);
static_assert(!kConSanGfx942TargetProfile.requires_raw_memory_order_qualifier);
static_assert(!kConSanGfx950TargetProfile.requires_raw_memory_order_qualifier);
static_assert(kConSanGfx1100TargetProfile.requires_raw_memory_order_qualifier);
static_assert(kConSanGfx1201TargetProfile.requires_raw_memory_order_qualifier);
static_assert(kConSanGfx1250TargetProfile.requires_raw_memory_order_qualifier);

} // namespace

const std::array<ConSanTargetProfile, 5> kConSanTargetProfiles = kProfiles;

const ConSanTargetProfile *consan_target_profile(rj_code_target_id_t target) {
  for (const ConSanTargetProfile &profile : kConSanTargetProfiles) {
    if (target == profile.target)
      return &profile;
  }
  return nullptr;
}

const ConSanTargetProfile *consan_target_profile(rj_code_arch_t arch) {
  for (const ConSanTargetProfile &profile : kConSanTargetProfiles) {
    if (arch == profile.arch)
      return &profile;
  }
  return nullptr;
}

bool consan_target_profiles_are_valid() {
  return consan_target_profiles_are_valid(kConSanTargetProfiles);
}

} // namespace rocjitsu
