// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "b96_fixture.h"
#include "remaining_subword_fixture.h"
#include "scalar_two_fixture.h"
#include "subword_fixture.h"
#include "two_address_fixture.h"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::test {

inline std::vector<LdsFormCase> retained_lds_cases() {
  struct NamedWords {
    std::string_view form;
    std::array<uint32_t, 2> words;
  };
  // Existing concrete witnesses from cdna_lds_fixture.h, b96_fixture.h,
  // ds_address_encoded_fields_test.cpp and two_address_fixture.h.
  constexpr std::array<NamedWords, 4> cdna_basic = {{
      {.form = "ds_write_b32", .words = {0xd81a0004u, 0x00000302u}},
      {.form = "ds_write_b128", .words = {0xd9be0010u, 0x00000401u}},
      {.form = "ds_read_b96", .words = {0xd9fc0004u, 0x04000002u}},
      {.form = "ds_write_b96", .words = {0xd9bc0004u, 0x00000402u}},
  }};
  constexpr std::array<NamedWords, 4> rdna_basic = {{
      {.form = "ds_store_b32", .words = {0xd8340004u, 0x00000302u}},
      {.form = "ds_load_b32", .words = {0xd8d80004u, 0x01000002u}},
      {.form = "ds_load_b96", .words = {0xdbf80004u, 0x04000002u}},
      {.form = "ds_store_b96", .words = {0xdb780004u, 0x00000402u}},
  }};
  constexpr std::array<NamedWords, 2> cdna_two = {{
      {.form = "ds_read2_b64", .words = {0xd8ee0301u, 0x04000002u}},
      {.form = "ds_write2st64_b64", .words = {0xd89e0502u, 0x00060402u}},
  }};
  constexpr std::array<NamedWords, 2> rdna_two = {{
      {.form = "ds_load_2addr_b64", .words = {0xd9dc0301u, 0x04000002u}},
      {.form = "ds_store_2addr_b32", .words = {0xd8380502u, 0x00040302u}},
  }};
  std::vector<LdsFormCase> forms;
  for (const B96Target &target : kB96Targets) {
    for (const NamedWords &fixture : target.cdna ? cdna_basic : rdna_basic)
      forms.push_back({.target = target.target,
                       .machine = target.machine,
                       .arch = target.arch,
                       .form = fixture.form,
                       .words = fixture.words});
  }
  for (const TwoAddressTarget &target : kTwoAddressTargets) {
    for (const NamedWords &fixture : target.cdna ? cdna_two : rdna_two)
      forms.push_back({.target = target.target,
                       .machine = target.machine,
                       .arch = target.arch,
                       .form = fixture.form,
                       .words = fixture.words});
  }
  for (std::span<const LdsFormCase> group :
       {std::span<const LdsFormCase>(kScalarTwoCases), std::span<const LdsFormCase>(kSubwordCases),
        std::span<const LdsFormCase>(kRemainingSubwordCases)})
    forms.insert(forms.end(), group.begin(), group.end());
  return forms;
}

} // namespace rocjitsu::test
