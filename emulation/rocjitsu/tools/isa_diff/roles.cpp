// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "roles.h"

#include "effects.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <string_view>

namespace rocjitsu::isa_diff {
namespace {

bool starts_with(std::string_view s, std::string_view p) { return s.substr(0, p.size()) == p; }

bool in(std::string_view s, std::initializer_list<std::string_view> names) {
  return std::ranges::find(names, s) != names.end();
}

/// Instructions whose two value sources are worth pairing (carry, borrow, overflow,
/// ordering, equality, complement).
bool is_pair_instruction(std::string_view m) {
  return in(m, {"s_add_u32",        "s_sub_u32",       "s_add_i32",         "s_sub_i32",
                "s_addc_u32",       "s_subb_u32",      "s_min_i32",         "s_min_u32",
                "s_max_i32",        "s_max_u32",       "s_absdiff_i32",     "s_mul_i32",
                "s_mul_hi_u32",     "s_mul_hi_i32",    "s_lshl1_add_u32",   "s_lshl2_add_u32",
                "s_lshl3_add_u32",  "s_lshl4_add_u32", "s_addk_i32",        "s_mulk_i32",
                "s_cselect_b32",    "s_cselect_b64",   "s_pack_ll_b32_b16", "s_pack_lh_b32_b16",
                "s_pack_hh_b32_b16"}) ||
         starts_with(m, "s_cmp_") || starts_with(m, "s_cmpk_") || starts_with(m, "s_and_b") ||
         starts_with(m, "s_or_b") || starts_with(m, "s_xor_b") || starts_with(m, "s_andn2_b") ||
         starts_with(m, "s_orn2_b") || starts_with(m, "s_nand_b") || starts_with(m, "s_nor_b") ||
         starts_with(m, "s_xnor_b");
}

unsigned low_loc(const LocSet &locs) {
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (locs.test(l))
      return l;
  return kNumLocs;
}

void mark(RegionRoles &r, const SourceOperand &s, uint32_t role) {
  for (unsigned l = 0; l < kNumLocs; ++l)
    if (s.locs.test(l))
      r.roles[l] |= role;
}

void add_instruction(RegionRoles &r, const Instruction &inst) {
  const std::string m(inst.mnemonic());
  std::vector<SourceOperand> src = source_operands(inst);
  for (const auto &s : src)
    r.read |= s.locs;
  // Drop fieldless operands (SCC carry-in etc.): they are not value sources here.
  std::erase_if(src, [](const SourceOperand &s) { return s.fieldless; });
  if (src.empty())
    return;
  auto at = [&](size_t i) -> const SourceOperand * { return i < src.size() ? &src[i] : nullptr; };

  if (starts_with(m, "s_lshl_b") || starts_with(m, "s_lshr_b") || starts_with(m, "s_ashr_i")) {
    if (auto *s = at(1))
      mark(r, *s, kRoleShiftAmount);
  } else if (starts_with(m, "s_bfm_b")) {
    for (size_t i : {0u, 1u})
      if (auto *s = at(i))
        mark(r, *s, kRoleShiftAmount);
  } else if (starts_with(m, "s_bfe_")) {
    if (auto *s = at(1))
      mark(r, *s, kRoleBfeControl);
  } else if (starts_with(m, "s_bitset") || starts_with(m, "s_bitcmp")) {
    if (auto *s = at(1))
      mark(r, *s, kRoleBitIndex);
  } else if (starts_with(m, "s_sext_i32_")) {
    if (auto *s = at(0))
      mark(r, *s, kRoleSextInput);
  } else if (m == "s_abs_i32") {
    if (auto *s = at(0))
      mark(r, *s, kRoleSignedEdge);
  }

  if (!is_pair_instruction(m) || src.size() < 2)
    return;
  const SourceOperand &a = src[0];
  const SourceOperand &b = src[1];
  mark(r, a, kRoleArith);
  mark(r, b, kRoleArith);
  if (a.locs.any() && b.locs.any() && a.locs.count() == b.locs.count() && a.locs.count() <= 2) {
    OperandPair p{low_loc(a.locs), low_loc(b.locs), a.locs.count() == 2, m};
    if (std::ranges::find(r.pairs, p) == r.pairs.end())
      r.pairs.push_back(p);
  } else if (a.locs.count() == 1 && b.constant) {
    r.const_relations.push_back({low_loc(a.locs), *b.constant, m});
  } else if (b.locs.count() == 1 && a.constant) {
    r.const_relations.push_back({low_loc(b.locs), *a.constant, m});
  }
}

} // namespace

RegionRoles infer_roles(const std::vector<RegionInst> &old_insts,
                        const std::vector<RegionInst> &new_insts) {
  RegionRoles r;
  r.decoded = true;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  if (!decoder) {
    r.decoded = false;
    return r;
  }
  for (const auto *side : {&old_insts, &new_insts}) {
    for (const RegionInst &ri : *side) {
      std::vector<uint32_t> window = ri.words;
      window.resize(window.size() + 3, 0);
      auto decoded = decoder->decode_window(window, ri.pc);
      if (decoded.failed()) {
        r.decoded = false;
        continue;
      }
      add_instruction(r, *decoded.value());
    }
  }
  // Same constant relation from both sides counts once.
  std::vector<ConstRelation> unique;
  for (const auto &c : r.const_relations)
    if (std::ranges::none_of(unique, [&](const ConstRelation &u) {
          return u.loc == c.loc && u.constant == c.constant;
        }))
      unique.push_back(c);
  r.const_relations = std::move(unique);
  return r;
}

std::string role_names(uint32_t roles) {
  static constexpr std::array<std::pair<uint32_t, const char *>, 6> kNames{{
      {kRoleShiftAmount, "shift_amount"},
      {kRoleBitIndex, "bit_index"},
      {kRoleBfeControl, "bfe_control"},
      {kRoleSextInput, "sext_input"},
      {kRoleSignedEdge, "signed_edge"},
      {kRoleArith, "arith"},
  }};
  std::string out;
  for (const auto &[bit, name] : kNames)
    if (roles & bit) {
      if (!out.empty())
        out += ',';
      out += name;
    }
  return out;
}

} // namespace rocjitsu::isa_diff
