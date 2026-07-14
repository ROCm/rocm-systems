// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/trampoline_builder.h"

#include "rocjitsu/code/patch/error_report.h"
#include "rocjitsu/code/patch/instruction_builder.h"

#include <cstring>

namespace rocjitsu {

namespace {

[[nodiscard]] bool check_size_and_words(const TrampolinePlan &plan, std::string *err) {
  if (plan.arch == ROCJITSU_CODE_ARCH_INVALID) {
    report(err, "trampoline plan: arch was not set");
    return false;
  }
  if (plan.original_size != 4 && plan.original_size != 8) {
    report(err, "trampoline plan: original_size must be 4 or 8");
    return false;
  }
  const size_t expected_words = plan.original_size / sizeof(uint32_t);
  if (plan.original_words.size() != expected_words) {
    report(err, "trampoline plan: original_words count does not match original_size");
    return false;
  }
  return true;
}

// TODO: the following functions are very similar to those in LivenessAnalysis
// but they take a RegisterSet instead of an Instruction. These functions
// probably belong there and with some refactoring, we can probably reduce the
// duplicated code. Would like another opinion before making that call though.
// `any_sgpr_in_range` is similar to a test used by `find_free_*`
// `find_free_sgpr_pair` is similar to `find_free_sgpr_pair`
// `find_free_sgpr` is similar to `find_free_sgpr`
[[nodiscard]] bool any_sgpr_in_range(const RegisterSet &set, uint16_t base, uint16_t count) {
  for (uint16_t i = 0; i < count; ++i) {
    if (set.contains(RegisterRef{RegClass::SGPR, static_cast<uint16_t>(base + i), 1}))
      return true;
  }
  return false;
}


// First even-aligned SGPR pair with both lanes free of @p unavailable, within the
// conservative cross-family allocatable bound. nullopt if none.
[[nodiscard]] std::optional<uint16_t> find_free_sgpr_pair(const RegisterSet &unavailable) {
  for (uint16_t base = 0; base + 1 < REGISTER_SET_ALLOCATABLE_SGPRS; base += 2) {
    if (!any_sgpr_in_range(unavailable, base, 2))
      return base;
  }
  return std::nullopt;
}

// First single SGPR free of @p unavailable, within the allocatable bound.
[[nodiscard]] std::optional<uint16_t> find_free_sgpr(const RegisterSet &unavailable) {
  for (uint16_t base = 0; base < REGISTER_SET_ALLOCATABLE_SGPRS; ++base) {
    if (!unavailable.contains(RegisterRef{RegClass::SGPR, base, 1}))
      return base;
  }
  return std::nullopt;
}

// Appends @p w to @p dst in host byte order. AMDGPU code objects are little-
// endian and rocjitsu only supports little-endian hosts (matches DBT's
// memcpy convention in binary_translator.cpp); if either invariant ever
// changes, this helper needs an explicit byte-swap.
void append_word(std::vector<uint8_t> &dst, uint32_t w) {
  uint8_t buf[sizeof(w)];
  std::memcpy(buf, &w, sizeof(w));
  dst.insert(dst.end(), buf, buf + sizeof(w));
}

} // namespace

std::optional<TrampolineBytes> TrampolineBuilder::build(const TrampolinePlan &plan,
                                                        std::string *error_out) {
  if (!check_size_and_words(plan, error_out))
    return std::nullopt;

  // Forward branch: from the anchor to the trampoline.
  const auto fwd = compute_sopp_branch_simm16(plan.anchor_offset, plan.trampoline_offset);
  if (!fwd) {
    report(error_out, "relocation trampoline forward branch exceeds s_branch simm16");
    return std::nullopt;
  }

  // Lay out trampoline body so we can compute the return branch offset. The
  // generic loops below handle any multi-item inline-asm shape; no reserve
  // hint because the per-item word counts aren't known up front and
  // vector::insert handles growth.
  std::vector<uint32_t> body;
  for (const InlineAsmItem &item : plan.before_items)
    body.insert(body.end(), item.words.begin(), item.words.end());
  if (plan.emit_original)
    body.insert(body.end(), plan.original_words.begin(), plan.original_words.end());
  for (const InlineAsmItem &item : plan.after_items)
    body.insert(body.end(), item.words.begin(), item.words.end());

  const uint64_t return_branch_pc = plan.trampoline_offset + body.size() * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(return_branch_pc, plan.return_target);
  if (!ret) {
    report(error_out, "relocation trampoline return branch exceeds s_branch simm16");
    return std::nullopt;
  }

  TrampolineBytes out;
  out.patched_anchor_bytes.reserve(plan.original_size);
  append_word(out.patched_anchor_bytes, build_s_branch(*fwd, plan.arch));
  if (plan.original_size == 8)
    append_word(out.patched_anchor_bytes, build_s_nop(0, plan.arch));

  out.trampoline_words = std::move(body);
  out.trampoline_words.push_back(build_s_branch(*ret, plan.arch));
  return out;
}

bool TrampolineBuilder::plan_probe_call(TrampolinePlan &plan, const RegisterSet &live_at_anchor,
                                        const RegisterSet &probe_body_clobbers,
                                        std::string *error_out) {
  // Link pair is the fixed ABI pair s[30:31] for now, so reject if either lane
  // is live
  // TODO: Supporting other pairs is deferred
  constexpr uint16_t kLinkPairBase = 30;
  if (any_sgpr_in_range(live_at_anchor, kLinkPairBase, 2)) {
    report(error_out, "probe-call resource planning: return-link pair s[30:31] is live at the "
                      "anchor; cannot yet save a live link pair");
    return false;
  }

  RegisterSet link_pair;
  link_pair.expand(RegisterRef{RegClass::SGPR, kLinkPairBase, 2});

  // Target-address pair: dead, even-aligned, and not the link pair. It is read by
  // s_swappc before the probe body runs, so it may overlap probe_body_clobbers.
  const RegisterSet target_unavail = live_at_anchor | link_pair;
  const std::optional<uint16_t> target_pair = find_free_sgpr_pair(target_unavail);
  if (!target_pair) {
    report(error_out, "probe-call resource planning: no dead SGPR pair available for the probe "
                      "target address");
    return false;
  }

  RegisterSet target_pair_set;
  target_pair_set.expand(RegisterRef{RegClass::SGPR, *target_pair, 2});

  // SCC temp: lives across the call (saved before materialization, restored
  // after), so it must avoid the live set, the link/target pairs, AND the probe
  // body clobbers.
  // TODO: allow for reuse of target_pair if unavailable
  const RegisterSet scc_unavail = target_unavail | target_pair_set | probe_body_clobbers;
  const std::optional<uint16_t> scc_temp = find_free_sgpr(scc_unavail);
  if (!scc_temp) {
    report(error_out, "probe-call resource planning: no dead SGPR available for the SCC "
                      "preservation temp");
    return false;
  }

  // Word count is derived from the resource decisions, not a fixed envelope size.
  // Each add/addc uses the 32-bit literal form (instruction + literal word) so the
  // count is independent of the (layout-dependent) addend values.
  uint32_t before_words = 0;
  before_words += 1;     // s_getpc_b64
  before_words += 2 + 2; // s_add_u32 + literal, s_addc_u32 + literal
  before_words += 1;     // s_swappc_b64
  if (plan.preserve_scc)
    before_words += 2; // s_cselect_b32 (save) + s_cmp_lg_u32 (restore)

  plan.is_probe_call = true;
  plan.link_pair_base = kLinkPairBase;
  plan.target_pair_base = *target_pair;
  plan.scc_temp = *scc_temp;
  plan.before_word_count = before_words;

  plan.builder_clobbers = link_pair | target_pair_set;
  plan.builder_clobbers.expand(RegisterRef{RegClass::SGPR, *scc_temp, 1});
  return true;
}

} // namespace rocjitsu
