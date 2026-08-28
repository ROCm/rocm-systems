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
  for (uint16_t base = 0; static_cast<size_t>(base) + 1 < REGISTER_SET_ALLOCATABLE_SGPRS;
       base += 2) {
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

bool TrampolineBuilder::plan_probe_call(TrampolinePlan &plan, ProbeCallingConvention cc,
                                        const RegisterSet &live_at_anchor,
                                        const RegisterSet &probe_body_clobbers,
                                        std::string *error_out) {
  // The link pair is whatever the probe's calling convention returns through,
  // so the call site and the probe body agree on one pair. An unknown
  // convention cannot be called.
  const std::optional<uint16_t> link_base = link_pair_for(cc);
  if (!link_base) {
    report(error_out, "probe-call resource planning: unknown probe calling convention; cannot "
                      "derive the return-link pair");
    return false;
  }
  const uint16_t kLinkPairBase = *link_base;

  // Reject if either lane of the link pair is live at the anchor; saving a live
  // link pair is deferred.
  if (any_sgpr_in_range(live_at_anchor, kLinkPairBase, 2)) {
    report(error_out, "probe-call resource planning: return-link pair s[30:31] is live at the "
                      "anchor; cannot yet save a live link pair");
    return false;
  }

  RegisterSet link_pair;
  link_pair.expand(RegisterRef{RegClass::SGPR, kLinkPairBase, 2});

  // NOTE: target/scc selection scans the conservative cross-ISA allocatable
  // bound (REGISTER_SET_ALLOCATABLE_SGPRS), not the patched kernel's actual
  // .sgpr_count. This is safe today only because the scan returns the lowest
  // dead registers, and we currently require the s30/31 regs. Handling this
  // is deferred.
  // Target-address pair: dead, even-aligned, and not the link pair. It is
  // read by s_swappc before the probe body runs, so it may overlap
  // probe_body_clobbers.
  const RegisterSet target_unavail = live_at_anchor | link_pair;
  const std::optional<uint16_t> target_pair = find_free_sgpr_pair(target_unavail);
  if (!target_pair) {
    report(error_out, "probe-call resource planning: no dead SGPR pair available for the probe "
                      "target address");
    return false;
  }

  RegisterSet target_pair_set;
  target_pair_set.expand(RegisterRef{RegClass::SGPR, *target_pair, 2});

  // SCC temp: only needed when we preserve SCC across the call. It lives across
  // the call (saved before materialization, restored after), so it must avoid
  // the live set, the link/target pairs, AND the probe body clobbers. When SCC
  // is not preserved we reserve nothing
  // TODO: allow for reuse of target_pair if unavailable
  std::optional<uint16_t> scc_temp;
  if (plan.preserve_scc) {
    const RegisterSet scc_unavail = target_unavail | target_pair_set | probe_body_clobbers;
    scc_temp = find_free_sgpr(scc_unavail);
    if (!scc_temp) {
      report(error_out, "probe-call resource planning: no dead SGPR available for the SCC "
                        "preservation temp");
      return false;
    }
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
  if (scc_temp)
    plan.scc_temp = *scc_temp;
  plan.before_word_count = before_words;

  plan.builder_clobbers = link_pair | target_pair_set;
  if (scc_temp)
    plan.builder_clobbers.expand(RegisterRef{RegClass::SGPR, *scc_temp, 1});
  return true;
}

std::optional<TrampolineBytes> TrampolineBuilder::emit_probe_call(const TrampolinePlan &plan,
                                                                  std::string *error_out) {
  if (!plan.is_probe_call) {
    report(error_out, "emit_probe_call: plan is not a probe call (run plan_probe_call first)");
    return std::nullopt;
  }

  const uint16_t link = plan.link_pair_base;
  const uint16_t target_lo = plan.target_pair_base;
  const uint16_t target_hi = static_cast<uint16_t>(plan.target_pair_base + 1);
  // Literal-constant scalar source code; the 32-bit literal follows the word.
  constexpr uint16_t kLiteralConstant = 0xFF;

  std::vector<uint32_t> env;

  // SCC save (prologue): capture SCC into the temp without disturbing it. The
  // matching restore is emitted after the call but still before the relocated
  // original.
  if (plan.preserve_scc)
    env.push_back(build_s_cselect_b32(plan.scc_temp, scalar_positive_inline_u32(1),
                                      scalar_positive_inline_u32(0), plan.arch));

  // Target-address materialization. s_getpc_b64 writes the runtime VA of the
  // *next* instruction (the s_add_u32 below) into the target pair; the
  // build-time delta to the probe body is then folded in via the 64-bit add
  // chain (s_add_u32 sets carry -> SCC, s_addc_u32 consumes it). Both sides are
  // .text-relative and share the load base, so the delta is a pure layout
  // distance. The adds always use the literal form so the word count is
  // independent of the (layout-dependent) delta value (see before_word_count).
  const size_t getpc_index = env.size();
  env.push_back(build_s_getpc_b64(target_lo, plan.arch));
  const uint64_t va_after_getpc =
      plan.trampoline_offset + static_cast<uint64_t>(getpc_index + 1) * sizeof(uint32_t);
  const uint64_t delta = static_cast<uint64_t>(static_cast<int64_t>(plan.probe_target_offset) -
                                               static_cast<int64_t>(va_after_getpc));
  env.push_back(build_s_add_u32(target_lo, target_lo, kLiteralConstant, plan.arch));
  env.push_back(static_cast<uint32_t>(delta & 0xFFFFFFFFu));
  env.push_back(build_s_addc_u32(target_hi, target_hi, kLiteralConstant, plan.arch));
  env.push_back(static_cast<uint32_t>(delta >> 32));

  // The call: writes the return PC into the cc-derived link pair, jumps to the
  // materialized target. The probe returns here via s_setpc_b64 of the same pair.
  env.push_back(build_s_swappc_b64(link, target_lo, plan.arch));

  // SCC restore (epilogue): set SCC from the saved temp before the relocated
  // original runs.
  if (plan.preserve_scc)
    env.push_back(build_s_cmp_lg_u32(plan.scc_temp, scalar_positive_inline_u32(0), plan.arch));

  // Plan/emit drift guard: the planner committed to this many envelope words and
  // the orchestrator sized the layout around it. A mismatch means the two
  // disagree about the envelope shape.
  if (env.size() != plan.before_word_count) {
    report(error_out, "emit_probe_call: synthesized envelope word count does not match the planned "
                      "before_word_count");
    return std::nullopt;
  }

  // Hand the synthesized envelope to build() for layout and branch math so the
  // SOPP range checks are shared with the inline path.
  TrampolinePlan emit_plan = plan;
  emit_plan.before_items.assign(1, InlineAsmItem{std::move(env)});
  emit_plan.after_items.clear();
  emit_plan.emit_original = true;
  return build(emit_plan, error_out);
}

} // namespace rocjitsu
