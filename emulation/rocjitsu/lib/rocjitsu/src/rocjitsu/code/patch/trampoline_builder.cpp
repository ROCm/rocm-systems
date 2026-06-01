// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/trampoline_builder.h"

#include "rocjitsu/code/patch/instruction_builder.h"

#include <cstring>

namespace rocjitsu {

namespace {

void report(std::string *out, const char *msg) {
  if (out)
    *out = msg;
}

// DBI currently only supports inlined nops as the instrumentation body. Any
// other inline-asm shape needs clobber declarations and liveness handling,
// which arrive in a later milestone — this guardrail keeps the first DBI
// build from silently emitting bytes whose register footprint hasn't been
// accounted for.
[[nodiscard]] bool check_inline_nop_smoke_guardrail(const TrampolinePlan &plan,
                                                    std::string *err) {
  if (!plan.emit_original) {
    report(err, "trampoline plan: emit_original must be true for the inline-nop smoke build");
    return false;
  }
  if (!plan.after_items.empty()) {
    report(err, "trampoline plan: after_items must be empty for the inline-nop smoke build");
    return false;
  }
  if (plan.before_items.size() != 1 || plan.before_items[0].words.size() != 1 ||
      plan.before_items[0].words[0] != build_s_nop(0, plan.arch)) {
    report(err,
           "trampoline plan: before_items must be exactly { { s_nop 0 } } "
           "for the inline-nop smoke build");
    return false;
  }
  return true;
}

[[nodiscard]] bool check_size_and_words(const TrampolinePlan &plan, std::string *err) {
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

void append_word(std::vector<uint8_t> &dst, uint32_t w) {
  uint8_t buf[sizeof(w)];
  std::memcpy(buf, &w, sizeof(w));
  dst.insert(dst.end(), buf, buf + sizeof(w));
}

} // namespace

std::optional<TrampolineBytes> TrampolineBuilder::build(const TrampolinePlan &plan,
                                                        std::string *error_out) {
  if (!check_inline_nop_smoke_guardrail(plan, error_out))
    return std::nullopt;
  if (!check_size_and_words(plan, error_out))
    return std::nullopt;

  // Forward branch: from the anchor to the trampoline.
  const auto fwd = compute_sopp_branch_simm16(plan.anchor_offset, plan.trampoline_offset);
  if (!fwd) {
    report(error_out, "relocation trampoline forward branch exceeds s_branch simm16");
    return std::nullopt;
  }

  // Lay out trampoline body so we can compute the return branch offset. For
  // the inline-nop smoke build the body is exactly [s_nop 0, original*]; the
  // generic loops below also handle the eventual multi-item inline-asm shape.
  std::vector<uint32_t> body;
  body.reserve(1 + plan.original_words.size() + 1);
  for (const InlineAsmItem &item : plan.before_items)
    body.insert(body.end(), item.words.begin(), item.words.end());
  if (plan.emit_original)
    body.insert(body.end(), plan.original_words.begin(), plan.original_words.end());
  for (const InlineAsmItem &item : plan.after_items)
    body.insert(body.end(), item.words.begin(), item.words.end());

  const uint64_t return_branch_pc =
      plan.trampoline_offset + body.size() * sizeof(uint32_t);
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

} // namespace rocjitsu
