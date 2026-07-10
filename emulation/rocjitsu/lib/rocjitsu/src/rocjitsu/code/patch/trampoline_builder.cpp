// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/trampoline_builder.h"

#include "rocjitsu/code/patch/error_report.h"
#include "rocjitsu/code/patch/instruction_builder.h"

#include <cstring>
#include <limits>

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

DbiPatchPlacementPlanner::DbiPatchPlacementPlanner(rj_code_arch_t arch, uint64_t original_text_size)
    : arch_(arch), original_text_size_(original_text_size), appended_cursor_(original_text_size) {}

bool DbiPatchPlacementPlanner::range_is_free(uint64_t begin, uint64_t end) const {
  if (begin >= end)
    return false;
  for (const auto [occupied_begin, occupied_end] : occupied_ranges_) {
    if (begin < occupied_end && occupied_begin < end)
      return false;
  }
  return true;
}

void DbiPatchPlacementPlanner::reserve_range(uint64_t begin, uint64_t end) {
  occupied_ranges_.emplace_back(begin, end);
}

std::optional<DbiPatchPlacement>
DbiPatchPlacementPlanner::plan(const DbiPatchPlacementRequest &request, std::string *error_out) {
  const auto checked_end = [](uint64_t begin, uint64_t size) -> std::optional<uint64_t> {
    if (size > std::numeric_limits<uint64_t>::max() - begin)
      return std::nullopt;
    return begin + size;
  };
  if (arch_ == ROCJITSU_CODE_ARCH_INVALID) {
    report(error_out, "DBI patch placement: architecture was not set");
    return std::nullopt;
  }
  if (request.original_size < sizeof(uint32_t) || request.original_size % sizeof(uint32_t) != 0 ||
      request.body_size == 0 || request.body_size % sizeof(uint32_t) != 0) {
    report(error_out, "DBI patch placement: sizes must be nonzero instruction multiples");
    return std::nullopt;
  }
  const auto anchor_end = checked_end(request.anchor_offset, request.original_size);
  if (!anchor_end || *anchor_end > original_text_size_) {
    report(error_out, "DBI patch placement: anchor exceeds original .text");
    return std::nullopt;
  }

  if (request.body_size <= request.inline_capacity) {
    const auto body_end = checked_end(request.anchor_offset, request.body_size);
    if (body_end && *body_end <= original_text_size_ &&
        range_is_free(request.anchor_offset, *body_end)) {
      reserve_range(request.anchor_offset, *body_end);
      return DbiPatchPlacement{
          .kind = DbiPatchPlacementKind::Inline,
          .anchor_offset = request.anchor_offset,
          .original_size = request.original_size,
          .body_offset = request.anchor_offset,
          .body_size = request.body_size,
          .return_branch_offset = 0,
          .return_target = *anchor_end,
      };
    }
  }

  const auto try_trampoline = [&](DbiPatchPlacementKind kind, uint64_t body_offset,
                                  uint64_t capacity) -> std::optional<DbiPatchPlacement> {
    const auto body_end = checked_end(body_offset, request.body_size);
    const auto reservation_end = body_end ? checked_end(*body_end, sizeof(uint32_t)) : std::nullopt;
    if (!body_end || !reservation_end || request.body_size + sizeof(uint32_t) > capacity ||
        !range_is_free(request.anchor_offset, *anchor_end) ||
        (kind == DbiPatchPlacementKind::LocalCave &&
         (!range_is_free(body_offset, *reservation_end) ||
          *reservation_end > original_text_size_)) ||
        !compute_sopp_branch_simm16(request.anchor_offset, body_offset) ||
        !compute_sopp_branch_simm16(*body_end, *anchor_end)) {
      return std::nullopt;
    }
    return DbiPatchPlacement{
        .kind = kind,
        .anchor_offset = request.anchor_offset,
        .original_size = request.original_size,
        .body_offset = body_offset,
        .body_size = request.body_size,
        .return_branch_offset = *body_end,
        .return_target = *anchor_end,
    };
  };

  if (request.local_cave) {
    if (auto placement = try_trampoline(DbiPatchPlacementKind::LocalCave,
                                        request.local_cave->offset, request.local_cave->capacity)) {
      reserve_range(request.anchor_offset, *anchor_end);
      reserve_range(placement->body_offset, placement->return_branch_offset + sizeof(uint32_t));
      return placement;
    }
  }

  if (request.allow_appended_cave) {
    if (auto placement = try_trampoline(DbiPatchPlacementKind::AppendedCave, appended_cursor_,
                                        std::numeric_limits<uint64_t>::max())) {
      reserve_range(request.anchor_offset, *anchor_end);
      appended_cursor_ = placement->return_branch_offset + sizeof(uint32_t);
      return placement;
    }
  }

  report(error_out,
         "DBI patch placement: no nonoverlapping reachable inline, local, or appended placement");
  return std::nullopt;
}

} // namespace rocjitsu
