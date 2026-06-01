// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instrumentor.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <array>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

void report(std::string *out, const char *msg) {
  if (out) *out = msg;
}

// PC-relative instructions that may not surface in flags() or
// branch_offset_bytes() on every ISA. Exact-match list plus a prefix match
// for the s_rfe_* family (variants differ across ISAs).
constexpr std::array<std::string_view, 4> kPcRelativeDenylist = {
    "s_getpc_b64",
    "s_call_b64",
    "s_setpc_b64",
    "s_swappc_b64",
};

constexpr std::string_view kRfePrefix = "s_rfe_";

[[nodiscard]] bool is_denylisted_mnemonic(std::string_view mnemonic) {
  for (auto m : kPcRelativeDenylist)
    if (mnemonic == m) return true;
  if (mnemonic.size() >= kRfePrefix.size() &&
      mnemonic.substr(0, kRfePrefix.size()) == kRfePrefix)
    return true;
  return false;
}

// Per-site result of patch_relocation_only's preflight: the chosen trampoline
// offset and the concrete bytes we'll splice in once all preflights succeed.
struct AppliedSite {
  const ResolvedInstrumentationSite *site;
  uint64_t trampoline_offset;
  TrampolineBytes bytes;
};

} // namespace

std::optional<ResolvedInstrumentationSite>
validate_relocation_anchor(const Instruction &anchor, uint64_t anchor_offset,
                           std::span<const uint8_t> text_bytes,
                           const InstrumentationPoint &pt,
                           [[maybe_unused]] rj_code_arch_t arch, std::string *error_out) {
  if (pt.filter_flags != 0) {
    report(error_out, "InstrumentationPoint::filter_flags must be 0 in the inline-nop smoke build");
    return std::nullopt;
  }
  // TODO(future milestone): support AfterInst / BlockEntry / BlockExit.
  if (pt.kind != InstrumentationKind::BeforeInst) {
    report(error_out,
           "InstrumentationPoint::kind must be BeforeInst in the inline-nop smoke build");
    return std::nullopt;
  }
  // TODO(future milestone): consume probe_obj / probe_symbol when probe-call
  // trampolines are supported.
  if (pt.probe_obj != nullptr) {
    report(error_out, "InstrumentationPoint::probe_obj must be null in the inline-nop smoke build");
    return std::nullopt;
  }
  if (!pt.probe_symbol.empty()) {
    report(error_out,
           "InstrumentationPoint::probe_symbol must be empty in the inline-nop smoke build");
    return std::nullopt;
  }
  // TODO(future milestone): consume force_full_exec when EXEC policy
  // management lands.
  if (pt.force_full_exec) {
    report(error_out,
           "InstrumentationPoint::force_full_exec must be false in the inline-nop smoke build");
    return std::nullopt;
  }
  if (anchor_offset % sizeof(uint32_t) != 0) {
    report(error_out, "anchor_offset must be dword aligned");
    return std::nullopt;
  }
  // Instruction::size() returns int by convention; the `!= 4 && != 8` check
  // also rejects negative values (which decoders never produce in practice).
  const int size = anchor.size();
  if (size != 4 && size != 8) {
    report(error_out, "anchor instruction size must be 4 or 8 bytes");
    return std::nullopt;
  }
  if (anchor_offset + static_cast<uint64_t>(size) > text_bytes.size()) {
    report(error_out, "anchor extends past end of .text");
    return std::nullopt;
  }
  if (anchor.raw_encoding() == nullptr) {
    report(error_out, "anchor instruction has no raw encoding bytes");
    return std::nullopt;
  }
  constexpr uint64_t kControlFlowFlags =
      BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR;
  if (anchor.flags() & kControlFlowFlags) {
    report(error_out, "anchor instruction is a branch / indirect / program terminator");
    return std::nullopt;
  }
  if (anchor.branch_offset_bytes().has_value()) {
    report(error_out, "anchor instruction has a PC-relative branch offset");
    return std::nullopt;
  }
  if (is_denylisted_mnemonic(anchor.mnemonic())) {
    report(error_out, "anchor mnemonic is in the PC-relative denylist");
    return std::nullopt;
  }

  ResolvedInstrumentationSite site;
  site.anchor_inst = &anchor;
  site.kind = pt.kind;
  site.anchor_offset = anchor_offset;
  site.original_size = static_cast<uint32_t>(size);
  site.original_bytes.assign(text_bytes.begin() + anchor_offset,
                             text_bytes.begin() + anchor_offset + size);
  site.mnemonic = std::string(anchor.mnemonic());
  return site;
}

bool validate_inline_nop_smoke_plan(const TrampolinePlan &plan, std::string *error_out) {
  if (!plan.emit_original) {
    report(error_out,
           "trampoline plan: emit_original must be true for the inline-nop smoke build");
    return false;
  }
  if (!plan.after_items.empty()) {
    report(error_out, "trampoline plan: after_items must be empty for the inline-nop smoke build");
    return false;
  }
  if (plan.before_items.size() != 1 || plan.before_items[0].words.size() != 1 ||
      plan.before_items[0].words[0] != build_s_nop(0, plan.arch)) {
    report(error_out,
           "trampoline plan: before_items must be exactly { { s_nop 0 } } "
           "for the inline-nop smoke build");
    return false;
  }
  return true;
}

TrampolinePlan make_relocation_only_plan(const ResolvedInstrumentationSite &site,
                                         rj_code_arch_t arch, uint64_t trampoline_offset) {
  TrampolinePlan plan;
  plan.arch = arch;
  plan.anchor_offset = site.anchor_offset;
  plan.original_size = site.original_size;
  plan.trampoline_offset = trampoline_offset;
  plan.return_target = site.anchor_offset + site.original_size;

  // Little-endian host assumption (consistent with DBT and the rest of the
  // codebase): host byte order matches AMDGPU's little-endian encoding, so
  // memcpy of the raw bytes into uint32_t words preserves semantics.
  const size_t num_words = site.original_size / sizeof(uint32_t);
  plan.original_words.resize(num_words);
  std::memcpy(plan.original_words.data(), site.original_bytes.data(), site.original_size);

  plan.before_items = {InlineAsmItem{{build_s_nop(0, arch)}}};
  plan.after_items = {};
  plan.emit_original = true;
  return plan;
}

Instrumentor::Instrumentor(const AmdGpuCodeObject &obj, rj_code_arch_t arch)
    : obj_(obj), arch_(arch) {}

Instrumentor::~Instrumentor() = default;

void Instrumentor::add_point(InstrumentationPoint pt) { points_.push_back(std::move(pt)); }

void Instrumentor::add_point_by_offset(uint64_t anchor_offset, InstrumentationKind kind) {
  InstrumentationPoint pt;
  pt.anchor_offset = anchor_offset;
  pt.kind = kind;
  points_.push_back(std::move(pt));
}

std::span<const std::unique_ptr<BasicBlock>> Instrumentor::owned_blocks() {
  ensure_blocks_built();
  return blocks_;
}

void Instrumentor::ensure_blocks_built() {
  if (blocks_built_) return;
  decoder_ = Decoder::create(arch_);
  blocks_ = BasicBlock::build(obj_, *decoder_);
  blocks_built_ = true;
}

const Instruction *Instrumentor::find_instruction_at_offset(uint64_t anchor_offset) {
  for (const auto &block : blocks_) {
    if (anchor_offset < block->start_offset() || anchor_offset >= block->end_offset())
      continue;
    uint64_t cur = block->start_offset();
    for (const Instruction &inst : block->instructions()) {
      if (cur == anchor_offset) return &inst;
      cur += static_cast<uint64_t>(inst.size());
    }
    return nullptr;
  }
  return nullptr;
}

std::optional<uint64_t>
Instrumentor::resolve_anchor_inst_to_offset(const Instruction *target, std::string *err) {
  for (const auto &block : blocks_) {
    uint64_t cur = block->start_offset();
    for (const Instruction &inst : block->instructions()) {
      if (&inst == target) return cur;
      cur += static_cast<uint64_t>(inst.size());
    }
  }
  report(err, "anchor_inst does not belong to any Instrumentor-owned block");
  return std::nullopt;
}

Instrumentor::ResolveResult Instrumentor::resolve_and_validate() {
  ResolveResult result;
  ensure_blocks_built();

  if (obj_.text_sections().empty()) {
    result.errors.emplace_back("code object has no .text section");
    return result;
  }
  // TODO(future milestone): support multi-text code objects. Anchor offsets
  // would need to identify which .text section they belong to.
  if (obj_.text_sections().size() > 1) {
    result.errors.emplace_back(
        "code object has multiple .text sections; the inline-nop smoke build supports only one");
    return result;
  }
  const Section *text = obj_.text_sections().front();
  const std::span<const uint8_t> text_bytes(
      reinterpret_cast<const uint8_t *>(text->data()), text->size());

  // All-or-nothing: per-point errors accumulate in `result.errors`; per-point
  // successes accumulate in `sites` but are only published to `result.sites`
  // if `errors` ends up empty. Multi-site callers therefore can't tell
  // *which* points succeeded when any fail — only the failures are itemized.
  // Acceptable for PC01-A (single-site only); revisit if a future milestone
  // exposes multi-site to callers.
  std::vector<ResolvedInstrumentationSite> sites;
  sites.reserve(points_.size());
  for (const auto &pt : points_) {
    const bool has_inst = pt.anchor_inst != nullptr;
    const bool has_off = pt.anchor_offset.has_value();
    if (has_inst == has_off) {
      result.errors.emplace_back(
          has_inst ? "InstrumentationPoint sets both anchor_inst and anchor_offset"
                   : "InstrumentationPoint sets neither anchor_inst nor anchor_offset");
      continue;
    }

    uint64_t offset = 0;
    const Instruction *anchor = nullptr;
    if (has_inst) {
      std::string err;
      auto off = resolve_anchor_inst_to_offset(pt.anchor_inst, &err);
      if (!off) {
        result.errors.push_back(std::move(err));
        continue;
      }
      offset = *off;
      anchor = pt.anchor_inst;
    } else {
      offset = *pt.anchor_offset;
      anchor = find_instruction_at_offset(offset);
      if (anchor == nullptr) {
        result.errors.emplace_back("no decoded instruction starts at the requested anchor_offset");
        continue;
      }
    }

    std::string err;
    auto site = validate_relocation_anchor(*anchor, offset, text_bytes, pt, arch_, &err);
    if (!site) {
      result.errors.push_back(std::move(err));
      continue;
    }
    sites.push_back(std::move(*site));
  }

  if (result.errors.empty()) result.sites = std::move(sites);
  return result;
}

InstrumentedCodeObject Instrumentor::patch_relocation_only() {
  InstrumentedCodeObject result;

  if (patched_) {
    result.errors.emplace_back(
        "patch_relocation_only has already been called on this Instrumentor");
    return result;
  }
  patched_ = true;

  // Inline-nop smoke restriction: exactly one queued point.
  if (points_.empty()) {
    result.errors.emplace_back("patch_relocation_only requires exactly one queued point; got zero");
    return result;
  }
  if (points_.size() > 1) {
    result.errors.emplace_back(
        "patch_relocation_only accepts only one queued point in the inline-nop smoke build");
    return result;
  }

  auto resolved = resolve_and_validate();
  if (!resolved.errors.empty()) {
    result.errors = std::move(resolved.errors);
    return result;
  }
  const auto &sites = resolved.sites;

  // Construct the patcher and preflight builder output before mutating it.
  // TODO(future cleanup): the "cave" terminology is inherited from the
  // patcher API (originally added for DBT). For DBI the appended section is
  // not really "unused space" — rename the patcher API to something like
  // `appended_section_*` so DBI reads naturally too.
  CodeObjectPatcher patcher(obj_);
  patcher.set_cave_start(patcher.text_size());

  std::vector<AppliedSite> applied;
  applied.reserve(sites.size());
  uint64_t cave_cursor = patcher.text_size();
  for (const auto &site : sites) {
    const uint64_t trampoline_offset = cave_cursor;
    TrampolinePlan plan = make_relocation_only_plan(site, arch_, trampoline_offset);
    // Defense-in-depth: make_relocation_only_plan always produces a
    // canonical inline-nop smoke plan today, so this never fires in
    // practice. If it does, that's a bug in plan construction.
    std::string err;
    if (!validate_inline_nop_smoke_plan(plan, &err)) {
      result.errors.push_back(std::move(err));
      return result;
    }
    auto bytes = TrampolineBuilder::build(plan, &err);
    if (!bytes) {
      result.errors.push_back(std::move(err));
      return result;
    }
    cave_cursor += bytes->trampoline_words.size() * sizeof(uint32_t);
    applied.push_back({&site, trampoline_offset, std::move(*bytes)});
  }

  // Preflight invariant: every per-site validation, branch-range check, and
  // trampoline-byte construction has succeeded before this point. The patcher
  // is now mutated. `append_cave_section` below is the one operation that
  // *can* still fail after mutation; in practice it only fails when the
  // `.text` section header is missing, which was caught earlier by the
  // `text_sections().empty()` check.
  // Splice patched anchors into a local copy of .text, then overwrite once.
  const auto text_span = patcher.text_bytes();
  std::vector<uint8_t> local_text(text_span.begin(), text_span.end());
  for (const auto &a : applied) {
    std::memcpy(local_text.data() + a.site->anchor_offset,
                a.bytes.patched_anchor_bytes.data(), a.site->original_size);
  }
  patcher.overwrite_text(local_text);

  // Append trampoline words and materialize the section.
  for (const auto &a : applied)
    patcher.append_cave_body(a.bytes.trampoline_words);
  if (!patcher.append_cave_section(".rj_trampolines")) {
    result.errors.emplace_back("failed to append .rj_trampolines section");
    return result;
  }

  // Emit and build patch summaries.
  result.elf_bytes = patcher.emit();
  for (const auto &a : applied) {
    InstrumentationPatch patch;
    patch.anchor_offset = a.site->anchor_offset;
    patch.original_size = a.site->original_size;
    patch.trampoline_offset = a.trampoline_offset;
    patch.return_target = a.site->anchor_offset + a.site->original_size;
    patch.original_bytes = a.site->original_bytes;
    patch.patched_anchor_bytes = a.bytes.patched_anchor_bytes;
    result.patches.push_back(std::move(patch));
  }
  return result;
}

} // namespace rocjitsu
