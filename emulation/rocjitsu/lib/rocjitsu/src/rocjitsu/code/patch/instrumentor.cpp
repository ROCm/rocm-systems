// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instrumentor.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/error_report.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/kernel_text_layout.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_clobber.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <array>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

// PC-reading / PC-relative instructions to reject as anchors: relocating any of
// them into a trampoline changes the PC value they read (s_getpc) or the target
// they branch to. s_getpc and the s_rfe_ family carry no control-flow flag or
// branch_offset_bytes, so validate_anchor's flag/offset checks miss them and the
// denylist is the only thing that catches them. The s_call / s_setpc / s_swappc
// entries are already rejected upstream by their INDIRECT_BRANCH / INDIRECT_CALL
// flags and are listed only as defense-in-depth. gfx1250 renames the whole
// family from *_b64 to *_i64 (e.g. s_getpc_b64 -> s_get_pc_i64), so both
// spellings are listed; the s_rfe_ prefix match below covers s_rfe_b64,
// s_rfe_i64, and s_rfe_restore_b64.
constexpr std::array<std::string_view, 8> kPcRelativeDenylist = {
    "s_getpc_b64",  "s_call_b64", "s_setpc_b64",  "s_swappc_b64",
    "s_get_pc_i64", "s_call_i64", "s_set_pc_i64", "s_swap_pc_i64",
};

constexpr std::string_view kRfePrefix = "s_rfe_";

[[nodiscard]] bool is_denylisted_mnemonic(std::string_view mnemonic) {
  for (auto m : kPcRelativeDenylist)
    if (mnemonic == m)
      return true;
  if (mnemonic.size() >= kRfePrefix.size() && mnemonic.substr(0, kRfePrefix.size()) == kRfePrefix)
    return true;
  return false;
}

// Per-site result of Instrumentor::patch's preflight: the chosen trampoline
// offset and the concrete bytes we'll splice in once all preflights succeed.
struct AppliedSite {
  const ResolvedInstrumentationSite *site;
  uint64_t trampoline_offset;
  TrampolineBytes bytes;

  // Probe-call facts for the patch record; defaulted for the inline-nop site.
  bool is_probe_call = false;
  uint64_t probe_target_offset = 0;
  uint16_t link_pair_base = 0;
  uint16_t target_pair_base = 0;
};

// Human-readable single-lane register name for spill diagnostics.
std::string reg_name(RegisterRef ref) {
  const char *prefix = "?";
  switch (ref.cls) {
  case RegClass::SGPR:
    prefix = "s";
    break;
  case RegClass::VGPR:
    prefix = "v";
    break;
  case RegClass::ACC_VGPR:
    prefix = "acc";
    break;
  default:
    break;
  }
  return std::string(prefix) + std::to_string(ref.index);
}

// Special machine state has no save/restore path across a probe call yet.
// Only SCC (via the trampoline envelope) and ordinary GPRs (via the spill
// policy) are handled today. Until each special register gains a real consumer,
// fail closed on any probe whose body touches them rather than letting it
// silently corrupt the host kernel's state.
bool check_probe_special_state(const ProbeClobberSummary &summary, std::string *error_out) {
  std::vector<const char *> touched;
  if (summary.touches_exec)
    touched.push_back("EXEC");
  if (summary.touches_vcc)
    touched.push_back("VCC");
  if (summary.touches_m0)
    touched.push_back("M0");
  if (summary.touches_flat_scratch)
    touched.push_back("FLAT_SCRATCH");
  if (touched.empty())
    return true;
  if (error_out != nullptr) {
    std::string msg = "probe body writes special machine state not yet preserved across a "
                      "probe call:";
    bool first = true;
    for (const char *name : touched) {
      msg += first ? " " : ", ";
      msg += name;
      first = false;
    }
    *error_out = std::move(msg);
  }
  return false;
}

// Check that the probe does not clobber the link pair.
bool check_probe_link_pair(const ProbeClobberSummary &summary, ProbeCallingConvention cc,
                           std::string *error_out) {
  const std::optional<uint16_t> link_base = link_pair_for(cc);
  if (!link_base)
    return true; // Unknown convention: plan_probe_call rejects it with a cc-specific error.
  RegisterSet link_pair;
  link_pair.expand(RegisterRef{RegClass::SGPR, *link_base, 2});
  if (!summary.ordinary_clobbers.intersects(link_pair))
    return true;
  if (error_out != nullptr) {
    const uint16_t hi = static_cast<uint16_t>(*link_base + 1);
    *error_out = "probe body overwrites its own return-link pair s[" + std::to_string(*link_base) +
                 ":" + std::to_string(hi) +
                 "] before returning; it would return through a corrupted PC";
  }
  return false;
}

} // namespace

bool is_relocatable_anchor(const Instruction &anchor, uint64_t anchor_offset,
                           std::span<const uint8_t> text_bytes,
                           [[maybe_unused]] rj_code_arch_t arch, std::string *error_out) {
  if (anchor_offset % sizeof(uint32_t) != 0) {
    report(error_out, "anchor_offset must be dword aligned");
    return false;
  }
  // Instruction::size() returns int by convention; the `!= 4 && != 8` check
  // also rejects negative values (which decoders never produce in practice).
  const int size = anchor.size();
  if (size != 4 && size != 8) {
    report(error_out, "anchor instruction size must be 4 or 8 bytes");
    return false;
  }
  // Subtraction-based bounds check: a huge anchor_offset would otherwise wrap
  // the addition and silently pass.
  const uint64_t size_u = static_cast<uint64_t>(size);
  if (anchor_offset > text_bytes.size() || size_u > text_bytes.size() - anchor_offset) {
    report(error_out, "anchor extends past end of .text");
    return false;
  }
  if (anchor.raw_encoding() == nullptr) {
    report(error_out, "anchor instruction has no raw encoding bytes");
    return false;
  }
  constexpr uint64_t kControlFlowFlags =
      BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR;
  if (anchor.flags() & kControlFlowFlags) {
    report(error_out, "anchor instruction is a branch / indirect / program terminator");
    return false;
  }
  if (anchor.branch_offset_bytes().has_value()) {
    report(error_out, "anchor instruction has a PC-relative branch offset");
    return false;
  }
  if (is_denylisted_mnemonic(anchor.mnemonic())) {
    report(error_out, "anchor mnemonic is in the PC-relative denylist");
    return false;
  }
  return true;
}

std::optional<ResolvedInstrumentationSite>
validate_anchor(const Instruction &anchor, uint64_t anchor_offset,
                std::span<const uint8_t> text_bytes, const InstrumentationPoint &pt,
                rj_code_arch_t arch, std::string *error_out) {
  auto fail = [&](const char *msg) {
    report(error_out, (msg + (", anchor_offset = " + std::to_string(anchor_offset))).c_str());
  };
  // TODO: consume filter_flags to filter anchors based on InstFlags.
  if (pt.filter_flags != 0) {
    fail("InstrumentationPoint::filter_flags must be 0 temporarily");
    return std::nullopt;
  }
  // TODO: support AfterInst / BlockEntry / BlockExit.
  if (pt.kind != InstrumentationKind::BeforeInst) {
    fail("InstrumentationPoint::kind must be BeforeInst temporarily");
    return std::nullopt;
  }
  // A probe call needs both halves of the request: the object to resolve the
  // symbol in, and the symbol name
  if ((pt.probe_obj != nullptr) != (!pt.probe_symbol.empty())) {
    report(error_out, "InstrumentationPoint probe_obj and probe_symbol must both be set "
                      "(a probe call) or both be empty (the inline nop)");
    return std::nullopt;
  }
  // TODO: consume force_full_exec when EXEC policy management is implemented
  if (pt.force_full_exec) {
    fail("InstrumentationPoint::force_full_exec must be false temporarily");
    return std::nullopt;
  }

  if (!is_relocatable_anchor(anchor, anchor_offset, text_bytes, arch, error_out))
    return std::nullopt;

  const auto size = static_cast<uint32_t>(anchor.size());
  ResolvedInstrumentationSite site;
  site.kind = pt.kind;
  site.anchor_offset = anchor_offset;
  site.original_size = size;
  site.original_bytes.assign(text_bytes.begin() + anchor_offset,
                             text_bytes.begin() + anchor_offset + size);
  site.mnemonic = std::string(anchor.mnemonic());
  return site;
}

bool validate_inline_nop_plan(const TrampolinePlan &plan, std::string *error_out) {
  if (!plan.emit_original) {
    report(error_out, "trampoline plan: emit_original must be true for the inlined nop");
    return false;
  }
  if (!plan.after_items.empty()) {
    report(error_out, "trampoline plan: after_items must be empty for the inline nop");
    return false;
  }
  if (plan.before_items.size() != 1 || plan.before_items[0].words.size() != 1 ||
      plan.before_items[0].words[0] != build_s_nop(0, plan.arch)) {
    report(error_out, "trampoline plan: before_items must be exactly { { s_nop 0 } } "
                      "for the inlined nop");
    return false;
  }
  return true;
}

// Only ordinary GPR clobbers feed the spill set. Special machine state in the
// summary (EXEC/VCC/M0/FLAT_SCRATCH) is rejected up front by
// check_probe_special_state
RegisterSet compute_instrumentation_clobbers(const ProbeClobberSummary &probe_summary,
                                             const RegisterSet &builder_clobbers) {
  return probe_summary.ordinary_clobbers | builder_clobbers;
}

RegisterSet compute_spill_set(const RegisterSet &live_at_anchor,
                              const RegisterSet &instrumentation_clobbers) {
  return live_at_anchor & instrumentation_clobbers;
}

bool check_spill_policy(const RegisterSet &spill_set, SpillPolicy policy, std::string *error_out) {
  if (policy == SpillPolicy::NoSpillsSupported && !spill_set.none()) {
    std::string msg = "probe-call requires spilling live registers, not yet supported:";
    bool first = true;
    spill_set.for_each([&](RegisterRef ref) {
      msg += first ? " " : ", ";
      msg += reg_name(ref);
      first = false;
    });
    report(error_out, msg.c_str());
    return false;
  }
  return true;
}

namespace {

// Fill the layout/identity fields shared by every trampoline plan
TrampolinePlan make_base_plan(const ResolvedInstrumentationSite &site, rj_code_arch_t arch,
                              uint64_t trampoline_offset) {
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
  return plan;
}

} // namespace

TrampolinePlan make_trampoline_plan(const ResolvedInstrumentationSite &site, rj_code_arch_t arch,
                                    uint64_t trampoline_offset) {
  TrampolinePlan plan = make_base_plan(site, arch, trampoline_offset);
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

bool Instrumentor::ensure_blocks_built(std::string *error_out) {
  if (blocks_built_)
    return true;
  auto decoder = Decoder::create(arch_);
  if (!decoder) {
    report(error_out, "no decoder available for the requested architecture");
    return false;
  }
  decoder_ = std::move(decoder);
  blocks_ = BasicBlock::build(obj_, *decoder_, arch_);
  for (const auto &block : blocks_) {
    uint64_t cur = block->start_offset();
    for (const Instruction &inst : block->instructions()) {
      offset_to_inst_.emplace(cur, &inst);
      cur += static_cast<uint64_t>(inst.size());
    }
  }
  blocks_built_ = true;
  return true;
}

const Instruction *Instrumentor::find_instruction_at_offset(uint64_t anchor_offset) const {
  auto it = offset_to_inst_.find(anchor_offset);
  return it == offset_to_inst_.end() ? nullptr : it->second;
}

Instrumentor::ResolvedPoints Instrumentor::resolve_points() {
  ResolvedPoints out;

  std::string err;
  if (!ensure_blocks_built(&err)) {
    out.errors.push_back(std::move(err));
    return out;
  }
  if (obj_.text_sections().empty()) {
    out.errors.emplace_back("code object has no .text section");
    return out;
  }
  // TODO: support multi-text code objects. Anchor offsets would need to identify
  // which .text section they belong to.
  if (obj_.text_sections().size() > 1) {
    out.errors.emplace_back("code object has multiple .text sections; currently supports only one");
    return out;
  }
  const Section *text = obj_.text_sections().front();
  const std::span<const uint8_t> text_bytes(reinterpret_cast<const uint8_t *>(text->data()),
                                            text->size());

  // Store probe objects and symbols together in probe_keys (object, symbol).
  std::vector<std::pair<const AmdGpuCodeObject *, std::string>> probe_keys;
  // Helper function to get a probe index for a given InstrumentationPoint
  // If the probe is new, then resolve it and get probe info; add it to
  // probe_keys and out.probes.
  auto resolve_probe_index = [&](const InstrumentationPoint &pt,
                                 std::string &perr) -> std::optional<size_t> {
    for (size_t i = 0; i < probe_keys.size(); ++i) {
      if (probe_keys[i].first == pt.probe_obj && probe_keys[i].second == pt.probe_symbol)
        return i;
    }
    auto sym = resolve_probe_symbol(*pt.probe_obj, pt.probe_symbol, &perr);
    if (!sym)
      return std::nullopt;
    auto callable = build_probe_callable(*pt.probe_obj, *sym, arch_, &perr);
    if (!callable)
      return std::nullopt;
    out.probes.push_back(std::move(*callable));
    probe_keys.emplace_back(pt.probe_obj, pt.probe_symbol);
    return out.probes.size() - 1;
  };

  // All-or-nothing: per-point errors accumulate
  std::vector<ResolvedInstrumentationSite> sites;
  std::unordered_set<uint64_t> site_offsets;
  sites.reserve(points_.size());
  for (const auto &pt : points_) {
    const Instruction *anchor = find_instruction_at_offset(pt.anchor_offset);
    if (anchor == nullptr) {
      out.errors.emplace_back("no decoded instruction starts at the requested anchor_offset = " +
                              std::to_string(pt.anchor_offset));
      continue;
    }

    if (site_offsets.find(pt.anchor_offset) != site_offsets.end()) {
      out.errors.emplace_back("multiple points requested the same anchor_offset = " +
                              std::to_string(pt.anchor_offset));
      continue;
    }

    std::string perr;
    auto site = validate_anchor(*anchor, pt.anchor_offset, text_bytes, pt, arch_, &perr);
    if (!site) {
      out.errors.push_back(std::move(perr));
      continue;
    }

    // Resolve the probe request. An unresolvable or non-relocatable probe is
    // a fatal, all-or-nothing validation error.
    if (pt.probe_obj != nullptr) {
      auto index = resolve_probe_index(pt, perr);
      if (!index) {
        out.errors.push_back(std::move(perr));
        continue;
      }
      site->probe_index = *index;
    }

    sites.push_back(std::move(*site));
    site_offsets.insert(site->anchor_offset);
  }

  if (out.errors.empty())
    out.sites = std::move(sites);
  else
    out.probes.clear(); // all-or-nothing: no partial registry on failure.
  return out;
}

Instrumentor::ValidationResult Instrumentor::validate_points() {
  ResolvedPoints resolved = resolve_points();
  ValidationResult result;
  result.errors = std::move(resolved.errors);
  result.sites = std::move(resolved.sites);
  return result;
}

InstrumentedCodeObject Instrumentor::patch() {
  // Slice off the debug summaries; move the base subobject into the return.
  auto debug = patch_with_debug_summaries();
  return std::move(static_cast<InstrumentedCodeObject &>(debug));
}

InstrumentedCodeObjectDebug Instrumentor::patch_with_debug_summaries() {
  InstrumentedCodeObjectDebug result;

  if (patched_) {
    result.errors.emplace_back(
        "Instrumentor::patch / patch_with_debug_summaries has already been called");
    return result;
  }
  patched_ = true;

  if (points_.empty()) {
    result.errors.emplace_back("Instrumentor::patch requires at least one queued point; got zero");
    return result;
  }

  ResolvedPoints resolved = resolve_points();
  if (!resolved.errors.empty()) {
    result.errors = std::move(resolved.errors);
    return result;
  }

  // Construct the patcher and preflight builder output before mutating it.
  // Each trampoline is appended directly after the original .text bytes as a
  // local code cave (as in the current DBT design).
  CodeObjectPatcher patcher(obj_);
  const uint64_t cave_start = patcher.text_size();

  // Liveness over the decoded blocks, built once and reused by every probe-call
  // site. validate_points() already built blocks_, and obj_ owns the
  // Instructions live_before() is keyed on.
  // TODO: scope this to the kernel containing each anchor, like the DBT path,
  // rather than treating every decoded block as one CFG. Safe here: kernels end
  // in s_endpgm, so there are no false cross-kernel fallthrough edges.
  std::vector<BasicBlock *> liveness_scope;
  liveness_scope.reserve(blocks_.size());
  for (const auto &block : blocks_)
    liveness_scope.push_back(block.get());
  const LivenessAnalysis liveness{KernelBlockScope(liveness_scope)};

  // Lay out the appended region as [probe bodies][trampolines]. Each distinct
  // probe body is copied once, ahead of the trampolines that call into it, so a
  // trampoline's target address is known before it is emitted and sites sharing
  // a probe share its single body.
  const auto &sites = resolved.sites;
  // Allocate offsets for each probe body, starting at the local cave (the first
  // byte after the original .text). The site loop below continues advancing this
  // same cursor so trampolines follow the probe bodies.
  uint64_t cave_cursor = cave_start;
  for (ProbeCallable &probe : resolved.probes) {
    probe.output_text_offset = cave_cursor;
    cave_cursor += probe.body_words.size() * sizeof(uint32_t);
  }

  // Temporary SGPR-allocation bound. The probe-call return-link pair is fixed
  // by the calling convention, so the kernel must allocate up to it.
  const std::optional<uint32_t> kernel_sgpr_count = obj_.min_kernel_sgpr_count(arch_);

  std::vector<AppliedSite> applied;
  applied.reserve(sites.size());
  // Allocate offsets for each site. Trampoline size is highly dependent on the
  // plan (inlined assembly? spills? arguments passed?) so set up the plan first.
  // Then build the trampoline based on the plan.
  for (const auto &site : sites) {
    const uint64_t trampoline_offset = cave_cursor;
    std::string err;
    std::optional<TrampolineBytes> bytes;
    AppliedSite record{&site, trampoline_offset, {}};

    // If this site calls a probe, then we need to know what registers it will
    // clobber
    if (site.is_probe_call()) {
      const ProbeCallable &probe = resolved.probes[*site.probe_index];

      // The kernel must own the fixed return-link pair.
      if (const std::optional<uint16_t> link_base = link_pair_for(probe.cc);
          link_base && kernel_sgpr_count &&
          !probe_link_pair_fits_in_kernel(*kernel_sgpr_count, *link_base)) {
        result.errors.push_back(
            "probe call needs the return-link pair s[" + std::to_string(*link_base) + ":" +
            std::to_string(*link_base + 1) + "] but the kernel allocates only " +
            std::to_string(*kernel_sgpr_count) + " SGPRs; rebuild the kernel with at least " +
            std::to_string(*link_base + 2) + " SGPRs");
        continue;
      }

      // Callee clobbers (probe body) + liveness at the anchor feed envelope
      // resource selection and the no-spill policy gate.
      auto summary = build_probe_clobber_summary(probe, &err);
      if (!summary) {
        result.errors.push_back(std::move(err));
        continue;
      }

      // Check for special machine state that has no save/restore path yet
      if (!check_probe_special_state(*summary, &err)) {
        result.errors.push_back(std::move(err));
        continue;
      }

      // The probe must not overwrite its own return-link pair before returning.
      if (!check_probe_link_pair(*summary, probe.cc, &err)) {
        result.errors.push_back(std::move(err));
        continue;
      }

      // Get liveness for this anchor
      const Instruction *anchor = find_instruction_at_offset(site.anchor_offset);
      if (anchor == nullptr) {
        result.errors.emplace_back("internal: anchor instruction vanished after validation");
        continue;
      }
      const RegisterSet &live = liveness.live_before(*anchor);

      // Set the probe's offset
      TrampolinePlan plan = make_base_plan(site, arch_, trampoline_offset);
      plan.probe_target_offset = probe.output_text_offset;
      // Given liveness, clobbers, and calling convention, select registers
      // for trampoline and determine how big the trampoline will be
      if (!TrampolineBuilder::plan_probe_call(plan, probe.cc, live, summary->ordinary_clobbers,
                                              &err)) {
        result.errors.push_back(std::move(err));
        continue;
      }

      // Check that we do not need to spill registers since that is not
      // implemented yet.
      const RegisterSet clobbers =
          compute_instrumentation_clobbers(*summary, plan.builder_clobbers);
      const RegisterSet spill = compute_spill_set(live, clobbers);
      if (!check_spill_policy(spill, SpillPolicy::NoSpillsSupported, &err)) {
        result.errors.push_back(std::move(err));
        continue;
      }

      // Get trampoline from TrampolineBuilder
      bytes = TrampolineBuilder::emit_probe_call(plan, &err);
      record.is_probe_call = true;
      record.probe_target_offset = probe.output_text_offset;
      record.link_pair_base = plan.link_pair_base;
      record.target_pair_base = plan.target_pair_base;
      // If this site does not call a probe
      // Currently, this means that the plan is to put a nop which means we do
      // not need to mess with spills
    } else {
      TrampolinePlan plan = make_trampoline_plan(site, arch_, trampoline_offset);
      // The only assembly we currently allow is an inlined nop
      if (!validate_inline_nop_plan(plan, &err)) {
        result.errors.push_back(std::move(err));
        continue;
      }
      bytes = TrampolineBuilder::build(plan, &err);
    }

    if (!bytes) {
      result.errors.push_back(std::move(err));
      continue;
    }
    // Update the the current cave offset based on the size of the trampoline
    // and record the built trampoline
    cave_cursor += bytes->trampoline_words.size() * sizeof(uint32_t);
    record.bytes = std::move(*bytes);
    applied.push_back(std::move(record));
  }

  // All-or-nothing: bail before mutating the patcher if any site failed.
  if (!result.errors.empty())
    return result;

  // Every per-site validation, branch-range check, and trampoline-byte
  // construction has succeeded up to this point. Assemble the new .text in one
  // buffer: the original bytes with each anchor spliced to its forward branch,
  // followed by every trampoline appended as the local cave. replace_text()
  // grows .text in place and fixes up the surrounding ELF (section/segment
  // sizes, moved symbols, descriptor entries).
  const auto text_span = patcher.text_bytes();
  std::vector<uint8_t> new_text(text_span.begin(), text_span.end());
  for (const auto &a : applied) {
    std::memcpy(new_text.data() + a.site->anchor_offset, a.bytes.patched_anchor_bytes.data(),
                a.site->original_size);
  }
  // Append in the laid-out order: probe bodies first (one per distinct probe),
  // then trampolines.
  for (const ProbeCallable &probe : resolved.probes)
    append_words(new_text, probe.body_words);
  for (const auto &a : applied)
    append_words(new_text, a.bytes.trampoline_words);
  if (!patcher.replace_text(new_text)) {
    result.errors.emplace_back("failed to replace .text with the instrumented code");
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
    patch.is_probe_call = a.is_probe_call;
    if (a.is_probe_call) {
      patch.probe_symbol = resolved.probes[*a.site->probe_index].symbol;
      patch.probe_target_offset = a.probe_target_offset;
      patch.link_pair_base = a.link_pair_base;
      patch.target_pair_base = a.target_pair_base;
    }
    result.patches.push_back(std::move(patch));
  }
  return result;
}

} // namespace rocjitsu
