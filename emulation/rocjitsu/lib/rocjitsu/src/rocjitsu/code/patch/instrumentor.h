// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instrumentor.h
/// @brief DBI orchestrator: resolves InstrumentationPoints to .text-relative
///        anchors, validates them, and drives byte-level patching via
///        TrampolineBuilder and CodeObjectPatcher.
///
/// Mental model
/// ------------
/// The Instrumentor is the central hub. Callers queue InstrumentationPoints
/// (the requests). On patch(), the orchestrator runs a multi-stage pipeline:
///
///   InstrumentationPoint        -- request: "instrument here, with this body"
///         |  validate_points()
///         v
///   ResolvedInstrumentationSite -- one validated anchor + snapshot of the
///         |                        original bytes (captured before mutation)
///         |  make_trampoline_plan() + TrampolineBuilder::build()
///         v
///   (preflight: per-site plan + built bytes, accumulated locally)
///         |  splice anchors + append trampoline caves into .text + emit()
///         v
///   InstrumentedCodeObject      -- patched ELF + per-site InstrumentationPatch
///
/// Current pipeline is all-or-nothing across all queued points: any per-site
/// failure aborts the whole patch.
/// Future work: predicate-based anchor selection (Instrumentor walks blocks
/// itself), per-site failure tolerance, probe-call bodies,
/// AfterInst / BlockEntry / BlockExit kinds, EXEC policy management.
/// As that lands the per-stage types will thicken
/// (e.g. ResolvedInstrumentationSite probably gains an ordered list of bodies)
/// and a layout/negotiation stage will appear between planning and splicing.
///
/// Layer split: this file is the orchestrator. TrampolineBuilder
/// (trampoline_builder.h) is the generic byte emitter — it knows nothing
/// about points or milestones. CodeObjectPatcher (code_object_patcher.h)
/// owns ELF mutation.

#pragma once

#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_clobber.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
class BasicBlock;
class Decoder;
class Instruction;

/// @brief Where the instrumentation should attach relative to the anchor.
///
/// Internal enum mirroring the future public `rj_code_instrument_kind_t`.
/// The inline-nop smoke build only supports BeforeInst; the other values are
/// placeholders so the descriptor evolves into the Section 2 probe shape
/// without renaming.
///
/// TODO: implement AfterInst, BlockEntry, and BlockExit.
/// Today the validator rejects any kind other than BeforeInst.
enum class InstrumentationKind {
  BeforeInst,
  AfterInst,
  BlockEntry,
  BlockExit,
};

/// @brief A request to instrument one site.
///
/// The anchor is identified by @ref anchor_offset (a .text-relative byte
/// offset). The orchestrator looks up the decoded Instruction at that offset
/// during validation.
struct InstrumentationPoint {
  uint64_t anchor_offset = 0;

  InstrumentationKind kind = InstrumentationKind::BeforeInst;

  // filter_flags / force_full_exec are not used yet. The validator rejects any
  // non-default value to keep the contract honest until each field is actually
  // implemented.
  // TODO: consume force_full_exec when EXEC policy management lands; consume
  // filter_flags to filter based on InstFlags.
  uint32_t filter_flags = 0;
  // probe_obj / probe_symbol are consumed: set both to request a probe-call
  // trampoline, or leave both empty for the inline nop.
  const AmdGpuCodeObject *probe_obj = nullptr;
  std::string probe_symbol;
  bool force_full_exec = false;
  // TODO: SCC will eventually need a per-point knob mirroring force_full_exec
  // (e.g. probe_consumes_scc) since the probe call clobbers it
};

/// @brief Per-site record produced after validation and byte capture.
struct ResolvedInstrumentationSite {
  InstrumentationKind kind = InstrumentationKind::BeforeInst;
  uint64_t anchor_offset = 0;
  uint32_t original_size = 0;
  std::vector<uint8_t> original_bytes;
  std::string mnemonic; // Diagnostic/debug only.

  // Index into the probe registry produced by resolve_points()
  // (ResolvedPoints::probes); nullopt if no probes
  std::optional<size_t> probe_index;

  [[nodiscard]] bool is_probe_call() const { return probe_index.has_value(); }
};

/// @brief Per-site patch record, for testing and debugging.
///
/// Returned from Instrumentor::patch_with_debug_summaries(), not from the
/// regular patch() entry point. The schema is not yet stable; the same
/// information is also recoverable from a fresh disassembly of the emitted
/// ELF.
struct InstrumentationPatch {
  uint64_t anchor_offset;
  uint32_t original_size;
  uint64_t trampoline_offset; // .text-relative.
  uint64_t return_target;
  std::vector<uint8_t> original_bytes;
  std::vector<uint8_t> patched_anchor_bytes;

  // Probe-call details, populated for probe-call sites
  bool is_probe_call = false;
  std::string probe_symbol;
  uint64_t probe_target_offset = 0; // .text-relative offset of the copied body.
  uint16_t link_pair_base = 0;      // cc-derived return-link pair.
  uint16_t target_pair_base = 0;    // dead pair holding the materialized address.
};

/// @brief Result of Instrumentor::patch().
///
/// All-or-nothing: when any fatal error occurs, `elf_bytes` is empty and
/// `errors` is non-empty. On success, `errors` is empty and `elf_bytes`
/// contains a re-parseable patched ELF.
struct InstrumentedCodeObject {
  std::vector<uint8_t> elf_bytes;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
};

/// @brief Result of Instrumentor::patch_with_debug_summaries().
///
/// Extends InstrumentedCodeObject with one InstrumentationPatch per applied
/// site. Test / debug surface only; the per-site schema is unstable.
struct InstrumentedCodeObjectDebug : InstrumentedCodeObject {
  std::vector<InstrumentationPatch> patches;
};

/// @brief Permanent structural checks: would @p anchor work as a trampoline
///        anchor at @p anchor_offset?
///
/// Pure predicate. Does not look at any InstrumentationPoint. Answers only
/// "can the trampoline machinery splice an `s_branch` here and relocate the
/// original safely?"
///
/// Rules enforced (last two will eventually be relaxed):
///   - @p anchor_offset is dword aligned.
///   - anchor.size() is 4 or 8 and fits inside @p text_bytes.
///   - anchor.raw_encoding() is non-null.
///   - anchor is not a branch/cond branch/indirect branch/indirect call/
///     program terminator, and branch_offset_bytes() is nullopt.
///   - anchor.mnemonic() is not in the small PC-relative denylist
///     (s_getpc_b64, s_call_b64, s_setpc_b64, s_swappc_b64, s_rfe_*).
///
/// @p arch is accepted now so a future denylist can grow ISA-specific entries
/// without an API change; today's checks are uniform across all AMDGPU ISAs.
[[nodiscard]] bool is_relocatable_anchor(const Instruction &anchor, uint64_t anchor_offset,
                                         std::span<const uint8_t> text_bytes, rj_code_arch_t arch,
                                         std::string *error_out = nullptr);

/// @brief Validate that @p anchor is a legal trampoline anchor for @p pt.
///
/// Called by Instrumentor::validate_points() and also directly by tests
/// (the free-function form lets test fixtures use synthetic TestInstruction
/// objects without standing up an AmdGpuCodeObject). The anchor identity is
/// already resolved by the caller; @p pt is read for `filter_flags`, `kind`,
/// and reserved fields. See is_relocatable_anchor for rules.
///
/// Temporary rules enforced here:
///   - @p pt.filter_flags is zero.
///   - @p pt.kind is BeforeInst (other kinds are unsupported in this milestone).
///   - @p pt.probe_obj and @p pt.probe_symbol are consistent: both set (a probe
///     call) or both empty (the inline nop). Setting only one is rejected.
///   - @p pt.force_full_exec is false.
[[nodiscard]] std::optional<ResolvedInstrumentationSite>
validate_anchor(const Instruction &anchor, uint64_t anchor_offset,
                std::span<const uint8_t> text_bytes, const InstrumentationPoint &pt,
                rj_code_arch_t arch, std::string *error_out = nullptr);

/// @brief Build a TrampolinePlan from a validated site.
///
/// Temporarily fills it ONLY with a plan to instrument with a nop.
/// Fills before_items = {{ s_nop 0 }}, after_items = {}, emit_original = true.
/// The caller chooses @p trampoline_offset (typically `patcher.text_size()`).
[[nodiscard]] TrampolinePlan make_trampoline_plan(const ResolvedInstrumentationSite &site,
                                                  rj_code_arch_t arch, uint64_t trampoline_offset);

/// @brief Verify @p plan matches the (temporary) inlined nop body shape:
///        exactly one `before_items` entry containing `s_nop 0`, empty
///        `after_items`, and `emit_original == true`.
///
/// One of several places where temporary inlined nop constraints are enforced.
/// The others: validate_anchor() rejects reserved InstrumentationPoint fields,
/// and Instrumentor::patch() rejects multi-text code objects and the multi-
/// point case. This one lives at the orchestrator boundary rather than inside
/// TrampolineBuilder so the builder stays generic.
/// Called by Instrumentor::patch() as a defense-check (make sure users/agents
/// do not misinterpret current DBI support) and also directly by tests
///
/// TODO: delete this when DBI supports more plans
[[nodiscard]] bool validate_inline_nop_plan(const TrampolinePlan &plan,
                                            std::string *error_out = nullptr);

//==============================================================================
// Spill formula and policy (orchestrator-owned).
//
// The probe-call spill set is the live registers an instrumentation envelope
// would clobber:
//
//   instrument_clobbers = probe_clobbers | builder_clobbers
//   spill_set           = live_at_anchor & instrument_clobbers
//
// The two inputs come from different owners: `probe_clobbers` is the callee fact
// from ProbeClobberSummary (probe_clobber.h); `builder_clobbers` is the
// call-envelope fact the builder's resource plan reports (link pair,
// target-address pair, SCC temp, ...). That builder plan and its dead-register
// selection land in a later slice; these helpers take `builder_clobbers` as a
// plain RegisterSet so the formula and v0 policy are testable now. Combining the
// two and applying policy is the Instrumentor's job, so it lives here rather
// than in the callee-only probe_clobber unit.
//==============================================================================

/// @brief What the builder will do about registers that are simultaneously live
///        at the anchor and clobbered by instrumentation.
///
/// Currently implements only the no-spill path: a non-empty spill set is a hard
/// failure rather than something the builder saves and restores. The enum exists
/// now so the contract has a name and a later spill-capable slice can add a
/// value without a rename.
enum class SpillPolicy {
  NoSpillsSupported, ///< Any non-empty spill set is a hard failure.
  // TODO: SpillsSupported once a bootstrap save/restore sequence exists.
};

/// @brief instrument_clobbers = probe body clobbers | builder envelope clobbers.
[[nodiscard]] RegisterSet compute_instrumentation_clobbers(const ProbeClobberSummary &probe_summary,
                                                           const RegisterSet &builder_clobbers);

/// @brief spill_set = live_at_anchor & instrumentation_clobbers.
[[nodiscard]] RegisterSet compute_spill_set(const RegisterSet &live_at_anchor,
                                            const RegisterSet &instrumentation_clobbers);

/// @brief Enforce the current spill policy, NoSpillsSupported.
///
/// Under @ref SpillPolicy::NoSpillsSupported a non-empty @p spill_set fails
/// closed: returns false and writes a diagnostic naming the live, clobbered
/// registers. An empty spill set always succeeds.
[[nodiscard]] bool check_spill_policy(const RegisterSet &spill_set, SpillPolicy policy,
                                      std::string *error_out = nullptr);

/// @brief Does a kernel that allocates @p kernel_sgpr_count SGPRs own the fixed
///        return-link pair s[link_base : link_base+1]?
///
/// The probe-call trampoline returns through a link pair fixed by the calling
/// convention (s[30:31] today). A kernel only owns SGPRs [0, sgpr_count), so it
/// is possible that it does not own the registers even if they look dead from
/// liveness. This coarse gate fails such a kernel closed early.
///
/// NOTE: this exists only because the link pair is hardcoded and the kernel
/// must be built padded up to it. Remove once the link pair is chosen from dead
/// registers within the kernel's real allocation (SGPR auto-grow).
[[nodiscard]] constexpr bool probe_link_pair_fits_in_kernel(uint32_t kernel_sgpr_count,
                                                            uint16_t link_base) {
  return kernel_sgpr_count >= static_cast<uint32_t>(link_base) + 2;
}

/// @brief DBI orchestrator. Collects InstrumentationPoints, validates each
///        anchor, plans + builds per-site trampolines, and drives the
///        CodeObjectPatcher to splice the patches into a new ELF. See the
///        mental-model diagram at the top of this file for the full pipeline.
///
/// Single-attempt: each Instrumentor instance owns one call to patch() or
/// patch_with_debug_summaries(), success or failure. Could expand to allow
/// calls again upon failure. Re-instrumenting and already instrumented code
/// is a much more difficult task.
///
/// Block decoding is lazy so the CFG is built on the first call that needs
/// it (validate_points / find_instruction_at_offset). An Instrumentor that
/// never gets patched also never decodes.
class Instrumentor {
public:
  /// @param obj  The code object to instrument. Borrowed; must outlive the
  ///             Instrumentor.
  /// @param arch The ISA used to decode @p obj. An arch that has no
  ///             registered Decoder (RV32I/RV64I/INVALID/etc.) surfaces as
  ///             a ValidationResult / InstrumentedCodeObject error at
  ///             validate_points() / patch() time rather than at construction.
  Instrumentor(const AmdGpuCodeObject &obj, rj_code_arch_t arch);
  ~Instrumentor();

  Instrumentor(const Instrumentor &) = delete;
  Instrumentor &operator=(const Instrumentor &) = delete;

  /// @brief Queue a point. The point is not validated until validate_points().
  void add_point(InstrumentationPoint pt);

  /// @brief Convenience: queue a point identified by its .text-relative byte
  ///        offset. Equivalent to constructing an InstrumentationPoint with
  ///        anchor_offset set. Handy when callers walk basic blocks themselves
  ///        and already have the offset of each candidate instruction.
  void add_point_by_offset(uint64_t anchor_offset,
                           InstrumentationKind kind = InstrumentationKind::BeforeInst);

  struct ValidationResult {
    std::vector<ResolvedInstrumentationSite> sites;
    std::vector<std::string> errors; // Fatal; sites is empty when non-empty.
  };

  /// @brief Look up each queued point's anchor and validate it.
  ///
  /// Currently all-or-nothing: on any failure, `sites` is empty and `errors`
  /// lists every fatal diagnostic encountered. On success, `sites` contains
  /// one record per queued point in insertion order and `errors` is empty.
  [[nodiscard]] ValidationResult validate_points();

  /// @brief Validate, plan, build, and apply the queued points; emit a
  ///        patched ELF.
  ///
  /// All per-site validation and builder output is preflighted before any
  /// patcher mutation, so a late failure (e.g. branch-range overflow) can't
  /// leak a half-built ELF. Thus `elf_bytes` is either fully populated or empty
  /// with diagnostics in `errors`.
  ///
  /// Requires at least one queued InstrumentationPoint; queuing zero is a fatal
  /// error. Shares the single-attempt budget with patch_with_debug_summaries();
  /// see the class-level note.
  [[nodiscard]] InstrumentedCodeObject patch();

  /// @brief Same as patch(), plus per-site InstrumentationPatch summaries.
  ///
  /// Currently intended for tests and debugging, although could eventually
  /// be used to communicate important information for the host (an ordering
  /// of which instructions were instrumented and should be analyzed further
  /// for post-processing. Per-site info not yet stable.
  [[nodiscard]] InstrumentedCodeObjectDebug patch_with_debug_summaries();

private:
  const AmdGpuCodeObject &obj_;
  rj_code_arch_t arch_;
  std::vector<InstrumentationPoint> points_;
  bool patched_ = false;

  // Lazily populated.
  std::unique_ptr<Decoder> decoder_;
  std::vector<std::unique_ptr<BasicBlock>> blocks_;
  // .text-relative byte offset -> decoded Instruction. Populated alongside
  // blocks_ so find_instruction_at_offset is O(1). Pointers are stable for
  // the lifetime of blocks_ (BasicBlock owns the Instructions via unique_ptr).
  std::unordered_map<uint64_t, const Instruction *> offset_to_inst_;
  bool blocks_built_ = false;

  // Returns false if no decoder exists for arch_ (RV32I/RV64I/INVALID/etc.);
  // in that case *error_out is set and blocks_built_ stays false so the failure
  // is reported instead of crashing in BasicBlock::build.
  [[nodiscard]] bool ensure_blocks_built(std::string *error_out = nullptr);
  [[nodiscard]] const Instruction *find_instruction_at_offset(uint64_t anchor_offset) const;

  // Everything one resolution pass produces.
  struct ResolvedPoints {
    std::vector<ResolvedInstrumentationSite> sites;
    std::vector<ProbeCallable> probes;
    std::vector<std::string> errors;
  };

  // Validate and resolve every queued point in one pass.
  [[nodiscard]] ResolvedPoints resolve_points();
};

} // namespace rocjitsu
