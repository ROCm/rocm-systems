// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file instrumentor.h
/// @brief DBI orchestrator: resolves InstrumentationPoints to .text-relative
///        anchors, validates them, and (in a later slice) drives byte-level
///        patching via TrampolineBuilder and CodeObjectPatcher.

#pragma once

#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/code/rj_code.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
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
/// TODO(future milestone): implement AfterInst, BlockEntry, and BlockExit.
/// Today the validator rejects any kind other than BeforeInst.
enum class InstrumentationKind {
  BeforeInst,
  AfterInst,
  BlockEntry,
  BlockExit,
};

/// @brief A request to instrument one site.
///
/// The anchor is identified by either @ref anchor_inst (when the caller
/// already has a decoded Instruction pointer from an Instrumentor-owned
/// block) or @ref anchor_offset (test/internal shortcut). Exactly one must
/// be set; setting both or neither is a fatal error during resolution.
struct InstrumentationPoint {
  const Instruction *anchor_inst = nullptr;
  std::optional<uint64_t> anchor_offset;

  InstrumentationKind kind = InstrumentationKind::BeforeInst;
  uint32_t filter_flags = 0; // Must be 0 in the inline-nop smoke build.

  // Reserved for later milestones — must remain default in the inline-nop
  // smoke build. The validator rejects any non-default value to keep the
  // contract honest until each field is actually implemented.
  // TODO(future milestone): consume probe_obj / probe_symbol when probe-call
  // trampolines are supported; consume force_full_exec when EXEC policy
  // management lands.
  const AmdGpuCodeObject *probe_obj = nullptr;
  std::string probe_symbol;
  bool force_full_exec = false;
};

/// @brief Per-site record produced after validation and byte capture.
///
/// @note `anchor_inst` points into BasicBlock storage owned by the
///       Instrumentor that produced the site. Consumers must not retain
///       sites past the Instrumentor's lifetime.
struct ResolvedInstrumentationSite {
  const Instruction *anchor_inst = nullptr;
  InstrumentationKind kind = InstrumentationKind::BeforeInst;
  uint64_t anchor_offset = 0;
  uint32_t original_size = 0; // 4 or 8 in the inline-nop smoke build.
  std::vector<uint8_t> original_bytes;
  std::string mnemonic; // Diagnostic/debug only.
};

/// @brief Per-site patch record returned by patch_relocation_only().
///
/// Intended for test assertions and debugging only — a fresh disassembly of
/// the emitted ELF would expose the same information. Not a proposed public
/// metadata surface.
struct InstrumentationPatch {
  uint64_t anchor_offset;
  uint32_t original_size;
  uint64_t trampoline_offset; // .text-relative.
  uint64_t return_target;
  std::vector<uint8_t> original_bytes;
  std::vector<uint8_t> patched_anchor_bytes;
};

/// @brief Result of patch_relocation_only().
///
/// All-or-nothing: when any fatal error occurs, `elf_bytes` and `patches`
/// are empty and `errors` is non-empty. On success, `errors` is empty,
/// `elf_bytes` contains a re-parseable patched ELF, and `patches` contains
/// one record per applied site.
struct InstrumentedCodeObject {
  std::vector<uint8_t> elf_bytes;
  std::vector<InstrumentationPatch> patches;
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
};

/// @brief Validate that @p anchor is a legal relocation target.
///
/// Called by Instrumentor::resolve_and_validate() and also directly by tests
/// (the free-function form lets test fixtures use synthetic TestInstruction
/// objects without standing up an AmdGpuCodeObject). The anchor identity is
/// already resolved by the caller; @p pt is read for `filter_flags`, `kind`,
/// and reserved fields. @p arch is accepted now so a future denylist can
/// grow ISA-specific entries without an API change; today's checks are
/// uniform across all AMDGPU ISAs.
///
/// Rules enforced:
///   - @p pt.filter_flags is zero.
///   - @p pt.kind is BeforeInst (other kinds are unsupported in this milestone).
///   - @p pt.probe_obj is null, @p pt.probe_symbol is empty, and
///     @p pt.force_full_exec is false.
///   - @p anchor_offset is dword aligned.
///   - anchor.size() is 4 or 8 and fits inside @p text_bytes.
///   - anchor.raw_encoding() is non-null.
///   - anchor is not a branch/cond branch/indirect branch/indirect call/
///     program terminator, and branch_offset_bytes() is nullopt.
///   - anchor.mnemonic() is not in the small PC-relative denylist
///     (s_getpc_b64, s_call_b64, s_setpc_b64, s_swappc_b64, s_rfe_*).
[[nodiscard]] std::optional<ResolvedInstrumentationSite>
validate_relocation_anchor(const Instruction &anchor, uint64_t anchor_offset,
                           std::span<const uint8_t> text_bytes,
                           const InstrumentationPoint &pt, rj_code_arch_t arch,
                           std::string *error_out = nullptr);

/// @brief Build the inline-nop smoke TrampolinePlan from a validated site.
///
/// Fills before_items = {{ s_nop 0 }}, after_items = {}, emit_original = true.
/// The caller chooses @p trampoline_offset (typically `patcher.text_size()`).
[[nodiscard]] TrampolinePlan
make_relocation_only_plan(const ResolvedInstrumentationSite &site, rj_code_arch_t arch,
                          uint64_t trampoline_offset);

/// @brief Verify @p plan matches the inline-nop smoke build's canonical body
///        shape: exactly one `before_items` entry containing `s_nop 0`, empty
///        `after_items`, and `emit_original == true`.
///
/// Lives at the orchestrator boundary rather than inside TrampolineBuilder so
/// the builder stays generic and ready for future probe-call / clobber-bearing
/// inline-asm bodies. Called by Instrumentor::patch_relocation_only as a
/// defense-in-depth check (make_relocation_only_plan always produces canonical
/// plans today) and also directly by tests so the guardrail logic is
/// exercised at the layer where it lives.
///
/// TODO(future milestone): delete this once arbitrary inline-asm bodies with
/// declared clobbers are supported.
[[nodiscard]] bool
validate_inline_nop_smoke_plan(const TrampolinePlan &plan,
                               std::string *error_out = nullptr);

/// @brief DBI orchestrator owning point collection and resolution.
///
/// PC01-A scope: resolve points to .text-relative offsets, validate, capture
/// original bytes. Patching (TrampolineBuilder lowering + CodeObjectPatcher
/// mutation) arrives in the next slice.
class Instrumentor {
public:
  Instrumentor(const AmdGpuCodeObject &obj, rj_code_arch_t arch);
  ~Instrumentor();

  Instrumentor(const Instrumentor &) = delete;
  Instrumentor &operator=(const Instrumentor &) = delete;

  /// @brief Queue a point. The point is not validated until resolve_and_validate().
  void add_point(InstrumentationPoint pt);

  /// @brief Convenience: queue a point identified only by .text-relative offset.
  ///        Equivalent to constructing an InstrumentationPoint with
  ///        anchor_offset set.
  void add_point_by_offset(uint64_t anchor_offset,
                           InstrumentationKind kind = InstrumentationKind::BeforeInst);

  /// @brief Force the lazy block build and return the owned blocks.
  ///
  /// Callers walk these to choose anchor candidates, then queue the chosen
  /// Instruction* via add_point(). Pointers obtained any other way fail
  /// resolution.
  [[nodiscard]] std::span<const std::unique_ptr<BasicBlock>> owned_blocks();

  struct ResolveResult {
    std::vector<ResolvedInstrumentationSite> sites;
    std::vector<std::string> errors; // Fatal; sites is empty when non-empty.
  };

  /// @brief Resolve and validate all queued points.
  ///
  /// All-or-nothing: on any failure, `sites` is empty and `errors` lists every
  /// fatal diagnostic encountered. On success, `sites` contains one record per
  /// queued point in insertion order and `errors` is empty.
  ///
  /// @note `sites[i].anchor_inst` points into BasicBlock storage owned by
  ///       this Instrumentor. Consumers must not retain sites past this
  ///       Instrumentor's lifetime.
  [[nodiscard]] ResolveResult resolve_and_validate();

  /// @brief Resolve, validate, plan, build, and patch the queued points.
  ///
  /// Inline-nop smoke build accepts exactly one queued point; queuing zero
  /// or more than one is a fatal error. The orchestrator preflights all
  /// builder output before mutating the patcher, so a branch-range failure
  /// can't leak a half-built ELF.
  ///
  /// Single-attempt: any call (success or failure) consumes the Instrumentor's
  /// budget. A second call returns an empty result with a fatal error, even
  /// if the first call failed for argument reasons (e.g., zero queued points).
  /// Callers that hit a recoverable error must construct a new Instrumentor.
  [[nodiscard]] InstrumentedCodeObject patch_relocation_only();

private:
  const AmdGpuCodeObject &obj_;
  rj_code_arch_t arch_;
  std::vector<InstrumentationPoint> points_;
  bool patched_ = false;

  // Lazily populated.
  std::unique_ptr<Decoder> decoder_;
  std::vector<std::unique_ptr<BasicBlock>> blocks_;
  bool blocks_built_ = false;

  void ensure_blocks_built();
  [[nodiscard]] const Instruction *find_instruction_at_offset(uint64_t anchor_offset);
  [[nodiscard]] std::optional<uint64_t>
  resolve_anchor_inst_to_offset(const Instruction *target, std::string *error_out);
};

} // namespace rocjitsu
