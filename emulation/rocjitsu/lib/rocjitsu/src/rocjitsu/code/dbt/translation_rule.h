// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translation_rule.h
/// @brief Core types for the three-tier semantic translation framework.
///
/// @details Provides a declarative, data-driven representation for cross-ISA
/// instruction translation. Three tiers:
///
/// 1. **InstructionDescriptor** — per-instruction metadata (operand widths,
///    register effects, flags). Auto-generated from ISA specification XML.
///
/// 2. **TranslationRule** — per-(source, target) instruction pair translation
///    action. Four kinds: Identity (no change), Substitute (opcode swap),
///    FieldRemap (field-level transformation chain), Expand (multi-instruction
///    lowering with optional LaneLayout).
///
/// 3. **LaneLayout** — matrix instruction data distribution descriptor.
///    Captures the (row, col, K) → (lane, vgpr, bit_offset) mapping
///    algebraically. Used by Expand rules for MFMA→WMMA translation to
///    derive the cross-lane shuffle sequence from the layout difference.

#pragma once

#include "util/bit.h"

#include <algorithm>
#include <array>
#include <compare>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {

class Instruction;
class LivenessAnalysis;

/// @brief Architecture-neutral resource accounting shared by semantic lowerings.
///
/// @details This state is created once per kernel from the current target kernel
/// descriptor translation, then passed through every semantic EXPAND rule for
/// that kernel. The `num_*` fields describe the descriptor state the lowering
/// started from. The `required_*` fields are feedback from lowerings that
/// allocated scratch registers beyond those descriptor counts. Feature-specific
/// lowering state belongs in a separate context component.
///
/// Descriptor translation happens before instruction translation, but semantic
/// lowerings only know their actual scratch choices after liveness has been
/// computed. Each kernel is lowered once while recording the highest SGPR/VGPR
/// and private-memory requirements here; BinaryTranslator then recomputes the
/// affected descriptor translations with those larger minimums before patching
/// descriptors into the output image. A second instruction pass is only needed
/// if a future lowering depends on descriptor-derived register numbers that can
/// change during that recomputation.
struct KernelResourceRequirements {
  /// @brief Initial target ordinary VGPR count from descriptor translation.
  uint32_t num_vgprs = 0;

  /// @brief Initial target AccVGPR count from descriptor translation.
  uint32_t num_agprs = 0;

  /// @brief Initial target AccVGPR base used by descriptor-backed AccVGPR allocation.
  uint32_t accum_offset = 0;

  /// @brief Initial target SGPR count from descriptor translation.
  uint32_t num_sgprs = 0;

  /// @brief Minimum ordinary VGPR count required after all semantic lowerings.
  ///
  /// @details Lowerings update this with require_vgprs() when a chosen scratch
  /// VGPR is outside the descriptor's initial ordinary VGPR allocation. The
  /// value is a count, not a register index, so callers pass the highest used
  /// VGPR index plus one.
  uint32_t required_vgpr_count = 0;

  /// @brief Minimum SGPR count required after all semantic lowerings.
  ///
  /// @details Lowerings update this with require_sgprs() when a chosen scratch
  /// SGPR is outside the descriptor's initial SGPR allocation. The value is a
  /// count, not a register index, so callers pass the highest used SGPR index
  /// plus one.
  uint32_t required_sgpr_count = 0;

  /// @brief Initial per-lane private segment size from descriptor translation.
  uint32_t private_segment_fixed_size = 0;

  /// @brief Whether private memory above the fixed segment is a dynamic call stack.
  bool uses_dynamic_stack = false;

  /// @brief Minimum per-lane private segment size required after spill-backed lowerings.
  uint32_t required_private_segment_fixed_size = 0;

  /// @brief End of persistent semantic spill storage before reusable temp slots.
  ///
  /// @details Persistent spill storage must not overlap the reusable
  /// per-instruction spill window. Virtual LDS uses this when a descriptor-full
  /// kernel has to save the backing-buffer pointer in private memory before the
  /// guest body is allowed to clobber the dispatch/kernarg pointer SGPRs.
  uint32_t semantic_spill_persistent_end = 0;

  /// @brief Construct an empty context component for tests or call sites without
  /// descriptor feedback.
  KernelResourceRequirements() = default;

  /// @brief Construct resource accounting for kernels that only need VGPR/SGPR
  /// descriptor state.
  /// @param vgprs Initial target ordinary VGPR count.
  /// @param sgprs Initial target SGPR count.
  KernelResourceRequirements(uint32_t vgprs, uint32_t sgprs) : num_vgprs(vgprs), num_sgprs(sgprs) {}

  /// @brief Construct full resource accounting from target descriptor translation.
  /// @param vgprs Initial target ordinary VGPR count.
  /// @param agprs Initial target AccVGPR count.
  /// @param accum_base Initial target AccVGPR base.
  /// @param sgprs Initial target SGPR count.
  /// @param private_bytes Initial per-lane private segment size.
  /// @param dynamic_stack Whether the kernel uses a runtime-managed dynamic stack.
  KernelResourceRequirements(uint32_t vgprs, uint32_t agprs, uint32_t accum_base, uint32_t sgprs,
                             uint32_t private_bytes = 0, bool dynamic_stack = false)
      : num_vgprs(vgprs), num_agprs(agprs), accum_offset(accum_base), num_sgprs(sgprs),
        private_segment_fixed_size(private_bytes), uses_dynamic_stack(dynamic_stack),
        required_private_segment_fixed_size(private_bytes),
        semantic_spill_persistent_end(private_bytes) {}

  /// @brief Record that semantic lowering requires at least @p count ordinary VGPRs.
  ///
  /// @details This is monotonic across all lowerings for the kernel. It only
  /// raises the required count and never reduces an earlier requirement.
  void require_vgprs(uint32_t count) {
    if (required_vgpr_count < count)
      required_vgpr_count = count;
  }

  /// @brief Record that semantic lowering requires at least @p count SGPRs.
  ///
  /// @details This is monotonic across all lowerings for the kernel. It only
  /// raises the required count and never reduces an earlier requirement.
  void require_sgprs(uint32_t count) {
    if (required_sgpr_count < count)
      required_sgpr_count = count;
  }

  /// @brief Record a minimum per-lane private segment size.
  ///
  /// @details SemanticSpillFrame calls this as anonymous transient spill slots
  /// are allocated. Keeping the high-water update here leaves descriptor
  /// feedback independent of the target instructions used to access the slots.
  void require_private_segment_bytes(uint32_t bytes) {
    if (required_private_segment_fixed_size < bytes)
      required_private_segment_fixed_size = bytes;
  }

  /// @brief Reserve @p dwords persistent 32-bit per-lane spill slots.
  ///
  /// @details Persistent semantic spill slots hold values across multiple
  /// replacement sequences. They are allocated before the reusable temp window
  /// so later per-instruction spills cannot overwrite them.
  ///
  /// @returns The per-lane byte offset of the reserved slots, or std::nullopt if
  /// aligning/extending the private segment would exceed the 32-bit private-size
  /// field. The private segment is initialized from the guest descriptor's
  /// private size, which can be near UINT32_MAX, so unchecked 32-bit arithmetic
  /// here could wrap to a low offset and corrupt guest scratch. Callers must
  /// treat nullopt as a kernel translation failure.
  [[nodiscard]] std::optional<uint32_t> reserve_persistent_semantic_spill_dwords(uint32_t dwords) {
    const auto reservation = semantic_spill_reservation(dwords);
    if (!reservation)
      return std::nullopt;
    semantic_spill_persistent_end = reservation->second;
    require_private_segment_bytes(reservation->second);
    return reservation->first;
  }

  /// @brief Reserve @p dwords reusable 32-bit per-lane spill slots.
  ///
  /// @details Semantic spill slots are appended after the kernel's original
  /// private segment and any persistent semantic spill storage at a 16-byte
  /// boundary, matching the patch-layer spill manager's flat-scratch layout.
  /// They are scratch temporaries for a single replacement sequence, not
  /// persistent virtual registers, so every lowering site in the kernel can
  /// reuse the same slot range. Descriptor recomputation later raises
  /// private_segment_fixed_size to cover the largest reservation.
  ///
  /// @returns The per-lane byte offset of the reserved slots, or std::nullopt on
  /// 32-bit overflow (see reserve_persistent_semantic_spill_dwords). Unlike the
  /// persistent variant this does not advance semantic_spill_persistent_end.
  [[nodiscard]] std::optional<uint32_t> reserve_semantic_spill_dwords(uint32_t dwords) {
    const auto reservation = semantic_spill_reservation(dwords);
    if (!reservation)
      return std::nullopt;
    require_private_segment_bytes(reservation->second);
    return reservation->first;
  }

private:
  /// @brief Compute the aligned base and one-past-end for a spill reservation.
  ///
  /// @returns {base_byte_offset, end_byte_offset} using checked 64-bit math, or
  /// std::nullopt if either would exceed the 32-bit private-size field.
  [[nodiscard]] std::optional<std::pair<uint32_t, uint32_t>>
  semantic_spill_reservation(uint32_t dwords) const {
    constexpr uint64_t kSpillAlignment = 16;
    constexpr uint64_t kMax = std::numeric_limits<uint32_t>::max();
    const uint64_t base =
        util::align_up(static_cast<uint64_t>(semantic_spill_persistent_end), kSpillAlignment);
    const uint64_t end = base + static_cast<uint64_t>(dwords) * 4u;
    if (base > kMax || end > kMax)
      return std::nullopt;
    return std::pair<uint32_t, uint32_t>{static_cast<uint32_t>(base), static_cast<uint32_t>(end)};
  }
};

/// @brief State owned by the virtual-LDS feature while lowering one kernel.
struct VirtualLdsTranslationState {
  /// @brief True when this kernel's LDS accesses target a global-memory backing buffer.
  ///
  /// @details Descriptor translation sets hardware LDS to zero in this mode, so
  /// any real LDS read/write instruction in the source body must be rewritten.
  /// Cross-lane DS instructions that do not access LDS storage, such as
  /// bpermute, may still use the DS unit directly.
  bool virtualize_lds = false;

  /// @brief 64-bit SGPR-pair base address for the virtual-LDS backing buffer.
  ///
  /// @details CDNA3 flat/global addressing uses this SGPR pair plus the source
  /// DS address VGPR and folded DS immediate offset. The runtime/prologue path
  /// is responsible for loading this pair before the rewritten body executes.
  uint16_t virtual_lds_base_sgpr = 0;

  /// @brief True when the virtual-LDS base SGPR pair is borrowed per DS use.
  ///
  /// @details Some kernels already allocate and touch every ordinary SGPR pair.
  /// For those kernels, DBT preserves the selected pair around each lowered LDS
  /// memory operation instead of permanently clobbering it in the entry
  /// prologue.
  bool virtual_lds_base_sgpr_spill_per_use = false;

  /// @brief True when the runtime backing pointer was spilled at entry.
  bool virtual_lds_base_pointer_spilled = false;

  /// @brief Private scratch offset of the spilled virtual-LDS backing pointer.
  uint32_t virtual_lds_base_pointer_spill_offset = 0;

  /// @brief Target kernarg segment pointer SGPR pair used by entry-only wrapper loads.
  uint16_t virtual_lds_kernarg_segment_ptr_sgpr = 0;

  /// @brief Kernarg-wrapper byte offset of the runtime virtual-LDS state.
  uint32_t virtual_lds_kernarg_pointer_offset = 0;
};

/// @brief Per-kernel state passed through semantic translation rules.
///
/// @details Inheritance preserves the compact field access used by existing
/// rules while making the ownership split explicit. Generic DBT/DBI resource
/// helpers can consume KernelResourceRequirements without depending on virtual
/// LDS, and pair-specific lowerings can consume VirtualLdsTranslationState.
struct TranslationContext : KernelResourceRequirements, VirtualLdsTranslationState {
  TranslationContext() = default;

  TranslationContext(uint32_t vgprs, uint32_t sgprs) : KernelResourceRequirements(vgprs, sgprs) {}

  TranslationContext(uint32_t vgprs, uint32_t agprs, uint32_t accum_base, uint32_t sgprs,
                     uint32_t private_bytes = 0, bool dynamic_stack = false)
      : KernelResourceRequirements(vgprs, agprs, accum_base, sgprs, private_bytes, dynamic_stack) {}
};

/// @brief Status returned by a semantic EXPAND rule lookup or expansion.
enum class ExpandStatus {
  NotHandled, ///< No rule matched this instruction.
  Success,    ///< A rule emitted replacement instruction words.
  Failed,     ///< A rule matched but could not safely emit a lowering.
};

/// @brief Structured result from a semantic EXPAND rule.
///
/// @details Empty replacement words are no longer overloaded to mean both
/// "there is no rule" and "a rule exists but failed." BinaryTranslator uses the
/// status to hard-fail unimplemented or failed EXPAND legalizations with useful
/// diagnostics instead of silently NOP-filling the source instruction.
struct ExpandResult {
  ExpandStatus status = ExpandStatus::NotHandled;
  std::vector<uint32_t> words;
  std::string message;
  std::vector<std::string> required_work;

  [[nodiscard]] static ExpandResult not_handled() { return {}; }

  [[nodiscard]] static ExpandResult success(std::vector<uint32_t> replacement_words) {
    ExpandResult result;
    result.status = ExpandStatus::Success;
    result.words = std::move(replacement_words);
    return result;
  }

  [[nodiscard]] static ExpandResult failed(std::string failure_message,
                                           std::vector<std::string> work = {}) {
    ExpandResult result;
    result.status = ExpandStatus::Failed;
    result.message = std::move(failure_message);
    result.required_work = std::move(work);
    return result;
  }
};

// ---------------------------------------------------------------------------
// Tier 1: Instruction Descriptor
// ---------------------------------------------------------------------------

/// @brief Flags describing instruction properties relevant to translation.
enum InstructionProperty : uint32_t {
  PROP_NONE = 0,
  PROP_EXEC_MASKED = 1u << 0,  ///< Instruction respects EXEC mask.
  PROP_IS_MATRIX = 1u << 1,    ///< Matrix multiply-accumulate (MFMA/WMMA).
  PROP_IS_BARRIER = 1u << 2,   ///< Synchronization barrier.
  PROP_HAS_LITERAL = 1u << 3,  ///< May have a trailing 32-bit literal.
  PROP_USES_ACCVGPR = 1u << 4, ///< Reads or writes AccVGPR file.
  PROP_IS_WAITCNT = 1u << 5,   ///< Wait-counter instruction.
  PROP_IGNORES_EXEC = 1u << 6, ///< Executes regardless of EXEC mask.
  PROP_CROSS_LANE = 1u << 7,   ///< Cross-lane data movement.
  PROP_DS_PERMUTE = 1u << 8,   ///< DS-based cross-lane permute (uses LDS crossbar).
  PROP_NEEDS_DSCNT = 1u << 9,  ///< Result requires s_wait_dscnt before use.
};

/// @brief GFX12 hazard class for s_delay_alu scheduling.
enum class HazardPipeline : uint8_t {
  None = 0,
  VALU = 1,  ///< Standard VALU pipeline (1-4 instruction latency).
  TRANS = 2, ///< Transcendental pipeline (1-3 instruction latency).
  SALU = 3,  ///< Scalar ALU pipeline (1-3 cycle latency).
};

/// @brief Register effect of an instruction: which register classes are
/// read, written, or clobbered.
struct RegEffect {
  uint8_t vgpr_read_count = 0;  ///< Number of VGPRs read (src operands).
  uint8_t vgpr_write_count = 0; ///< Number of VGPRs written (dst operands).
  uint8_t sgpr_read_count = 0;
  uint8_t sgpr_write_count = 0;
  bool reads_exec = false;
  bool writes_exec = false;
  bool reads_vcc = false;
  bool writes_vcc = false;
};

/// @brief Per-instruction metadata, auto-generated from ISA specification.
struct InstructionDescriptor {
  uint16_t encoding_id;   ///< Encoding format ID (VOP1, VOP3P, etc.).
  uint16_t opcode;        ///< Opcode within the encoding format.
  const char *mnemonic;   ///< Human-readable mnemonic.
  uint8_t num_src;        ///< Number of source operands.
  uint8_t num_dst;        ///< Number of destination operands.
  uint16_t src_widths[4]; ///< Source operand widths in bits.
  uint16_t dst_widths[2]; ///< Destination operand widths in bits.
  uint32_t properties;    ///< Bitwise OR of InstructionProperty flags.
  RegEffect reg_effect;   ///< Register read/write/clobber effects.
};

// ---------------------------------------------------------------------------
// Tier 2: Translation Rule
// ---------------------------------------------------------------------------

/// @brief Describes a single field transformation within a FieldRemap rule.
///
/// @details Each FieldMap extracts bits from a source encoding field, applies
/// a shift and mask, optionally clamps to a maximum, and writes to a target
/// field. A chain of FieldMaps can express transformations like GFX9 waitcnt
/// splitting (one source field → multiple target fields with different
/// bit ranges and saturation).
struct FieldMap {
  uint8_t src_word;  ///< Source instruction word index (0, 1, or 2).
  uint8_t src_shift; ///< Right-shift to extract the source field.
  uint8_t src_width; ///< Width of the source field in bits.
  uint8_t dst_word;  ///< Target instruction word index.
  uint8_t dst_shift; ///< Left-shift for placement in the target word.
  uint8_t dst_width; ///< Width of the target field in bits.
  uint8_t clamp_max; ///< Saturate value (0 = no clamping).
};

/// @brief Translation rule action kind.
enum class RuleAction : uint8_t {
  Identity,   ///< No translation needed — encoding is compatible.
  Substitute, ///< Same encoding format, different opcode.
  FieldRemap, ///< Field-level transformation chain.
  Expand,     ///< Multi-instruction lowering (code cave eligible).
};

struct LaneLayout;

/// @brief Function type for Expand rule expansion generators.
///
/// @param inst          The decoded guest instruction to expand.
/// @param arch          Target ISA architecture.
/// @param offset        Byte offset of the instruction in .text.
/// @param source_text   Full source .text bytes, used when trailing modifier/literal words are
///                      not retained by the decoded Instruction object.
/// @param liveness      Kernel-scoped live-before data for safe scratch register allocation.
/// @param guest_layout  Source matrix lane layout (nullptr if not a matrix op).
/// @param host_layout   Target matrix lane layout (nullptr if not a matrix op).
/// @returns Structured expansion status and replacement words.
using ExpandFn = ExpandResult (*)(const Instruction &inst, uint32_t arch, uint64_t offset,
                                  std::span<const uint8_t> source_text,
                                  const LivenessAnalysis &liveness, TranslationContext &context,
                                  const LaneLayout *guest_layout, const LaneLayout *host_layout);

/// @brief Read-only test for whether an implemented expansion is still actionable.
///
/// @details The predicate observes only the decoded instruction stream. It must
/// share any operand or neighboring-instruction trigger checks used by the
/// corresponding ExpandFn, but must not allocate resources or emit words.
using ResidualExpandFn = bool (*)(const Instruction &inst);

/// @brief Whether and how a rewrite participates in final-stream discharge.
enum class RewriteDischargeDisposition : uint8_t {
  Unregistered,          ///< No audit contract was declared; audited profiles reject this state.
  Checked,               ///< Run the registered read-only predicate on the final stream.
  NoSuccessfulExpansion, ///< The rule may decline or fail, but must never emit output.
};

/// @brief Decoded context required by one residual rewrite predicate.
enum class RewriteDischargeContext : uint8_t {
  Instruction, ///< The predicate observes only the candidate instruction.
  BasicBlock,  ///< The predicate observes neighboring instructions in the decoded basic block.
};

/// @brief Explicit final-stream audit contract for one rewrite.
///
/// @details Audited profiles require every rewrite to use checked() or
/// no_success(). A checked predicate declares the least decoded context it
/// needs: instruction-local checks support a constant-memory stream scan, while
/// basic-block checks require the verifier to retain the decoded CFG.
/// The default Unregistered state keeps older, unaudited profiles source
/// compatible without allowing an audited profile to silently omit a rule.
class RewriteDischarge {
public:
  RewriteDischargeDisposition disposition = RewriteDischargeDisposition::Unregistered;
  ResidualExpandFn check = nullptr;
  const char *rationale = nullptr;
  RewriteDischargeContext context = RewriteDischargeContext::Instruction;

  [[nodiscard]] static constexpr RewriteDischarge
  checked(ResidualExpandFn predicate, RewriteDischargeContext required_context) {
    return {RewriteDischargeDisposition::Checked, predicate, nullptr, required_context};
  }

  [[nodiscard]] static constexpr RewriteDischarge no_success(const char *reason) {
    return {RewriteDischargeDisposition::NoSuccessfulExpansion, nullptr, reason,
            RewriteDischargeContext::Instruction};
  }

  [[nodiscard]] constexpr bool valid() const {
    if (disposition == RewriteDischargeDisposition::Checked) {
      const bool valid_context = context == RewriteDischargeContext::Instruction ||
                                 context == RewriteDischargeContext::BasicBlock;
      return check != nullptr && rationale == nullptr && valid_context;
    }
    if (disposition == RewriteDischargeDisposition::NoSuccessfulExpansion)
      return check == nullptr && rationale != nullptr && rationale[0] != '\0' &&
             context == RewriteDischargeContext::Instruction;
    return false;
  }

  /// @brief Whether @p status is compatible with the declared audit contract.
  [[nodiscard]] constexpr bool allows(ExpandStatus status) const {
    return disposition != RewriteDischargeDisposition::NoSuccessfulExpansion ||
           status != ExpandStatus::Success;
  }
};

/// @brief Applicability test for a non-opcode-keyed instruction rewrite.
using InstructionRewriteAppliesFn = bool (*)(const Instruction &inst);

/// @brief Lowering callback for a non-opcode-keyed instruction rewrite.
using InstructionRewriteFn = ExpandResult (*)(const Instruction &inst, uint64_t offset,
                                              std::span<const uint8_t> source_text,
                                              const LivenessAnalysis &liveness,
                                              TranslationContext &context);

/// @brief One rewrite selected by operands or other instruction-local state.
///
/// @details These rules are kept separate from the sorted opcode table so that
/// table lookup remains cheap. The same declaration drives applicability,
/// liveness selection, lowering, and final-stream discharge.
class RegisteredInstructionRewrite {
public:
  const char *name = nullptr;
  InstructionRewriteAppliesFn applies = nullptr;
  InstructionRewriteFn lower = nullptr;
  bool requires_liveness = true;
  RewriteDischarge discharge;

  [[nodiscard]] constexpr bool valid() const {
    return name != nullptr && name[0] != '\0' && applies != nullptr && lower != nullptr &&
           discharge.valid();
  }
};

/// @brief A single translation rule for one (source, target) instruction.
///
/// @details Keyed by (src_encoding_id, src_opcode) for lookup via binary search.
/// Different encoding formats can share the same opcode (e.g., SOPP s_waitcnt
/// and SOP2 s_and_b32 both use opcode 12), so encoding_id is required to
/// disambiguate.
struct TranslationRule {
  uint16_t src_encoding_id; ///< Source encoding format ID (e.g., SOPP, VOP3P).
  uint16_t src_opcode;      ///< Source opcode within the encoding format.
  RuleAction action;        ///< What kind of translation to apply.

  uint16_t dst_opcode; ///< Target opcode (for Substitute).

  uint8_t num_field_maps;     ///< Number of FieldMap entries (for FieldRemap).
  const FieldMap *field_maps; ///< Field transformation chain (for FieldRemap).

  ExpandFn expand_fn;             ///< Expansion generator (for Expand).
  const LaneLayout *guest_layout; ///< Source matrix layout (for matrix Expand).
  const LaneLayout *host_layout;  ///< Target matrix layout (for matrix Expand).
  bool requires_liveness = true;  ///< Conservative default; tables opt out after auditing.
  /// Explicit final-stream audit disposition for this expansion.
  RewriteDischarge discharge;

  constexpr TranslationRule(uint16_t source_encoding_id, uint16_t source_opcode,
                            RuleAction rule_action, uint16_t target_opcode, uint8_t field_map_count,
                            const FieldMap *maps, ExpandFn expansion,
                            const LaneLayout *source_layout, const LaneLayout *target_layout,
                            bool needs_liveness = true, RewriteDischarge audit = {})
      : src_encoding_id(source_encoding_id), src_opcode(source_opcode), action(rule_action),
        dst_opcode(target_opcode), num_field_maps(field_map_count), field_maps(maps),
        expand_fn(expansion), guest_layout(source_layout), host_layout(target_layout),
        requires_liveness(needs_liveness), discharge(audit) {}

  constexpr auto operator<=>(const TranslationRule &rhs) const {
    if (auto cmp = src_encoding_id <=> rhs.src_encoding_id; cmp != 0)
      return cmp;
    return src_opcode <=> rhs.src_opcode;
  }
  constexpr bool operator==(const TranslationRule &rhs) const {
    return src_encoding_id == rhs.src_encoding_id && src_opcode == rhs.src_opcode;
  }
};

/// @brief Whether a semantic rule table is strictly ordered for binary search.
///
/// @details SemanticTranslator::find_expand_rule() binary-searches these tables,
/// so an entry in the wrong place misses its own rule rather than failing.
/// Encoding ids are derived rather than written down -- a SOPK id, for instance,
/// is the SOPK base plus the opcode -- which makes a misplaced entry easy to
/// introduce and silent to observe. Every table static_asserts this.
///
/// Strict rather than merely nondecreasing: the search takes the first match, so
/// a duplicated (encoding id, opcode) leaves the second rule permanently
/// unreachable -- the same silent miss, arrived at from the other direction.
/// Catching that case depends on the comparison below ordering rules by their
/// key alone, which is what makes two entries that share a key but differ in
/// action or handler compare greater-equal here. Defaulting those operators
/// would keep the out-of-order half working and silently drop the duplicate
/// half.
[[nodiscard]] constexpr bool translation_rules_sorted(std::span<const TranslationRule> rules) {
  return std::ranges::adjacent_find(rules, std::ranges::greater_equal{}) == rules.end();
}

namespace translation_rule_detail {

/// @brief Two rules sharing a key but differing in everything else.
///
/// @details The duplicate half of translation_rules_sorted() works only while
/// the comparison ignores these trailing fields. Defaulting the operators would
/// order this pair by them instead, so the pair would compare ascending, every
/// table in the tree would still satisfy its own assertion, and the duplicate
/// check would be silently gone. Pinning it here fails at the definition rather
/// than at some future table that happens to duplicate a key.
inline constexpr std::array<TranslationRule, 2> kSameKeyDifferentTail = {{
    {1, 2, RuleAction::Identity, 0, 0, nullptr, nullptr, nullptr, nullptr, true},
    {1, 2, RuleAction::Expand, 3, 0, nullptr, nullptr, nullptr, nullptr, false},
}};
static_assert(!translation_rules_sorted(kSameKeyDifferentTail),
              "TranslationRule must order by (encoding id, opcode) alone, or "
              "translation_rules_sorted() stops rejecting duplicate keys");

} // namespace translation_rule_detail

/// @brief Complete handwritten rewrite declaration for one translation profile.
///
/// @details Opcode-keyed expansions and non-table instruction rewrites retain
/// their existing dispatch mechanisms, but are selected here as one profile.
/// Verification is available only when the registry is nonempty, the opcode
/// rules have strictly ordered unique keys, and every declared rule carries a
/// valid checked or no-success disposition.
class RewriteRegistry {
public:
  std::span<const TranslationRule> opcode_rules;
  std::span<const RegisteredInstructionRewrite> instruction_rules;

  /// @brief Whether a non-opcode rule requires contextual final-stream analysis.
  ///
  /// @details These predicates cannot be evaluated safely by the streaming
  /// verifier because their neighboring instructions do not exist until the CFG
  /// is built. Conservatively request that CFG independent of the lowering
  /// selector, which is allowed to differ from the residual predicate.
  [[nodiscard]] constexpr bool instruction_rewrites_require_basic_block() const {
    for (const RegisteredInstructionRewrite &rule : instruction_rules) {
      if (rule.discharge.disposition == RewriteDischargeDisposition::Checked &&
          rule.discharge.context == RewriteDischargeContext::BasicBlock) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] constexpr bool has_complete_discharge() const {
    if (opcode_rules.empty() && instruction_rules.empty())
      return false;
    for (size_t rule_index = 0; rule_index < opcode_rules.size(); ++rule_index) {
      const TranslationRule &rule = opcode_rules[rule_index];
      if (rule.action != RuleAction::Expand || rule.expand_fn == nullptr || !rule.discharge.valid())
        return false;
      if (rule_index != 0 && !(opcode_rules[rule_index - 1] < rule))
        return false;
    }
    for (const RegisteredInstructionRewrite &rule : instruction_rules) {
      if (!rule.valid())
        return false;
    }
    return true;
  }
};

// ---------------------------------------------------------------------------
// Tier 3: Lane Layout Descriptor
// ---------------------------------------------------------------------------

/// @brief Describes how a matrix instruction distributes data across
/// wavefront lanes and VGPRs.
///
/// @details Each matrix instruction (MFMA, WMMA) has a specific mapping from
/// matrix element coordinates (row, col, K) to hardware positions
/// (lane_id, vgpr_index, bit_offset). This descriptor captures that mapping
/// compactly, enabling the Expand rule to derive the cross-lane shuffle
/// sequence algorithmically rather than hand-coding each instruction pair.
///
/// The key fields are:
/// - active_lane_mask: which 16-lane groups receive output (bitmask over 4 groups)
/// - src_vgprs / dst_vgprs: operand register counts per lane
///
/// For MFMA→WMMA translation, the difference in active_lane_mask between
/// source and target tells us which lanes need data from cross-lane shuffles,
/// and the src_vgprs difference tells us whether source expansion is needed.
/// @brief Which lane formula to use for mapping rows to lanes.
enum class LayoutKind : uint8_t {
  MFMA, ///< lane = 16*(row/4) + col (CDNA sequential groups)
  WMMA, ///< lane = 32*((row/4)%2) + 16*(row/8) + col (RDNA interleaved)
};

struct LaneLayout {
  LayoutKind kind = LayoutKind::MFMA;
  uint8_t m = 0;
  uint8_t n = 0;
  uint8_t k = 0;
  uint8_t wave_size = 0;
  uint8_t src_vgprs = 0;
  uint8_t dst_vgprs = 0;
  uint8_t active_lane_groups = 0;
  uint8_t num_passes = 1;
  uint16_t shuffle_pattern = 0;
};

/// @brief Result of computing the lane permutation between two layouts.
struct LanePermutation {
  uint32_t xor_byte_mask = 0; ///< Byte-address XOR for the affected lane range.
  uint8_t range_start = 0;    ///< First lane needing permutation (inclusive).
  uint8_t range_end = 0;      ///< One past last lane needing permutation (exclusive).
};

/// @brief Compute the row-to-lane base for a given layout and row index.
[[nodiscard]] constexpr uint8_t lane_for_row(const LaneLayout &layout, uint8_t row) {
  if (layout.kind == LayoutKind::WMMA)
    return static_cast<uint8_t>(32 * ((row / 4) % 2) + 16 * (row / 8));
  return static_cast<uint8_t>(16 * (row / 4)); // MFMA
}

/// @brief Compute the lane permutation needed to convert guest layout to host layout.
[[nodiscard]] LanePermutation compute_lane_permutation(const LaneLayout &guest,
                                                       const LaneLayout &host);

/// @brief Predefined lane layout for MFMA v_mfma_f32_16x16x16_f16 on CDNA4.
inline constexpr LaneLayout kMfmaF32_16x16x16_F16_Cdna4 = {
    .kind = LayoutKind::MFMA,
    .m = 16,
    .n = 16,
    .k = 16,
    .wave_size = 64,
    .src_vgprs = 2,
    .dst_vgprs = 4,
    .active_lane_groups = 0xF,
    .num_passes = 1,
    .shuffle_pattern = 0,
};

/// @brief Predefined lane layout for WMMA v_wmma_f32_16x16x16_f16 on RDNA4.
///
/// WMMA writes all 64 lanes but swaps groups 1 and 2 vs MFMA:
///   MFMA: rows 0-3 @lanes 0-15, 4-7 @16-31, 8-11 @32-47, 12-15 @48-63
///   WMMA: rows 0-3 @lanes 0-15, 8-11 @16-31, 4-7 @32-47, 12-15 @48-63
/// A single ds_bpermute with XOR-48 byte addresses at lanes 16-47 corrects this.
inline constexpr LaneLayout kWmmaF32_16x16x16_F16_Rdna4 = {
    .kind = LayoutKind::WMMA,
    .m = 16,
    .n = 16,
    .k = 16,
    .wave_size = 64,
    .src_vgprs = 2,
    .dst_vgprs = 4,
    .active_lane_groups = 0xF, // all 4 groups written
    .num_passes = 1,           // single WMMA + ds_bpermute lane remap
    .shuffle_pattern = 0,      // ds_bpermute handles permutation, not ds_swizzle
};

} // namespace rocjitsu
