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

#include <compare>
#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu {

class Instruction;
class LivenessAnalysis;

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
/// @param liveness      Kernel-scoped live-before data for safe scratch register allocation.
/// @param guest_layout  Source matrix lane layout (nullptr if not a matrix op).
/// @param host_layout   Target matrix lane layout (nullptr if not a matrix op).
/// @returns Replacement instruction words, or empty vector if unhandled.
using ExpandFn = std::vector<uint32_t> (*)(const Instruction &inst, uint32_t arch, uint64_t offset,
                                           const LivenessAnalysis &liveness,
                                           const LaneLayout *guest_layout,
                                           const LaneLayout *host_layout);

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

  constexpr auto operator<=>(const TranslationRule &rhs) const {
    if (auto cmp = src_encoding_id <=> rhs.src_encoding_id; cmp != 0)
      return cmp;
    return src_opcode <=> rhs.src_opcode;
  }
  constexpr bool operator==(const TranslationRule &rhs) const {
    return src_encoding_id == rhs.src_encoding_id && src_opcode == rhs.src_opcode;
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
