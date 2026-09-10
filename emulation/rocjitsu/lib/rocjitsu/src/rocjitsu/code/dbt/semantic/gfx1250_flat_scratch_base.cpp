// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/gfx1250_flat_scratch_base.cpp
/// @brief gfx1250 B0-to-A0 lowering for FLAT_SCRATCH_BASE source operands.

#include "rocjitsu/code/dbt/semantic/gfx1250_flat_scratch_base.h"

#include "rocjitsu/code/analysis/liveness.h"
#include "rocjitsu/code/dbt/hazard_tracker.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu {
namespace {

/// @brief Scalar selectors that name the flat-scratch base.
/// @details Both deliver the whole 64-bit base in a 64-bit source position; the
/// low selector is the spelling the A0 profile reads. See gfx1250
/// operand_types.h (OPR_SSRC_SRC_FLAT_SCRATCH_BASE_LO / _HI).
constexpr uint32_t kFlatScratchBaseLo = 230;
constexpr uint32_t kFlatScratchBaseHi = 231;
constexpr int k64BitOperand = 64;

/// @brief True when an encoding field value names the flat-scratch base.
[[nodiscard]] bool names_flat_scratch_base(uint32_t value) {
  return value == kFlatScratchBaseLo || value == kFlatScratchBaseHi;
}

/// @brief Highest scalar selector usable as an ordinary SGPR source operand.
/// @details Selectors above this range name architectural values rather than
/// the general-purpose file, so a borrowed pair must stay below it.
constexpr uint16_t kMaxOrdinarySgpr = 105;

/// @brief Complete any producer of the aliased s102:s103 state before reading
/// the architectural flat-scratch selectors.
[[nodiscard]] uint32_t flat_scratch_selector_wait() {
  return cdna5::build_sopp(cdna5::kSWaitAluSopp, {.simm16 = 0})[0];
}

/// @brief Width of a vector source field wide enough to name a scalar value.
/// @details The compact vector formats give their second source an eight-bit
/// field that indexes the vector file directly, so it can hold the selector's
/// numeric value without naming it. Only the nine-bit form reaches the scalar
/// encodings at all.
constexpr uint8_t kSelectorCapableVectorFieldWidth = 9;

/// @brief One source-operand field within a base instruction encoding.
struct SourceField {
  uint8_t word;  ///< Index of the containing 32-bit word.
  uint8_t shift; ///< Bit position of the field within that word.
  uint8_t width; ///< Field width in bits.
};

/// @brief Layout of the source-operand fields for one base encoding.
struct EncodingSourceFields {
  std::array<SourceField, 3> fields{};
  uint8_t count = 0;
  bool vector = false; ///< True when sources are read by the vector ALU.
  /// True when this EXEC-masked encoding may stage through an overwritten destination.
  bool destination_stageable = false;
};

/// @brief Identify the base encoding from its self-describing high bits.
///
/// @details Encoding IDs carry high opcode bits and are not contiguous per
/// format, so they cannot be range-tested reliably. The leading bits of the
/// first word identify the format directly and are the same constants the
/// gfx1250 builders emit. Source-operand fields are listed in the order the
/// decoder reports them, so field N corresponds to source operand N.
///
/// Formats whose sources cannot be a 64-bit scalar special value are omitted;
/// the caller treats an unrecognized format as a rewrite it cannot encode
/// rather than copying the instruction unchanged.
[[nodiscard]] std::optional<EncodingSourceFields> source_fields(uint32_t word0) {
  if ((word0 >> 23) == 381) // SOP1
    return EncodingSourceFields{.fields = {{{0, 0, 8}}}, .count = 1, .vector = false};
  if ((word0 >> 23) == 382) // SOPC
    return EncodingSourceFields{.fields = {{{0, 0, 8}, {0, 8, 8}}}, .count = 2, .vector = false};
  if ((word0 >> 30) == 2) // SOP2
    return EncodingSourceFields{.fields = {{{0, 0, 8}, {0, 8, 8}}}, .count = 2, .vector = false};
  if ((word0 >> 26) == 53) // VOP3
    return EncodingSourceFields{.fields = {{{1, 0, 9}, {1, 9, 9}, {1, 18, 9}}},
                                .count = 3,
                                .vector = true,
                                .destination_stageable = true};
  // VOP3P shares VOP3's second-word source layout but not its leading bits, and
  // its packed sources are 64 bits wide, so it reaches the selector the same way.
  if ((word0 >> 24) == 204) // VOP3P
    return EncodingSourceFields{.fields = {{{1, 0, 9}, {1, 9, 9}, {1, 18, 9}}},
                                .count = 3,
                                .vector = true,
                                .destination_stageable = true};
  if ((word0 >> 25) == 63) // VOP1
    return EncodingSourceFields{
        .fields = {{{0, 0, 9}}}, .count = 1, .vector = true, .destination_stageable = true};
  if ((word0 >> 25) == 62) // VOPC
    return EncodingSourceFields{.fields = {{{0, 0, 9}, {0, 9, 8}}}, .count = 2, .vector = true};
  // VOP2 is the remaining format whose leading bit is clear; the compact
  // formats above are tested first, so reaching here identifies it.
  if ((word0 >> 31) == 0) // VOP2
    return EncodingSourceFields{.fields = {{{0, 0, 9}, {0, 9, 8}}},
                                .count = 2,
                                .vector = true,
                                .destination_stageable = true};
  return std::nullopt;
}

/// @brief Replace one bit field in a 32-bit instruction word.
void set_word_field(uint32_t &word, uint32_t value, uint32_t shift, uint32_t width) {
  const uint32_t mask = ((uint32_t{1} << width) - 1) << shift;
  word = (word & ~mask) | ((value << shift) & mask);
}

/// @brief Read one source-operand field out of the instruction words.
/// @returns The field's value, or nullopt when it lies outside @p words.
[[nodiscard]] std::optional<uint32_t> read_word_field(std::span<const uint32_t> words,
                                                      const SourceField &field) {
  if (field.word >= words.size())
    return std::nullopt;
  const uint32_t mask = (uint32_t{1} << field.width) - 1;
  return (words[field.word] >> field.shift) & mask;
}

/// @brief Words of the base encoding that a layout's source fields reach into.
/// @details Derived from the layout rather than tabulated separately so the two
/// cannot drift apart as encodings are added.
[[nodiscard]] size_t modelled_word_count(const EncodingSourceFields &layout) {
  size_t words = 0;
  for (uint8_t i = 0; i < layout.count; ++i)
    words = std::max<size_t>(words, static_cast<size_t>(layout.fields[i].word) + 1);
  return words;
}

/// @brief True when source operand @p index names the base in a 64-bit position.
///
/// @details The selector is read from the encoding field, never from the
/// decoded operand. A literal source reports the literal's *value* through
/// Operand::encoding_value(), so an ordinary constant of 230 or 231 is
/// indistinguishable there from the selector; the field itself still reads 255
/// (literal) or 254 (literal64) and names no register at all. Trusting the
/// operand rewrites the very field that marks the literal as present, which
/// leaves its dword behind as a standalone illegal instruction.
///
/// The field must also be one that can reach the scalar encodings before its
/// value means anything. A vector field that indexes the register file directly
/// can hold the selector's number while naming an ordinary register.
///
/// Operand::is_vgpr() cannot make that distinction: it is a construction-time
/// capability of the operand *type*, and the ordinary vector source type is
/// also the one that accepts scalar values, so it reports true for both.
[[nodiscard]] bool is_flat_scratch_base_64bit_source(const Instruction &inst, int index,
                                                     const EncodingSourceFields &layout,
                                                     std::span<const uint32_t> words) {
  const Operand *op = inst.src_operand(index);
  if (op == nullptr || op->size_bits() != k64BitOperand)
    return false;
  const SourceField &field = layout.fields[static_cast<size_t>(index)];
  if (layout.vector && field.width != kSelectorCapableVectorFieldWidth)
    return false;
  const std::optional<uint32_t> encoded = read_word_field(words, field);
  return encoded.has_value() && names_flat_scratch_base(*encoded);
}

/// @brief True when @p op is a decoded literal rather than a named operand.
///
/// @details Used only by the conservative scans below, which have no field
/// layout to read and so must fall back to the operand. It recognizes the
/// 64-bit literal form, the only one the shared Operand interface exposes; a
/// 32-bit literal widened into a 64-bit source position is indistinguishable
/// from a selector by value alone. That residual gap is why the modelled
/// encodings above read the encoding field instead of asking here, and it can
/// only cause a refusal, never a miscompile.
///
/// TODO: close the gap with a generic immediate predicate on Operand, set from
/// the per-arch is_immediate_type() the way is_vgpr_ already is. That is a
/// change to the amdisa codegen templates and every generated arch.
[[nodiscard]] bool is_decoded_literal(const Operand &op) {
  return op.literal64_value().has_value();
}

/// @brief Copy the instruction's words from the authoritative source image.
///
/// @details The rewrite operates on the exact source bytes, including any
/// literal or modifier words, so copy the complete bounded instruction span.
[[nodiscard]] std::optional<std::vector<uint32_t>>
instruction_words(const Instruction &inst, uint64_t offset, std::span<const uint8_t> source_text) {
  const size_t size = static_cast<size_t>(inst.size());
  if (size < sizeof(uint32_t) || size % sizeof(uint32_t) != 0 ||
      offset + size > source_text.size()) {
    return std::nullopt;
  }
  std::vector<uint32_t> words(size / sizeof(uint32_t));
  std::memcpy(words.data(), source_text.data() + offset, size);
  return words;
}

/// @brief True when a decoded source beyond the modelled fields names the base.
///
/// @details A decoder may report more sources than the encoding has source
/// fields: a compact accumulate form repeats its destination as a source, and a
/// literal occupies a position of its own. Neither can carry the selector, so
/// their presence alone is not a reason to refuse the instruction. One that does
/// carry it would have nowhere to be rewritten, so it is reported and refused.
///
/// Operands the ISA types as vector registers are skipped: every position
/// reaching here addresses the vector file directly, so such an operand holds a
/// register number that may coincide with the selector's value.
[[nodiscard]] bool unmodelled_source_names_selector(const Instruction &inst, int first) {
  for (int i = first; i < inst.num_src_operands(); ++i) {
    const Operand *op = inst.src_operand(i);
    if (op == nullptr || op->is_vgpr() || op->size_bits() != k64BitOperand)
      continue;
    if (is_decoded_literal(*op))
      continue;
    const int value = op->encoding_value();
    if (value >= 0 && names_flat_scratch_base(static_cast<uint32_t>(value)))
      return true;
  }
  return false;
}

/// @brief Conservative scan used when the encoding has no modelled layout.
/// @details Without field widths the selector-capable test cannot be applied,
/// so any remaining 64-bit source carrying the value is reported and the
/// lowering then refuses the instruction rather than copying it unexamined.
///
/// Operands the ISA types as vector registers are excluded first. Every
/// encoding reaching here addresses the vector file directly, so such an
/// operand holds a register number rather than a selector; a wide vector
/// address pair can otherwise share the selector's number and be refused.
/// Operand::is_vgpr() is usable for exactly that reason here and not in the
/// modelled formats, whose ordinary source type also accepts scalar values.
[[nodiscard]] bool instruction_names_selector_in_any_64bit_source(const Instruction &inst) {
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const Operand *op = inst.src_operand(i);
    if (op == nullptr || op->is_vgpr() || op->size_bits() != k64BitOperand)
      continue;
    if (is_decoded_literal(*op))
      continue;
    const int value = op->encoding_value();
    if (value >= 0 && names_flat_scratch_base(static_cast<uint32_t>(value)))
      return true;
  }
  return false;
}

[[nodiscard]] bool register_ranges_overlap(uint16_t lhs_base, uint16_t lhs_width, uint16_t rhs_base,
                                           uint16_t rhs_width) {
  return lhs_base < static_cast<uint32_t>(rhs_base) + rhs_width &&
         rhs_base < static_cast<uint32_t>(lhs_base) + lhs_width;
}

/// @brief Reuse an overwritten destination pair to stage the flat-scratch base.
///
/// @details Two 32-bit vector moves can read the low and high special selectors
/// on A0 even though a 64-bit vector source cannot. The gfx1250 A0 source-
/// operand table assigns selectors 230 and 231 to the two 32-bit halves; the
/// corresponding VOP1 encodings also round-trip through llvm-mc for gfx1250.
/// The destination is safe
/// scratch only when it is exactly one VGPR pair, no other source aliases it,
/// and each rewritten source role selects the same physical VGPR bank as the
/// destination role. Tied operands are rejected by the explicit-source scan;
/// gfx1250 partial-write and read-modify-write destinations are rejected by the
/// separate implicit-use scan. Unknown or mismatched banking fails closed.
///
/// The staging moves are EXEC-masked VALU operations. They are safe here because
/// the instruction they precede is also an EXEC-masked VALU operation; an
/// encoding with different lane-mask semantics must not use this fallback.
[[nodiscard]] std::optional<uint16_t> reusable_destination_pair(const Instruction &inst,
                                                                const EncodingSourceFields &layout,
                                                                std::span<const uint32_t> words,
                                                                const LivenessAnalysis &liveness) {
  if (!layout.destination_stageable || inst.num_dst_operands() != 1 ||
      !liveness.global_vgpr_usage_is_complete())
    return std::nullopt;

  const Operand *dst = inst.dst_operand(0);
  if (dst == nullptr)
    return std::nullopt;
  const auto dst_ref = dst->to_register_ref();
  if (!dst_ref || dst_ref->cls != RegClass::VGPR || dst_ref->width != 2 ||
      static_cast<uint32_t>(dst_ref->index) + dst_ref->width > 256u)
    return std::nullopt;
  const auto dst_bank = liveness.vgpr_msb_bank_before(inst, dst->vgpr_msb_role());
  if (!dst_bank)
    return std::nullopt;
  const uint16_t dst_phys =
      static_cast<uint16_t>(dst_ref->index + static_cast<uint16_t>(*dst_bank) * 256u);

  bool found_rewritable_source = false;
  for (int index = 0; index < inst.num_src_operands(); ++index) {
    const Operand *src = inst.src_operand(index);
    if (src == nullptr)
      continue;
    const bool rewritable = index < static_cast<int>(layout.count) &&
                            is_flat_scratch_base_64bit_source(inst, index, layout, words);
    if (rewritable) {
      found_rewritable_source = true;
      const auto src_bank = liveness.vgpr_msb_bank_before(inst, src->vgpr_msb_role());
      if (!src_bank || *src_bank != *dst_bank)
        return std::nullopt;
      continue;
    }

    const auto src_ref = src->to_register_ref();
    if (!src_ref || src_ref->cls != RegClass::VGPR)
      continue;
    if (static_cast<uint32_t>(src_ref->index) + src_ref->width > 256u)
      return std::nullopt;
    const auto src_bank = liveness.vgpr_msb_bank_before(inst, src->vgpr_msb_role());
    if (!src_bank) {
      if (register_ranges_overlap(dst_ref->index, dst_ref->width, src_ref->index, src_ref->width))
        return std::nullopt;
      continue;
    }
    const uint16_t src_phys =
        static_cast<uint16_t>(src_ref->index + static_cast<uint16_t>(*src_bank) * 256u);
    if (register_ranges_overlap(dst_phys, dst_ref->width, src_phys, src_ref->width))
      return std::nullopt;
  }

  RegisterSet implicit_uses;
  inst.implicit_uses(implicit_uses);
  bool implicit_alias = false;
  implicit_uses.for_each([&](RegisterRef ref) {
    if (ref.cls == RegClass::VGPR &&
        register_ranges_overlap(dst_ref->index, dst_ref->width,
                                static_cast<uint16_t>(ref.index & 0xffu), ref.width))
      implicit_alias = true;
  });
  if (implicit_alias || !found_rewritable_source)
    return std::nullopt;
  return dst_ref->index;
}

} // namespace

bool gfx1250_reads_flat_scratch_base_64bit(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr)
    return false;
  const std::optional<EncodingSourceFields> layout = source_fields(raw[0]);
  // An unmodelled encoding is reported so the lowering can refuse it rather
  // than let it reach the copy path unexamined.
  if (!layout)
    return instruction_names_selector_in_any_64bit_source(inst);
  // Only the base encoding is read here. Every modelled source field lies
  // inside it, so the decoded size bounds the span without depending on
  // whether raw_encoding() also addresses trailing literal or modifier words.
  const size_t available =
      inst.size() > 0 ? static_cast<size_t>(inst.size()) / sizeof(uint32_t) : 0;
  const std::span<const uint32_t> words(raw, std::min(modelled_word_count(*layout), available));
  const int sources = std::min(inst.num_src_operands(), static_cast<int>(layout->count));
  for (int i = 0; i < sources; ++i) {
    if (is_flat_scratch_base_64bit_source(inst, i, *layout, words))
      return true;
  }
  return false;
}

bool gfx1250_flat_scratch_base_residual(const Instruction &inst) {
  const uint32_t *raw = inst.raw_encoding();
  if (raw == nullptr)
    return false;
  const std::optional<EncodingSourceFields> layout = source_fields(raw[0]);
  if (!layout)
    return instruction_names_selector_in_any_64bit_source(inst);

  const size_t available =
      inst.size() > 0 ? static_cast<size_t>(inst.size()) / sizeof(uint32_t) : 0;
  const std::span<const uint32_t> words(raw, std::min(modelled_word_count(*layout), available));
  const int sources = std::min(inst.num_src_operands(), static_cast<int>(layout->count));
  for (int source_index = 0; source_index < sources; ++source_index) {
    if (!is_flat_scratch_base_64bit_source(inst, source_index, *layout, words))
      continue;
    if (layout->vector)
      return true;
    const Operand *operand = inst.src_operand(source_index);
    if (operand != nullptr && operand->encoding_value() == kFlatScratchBaseHi)
      return true;
  }
  return false;
}

ExpandResult gfx1250_lower_flat_scratch_base_source(const Instruction &inst, uint64_t offset,
                                                    std::span<const uint8_t> source_text,
                                                    const LivenessAnalysis &liveness,
                                                    TranslationContext &) {
  if (!gfx1250_reads_flat_scratch_base_64bit(inst))
    return ExpandResult::not_handled();

  auto words = instruction_words(inst, offset, source_text);
  if (!words) {
    return ExpandResult::failed(
        "gfx1250 flat-scratch-base rewrite could not read the complete instruction");
  }

  const std::optional<EncodingSourceFields> layout = source_fields((*words)[0]);
  if (!layout) {
    return ExpandResult::failed(
        "gfx1250 flat-scratch-base rewrite does not model this instruction encoding",
        {"Add the source-field layout for this encoding."});
  }
  if (unmodelled_source_names_selector(inst, layout->count)) {
    return ExpandResult::failed(
        "gfx1250 flat-scratch-base rewrite cannot map operands onto encoding fields");
  }

  const int rewritable = std::min(inst.num_src_operands(), static_cast<int>(layout->count));
  std::vector<uint32_t> prologue;
  std::optional<uint16_t> borrowed_pair;
  std::optional<uint16_t> staged_vgpr_pair;
  if (layout->vector) {
    borrowed_pair = liveness.find_free_sgpr_pair(&inst);
    if (!borrowed_pair || *borrowed_pair + 1 > kMaxOrdinarySgpr) {
      borrowed_pair.reset();
      staged_vgpr_pair = reusable_destination_pair(inst, *layout, *words, liveness);
      if (!staged_vgpr_pair) {
        return ExpandResult::failed(
            "gfx1250 flat-scratch-base rewrite could not allocate safe temporary storage",
            {"Free an aligned SGPR pair or a non-aliasing destination VGPR pair around this "
             "instruction."});
      }
    }
    if (borrowed_pair) {
      prologue.push_back(flat_scratch_selector_wait());
      // No descriptor growth is involved. This target's scalar file is fixed
      // rather than sized by the descriptor, and the translator does not write
      // the legacy granulated field for it, so raising a requirement here would
      // change nothing. The bound that matters is that the pair stays inside the
      // architecturally addressable range, which the check above enforces.
      const auto move =
          cdna5::build_sop1(cdna5::kSMovB64Sop1, {.ssrc0 = static_cast<uint8_t>(kFlatScratchBaseLo),
                                                  .sdst = static_cast<uint8_t>(*borrowed_pair)});
      // Appended one word at a time rather than as an iterator range: the
      // range form of insert() reduces to a bulk copy whose bounds GCC cannot
      // relate back to a single-element std::array, and it reports the
      // one-past-the-end pointer as an out-of-bounds access.
      for (const uint32_t word : move)
        prologue.push_back(word);
    }
  }
  for (int i = 0; i < rewritable; ++i) {
    if (!is_flat_scratch_base_64bit_source(inst, i, *layout, *words))
      continue;

    const SourceField &field = layout->fields[static_cast<size_t>(i)];
    if (field.word >= words->size()) {
      return ExpandResult::failed(
          "gfx1250 flat-scratch-base source field lies outside the decoded instruction");
    }

    if (!layout->vector) {
      // A scalar read reaches the whole base through the low selector.
      set_word_field((*words)[field.word], kFlatScratchBaseLo, field.shift, field.width);
      continue;
    }

    // One temporary serves every affected source position in this instruction.
    if (staged_vgpr_pair) {
      set_word_field((*words)[field.word], 256u + *staged_vgpr_pair, field.shift, field.width);
      continue;
    }
    set_word_field((*words)[field.word], *borrowed_pair, field.shift, field.width);
  }

  if (staged_vgpr_pair) {
    std::vector<uint32_t> replacement;
    using Pipeline = HazardTracker::Pipeline;
    HazardTracker hazards;
    const auto low =
        cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = kFlatScratchBaseLo,
                                                .vdst = static_cast<uint8_t>(*staged_vgpr_pair)});
    const auto high = cdna5::build_vop1(
        cdna5::kVMovB32Vop1,
        {.src0 = kFlatScratchBaseHi, .vdst = static_cast<uint8_t>(*staged_vgpr_pair + 1)});
    replacement.push_back(flat_scratch_selector_wait());
    hazards.emit_raw(replacement, low[0]);
    hazards.emit(replacement, high[0], Pipeline::VALU);
    hazards.emit(replacement, (*words)[0], Pipeline::VALU);
    replacement.insert(replacement.end(), words->begin() + 1, words->end());
    return ExpandResult::success(std::move(replacement));
  }

  prologue.insert(prologue.end(), words->begin(), words->end());
  return ExpandResult::success(std::move(prologue));
}

} // namespace rocjitsu
