// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/gfx1250_flat_scratch_base.cpp
/// @brief gfx1250 B0-to-A0 lowering for FLAT_SCRATCH_BASE source operands.

#include "rocjitsu/code/dbt/semantic/gfx1250_flat_scratch_base.h"

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/builders.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <vector>

namespace rocjitsu {
namespace {

/// @brief Scalar selectors that name the flat-scratch base.
/// @details Both deliver the whole 64-bit base in a 64-bit source position; the
/// low selector is the spelling the A0 profile reads. See gfx1250
/// operand_types.h (OPR_SSRC_SRC_FLAT_SCRATCH_BASE_LO / _HI).
constexpr int kFlatScratchBaseLo = 230;
constexpr int kFlatScratchBaseHi = 231;
constexpr int k64BitOperand = 64;

/// @brief Highest scalar selector usable as an ordinary SGPR source operand.
/// @details Selectors above this range name architectural values rather than
/// the general-purpose file, so a borrowed pair must stay below it.
constexpr uint16_t kMaxOrdinarySgpr = 105;

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
    return EncodingSourceFields{
        .fields = {{{1, 0, 9}, {1, 9, 9}, {1, 18, 9}}}, .count = 3, .vector = true};
  // VOP3P shares VOP3's second-word source layout but not its leading bits, and
  // its packed sources are 64 bits wide, so it reaches the selector the same way.
  if ((word0 >> 24) == 204) // VOP3P
    return EncodingSourceFields{
        .fields = {{{1, 0, 9}, {1, 9, 9}, {1, 18, 9}}}, .count = 3, .vector = true};
  if ((word0 >> 25) == 63) // VOP1
    return EncodingSourceFields{.fields = {{{0, 0, 9}}}, .count = 1, .vector = true};
  if ((word0 >> 25) == 62) // VOPC
    return EncodingSourceFields{.fields = {{{0, 0, 9}, {0, 9, 8}}}, .count = 2, .vector = true};
  // VOP2 is the remaining format whose leading bit is clear; the compact
  // formats above are tested first, so reaching here identifies it.
  if ((word0 >> 31) == 0) // VOP2
    return EncodingSourceFields{.fields = {{{0, 0, 9}, {0, 9, 8}}}, .count = 2, .vector = true};
  return std::nullopt;
}

/// @brief Replace one bit field in a 32-bit instruction word.
void set_word_field(uint32_t &word, uint32_t value, uint32_t shift, uint32_t width) {
  const uint32_t mask = ((uint32_t{1} << width) - 1) << shift;
  word = (word & ~mask) | ((value << shift) & mask);
}

/// @brief True when source operand @p index names the base in a 64-bit position.
///
/// @details The value alone is not decisive. A vector field that indexes the
/// register file directly can hold the selector's number while meaning an
/// ordinary register, so the field has to be one that reaches the scalar
/// encodings before the value is compared.
///
/// Operand::is_vgpr() cannot make that distinction: it is a construction-time
/// capability of the operand *type*, and the ordinary vector source type is
/// also the one that accepts scalar values, so it reports true for both.
[[nodiscard]] bool is_flat_scratch_base_64bit_source(const Instruction &inst, int index,
                                                     const EncodingSourceFields &layout) {
  const Operand *op = inst.src_operand(index);
  if (op == nullptr || op->size_bits() != k64BitOperand)
    return false;
  const SourceField &field = layout.fields[static_cast<size_t>(index)];
  if (layout.vector && field.width != kSelectorCapableVectorFieldWidth)
    return false;
  const int value = op->encoding_value();
  return value == kFlatScratchBaseLo || value == kFlatScratchBaseHi;
}

/// @brief Copy the instruction's words from the authoritative source image.
///
/// @details raw_encoding() addresses only the base-format subobject, so any
/// literal or modifier word that follows it must come from the text image.
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
    const int value = op->encoding_value();
    if (value == kFlatScratchBaseLo || value == kFlatScratchBaseHi)
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
    const int value = op->encoding_value();
    if (value == kFlatScratchBaseLo || value == kFlatScratchBaseHi)
      return true;
  }
  return false;
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
  const int sources = std::min(inst.num_src_operands(), static_cast<int>(layout->count));
  for (int i = 0; i < sources; ++i) {
    if (is_flat_scratch_base_64bit_source(inst, i, *layout))
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
  for (int i = 0; i < rewritable; ++i) {
    if (!is_flat_scratch_base_64bit_source(inst, i, *layout))
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

    // A vector read takes the base from an ordinary SGPR pair. One scalar move
    // serves every affected source position in this instruction.
    if (!borrowed_pair) {
      borrowed_pair = liveness.find_free_sgpr_pair(&inst);
      if (!borrowed_pair || *borrowed_pair + 1 > kMaxOrdinarySgpr) {
        return ExpandResult::failed(
            "gfx1250 flat-scratch-base rewrite could not allocate a dead SGPR pair",
            {"Free an aligned SGPR pair around this instruction."});
      }
      // No descriptor growth is involved. This target's scalar file is fixed
      // rather than sized by the descriptor, and the translator does not write
      // the legacy granulated field for it, so raising a requirement here would
      // change nothing. The bound that matters is that the pair stays inside the
      // architecturally addressable range, which the check above enforces.
      const auto move = gfx1250::build_sop1(gfx1250::kSMovB64Sop1,
                                            {.ssrc0 = static_cast<uint8_t>(kFlatScratchBaseLo),
                                             .sdst = static_cast<uint8_t>(*borrowed_pair)});
      // Appended one word at a time rather than as an iterator range: the
      // range form of insert() reduces to a bulk copy whose bounds GCC cannot
      // relate back to a single-element std::array, and it reports the
      // one-past-the-end pointer as an out-of-bounds access.
      for (const uint32_t word : move)
        prologue.push_back(word);
    }
    set_word_field((*words)[field.word], *borrowed_pair, field.shift, field.width);
  }

  prologue.insert(prologue.end(), words->begin(), words->end());
  return ExpandResult::success(std::move(prologue));
}

} // namespace rocjitsu
