// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "ordinary_wrong_address_oracle.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::test {

// Independently reconstruct full eight-byte GLOBAL guest words. No production
// displacement helper, acquire record, or candidate-derived mask is consulted.
inline bool legacy_ordinary_output_matches(const FaultMutationContext &source, uint64_t offset,
                                           uint32_t delta, std::span<const uint8_t> candidate,
                                           std::string_view expected_form) {
  if (source.source().text_sections().size() != 1 || delta == 0 || delta % 4 != 0 ||
      delta > 0x7fffff)
    return false;
  const Section &original_text = *source.source().text_sections().front();
  if (offset > original_text.size() || 8 > original_text.size() - offset)
    return false;
  const uint8_t *original = reinterpret_cast<const uint8_t *>(original_text.data());
  std::array<uint32_t, 2> original_words = {};
  std::memcpy(original_words.data(), original + offset, sizeof(original_words));
  const int64_t encoded = original_words[0] & 0x1fffu;
  const int64_t original_offset = encoded >= 0x1000 ? encoded - 0x2000 : encoded;
  const int64_t next = original_offset + delta;
  if (next < -4096 || next > 4095 || next % 4 != 0)
    return false;
  const std::array<uint32_t, 2> expected_words = {
      (original_words[0] & ~0x1fffu) | (static_cast<uint32_t>(next) & 0x1fffu), original_words[1]};
  std::vector<uint8_t> replacement(sizeof(expected_words));
  std::memcpy(replacement.data(), expected_words.data(), replacement.size());
  const std::array<ExpectedReplacement, 1> edits = {ExpectedReplacement{
      .offset = offset,
      .expected = std::vector<uint8_t>(original + offset, original + offset + 8),
      .replacement = replacement}};
  const IsaTargetDescriptor *target =
      default_isa_target_registry().find(source.source().target_id());
  if (!target)
    return false;
  const std::optional<std::vector<uint8_t>> expected =
      expected_replacement_image(source.source(), edits, target->family);
  if (!expected || !std::ranges::equal(*expected, candidate))
    return false;
  const FaultMutationContext mutant(candidate);
  if (!mutant.source().is_valid() || mutant.source().text_sections().size() != 1 ||
      mutant.source().target_id() != source.source().target_id() ||
      !descriptors_preserved(source, mutant))
    return false;
  const Section &text = *mutant.source().text_sections().front();
  const std::optional<uint64_t> guest =
      ordinary_emitted_guest(text, offset, source.source().target_id());
  if (!guest || *guest > text.size() || text.size() - *guest < 12)
    return false;
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), source.source().target_id());
  if (!decoder || !branch_reaches(*decoder, text, offset, *guest) ||
      !branch_reaches(*decoder, text, *guest + 8, offset + 8))
    return false;
  std::array<uint32_t, 2> actual_words = {};
  std::memcpy(actual_words.data(), text.data() + *guest, sizeof(actual_words));
  const DecodeResult before = decoder->decode_window(original_words, offset);
  const DecodeResult after = decoder->decode_window(actual_words, *guest);
  if (before.failed() || after.failed() || before.value()->size() != 8 ||
      after.value()->size() != 8 || before.value()->mnemonic() != expected_form ||
      after.value()->mnemonic() != expected_form ||
      before.value()->opcode() != after.value()->opcode() ||
      before.value()->num_src_operands() != after.value()->num_src_operands() ||
      before.value()->num_dst_operands() != after.value()->num_dst_operands() ||
      actual_words != expected_words)
    return false;
  for (int index = 0; index < before.value()->num_src_operands(); ++index) {
    const Operand *old_operand = before.value()->src_operand(index);
    const Operand *new_operand = after.value()->src_operand(index);
    if (!old_operand || !new_operand ||
        old_operand->encoding_value() != new_operand->encoding_value() ||
        old_operand->decoded_vgpr() != new_operand->decoded_vgpr())
      return false;
  }
  for (int index = 0; index < before.value()->num_dst_operands(); ++index) {
    const Operand *old_operand = before.value()->dst_operand(index);
    const Operand *new_operand = after.value()->dst_operand(index);
    if (!old_operand || !new_operand ||
        old_operand->encoding_value() != new_operand->encoding_value() ||
        old_operand->decoded_vgpr() != new_operand->decoded_vgpr())
      return false;
  }
  return true;
}

} // namespace rocjitsu::test
