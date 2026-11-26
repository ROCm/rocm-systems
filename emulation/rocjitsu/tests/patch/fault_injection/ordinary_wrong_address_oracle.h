// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "barrier_oracle.h"
#include "ordinary_fixture.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::test {

inline constexpr std::array<OrdinaryTargetCase, 2> kCdna5OrdinaryAddressCases = {
    kOrdinaryTargetCases[0], kOrdinaryTargetCases[1]};
inline constexpr std::array<OrdinaryTargetCase, 4> kRdna4OrdinaryAddressCases = {
    kOrdinaryTargetCases[2], kOrdinaryTargetCases[3], kOrdinaryTargetCases[4],
    kOrdinaryTargetCases[5]};

struct OrdinaryAddressParityCase {
  std::vector<uint8_t> image;
  const char *kernel;
  uint64_t offset;
  const char *name;
};

inline std::vector<OrdinaryAddressParityCase>
rdna4_ordinary_address_parity_cases(const OrdinaryTargetCase &fixture) {
  std::vector<OrdinaryAddressParityCase> cases = {
      {.image = make_ordinary_elf(fixture, 2, 0, false, true),
       .kernel = "test_kernel",
       .offset = 0,
       .name = "LaterCache"},
      {.image = make_ordinary_poll_elf(fixture, 32),
       .kernel = "test_kernel",
       .offset = 128,
       .name = "MaximumBoundedLoopPrefix"},
  };
  std::vector<uint32_t> alu_wait = ordinary_acquire_words();
  alu_wait[3] = 0xbf88ff9eu;
  cases.push_back({.image = make_ordinary_words_elf(fixture, std::move(alu_wait)),
                   .kernel = "test_kernel",
                   .offset = 0,
                   .name = "OriginalAnyWaitWitness"});
  for (bool shared : {false, true}) {
    const uint64_t offset = shared ? 512 : 0;
    std::vector<uint8_t> image =
        shared ? make_shared_ordinary_elf(fixture) : make_ordinary_elf(fixture);
    if (!zero_ordinary_function_size(image, offset))
      image.clear();
    cases.push_back({.image = std::move(image),
                     .kernel = shared ? "test_kernel0" : "test_kernel",
                     .offset = offset,
                     .name = shared ? "ZeroSizedHelper" : "ZeroSizedKernel"});
  }
  return cases;
}

inline std::array<uint32_t, 3> cdna5_ordinary_address_words(uint32_t scope = 2, int32_t offset = 0,
                                                            bool modifiers = false) {
  const uint8_t flag = static_cast<uint8_t>(modifiers);
  return cdna5::build_vglobal(cdna5::kGlobalLoadB32Vglobal,
                              {.saddr = 4,
                               .nv = flag,
                               .vdst = 7,
                               .scale_offset = flag,
                               .sve = flag,
                               .scope = static_cast<uint8_t>(scope),
                               .th = modifiers ? uint8_t{6} : uint8_t{0},
                               .vsrc = modifiers ? uint8_t{6} : uint8_t{0},
                               .vaddr = 10,
                               .ioffset = static_cast<uint32_t>(offset) & 0xffffffu});
}

inline std::vector<uint8_t> make_cdna5_ordinary_address_elf(const OrdinaryTargetCase &fixture,
                                                            uint32_t scope = 2, int32_t offset = 0,
                                                            bool modifiers = false) {
  const std::array<uint32_t, 3> load = cdna5_ordinary_address_words(scope, offset, modifiers);
  std::vector<uint32_t> words = ordinary_acquire_words(scope, offset, true);
  std::copy(load.begin(), load.end(), words.begin());
  return make_ordinary_words_elf(fixture, std::move(words));
}

inline std::array<uint32_t, 3> modern_ordinary_address_words(const OrdinaryTargetCase &fixture,
                                                             uint32_t scope = 2, int32_t offset = 0,
                                                             bool modifiers = false) {
  if (fixture.arch == IsaFamily::Cdna5)
    return cdna5_ordinary_address_words(scope, offset, modifiers);
  const uint8_t flag = static_cast<uint8_t>(modifiers);
  return rdna4::build_vglobal(rdna4::kGlobalLoadB32Vglobal,
                              {.saddr = 4,
                               .nv = flag,
                               .vdst = 7,
                               .sve = flag,
                               .scope = static_cast<uint8_t>(scope),
                               .th = modifiers ? uint8_t{6} : uint8_t{0},
                               .vsrc = modifiers ? uint8_t{6} : uint8_t{0},
                               .vaddr = 10,
                               .ioffset = static_cast<uint32_t>(offset) & 0xffffffu});
}

inline std::vector<uint8_t> make_modern_ordinary_address_elf(const OrdinaryTargetCase &fixture,
                                                             uint32_t scope = 2, int32_t offset = 0,
                                                             bool modifiers = false) {
  const std::array<uint32_t, 3> load =
      modern_ordinary_address_words(fixture, scope, offset, modifiers);
  std::vector<uint32_t> words = ordinary_acquire_words(scope, offset, true);
  std::copy(load.begin(), load.end(), words.begin());
  return make_ordinary_words_elf(fixture, std::move(words));
}

inline std::optional<uint64_t> ordinary_emitted_guest(const Section &text, uint64_t source_offset,
                                                      rj_code_target_id_t target) {
  if (source_offset > text.size() || text.size() - source_offset < 4 ||
      source_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - 4)
    return std::nullopt;
  uint32_t word = 0;
  std::memcpy(&word, text.data() + source_offset, sizeof(word));
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
  if (!decoder)
    return std::nullopt;
  const DecodeResult decoded = decoder->decode_window(std::span(&word, 1), source_offset);
  if (decoded.failed() || decoded.value()->size() != 4 || decoded.value()->mnemonic() != "s_branch")
    return std::nullopt;
  const std::optional<int64_t> delta = decoded.value()->branch_offset_bytes();
  const int64_t base = static_cast<int64_t>(source_offset + 4);
  if (!delta || (*delta > 0 && base > std::numeric_limits<int64_t>::max() - *delta))
    return std::nullopt;
  const int64_t destination = base + *delta;
  if (destination < 0 || static_cast<uint64_t>(destination) > text.size())
    return std::nullopt;
  return static_cast<uint64_t>(destination);
}

// Reconstruct the allowed load-only delta from pristine bytes, without the
// production setter, acquire record, emitted ranges, or candidate-derived mask.
inline bool ordinary_wrong_address_output_matches(const FaultMutationContext &source,
                                                  uint64_t selected_offset, uint32_t byte_delta,
                                                  std::span<const uint8_t> candidate) {
  if (source.source().text_sections().size() != 1 || byte_delta == 0 || byte_delta % 4 != 0 ||
      byte_delta > 0x7fffff)
    return false;
  const Section &original_text = *source.source().text_sections().front();
  if (selected_offset > original_text.size() || original_text.size() - selected_offset < 12)
    return false;
  const auto *original = reinterpret_cast<const uint8_t *>(original_text.data());
  std::array<uint32_t, 3> original_words = {};
  std::memcpy(original_words.data(), original + selected_offset, sizeof(original_words));
  const uint32_t encoded = original_words[2] >> 8;
  const int64_t old_offset = encoded >= 0x800000u ? static_cast<int64_t>(encoded) - 0x1000000
                                                  : static_cast<int64_t>(encoded);
  const int64_t next = old_offset + byte_delta;
  if (next < -0x800000 || next > 0x7fffff || next % 4 != 0)
    return false;
  std::array<uint32_t, 3> expected_words = original_words;
  expected_words[2] =
      (original_words[2] & 0xffu) | ((static_cast<uint32_t>(next) & 0xffffffu) << 8);
  const std::vector<uint8_t> expected_source(original + selected_offset,
                                             original + selected_offset + 12);
  std::vector<uint8_t> replacement(sizeof(expected_words));
  std::memcpy(replacement.data(), expected_words.data(), replacement.size());
  const std::array<ExpectedReplacement, 1> edits = {ExpectedReplacement{
      .offset = selected_offset, .expected = expected_source, .replacement = replacement}};
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
      ordinary_emitted_guest(text, selected_offset, source.source().target_id());
  if (!guest || *guest > text.size() || text.size() - *guest < 16)
    return false;
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), source.source().target_id());
  if (!decoder || !branch_reaches(*decoder, text, selected_offset, *guest) ||
      !branch_reaches(*decoder, text, *guest + 12, selected_offset + 12))
    return false;
  std::array<uint32_t, 3> guest_words = {};
  std::memcpy(guest_words.data(), text.data() + *guest, sizeof(guest_words));
  const DecodeResult before = decoder->decode_window(original_words, selected_offset);
  const DecodeResult after = decoder->decode_window(guest_words, *guest);
  if (before.failed() || after.failed() || before.value()->size() != 12 ||
      after.value()->size() != 12 || before.value()->mnemonic() != "global_load_b32" ||
      after.value()->mnemonic() != before.value()->mnemonic() ||
      before.value()->opcode() != after.value()->opcode() ||
      before.value()->num_src_operands() != after.value()->num_src_operands() ||
      before.value()->num_dst_operands() != after.value()->num_dst_operands() ||
      guest_words != expected_words)
    return false;
  for (int index = 0; index < before.value()->num_operands(); ++index) {
    const Operand *old_operand =
        index < before.value()->num_src_operands()
            ? before.value()->src_operand(index)
            : before.value()->dst_operand(index - before.value()->num_src_operands());
    const Operand *new_operand =
        index < after.value()->num_src_operands()
            ? after.value()->src_operand(index)
            : after.value()->dst_operand(index - after.value()->num_src_operands());
    if (!old_operand || !new_operand ||
        old_operand->encoding_value() != new_operand->encoding_value() ||
        old_operand->size_bits() != new_operand->size_bits() ||
        old_operand->decoded_vgpr() != new_operand->decoded_vgpr())
      return false;
  }
  return true;
}

} // namespace rocjitsu::test
