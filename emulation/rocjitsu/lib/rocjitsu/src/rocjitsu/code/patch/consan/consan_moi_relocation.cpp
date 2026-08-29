// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <utility>

namespace rocjitsu::consan_moi_detail {

std::optional<std::vector<uint32_t>>
decode_relocatable_entry_instruction(std::span<const uint8_t> text, uint64_t offset,
                                     rj_code_arch_t arch, std::vector<std::string> &errors,
                                     Decoder *reusable_decoder) {
  std::unique_ptr<Decoder> owned_decoder;
  Decoder *decoder = reusable_decoder;
  if (decoder == nullptr) {
    owned_decoder = Decoder::create(arch);
    decoder = owned_decoder.get();
  }
  if (!decoder) {
    errors.emplace_back("ConSan MOI could not create an entry-instruction decoder");
    return std::nullopt;
  }
  const auto decode_one = [&](uint64_t instruction_offset)
      -> std::optional<std::pair<std::vector<uint32_t>, std::string>> {
    if (instruction_offset > text.size() || sizeof(uint32_t) > text.size() - instruction_offset) {
      errors.emplace_back("ConSan MOI kernel entry instruction exceeds .text");
      return std::nullopt;
    }
    std::array<uint32_t, 4> decode_words{};
    const size_t available =
        std::min<size_t>(sizeof(decode_words), text.size() - instruction_offset);
    std::memcpy(decode_words.data(), text.data() + instruction_offset, available);
    std::unique_ptr<Instruction> instruction = decode_bounded_instruction(
        *decoder, std::span<const uint32_t>(decode_words).first(available / sizeof(uint32_t)),
        instruction_offset);
    if (!instruction) {
      errors.emplace_back("ConSan MOI could not decode the kernel entry instruction");
      return std::nullopt;
    }
    if (instruction->size() <= 0 || instruction->size() % sizeof(uint32_t) != 0 ||
        static_cast<size_t>(instruction->size()) > available) {
      errors.emplace_back("ConSan MOI kernel entry instruction has invalid decoded bounds");
      return std::nullopt;
    }
    const std::string mnemonic(instruction->mnemonic());
    if (mnemonic.starts_with("s_branch") || mnemonic.starts_with("s_cbranch") ||
        mnemonic.starts_with("s_call") || mnemonic.starts_with("s_getpc") ||
        mnemonic.starts_with("s_setpc") || mnemonic.starts_with("s_swappc")) {
      errors.emplace_back(
          "ConSan MOI dynamic-stack entry trampoline cannot relocate PC-relative control flow");
      return std::nullopt;
    }
    std::vector<uint32_t> words(static_cast<size_t>(instruction->size()) / sizeof(uint32_t));
    std::memcpy(words.data(), text.data() + instruction_offset,
                static_cast<size_t>(instruction->size()));
    return std::pair{std::move(words), mnemonic};
  };

  auto first = decode_one(offset);
  if (!first)
    return std::nullopt;
  std::vector<uint32_t> words = std::move(first->first);
  if (first->second != "s_clause")
    return words;

  const uint32_t following_instruction_count = (words.front() & 0xffffu) + 1u;
  uint64_t cursor = offset + words.size() * sizeof(uint32_t);
  for (uint32_t index = 0; index < following_instruction_count; ++index) {
    auto following = decode_one(cursor);
    if (!following)
      return std::nullopt;
    if (following->second == "s_clause") {
      errors.emplace_back("ConSan MOI cannot relocate a nested s_clause entry run");
      return std::nullopt;
    }
    cursor += following->first.size() * sizeof(uint32_t);
    words.insert(words.end(), following->first.begin(), following->first.end());
  }
  return words;
}

uint32_t count_nop_padding(std::span<const uint8_t> bytes, uint64_t offset, rj_code_arch_t arch) {
  if (offset > bytes.size())
    return 0;

  const uint32_t nop = build_s_nop(0, arch);
  uint32_t count = 0;
  while (bytes.size() - offset >= sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    if (word != nop)
      break;
    ++count;
    offset += sizeof(uint32_t);
  }
  return count;
}

void append_word_bytes(std::vector<uint8_t> &bytes, uint32_t word) {
  const auto *begin = reinterpret_cast<const uint8_t *>(&word);
  bytes.insert(bytes.end(), begin, begin + sizeof(word));
}

void append_words_bytes(std::vector<uint8_t> &bytes, std::span<const uint32_t> words) {
  for (uint32_t word : words)
    append_word_bytes(bytes, word);
}

std::optional<DbiPatchPlacement>
plan_prebuilt_appended_cave(DbiPatchPlacementPlanner &planner, uint64_t anchor_offset,
                            uint32_t original_size, std::span<const uint32_t> cave_words,
                            std::vector<std::string> &errors, std::string_view probe_name) {
  if (cave_words.empty()) {
    errors.emplace_back(std::string(probe_name) + " produced an empty cave");
    return std::nullopt;
  }
  DbiPatchPlacementRequest request;
  request.anchor_offset = anchor_offset;
  request.original_size = original_size;
  request.body_size = static_cast<uint64_t>(cave_words.size() - 1u) * sizeof(uint32_t);
  request.inline_capacity = 0;
  std::string placement_error;
  const auto placement = planner.plan(request, &placement_error);
  if (!placement || placement->kind != DbiPatchPlacementKind::AppendedCave ||
      placement->return_branch_offset !=
          placement->body_offset + (cave_words.size() - 1u) * sizeof(uint32_t)) {
    errors.emplace_back(std::string(probe_name) + " placement failed: " + placement_error);
    return std::nullopt;
  }
  return placement;
}

} // namespace rocjitsu::consan_moi_detail
