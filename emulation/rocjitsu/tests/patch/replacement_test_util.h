// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"
#include "util/result.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rocjitsu::test {

struct ExpectedReplacement {
  uint64_t offset;
  std::vector<uint8_t> expected;
  std::vector<uint8_t> replacement;
};

// Compute the displacement independently of the production planner/range helper.
inline std::optional<uint32_t> expected_replacement_branch(uint64_t from, uint64_t to,
                                                           rj_code_arch_t arch) {
  if (from > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - 4 ||
      to > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  const int64_t distance = static_cast<int64_t>(to) - static_cast<int64_t>(from) - 4;
  if (distance % 4 != 0 || distance / 4 < std::numeric_limits<int16_t>::min() ||
      distance / 4 > std::numeric_limits<int16_t>::max())
    return std::nullopt;
  const util::FailureOr<uint32_t> branch = build_s_branch(0, arch);
  if (branch.failed())
    return std::nullopt;
  return branch.value() | static_cast<uint16_t>(distance / 4);
}

// This oracle constructs source splices and replacement caves without consuming
// emitted site ranges, plans or candidate bytes. The shared patcher supplies only
// ELF layout/metadata updates; its writer has separate regression coverage.
inline std::optional<std::vector<uint8_t>>
expected_replacement_image(const AmdGpuCodeObject &source,
                           std::span<const ExpectedReplacement> edits, rj_code_arch_t arch) {
  if (!source.is_valid() || source.text_sections().size() != 1)
    return std::nullopt;
  const Section &text = *source.text_sections().front();
  const auto *data = reinterpret_cast<const uint8_t *>(text.data());
  std::vector<uint8_t> expected_text(data, data + text.size());
  const util::FailureOr<uint32_t> maybe_nop = build_s_nop(0, arch);
  if (maybe_nop.failed())
    return std::nullopt;
  const uint32_t nop = maybe_nop.value();
  std::unique_ptr<Decoder> decoder =
      source.target_id() == ROCJITSU_CODE_TARGET_INVALID
          ? Decoder::create(arch)
          : Decoder::create(default_isa_target_registry(), source.target_id());
  if (!decoder)
    return std::nullopt;
  for (const ExpectedReplacement &edit : edits) {
    if (edit.expected.empty() || edit.expected.size() % 4 != 0 || edit.offset > text.size() ||
        edit.expected.size() > text.size() - edit.offset || edit.replacement.size() % 4 != 0)
      return std::nullopt;
    const uint64_t trampoline_offset = expected_text.size();
    const std::optional<uint32_t> splice =
        expected_replacement_branch(edit.offset, trampoline_offset, arch);
    if (!splice)
      return std::nullopt;
    std::memcpy(expected_text.data() + edit.offset, &*splice, sizeof(*splice));
    for (size_t padding = 4; padding < edit.expected.size(); padding += 4)
      std::memcpy(expected_text.data() + edit.offset + padding, &nop, sizeof(nop));
    expected_text.insert(expected_text.end(), edit.replacement.begin(), edit.replacement.end());
    std::vector<uint32_t> words(edit.replacement.size() / 4);
    if (!words.empty())
      std::memcpy(words.data(), edit.replacement.data(), edit.replacement.size());
    size_t cursor = 0;
    bool terminal = false;
    while (cursor < words.size()) {
      const DecodeResult decoded =
          decoder->decode_window(std::span(words).subspan(cursor), trampoline_offset + cursor * 4);
      if (decoded.failed() || decoded.value()->size() == 0 || decoded.value()->size() % 4 != 0 ||
          static_cast<size_t>(decoded.value()->size()) / 4 > words.size() - cursor)
        return std::nullopt;
      terminal = (decoded.value()->flags() & PROGRAM_TERMINATOR) != 0;
      cursor += static_cast<size_t>(decoded.value()->size()) / 4;
    }
    if (!terminal) {
      const std::optional<uint32_t> continuation = expected_replacement_branch(
          expected_text.size(), edit.offset + edit.expected.size(), arch);
      if (!continuation)
        return std::nullopt;
      const auto *bytes = reinterpret_cast<const uint8_t *>(&*continuation);
      expected_text.insert(expected_text.end(), bytes, bytes + sizeof(*continuation));
    }
  }
  CodeObjectPatcher expected_patcher(source);
  if (!expected_patcher.replace_text(expected_text))
    return std::nullopt;
  return expected_patcher.emit();
}

} // namespace rocjitsu::test
