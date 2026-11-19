// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/text_padding.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/relocation_function_table.h"

#include <algorithm>
#include <iterator>
#include <span>

namespace rocjitsu {

TextAlignmentGaps::TextAlignmentGaps(const CodeObject &object, rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_RDNA4 || object.text_sections().size() != 1)
    return;
  const auto *elf = dynamic_cast<const AmdGpuCodeObject *>(&object);
  if (!elf)
    return;
  const auto ranges = discover_text_function_symbol_ranges(*elf);
  if (ranges.empty())
    return;
  const auto *text = object.text_sections().front();
  const std::span bytes(text->data(), text->size());
  uint64_t end = ranges.front().start_offset + ranges.front().size;
  for (const auto &range : std::span(ranges).subspan(1)) {
    if (end < range.start_offset &&
        std::ranges::all_of(bytes.subspan(end, range.start_offset - end),
                            [](auto byte) { return byte == 0; }))
      gaps_.push_back({end, range.start_offset});
    end = std::max(end, range.start_offset + range.size);
  }
}

bool TextAlignmentGaps::contains(uint64_t byte_offset) const {
  const auto next = std::ranges::upper_bound(gaps_, byte_offset, {}, &Gap::begin);
  return next != gaps_.begin() && byte_offset < std::prev(next)->end;
}

} // namespace rocjitsu
