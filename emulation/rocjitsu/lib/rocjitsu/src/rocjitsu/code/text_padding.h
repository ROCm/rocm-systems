// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/isa_family.h"

#include <cstdint>
#include <vector>

namespace rocjitsu {

/// @brief Whether a word at a prospective instruction start is linker padding.
/// @details gfx1250 zero-filled alignment is not an instruction. Never apply
/// this predicate to literal/interior words of an already decoded instruction.
[[nodiscard]] inline bool is_text_padding_word(rocjitsu::IsaFamily arch, uint32_t word) {
  return arch == rocjitsu::IsaFamily::Cdna5 && word == 0;
}

class CodeObject;

/// Zero-filled intervals strictly between sized RDNA4 function symbols. An in-function
/// zero word is v_illegal and must still be presented to the instruction decoder.
class TextAlignmentGaps {
public:
  TextAlignmentGaps(const CodeObject &object, rocjitsu::IsaFamily arch);
  [[nodiscard]] bool contains(uint64_t byte_offset) const;

private:
  struct Gap {
    uint64_t begin;
    uint64_t end;
  };
  std::vector<Gap> gaps_;
};

} // namespace rocjitsu
