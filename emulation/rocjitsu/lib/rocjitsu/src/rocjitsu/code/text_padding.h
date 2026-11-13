// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"

#include <cstdint>

namespace rocjitsu {

/// @brief Whether a word at a prospective instruction start is linker padding.
/// @details gfx1250 zero-filled alignment is not an instruction. Never apply
/// this predicate to literal/interior words of an already decoded instruction.
[[nodiscard]] inline bool is_text_padding_word(rj_code_arch_t arch, uint32_t word) {
  return arch == ROCJITSU_CODE_ARCH_CDNA5 && word == 0;
}

} // namespace rocjitsu
