// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/code_object_patcher.h"

#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu {

/// Relocate address-bearing DWARF sections after DBT has rearranged `.text`.
///
/// The operation is transactional: on failure, @p image and @p sections are
/// unchanged. Unsupported address-bearing encodings fail closed so a caller
/// never publishes debug information that silently describes different code.
[[nodiscard]] bool relocate_dwarf(std::vector<uint8_t> &image, const Elf64_Ehdr &header,
                                  std::vector<Elf64_Shdr> &sections, size_t text_index,
                                  uint64_t old_text_size, uint64_t new_text_size,
                                  std::span<const TextOffsetRelocation> relocations);

} // namespace rocjitsu
