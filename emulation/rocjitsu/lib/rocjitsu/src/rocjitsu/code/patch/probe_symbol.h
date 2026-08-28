// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file probe_symbol.h
/// @brief Resolve a named probe function symbol in an AMDGPU code object to a
///        body byte range.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace rocjitsu {

class AmdGpuCodeObject;

/// @brief A probe function symbol resolved to a body byte range, expressed in
///        ELF-image file-offset coordinates (the same coordinate space as
///        AmdGpuCodeObject::image_data()).
struct ResolvedProbeSymbol {
  std::string name;              ///< Matched symbol name.
  uint16_t section_index = 0;    ///< Owning section header index (st_shndx).
  uint64_t st_value = 0;         ///< Raw symbol value (vaddr / section-relative).
  uint64_t body_file_offset = 0; ///< File offset of the body in the ELF image.
  uint64_t body_size = 0;        ///< Body length in bytes (st_size).
};

/// @brief Find exactly one defined STT_FUNC symbol named @p symbol_name in
///        @p obj and map it to a body byte range.
///
/// Search order is SHT_SYMTAB first, then SHT_DYNSYM; whichever table contains
/// a name match is the one used (so a symbol present in both tables is not
/// mistaken for a duplicate). "Exactly one" is enforced within that table.
///
/// Fails (returns std::nullopt) when the symbol is:
///   - missing from both tables,
///   - present more than once in the chosen table,
///   - undefined (st_shndx == SHN_UNDEF) or not backed by a normal section,
///   - not STT_FUNC,
///   - backed by a non-executable section (no SHF_EXECINSTR),
///   - zero-sized,
///   - not dword-aligned in value or size,
///   - or describes a byte range outside its section / the ELF image.
///
/// Also fails on a malformed ELF (truncated headers, bad string/symbol table
/// links, etc.).
[[nodiscard]] std::optional<ResolvedProbeSymbol>
resolve_probe_symbol(const AmdGpuCodeObject &obj, std::string_view symbol_name,
                     std::string *error_out = nullptr);

} // namespace rocjitsu
