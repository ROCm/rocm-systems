// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sdwa_disassembly.h
/// @brief SDWA text formatting without instruction execution dependencies.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SDWA_DISASSEMBLY_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SDWA_DISASSEMBLY_H_

#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/operand.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace rocjitsu::amdgpu::sdwa {

/// @brief Append an SDWA source, including modifiers encoded in the extension word.
/// The semantic source format selects which modifier family is meaningful:
/// integer-like sources use sign extension, while floating-point sources use
/// negate and absolute-value modifiers.
inline void append_source(std::string &out, const Operand &source, SourceModifierFormat format,
                          bool sign_extend, bool negate, bool absolute) {
  if (format == SourceModifierFormat::NONE && sign_extend)
    out += "sext(";
  if (format != SourceModifierFormat::NONE && negate)
    out += '-';
  if (format != SourceModifierFormat::NONE && absolute)
    out += '|';
  out += source.name();
  if (format != SourceModifierFormat::NONE && absolute)
    out += '|';
  if (format == SourceModifierFormat::NONE && sign_extend)
    out += ')';
}

inline std::string selection_name(uint32_t selection) {
  constexpr std::array<std::string_view, 7> names = {"BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3",
                                                     "WORD_0", "WORD_1", "DWORD"};
  if (selection < names.size())
    return std::string(names[selection]);
  return "invalid(" + std::to_string(selection) + ")";
}

inline std::string destination_unused_name(uint32_t unused) {
  constexpr std::array<std::string_view, 3> names = {"UNUSED_PAD", "UNUSED_SEXT",
                                                     "UNUSED_PRESERVE"};
  if (unused < names.size())
    return std::string(names[unused]);
  return "invalid(" + std::to_string(unused) + ")";
}

inline void append_source_attributes(std::string &out, uint32_t src0_selection, const Operand *src1,
                                     uint32_t src1_selection) {
  out += " src0_sel:";
  out += selection_name(src0_selection);
  if (src1) {
    out += " src1_sel:";
    out += selection_name(src1_selection);
  }
}

inline void append_destination_attributes(std::string &out, bool clamp, uint32_t omod,
                                          uint32_t destination_selection,
                                          uint32_t destination_unused, uint32_t src0_selection,
                                          const Operand *src1, uint32_t src1_selection) {
  if (clamp)
    out += " clamp";
  switch (omod) {
  case 1:
    out += " mul:2";
    break;
  case 2:
    out += " mul:4";
    break;
  case 3:
    out += " div:2";
    break;
  default:
    break;
  }
  out += " dst_sel:";
  out += selection_name(destination_selection);
  out += " dst_unused:";
  out += destination_unused_name(destination_unused);
  append_source_attributes(out, src0_selection, src1, src1_selection);
}

} // namespace rocjitsu::amdgpu::sdwa

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SDWA_DISASSEMBLY_H_
