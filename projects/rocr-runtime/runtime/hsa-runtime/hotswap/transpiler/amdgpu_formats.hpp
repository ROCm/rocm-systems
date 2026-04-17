#ifndef HOTSWAP_TRANSPILER_AMDGPU_FORMATS_HPP
#define HOTSWAP_TRANSPILER_AMDGPU_FORMATS_HPP

#include <cstdint>

// Source tree: lib/Target/AMDGPU/SIDefines.h — target-private but exposed
// through the LLVM build tree via our CMake include path. Provides the
// authoritative `SIInstrFlags` enum and `AMDGPU::OPERAND_INPUT_MODS` operand
// type used by the disassembler's TSFlags / OperandType fields.
#include "SIDefines.h"

namespace transpiler {

// Alias `transpiler::SIInstrFlags` to the LLVM namespace so existing call
// sites (`SIInstrFlags::SOPP`, `SIInstrFlags::FLAT`, etc.) keep compiling.
namespace SIInstrFlags = llvm::SIInstrFlags;

// AMDGPU target-specific operand type for VOP3 source modifiers (abs, neg).
// Defined in llvm::AMDGPU::OperandType from SIDefines.h.
constexpr unsigned OPERAND_INPUT_MODS = llvm::AMDGPU::OPERAND_INPUT_MODS;

enum class FormatKind : uint8_t {
  SOP1,
  SOP2,
  SOPC,
  SOPK,
  SOPP,
  VOP1,
  VOP2,
  VOP3,
  VOPC,
  VOP3P,
  SMEM,
  FLAT,
  MUBUF,
  DS,
  MFMA,
  DPP,
  SDWA,
  VOPD,
  Unknown,
};

// VOPD detection cannot use TSFlags — the VOPD3 bit position varies across
// LLVM versions.  Use classifyFormatWithMnemonic() which checks the mnemonic.
inline FormatKind classifyFormat(uint64_t tsFlags) {
  if (tsFlags & SIInstrFlags::IsMAI)  return FormatKind::MFMA;
  // DPP/SDWA must be checked before VOP1/VOP2 because they have both bits set.
  if (tsFlags & SIInstrFlags::DPP)    return FormatKind::DPP;
  if (tsFlags & SIInstrFlags::SDWA)   return FormatKind::SDWA;
  if (tsFlags & SIInstrFlags::SOPP)   return FormatKind::SOPP;
  if (tsFlags & SIInstrFlags::SOPC)   return FormatKind::SOPC;
  if (tsFlags & SIInstrFlags::SOP1)   return FormatKind::SOP1;
  if (tsFlags & SIInstrFlags::SOP2)   return FormatKind::SOP2;
  if (tsFlags & SIInstrFlags::SOPK)   return FormatKind::SOPK;
  if (tsFlags & SIInstrFlags::VOPC)   return FormatKind::VOPC;
  if (tsFlags & SIInstrFlags::VOP3P)  return FormatKind::VOP3P;
  if (tsFlags & SIInstrFlags::VOP3)   return FormatKind::VOP3;
  if (tsFlags & SIInstrFlags::VOP2)   return FormatKind::VOP2;
  if (tsFlags & SIInstrFlags::VOP1)   return FormatKind::VOP1;
  if (tsFlags & SIInstrFlags::SMRD)   return FormatKind::SMEM;
  if (tsFlags & SIInstrFlags::FLAT)   return FormatKind::FLAT;
  if (tsFlags & SIInstrFlags::MUBUF)  return FormatKind::MUBUF;
  if (tsFlags & SIInstrFlags::DS)     return FormatKind::DS;
  return FormatKind::Unknown;
}

inline const char *formatName(FormatKind fk) {
  switch (fk) {
  case FormatKind::SOP1:    return "SOP1";
  case FormatKind::SOP2:    return "SOP2";
  case FormatKind::SOPC:    return "SOPC";
  case FormatKind::SOPK:    return "SOPK";
  case FormatKind::SOPP:    return "SOPP";
  case FormatKind::VOP1:    return "VOP1";
  case FormatKind::VOP2:    return "VOP2";
  case FormatKind::VOP3:    return "VOP3";
  case FormatKind::VOPC:    return "VOPC";
  case FormatKind::VOP3P:   return "VOP3P";
  case FormatKind::SMEM:    return "SMEM";
  case FormatKind::FLAT:    return "FLAT";
  case FormatKind::MUBUF:   return "MUBUF";
  case FormatKind::DS:      return "DS";
  case FormatKind::MFMA:    return "MFMA";
  case FormatKind::DPP:     return "DPP";
  case FormatKind::SDWA:    return "SDWA";
  case FormatKind::VOPD:    return "VOPD";
  case FormatKind::Unknown: return "Unknown";
  }
  return "Unknown";
}

} // namespace transpiler

#endif
