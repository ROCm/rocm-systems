#ifndef HOTSWAP_TRANSPILER_AMDGPU_FORMATS_HPP
#define HOTSWAP_TRANSPILER_AMDGPU_FORMATS_HPP

#include <cstdint>

// Source tree: lib/Target/AMDGPU/SIDefines.h — target-private but exposed
// through the LLVM build tree via our CMake include path. Provides the
// authoritative `SIInstrFlags` enum and `AMDGPU::OPERAND_INPUT_MODS` operand
// type used by the disassembler's TSFlags / OperandType fields.
#include "SIDefines.h"

#include "Utils/AMDGPUBaseInfo.h" // AMDGPU::isVOPD

namespace transpiler {

// Alias `transpiler::SIInstrFlags` to the LLVM namespace so existing call
// sites (`SIInstrFlags::SOPP`, `SIInstrFlags::FLAT`, etc.) keep compiling.
namespace SIInstrFlags = llvm::SIInstrFlags;

// AMDGPU target-specific operand type for VOP3 source modifiers (abs, neg).
// Defined in llvm::AMDGPU::OperandType from SIDefines.h.
constexpr unsigned OPERAND_INPUT_MODS = llvm::AMDGPU::OPERAND_INPUT_MODS;

// Human-readable format label for diagnostics. There is no runtime dispatch
// on this string — it is consumed only by error messages in the decoder.
// The precedence of the TSFlags tests below mirrors LLVM's own decoder:
//   * `IsMAI` is a VOP3 subclass, so check before VOP3.
//   * `DPP` / `SDWA` are orthogonal encoding bits that coexist with
//     VOP1/VOP2/VOPC; check them first so those aren't misnamed as VOP1/2.
//   * `VOP3P` coexists with `VOP3` on some subtargets; check VOP3P first.
//   * VOPD has no dedicated TSFlags bit (LLVM's VOPD3 bit varies across
//     versions); use `AMDGPU::isVOPD(opc)` instead.
inline const char *formatName(uint64_t flags, unsigned opc) {
  if (llvm::AMDGPU::isVOPD(opc))     return "VOPD";
  if (flags & SIInstrFlags::IsMAI)   return "MFMA";
  if (flags & SIInstrFlags::DPP)     return "DPP";
  if (flags & SIInstrFlags::SDWA)    return "SDWA";
  if (flags & SIInstrFlags::SOPP)    return "SOPP";
  if (flags & SIInstrFlags::SOPC)    return "SOPC";
  if (flags & SIInstrFlags::SOP1)    return "SOP1";
  if (flags & SIInstrFlags::SOP2)    return "SOP2";
  if (flags & SIInstrFlags::SOPK)    return "SOPK";
  if (flags & SIInstrFlags::VOPC)    return "VOPC";
  if (flags & SIInstrFlags::VOP3P)   return "VOP3P";
  if (flags & SIInstrFlags::VOP3)    return "VOP3";
  if (flags & SIInstrFlags::VOP2)    return "VOP2";
  if (flags & SIInstrFlags::VOP1)    return "VOP1";
  if (flags & SIInstrFlags::SMRD)    return "SMEM";
  if (flags & SIInstrFlags::FLAT)    return "FLAT";
  if (flags & SIInstrFlags::MUBUF)   return "MUBUF";
  if (flags & SIInstrFlags::DS)      return "DS";
  // VIMAGE: gfx12+ vector image / tensor encoding family. Pure-image
  // members carry `SIInstrFlags::VIMAGE` directly; the gfx1250 TENSOR
  // pseudos (`tensor_load_to_lds_d{2,4}`,
  // `tensor_store_from_lds_d{2,4}`, MIMGInstructions.td:2049-2113) do
  // NOT — they extend `InstSI` directly and only set `let VALU = 1`
  // and `let TENSOR_CNT = 1`, so the `VIMAGE` field stays 0. Detect
  // them via the `TENSOR_CNT` TSFlags bit instead. The only other
  // user of that bit is `s_wait_tensorcnt` (SOPP), which already
  // matches the SOPP arm above and never reaches this fallthrough.
  // Routing both arms to the same `"VIMAGE"` label lets
  // `kerneldex`/`raise_cli` bucket the cross-target failures as
  // `[format=VIMAGE]` rather than `[format=Unknown]`, which is what
  // the handler refusal contract is keyed on.
  if (flags & SIInstrFlags::VIMAGE)     return "VIMAGE";
  if (flags & SIInstrFlags::TENSOR_CNT) return "VIMAGE";
  return "Unknown";
}

} // namespace transpiler

#endif
