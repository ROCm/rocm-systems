// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file machine_insts_scalar.h
/// @brief Shared scalar instruction encoding structs for all AMDGPU ISAs.
///
/// The five scalar encoding formats (SOP1, SOP2, SOPC, SOPK, SOPP) and their
/// literal-constant variants have identical bit-field layouts across GFX9,
/// GFX10, GFX11, and GFX12. This header defines them once in the
/// `rocjitsu::amdgpu` namespace; per-ISA `machine_insts.h` can import them
/// via type aliases when the codegen detects a field-identical match.
///
/// Verified identical across: CDNA1, CDNA2, CDNA3, CDNA4, RDNA1, RDNA2,
/// RDNA3, RDNA3.5, RDNA4.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MACHINE_INSTS_SCALAR_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MACHINE_INSTS_SCALAR_H_

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

// ---------------------------------------------------------------------------
// Base scalar encodings (single 32-bit word)
// ---------------------------------------------------------------------------

struct Sop1MachineInst {
  uint32_t ssrc0 : 8;
  uint32_t op : 8;
  uint32_t sdst : 7;
  uint32_t encoding : 9;
};

struct SopcMachineInst {
  uint32_t ssrc0 : 8;
  uint32_t ssrc1 : 8;
  uint32_t op : 7;
  uint32_t encoding : 9;
};

struct SoppMachineInst {
  uint32_t simm16 : 16;
  uint32_t op : 7;
  uint32_t encoding : 9;
};

struct SopkMachineInst {
  uint32_t simm16 : 16;
  uint32_t sdst : 7;
  uint32_t op : 5;
  uint32_t encoding : 4;
};

struct Sop2MachineInst {
  uint32_t ssrc0 : 8;
  uint32_t ssrc1 : 8;
  uint32_t sdst : 7;
  uint32_t op : 7;
  uint32_t encoding : 2;
};

// ---------------------------------------------------------------------------
// Scalar literal-constant variants (base word + 32-bit immediate)
// ---------------------------------------------------------------------------

struct Sop1InstLiteralMachineInst {
  uint32_t ssrc0 : 8;
  uint32_t op : 8;
  uint32_t sdst : 7;
  uint32_t encoding : 9;
  uint32_t simm32 : 32;
};

struct Sop2InstLiteralMachineInst {
  uint32_t ssrc0 : 8;
  uint32_t ssrc1 : 8;
  uint32_t sdst : 7;
  uint32_t op : 7;
  uint32_t encoding : 2;
  uint32_t simm32 : 32;
};

struct SopcInstLiteralMachineInst {
  uint32_t ssrc0 : 8;
  uint32_t ssrc1 : 8;
  uint32_t op : 7;
  uint32_t encoding : 9;
  uint32_t simm32 : 32;
};

struct SopkInstLiteralMachineInst {
  uint32_t simm16 : 16;
  uint32_t sdst : 7;
  uint32_t op : 5;
  uint32_t encoding : 4;
  uint32_t simm32 : 32;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MACHINE_INSTS_SCALAR_H_
