// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file machine_insts_cdna.h
/// @brief Shared vector/memory instruction encoding structs for CDNA1–4 (GFX9 family).
///
/// Contains encoding structs that are field-name-and-bit-width identical across
/// all four CDNA generations. Per-ISA `machine_insts.h` can import them via
/// type aliases when the codegen detects a field-identical match.
///
/// **NOT included here** (differ across generations):
///   - DsMachineInst       — CDNA1 uses `pad_25`, CDNA2/3/4 use `acc`
///   - MubufMachineInst    — coherency fields differ (glc/slc → scc/slc → sc0/sc1/nt)
///   - MtbufMachineInst    — same coherency field evolution
///   - FlatMachineInst     — same coherency field evolution
///   - FlatGlblMachineInst / FlatScratchMachineInst — same
///   - Vop3pMachineInst    — CDNA4 renames `op_sel_hi_2` to `pad_14`
///   - Vop3pMfmaMachineInst — CDNA1 uses `pad_15`, CDNA2/3/4 use `acc_cd`
///   - ExpMachineInst      — only present on CDNA1/2 (GFX9 export path)
///   - MimgMachineInst     — only present on CDNA1/2; fields differ between them
///   - VintrpMachineInst   — CDNA1 only
///
/// Verified identical across: CDNA1, CDNA2, CDNA3, CDNA4.

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MACHINE_INSTS_CDNA_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MACHINE_INSTS_CDNA_H_

#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

// ---------------------------------------------------------------------------
// SMEM — Scalar memory (identical across all 4 CDNA ISAs)
// ---------------------------------------------------------------------------

struct SmemMachineInst {
  uint32_t sbase : 6;
  uint32_t sdata : 7;
  uint32_t pad_13 : 1;
  uint32_t soffset_en : 1;
  uint32_t nv : 1;
  uint32_t glc : 1;
  uint32_t imm : 1;
  uint32_t op : 8;
  uint32_t encoding : 6;
  uint32_t offset : 21;
  uint32_t pad_53_56 : 4;
  uint32_t soffset : 7;
};

// ---------------------------------------------------------------------------
// VOP1 / VOP2 / VOPC — Vector ALU base encodings
// ---------------------------------------------------------------------------

struct Vop1MachineInst {
  uint32_t src0 : 9;
  uint32_t op : 8;
  uint32_t vdst : 8;
  uint32_t encoding : 7;
};

struct VopcMachineInst {
  uint32_t src0 : 9;
  uint32_t vsrc1 : 8;
  uint32_t op : 8;
  uint32_t encoding : 7;
};

struct Vop2MachineInst {
  uint32_t src0 : 9;
  uint32_t vsrc1 : 8;
  uint32_t vdst : 8;
  uint32_t op : 6;
  uint32_t encoding : 1;
};

// ---------------------------------------------------------------------------
// VOP3 / VOP3 with SDST — Vector ALU 64-bit encodings
// ---------------------------------------------------------------------------

struct Vop3MachineInst {
  uint32_t vdst : 8;
  uint32_t abs : 3;
  uint32_t op_sel : 4;
  uint32_t clamp : 1;
  uint32_t op : 10;
  uint32_t encoding : 6;
  uint32_t src0 : 9;
  uint32_t src1 : 9;
  uint32_t src2 : 9;
  uint32_t omod : 2;
  uint32_t neg : 3;
};

struct Vop3SdstEncMachineInst {
  uint32_t vdst : 8;
  uint32_t sdst : 7;
  uint32_t clamp : 1;
  uint32_t op : 10;
  uint32_t encoding : 6;
  uint32_t src0 : 9;
  uint32_t src1 : 9;
  uint32_t src2 : 9;
  uint32_t omod : 2;
  uint32_t neg : 3;
};

// ---------------------------------------------------------------------------
// DPP variants (GFX9 — single DPP format, no DPP8/DPP16 split)
// ---------------------------------------------------------------------------

struct Vop1VopDppMachineInst {
  uint32_t src0 : 9;
  uint32_t op : 8;
  uint32_t vdst : 8;
  uint32_t encoding : 7;
  uint32_t vsrc0 : 8;
  uint32_t dpp_ctrl : 9;
  uint32_t pad_49_50 : 2;
  uint32_t bound_ctrl : 1;
  uint32_t src0_neg : 1;
  uint32_t src0_abs : 1;
  uint32_t src1_neg : 1;
  uint32_t src1_abs : 1;
  uint32_t bank_mask : 4;
  uint32_t row_mask : 4;
};

struct Vop2VopDppMachineInst {
  uint32_t src0 : 9;
  uint32_t vsrc1 : 8;
  uint32_t vdst : 8;
  uint32_t op : 6;
  uint32_t encoding : 1;
  uint32_t vsrc0 : 8;
  uint32_t dpp_ctrl : 9;
  uint32_t pad_49_50 : 2;
  uint32_t bound_ctrl : 1;
  uint32_t src0_neg : 1;
  uint32_t src0_abs : 1;
  uint32_t src1_neg : 1;
  uint32_t src1_abs : 1;
  uint32_t bank_mask : 4;
  uint32_t row_mask : 4;
};

// ---------------------------------------------------------------------------
// SDWA variants (GFX9 — Sub-Dword Addressing)
// ---------------------------------------------------------------------------

struct Vop1VopSdwaMachineInst {
  uint32_t src0 : 9;
  uint32_t op : 8;
  uint32_t vdst : 8;
  uint32_t encoding : 7;
  uint32_t vsrc0 : 8;
  uint32_t dst_sel : 3;
  uint32_t dst_unused : 2;
  uint32_t clamp : 1;
  uint32_t omod : 2;
  uint32_t src0_sel : 3;
  uint32_t src0_sext : 1;
  uint32_t src0_neg : 1;
  uint32_t src0_abs : 1;
  uint32_t pad_54 : 1;
  uint32_t s0 : 1;
  uint32_t src1_sel : 3;
  uint32_t src1_sext : 1;
  uint32_t src1_neg : 1;
  uint32_t src1_abs : 1;
  uint32_t pad_62 : 1;
  uint32_t s1 : 1;
};

struct Vop2VopSdwaMachineInst {
  uint32_t src0 : 9;
  uint32_t vsrc1 : 8;
  uint32_t vdst : 8;
  uint32_t op : 6;
  uint32_t encoding : 1;
  uint32_t vsrc0 : 8;
  uint32_t dst_sel : 3;
  uint32_t dst_unused : 2;
  uint32_t clamp : 1;
  uint32_t omod : 2;
  uint32_t src0_sel : 3;
  uint32_t src0_sext : 1;
  uint32_t src0_neg : 1;
  uint32_t src0_abs : 1;
  uint32_t pad_54 : 1;
  uint32_t s0 : 1;
  uint32_t src1_sel : 3;
  uint32_t src1_sext : 1;
  uint32_t src1_neg : 1;
  uint32_t src1_abs : 1;
  uint32_t pad_62 : 1;
  uint32_t s1 : 1;
};

struct Vop2VopSdwaSdstEncMachineInst {
  uint32_t src0 : 9;
  uint32_t vsrc1 : 8;
  uint32_t vdst : 8;
  uint32_t op : 6;
  uint32_t encoding : 1;
  uint32_t vsrc0 : 8;
  uint32_t sdst : 7;
  uint32_t sd : 1;
  uint32_t src0_sel : 3;
  uint32_t src0_sext : 1;
  uint32_t src0_neg : 1;
  uint32_t src0_abs : 1;
  uint32_t pad_54 : 1;
  uint32_t s0 : 1;
  uint32_t src1_sel : 3;
  uint32_t src1_sext : 1;
  uint32_t src1_neg : 1;
  uint32_t src1_abs : 1;
  uint32_t pad_62 : 1;
  uint32_t s1 : 1;
};

struct VopcVopSdwaSdstEncMachineInst {
  uint32_t src0 : 9;
  uint32_t vsrc1 : 8;
  uint32_t op : 8;
  uint32_t encoding : 7;
  uint32_t vsrc0 : 8;
  uint32_t sdst : 7;
  uint32_t sd : 1;
  uint32_t src0_sel : 3;
  uint32_t src0_sext : 1;
  uint32_t src0_neg : 1;
  uint32_t src0_abs : 1;
  uint32_t pad_54 : 1;
  uint32_t s0 : 1;
  uint32_t src1_sel : 3;
  uint32_t src1_sext : 1;
  uint32_t src1_neg : 1;
  uint32_t src1_abs : 1;
  uint32_t pad_62 : 1;
  uint32_t s1 : 1;
};

// ---------------------------------------------------------------------------
// Vector literal-constant variants (base word + 32-bit immediate)
// ---------------------------------------------------------------------------

struct Vop1InstLiteralMachineInst {
  uint32_t src0 : 9;
  uint32_t op : 8;
  uint32_t vdst : 8;
  uint32_t encoding : 7;
  uint32_t simm32 : 32;
};

struct Vop2InstLiteralMachineInst {
  uint32_t src0 : 9;
  uint32_t vsrc1 : 8;
  uint32_t vdst : 8;
  uint32_t op : 6;
  uint32_t encoding : 1;
  uint32_t simm32 : 32;
};

struct VopcInstLiteralMachineInst {
  uint32_t src0 : 9;
  uint32_t vsrc1 : 8;
  uint32_t op : 8;
  uint32_t encoding : 7;
  uint32_t simm32 : 32;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MACHINE_INSTS_CDNA_H_
