// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file legalization/gfx1250_b0_to_a0.cpp
/// @brief Handwritten gfx1250 B0-to-A0 legalization classification.

#include "rocjitsu/code/dbt/legalization/gfx1250_b0_to_a0.h"

#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/isa/instruction.h"

#include <array>
#include <cstring>
#include <string_view>

namespace rocjitsu {
namespace {

/// @brief Exact instruction names whose A0 workaround needs an expansion.
///
/// @details Keep this list aligned with the implemented B0-to-A0 semantic
/// rules. Prefix-classified WMMA/SWMMAC and cluster-load instructions are
/// handled separately because their contextual workarounds apply to families.
///
/// NOT-YET-SUPPORTED (classified here but with no semantic expander):
/// s_barrier_signal_isfirst, v_cvt_pk_fp8_f32, and v_cvt_sr_fp8_f32 are classified
/// as needing an expansion but have no rule to produce one, so translating a kernel
/// that uses them hard-fails (fail-closed) rather than passing the instruction
/// through unchanged. Classifying them keeps that failure explicit and located.
/// Add the semantic rule (and drop this note) once the expansion is implemented.
inline constexpr std::array<std::string_view, 18> kExactErrataMnemonics = {
    "s_barrier_signal_isfirst",
    "ds_load_2addr_b32",
    "ds_load_2addr_b64",
    "ds_load_2addr_stride64_b32",
    "ds_load_2addr_stride64_b64",
    "ds_store_2addr_b32",
    "ds_store_2addr_b64",
    "ds_store_2addr_stride64_b32",
    "ds_store_2addr_stride64_b64",
    "ds_storexchg_2addr_rtn_b32",
    "ds_storexchg_2addr_rtn_b64",
    "ds_storexchg_2addr_stride64_rtn_b32",
    "ds_storexchg_2addr_stride64_rtn_b64",
    "ds_load_addtid_b32",
    "ds_store_addtid_b32",
    "v_cvt_pk_fp8_f32",
    "v_cvt_sr_fp8_f32",
    "tensor_load_to_lds",
};

[[nodiscard]] bool requires_errata_expansion(std::string_view mnemonic) {
  // This is deliberately more conservative than the reference patch
  // patterns. Rocjitsu relocates and expands instructions, so it cannot retain
  // a source clause without revalidating the translated membership and
  // placement constraints.
  if (mnemonic == "s_clause")
    return true;

  for (std::string_view exact : kExactErrataMnemonics) {
    if (mnemonic == exact)
      return true;
  }

  // Every cluster-load form needs either demotion to a global load or an M0
  // cluster-mask sequence. Operand inspection will choose the precise rule.
  if (mnemonic.starts_with("cluster_load_"))
    return true;

  // The reference patch accepts every encoding suffix in this conversion
  // family. The semantic rule further restricts the expansion to the
  // operand/modifier combinations that actually need the A0 workaround.
  if (mnemonic.starts_with("v_cvt_f32_fp8"))
    return true;

  // These eight K=128 FP8/BF8 forms exist on B0 but must be split into K=64
  // operations for A0. Match the closed family precisely: ordinary K=128
  // F8F6F4 is the A0 replacement for another workaround and is not a split
  // candidate itself.
  const bool is_k128_fp8_bf8 = (mnemonic.starts_with("v_wmma_f16_16x16x128_") ||
                                mnemonic.starts_with("v_wmma_f32_16x16x128_")) &&
                               (mnemonic.ends_with("_fp8_fp8") || mnemonic.ends_with("_fp8_bf8") ||
                                mnemonic.ends_with("_bf8_fp8") || mnemonic.ends_with("_bf8_bf8"));
  if (is_k128_fp8_bf8 || mnemonic == "v_wmma_f32_32x16x128_f4")
    return true;

  // Scale16 and regular Scale have separate mandatory encoding/scale-source
  // workarounds. Keep them fail-closed until their semantic rules land.
  if (mnemonic.starts_with("v_wmma_scale"))
    return true;

  // The A0 co-execution distance exceeds B0 only for integer IU8/IU4 WMMA or
  // SWMMAC. FP16/BF16 need four safe slots on both steppings, while floating
  // FP8 forms need no additional A0 padding. The integer forms remain
  // fail-closed until a CFG-aware spacing pass can inspect following VALU.
  const bool is_wmma_like = mnemonic.starts_with("v_wmma_") || mnemonic.starts_with("v_swmmac_");
  return is_wmma_like && (mnemonic.find("_iu8") != std::string_view::npos ||
                          mnemonic.find("_iu4") != std::string_view::npos);
}

/// @brief True when a B0 FP8 conversion selects the B0-only E5M3 mode.
///
/// @details The affected VOP3 conversions reuse CLAMP as the E5M3 selector on
/// B0. A0 implements the same CLAMP=0 E4M3 operation, so those instructions
/// must remain on the ordinary byte-copy path. CLAMP lives in the eight-byte
/// VOP3 base encoding; a trailing literal increases the decoded size without
/// moving that field.
[[nodiscard]] bool requires_fp8_clamp_emulation(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();
  const bool affected = mnemonic == "v_cvt_pk_fp8_f32" || mnemonic == "v_cvt_sr_fp8_f32" ||
                        mnemonic.starts_with("v_cvt_f32_fp8");
  const bool is_vop3 = inst.encoding_id() >= gfx1250::encoding::kVop3 &&
                       inst.encoding_id() <= gfx1250::encoding::kVop3OpHi6;
  if (!affected || !is_vop3 || inst.size() < static_cast<int>(sizeof(gfx1250::Vop3MachineInst)) ||
      inst.raw_encoding() == nullptr)
    return false;

  gfx1250::Vop3MachineInst encoding{};
  std::memcpy(&encoding, inst.raw_encoding(), sizeof(encoding));
  return encoding.clamp != 0;
}

} // namespace

const InstructionLegalization *gfx1250_b0_to_a0_legalization(const Instruction &inst) {
  // CLAMP=0 is the common E4M3 operation on both steppings. CLAMP=1 selects
  // the B0-only E5M3 behavior and therefore requires a semantic expansion.
  const std::string_view mnemonic = inst.mnemonic();
  const bool fp8_clamp_family = mnemonic == "v_cvt_pk_fp8_f32" || mnemonic == "v_cvt_sr_fp8_f32" ||
                                mnemonic.starts_with("v_cvt_f32_fp8");
  if (fp8_clamp_family && !requires_fp8_clamp_emulation(inst))
    return nullptr;

  if (!requires_errata_expansion(inst.mnemonic()))
    return nullptr;

  // The runtime uses only the action and target opcode for this revision-specific
  // classification. Source keys remain zero because matching is performed on
  // the fully decoded mnemonic, which is necessary for contextual gfx1250
  // variants that share structural opcode fields.
  static constexpr InstructionLegalization kExpand{
      .src_opcode = 0,
      .src_encoding_id = 0,
      .action = Action::Expand,
      .target_opcode = 0,
  };
  return &kExpand;
}

} // namespace rocjitsu
