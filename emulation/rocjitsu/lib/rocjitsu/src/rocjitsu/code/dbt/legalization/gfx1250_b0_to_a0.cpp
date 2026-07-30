// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file legalization/gfx1250_b0_to_a0.cpp
/// @brief Handwritten gfx1250 B0-to-A0 legalization classification.

#include "rocjitsu/code/dbt/legalization/gfx1250_b0_to_a0.h"

#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/isa/instruction.h"

#include "util/log.h"

#include <array>
#include <cstring>
#include <string_view>

namespace rocjitsu {
namespace {

/// @brief Exact instruction names handled by the B0-to-A0 expansion profile.
///
/// @details Keep this list aligned with the implemented B0-to-A0 semantic
/// rules. Prefix-classified WMMA/SWMMAC and cluster-load instructions are
/// handled separately by family-level translation rules.
///
/// NOT-YET-SUPPORTED (classified as needing an expansion but with no semantic
/// expander, so translating a kernel that uses them fails closed rather than
/// passing the instruction through unchanged):
///   * s_barrier_signal_isfirst,
///   * v_cvt_pk_fp8_f32, v_cvt_sr_fp8_f32 (only when CLAMP selects the B0-only
///     mode; the ordinary form stays on the copy path),
///   * v_wmma_scale / v_wmma_scale16 forms without an implemented rule,
///   * integer IU8/IU4 WMMA/SWMMAC.
/// Separately, a 64-bit source using FLAT_SCRATCH_BASE_HI is classified via
/// operand inspection (see uses_flat_scratch_base_hi_64bit_source), and the
/// barrier-state and sleep/monitor families are DEFERRED with a pass-through
/// warning rather than fail-closed (see is_deferred_gfx1250_family).
/// Classifying the fail-closed cases keeps the failure explicit and located; add
/// the semantic rule (and update this note) once each expansion is implemented.
inline constexpr std::array<std::string_view, 18> kExactB0ToA0TranslationMnemonics = {
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

[[nodiscard]] bool requires_b0_to_a0_expansion(std::string_view mnemonic) {
  // This is deliberately more conservative than the reference patch
  // patterns. Rocjitsu relocates and expands instructions, so it cannot retain
  // a source clause without revalidating the translated membership and
  // placement constraints.
  if (mnemonic == "s_clause")
    return true;

  for (std::string_view exact : kExactB0ToA0TranslationMnemonics) {
    if (mnemonic == exact)
      return true;
  }

  // Every cluster-load form is kept as a cluster load and wrapped to run with
  // M0 forced to zero (save M0, set M0 = 0, load, restore M0). The semantic rule
  // performs the rewrite.
  if (mnemonic.starts_with("cluster_load_"))
    return true;

  // The reference patch accepts every encoding suffix in this conversion
  // family. The semantic rule further restricts the expansion to the
  // operand/modifier combinations selected by the B0-to-A0 profile.
  if (mnemonic.starts_with("v_cvt_f32_fp8"))
    return true;

  // The eight K=128 FP8/BF8 forms and the standalone 32x16 FP4 WMMA exist on B0
  // but not A0, so they require semantic expansion. The f32 K=128 family has an
  // exact neutral regular-Scale lowering; f16 and standalone 32x16 FP4 still fail
  // closed in their semantic rules.
  const bool is_k128_fp8_bf8 = (mnemonic.starts_with("v_wmma_f16_16x16x128_") ||
                                mnemonic.starts_with("v_wmma_f32_16x16x128_")) &&
                               (mnemonic.ends_with("_fp8_fp8") || mnemonic.ends_with("_fp8_bf8") ||
                                mnemonic.ends_with("_bf8_fp8") || mnemonic.ends_with("_bf8_bf8"));
  if (is_k128_fp8_bf8 || mnemonic == "v_wmma_f32_32x16x128_f4")
    return true;

  // A0 trap/CWSR recovery requires every low-precision F8F6F4 WMMA to carry its
  // load-scale prefix, even when the requested scale is 1.0. Standalone input
  // is therefore wrapped with inline-zero neutral E8M0 scales. Scale16,
  // regular Scale, and B0-only M=32 forms have separate encoding and
  // scale-source translations.
  if (mnemonic == "v_wmma_f32_16x16x128_f8f6f4" || mnemonic.starts_with("v_wmma_scale"))
    return true;

  // K=64 FP8/BF8 WMMA is present on A0 and retains its architectural encoding.
  // It stays on the ordinary copy path. K=128 F8F6F4 is also present on A0, but
  // the A0 profile uses its scaled wrapper in software-visible code. Semantic
  // lowerings may still use it as the matrix body of that atomic four-DWORD
  // wrapper.
  //
  // FP8/BF8 SWMMAC is present on both A0 and B0. Unlike dense K=128 WMMA,
  // the gfx1250 A0-to-B0 change table does not classify these sparse forms as
  // B0 additions, and their opcodes remain inside the A0 seven-bit VOP3P
  // opcode field. They therefore stay on the same-stepping byte-copy path.

  // The A0 co-execution distance exceeds B0 only for integer IU8/IU4 WMMA or
  // SWMMAC. FP16/BF16 need four spacing slots on both steppings, while floating
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

/// @brief True when any source operand uses the special FLAT_SCRATCH_BASE_HI
/// value in a 64-bit source position.
/// @details This special scalar source is not usable in a 64-bit source position
/// on this target, but the restriction is operand-sensitive rather than tied to a
/// mnemonic family, so it needs per-operand inspection. A 32-bit use of the same
/// value is unaffected. Encoding value 231 identifies the special source (see
/// gfx1250/operand_types.h).
[[nodiscard]] bool uses_flat_scratch_base_hi_64bit_source(const Instruction &inst) {
  constexpr int kFlatScratchBaseHiEncoding = 231;
  constexpr int k64BitOperand = 64;
  for (int i = 0; i < inst.num_src_operands(); ++i) {
    const Operand *op = inst.src_operand(i);
    if (op != nullptr && op->size_bits() == k64BitOperand &&
        op->encoding_value() == kFlatScratchBaseHiEncoding)
      return true;
  }
  return false;
}

/// @brief True for instruction families whose A0 handling is deferred pending
/// confirmation of the exact translated set.
/// @details The barrier-state query and the sleep/monitor families may need
/// target-specific translation that is not yet implemented. Rather than fail closed
/// (which would refuse otherwise-translatable kernels that use very common ops
/// such as s_sleep), these are passed through unchanged for now and a warning is
/// emitted so the omission is visible. Revisit once the precise set is
/// confirmed; if translation is required, move the relevant members to
/// requires_b0_to_a0_expansion() so they fail closed instead.
[[nodiscard]] bool is_deferred_gfx1250_family(std::string_view mnemonic) {
  return mnemonic == "s_get_barrier_state" || mnemonic == "s_sleep" || mnemonic == "s_sleep_var" ||
         mnemonic == "s_monitor_sleep";
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

  if (!requires_b0_to_a0_expansion(inst.mnemonic()) &&
      !uses_flat_scratch_base_hi_64bit_source(inst)) {
    // Deferred families pass through unchanged but warn, so the not-yet-handled
    // case is visible rather than silent. See is_deferred_gfx1250_family.
    if (is_deferred_gfx1250_family(mnemonic))
      util::Logger::warn("gfx1250 translation passes through '", mnemonic,
                         "' unchanged; target-specific handling is not yet implemented");
    return nullptr;
  }

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
