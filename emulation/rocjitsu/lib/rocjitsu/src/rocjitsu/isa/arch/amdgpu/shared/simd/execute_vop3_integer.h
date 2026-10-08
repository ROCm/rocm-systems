// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP3_INTEGER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP3_INTEGER_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd/common.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/portable_math.h"

namespace rocjitsu::amdgpu {

/// v_cndmask_b32 VOP3 form: dst[lane] = (sel[lane]) ? src1 : src0, where `sel`
/// is the wave-mask value read from scalar `src2` (instead of the fixed VCC
/// used by the VOP2 form). Wave32-only gfx1250 uses one SGPR; wave64-capable
/// targets use a pair when running Wave64. VOP3 source modifiers are bitwise sign modifiers for
/// this B32 select: abs clears bit 31 and neg flips bit 31 before selection.
/// `src2` is an SGPR/inline operand, not a VGPR, so it does not participate in
/// the simd_capable gate; src0/src1/vdst do.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cndmask_vop3_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint64_t sel64 = read_wave_mask_scalar(inst.src2, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_b32_src_mod(src0.template load_native<T>(base), inst.inst_.abs,
                                          inst.inst_.neg, 0);
    const auto b = apply_vop3_b32_src_mod(src1.template load_native<T>(base), inst.inst_.abs,
                                          inst.inst_.neg, 1);
    const uint64_t sel_bits = (sel64 >> base) & chunk_full;
    auto r = a;
    util::stdx::where(util::simd_mask_from_bits<util::native<T>>(sel_bits), r) = b;
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 cndmask path; see the binary-path note.
template <typename Inst> [[nodiscard]] bool try_execute_cndmask_vop3_simd(Inst &, Wavefront &) {
  return false;
}

/// v_cndmask_b16 VOP3 form: same per-lane select as cndmask_vop3 but the
/// dst (and each source) is the low 16 bits of the 32-bit VGPR; the high 16
/// are zeroed (matching the scalar body's `uint32_t(uint16_t(...))` pattern).
/// The select shape is identical to the b32 form — the only addition is a
/// `& 0xFFFFu` mask before the masked store. RDNA3+; CDNA4 does not decode.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cndmask_b16_vop3_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint64_t sel64 = read_wave_mask_scalar(inst.src2, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t sel_bits = (sel64 >> base) & chunk_full;
    auto r = a;
    util::stdx::where(util::simd_mask_from_bits<util::native<T>>(sel_bits), r) = b;
    r = r & util::native<T>(0xFFFFu);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_cndmask_b16_vop3_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3 integer/bitwise binary SIMD fast path. Same shape as
/// try_execute_binary_vop2_simd but reads the VOP3 operands `src0`/`src1`
/// (instead of `src0`/`vsrc1`). The generated integer/bitwise VOP3 bodies apply
/// no source/result modifiers other than integer CLAMP. Saturating integer
/// arithmetic falls back to the scalar body; the plain op is bit-identical to
/// that body when CLAMP is clear. T is a 32-bit integer lane type.
template <typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_simd(Inst &inst, Wavefront &wf, BinOp bin_op) {
  if (inst.inst_.clamp != 0u || simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) ||
      !inst.src0.simd_capable() || !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const auto r = bin_op(a, b);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 integer binary path; see the VOP2
/// binary-path note above.
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 binary operations whose operands are encoded as true16 sources but
/// whose destination is a full b32 pack result.
template <typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_true16_src_simd(Inst &inst, Wavefront &wf,
                                                                  BinOp bin_op) {
  static_assert(std::is_same_v<T, uint32_t>);
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t opsel = vop3_opsel(inst.inst_);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = select_vop3_true16_src(src0.template load_native<T>(base), opsel, 0);
    const auto b = select_vop3_true16_src(src1.template load_native<T>(base), opsel, 1);
    const auto r = bin_op(a, b);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_true16_src_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 f16 binary fast path. The generic form matches the ordinary scalar
/// body's low-half read plus full-dword zero-extending write. The true16 form
/// selects source halves with op_sel[0:1] and writes the destination half per
/// the ISA's op_sel[3] policy. The packed-f16 binary functors do not apply
/// abs/neg/omod/clamp, so both forms bail to scalar whenever a modifier is
/// present.
template <bool True16, typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_f16_simd(Inst &inst, Wavefront &wf,
                                                           BinOp bin_op) {
  static_assert(std::is_same_v<T, uint32_t>);
  if (inst.inst_.abs != 0u || inst.inst_.neg != 0u || inst.inst_.omod != 0u ||
      inst.inst_.clamp != 0u)
    return false;
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t opsel = vop3_opsel(inst.inst_);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  if constexpr (True16) {
    auto dst = regs.readwrite_operand(inst.vdst, exec);
    if (!dst.has_storage())
      return false;
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      auto a = src0.template load_native<T>(base);
      auto b = src1.template load_native<T>(base);
      a = select_vop3_true16_src(a, opsel, 0);
      b = select_vop3_true16_src(b, opsel, 1);
      const auto r = bin_op(a, b);
      const auto out_half = r & util::broadcast<T>(0xffffu);
      auto prev = dst.template load_native<T>(base);
      auto out = (opsel & 0x8u) ? ((prev & util::broadcast<T>(0x0000ffffu)) | (out_half << 16))
                                : ((prev & util::broadcast<T>(0xffff0000u)) | out_half);
      if (!(opsel & 0x8u) && cdna_vop3_low_dst_zeroes_high(wf))
        out = out_half;
      dst.template store_native<T>(base, out, chunk);
    }
  } else {
    auto dst = regs.write_operand(inst.vdst, exec);
    for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
      const uint64_t chunk = (exec >> base) & chunk_full;
      if (chunk == 0)
        continue;
      const auto a = src0.template load_native<T>(base);
      const auto b = src1.template load_native<T>(base);
      const auto r = bin_op(a, b);
      dst.template store_native<T>(base, r, chunk);
    }
  }
  return true;
}

template <bool True16, typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_f16_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 integer/bitwise ternary SIMD fast path. Reads `src0`/`src1`/`src2`,
/// runs `tern_op(a, b, c)`, and masked-stores the result. The generated scalar
/// bodies for these ternary integer ops apply no source/result modifiers except
/// integer CLAMP. Saturating arithmetic falls back to the scalar body; the plain
/// functor is bit-identical to that body when CLAMP is clear. T is a 32-bit
/// integer lane type (typically uint32_t). Same SIMD-capable / EXEC-chunk loop
/// as the binary VOP3 path.
template <typename T, typename Inst, typename TernOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_simd(Inst &inst, Wavefront &wf, TernOp tern_op) {
  if (inst.inst_.clamp != 0u || simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) ||
      !inst.src0.simd_capable() || !inst.src1.simd_capable() || !inst.src2.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const auto c = src2.template load_native<T>(base);
    const auto r = tern_op(a, b, c);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 integer ternary path; see binary-path note.
template <typename T, typename Inst, typename TernOp>
[[nodiscard]] bool try_execute_ternary_vop3_simd(Inst &, Wavefront &, TernOp) {
  return false;
}

/// VOP3 ternary operations with true16 SRC0/SRC1 and a full-width SRC2/result,
/// such as V_MAD_[IU]32_[IU]16.
template <typename T, typename Inst, typename TernOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_true16_src01_simd(Inst &inst, Wavefront &wf,
                                                                     TernOp tern_op) {
  static_assert(std::is_same_v<T, uint32_t>);
  if (inst.inst_.clamp != 0u || simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) ||
      !inst.src0.simd_capable() || !inst.src1.simd_capable() || !inst.src2.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t opsel = vop3_opsel(inst.inst_);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = select_vop3_true16_src(src0.template load_native<T>(base), opsel, 0);
    const auto b = select_vop3_true16_src(src1.template load_native<T>(base), opsel, 1);
    const auto c = src2.template load_native<T>(base);
    const auto r = tern_op(a, b, c);
    dst.template store_native<T>(base, r, chunk);
  }
  return true;
}

template <typename T, typename Inst, typename TernOp>
[[nodiscard]] bool try_execute_ternary_vop3_true16_src01_simd(Inst &, Wavefront &, TernOp) {
  return false;
}

/// VOP3 ternary operations whose three sources and destination are true16,
/// such as min3/max3/med3 i16/u16 on RDNA true16 encodings.
template <typename T, typename Inst, typename TernOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_true16_simd(Inst &inst, Wavefront &wf,
                                                               TernOp tern_op) {
  static_assert(std::is_same_v<T, uint32_t>);
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint32_t opsel = vop3_opsel(inst.inst_);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand(inst.src2, exec);
  auto dst = regs.readwrite_operand(inst.vdst, exec);
  if (!dst.has_storage())
    return false;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = select_vop3_true16_src(src0.template load_native<T>(base), opsel, 0);
    const auto b = select_vop3_true16_src(src1.template load_native<T>(base), opsel, 1);
    const auto c = select_vop3_true16_src(src2.template load_native<T>(base), opsel, 2);
    const auto r = tern_op(a, b, c);
    const auto out_half = r & util::broadcast<T>(0xffffu);
    auto prev = dst.template load_native<T>(base);
    auto out = (opsel & 0x8u) ? ((prev & util::broadcast<T>(0x0000ffffu)) | (out_half << 16))
                              : ((prev & util::broadcast<T>(0xffff0000u)) | out_half);
    if (!(opsel & 0x8u) && cdna_vop3_low_dst_zeroes_high(wf))
      out = out_half;
    dst.template store_native<T>(base, out, chunk);
  }
  return true;
}

template <typename T, typename Inst, typename TernOp>
[[nodiscard]] bool try_execute_ternary_vop3_true16_simd(Inst &, Wavefront &, TernOp) {
  return false;
}

/// VOP3 64-bit-lane reverse-shift fast path (v_lshlrev_b64 / v_lshrrev_b64 /
/// v_ashrrev_i64). The shift amount is a 32-bit src0 (read as a native_width64
/// narrow lane, widened to 64-bit and masked to [0,63]); the shifted value is
/// the 64-bit src1; the result is 64-bit. `shift_op(value, shift)` receives both
/// as `native<uint64_t>` (shift already widened+masked), so the functor is a
/// plain `v << sh` / `v >> sh` (logical) or an arithmetic-shift cast for the
/// signed form — bit-identical to the scalar body's `& 63u` shift.
template <typename Inst, typename ShiftOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_shift64_vop3_simd(Inst &inst, Wavefront &wf,
                                                        ShiftOp shift_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  // src0 = 32-bit shift amount (narrow lane), src1 = 64-bit value, dst = 64-bit.
  RegisterAccess regs(wf);
  auto shift_src = regs.read_operand(inst.src0, exec);
  auto value_src = regs.read_operand64(inst.src1, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = shift_src.template load_narrow<uint32_t>(base);
    const auto v = value_src.template load_native<uint64_t>(base);
    const auto sh = util::stdx::static_simd_cast<util::native<uint64_t>>(s) & 63ull;
    dst.template store_native<uint64_t>(base, shift_op(v, sh), chunk);
  }
  return true;
}

template <typename Inst, typename ShiftOp>
[[nodiscard]] bool try_execute_shift64_vop3_simd(Inst &, Wavefront &, ShiftOp) {
  return false;
}

/// VOP3 v_lshl_add_u64 fast path: dst = (src0 << (src1 & 63)) + src2, all 64-bit
/// except the 32-bit shift amount src1. The scalar body shifts by the raw src1
/// (C++ UB at >=64, but x86 masks the count to 6 bits); masking to 63 here
/// reproduces that x86 scalar result for every shift value.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_lshl_add_u64_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  // src0 = 64-bit value, src1 = 32-bit shift (narrow), src2 = 64-bit addend.
  RegisterAccess regs(wf);
  auto value_src = regs.read_operand64(inst.src0, exec);
  auto shift_src = regs.read_operand(inst.src1, exec);
  auto addend_src = regs.read_operand64(inst.src2, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto v = value_src.template load_native<uint64_t>(base);
    const auto s = shift_src.template load_narrow<uint32_t>(base);
    const auto c = addend_src.template load_native<uint64_t>(base);
    const auto sh = util::stdx::static_simd_cast<util::native<uint64_t>>(s) & 63ull;
    dst.template store_native<uint64_t>(base, simd_lshl_u64(v, sh) + c, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_lshl_add_u64_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3 wide 32x32->64 multiply-add fast path (v_mad_u64_u32 / v_mad_i64_i32).
/// src0/src1 are 32-bit multiplicands (read as narrow lanes), src2 is the 64-bit
/// addend, dst is 64-bit, and sdst is the per-lane overflow/carryout mask.
/// `mad_op(s0, s1, c)` receives the two narrow operands and the 64-bit addend
/// and returns both the low 64-bit result and carry/overflow mask.
template <typename Inst, typename MadOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_mad_wide64_vop3_result_simd(Inst &inst, Wavefront &wf,
                                                                  MadOp mad_op,
                                                                  WriteResult write_result) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if constexpr (requires { inst.inst_.clamp; }) {
    if (inst.inst_.clamp)
      return false;
  }
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t carry_out = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto src2 = regs.read_operand64(inst.src2, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_narrow<uint32_t>(base);
    const auto b = src1.template load_narrow<uint32_t>(base);
    const auto c = src2.template load_native<uint64_t>(base);
    const auto r = mad_op(a, b, c);
    dst.template store_native<uint64_t>(base, r.value, chunk);
    const uint64_t carry_bits = util::simd_mask_to_bits(r.carry);
    carry_out = (carry_out & ~(chunk << base)) | ((carry_bits & chunk) << base);
  }
  write_result(carry_out);
  return true;
}

template <typename Inst, typename MadOp, typename WriteResult>
[[nodiscard]] bool try_execute_mad_wide64_vop3_result_simd(Inst &, Wavefront &, MadOp,
                                                           WriteResult) {
  return false;
}

template <typename Inst, typename MadOp>
[[nodiscard]] inline bool try_execute_mad_wide64_vop3_simd(Inst &inst, Wavefront &wf,
                                                           MadOp mad_op) {
  return try_execute_mad_wide64_vop3_result_simd(
      inst, wf, mad_op, [&](uint64_t result) { write_wave_mask_scalar(inst.sdst, wf, result); });
}

/// VOP3 carry-OUT binary fast path (v_add_co/sub_co/subrev_co_u32). No carry-in
/// (the SimdCarry functor's third arg is a zero vector); the per-lane carry-out
/// is written to the SGPR-pair `sdst` (not the fixed VCC). In wave32, explicit
/// scalar mask destinations write only the low SGPR and preserve the high SGPR;
/// wave64 writes the full pair.
/// `sdst` is an SGPR operand, so it does not participate in the simd_capable
/// gate; src0/src1/vdst do. (These VOP3 forms carry no `src2` member, so the
/// carry-in path lives in the separate _cin glue below.)
template <typename Inst, typename CarryOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_co_result_simd(Inst &inst, Wavefront &wf,
                                                                 CarryOp carry_op,
                                                                 WriteResult write_result) {
  if (simd_force_scalar() || inst.inst_.clamp || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t carry_out = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  const auto zero_cin = util::broadcast<T>(0u);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const auto r = carry_op(a, b, zero_cin);
    dst.template store_native<T>(base, r.value, chunk);
    const uint64_t carry_bits = util::simd_mask_to_bits(r.carry);
    carry_out = (carry_out & ~(chunk << base)) | ((carry_bits & chunk) << base);
  }
  write_result(carry_out);
  return true;
}

template <typename Inst, typename CarryOp, typename WriteResult>
[[nodiscard]] bool try_execute_binary_vop3_co_result_simd(Inst &, Wavefront &, CarryOp,
                                                          WriteResult) {
  return false;
}

template <typename Inst, typename CarryOp>
[[nodiscard]] inline bool try_execute_binary_vop3_co_simd(Inst &inst, Wavefront &wf,
                                                          CarryOp carry_op) {
  return try_execute_binary_vop3_co_result_simd(
      inst, wf, carry_op, [&](uint64_t result) { write_wave_mask_scalar(inst.sdst, wf, result); });
}

/// VOP3 carry-IN binary fast path (v_addc_co/subb_co/subbrev_co_u32). Same as
/// the _co glue but the per-lane carry-in is read from the SGPR-pair `src2`
/// (these forms have a src2 member) and expanded to a 0/1-per-lane vector;
/// carry-out goes to `sdst` with the same wave32/wave64 width rule as _co.
/// src2/sdst are SGPR operands (not gated).
template <typename Inst, typename CarryOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_cin_result_simd(Inst &inst, Wavefront &wf,
                                                                  CarryOp carry_op,
                                                                  WriteResult write_result) {
  if (simd_force_scalar() || inst.inst_.clamp || !inst.src0.simd_capable() ||
      !inst.src1.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const uint64_t cin_all = read_wave_mask_scalar(inst.src2, wf);
  uint64_t carry_out = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t cin_bits = (cin_all >> base) & chunk_full;
    const auto cin = util::simd_u32_lanes_from_bits(cin_bits);
    const auto r = carry_op(a, b, cin);
    dst.template store_native<T>(base, r.value, chunk);
    const uint64_t carry_bits = util::simd_mask_to_bits(r.carry);
    carry_out = (carry_out & ~(chunk << base)) | ((carry_bits & chunk) << base);
  }
  write_result(carry_out);
  return true;
}

template <typename Inst, typename CarryOp, typename WriteResult>
[[nodiscard]] bool try_execute_binary_vop3_cin_result_simd(Inst &, Wavefront &, CarryOp,
                                                           WriteResult) {
  return false;
}

template <typename Inst, typename CarryOp>
[[nodiscard]] inline bool try_execute_binary_vop3_cin_simd(Inst &inst, Wavefront &wf,
                                                           CarryOp carry_op) {
  return try_execute_binary_vop3_cin_result_simd(
      inst, wf, carry_op, [&](uint64_t result) { write_wave_mask_scalar(inst.sdst, wf, result); });
}

/// Common observed word access for local true16 and packed conversion paths.
/// The callback always sees unsigned words; it owns arithmetic and narrowing.
/// Partial-byte stores preserve the other half without reporting a false read.
template <unsigned Arity, bool E32, bool HalfDst, unsigned HalfInputs, typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_words_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst))
    return false;
  auto canonical = [](const auto &operand) {
    if constexpr (E32)
      return e32_word_operand(operand);
    else
      return operand;
  };
  auto first = canonical(inst.src0);
  auto destination = canonical(inst.vdst);
  using OperandT = decltype(first);
  std::optional<OperandT> second, third;
  if constexpr (Arity >= 2) {
    if constexpr (requires { inst.src1; })
      second.emplace(canonical(inst.src1));
    else
      second.emplace(canonical(inst.vsrc1));
  }
  if constexpr (Arity >= 3)
    third.emplace(canonical(inst.src2));
  if (!first.simd_capable() || !destination.simd_capable() || (second && !second->simd_capable()) ||
      (third && !third->simd_capable()))
    return false;
  uint32_t selectors = 0;
  if constexpr (E32) {
    auto high = [](const auto &operand) {
      return e32_packed_vgpr_operand(operand) && (operand.encoding_value() & 0x80);
    };
    selectors = high(inst.src0) ? 1 : 0;
    if constexpr (Arity >= 2) {
      if constexpr (requires { inst.src1; })
        selectors |= high(inst.src1) ? 2 : 0;
      else
        selectors |= high(inst.vsrc1) ? 2 : 0;
    }
    selectors |= (inst.inst_.vdst & 0x80) ? 8 : 0;
  } else if constexpr (HalfDst || HalfInputs) {
    selectors = vop3_opsel(inst.inst_);
  }
  using U = util::native<uint32_t>;
  constexpr uint32_t W = U::size();
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  auto bytes = [&](unsigned i) -> uint8_t {
    return (HalfInputs & (1u << i)) ? ((selectors & (1u << i)) ? 0xc : 0x3) : 0xf;
  };
  RegisterAccess regs(wf);
  auto a = regs.read_operand(first, exec, bytes(0));
  std::optional<RegisterAccess::OperandReadView> b, c;
  if constexpr (Arity >= 2)
    b.emplace(regs.read_operand(*second, exec, bytes(1)));
  if constexpr (Arity >= 3)
    c.emplace(regs.read_operand(*third, exec, bytes(2)));
  const bool high_dst = HalfDst && (selectors & 8);
  const bool zero_high = HalfDst && !E32 && !high_dst && cdna_vop3_low_dst_zeroes_high(wf);
  auto dst =
      regs.write_operand(destination, exec, HalfDst && !zero_high ? (high_dst ? 0xc : 3) : 0xf);
  for (uint32_t base = 0; base < wf.wf_size(); base += W) {
    const uint64_t chunk = (exec >> base) & util::mask<uint64_t>(W);
    if (!chunk)
      continue;
    auto load = [&](const auto &view, unsigned i) {
      U v = view.template load_native<uint32_t>(base);
      if (HalfInputs & (1u << i))
        v = (v >> ((selectors & (1u << i)) ? 16 : 0)) & U(0xffffu);
      return v;
    };
    U result;
    if constexpr (Arity == 1)
      result = op(load(a, 0));
    else if constexpr (Arity == 2) {
      if constexpr (requires { op(load(a, 0), load(*b, 1), base); })
        result = op(load(a, 0), load(*b, 1), base);
      else
        result = op(load(a, 0), load(*b, 1));
    } else
      result = op(load(a, 0), load(*b, 1), load(*c, 2));
    if constexpr (HalfDst)
      result = (result & U(0xffffu)) << (high_dst ? 16 : 0);
    dst.template store_native<uint32_t>(base, result, chunk);
  }
  return true;
}

template <unsigned Arity, bool E32, bool HalfDst, unsigned HalfInputs, typename Inst, typename Op>
[[nodiscard]] bool try_execute_words_simd(Inst &, Wavefront &, Op) {
  return false;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP3_INTEGER_H_
