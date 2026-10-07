// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP1_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP1_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd/common.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/portable_math.h"

namespace rocjitsu::amdgpu {

/// VOP1 unary SIMD fast path. Reads `src0` as `Tin`, applies `un_op`
/// (`native<Tin> -> native<Tout>`), masked-stores the result to `vdst` as
/// `Tout`. `Tin` and `Tout` are both 32-bit lane types (possibly different,
/// e.g. int32->float32 for v_cvt_f32_i32). Same contract as the VOP2 path:
/// returns true when the SIMD path executed; false on the `force_scalar`
/// override or when either operand reports `!simd_capable()`.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop1_simd(Inst &inst, Wavefront &wf, UnOp un_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<Tout>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<Tin>(base);
    const auto r = un_op(a);
    dst.template store_native<Tout>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the unary path; see the binary-path note above.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop1_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// 64-bit-lane VOP1 unary SIMD fast path. The 64-bit counterpart of
/// try_execute_unary_vop1_simd: reads src0 as `native<T>` through a 64-bit
/// RegisterAccess operand view, applies `un_op` (`native<T> ->
/// native<T>`), and masked-stores the result to vdst. `T` is `double` for the
/// f64 math ops (ceil/floor/trunc/rndne/fract/rcp/rsq/sqrt) and `uint64_t` for
/// the pure 64-bit move (v_mov_b64). Same contract as the other paths: returns
/// true when the SIMD path executed; false on the `force_scalar` override or
/// when either operand reports `!simd_capable()`.
///
/// The rounding ops map to `vroundpd`, sqrt to `vsqrtpd`, and `1.0 / x` to
/// `vdivpd` — all correctly-rounded IEEE operations, bit-identical to the scalar
/// `std::ceil`/`std::sqrt`/... for every finite and infinite input. NaN-*input*
/// lanes may differ in propagated NaN payload (accepted — the result is a NaN
/// either way), guarded by the UtilSimd.*F64*_VectorMatchesScalar_BitExact tests.
template <typename T, typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop1_f64_simd(Inst &inst, Wavefront &wf, UnOp un_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    dst.template store_native<T>(base, un_op(a), chunk);
  }
  return true;
}

/// Unconstrained fallback for the f64 unary path; see the binary-path note.
template <typename T, typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop1_f64_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// Mixed-width VOP1 conversion SIMD fast path, f64 source -> 32-bit dst
/// (v_cvt_f32_f64, v_cvt_i32_f64, v_cvt_u32_f64). Reads src0 as `native<double>`
/// through a 64-bit RegisterAccess operand view, applies `cvt_op`
/// (`native<double> -> narrow32<Tout>`), and masked-stores the `native_width64`
/// 32-bit results to vdst. The double->Tout step is a single `static_simd_cast`
/// (cvt_f32_f64 maps to vcvtpd2ps, correctly rounded; the int forms clamp/NaN in
/// the double domain first, then one cast). Bit-identical to the scalar body for
/// finite/Inf inputs; a NaN *result* may differ in payload (accepted), which the
/// A/B test skips. Returns true when the SIMD path executed.
template <typename Tout, typename Inst, typename CvtOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cvt_f64_to_b32_simd(Inst &inst, Wavefront &wf, CvtOp cvt_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto dst = regs.write_operand(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = src0.template load_native<double>(base);
    const util::narrow32<Tout> r = cvt_op(s);
    dst.template store_narrow<Tout>(base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the f64->b32 cvt path; see the binary-path note.
template <typename Tout, typename Inst, typename CvtOp>
[[nodiscard]] bool try_execute_cvt_f64_to_b32_simd(Inst &, Wavefront &, CvtOp) {
  return false;
}

/// Mixed-width VOP1 conversion SIMD fast path, 32-bit source -> f64 dst
/// (v_cvt_f64_f32, v_cvt_f64_i32, v_cvt_f64_u32). Reads src0 as `narrow32<Tin>`
/// (native_width64 32-bit lanes), applies `cvt_op` (`narrow32<Tin> ->
/// native<double>`), and masked-stores the result to the 64-bit vdst through a
/// 64-bit RegisterAccess write view. Each conversion is an exact widening
/// `static_simd_cast` (vcvtps2pd / int->double), bit-identical to the scalar body
/// for every input. Returns true when the SIMD path executed.
template <typename Tin, typename Inst, typename CvtOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cvt_b32_to_f64_simd(Inst &inst, Wavefront &wf, CvtOp cvt_op) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto dst = regs.write_operand64(inst.vdst, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto in = src0.template load_narrow<Tin>(base);
    dst.template store_native<double>(base, cvt_op(in), chunk);
  }
  return true;
}

/// Unconstrained fallback for the b32->f64 cvt path; see the binary-path note.
template <typename Tin, typename Inst, typename CvtOp>
[[nodiscard]] bool try_execute_cvt_b32_to_f64_simd(Inst &, Wavefront &, CvtOp) {
  return false;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOP1_H_
