// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOPC_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOPC_H_

#include "rocjitsu/isa/arch/amdgpu/shared/simd/common.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd/portable_math.h"

namespace rocjitsu::amdgpu {

/// VOPC compare SIMD fast path: per active EXEC lane, `cmp_op(src0, vsrc1)`
/// produces a `simd_mask` whose bit is packed into VCC at the lane position;
/// inactive-lane VCC bits are preserved (mirroring the scalar body, which
/// flips only active-lane bits). VOPC writes VCC only — there is no vdst
/// operand and CDNA4 has no v_cmpx (EXEC-writing) form, so this single shape
/// covers every compare. `T` is the 32-bit lane read type (uint32_t raw
/// encodings for the f16/f32 relations, int32_t/uint32_t for the integer ones);
/// the f16 and 16-bit integer relations read as 32-bit lanes and narrow/convert
/// inside the functor. The VCC merge is identical to the carry path's.
///
/// Float relations never see host floats: their functors call
/// comparison::evaluate on the raw encodings with the captured MODE
/// input-denormal policy, the same evaluation the scalar body uses, so host
/// DAZ cannot alter a lane and the compares are bit-exact for every input
/// including NaN/Inf/±0/subnormals (no accepted-divergence carve-out, unlike
/// fma / min-max).
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                WriteResult write_result) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t vcc = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.vsrc1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t cmp_bits = util::simd_mask_to_bits(cmp_op(a, b));
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(vcc);
  return true;
}

/// Unconstrained fallback for the VOPC path; see the binary-path note above.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

template <typename T, typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc_simd(Inst &inst, Wavefront &wf, CmpOp op) {
  return try_execute_vopc_simd<T>(inst, wf, op, [&wf](uint64_t value) { wf.set_vcc_mask(value); });
}

/// 64-bit-lane VOPC compare SIMD fast path (f64/i64/u64 relations). Identical to
/// try_execute_vopc_simd but reads each operand as `native<T>` (T = uint64_t
/// raw encodings for f64, int64_t / uint64_t for the integer relations) through
/// 64-bit RegisterAccess operand views, so it processes `native_width64` lanes
/// per chunk. f64 functors evaluate through comparison::evaluate as above.
/// Same VCC merge.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc64_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                  WriteResult write_result) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t vcc = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.vsrc1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t cmp_bits = cmp_bits64<T>(a, b, cmp_op);
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(vcc);
  return true;
}

/// Unconstrained fallback for the 64-bit VOPC path; see the binary-path note.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc64_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

template <typename T, typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc64_simd(Inst &inst, Wavefront &wf, CmpOp op) {
  return try_execute_vopc64_simd<T>(inst, wf, op,
                                    [&wf](uint64_t value) { wf.set_vcc_mask(value); });
}

/// Mixed-width v_cmp_class_f64 SIMD fast path. v_cmp_class_f64 tests a 64-bit f64
/// src0 against a 32-bit class mask in vsrc1, so unlike the relational VOPC64 path
/// the two operands have different widths: src0 is read as `native<uint64_t>` raw
/// bits through a 64-bit RegisterAccess operand view, and the per-lane
/// mask as a `native_width64`-wide `narrow32<uint32_t>`. The functor
/// classifies the f64 from its raw bits and tests the class against the mask,
/// returning a `native_width64`-wide mask packed into VCC exactly like the other
/// VOPC paths (active lanes only, inactive bits preserved). The classification is
/// pure bit decode, bit-exact with the scalar body for every input.
template <typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_class_f64_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                          WriteResult write_result) {
  if (simd_force_scalar() || !sdwa::supports_direct_simd_store(inst) || !inst.src0.simd_capable() ||
      !inst.vsrc1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t vcc = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto mask_src = regs.read_operand(inst.vsrc1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = src0.template load_native<uint64_t>(base);
    const auto mask = mask_src.template load_narrow<uint32_t>(base);
    const uint64_t cmp_bits = cmp_class_f64_bits(s, mask, cmp_op);
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(vcc);
  return true;
}

/// Unconstrained fallback for the f64 class path; see the binary-path note.
template <typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc_class_f64_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

template <typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc_class_f64_simd(Inst &inst, Wavefront &wf, CmpOp op) {
  return try_execute_vopc_class_f64_simd(inst, wf, op,
                                         [&wf](uint64_t value) { wf.set_vcc_mask(value); });
}

/// VOP3 v_cmp_class_f16/f32 SIMD fast path (32-bit value). The VOP3 form differs
/// from the VOPC form in three ways, all handled here: (1) the raw result is
/// passed to a caller-provided architectural commit; (2) the per-instruction
/// `abs`/`neg` source modifiers are applied
/// to src0's selected raw bits before classification; `signmask` is passed per op
/// (0x8000 for f16, 0x80000000 for f32, since both share a uint32 lane); (3) the
/// class mask is read from `inst.src1`, not `inst.vsrc1`. In true16 mode, f16
/// VOP3 inputs use op_sel to select each source half before the classify functor
/// sees the value and mask.
template <bool True16, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3_class_b32_simd(Inst &inst, Wavefront &wf,
                                                          uint32_t signmask, CmpOp cmp_op,
                                                          WriteResult write_result) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const bool do_abs = (inst.inst_.abs & (1u << 0)) != 0;
  const bool do_neg = (inst.inst_.neg & (1u << 0)) != 0;
  const bool true16 = True16 && signmask == 0x8000u;
  const uint32_t opsel = true16 ? vop3_opsel(inst.inst_) : 0u;
  const auto sm = util::broadcast<T>(signmask);
  uint64_t vcc = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = src0.template load_native<T>(base);
    auto b = src1.template load_native<T>(base);
    if (true16) {
      a = select_vop3_true16_src(a, opsel, 0);
      b = select_vop3_true16_src(b, opsel, 1);
    }
    if (do_abs)
      a = a & ~sm;
    if (do_neg)
      a = a ^ sm;
    const uint64_t cmp_bits = util::simd_mask_to_bits(cmp_op(a, b));
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(vcc);
  return true;
}

/// Unconstrained fallback for the VOP3 b32 class path; see the binary-path note.
template <bool True16, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vop3_class_b32_simd(Inst &, Wavefront &, uint32_t, CmpOp,
                                                   WriteResult) {
  return false;
}

/// VOP3 v_cmp_class_f64 SIMD fast path. The 64-bit-value counterpart of
/// try_execute_vop3_class_b32_simd: src0 is read as `native<uint64_t>` raw bits
/// through a 64-bit RegisterAccess operand view and the class mask as a
/// `narrow32<uint32_t>` from `inst.src1`. The packed result is passed to the
/// generated wrapper's result writer, which owns the final SGPR/EXEC merge.
/// abs/neg are applied to the 64-bit raw bits (signmask 0x8000000000000000),
/// using the same classify functor as the VOPC f64 class path.
template <typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3_class_f64_simd(Inst &inst, Wavefront &wf,
                                                          uint64_t signmask, CmpOp cmp_op,
                                                          WriteResult write_result) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  const bool do_abs = (inst.inst_.abs & (1u << 0)) != 0;
  const bool do_neg = (inst.inst_.neg & (1u << 0)) != 0;
  const auto sm = util::broadcast64<uint64_t>(signmask);
  uint64_t vcc = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto mask_src = regs.read_operand(inst.src1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto s = src0.template load_native<uint64_t>(base);
    if (do_abs)
      s = s & ~sm;
    if (do_neg)
      s = s ^ sm;
    const auto mask = mask_src.template load_narrow<uint32_t>(base);
    const uint64_t cmp_bits = cmp_class_f64_bits(s, mask, cmp_op);
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(vcc);
  return true;
}

/// Unconstrained fallback for the VOP3 f64 class path; see the binary-path note.
template <typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vop3_class_f64_simd(Inst &, Wavefront &, uint64_t, CmpOp,
                                                   WriteResult) {
  return false;
}

/// VOP3 raw-lane VOPC compare SIMD fast path (32-bit lane). The VOP3 form
/// of v_cmp_<rel>_<i16|u16|i32|u32|f16|f32> reads src0/src1 (not src0/vsrc1) and
/// writes the per-lane compare result into an arbitrary wave-mask scalar
/// destination through a caller-provided architectural commit. The functor
/// receives the raw source encodings: integer relations use no modifiers, and
/// float relations apply their captured abs/neg fields in shared/comparison.h.
/// The raw result contains active EXEC lanes only. Returns true when the SIMD
/// path executed.
template <typename T, bool True16 = false, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_vop3_int_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                         WriteResult write_result) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t dst = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand(inst.src0, exec);
  auto src1 = regs.read_operand(inst.src1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = src0.template load_native<T>(base);
    auto b = src1.template load_native<T>(base);
    if constexpr (True16) {
      a = select_vop3_true16_src(a, vop3_opsel(inst.inst_), 0);
      b = select_vop3_true16_src(b, vop3_opsel(inst.inst_), 1);
    }
    const uint64_t cmp_bits = util::simd_mask_to_bits(cmp_op(a, b));
    dst = (dst & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(dst);
  return true;
}

/// Unconstrained fallback for the VOP3 integer VOPC path; see the binary-path note.
template <typename T, bool True16 = false, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc_vop3_int_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

/// 64-bit-lane VOP3 raw-lane VOPC compare SIMD fast path (i64/u64/f64).
/// Identical to try_execute_vopc_vop3_int_simd but reads each operand as
/// `native<T>` (T = int64_t / uint64_t) through 64-bit RegisterAccess operand
/// views, so it processes `native_width64` lanes per chunk. The caller performs
/// the result commit; any float modifiers are applied by the functor.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc64_vop3_int_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op,
                                                           WriteResult write_result) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = dpp::execution_lane_mask(inst, wf);
  uint64_t dst = 0;
  RegisterAccess regs(wf);
  auto src0 = regs.read_operand64(inst.src0, exec);
  auto src1 = regs.read_operand64(inst.src1, exec);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = src0.template load_native<T>(base);
    const auto b = src1.template load_native<T>(base);
    const uint64_t cmp_bits = cmp_bits64<T>(a, b, cmp_op);
    dst = (dst & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  write_result(dst);
  return true;
}

/// Unconstrained fallback for the 64-bit VOP3 integer VOPC path; see the binary-path note.
template <typename T, typename Inst, typename CmpOp, typename WriteResult>
[[nodiscard]] bool try_execute_vopc64_vop3_int_simd(Inst &, Wavefront &, CmpOp, WriteResult) {
  return false;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_EXECUTE_VOPC_H_
