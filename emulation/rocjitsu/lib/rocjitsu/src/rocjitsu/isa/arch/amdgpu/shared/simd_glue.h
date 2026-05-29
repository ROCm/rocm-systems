// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Operand-aware SIMD glue for the auto-generated execute_<mnemonic>
// kernels in execute_shared.h. Hand-maintained; lives separately from
// the generated header so the same code is not duplicated in
// simd_codegen.py (raw string) and execute_shared.h (emitted output).
//
// Layering: this header sees rocjitsu types (Wavefront, plus the Op /
// Inst template parameters) and bridges to the generic util SIMD
// primitives in util/simd.h. The generic util layer never depends on
// rocjitsu; only this direction is permitted.

#ifndef ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
#define ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_

#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd.h"

#include <cstddef>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// Explicit-width alias for IEEE-754 binary32. C++23 has std::float32_t
/// in <stdfloat>; rocjitsu is on C++20 so a local alias.
using float32_t = float;

/// Per-thread runtime override that disables the SIMD fast path in
/// kernels that have one. Forwards to util::force_scalar().
inline bool &simd_force_scalar() { return util::force_scalar(); }

/// In-vector VOP3 source modifier (f32), bit-exact with the scalar lambda the
/// generated bodies emit per source: `abs` first (`std::fabs`), then `neg`
/// (`-x`). `abs`/`neg` are the raw VOP3 modifier fields; the bit for source
/// index `SrcIdx` selects whether the modifier applies. std::fabs clears the
/// sign bit and unary minus flips it (both NaN-payload preserving), so the
/// vector form is a pure sign-bit AND/XOR — bit-identical on every input.
template <unsigned SrcIdx>
inline util::native<float> apply_vop3_src_mod_f32(util::native<float> v, uint32_t abs,
                                                  uint32_t neg) {
  using U = util::native<uint32_t>;
  U b = std::bit_cast<U>(v);
  if (abs & (1u << SrcIdx))
    b = b & 0x7FFFFFFFu;
  if (neg & (1u << SrcIdx))
    b = b ^ 0x80000000u;
  return std::bit_cast<util::native<float>>(b);
}

/// In-vector VOP3 source modifier (f64), the f64 counterpart of
/// apply_vop3_src_mod_f32: abs first (std::fabs = sign-bit clear), then neg
/// (unary minus = sign-bit flip). Both are sign-bit-only on IEEE binary64, so
/// the vector form is a pure AND/XOR — bit-identical incl. NaN payload,
/// matching the scalar lambda the f64 VOP3 bodies emit.
template <unsigned SrcIdx>
inline util::native<double> apply_vop3_src_mod_f64(util::native<double> v, uint32_t abs,
                                                   uint32_t neg) {
  using U = util::native<uint64_t>;
  U b = std::bit_cast<U>(v);
  if (abs & (1u << SrcIdx))
    b = b & 0x7FFFFFFFFFFFFFFFull;
  if (neg & (1u << SrcIdx))
    b = b ^ 0x8000000000000000ull;
  return std::bit_cast<util::native<double>>(b);
}

/// In-vector VOP3 destination modifier (f64), counterpart of
/// apply_vop3_dst_mod_f32: omod scales by an exact power of two then clamp
/// saturates to [0, 1]. Ordered compares are false for NaN, so NaN passes
/// through unchanged — matches `std::clamp(v, 0.0, 1.0)`.
inline util::native<double> apply_vop3_dst_mod_f64(util::native<double> v, uint32_t omod,
                                                   uint32_t clamp) {
  if (omod == 1)
    v = v * 2.0;
  else if (omod == 2)
    v = v * 4.0;
  else if (omod == 3)
    v = v * 0.5;
  if (clamp) {
    util::stdx::where(v < 0.0, v) = 0.0;
    util::stdx::where(v > 1.0, v) = 1.0;
  }
  return v;
}

/// In-vector VOP3 destination modifier (f32), bit-exact with the scalar tail:
/// `omod` scales by an exact power of two (1->*2, 2->*4, 3->*0.5; IEEE-exact,
/// no rounding), then `clamp` saturates to [0,1]. The clamp uses ordered
/// comparisons (`v < 0`, `v > 1`), which are false for NaN, so NaN passes
/// through unchanged — matching `std::clamp(v, 0.f, 1.f)`.
inline util::native<float> apply_vop3_dst_mod_f32(util::native<float> v, uint32_t omod,
                                                  uint32_t clamp) {
  if (omod == 1)
    v = v * 2.0f;
  else if (omod == 2)
    v = v * 4.0f;
  else if (omod == 3)
    v = v * 0.5f;
  if (clamp) {
    util::stdx::where(v < 0.0f, v) = 0.0f;
    util::stdx::where(v > 1.0f, v) = 1.0f;
  }
  return v;
}

/// SIMD load of an operand at `lane_base`. Returns a contiguous SIMD
/// load when the operand resolves to per-lane VGPR storage; otherwise
/// broadcasts the operand's scalar value. Constrained on
/// `util::has_stdx_simd` so toolchains without `<experimental/simd>`
/// remove this overload from the candidate set entirely.
template <typename T, typename Op>
  requires(util::has_stdx_simd)
inline util::native<T> read_simd(const Op &op, const Wavefront &wf, uint32_t lane_base) {
  static_assert(sizeof(T) == sizeof(uint32_t), "read_simd: T must be a 32-bit lane type");
  const uint32_t *p = SimdAccess::lane_ptr(op, wf, lane_base);
  if (p)
    return util::load<T>(p);
  return util::broadcast<T>(op.read_scalar(wf));
}

/// SIMD store of `v` into an operand at `lane_base`, blending in only
/// the lanes whose bit is set in `mask`. Falls back to per-lane
/// `write_lane_chunk` when the operand is not contiguous VGPR storage.
template <typename T, typename Op>
  requires(util::has_stdx_simd)
inline void write_simd(const Op &op, Wavefront &wf, uint32_t lane_base, util::native<T> v,
                       uint64_t mask) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  constexpr std::size_t W = util::native_width_v<T>;
  if (uint32_t *p = SimdAccess::dst_ptr(op, wf, lane_base)) {
    util::masked_store<T>(p, v, mask);
    return;
  }
  alignas(util::native<T>) uint32_t buf[W];
  util::blit_to_buffer<T>(buf, v);
  op.write_lane_chunk(wf, lane_base, static_cast<uint32_t>(W), buf, mask);
}

/// 64-bit-lane load of an operand at `lane_base` (T is double / uint64_t).
/// Returns a combined `native<T>` from the operand's split lo/hi VGPR pair when
/// it resolves to per-lane storage; otherwise broadcasts the operand's 64-bit
/// scalar value (`read_scalar64`). Constrained on `util::has_stdx_simd`.
template <typename T, typename Op>
  requires(util::has_stdx_simd)
inline util::native<T> read_simd64(const Op &op, const Wavefront &wf, uint32_t lane_base) {
  static_assert(sizeof(T) == sizeof(uint64_t), "read_simd64: T must be a 64-bit lane type");
  auto [lo, hi] = SimdAccess::lane_ptr64(op, wf, lane_base);
  if (lo)
    return util::load64<T>(lo, hi);
  return util::broadcast64<T>(op.read_scalar64(wf));
}

/// 64-bit-lane masked store of `v` into an operand at `lane_base`, writing only
/// the lanes set in `mask`. Falls back to per-lane `write_lane64` when the dst is
/// not contiguous VGPR storage.
template <typename T, typename Op>
  requires(util::has_stdx_simd)
inline void write_simd64(const Op &op, Wavefront &wf, uint32_t lane_base, util::native<T> v,
                         uint64_t mask) {
  static_assert(sizeof(T) == sizeof(uint64_t));
  constexpr std::size_t W = util::native_width64;
  auto [lo, hi] = SimdAccess::dst_ptr64(op, wf, lane_base);
  if (lo) {
    util::masked_store64<T>(lo, hi, v, mask);
    return;
  }
  alignas(util::native<T>) uint64_t buf[W];
  util::stdx::native_simd<uint64_t> bits = [&] {
    if constexpr (std::is_same_v<T, uint64_t>)
      return v;
    else
      return std::bit_cast<util::stdx::native_simd<uint64_t>>(v);
  }();
  bits.copy_to(buf, util::stdx::vector_aligned);
  for (std::size_t i = 0; i < W; ++i)
    if (mask & (1ULL << i))
      op.write_lane64(wf, lane_base + static_cast<uint32_t>(i), buf[i]);
}

/// VOP2 binary SIMD fast path. Returns true when the SIMD path executed
/// and the caller should skip its scalar per-lane loop; false on the
/// `force_scalar` override or when any operand reports `!simd_capable()`.
/// Constrained on `util::has_stdx_simd`; an unconstrained overload
/// below returns false unconditionally when the constraint cannot be
/// satisfied, so callers can write the probe without an `if constexpr`
/// guard.
template <typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop2_simd(Inst &inst, Wavefront &wf, BinOp bin_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vsrc1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.vsrc1, wf, base);
    write_simd<T>(inst.vdst, wf, base, bin_op(a, b), chunk);
  }
  return true;
}

/// Unconstrained fallback selected when `util::has_stdx_simd` is false.
/// Trivially inlined to `return false;` so the generated probe at the
/// call site costs nothing on toolchains without `<experimental/simd>`.
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] inline bool try_execute_binary_vop2_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// Re-type a `simd_mask` (e.g. the result of a float comparison) to the mask
/// type of `native<To>`, so it can drive `util::stdx::where` on a `native<To>`
/// value. Needed by the clamp/NaN cvt-to-int functors, which compute masks in
/// the float domain but blend into an int result. Wraps the libstdc++
/// `__proposed` mask cast in one place; `<experimental/simd>` is libstdc++-only
/// so the dependency is acceptable.
template <typename To, typename Mask>
  requires(util::has_stdx_simd)
inline auto simd_mask_as(const Mask &m) {
  return util::stdx::__proposed::static_simd_cast<util::native<To>>(m);
}

/// VOP1 unary SIMD fast path. Reads `src0` as `Tin`, applies `un_op`
/// (`native<Tin> -> native<Tout>`), masked-stores the result to `vdst` as
/// `Tout`. `Tin` and `Tout` are both 32-bit lane types (possibly different,
/// e.g. int32->float32 for v_cvt_f32_i32). Same contract as the VOP2 path:
/// returns true when the SIMD path executed; false on the `force_scalar`
/// override or when either operand reports `!simd_capable()`.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop1_simd(Inst &inst, Wavefront &wf, UnOp un_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<Tout>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<Tin>(inst.src0, wf, base);
    write_simd<Tout>(inst.vdst, wf, base, un_op(a), chunk);
  }
  return true;
}

/// Unconstrained fallback for the unary path; see the binary-path note above.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
[[nodiscard]] inline bool try_execute_unary_vop1_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// Result of a carry-bearing VOP2 functor: the 32-bit per-lane result and the
/// per-lane carry/borrow as a `simd_mask`. A class template (not a fixed type)
/// so it never names `native<uint32_t>::mask_type` outside the SIMD build —
/// the carry functors below build it through `make_simd_carry`, whose return
/// type is deduced and only instantiated on the constrained code path.
template <typename Value, typename Mask> struct SimdCarry {
  Value value;
  Mask carry;
};

/// Deduce-and-wrap helper for the carry functors. Keeps each functor a single
/// expression while leaving the mask type implicit.
template <typename Value, typename Mask>
inline SimdCarry<Value, Mask> make_simd_carry(Value value, Mask carry) {
  return {value, carry};
}

/// VOP2 carry SIMD fast path (v_add_co/sub_co/subrev_co/addc/subb/subbrev_u32).
/// The lane type is fixed to uint32_t. `carry_op` is invoked as
///   carry_op(native<uint32_t> src0, native<uint32_t> vsrc1, native<uint32_t> cin)
///     -> SimdCarry<native<uint32_t>, mask>
/// where `cin` carries the incoming VCC bit (0/1 per lane); ops without a
/// carry-in ignore it. The result is masked-stored to vdst and the carry mask
/// is merged into VCC at the chunk's bit offset for active EXEC lanes only —
/// inactive-lane VCC bits are preserved, matching the scalar bodies.
template <typename Inst, typename CarryOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop2_carry_simd(Inst &inst, Wavefront &wf,
                                                             CarryOp carry_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vsrc1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  // Carry-in reads the incoming VCC; the result accumulates into a copy. Each
  // lane touches only its own VCC bit, so the snapshot/copy split mirrors the
  // scalar body, which reads wf.vcc() for carry-in and writes set_vcc() once.
  const uint64_t vcc_in = wf.vcc();
  uint64_t vcc_out = vcc_in;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.vsrc1, wf, base);
    // Expand the incoming VCC bits for this chunk to a 0/1-per-lane vector.
    const uint64_t cin_bits = (vcc_in >> base) & chunk_full;
    alignas(util::native<T>) uint32_t cinbuf[W];
    for (std::size_t i = 0; i < W; ++i)
      cinbuf[i] = static_cast<uint32_t>((cin_bits >> i) & 1u);
    const auto cin = util::load<T>(cinbuf);
    const auto r = carry_op(a, b, cin);
    write_simd<T>(inst.vdst, wf, base, r.value, chunk);
    // Pack the per-lane carry mask into the low W bits, then merge into VCC for
    // active lanes only (clear active bits, set from carry; preserve the rest).
    uint64_t carry_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (r.carry[i])
        carry_bits |= (1ULL << i);
    vcc_out = (vcc_out & ~(chunk << base)) | ((carry_bits & chunk) << base);
  }
  wf.set_vcc(vcc_out);
  return true;
}

/// Unconstrained fallback for the carry path; see the binary-path note above.
template <typename Inst, typename CarryOp>
[[nodiscard]] inline bool try_execute_binary_vop2_carry_simd(Inst &, Wavefront &, CarryOp) {
  return false;
}

/// VOP2 ternary (fused multiply-add) SIMD fast path. Covers the three operand
/// shapes of the VOP2 FMA/MAC/MAD family:
///   * dst-accumulate (v_fmac/v_mac):  dst = fma(src0, vsrc1, dst)
///   * literal addend (v_fmaak/v_madak): dst = fma(src0, vsrc1, K)
///   * literal multiplier (v_fmamk/v_madmk): dst = fma(src0, K, vsrc1)
/// All read src0/vsrc1/vdst as `native<T>` and receive the broadcast inline
/// literal `k` (zero for forms without one). `fma_op(s0, s1, dvst, k)` selects
/// the shape and, for f16, does the f16<->f32 conversions in the functor. The
/// result is masked-stored to vdst.
///
/// `util::stdx::fma` is bit-identical to the scalar `std::fma` for all finite
/// and infinite inputs (including Inf*0 -> NaN). When an *input* is NaN the
/// packed and scalar FMA may propagate a different NaN operand (a toolchain-
/// dependent payload, observed on g++-13/AVX-512); that NaN-payload divergence
/// is accepted — the result is a NaN either way. The finite/Inf bit-exactness
/// the fast path relies on is guarded by UtilSimd.Fma_VectorMatchesScalar_*.
template <typename T, typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop2_simd(Inst &inst, Wavefront &wf,
                                                        util::native<T> k, FmaOp fma_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vsrc1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.vsrc1, wf, base);
    const auto d = read_simd<T>(inst.vdst, wf, base); // dst-accumulate source
    write_simd<T>(inst.vdst, wf, base, fma_op(a, b, d, k), chunk);
  }
  return true;
}

/// Unconstrained fallback for the ternary path; see the binary-path note above.
template <typename T, typename Inst, typename FmaOp>
[[nodiscard]] inline bool try_execute_ternary_vop2_simd(Inst &, Wavefront &, util::native<T>,
                                                        FmaOp) {
  return false;
}

/// 64-bit-lane VOP2 fused-multiply-add SIMD fast path (v_fmac_f64): the only
/// f64 VOP2 op reachable on CDNA4 is the dst-accumulate form
/// `dst = fma(src0, vsrc1, dst)`. Reads all three operands as `native<T>`
/// (T = double) through the split lo/hi VGPR-pair path and masked-stores the
/// result. `fma_op(s0, s1, dvst)` is the dst-accumulate functor.
///
/// `util::stdx::fma` over `native<double>` is bit-identical to the scalar
/// `std::fma` for all finite and infinite inputs; NaN-*input* lanes may differ
/// in propagated NaN payload (accepted — the result is a NaN either way),
/// guarded by UtilSimd.FmaF64_VectorMatchesScalar_BitExact.
template <typename T, typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop2_f64_simd(Inst &inst, Wavefront &wf,
                                                            FmaOp fma_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vsrc1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd64<T>(inst.src0, wf, base);
    const auto b = read_simd64<T>(inst.vsrc1, wf, base);
    const auto d = read_simd64<T>(inst.vdst, wf, base); // dst-accumulate source
    write_simd64<T>(inst.vdst, wf, base, fma_op(a, b, d), chunk);
  }
  return true;
}

/// Unconstrained fallback for the f64 ternary path; see the binary-path note.
template <typename T, typename Inst, typename FmaOp>
[[nodiscard]] inline bool try_execute_ternary_vop2_f64_simd(Inst &, Wavefront &, FmaOp) {
  return false;
}

/// 64-bit-lane VOP1 unary SIMD fast path. The 64-bit counterpart of
/// try_execute_unary_vop1_simd: reads src0 as `native<T>` through the split
/// lo/hi VGPR-pair path (read_simd64), applies `un_op` (`native<T> ->
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
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd64<T>(inst.src0, wf, base);
    write_simd64<T>(inst.vdst, wf, base, un_op(a), chunk);
  }
  return true;
}

/// Unconstrained fallback for the f64 unary path; see the binary-path note.
template <typename T, typename Inst, typename UnOp>
[[nodiscard]] inline bool try_execute_unary_vop1_f64_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// Narrow (native_width64-wide) load of a 32-bit operand at `lane_base`. The
/// 32-bit-source counterpart of `read_simd` for the f64<->32-bit conversion glue,
/// which processes `native_width64` (8 on AVX-512) lanes per chunk to stay aligned
/// with the 64-bit f64 side. Returns a contiguous narrow load from per-lane VGPR
/// storage, else broadcasts the operand's scalar value.
template <typename T, typename Op>
  requires(util::has_stdx_simd)
inline util::narrow32<T> read_narrow(const Op &op, const Wavefront &wf, uint32_t lane_base) {
  static_assert(sizeof(T) == sizeof(uint32_t), "read_narrow: T must be a 32-bit lane type");
  const uint32_t *p = SimdAccess::lane_ptr(op, wf, lane_base);
  if (p)
    return util::load_narrow<T>(p);
  return util::broadcast_narrow<T>(op.read_scalar(wf));
}

/// Mixed-width VOP1 conversion SIMD fast path, f64 source -> 32-bit dst
/// (v_cvt_f32_f64, v_cvt_i32_f64, v_cvt_u32_f64). Reads src0 as `native<double>`
/// through the split lo/hi VGPR-pair path (read_simd64), applies `cvt_op`
/// (`native<double> -> narrow32<Tout>`), and masked-stores the `native_width64`
/// 32-bit results to vdst. The double->Tout step is a single `static_simd_cast`
/// (cvt_f32_f64 maps to vcvtpd2ps, correctly rounded; the int forms clamp/NaN in
/// the double domain first, then one cast). Bit-identical to the scalar body for
/// finite/Inf inputs; a NaN *result* may differ in payload (accepted), which the
/// A/B test skips. Returns true when the SIMD path executed.
template <typename Tout, typename Inst, typename CvtOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cvt_f64_to_b32_simd(Inst &inst, Wavefront &wf, CvtOp cvt_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = read_simd64<double>(inst.src0, wf, base);
    const util::narrow32<Tout> r = cvt_op(s);
    if (uint32_t *p = SimdAccess::dst_ptr(inst.vdst, wf, base)) {
      util::masked_store_narrow<Tout>(p, r, chunk);
      continue;
    }
    // Non-VGPR dst: spill the narrow result to a uint32 chunk buffer (per-lane
    // bit_cast — fixed_size_simd is not bit_castable as a whole) and fall back
    // to the operand's write_lane_chunk.
    alignas(util::narrow32<Tout>) Tout vals[W];
    r.copy_to(vals, util::stdx::vector_aligned);
    uint32_t buf[W];
    for (std::size_t i = 0; i < W; ++i)
      buf[i] = std::bit_cast<uint32_t>(vals[i]);
    inst.vdst.write_lane_chunk(wf, base, static_cast<uint32_t>(W), buf, chunk);
  }
  return true;
}

/// Unconstrained fallback for the f64->b32 cvt path; see the binary-path note.
template <typename Tout, typename Inst, typename CvtOp>
[[nodiscard]] inline bool try_execute_cvt_f64_to_b32_simd(Inst &, Wavefront &, CvtOp) {
  return false;
}

/// Mixed-width VOP1 conversion SIMD fast path, 32-bit source -> f64 dst
/// (v_cvt_f64_f32, v_cvt_f64_i32, v_cvt_f64_u32). Reads src0 as `narrow32<Tin>`
/// (native_width64 32-bit lanes), applies `cvt_op` (`narrow32<Tin> ->
/// native<double>`), and masked-stores the result to the 64-bit vdst through the
/// split lo/hi VGPR-pair path (write_simd64). Each conversion is an exact widening
/// `static_simd_cast` (vcvtps2pd / int->double), bit-identical to the scalar body
/// for every input. Returns true when the SIMD path executed.
template <typename Tin, typename Inst, typename CvtOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cvt_b32_to_f64_simd(Inst &inst, Wavefront &wf, CvtOp cvt_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto in = read_narrow<Tin>(inst.src0, wf, base);
    write_simd64<double>(inst.vdst, wf, base, cvt_op(in), chunk);
  }
  return true;
}

/// Unconstrained fallback for the b32->f64 cvt path; see the binary-path note.
template <typename Tin, typename Inst, typename CvtOp>
[[nodiscard]] inline bool try_execute_cvt_b32_to_f64_simd(Inst &, Wavefront &, CvtOp) {
  return false;
}

/// v_cndmask_b32 SIMD fast path: dst[lane] = (VCC bit) ? vsrc1 : src0. VCC is an
/// input side-channel here (no carry-out). The per-lane select bits for a chunk
/// are read from VCC at the chunk's bit offset, expanded to a 0/1-per-lane
/// vector, and used to blend src0/vsrc1 with `where`. A pure 32-bit bit select,
/// so the result is bit-identical to the scalar body for every input.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cndmask_vop2_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vsrc1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const uint64_t vcc = wf.vcc();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.vsrc1, wf, base);
    const uint64_t sel_bits = (vcc >> base) & chunk_full;
    alignas(util::native<T>) uint32_t selbuf[W];
    for (std::size_t i = 0; i < W; ++i)
      selbuf[i] = static_cast<uint32_t>((sel_bits >> i) & 1u);
    auto r = a;
    util::stdx::where(util::load<T>(selbuf) != 0u, r) = b;
    write_simd<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the cndmask path; see the binary-path note above.
template <typename Inst>
[[nodiscard]] inline bool try_execute_cndmask_vop2_simd(Inst &, Wavefront &) {
  return false;
}

/// v_cndmask_b32 VOP3 form: dst[lane] = (sel[lane]) ? src1 : src0, where `sel`
/// is the 64-bit value read from the SGPR-pair `src2` (instead of the fixed VCC
/// used by the VOP2 form). The scalar body applies no modifiers (the result is
/// a bit-exact integer select on whichever operand the selector picks), so the
/// VOP3 modifier fields are intentionally not consulted here — bit-identical to
/// the scalar body regardless of abs/neg/omod/clamp encoding. `src2` is an
/// SGPR/inline operand, not a VGPR, so it does not participate in the
/// simd_capable gate; src0/src1/vdst do.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cndmask_vop3_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const uint64_t sel64 = inst.src2.read_scalar64(wf);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.src1, wf, base);
    const uint64_t sel_bits = (sel64 >> base) & chunk_full;
    alignas(util::native<T>) uint32_t selbuf[W];
    for (std::size_t i = 0; i < W; ++i)
      selbuf[i] = static_cast<uint32_t>((sel_bits >> i) & 1u);
    auto r = a;
    util::stdx::where(util::load<T>(selbuf) != 0u, r) = b;
    write_simd<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 cndmask path; see the binary-path note.
template <typename Inst>
[[nodiscard]] inline bool try_execute_cndmask_vop3_simd(Inst &, Wavefront &) {
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
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const uint64_t sel64 = inst.src2.read_scalar64(wf);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.src1, wf, base);
    const uint64_t sel_bits = (sel64 >> base) & chunk_full;
    alignas(util::native<T>) uint32_t selbuf[W];
    for (std::size_t i = 0; i < W; ++i)
      selbuf[i] = static_cast<uint32_t>((sel_bits >> i) & 1u);
    auto r = a;
    util::stdx::where(util::load<T>(selbuf) != 0u, r) = b;
    r = r & util::native<T>(0xFFFFu);
    write_simd<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst>
[[nodiscard]] inline bool try_execute_cndmask_b16_vop3_simd(Inst &, Wavefront &) {
  return false;
}

/// VOPC compare SIMD fast path: per active EXEC lane, `cmp_op(src0, vsrc1)`
/// produces a `simd_mask` whose bit is packed into VCC at the lane position;
/// inactive-lane VCC bits are preserved (mirroring the scalar body, which
/// flips only active-lane bits). VOPC writes VCC only — there is no vdst
/// operand and CDNA4 has no v_cmpx (EXEC-writing) form, so this single shape
/// covers every compare. `T` is the 32-bit lane read type (float32_t for the
/// f32 relations, int32_t/uint32_t for the integer ones); the f16 and 16-bit
/// integer relations also read as 32-bit lanes and narrow/convert inside the
/// functor. The VCC merge is identical to the carry path's.
///
/// Float comparison operators (and stdx::isnan, used by the ordered/unordered
/// relations) produce the same per-lane boolean as the scalar `<`/`==`/isnan,
/// for all inputs including NaN/Inf/±0 — so the compares are bit-exact with no
/// accepted-divergence carve-out (unlike fma / min-max).
template <typename T, typename Inst, typename CmpOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vsrc1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.vsrc1, wf, base);
    const auto m = cmp_op(a, b);
    uint64_t cmp_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (m[i])
        cmp_bits |= (1ULL << i);
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  wf.set_vcc(vcc);
  return true;
}

/// Unconstrained fallback for the VOPC path; see the binary-path note above.
template <typename T, typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc_simd(Inst &, Wavefront &, CmpOp) {
  return false;
}

/// 64-bit-lane VOPC compare SIMD fast path (f64/i64/u64 relations). Identical to
/// try_execute_vopc_simd but reads each operand as `native<T>` (T = double /
/// int64_t / uint64_t) through the split lo/hi VGPR-pair path (read_simd64), so
/// it processes `native_width64` lanes per chunk. Same VCC merge / preservation.
template <typename T, typename Inst, typename CmpOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc64_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vsrc1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd64<T>(inst.src0, wf, base);
    const auto b = read_simd64<T>(inst.vsrc1, wf, base);
    const auto m = cmp_op(a, b);
    uint64_t cmp_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (m[i])
        cmp_bits |= (1ULL << i);
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  wf.set_vcc(vcc);
  return true;
}

/// Unconstrained fallback for the 64-bit VOPC path; see the binary-path note.
template <typename T, typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc64_simd(Inst &, Wavefront &, CmpOp) {
  return false;
}

/// Mixed-width v_cmp_class_f64 SIMD fast path. v_cmp_class_f64 tests a 64-bit f64
/// src0 against a 32-bit class mask in vsrc1, so unlike the relational VOPC64 path
/// the two operands have different widths: src0 is read as `native<uint64_t>` raw
/// bits through the split lo/hi VGPR-pair path (read_simd64), and the per-lane
/// mask as a `native_width64`-wide `narrow32<uint32_t>` (read_narrow). The functor
/// classifies the f64 from its raw bits and tests the class against the mask,
/// returning a `native_width64`-wide mask packed into VCC exactly like the other
/// VOPC paths (active lanes only, inactive bits preserved). The classification is
/// pure bit decode, bit-exact with the scalar body for every input.
template <typename Inst, typename CmpOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_class_f64_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vsrc1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  uint64_t vcc = wf.vcc();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = read_simd64<uint64_t>(inst.src0, wf, base);
    const auto mask = read_narrow<uint32_t>(inst.vsrc1, wf, base);
    const auto m = cmp_op(s, mask);
    uint64_t cmp_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (m[i])
        cmp_bits |= (1ULL << i);
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  wf.set_vcc(vcc);
  return true;
}

/// Unconstrained fallback for the f64 class path; see the binary-path note.
template <typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc_class_f64_simd(Inst &, Wavefront &, CmpOp) {
  return false;
}

/// VOP3 v_cmp_class_f16/f32 SIMD fast path (32-bit value). The VOP3 form differs
/// from the VOPC form in three ways, all handled here: (1) the result merges into
/// an arbitrary SGPR-pair dst via `inst.vdst.read_scalar64`/`write_scalar64`, not
/// the fixed VCC; (2) the per-instruction `abs`/`neg` source modifiers are applied
/// to src0's raw bits before classification — `abs` clears the sign bit
/// (`& ~signmask`), `neg` flips it (`^ signmask`), applied abs-then-neg to match
/// the scalar body's std::fabs/negate (bit-identical incl. NaN); `signmask` is
/// passed per op (0x8000 for f16, 0x80000000 for f32, since both share a uint32
/// lane); (3) the class mask is read from `inst.src1`, not `inst.vsrc1`. The
/// classify functor is identical to the VOPC class functor (it sees the already
/// modified bits). Pure bit decode, bit-exact with the scalar body.
template <typename Inst, typename CmpOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3_class_b32_simd(Inst &inst, Wavefront &wf,
                                                          uint32_t signmask, CmpOp cmp_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const bool do_abs = (inst.inst_.abs & (1u << 0)) != 0;
  const bool do_neg = (inst.inst_.neg & (1u << 0)) != 0;
  const auto sm = util::broadcast<T>(signmask);
  uint64_t vcc = inst.vdst.read_scalar64(wf);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = read_simd<T>(inst.src0, wf, base);
    if (do_abs)
      a = a & ~sm;
    if (do_neg)
      a = a ^ sm;
    const auto b = read_simd<T>(inst.src1, wf, base);
    const auto m = cmp_op(a, b);
    uint64_t cmp_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (m[i])
        cmp_bits |= (1ULL << i);
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  inst.vdst.write_scalar64(wf, vcc);
  return true;
}

/// Unconstrained fallback for the VOP3 b32 class path; see the binary-path note.
template <typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vop3_class_b32_simd(Inst &, Wavefront &, uint32_t, CmpOp) {
  return false;
}

/// VOP3 v_cmp_class_f64 SIMD fast path. The 64-bit-value counterpart of
/// try_execute_vop3_class_b32_simd: src0 is read as `native<uint64_t>` raw bits
/// (read_simd64), the class mask as a `narrow32<uint32_t>` from `inst.src1`, and
/// the result merges into the SGPR-pair dst. abs/neg are applied to the 64-bit raw
/// bits (signmask 0x8000000000000000). Same VCC-style merge / preservation; same
/// classify functor as the VOPC f64 class path.
template <typename Inst, typename CmpOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3_class_f64_simd(Inst &inst, Wavefront &wf,
                                                          uint64_t signmask, CmpOp cmp_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const bool do_abs = (inst.inst_.abs & (1u << 0)) != 0;
  const bool do_neg = (inst.inst_.neg & (1u << 0)) != 0;
  const auto sm = util::broadcast64<uint64_t>(signmask);
  uint64_t vcc = inst.vdst.read_scalar64(wf);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto s = read_simd64<uint64_t>(inst.src0, wf, base);
    if (do_abs)
      s = s & ~sm;
    if (do_neg)
      s = s ^ sm;
    const auto mask = read_narrow<uint32_t>(inst.src1, wf, base);
    const auto m = cmp_op(s, mask);
    uint64_t cmp_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (m[i])
        cmp_bits |= (1ULL << i);
    vcc = (vcc & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  inst.vdst.write_scalar64(wf, vcc);
  return true;
}

/// Unconstrained fallback for the VOP3 f64 class path; see the binary-path note.
template <typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vop3_class_f64_simd(Inst &, Wavefront &, uint64_t, CmpOp) {
  return false;
}

/// VOP3 integer/bitwise binary SIMD fast path. Same shape as
/// try_execute_binary_vop2_simd but reads the VOP3 operands `src0`/`src1`
/// (instead of `src0`/`vsrc1`). The generated integer/bitwise VOP3 bodies apply
/// no source/result modifiers (abs/neg/omod are float-only; clamp on an integer
/// op means saturate, which these wrap-around/bitwise twins do not request), so
/// the plain op is bit-identical to the scalar body on every input. T is a
/// 32-bit integer lane type.
template <typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_simd(Inst &inst, Wavefront &wf, BinOp bin_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.src1, wf, base);
    write_simd<T>(inst.vdst, wf, base, bin_op(a, b), chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 integer binary path; see the VOP2
/// binary-path note above.
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] inline bool try_execute_binary_vop3_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 f32 binary SIMD fast path. Reads `src0`/`src1`, applies the per-source
/// abs/neg modifiers, runs `bin_op`, then applies the result omod/clamp — the
/// exact order of the generated scalar body (abs->neg per source, op,
/// omod->clamp on the result). The modifier helpers are bit-exact, so unlike the
/// VOP2 path this fast path stays correct even when modifiers are set; no bail.
template <typename T, typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_fp_simd(Inst &inst, Wavefront &wf, BinOp bin_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(read_simd<T>(inst.src0, wf, base), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(read_simd<T>(inst.src1, wf, base), abs, neg);
    const auto r = apply_vop3_dst_mod_f32(bin_op(a, b), omod, clamp);
    write_simd<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f32 binary path.
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] inline bool try_execute_binary_vop3_fp_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 integer/bitwise VOPC compare SIMD fast path (32-bit lane). The VOP3 form
/// of v_cmp_<rel>_<i16|u16|i32|u32> reads src0/src1 (not src0/vsrc1) and writes
/// the per-lane compare result into an arbitrary SGPR-pair dst via
/// `inst.vdst.read_scalar64`/`write_scalar64` instead of the fixed VCC. The
/// integer/bitwise scalar bodies apply no source/result modifiers (abs/neg/omod
/// are float-only; clamp on integer is unused here), so the plain functor is
/// bit-identical to the scalar body on every input. Mirrors the VOPC merge:
/// active EXEC lanes only, inactive SGPR-pair bits preserved. Returns true when
/// the SIMD path executed.
template <typename T, typename Inst, typename CmpOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_vop3_int_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  uint64_t dst = inst.vdst.read_scalar64(wf);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.src1, wf, base);
    const auto m = cmp_op(a, b);
    uint64_t cmp_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (m[i])
        cmp_bits |= (1ULL << i);
    dst = (dst & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  inst.vdst.write_scalar64(wf, dst);
  return true;
}

/// Unconstrained fallback for the VOP3 integer VOPC path; see the binary-path note.
template <typename T, typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc_vop3_int_simd(Inst &, Wavefront &, CmpOp) {
  return false;
}

/// 64-bit-lane VOP3 integer/bitwise VOPC compare SIMD fast path (i64/u64).
/// Identical to try_execute_vopc_vop3_int_simd but reads each operand as
/// `native<T>` (T = int64_t / uint64_t) through the split lo/hi VGPR-pair path
/// (read_simd64), so it processes `native_width64` lanes per chunk. Same SGPR-pair
/// merge / preservation. No modifiers.
template <typename T, typename Inst, typename CmpOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc64_vop3_int_simd(Inst &inst, Wavefront &wf,
                                                           CmpOp cmp_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  uint64_t dst = inst.vdst.read_scalar64(wf);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd64<T>(inst.src0, wf, base);
    const auto b = read_simd64<T>(inst.src1, wf, base);
    const auto m = cmp_op(a, b);
    uint64_t cmp_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (m[i])
        cmp_bits |= (1ULL << i);
    dst = (dst & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  inst.vdst.write_scalar64(wf, dst);
  return true;
}

/// Unconstrained fallback for the 64-bit VOP3 integer VOPC path; see the binary-path note.
template <typename T, typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc64_vop3_int_simd(Inst &, Wavefront &, CmpOp) {
  return false;
}

/// VOP3 f32 VOPC compare SIMD fast path. Same SGPR-pair-dst merge as the
/// integer VOP3 path but reads src0/src1 as `native<float>` and applies the
/// per-source abs/neg VOP3 modifiers — bit-identical to the scalar body which
/// does `std::fabs` then unary minus per source before comparing. The compare
/// itself is the existing VOPC f32 functor (omod/clamp are not applied because
/// the compare result is a single bit, not an f32; the scalar bodies for these
/// kernels likewise ignore omod/clamp). NaN handling mirrors the scalar
/// `<`/`==`/etc. exactly. Returns true when the SIMD path executed.
template <typename Inst, typename CmpOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_vop3_fp32_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  using T = float32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  uint64_t dst = inst.vdst.read_scalar64(wf);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(read_simd<T>(inst.src0, wf, base), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(read_simd<T>(inst.src1, wf, base), abs, neg);
    const auto m = cmp_op(a, b);
    uint64_t cmp_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (m[i])
        cmp_bits |= (1ULL << i);
    dst = (dst & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  inst.vdst.write_scalar64(wf, dst);
  return true;
}

/// Unconstrained fallback for the VOP3 f32 VOPC path; see the binary-path note.
template <typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc_vop3_fp32_simd(Inst &, Wavefront &, CmpOp) {
  return false;
}

/// VOP3 f16 VOPC compare SIMD fast path. The scalar body widens each f16 src
/// to f32 (`util::f16_to_f32`) and only then applies abs/neg (std::fabs / unary
/// minus on the f32). The vector path matches that order exactly: read
/// src0/src1 as raw uint32 lanes (low 16 = f16 bits), widen via
/// `util::f16_to_f32_simd`, then apply the f32 modifier helper, then call the
/// compare functor on f32 operands. The compare functor is the same as the f32
/// VOP3 VOPC one (it takes already-widened, already-modified `native<float>`).
/// Bit-identical to the scalar body for every input incl. NaN.
template <typename Inst, typename CmpOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc_vop3_fp16_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  uint64_t dst = inst.vdst.read_scalar64(wf);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(
        util::f16_to_f32_simd(read_simd<T>(inst.src0, wf, base)), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(
        util::f16_to_f32_simd(read_simd<T>(inst.src1, wf, base)), abs, neg);
    const auto m = cmp_op(a, b);
    uint64_t cmp_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (m[i])
        cmp_bits |= (1ULL << i);
    dst = (dst & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  inst.vdst.write_scalar64(wf, dst);
  return true;
}

/// Unconstrained fallback for the VOP3 f16 VOPC path; see the binary-path note.
template <typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc_vop3_fp16_simd(Inst &, Wavefront &, CmpOp) {
  return false;
}

/// VOP3 f64 VOPC compare SIMD fast path. 64-bit-lane counterpart of the f32
/// path: reads src0/src1 as `native<double>` through the split lo/hi VGPR-pair
/// path (read_simd64), applies the per-source abs/neg modifiers in the f64
/// domain (apply_vop3_src_mod_f64; sign-bit AND/XOR — bit-identical incl. NaN
/// payload), and calls the compare functor on `native<double>` operands. The
/// SGPR-pair merge is the same VCC-style pack as the other VOP3 VOPC paths,
/// processed `native_width64` lanes per chunk. Bit-identical to the scalar
/// body for every input.
template <typename Inst, typename CmpOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc64_vop3_fp64_simd(Inst &inst, Wavefront &wf,
                                                            CmpOp cmp_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  uint64_t dst = inst.vdst.read_scalar64(wf);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(read_simd64<T>(inst.src0, wf, base), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(read_simd64<T>(inst.src1, wf, base), abs, neg);
    const auto m = cmp_op(a, b);
    uint64_t cmp_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (m[i])
        cmp_bits |= (1ULL << i);
    dst = (dst & ~(chunk << base)) | ((cmp_bits & chunk) << base);
  }
  inst.vdst.write_scalar64(wf, dst);
  return true;
}

/// Unconstrained fallback for the VOP3 f64 VOPC path; see the binary-path note.
template <typename Inst, typename CmpOp>
[[nodiscard]] inline bool try_execute_vopc64_vop3_fp64_simd(Inst &, Wavefront &, CmpOp) {
  return false;
}

/// VOP3 f64 binary SIMD fast path. 64-bit-lane counterpart of
/// try_execute_binary_vop3_fp_simd: reads src0/src1 as `native<double>` through
/// the split lo/hi VGPR-pair path (read_simd64), applies the per-source abs/neg
/// VOP3 modifiers in the f64 domain (apply_vop3_src_mod_f64), runs `bin_op`,
/// applies the result omod/clamp (apply_vop3_dst_mod_f64), and masked-stores the
/// result through write_simd64. All modifier helpers are bit-exact, so the fast
/// path stays correct even with modifiers set; no bail.
template <typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_fp64_simd(Inst &inst, Wavefront &wf,
                                                            BinOp bin_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(read_simd64<T>(inst.src0, wf, base), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(read_simd64<T>(inst.src1, wf, base), abs, neg);
    const auto r = apply_vop3_dst_mod_f64(bin_op(a, b), omod, clamp);
    write_simd64<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f64 binary path; see the binary-path note.
template <typename Inst, typename BinOp>
[[nodiscard]] inline bool try_execute_binary_vop3_fp64_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 f64 unary SIMD fast path. 64-bit-lane counterpart of
/// try_execute_unary_vop3_fp_simd: reads `src0` as `native<double>`, applies
/// the src0 abs/neg modifiers (apply_vop3_src_mod_f64), runs `un_op`, applies
/// the result omod/clamp (apply_vop3_dst_mod_f64). All bit-exact.
template <typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop3_fp64_simd(Inst &inst, Wavefront &wf, UnOp un_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(read_simd64<T>(inst.src0, wf, base), abs, neg);
    const auto r = apply_vop3_dst_mod_f64(un_op(a), omod, clamp);
    write_simd64<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f64 unary path; see the binary-path note.
template <typename Inst, typename UnOp>
[[nodiscard]] inline bool try_execute_unary_vop3_fp64_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// VOP3 f16 unary SIMD fast path. Mirrors the scalar body's
/// f16_to_f32 -> abs/neg -> op -> omod/clamp -> f32_to_f16 chain:
/// read raw uint32 lanes (low 16 = f16 bits), widen via util::f16_to_f32_simd,
/// apply src0 abs/neg in f32, run `un_op` (`native<float> -> native<float>`),
/// apply omod/clamp in f32, narrow via util::f32_to_f16_simd, write_simd<uint32_t>.
/// All steps bit-exact per the f16 VOP3 cmp slice's widening probe (f16_to_f32
/// + f32_to_f16 verified exhaustive incl. NaN payload). High 16 bits of the dst
/// are written zero (matching write_lane(f32_to_f16(...)) which zero-extends).
template <typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop3_fp16_simd(Inst &inst, Wavefront &wf, UnOp un_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto in = util::f16_to_f32_simd(read_simd<T>(inst.src0, wf, base));
    const auto a = apply_vop3_src_mod_f32<0>(in, abs, neg);
    const auto r = apply_vop3_dst_mod_f32(un_op(a), omod, clamp);
    write_simd<T>(inst.vdst, wf, base, util::f32_to_f16_simd(r), chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f16 unary path; see the binary-path note.
template <typename Inst, typename UnOp>
[[nodiscard]] inline bool try_execute_unary_vop3_fp16_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// VOP3 integer/bitwise ternary SIMD fast path. Reads `src0`/`src1`/`src2`,
/// runs `tern_op(a, b, c)`, and masked-stores the result. The generated scalar
/// bodies for these ternary integer ops apply no source/result modifiers (abs/
/// neg/omod are float-only; clamp is unused on integer 3-source ops), so the
/// plain functor is bit-identical to the scalar body. T is a 32-bit integer
/// lane type (typically uint32_t). Same SIMD-capable / EXEC-chunk loop as the
/// binary VOP3 path.
template <typename T, typename Inst, typename TernOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_simd(Inst &inst, Wavefront &wf, TernOp tern_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.src1, wf, base);
    const auto c = read_simd<T>(inst.src2, wf, base);
    write_simd<T>(inst.vdst, wf, base, tern_op(a, b, c), chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 integer ternary path; see binary-path note.
template <typename T, typename Inst, typename TernOp>
[[nodiscard]] inline bool try_execute_ternary_vop3_simd(Inst &, Wavefront &, TernOp) {
  return false;
}

/// VOP3 f32 ternary SIMD fast path (FMA / MAD family). Reads src0/src1/src2 as
/// `native<float>`, applies the per-source abs/neg VOP3 modifiers, runs
/// `tern_op(a, b, c)`, applies the result omod/clamp. NaN-payload divergence
/// between stdx::fma and std::fma is the standard accepted carve-out (the A/B
/// test skips NaN-input lanes), same as the existing VOP2 ternary path.
template <typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_fp_simd(Inst &inst, Wavefront &wf,
                                                           FmaOp tern_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = float32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(read_simd<T>(inst.src0, wf, base), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(read_simd<T>(inst.src1, wf, base), abs, neg);
    const auto c = apply_vop3_src_mod_f32<2>(read_simd<T>(inst.src2, wf, base), abs, neg);
    const auto r = apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp);
    write_simd<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] inline bool try_execute_ternary_vop3_fp_simd(Inst &, Wavefront &, FmaOp) {
  return false;
}

/// VOP3 f16 ternary SIMD fast path. Mirrors the scalar f16 chain across three
/// sources: widen each via util::f16_to_f32_simd, apply f32 abs/neg, run
/// `tern_op` on native<float>, apply omod/clamp, narrow via f32_to_f16_simd.
template <typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_fp16_simd(Inst &inst, Wavefront &wf,
                                                             FmaOp tern_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(
        util::f16_to_f32_simd(read_simd<T>(inst.src0, wf, base)), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(
        util::f16_to_f32_simd(read_simd<T>(inst.src1, wf, base)), abs, neg);
    const auto c = apply_vop3_src_mod_f32<2>(
        util::f16_to_f32_simd(read_simd<T>(inst.src2, wf, base)), abs, neg);
    const auto r = apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp);
    write_simd<T>(inst.vdst, wf, base, util::f32_to_f16_simd(r), chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] inline bool try_execute_ternary_vop3_fp16_simd(Inst &, Wavefront &, FmaOp) {
  return false;
}

/// VOP3 f64 ternary SIMD fast path. 64-bit-lane counterpart: read src0/src1/src2
/// via read_simd64<double>, apply abs/neg in f64, run `tern_op`, apply
/// omod/clamp, write_simd64.
template <typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ternary_vop3_fp64_simd(Inst &inst, Wavefront &wf,
                                                             FmaOp tern_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(read_simd64<T>(inst.src0, wf, base), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(read_simd64<T>(inst.src1, wf, base), abs, neg);
    const auto c = apply_vop3_src_mod_f64<2>(read_simd64<T>(inst.src2, wf, base), abs, neg);
    const auto r = apply_vop3_dst_mod_f64(tern_op(a, b, c), omod, clamp);
    write_simd64<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] inline bool try_execute_ternary_vop3_fp64_simd(Inst &, Wavefront &, FmaOp) {
  return false;
}

/// VOP3 dst-accumulate FMA fast path (f32). Counterpart of
/// try_execute_ternary_vop3_fp_simd for the v_fmac / v_mac family, where the
/// third FMA operand IS the destination register (no separate src2 Operand
/// in the per-isa codegen class). The scalar body reads vdst as the
/// accumulator without applying abs/neg to it (per scalar; verified for
/// v_fmac_f32_vop3 + v_mac_f32_vop3). SIMD mirrors: read inst.vdst as the
/// third operand, apply abs/neg to src0/src1 only, run `tern_op`, apply
/// result omod/clamp, masked-store back to inst.vdst (overwriting accumulator).
template <typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_fmac_vop3_fp_simd(Inst &inst, Wavefront &wf, FmaOp tern_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = float32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(read_simd<T>(inst.src0, wf, base), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(read_simd<T>(inst.src1, wf, base), abs, neg);
    const auto c = read_simd<T>(inst.vdst, wf, base); // accumulator, NO modifier
    const auto r = apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp);
    write_simd<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] inline bool try_execute_fmac_vop3_fp_simd(Inst &, Wavefront &, FmaOp) {
  return false;
}

/// VOP3 dst-accumulate FMA fast path (f16). f16 widen chain across src0/src1
/// + vdst (accumulator). NO abs/neg on accumulator (per scalar).
template <typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_fmac_vop3_fp16_simd(Inst &inst, Wavefront &wf,
                                                          FmaOp tern_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(
        util::f16_to_f32_simd(read_simd<T>(inst.src0, wf, base)), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(
        util::f16_to_f32_simd(read_simd<T>(inst.src1, wf, base)), abs, neg);
    const auto c = util::f16_to_f32_simd(read_simd<T>(inst.vdst, wf, base)); // accumulator, no mod
    const auto r = apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp);
    write_simd<T>(inst.vdst, wf, base, util::f32_to_f16_simd(r), chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] inline bool try_execute_fmac_vop3_fp16_simd(Inst &, Wavefront &, FmaOp) {
  return false;
}

/// VOP3 dst-accumulate FMA fast path (f64).
template <typename Inst, typename FmaOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_fmac_vop3_fp64_simd(Inst &inst, Wavefront &wf,
                                                          FmaOp tern_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(read_simd64<T>(inst.src0, wf, base), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(read_simd64<T>(inst.src1, wf, base), abs, neg);
    const auto c = read_simd64<T>(inst.vdst, wf, base); // accumulator, no mod
    const auto r = apply_vop3_dst_mod_f64(tern_op(a, b, c), omod, clamp);
    write_simd64<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] inline bool try_execute_fmac_vop3_fp64_simd(Inst &, Wavefront &, FmaOp) {
  return false;
}

/// VOP3 mixed-width ldexp fast path (f32 = std::ldexp(f32 src0, int32 src1)).
/// Reads src0 as native<float>, applies src0 abs/neg in f32, reads src1 as
/// native<int32_t> (per-lane exponent), runs `op(a, e)` (stdx::ldexp), applies
/// result omod/clamp, write_simd. stdx::ldexp is bit-exact to std::ldexp for
/// every input incl. NaN (proven via the VOP2 v_ldexp_f16 path).
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ldexp_vop3_fp32_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = float32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(read_simd<T>(inst.src0, wf, base), abs, neg);
    const auto e = read_simd<int32_t>(inst.src1, wf, base);
    const auto r = apply_vop3_dst_mod_f32(op(a, e), omod, clamp);
    write_simd<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] inline bool try_execute_ldexp_vop3_fp32_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3 mixed-width ldexp fast path (f64 = std::ldexp(f64 src0, int32 src1)).
/// Reads src0 as native<double> via read_simd64, applies src0 abs/neg in f64,
/// reads src1 as narrow32<int32_t> (native_width64-wide), runs `op(a, e)`,
/// applies result omod/clamp, write_simd64.
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_ldexp_vop3_fp64_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(read_simd64<T>(inst.src0, wf, base), abs, neg);
    const auto e = read_narrow<int32_t>(inst.src1, wf, base);
    const auto r = apply_vop3_dst_mod_f64(op(a, e), omod, clamp);
    write_simd64<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] inline bool try_execute_ldexp_vop3_fp64_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3 f32 unary SIMD fast path. Reads `src0` as f32, applies the src0 abs/neg
/// modifiers, runs `un_op` (`native<float> -> native<float>`), then applies the
/// result omod/clamp — the scalar body's order (abs->neg, op, omod->clamp). Tin
/// and Tout are both float32_t (the plain int/cvt unary VOP3 forms apply no
/// modifiers and reuse the VOP1 unary path directly). The modifier helpers are
/// bit-exact, so the fast path stays correct with modifiers set; no bail.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_unary_vop3_fp_simd(Inst &inst, Wavefront &wf, UnOp un_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vdst.simd_capable())
    return false;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<Tout>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(read_simd<Tin>(inst.src0, wf, base), abs, neg);
    const auto r = apply_vop3_dst_mod_f32(un_op(a), omod, clamp);
    write_simd<Tout>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f32 unary path.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
[[nodiscard]] inline bool try_execute_unary_vop3_fp_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// v_div_fixup_f32 SIMD helper. Mirrors the scalar `else if` cascade
/// (execute_shared.h:execute_v_div_fixup_f32_vop3): given three already-
/// modified f32 operands `p` (the fma scaffold input), `b` (numerator), and
/// `c` (denominator), pick the result among NaN/Inf/zero copysign cases
/// according to AMD's div_fixup ULP table. The cascade is applied lowest-
/// priority first so that the higher-priority `where` blends overwrite —
/// equivalent to scalar's first-match `else if`. The sign-of-quotient
/// `b ^ c` is computed once in the integer domain and reinterpreted as
/// float, matching the scalar's `bit_cast<float>(bits(b) ^ bits(c))`. Both
/// `std::copysign(Inf, bxc)` and `std::copysign(0, bxc)` reduce to
/// `stdx::copysign(target, bxc)`.
inline util::native<float> div_fixup_f32_simd(util::native<float> p, util::native<float> b,
                                              util::native<float> c) {
  using F = util::native<float>;
  using U = util::native<uint32_t>;
  const auto bxc = std::bit_cast<F>(std::bit_cast<U>(b) ^ std::bit_cast<U>(c));
  const auto inf_val = util::stdx::copysign(F(std::numeric_limits<float>::infinity()), bxc);
  const auto zero_val = util::stdx::copysign(F(0.0f), bxc);
  const auto qnan = F(std::numeric_limits<float>::quiet_NaN());
  const auto b_nan = util::stdx::isnan(b);
  const auto c_nan = util::stdx::isnan(c);
  const auto b_inf = util::stdx::isinf(b);
  const auto c_inf = util::stdx::isinf(c);
  const auto b_zero = (b == F(0.0f));
  const auto c_zero = (c == F(0.0f));
  F r = p;
  util::stdx::where(b_inf, r) = zero_val;
  util::stdx::where(c_inf, r) = inf_val;
  util::stdx::where(c_zero, r) = zero_val;
  util::stdx::where(b_zero, r) = inf_val;
  util::stdx::where(b_inf && c_inf, r) = qnan;
  util::stdx::where(b_zero && c_zero, r) = qnan;
  util::stdx::where(c_nan, r) = c;
  util::stdx::where(b_nan, r) = b;
  return r;
}

/// f64 counterpart of div_fixup_f32_simd — same cascade, 64-bit-lane domain.
inline util::native<double> div_fixup_f64_simd(util::native<double> p, util::native<double> b,
                                               util::native<double> c) {
  using D = util::native<double>;
  using U = util::native<uint64_t>;
  const auto bxc = std::bit_cast<D>(std::bit_cast<U>(b) ^ std::bit_cast<U>(c));
  const auto inf_val = util::stdx::copysign(D(std::numeric_limits<double>::infinity()), bxc);
  const auto zero_val = util::stdx::copysign(D(0.0), bxc);
  const auto qnan = D(std::numeric_limits<double>::quiet_NaN());
  const auto b_nan = util::stdx::isnan(b);
  const auto c_nan = util::stdx::isnan(c);
  const auto b_inf = util::stdx::isinf(b);
  const auto c_inf = util::stdx::isinf(c);
  const auto b_zero = (b == D(0.0));
  const auto c_zero = (c == D(0.0));
  D r = p;
  util::stdx::where(b_inf, r) = zero_val;
  util::stdx::where(c_inf, r) = inf_val;
  util::stdx::where(c_zero, r) = zero_val;
  util::stdx::where(b_zero, r) = inf_val;
  util::stdx::where(b_inf && c_inf, r) = qnan;
  util::stdx::where(b_zero && c_zero, r) = qnan;
  util::stdx::where(c_nan, r) = c;
  util::stdx::where(b_nan, r) = b;
  return r;
}

/// VOP3 div_fmas SIMD fast path (f32). The scalar body is `fma(s0, s1, s2)`
/// followed by a per-lane `ldexp(result, 32)` gated by the VCC bit; no
/// omod/clamp are applied (the encoded modifier fields are intentionally
/// ignored). Per-source abs/neg modifiers ARE applied (matching scalar).
/// This is a dedicated glue (rather than routing through the existing
/// ternary fp glue) because the omod/clamp policy differs and the VCC-
/// driven ldexp gate is unique to div_fmas. NaN-input lanes skipped by test
/// (gcc-13 packed FMA quiets a different NaN operand vs scalar std::fma —
/// same accepted carve-out as the rest of the ternary fp suite).
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_div_fmas_f32_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = float32_t;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const uint64_t vcc = wf.vcc();
  using IExp = util::stdx::fixed_size_simd<int, util::native<float>::size()>;
  const IExp shift_32 = IExp(32);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(read_simd<T>(inst.src0, wf, base), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(read_simd<T>(inst.src1, wf, base), abs, neg);
    const auto c = apply_vop3_src_mod_f32<2>(read_simd<T>(inst.src2, wf, base), abs, neg);
    auto r = util::stdx::fma(a, b, c);
    const auto scaled = util::stdx::ldexp(r, shift_32);
    const uint64_t sel_bits = (vcc >> base) & chunk_full;
    alignas(util::native<uint32_t>) uint32_t selbuf[W];
    for (std::size_t i = 0; i < W; ++i)
      selbuf[i] = static_cast<uint32_t>((sel_bits >> i) & 1u);
    const auto vcc_mask_u = util::load<uint32_t>(selbuf) != 0u;
    util::stdx::where(simd_mask_as<float>(vcc_mask_u), r) = scaled;
    write_simd<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst>
[[nodiscard]] inline bool try_execute_div_fmas_f32_simd(Inst &, Wavefront &) {
  return false;
}

/// f64 counterpart of try_execute_div_fmas_f32_simd. Same VCC-gated ldexp
/// shape with shift=64 (per AMD spec for div_fmas_f64). 64-bit-lane reads,
/// abs/neg in f64 domain via apply_vop3_src_mod_f64.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_div_fmas_f64_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = double;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const uint64_t vcc = wf.vcc();
  using IExp = util::stdx::fixed_size_simd<int, util::native_width64>;
  const IExp shift_64 = IExp(64);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(read_simd64<T>(inst.src0, wf, base), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(read_simd64<T>(inst.src1, wf, base), abs, neg);
    const auto c = apply_vop3_src_mod_f64<2>(read_simd64<T>(inst.src2, wf, base), abs, neg);
    auto r = util::stdx::fma(a, b, c);
    const auto scaled = util::stdx::ldexp(r, shift_64);
    // VCC mask built directly in the f64 abi via a generator; avoids a
    // cross-abi mask cast (the f32 path can ride load+simd_mask_as because
    // native<uint32_t> shares abi with native<float>, but native<uint64_t>
    // does not match native<double>::abi on AVX-512).
    const uint64_t vcc_chunk = vcc >> base;
    const auto vcc_mask = util::native<double>([&](std::size_t i) {
                            return ((vcc_chunk >> i) & 1ULL) ? 1.0 : 0.0;
                          }) != 0.0;
    util::stdx::where(vcc_mask, r) = scaled;
    write_simd64<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst>
[[nodiscard]] inline bool try_execute_div_fmas_f64_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3 binary carry SIMD fast path (sdst-enc forms: add_co / sub_co /
/// subrev_co / addc_co / subb_co / subbrev_co). Mirrors the VOP2 carry
/// glue (read incoming VCC as carry-in, run the per-lane carry op via
/// SimdCarry, merge the per-lane carry-out into a 64-bit mask), but
/// reads src0/src1 from the VOP3 operand layout and writes the
/// resulting carry mask into the per-instruction SGPR-pair sdst rather
/// than wf.vcc(). Inactive-lane bits in the carry mask are preserved
/// from the incoming VCC, matching the scalar `uint64_t vcc = wf.vcc();
/// ... inst.sdst.write_scalar64(wf, vcc)` pattern. No abs/neg/omod/
/// clamp — the scalar bodies ignore modifiers on the integer carry ops.
template <typename Inst, typename CarryOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_carry_simd(Inst &inst, Wavefront &wf,
                                                             CarryOp carry_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  // sdst-enc carry ops seed the merged carry word from the incoming VCC so
  // inactive lanes are preserved through to sdst, matching the scalar
  // `uint64_t vcc = wf.vcc()` init. The carry-in forms also read each chunk's
  // VCC bits into a per-lane 0/1 vector that the functor accepts as `cin`.
  const uint64_t vcc_in = wf.vcc();
  uint64_t sdst_out = vcc_in;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.src1, wf, base);
    const uint64_t cin_bits = (vcc_in >> base) & chunk_full;
    alignas(util::native<T>) uint32_t cinbuf[W];
    for (std::size_t i = 0; i < W; ++i)
      cinbuf[i] = static_cast<uint32_t>((cin_bits >> i) & 1u);
    const auto cin = util::load<T>(cinbuf);
    const auto r = carry_op(a, b, cin);
    write_simd<T>(inst.vdst, wf, base, r.value, chunk);
    uint64_t carry_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (r.carry[i])
        carry_bits |= (1ULL << i);
    sdst_out = (sdst_out & ~(chunk << base)) | ((carry_bits & chunk) << base);
  }
  inst.sdst.write_scalar64(wf, sdst_out);
  return true;
}

template <typename Inst, typename CarryOp>
[[nodiscard]] inline bool try_execute_binary_vop3_carry_simd(Inst &, Wavefront &, CarryOp) {
  return false;
}

/// VOP3 binary carry SIMD fast path, src2-cin variant. Covers the six
/// carry-in forms: CDNA4 sdst-enc `v_addc_co_u32_vop3` / `v_subb_co_u32_vop3`
/// / `v_subbrev_co_u32_vop3` and the RDNA-only `v_add_co_ci_u32_vop3` /
/// `v_sub_co_ci_u32_vop3` / `v_subrev_co_ci_u32_vop3`. All six scalar bodies
/// share the same shape: per-lane carry-in is read from the SGPR-pair
/// `inst.src2.read_scalar64(wf)` (NOT from VCC — the RDNA forms also pull
/// cin from src2; the decoder binds src2 to VCC there but the body is
/// uniform), and the merged carry-out is written to `inst.sdst.write_scalar64`
/// with inactive-lane bits preserved from the incoming VCC. Mirrors
/// `try_execute_binary_vop3_carry_simd` exactly except for the cin source.
template <typename Inst, typename CarryOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_carry_src2_simd(Inst &inst, Wavefront &wf,
                                                                  CarryOp carry_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const uint64_t vcc_in = wf.vcc();
  const uint64_t cin_word = inst.src2.read_scalar64(wf);
  uint64_t sdst_out = vcc_in;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.src1, wf, base);
    const uint64_t cin_bits = (cin_word >> base) & chunk_full;
    alignas(util::native<T>) uint32_t cinbuf[W];
    for (std::size_t i = 0; i < W; ++i)
      cinbuf[i] = static_cast<uint32_t>((cin_bits >> i) & 1u);
    const auto cin = util::load<T>(cinbuf);
    const auto r = carry_op(a, b, cin);
    write_simd<T>(inst.vdst, wf, base, r.value, chunk);
    uint64_t carry_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (r.carry[i])
        carry_bits |= (1ULL << i);
    sdst_out = (sdst_out & ~(chunk << base)) | ((carry_bits & chunk) << base);
  }
  inst.sdst.write_scalar64(wf, sdst_out);
  return true;
}

template <typename Inst, typename CarryOp>
[[nodiscard]] inline bool try_execute_binary_vop3_carry_src2_simd(Inst &, Wavefront &, CarryOp) {
  return false;
}

/// VOP3 v_mov_b16 SIMD fast path. Single op (RDNA3+ only), fixed-shape body:
/// read u32 src0, mask the low 16 bits, convert to f32 (exact for every u16
/// value), apply the uniform omod (`*2 / *4 / *0.5`) and clamp ([0, 1])
/// modifiers, truncating-cast back to int32, mask 16, write u32. No abs/neg/
/// op_sel — the scalar body ignores them. Functorless / fixed-op.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_mov_b16_vop3_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  using F = util::native<float>;
  using I = util::native<int32_t>;
  using U = util::native<uint32_t>;
  const F kZero(0.0f);
  const F kOne(1.0f);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    U raw = read_simd<uint32_t>(inst.src0, wf, base);
    U u16 = raw & 0xFFFFu;
    F f = util::stdx::static_simd_cast<F>(u16);
    if (omod == 1)
      f = f * F(2.0f);
    else if (omod == 2)
      f = f * F(4.0f);
    else if (omod == 3)
      f = f * F(0.5f);
    if (clamp) {
      util::stdx::where(f < kZero, f) = kZero;
      util::stdx::where(f > kOne, f) = kOne;
    }
    I i = util::stdx::static_simd_cast<I>(f);
    U out = util::stdx::static_simd_cast<U>(i) & 0xFFFFu;
    write_simd<uint32_t>(inst.vdst, wf, base, out, chunk);
  }
  return true;
}

template <typename Inst>
[[nodiscard]] inline bool try_execute_mov_b16_vop3_simd(Inst &, Wavefront &) {
  return false;
}

/// Destination shape for the VOP3P fma_mix / mad_mix family. F32 writes a
/// full 32-bit float into vdst; F16_LO writes the f16-narrowed result into
/// the low half of vdst (high half preserved); F16_HI writes it into the
/// high half (low half preserved). Selected by the per-mnemonic glue probe.
enum class FmaMixDst { F32, F16_LO, F16_HI };

/// VOP3P fma_mix / mad_mix SIMD fast path. Six ops share one body because
/// the scalar reference computes `a * b + c` (plain `*+`, not std::fma) for
/// all six and differs only in (a) the f16-vs-f32 widening shape per source
/// and (b) the f16-lo/f16-hi/f32 narrowing shape on the destination.
///
/// Per-source data fetch is gated by `op_sel_hi` (src0/src1) and
/// `op_sel_hi_2` (src2): when the bit is 0 the source is read as f32; when
/// 1 the source is read as a 32-bit word and the low or high f16 half
/// (selected by the matching `op_sel` bit) is widened via f16_to_f32_simd.
/// Per-source sign-flip is gated by `neg` (xor of bit 31). VOP3P does not
/// carry abs or omod. Result-clamp saturates to [0, 1] via stdx::where.
///
/// All modifier fields are uniform across the wave so the per-source mode
/// branches live outside the chunk loop and feed into the same `a*b+c`
/// inner kernel regardless of fetch shape, keeping the SIMD path branch-
/// predictable on every chunk.
template <FmaMixDst DstMode, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_fma_mix_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  using T = float32_t;
  const uint32_t op_sel = inst.inst_.op_sel;
  const uint32_t op_sel_hi = inst.inst_.op_sel_hi;
  const uint32_t op_sel_hi_2 = inst.inst_.op_sel_hi_2;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  using U = util::native<uint32_t>;
  using F = util::native<float>;
  const F kZero(0.0f);
  const F kOne(1.0f);
  const U kSignBit(0x80000000u);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto load_src = [&](auto &op, uint32_t sel_hi_bit, uint32_t sel_bit, uint32_t neg_bit) -> F {
      F v;
      if (sel_hi_bit) {
        U raw = read_simd<uint32_t>(op, wf, base);
        U halves = sel_bit ? (raw >> 16) : (raw & 0xFFFFu);
        v = util::f16_to_f32_simd(halves);
      } else {
        v = read_simd<float>(op, wf, base);
      }
      if (neg_bit)
        v = std::bit_cast<F>(std::bit_cast<U>(v) ^ kSignBit);
      return v;
    };
    F a = load_src(inst.src0, op_sel_hi & 1u, op_sel & 1u, neg & 1u);
    F b = load_src(inst.src1, (op_sel_hi >> 1) & 1u, (op_sel >> 1) & 1u, (neg >> 1) & 1u);
    F c = load_src(inst.src2, op_sel_hi_2, (op_sel >> 2) & 1u, (neg >> 2) & 1u);
    F r = a * b + c;
    if (clamp) {
      util::stdx::where(r < kZero, r) = kZero;
      util::stdx::where(r > kOne, r) = kOne;
    }
    if constexpr (DstMode == FmaMixDst::F32) {
      write_simd<float>(inst.vdst, wf, base, r, chunk);
    } else {
      U h = util::f32_to_f16_simd(r); // low16 = f16, high16 zero
      U prev = read_simd<uint32_t>(inst.vdst, wf, base);
      U packed;
      if constexpr (DstMode == FmaMixDst::F16_LO) {
        packed = (prev & 0xFFFF0000u) | h;
      } else { // F16_HI
        packed = (prev & 0x0000FFFFu) | (h << 16);
      }
      write_simd<uint32_t>(inst.vdst, wf, base, packed, chunk);
    }
  }
  return true;
}

template <FmaMixDst DstMode, typename Inst>
[[nodiscard]] inline bool try_execute_vop3p_fma_mix_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3P packed-16 binary integer SIMD fast path. Covers the pk_add /
/// pk_sub / pk_mul_lo / pk_min / pk_max / pk_*shrev family on i16/u16/b16.
/// Scalar pattern (verified across pk_add_i16, pk_add_u16, pk_mul_lo_u16,
/// pk_lshlrev_b16, pk_min/max_*_16, etc.): two packed 16-bit values per
/// 32-bit lane, each output half computed from the matching halves of
/// src0/src1 picked by op_sel (low half) / op_sel_hi (high half). Default
/// packing is op_sel = 0 (both srcs feed their low half into the low
/// output) and op_sel_hi = 3 (both srcs feed their high half into the
/// high output) — the LLVM-AS encoder emits this for the default-mode
/// pk mnemonics, and the SIMD fast path bails when any other combination
/// is requested. The scalar bodies do NOT apply neg / neg_hi / clamp on
/// these integer pk ops, so the SIMD path also passes through. Functor
/// receives the two source u32 lane vectors (each holding {low16, high16}
/// packed) and returns the same shape; the per-half decompose / recompose
/// lives inside the functor for op-specific flexibility (e.g. mul_lo
/// requires the wider product to be masked to 16 bits before pack).
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_binary_int_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  if (inst.inst_.op_sel != 0u || inst.inst_.op_sel_hi != 3u)
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = read_simd<T>(inst.src0, wf, base);
    const auto b = read_simd<T>(inst.src1, wf, base);
    const auto r = op(a, b);
    write_simd<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] inline bool try_execute_vop3p_pk_binary_int_simd(Inst &, Wavefront &, Op) {
  return false;
}

} // namespace amdgpu
} // namespace rocjitsu

/// Probe macro emitted at the top of each SIMD-eligible execute_<mnemonic>
/// kernel by simd_codegen.py. Expands to the binary-VOP2 fast-path call and
/// an early `return` on success, keeping each generated body to a single
/// line. Variadic in the operator argument so functor lambdas (which contain
/// commas) pass through as one token sequence. Relies on the kernel's `inst`
/// and `wf` parameters being in scope.
#define ROCJITSU_TRY_SIMD_VOP2_BINARY(T, ...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_binary_vop2_simd<T>(inst, wf, __VA_ARGS__))                  \
  return

/// VOP1 unary counterpart of ROCJITSU_TRY_SIMD_VOP2_BINARY. Variadic in the
/// operator argument so functor lambdas pass through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOP1_UNARY(Tin, Tout, ...)                                               \
  if (::rocjitsu::amdgpu::try_execute_unary_vop1_simd<Tin, Tout>(inst, wf, __VA_ARGS__))           \
  return

/// Carry-VOP2 counterpart. Lane type is fixed to uint32_t, so unlike the binary
/// macro this takes only the functor. Variadic so the functor's commas pass
/// through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOP2_CARRY(...)                                                          \
  if (::rocjitsu::amdgpu::try_execute_binary_vop2_carry_simd(inst, wf, __VA_ARGS__))               \
  return

/// Ternary (FMA/MAC/MAD) VOP2 counterpart. `KEXPR` is the inline-literal bits
/// (an `inst.`-qualified expression, or `0u` for forms without a literal),
/// broadcast to every lane before the call. Variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP2_TERNARY(T, KEXPR, ...)                                              \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop2_simd<T>(inst, wf, ::util::broadcast<T>(KEXPR),  \
                                                           __VA_ARGS__))                           \
  return

/// v_cndmask_b32 counterpart. Fixed op (VCC-driven select), so no type or
/// functor argument.
#define ROCJITSU_TRY_SIMD_VOP2_CNDMASK()                                                           \
  if (::rocjitsu::amdgpu::try_execute_cndmask_vop2_simd(inst, wf))                                 \
  return

/// v_cndmask_b32 VOP3 counterpart. Same shape, but the selector comes from the
/// SGPR-pair `src2` instead of fixed VCC.
#define ROCJITSU_TRY_SIMD_VOP3_CNDMASK()                                                           \
  if (::rocjitsu::amdgpu::try_execute_cndmask_vop3_simd(inst, wf))                                 \
  return

/// 16-bit variant of v_cndmask_b32 VOP3 (RDNA3+). Same select + low-16 mask
/// on the result.
#define ROCJITSU_TRY_SIMD_VOP3_CNDMASK_B16()                                                       \
  if (::rocjitsu::amdgpu::try_execute_cndmask_b16_vop3_simd(inst, wf))                             \
  return

/// 64-bit-lane VOP2 FMA counterpart (v_fmac_f64). Lane type is fixed to double,
/// so this takes only the dst-accumulate functor. Variadic so the functor's
/// commas pass through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOP2_FMA_F64(...)                                                        \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop2_f64_simd<double>(inst, wf, __VA_ARGS__))        \
  return

/// 64-bit-lane VOP1 unary counterpart. `T` is the 64-bit lane type (`double`
/// for the f64 math ops, `uint64_t` for v_mov_b64). Variadic in the functor so
/// its commas pass through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOP1_UNARY_F64(T, ...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_unary_vop1_f64_simd<T>(inst, wf, __VA_ARGS__))               \
  return

/// Mixed-width cvt counterpart, f64 source -> 32-bit dst. `Tout` is the 32-bit
/// result lane type; the functor (`native<double> -> narrow32<Tout>`) is variadic
/// so its commas pass through as one token sequence.
#define ROCJITSU_TRY_SIMD_CVT_F64_TO_B32(Tout, ...)                                                \
  if (::rocjitsu::amdgpu::try_execute_cvt_f64_to_b32_simd<Tout>(inst, wf, __VA_ARGS__))            \
  return

/// Mixed-width cvt counterpart, 32-bit source -> f64 dst. `Tin` is the 32-bit
/// source lane type; the functor (`narrow32<Tin> -> native<double>`) is variadic.
#define ROCJITSU_TRY_SIMD_CVT_B32_TO_F64(Tin, ...)                                                 \
  if (::rocjitsu::amdgpu::try_execute_cvt_b32_to_f64_simd<Tin>(inst, wf, __VA_ARGS__))             \
  return

/// VOPC compare counterpart. `T` is the 32-bit lane read type; the comparison
/// functor (which may convert/narrow inside) is variadic so its commas pass
/// through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOPC(T, ...)                                                             \
  if (::rocjitsu::amdgpu::try_execute_vopc_simd<T>(inst, wf, __VA_ARGS__))                         \
  return

/// 64-bit-lane VOPC compare counterpart (f64/i64/u64). `T` is the 64-bit lane
/// read type; the comparison functor is variadic so its commas pass through.
#define ROCJITSU_TRY_SIMD_VOPC64(T, ...)                                                           \
  if (::rocjitsu::amdgpu::try_execute_vopc64_simd<T>(inst, wf, __VA_ARGS__))                       \
  return

/// Mixed-width v_cmp_class_f64 counterpart (64-bit value, 32-bit mask). No type
/// argument; the class functor `(native<uint64_t> bits, narrow32<uint32_t> mask)
/// -> mask` is variadic so its commas pass through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOPC_CLASS_F64(...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_vopc_class_f64_simd(inst, wf, __VA_ARGS__))                  \
  return

/// VOP3 v_cmp_class_f16/f32 counterpart (32-bit value, abs/neg modifiers, src1
/// mask, SGPR-pair dst). `SM` is the per-op sign-bit mask (0x8000 / 0x80000000);
/// the class functor is variadic so its commas pass through.
#define ROCJITSU_TRY_SIMD_VOP3_CLASS_B32(SM, ...)                                                  \
  if (::rocjitsu::amdgpu::try_execute_vop3_class_b32_simd(inst, wf, SM, __VA_ARGS__))              \
  return

/// VOP3 v_cmp_class_f64 counterpart (64-bit value). `SM` is the f64 sign-bit mask
/// (0x8000000000000000); the class functor is variadic.
#define ROCJITSU_TRY_SIMD_VOP3_CLASS_F64(SM, ...)                                                  \
  if (::rocjitsu::amdgpu::try_execute_vop3_class_f64_simd(inst, wf, SM, __VA_ARGS__))              \
  return

/// VOP3 integer/bitwise binary counterpart (reads src0/src1, no modifiers).
/// `T` is the 32-bit integer lane type; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_INT(T, ...)                                                  \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_simd<T>(inst, wf, __VA_ARGS__))                  \
  return

/// VOP3 f32 binary counterpart (reads src0/src1, applies abs/neg/omod/clamp).
/// `T` is the 32-bit float lane type; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_FP(T, ...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_fp_simd<T>(inst, wf, __VA_ARGS__))               \
  return

/// VOP3 f32 unary counterpart (reads src0, applies abs/neg/omod/clamp). `Tin`
/// and `Tout` are both float32_t; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_UNARY_FP(Tin, Tout, ...)                                            \
  if (::rocjitsu::amdgpu::try_execute_unary_vop3_fp_simd<Tin, Tout>(inst, wf, __VA_ARGS__))        \
  return

/// VOP3 integer/bitwise VOPC compare counterpart (32-bit lane, no modifiers,
/// SGPR-pair dst). `T` is the 32-bit integer lane read type; variadic in the
/// functor so its commas pass through as one token sequence.
#define ROCJITSU_TRY_SIMD_VOPC_VOP3_INT(T, ...)                                                    \
  if (::rocjitsu::amdgpu::try_execute_vopc_vop3_int_simd<T>(inst, wf, __VA_ARGS__))                \
  return

/// 64-bit-lane VOP3 integer/bitwise VOPC compare counterpart (i64/u64, no
/// modifiers, SGPR-pair dst). `T` is the 64-bit integer lane read type;
/// variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOPC64_VOP3_INT(T, ...)                                                  \
  if (::rocjitsu::amdgpu::try_execute_vopc64_vop3_int_simd<T>(inst, wf, __VA_ARGS__))              \
  return

/// VOP3 f32 VOPC compare counterpart (per-source abs/neg modifiers, SGPR-pair
/// dst). Lane type is fixed to float32_t; the functor takes already-modified
/// `native<float>` arguments and is variadic so its commas pass through.
#define ROCJITSU_TRY_SIMD_VOPC_VOP3_FP32(...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_vopc_vop3_fp32_simd(inst, wf, __VA_ARGS__))                  \
  return

/// VOP3 f16 VOPC compare counterpart. Lane type is fixed to uint32_t (raw f16
/// bits in low 16); the glue widens to f32 then applies the abs/neg modifier.
/// The functor takes the same already-widened, already-modified `native<float>`
/// arguments as the f32 path; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOPC_VOP3_FP16(...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_vopc_vop3_fp16_simd(inst, wf, __VA_ARGS__))                  \
  return

/// VOP3 f64 VOPC compare counterpart (per-source abs/neg modifiers, 64-bit
/// lane via split lo/hi VGPR-pair, SGPR-pair dst). Lane type is fixed to
/// `double`; the functor takes already-modified `native<double>` arguments
/// and is variadic so its commas pass through.
#define ROCJITSU_TRY_SIMD_VOPC64_VOP3_FP64(...)                                                    \
  if (::rocjitsu::amdgpu::try_execute_vopc64_vop3_fp64_simd(inst, wf, __VA_ARGS__))                \
  return

/// VOP3 integer/bitwise ternary counterpart (reads src0/src1/src2, no
/// modifiers). `T` is the 32-bit integer lane type; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_INT(T, ...)                                                 \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_simd<T>(inst, wf, __VA_ARGS__))                 \
  return

/// VOP3 f32 ternary counterpart (per-source abs/neg, result omod/clamp).
/// Functor takes already-modified `native<float>` arguments; variadic.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP32(...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_fp_simd(inst, wf, __VA_ARGS__))                 \
  return

/// VOP3 f16 ternary counterpart (raw uint32 lanes; widen f16->f32 each src,
/// abs/neg, op, omod/clamp, narrow). Variadic.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP16(...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_fp16_simd(inst, wf, __VA_ARGS__))               \
  return

/// VOP3 f64 ternary counterpart (read_simd64, per-source abs/neg, omod/clamp).
/// Variadic.
#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP64(...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_fp64_simd(inst, wf, __VA_ARGS__))               \
  return

/// VOP3 dst-accumulate FMA counterpart (f32). Per-isa class has no src2;
/// vdst is the accumulator. abs/neg apply to src0/src1 only.
#define ROCJITSU_TRY_SIMD_FMAC_VOP3_FP32(...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_fmac_vop3_fp_simd(inst, wf, __VA_ARGS__))                    \
  return

/// VOP3 dst-accumulate FMA counterpart (f16). Widen chain, vdst is the
/// (widened) accumulator.
#define ROCJITSU_TRY_SIMD_FMAC_VOP3_FP16(...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_fmac_vop3_fp16_simd(inst, wf, __VA_ARGS__))                  \
  return

/// VOP3 dst-accumulate FMA counterpart (f64).
#define ROCJITSU_TRY_SIMD_FMAC_VOP3_FP64(...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_fmac_vop3_fp64_simd(inst, wf, __VA_ARGS__))                  \
  return

/// VOP3 ldexp counterpart (f32 src0 + int32 src1 exp). Variadic functor.
#define ROCJITSU_TRY_SIMD_LDEXP_VOP3_FP32(...)                                                     \
  if (::rocjitsu::amdgpu::try_execute_ldexp_vop3_fp32_simd(inst, wf, __VA_ARGS__))                 \
  return

/// VOP3 ldexp counterpart (f64 src0 + int32 src1 exp). Variadic functor.
#define ROCJITSU_TRY_SIMD_LDEXP_VOP3_FP64(...)                                                     \
  if (::rocjitsu::amdgpu::try_execute_ldexp_vop3_fp64_simd(inst, wf, __VA_ARGS__))                 \
  return

/// VOP3 div_fmas counterpart (fma(s0,s1,s2) followed by VCC-gated ldexp; no
/// omod/clamp). No functor — the op is fixed (operand order, ldexp shift) and
/// distinct for f32 vs f64.
#define ROCJITSU_TRY_SIMD_DIV_FMAS_VOP3_FP32()                                                     \
  if (::rocjitsu::amdgpu::try_execute_div_fmas_f32_simd(inst, wf))                                 \
  return

#define ROCJITSU_TRY_SIMD_DIV_FMAS_VOP3_FP64()                                                     \
  if (::rocjitsu::amdgpu::try_execute_div_fmas_f64_simd(inst, wf))                                 \
  return

/// VOP3 f64 binary counterpart (read_simd64, per-source abs/neg, omod/clamp on
/// the result). Variadic in the functor so its commas pass through as one
/// token sequence.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_FP64(...)                                                    \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_fp64_simd(inst, wf, __VA_ARGS__))                \
  return

/// VOP3 f64 unary counterpart (read_simd64, src0 abs/neg, omod/clamp on
/// the result). Variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_UNARY_FP64(...)                                                     \
  if (::rocjitsu::amdgpu::try_execute_unary_vop3_fp64_simd(inst, wf, __VA_ARGS__))                 \
  return

/// VOP3 f16 unary counterpart (raw uint32 lanes; widen f16->f32, src0 abs/neg,
/// op, omod/clamp, narrow f32->f16). The functor takes already-widened-and-
/// modified `native<float>` and returns `native<float>`; variadic.
#define ROCJITSU_TRY_SIMD_VOP3_UNARY_FP16(...)                                                     \
  if (::rocjitsu::amdgpu::try_execute_unary_vop3_fp16_simd(inst, wf, __VA_ARGS__))                 \
  return

/// VOP3 binary carry probe. Functor takes (a, b, cin) and returns a
/// SimdCarry-shaped struct, same contract as the VOP2 carry path.
#define ROCJITSU_TRY_SIMD_VOP3_CARRY(...)                                                          \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_carry_simd(inst, wf, __VA_ARGS__))               \
  return

/// VOP3 binary carry probe, src2-cin variant (addc / subb / subbrev,
/// CDNA4 sdst-enc + RDNA co_ci). Functor takes (a, b, cin) and returns a
/// SimdCarry-shaped struct.
#define ROCJITSU_TRY_SIMD_VOP3_CARRY_SRC2(...)                                                     \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_carry_src2_simd(inst, wf, __VA_ARGS__))          \
  return

/// VOP3 v_mov_b16 probe (RDNA3+ only).
#define ROCJITSU_TRY_SIMD_VOP3_MOV_B16()                                                           \
  if (::rocjitsu::amdgpu::try_execute_mov_b16_vop3_simd(inst, wf))                                 \
  return

/// VOP3P fma_mix / mad_mix probes. The destination shape is the only thing
/// that differs across the six ops, so the probe is parameterised by
/// `FmaMixDst::{F32,F16_LO,F16_HI}` and the shared scalar formula `a*b+c`
/// (incl. clamp) lives in the glue. No functor argument.
#define ROCJITSU_TRY_SIMD_VOP3P_FMA_MIX_F32()                                                      \
  if (::rocjitsu::amdgpu::try_execute_vop3p_fma_mix_simd<::rocjitsu::amdgpu::FmaMixDst::F32>(inst, \
                                                                                             wf))  \
  return

#define ROCJITSU_TRY_SIMD_VOP3P_FMA_MIX_F16_LO()                                                   \
  if (::rocjitsu::amdgpu::try_execute_vop3p_fma_mix_simd<::rocjitsu::amdgpu::FmaMixDst::F16_LO>(   \
          inst, wf))                                                                               \
  return

#define ROCJITSU_TRY_SIMD_VOP3P_FMA_MIX_F16_HI()                                                   \
  if (::rocjitsu::amdgpu::try_execute_vop3p_fma_mix_simd<::rocjitsu::amdgpu::FmaMixDst::F16_HI>(   \
          inst, wf))                                                                               \
  return

/// VOP3P packed-16 integer binary probe. Functor takes two u32 simd vectors
/// (each holding {low16, high16} packed) and returns the same shape with
/// the per-half op applied; the glue gates op_sel/op_sel_hi to the default
/// packing (0 / 3) and falls back to scalar otherwise.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_INT(...)                                                 \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_binary_int_simd(inst, wf, __VA_ARGS__))             \
  return

#endif // ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
