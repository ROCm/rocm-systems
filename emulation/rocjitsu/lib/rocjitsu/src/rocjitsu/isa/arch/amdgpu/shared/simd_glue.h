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
#include "simdojo/components/vector_reg.h"
#include "util/simd.h"

#include <cstddef>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// Explicit-width alias for IEEE-754 binary32. C++23 has std::float32_t
/// in <stdfloat>; rocjitsu is on C++20 so a local alias.
using float32_t = float;

/// Process-wide, immutable override that disables the SIMD fast path in
/// kernels that have one. Forwards to util::force_scalar() (read once from
/// RJ_FORCE_SCALAR); returned by value, so there is no mutable global to
/// flip at runtime.
inline bool simd_force_scalar() { return util::force_scalar(); }

/// In-vector VOP3 source modifier (f32), bit-exact with the scalar lambda the
/// generated bodies emit per source: `abs` first (`std::fabs`), then `neg`
/// (`-x`). `abs`/`neg` are the raw VOP3 modifier fields; the bit for source
/// index `SrcIdx` selects whether the modifier applies. std::fabs clears the
/// sign bit and unary minus flips it (both NaN-payload preserving), so the
/// vector form is a pure sign-bit AND/XOR — bit-identical on every input.
template <unsigned SrcIdx>
util::native<float> apply_vop3_src_mod_f32(util::native<float> v, uint32_t abs, uint32_t neg) {
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
util::native<double> apply_vop3_src_mod_f64(util::native<double> v, uint32_t abs, uint32_t neg) {
  using U = util::native<uint64_t>;
  U b = std::bit_cast<U>(v);
  if (abs & (1u << SrcIdx))
    b = b & 0x7FFFFFFFFFFFFFFFull;
  if (neg & (1u << SrcIdx))
    b = b ^ 0x8000000000000000ull;
  return std::bit_cast<util::native<double>>(b);
}

/// In-vector VOP3 destination modifier, bit-exact with the scalar tail: `omod`
/// scales by an exact power of two (1->*2, 2->*4, 3->*0.5; IEEE-exact, no
/// rounding), then `clamp` saturates to [0,1]. The clamp uses ordered compares
/// (`v < 0`, `v > 1`), which are false for NaN, so NaN passes through unchanged —
/// matching `std::clamp(v, T(0), T(1))`. Instantiated for float and double; the
/// `_f32`/`_f64` wrappers below name the two lane types the VOP3 paths use.
template <typename T>
util::native<T> apply_vop3_dst_mod(util::native<T> v, uint32_t omod, uint32_t clamp) {
  if (omod == 1)
    v = v * T(2);
  else if (omod == 2)
    v = v * T(4);
  else if (omod == 3)
    v = v * T(0.5);
  if (clamp) {
    util::stdx::where(v < T(0), v) = T(0);
    util::stdx::where(v > T(1), v) = T(1);
  }
  return v;
}

inline util::native<double> apply_vop3_dst_mod_f64(util::native<double> v, uint32_t omod,
                                                   uint32_t clamp) {
  return apply_vop3_dst_mod<double>(v, omod, clamp);
}

inline util::native<float> apply_vop3_dst_mod_f32(util::native<float> v, uint32_t omod,
                                                  uint32_t clamp) {
  return apply_vop3_dst_mod<float>(v, omod, clamp);
}

/// Resolve an operand's read-side VGPR storage to a `VgprStorage*` (the
/// per-lane register object), or null for a non-VGPR operand (SGPR / immediate
/// / inline-const / DPP delegate). This is the single virtual dispatch
/// (`simd_vgpr_storage`) the SIMD hot path takes to bind a source register; the
/// per-chunk loop then issues a value-semantic `r->simd_load<T>(base)` directly
/// off the resolved (loop-invariant) object with no further dispatch and no raw
/// pointer in the kernel body — the storage pointer never escapes `VgprStorage`.
template <typename Op> const VgprStorage *simd_src_reg(const Op &op, const Wavefront &wf) {
  const VgprStorage *r = SimdAccess::vgpr_storage(op, wf);
  if (r)
    SimdAccess::notify_read(op, wf, 0, wf.wf_size(), 0xF);
  return r;
}

/// Mutable counterpart of `simd_src_reg` for the dst write path (single
/// `simd_vgpr_storage_mut` dispatch).
template <typename Op> VgprStorage *simd_dst_reg(const Op &op, Wavefront &wf) {
  return SimdAccess::vgpr_storage_mut(op, wf);
}

/// Resolve a 64-bit-lane source operand's lo/hi register pair in one virtual
/// dispatch (`simd_vgpr_storage64`). `{lo, hi}` are `VgprStorage*` (lo = reg N,
/// hi = reg N+1), or `{nullptr, nullptr}` for a non-VGPR operand. The glue's
/// 64-bit kernels structured-bind this and issue value-semantic
/// `lo->simd_load64<T>(*hi, base)` — no raw pointer in the kernel body.
template <typename Op> ConstVgprStoragePair64 simd_src_reg64(const Op &op, const Wavefront &wf) {
  ConstVgprStoragePair64 p = SimdAccess::vgpr_storage64(op, wf);
  if (p.lo)
    SimdAccess::notify_read(op, wf, 0, wf.wf_size(), 0xF);
  return p;
}

/// Mutable counterpart of `simd_src_reg64` for the 64-bit dst write path
/// (single `simd_vgpr_storage64_mut` dispatch).
template <typename Op> VgprStoragePair64 simd_dst_reg64(const Op &op, Wavefront &wf) {
  return SimdAccess::vgpr_storage64_mut(op, wf);
}

/// Value-semantic 32-bit load of a resolved source register at `base`: a
/// contiguous `r->simd_load<T>(base)` when `r` is per-lane VGPR storage,
/// otherwise the pre-broadcast scalar `bcast`. The resolved `r` is loop-
/// invariant (hoisted before the chunk loop), so this is a plain null test +
/// inlined load — no raw pointer escapes the kernel body.
template <typename T>
util::native<T> simd_load_or(const VgprStorage *r, uint32_t base, util::native<T> bcast) {
  return r ? r->template simd_load<T>(base) : bcast;
}

/// Narrow (native_width64-wide) counterpart of `simd_load_or` for the
/// mixed-width f64<->32-bit glue.
template <typename T>
util::narrow32<T> simd_load_narrow_or(const VgprStorage *r, uint32_t base,
                                      util::narrow32<T> bcast) {
  return r ? r->template simd_load_narrow<T>(base) : bcast;
}

/// Value-semantic 64-bit load of a resolved source register pair at `base`: a
/// combined `lo->simd_load64<T>(*hi, base)` when the pair is per-lane VGPR
/// storage, otherwise the pre-broadcast scalar `bcast`. The resolved pair is
/// loop-invariant — no raw pointer escapes the kernel body.
template <typename T>
util::native<T> simd_load64_or(const ConstVgprStoragePair64 &p, uint32_t base,
                               util::native<T> bcast) {
  return p.lo ? p.lo->template simd_load64<T>(*p.hi, base) : bcast;
}

/// Mutable-pair overload of `simd_load64_or`, for the dst-accumulate (fmac) f64
/// paths whose vdst pair is both the accumulator source and the destination.
template <typename T>
util::native<T> simd_load64_or(const VgprStoragePair64 &p, uint32_t base, util::native<T> bcast) {
  return p.lo ? p.lo->template simd_load64<T>(*p.hi, base) : bcast;
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
  if (const VgprStorage *r = SimdAccess::vgpr_storage(op, wf)) {
    constexpr auto W = static_cast<uint32_t>(util::native_width_v<T>);
    SimdAccess::notify_read(op, wf, lane_base, lane_base + W, 0xF);
    return r->template simd_load<T>(lane_base);
  }
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
  if (VgprStorage *r = SimdAccess::vgpr_storage_mut(op, wf)) {
    r->template simd_store<T>(lane_base, v, mask);
    return;
  }
  alignas(util::native<T>) uint32_t buf[W];
  util::blit_to_buffer<T>(buf, v);
  op.write_lane_chunk(wf, lane_base, static_cast<uint32_t>(W), buf, mask);
}

/// Pre-resolved-register counterpart of write_simd, for the 32-bit fast paths
/// that resolve the dst's `VgprStorage*` ONCE before the chunk loop and issue a
/// value-semantic `rd->simd_store<T>(base, ...)` per chunk. `rd` is that hoisted
/// (loop-invariant) register object, or null when the dst is not contiguous VGPR
/// storage (then fall back to the operand's write_lane_chunk). Identical
/// masked-store / fallback semantics to write_simd — the 32-bit sibling of
/// write_simd64_at. No raw pointer escapes the kernel body.
template <typename T, typename Op>
  requires(util::has_stdx_simd)
inline void write_simd_at(VgprStorage *rd, const Op &op, Wavefront &wf, uint32_t base,
                          util::native<T> v, uint64_t mask) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  constexpr std::size_t W = util::native_width_v<T>;
  if (rd) {
    rd->template simd_store<T>(base, v, mask);
    return;
  }
  alignas(util::native<T>) uint32_t buf[W];
  util::blit_to_buffer<T>(buf, v);
  op.write_lane_chunk(wf, base, static_cast<uint32_t>(W), buf, mask);
}

/// Narrow (native_width64-wide) counterpart of write_simd_at, for the f64->32-bit
/// cvt dst path: value-semantic `rd->simd_store_narrow<T>(base, ...)` when the dst
/// resolves to per-lane VGPR storage, else spill to a uint32 chunk buffer and fall
/// back to the operand's write_lane_chunk. No raw pointer escapes the kernel body.
template <typename T, typename Op>
  requires(util::has_stdx_simd)
inline void write_simd_narrow_at(VgprStorage *rd, const Op &op, Wavefront &wf, uint32_t base,
                                 util::narrow32<T> v, uint64_t mask) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  constexpr std::size_t W = util::native_width64;
  if (rd) {
    rd->template simd_store_narrow<T>(base, v, mask);
    return;
  }
  alignas(util::narrow32<T>) T vals[W];
  v.copy_to(vals, util::stdx::vector_aligned);
  uint32_t buf[W];
  for (std::size_t i = 0; i < W; ++i)
    buf[i] = std::bit_cast<uint32_t>(vals[i]);
  op.write_lane_chunk(wf, base, static_cast<uint32_t>(W), buf, mask);
}

/// 64-bit-lane load of an operand at `lane_base` (T is double / uint64_t).
/// Returns a combined `native<T>` from the operand's split lo/hi VGPR pair when
/// it resolves to per-lane storage; otherwise broadcasts the operand's 64-bit
/// scalar value (`read_scalar64`). Constrained on `util::has_stdx_simd`.
template <typename T, typename Op>
  requires(util::has_stdx_simd)
inline util::native<T> read_simd64(const Op &op, const Wavefront &wf, uint32_t lane_base) {
  static_assert(sizeof(T) == sizeof(uint64_t), "read_simd64: T must be a 64-bit lane type");
  ConstVgprStoragePair64 p = simd_src_reg64(op, wf);
  if (p.lo)
    return p.lo->template simd_load64<T>(*p.hi, lane_base);
  return util::broadcast64<T>(op.read_scalar64(wf));
}

/// Per-half f32 reader for the packed-f32 VOP3P family (v_pk_add/mul/fma_f32).
/// In a VGPR pair {N, N+1}, register N holds the LO f32 of every lane and N+1
/// the HI f32, so each half is a native-width native<float> read of one
/// register (no 64-bit-lane / narrow32 detour). For a non-VGPR source the
/// pk_f32 scalar bodies splat the single 32-bit operand into BOTH halves, so
/// lo == hi == broadcast(read_scalar). The `p.lo != nullptr` gate is bit-exact
/// to the scalar's `encoding_value_ in [256,511]` VGPR-range test.
struct PkF32Halves {
  util::native<float> lo;
  util::native<float> hi;
};
template <typename Op>
  requires(util::has_stdx_simd)
inline PkF32Halves read_pkf32_halves(const Op &op, const Wavefront &wf, uint32_t lane_base) {
  ConstVgprStoragePair64 p = simd_src_reg64(op, wf);
  if (p.lo)
    return {p.lo->template simd_load<float>(lane_base), p.hi->template simd_load<float>(lane_base)};
  const util::native<float> b = util::broadcast<float>(op.read_scalar(wf));
  return {b, b};
}

/// In-vector f32 sign flip (neg modifier): XOR the sign bit. Bit-exact to the
/// scalar `x = -x` for all values incl. ±0 / ±Inf / NaN (payload preserved).
inline util::native<float> pkf32_neg(util::native<float> v, bool do_neg) {
  if (!do_neg)
    return v;
  return std::bit_cast<util::native<float>>(std::bit_cast<util::native<uint32_t>>(v) ^
                                            util::native<uint32_t>(0x80000000u));
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
  VgprStoragePair64 p = simd_dst_reg64(op, wf);
  if (p.lo) {
    p.lo->template simd_store64<T>(*p.hi, lane_base, v, mask);
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

/// Pre-resolved-register counterpart of write_simd64, for the fast paths that
/// resolve the dst's split lo/hi `VgprStorage*` pair ONCE (simd_dst_reg64)
/// before the chunk loop and issue a value-semantic
/// `p.lo->simd_store64<T>(*p.hi, base, ...)` per chunk. `p` is that hoisted
/// (loop-invariant) register pair, or `{nullptr, nullptr}` when the dst is not
/// contiguous VGPR storage (then fall back to the operand's write_lane64).
/// Identical masked-store / fallback semantics to write_simd64. No raw pointer
/// escapes the kernel body.
template <typename T, typename Op>
  requires(util::has_stdx_simd)
inline void write_simd64_at(VgprStoragePair64 p, const Op &op, Wavefront &wf, uint32_t base,
                            util::native<T> v, uint64_t mask) {
  static_assert(sizeof(T) == sizeof(uint64_t));
  constexpr std::size_t W = util::native_width64;
  if (p.lo) {
    p.lo->template simd_store64<T>(*p.hi, base, v, mask);
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
      op.write_lane64(wf, base + static_cast<uint32_t>(i), buf[i]);
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
  // Resolve each operand's `VgprStorage*` ONCE (a single virtual
  // simd_vgpr_storage dispatch -> resolved_vgpr_offset -> the localized
  // vgpr_reg<64> cast). The resolved register object is chunk-independent, so the
  // per-chunk loop issues a value-semantic `r->simd_load<T>(base)` off it instead
  // of re-dispatching the resolution every chunk — and no raw base pointer
  // escapes the kernel body. Operands that aren't contiguous VGPR storage
  // (SGPR/imm/inline-const) resolve to null and broadcast a single scalar read; a
  // null dst falls back to write_lane_chunk.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.vsrc1, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.vsrc1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<T>(r0, base, a_bcast);
    const auto b = simd_load_or<T>(r1, base, b_bcast);
    const auto r = bin_op(a, b);
    write_simd_at<T>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback selected when `util::has_stdx_simd` is false.
/// Trivially inlined to `return false;` so the generated probe at the
/// call site costs nothing on toolchains without `<experimental/simd>`.
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop2_simd(Inst &, Wavefront &, BinOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<Tin>{} : util::broadcast<Tin>(inst.src0.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<Tin>(r0, base, a_bcast);
    const auto r = un_op(a);
    write_simd_at<Tout>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the unary path; see the binary-path note above.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop1_simd(Inst &, Wavefront &, UnOp) {
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
SimdCarry<Value, Mask> make_simd_carry(Value value, Mask carry) {
  return {value, carry};
}

/// VOP2 carry SIMD fast path (v_add_co/sub_co/subrev_co/addc/subb/subbrev_u32).
/// The lane type is fixed to uint32_t. `carry_op` is invoked as
///   carry_op(native<uint32_t> src0, native<uint32_t> vsrc1, native<uint32_t> cin)
///     -> SimdCarry<native<uint32_t>, mask>
/// where `cin` carries the incoming VCC bit (0/1 per lane); ops without a
/// carry-in ignore it. The result is masked-stored to vdst and the carry mask
/// is merged into VCC at the chunk's bit offset for active EXEC lanes only —
/// inactive-lane VCC bits are zeroed, matching the scalar bodies.
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
  // Carry-in reads the incoming VCC; the result accumulates from zero so that
  // inactive lanes are zeroed (matching hardware and the scalar bodies).
  const uint64_t vcc_in = wf.vcc();
  uint64_t vcc_out = 0;
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.vsrc1, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.vsrc1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<T>(r0, base, a_bcast);
    const auto b = simd_load_or<T>(r1, base, b_bcast);
    // Expand the incoming VCC bits for this chunk to a 0/1-per-lane vector.
    const uint64_t cin_bits = (vcc_in >> base) & chunk_full;
    alignas(util::native<T>) uint32_t cinbuf[W];
    for (std::size_t i = 0; i < W; ++i)
      cinbuf[i] = static_cast<uint32_t>((cin_bits >> i) & 1u);
    const auto cin = util::load<T>(cinbuf);
    const auto r = carry_op(a, b, cin);
    write_simd_at<T>(rd, inst.vdst, wf, base, r.value, chunk);
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
[[nodiscard]] bool try_execute_binary_vop2_carry_simd(Inst &, Wavefront &, CarryOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd. vdst
  // is both the third (accumulate) source and the destination — one pointer.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.vsrc1, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.vsrc1.read_scalar(wf));
  const auto d_bcast = rd ? util::native<T>{} : util::broadcast<T>(inst.vdst.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<T>(r0, base, a_bcast);
    const auto b = simd_load_or<T>(r1, base, b_bcast);
    const auto d = simd_load_or<T>(rd, base, d_bcast); // dst-accumulate source
    const auto r = fma_op(a, b, d, k);
    write_simd_at<T>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the ternary path; see the binary-path note above.
template <typename T, typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop2_simd(Inst &, Wavefront &, util::native<T>, FmaOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd. vdst
  // is both the dst-accumulate source and the destination — one lo/hi pair.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const ConstVgprStoragePair64 rs1 = simd_src_reg64(inst.vsrc1, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto a_bcast =
      rs0.lo ? util::native<T>{} : util::broadcast64<T>(inst.src0.read_scalar64(wf));
  const auto b_bcast =
      rs1.lo ? util::native<T>{} : util::broadcast64<T>(inst.vsrc1.read_scalar64(wf));
  const auto d_bcast =
      rd64.lo ? util::native<T>{} : util::broadcast64<T>(inst.vdst.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load64_or<T>(rs0, base, a_bcast);
    const auto b = simd_load64_or<T>(rs1, base, b_bcast);
    const auto d = simd_load64_or<T>(rd64, base, d_bcast); // dst-accumulate
    write_simd64_at<T>(rd64, inst.vdst, wf, base, fma_op(a, b, d), chunk);
  }
  return true;
}

/// Unconstrained fallback for the f64 ternary path; see the binary-path note.
template <typename T, typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop2_f64_simd(Inst &, Wavefront &, FmaOp) {
  return false;
}

/// 64-bit-lane VOP2 binary SIMD fast path (v_add_f64 / v_mul_f64 /
/// v_max_num_f64 / v_min_num_f64). VOP2 has no abs/neg/omod/clamp fields and
/// reads its second source as `vsrc1` (not `src1`); otherwise identical to the
/// f64 FMA vop2 path minus the dst-accumulate operand. add/mul are bit-exact;
/// fmax/fmin carry the accepted NaN-payload / signed-zero-tie carve-out (same as
/// the f64 vop3 binary path and every other min/max).
template <typename Inst, typename BinOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop2_f64_simd(Inst &inst, Wavefront &wf,
                                                           BinOp bin_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vsrc1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = double;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const ConstVgprStoragePair64 rs1 = simd_src_reg64(inst.vsrc1, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto a_bcast =
      rs0.lo ? util::native<T>{} : util::broadcast64<T>(inst.src0.read_scalar64(wf));
  const auto b_bcast =
      rs1.lo ? util::native<T>{} : util::broadcast64<T>(inst.vsrc1.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load64_or<T>(rs0, base, a_bcast);
    const auto b = simd_load64_or<T>(rs1, base, b_bcast);
    write_simd64_at<T>(rd64, inst.vdst, wf, base, bin_op(a, b), chunk);
  }
  return true;
}

/// Unconstrained fallback for the f64 vop2 binary path.
template <typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop2_f64_simd(Inst &, Wavefront &, BinOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto a_bcast =
      rs0.lo ? util::native<T>{} : util::broadcast64<T>(inst.src0.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load64_or<T>(rs0, base, a_bcast);
    write_simd64_at<T>(rd64, inst.vdst, wf, base, un_op(a), chunk);
  }
  return true;
}

/// Unconstrained fallback for the f64 unary path; see the binary-path note.
template <typename T, typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop1_f64_simd(Inst &, Wavefront &, UnOp) {
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
  if (const VgprStorage *r = SimdAccess::vgpr_storage(op, wf)) {
    constexpr auto W = static_cast<uint32_t>(util::native_width64);
    SimdAccess::notify_read(op, wf, lane_base, lane_base + W, 0xF);
    return r->template simd_load_narrow<T>(lane_base);
  }
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd. The
  // dst is 32-bit (native_width64 narrow lanes) while the source is 64-bit.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto s_bcast =
      rs0.lo ? util::native<double>{} : util::broadcast64<double>(inst.src0.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = simd_load64_or<double>(rs0, base, s_bcast);
    const util::narrow32<Tout> r = cvt_op(s);
    // Value-semantic narrow store (per-lane bit_cast spill + write_lane_chunk
    // fallback for a non-VGPR dst lives inside write_simd_narrow_at).
    write_simd_narrow_at<Tout>(rd, inst.vdst, wf, base, r, chunk);
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd. The
  // source is 32-bit (native_width64 narrow lanes) while the dst is 64-bit.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto in_bcast =
      r0 ? util::narrow32<Tin>{} : util::broadcast_narrow<Tin>(inst.src0.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto in = simd_load_narrow_or<Tin>(r0, base, in_bcast);
    write_simd64_at<double>(rd64, inst.vdst, wf, base, cvt_op(in), chunk);
  }
  return true;
}

/// Unconstrained fallback for the b32->f64 cvt path; see the binary-path note.
template <typename Tin, typename Inst, typename CvtOp>
[[nodiscard]] bool try_execute_cvt_b32_to_f64_simd(Inst &, Wavefront &, CvtOp) {
  return false;
}

/// VOP3 mixed-width f64-source -> 32-bit-fp-dst fast path: the 64-bit-in /
/// 32-bit-out counterpart of the f64 unary FP glue, for v_frexp_exp_i32_f64
/// (the lone f64 VOP3 cvt whose body keeps the modifiers around an
/// f64 -> float(exp) -> bit_cast tail). Reads src0 as native<double>
/// (native_width64 lanes), applies src0 abs/neg in the f64 domain, runs cvt_op
/// (native<double> -> narrow32<float> = the exponent as float), then applies the
/// result omod/clamp INLINE at narrow32 width (apply_vop3_dst_mod_f32 is
/// native<float>-wide — a different width than narrow32<float>), and stores the
/// 32-bit results. All steps bit-exact: frexp_exp_f64_simd is the proven VOP1
/// helper (op 48), apply_vop3_src_mod_f64 is the same helper the tested f64
/// unary ops use, and the (float)(uint32) cast + power-of-two omod + ordered
/// clamp mirror the scalar tail (the exp is always finite, so NaN handling is
/// moot).
template <typename Inst, typename CvtOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_cvt_vop3_f64_to_b32_fp_simd(Inst &inst, Wavefront &wf,
                                                                  CvtOp cvt_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vdst.simd_capable())
    return false;
  const uint32_t abs = inst.inst_.abs;
  const uint32_t neg = inst.inst_.neg;
  const uint32_t omod = inst.inst_.omod;
  const uint32_t clamp = inst.inst_.clamp;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto s_bcast =
      rs0.lo ? util::native<double>{} : util::broadcast64<double>(inst.src0.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = apply_vop3_src_mod_f64<0>(simd_load64_or<double>(rs0, base, s_bcast), abs, neg);
    util::narrow32<float> v = cvt_op(s);
    // Inline omod/clamp at narrow32<float> width (mirrors apply_vop3_dst_mod):
    // IEEE-exact power-of-two scale, ordered-compare saturation to [0,1].
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
    write_simd_narrow_at<float>(rd, inst.vdst, wf, base, v, chunk);
  }
  return true;
}

template <typename Inst, typename CvtOp>
[[nodiscard]] bool try_execute_cvt_vop3_f64_to_b32_fp_simd(Inst &, Wavefront &, CvtOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.vsrc1, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.vsrc1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<T>(r0, base, a_bcast);
    const auto b = simd_load_or<T>(r1, base, b_bcast);
    const uint64_t sel_bits = (vcc >> base) & chunk_full;
    alignas(util::native<T>) uint32_t selbuf[W];
    for (std::size_t i = 0; i < W; ++i)
      selbuf[i] = static_cast<uint32_t>((sel_bits >> i) & 1u);
    auto r = a;
    util::stdx::where(util::load<T>(selbuf) != 0u, r) = b;
    write_simd_at<T>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the cndmask path; see the binary-path note above.
template <typename Inst> [[nodiscard]] bool try_execute_cndmask_vop2_simd(Inst &, Wavefront &) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<T>(r0, base, a_bcast);
    const auto b = simd_load_or<T>(r1, base, b_bcast);
    const uint64_t sel_bits = (sel64 >> base) & chunk_full;
    alignas(util::native<T>) uint32_t selbuf[W];
    for (std::size_t i = 0; i < W; ++i)
      selbuf[i] = static_cast<uint32_t>((sel_bits >> i) & 1u);
    auto r = a;
    util::stdx::where(util::load<T>(selbuf) != 0u, r) = b;
    write_simd_at<T>(rd, inst.vdst, wf, base, r, chunk);
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
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const uint64_t sel64 = inst.src2.read_scalar64(wf);
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<T>(r0, base, a_bcast);
    const auto b = simd_load_or<T>(r1, base, b_bcast);
    const uint64_t sel_bits = (sel64 >> base) & chunk_full;
    alignas(util::native<T>) uint32_t selbuf[W];
    for (std::size_t i = 0; i < W; ++i)
      selbuf[i] = static_cast<uint32_t>((sel_bits >> i) & 1u);
    auto r = a;
    util::stdx::where(util::load<T>(selbuf) != 0u, r) = b;
    r = r & util::native<T>(0xFFFFu);
    write_simd_at<T>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_cndmask_b16_vop3_simd(Inst &, Wavefront &) {
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
  uint64_t vcc = 0;
  // Resolve source base pointers once; see try_execute_binary_vop2_simd. VOPC
  // writes only the VCC mask, so there is no dst pointer to hoist.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.vsrc1, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.vsrc1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<T>(r0, base, a_bcast);
    const auto b = simd_load_or<T>(r1, base, b_bcast);
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
[[nodiscard]] bool try_execute_vopc_simd(Inst &, Wavefront &, CmpOp) {
  return false;
}

/// 64-bit-lane VOPC compare SIMD fast path (f64/i64/u64 relations). Identical to
/// try_execute_vopc_simd but reads each operand as `native<T>` (T = double /
/// int64_t / uint64_t) through the split lo/hi VGPR-pair path (read_simd64), so
/// it processes `native_width64` lanes per chunk. Same VCC merge.
template <typename T, typename Inst, typename CmpOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vopc64_simd(Inst &inst, Wavefront &wf, CmpOp cmp_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vsrc1.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  uint64_t vcc = 0;
  // Resolve source base pointers once; see try_execute_binary_vop2_simd. VOPC
  // writes only the VCC mask, so there is no dst pointer to hoist.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const ConstVgprStoragePair64 rs1 = simd_src_reg64(inst.vsrc1, wf);
  const auto a_bcast =
      rs0.lo ? util::native<T>{} : util::broadcast64<T>(inst.src0.read_scalar64(wf));
  const auto b_bcast =
      rs1.lo ? util::native<T>{} : util::broadcast64<T>(inst.vsrc1.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load64_or<T>(rs0, base, a_bcast);
    const auto b = simd_load64_or<T>(rs1, base, b_bcast);
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
[[nodiscard]] bool try_execute_vopc64_simd(Inst &, Wavefront &, CmpOp) {
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
  uint64_t vcc = 0;
  // Resolve source base pointers once; see try_execute_binary_vop2_simd. VOPC
  // writes only the VCC mask, so there is no dst pointer to hoist.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const VgprStorage *rm = simd_src_reg(inst.vsrc1, wf);
  const auto s_bcast =
      rs0.lo ? util::native<uint64_t>{} : util::broadcast64<uint64_t>(inst.src0.read_scalar64(wf));
  const auto mask_bcast = rm ? util::narrow32<uint32_t>{}
                             : util::broadcast_narrow<uint32_t>(inst.vsrc1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = simd_load64_or<uint64_t>(rs0, base, s_bcast);
    const auto mask = simd_load_narrow_or<uint32_t>(rm, base, mask_bcast);
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
[[nodiscard]] bool try_execute_vopc_class_f64_simd(Inst &, Wavefront &, CmpOp) {
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
  uint64_t vcc = 0;
  // Resolve source base pointers once; see try_execute_binary_vop2_simd. The
  // class test writes only the SGPR-pair mask, so there is no dst pointer to hoist.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto a = simd_load_or<T>(r0, base, a_bcast);
    if (do_abs)
      a = a & ~sm;
    if (do_neg)
      a = a ^ sm;
    const auto b = simd_load_or<T>(r1, base, b_bcast);
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
[[nodiscard]] bool try_execute_vop3_class_b32_simd(Inst &, Wavefront &, uint32_t, CmpOp) {
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
  uint64_t vcc = 0;
  // Resolve source base pointers once; see try_execute_binary_vop2_simd. The
  // class test writes only the SGPR-pair mask, so there is no dst pointer to hoist.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const VgprStorage *rm = simd_src_reg(inst.src1, wf);
  const auto s_bcast =
      rs0.lo ? util::native<uint64_t>{} : util::broadcast64<uint64_t>(inst.src0.read_scalar64(wf));
  const auto mask_bcast =
      rm ? util::narrow32<uint32_t>{} : util::broadcast_narrow<uint32_t>(inst.src1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    auto s = simd_load64_or<uint64_t>(rs0, base, s_bcast);
    if (do_abs)
      s = s & ~sm;
    if (do_neg)
      s = s ^ sm;
    const auto mask = simd_load_narrow_or<uint32_t>(rm, base, mask_bcast);
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
[[nodiscard]] bool try_execute_vop3_class_f64_simd(Inst &, Wavefront &, uint64_t, CmpOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<T>(r0, base, a_bcast);
    const auto b = simd_load_or<T>(r1, base, b_bcast);
    const auto r = bin_op(a, b);
    write_simd_at<T>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 integer binary path; see the VOP2
/// binary-path note above.
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_simd(Inst &, Wavefront &, BinOp) {
  return false;
}

/// VOP3 f16 binary fast path. The packed-f16 binary functors widen f16->f32
/// inside `bin_op` and narrow back, but do NOT apply the VOP3 abs/neg/omod/clamp
/// modifiers (there is no fp16 binary modifier glue, unlike the f32 path). The
/// generated scalar body DOES apply them around the f16<->f32 round trip, so bail
/// to scalar whenever any modifier field is set; the (common) unmodified case
/// still takes the integer fast path.
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_f16_simd(Inst &inst, Wavefront &wf, BinOp bin_op) {
  if (inst.inst_.abs != 0u || inst.inst_.neg != 0u || inst.inst_.omod != 0u ||
      inst.inst_.clamp != 0u)
    return false;
  return try_execute_binary_vop3_simd<T>(inst, wf, bin_op);
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(simd_load_or<T>(r0, base, a_bcast), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(simd_load_or<T>(r1, base, b_bcast), abs, neg);
    const auto r = apply_vop3_dst_mod_f32(bin_op(a, b), omod, clamp);
    write_simd_at<T>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f32 binary path.
template <typename T, typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_fp_simd(Inst &, Wavefront &, BinOp) {
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
  uint64_t dst = 0;
  // Resolve source base pointers once; see try_execute_binary_vop2_simd. The
  // compare writes only the SGPR-pair mask, so there is no dst pointer to hoist.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<T>(r0, base, a_bcast);
    const auto b = simd_load_or<T>(r1, base, b_bcast);
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
[[nodiscard]] bool try_execute_vopc_vop3_int_simd(Inst &, Wavefront &, CmpOp) {
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
  uint64_t dst = 0;
  // Resolve source base pointers once; see try_execute_binary_vop2_simd. The
  // compare writes only the SGPR-pair mask, so there is no dst pointer to hoist.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const ConstVgprStoragePair64 rs1 = simd_src_reg64(inst.src1, wf);
  const auto a_bcast =
      rs0.lo ? util::native<T>{} : util::broadcast64<T>(inst.src0.read_scalar64(wf));
  const auto b_bcast =
      rs1.lo ? util::native<T>{} : util::broadcast64<T>(inst.src1.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load64_or<T>(rs0, base, a_bcast);
    const auto b = simd_load64_or<T>(rs1, base, b_bcast);
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
[[nodiscard]] bool try_execute_vopc64_vop3_int_simd(Inst &, Wavefront &, CmpOp) {
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
  uint64_t dst = 0;
  // Resolve source base pointers once; see try_execute_binary_vop2_simd. The
  // compare writes only the SGPR-pair mask, so there is no dst pointer to hoist.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(simd_load_or<T>(r0, base, a_bcast), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(simd_load_or<T>(r1, base, b_bcast), abs, neg);
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
[[nodiscard]] bool try_execute_vopc_vop3_fp32_simd(Inst &, Wavefront &, CmpOp) {
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
  uint64_t dst = 0;
  // Resolve source base pointers once; see try_execute_binary_vop2_simd. The
  // compare writes only the SGPR-pair mask, so there is no dst pointer to hoist.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(
        util::f16_to_f32_simd(simd_load_or<T>(r0, base, a_bcast)), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(
        util::f16_to_f32_simd(simd_load_or<T>(r1, base, b_bcast)), abs, neg);
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
[[nodiscard]] bool try_execute_vopc_vop3_fp16_simd(Inst &, Wavefront &, CmpOp) {
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
  uint64_t dst = 0;
  // Resolve source base pointers once; see try_execute_binary_vop2_simd. The
  // compare writes only the SGPR-pair mask, so there is no dst pointer to hoist.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const ConstVgprStoragePair64 rs1 = simd_src_reg64(inst.src1, wf);
  const auto a_bcast =
      rs0.lo ? util::native<T>{} : util::broadcast64<T>(inst.src0.read_scalar64(wf));
  const auto b_bcast =
      rs1.lo ? util::native<T>{} : util::broadcast64<T>(inst.src1.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(simd_load64_or<T>(rs0, base, a_bcast), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(simd_load64_or<T>(rs1, base, b_bcast), abs, neg);
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
[[nodiscard]] bool try_execute_vopc64_vop3_fp64_simd(Inst &, Wavefront &, CmpOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const ConstVgprStoragePair64 rs1 = simd_src_reg64(inst.src1, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto a_bcast =
      rs0.lo ? util::native<T>{} : util::broadcast64<T>(inst.src0.read_scalar64(wf));
  const auto b_bcast =
      rs1.lo ? util::native<T>{} : util::broadcast64<T>(inst.src1.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(simd_load64_or<T>(rs0, base, a_bcast), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(simd_load64_or<T>(rs1, base, b_bcast), abs, neg);
    const auto r = apply_vop3_dst_mod_f64(bin_op(a, b), omod, clamp);
    write_simd64_at<T>(rd64, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f64 binary path; see the binary-path note.
template <typename Inst, typename BinOp>
[[nodiscard]] bool try_execute_binary_vop3_fp64_simd(Inst &, Wavefront &, BinOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto a_bcast =
      rs0.lo ? util::native<T>{} : util::broadcast64<T>(inst.src0.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(simd_load64_or<T>(rs0, base, a_bcast), abs, neg);
    const auto r = apply_vop3_dst_mod_f64(un_op(a), omod, clamp);
    write_simd64_at<T>(rd64, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f64 unary path; see the binary-path note.
template <typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop3_fp64_simd(Inst &, Wavefront &, UnOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto in = util::f16_to_f32_simd(simd_load_or<T>(r0, base, a_bcast));
    const auto a = apply_vop3_src_mod_f32<0>(in, abs, neg);
    const auto r = apply_vop3_dst_mod_f32(un_op(a), omod, clamp);
    const auto out = util::f32_to_f16_simd(r);
    write_simd_at<T>(rd, inst.vdst, wf, base, out, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f16 unary path; see the binary-path note.
template <typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop3_fp16_simd(Inst &, Wavefront &, UnOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  const VgprStorage *r2 = simd_src_reg(inst.src2, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  const auto c_bcast = r2 ? util::native<T>{} : util::broadcast<T>(inst.src2.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<T>(r0, base, a_bcast);
    const auto b = simd_load_or<T>(r1, base, b_bcast);
    const auto c = simd_load_or<T>(r2, base, c_bcast);
    const auto r = tern_op(a, b, c);
    write_simd_at<T>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 integer ternary path; see binary-path note.
template <typename T, typename Inst, typename TernOp>
[[nodiscard]] bool try_execute_ternary_vop3_simd(Inst &, Wavefront &, TernOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  const VgprStorage *r2 = simd_src_reg(inst.src2, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  const auto c_bcast = r2 ? util::native<T>{} : util::broadcast<T>(inst.src2.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(simd_load_or<T>(r0, base, a_bcast), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(simd_load_or<T>(r1, base, b_bcast), abs, neg);
    const auto c = apply_vop3_src_mod_f32<2>(simd_load_or<T>(r2, base, c_bcast), abs, neg);
    const auto r = apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp);
    write_simd_at<T>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop3_fp_simd(Inst &, Wavefront &, FmaOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  const VgprStorage *r2 = simd_src_reg(inst.src2, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  const auto c_bcast = r2 ? util::native<T>{} : util::broadcast<T>(inst.src2.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(
        util::f16_to_f32_simd(simd_load_or<T>(r0, base, a_bcast)), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(
        util::f16_to_f32_simd(simd_load_or<T>(r1, base, b_bcast)), abs, neg);
    const auto c = apply_vop3_src_mod_f32<2>(
        util::f16_to_f32_simd(simd_load_or<T>(r2, base, c_bcast)), abs, neg);
    const auto r = apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp);
    const auto out = util::f32_to_f16_simd(r);
    write_simd_at<T>(rd, inst.vdst, wf, base, out, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop3_fp16_simd(Inst &, Wavefront &, FmaOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const ConstVgprStoragePair64 rs1 = simd_src_reg64(inst.src1, wf);
  const ConstVgprStoragePair64 rs2 = simd_src_reg64(inst.src2, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto a_bcast =
      rs0.lo ? util::native<T>{} : util::broadcast64<T>(inst.src0.read_scalar64(wf));
  const auto b_bcast =
      rs1.lo ? util::native<T>{} : util::broadcast64<T>(inst.src1.read_scalar64(wf));
  const auto c_bcast =
      rs2.lo ? util::native<T>{} : util::broadcast64<T>(inst.src2.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(simd_load64_or<T>(rs0, base, a_bcast), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(simd_load64_or<T>(rs1, base, b_bcast), abs, neg);
    const auto c = apply_vop3_src_mod_f64<2>(simd_load64_or<T>(rs2, base, c_bcast), abs, neg);
    const auto r = apply_vop3_dst_mod_f64(tern_op(a, b, c), omod, clamp);
    write_simd64_at<T>(rd64, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_ternary_vop3_fp64_simd(Inst &, Wavefront &, FmaOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd. vdst
  // is both the third (accumulator) source and the destination — one pointer.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  const auto c_bcast = rd ? util::native<T>{} : util::broadcast<T>(inst.vdst.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(simd_load_or<T>(r0, base, a_bcast), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(simd_load_or<T>(r1, base, b_bcast), abs, neg);
    const auto c = simd_load_or<T>(rd, base, c_bcast); // accumulator, NO modifier
    const auto r = apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp);
    write_simd_at<T>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_fmac_vop3_fp_simd(Inst &, Wavefront &, FmaOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd. vdst
  // is both the third (accumulate) source and the destination — one pointer.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  const auto c_bcast = rd ? util::native<T>{} : util::broadcast<T>(inst.vdst.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(
        util::f16_to_f32_simd(simd_load_or<T>(r0, base, a_bcast)), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(
        util::f16_to_f32_simd(simd_load_or<T>(r1, base, b_bcast)), abs, neg);
    const auto c = util::f16_to_f32_simd(simd_load_or<T>(rd, base, c_bcast)); // accum, no mod
    const auto r = apply_vop3_dst_mod_f32(tern_op(a, b, c), omod, clamp);
    const auto out = util::f32_to_f16_simd(r);
    write_simd_at<T>(rd, inst.vdst, wf, base, out, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_fmac_vop3_fp16_simd(Inst &, Wavefront &, FmaOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd. vdst
  // is both the accumulator source and the destination — one lo/hi pair.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const ConstVgprStoragePair64 rs1 = simd_src_reg64(inst.src1, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto a_bcast =
      rs0.lo ? util::native<T>{} : util::broadcast64<T>(inst.src0.read_scalar64(wf));
  const auto b_bcast =
      rs1.lo ? util::native<T>{} : util::broadcast64<T>(inst.src1.read_scalar64(wf));
  const auto c_bcast =
      rd64.lo ? util::native<T>{} : util::broadcast64<T>(inst.vdst.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(simd_load64_or<T>(rs0, base, a_bcast), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(simd_load64_or<T>(rs1, base, b_bcast), abs, neg);
    const auto c = simd_load64_or<T>(rd64, base, c_bcast); // accumulator, no mod
    const auto r = apply_vop3_dst_mod_f64(tern_op(a, b, c), omod, clamp);
    write_simd64_at<T>(rd64, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename FmaOp>
[[nodiscard]] bool try_execute_fmac_vop3_fp64_simd(Inst &, Wavefront &, FmaOp) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *re = simd_src_reg(inst.src1, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto e_bcast =
      re ? util::native<int32_t>{} : util::broadcast<int32_t>(inst.src1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(simd_load_or<T>(r0, base, a_bcast), abs, neg);
    const auto e = simd_load_or<int32_t>(re, base, e_bcast);
    const auto r = apply_vop3_dst_mod_f32(op(a, e), omod, clamp);
    write_simd_at<T>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_ldexp_vop3_fp32_simd(Inst &, Wavefront &, Op) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const VgprStorage *re = simd_src_reg(inst.src1, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto a_bcast =
      rs0.lo ? util::native<T>{} : util::broadcast64<T>(inst.src0.read_scalar64(wf));
  const auto e_bcast =
      re ? util::narrow32<int32_t>{} : util::broadcast_narrow<int32_t>(inst.src1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(simd_load64_or<T>(rs0, base, a_bcast), abs, neg);
    const auto e = simd_load_narrow_or<int32_t>(re, base, e_bcast);
    const auto r = apply_vop3_dst_mod_f64(op(a, e), omod, clamp);
    write_simd64_at<T>(rd64, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_ldexp_vop3_fp64_simd(Inst &, Wavefront &, Op) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<Tin>{} : util::broadcast<Tin>(inst.src0.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(simd_load_or<Tin>(r0, base, a_bcast), abs, neg);
    const auto r = apply_vop3_dst_mod_f32(un_op(a), omod, clamp);
    write_simd_at<Tout>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

/// Unconstrained fallback for the VOP3 f32 unary path.
template <typename Tin, typename Tout, typename Inst, typename UnOp>
[[nodiscard]] bool try_execute_unary_vop3_fp_simd(Inst &, Wavefront &, UnOp) {
  return false;
}

/// v_div_fixup_f32 SIMD helper. Mirrors the scalar `else if` cascade
/// (execute_shared.h:execute_v_div_fixup_f32_vop3): given three already-
/// modified f32 operands `p` (the fma scaffold input), `b` (numerator), and
/// `c` (denominator), pick the result among NaN/Inf/zero copysign cases
/// according to AMD's div_fixup ULP table. The cascade is applied lowest-
/// priority first so that the higher-priority `where` blends overwrite —
/// equivalent to scalar's first-match `else if`; source-2 NaN therefore
/// overwrites source-1 NaN. The sign-of-quotient `b ^ c` is computed once
/// in the integer domain and reinterpreted as
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
  util::stdx::where(b_nan, r) = b;
  util::stdx::where(c_nan, r) = c;
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
  util::stdx::where(b_nan, r) = b;
  util::stdx::where(c_nan, r) = c;
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  const VgprStorage *r2 = simd_src_reg(inst.src2, wf);
  VgprStorage *rd = simd_dst_reg(inst.vdst, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  const auto c_bcast = r2 ? util::native<T>{} : util::broadcast<T>(inst.src2.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f32<0>(simd_load_or<T>(r0, base, a_bcast), abs, neg);
    const auto b = apply_vop3_src_mod_f32<1>(simd_load_or<T>(r1, base, b_bcast), abs, neg);
    const auto c = apply_vop3_src_mod_f32<2>(simd_load_or<T>(r2, base, c_bcast), abs, neg);
    auto r = util::stdx::fma(a, b, c);
    const auto scaled = util::stdx::ldexp(r, shift_32);
    const uint64_t sel_bits = (vcc >> base) & chunk_full;
    alignas(util::native<uint32_t>) uint32_t selbuf[W];
    for (std::size_t i = 0; i < W; ++i)
      selbuf[i] = static_cast<uint32_t>((sel_bits >> i) & 1u);
    const auto vcc_mask_u = util::load<uint32_t>(selbuf) != 0u;
    util::stdx::where(simd_mask_as<float>(vcc_mask_u), r) = scaled;
    write_simd_at<T>(rd, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_div_fmas_f32_simd(Inst &, Wavefront &) {
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
  // Resolve operand base pointers once; see try_execute_binary_vop2_simd.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const ConstVgprStoragePair64 rs1 = simd_src_reg64(inst.src1, wf);
  const ConstVgprStoragePair64 rs2 = simd_src_reg64(inst.src2, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto a_bcast =
      rs0.lo ? util::native<T>{} : util::broadcast64<T>(inst.src0.read_scalar64(wf));
  const auto b_bcast =
      rs1.lo ? util::native<T>{} : util::broadcast64<T>(inst.src1.read_scalar64(wf));
  const auto c_bcast =
      rs2.lo ? util::native<T>{} : util::broadcast64<T>(inst.src2.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = apply_vop3_src_mod_f64<0>(simd_load64_or<T>(rs0, base, a_bcast), abs, neg);
    const auto b = apply_vop3_src_mod_f64<1>(simd_load64_or<T>(rs1, base, b_bcast), abs, neg);
    const auto c = apply_vop3_src_mod_f64<2>(simd_load64_or<T>(rs2, base, c_bcast), abs, neg);
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
    write_simd64_at<T>(rd64, inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_div_fmas_f64_simd(Inst &, Wavefront &) {
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
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  // src0 = 32-bit shift amount (narrow lane), src1 = 64-bit value, dst = 64-bit.
  const VgprStorage *rs = simd_src_reg(inst.src0, wf);
  const ConstVgprStoragePair64 rs1 = simd_src_reg64(inst.src1, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto s_bcast =
      rs ? util::narrow32<uint32_t>{} : util::broadcast_narrow<uint32_t>(inst.src0.read_scalar(wf));
  const auto v_bcast =
      rs1.lo ? util::native<uint64_t>{} : util::broadcast64<uint64_t>(inst.src1.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto s = simd_load_narrow_or<uint32_t>(rs, base, s_bcast);
    const auto v = simd_load64_or<uint64_t>(rs1, base, v_bcast);
    const auto sh = util::stdx::static_simd_cast<util::native<uint64_t>>(s) & 63ull;
    write_simd64_at<uint64_t>(rd64, inst.vdst, wf, base, shift_op(v, sh), chunk);
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
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  // src0 = 64-bit value, src1 = 32-bit shift (narrow), src2 = 64-bit addend.
  const ConstVgprStoragePair64 rs0 = simd_src_reg64(inst.src0, wf);
  const VgprStorage *rs = simd_src_reg(inst.src1, wf);
  const ConstVgprStoragePair64 rs2 = simd_src_reg64(inst.src2, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto v_bcast =
      rs0.lo ? util::native<uint64_t>{} : util::broadcast64<uint64_t>(inst.src0.read_scalar64(wf));
  const auto s_bcast =
      rs ? util::narrow32<uint32_t>{} : util::broadcast_narrow<uint32_t>(inst.src1.read_scalar(wf));
  const auto c_bcast =
      rs2.lo ? util::native<uint64_t>{} : util::broadcast64<uint64_t>(inst.src2.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto v = simd_load64_or<uint64_t>(rs0, base, v_bcast);
    const auto s = simd_load_narrow_or<uint32_t>(rs, base, s_bcast);
    const auto c = simd_load64_or<uint64_t>(rs2, base, c_bcast);
    const auto sh = util::stdx::static_simd_cast<util::native<uint64_t>>(s) & 63ull;
    write_simd64_at<uint64_t>(rd64, inst.vdst, wf, base, (v << sh) + c, chunk);
  }
  return true;
}
template <typename Inst> [[nodiscard]] bool try_execute_lshl_add_u64_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3 wide 32x32->64 multiply-add fast path (v_mad_u64_u32 / v_mad_i64_i32).
/// src0/src1 are 32-bit multiplicands (read as narrow lanes), src2 is the 64-bit
/// addend, dst is 64-bit. `mad_op(s0, s1, c)` receives the two narrow operands
/// and the 64-bit addend and does the (signedness-aware) widen, 64-bit multiply
/// (low 64), and add — matching the scalar `(wide)s0 * (wide)s1 + s2`.
template <typename Inst, typename MadOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_mad_wide64_vop3_simd(Inst &inst, Wavefront &wf,
                                                           MadOp mad_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  const ConstVgprStoragePair64 rs2 = simd_src_reg64(inst.src2, wf);
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  const auto a_bcast =
      r0 ? util::narrow32<uint32_t>{} : util::broadcast_narrow<uint32_t>(inst.src0.read_scalar(wf));
  const auto b_bcast =
      r1 ? util::narrow32<uint32_t>{} : util::broadcast_narrow<uint32_t>(inst.src1.read_scalar(wf));
  const auto c_bcast =
      rs2.lo ? util::native<uint64_t>{} : util::broadcast64<uint64_t>(inst.src2.read_scalar64(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_narrow_or<uint32_t>(r0, base, a_bcast);
    const auto b = simd_load_narrow_or<uint32_t>(r1, base, b_bcast);
    const auto c = simd_load64_or<uint64_t>(rs2, base, c_bcast);
    write_simd64_at<uint64_t>(rd64, inst.vdst, wf, base, mad_op(a, b, c), chunk);
  }
  return true;
}
template <typename Inst, typename MadOp>
[[nodiscard]] bool try_execute_mad_wide64_vop3_simd(Inst &, Wavefront &, MadOp) {
  return false;
}

/// VOP3 carry-OUT binary fast path (v_add_co/sub_co/subrev_co_u32). No carry-in
/// (the SimdCarry functor's third arg is a zero vector); the per-lane carry-out
/// is merged into a copy of the current VCC and written to the SGPR-pair `sdst`
/// (not the fixed VCC) — exactly the scalar body's `inst.sdst.write_scalar64`.
/// `sdst` is an SGPR operand, so it does not participate in the simd_capable
/// gate; src0/src1/vdst do. (These VOP3 forms carry no `src2` member, so the
/// carry-in path lives in the separate _cin glue below.)
template <typename Inst, typename CarryOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_co_simd(Inst &inst, Wavefront &wf,
                                                          CarryOp carry_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  uint64_t carry_out = 0;
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  const auto zero_cin = util::broadcast<T>(0u);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<T>(r0, base, a_bcast);
    const auto b = simd_load_or<T>(r1, base, b_bcast);
    const auto r = carry_op(a, b, zero_cin);
    write_simd<T>(inst.vdst, wf, base, r.value, chunk);
    uint64_t carry_bits = 0;
    for (std::size_t i = 0; i < W; ++i)
      if (r.carry[i])
        carry_bits |= (1ULL << i);
    carry_out = (carry_out & ~(chunk << base)) | ((carry_bits & chunk) << base);
  }
  inst.sdst.write_scalar64(wf, carry_out);
  return true;
}
template <typename Inst, typename CarryOp>
[[nodiscard]] bool try_execute_binary_vop3_co_simd(Inst &, Wavefront &, CarryOp) {
  return false;
}

/// VOP3 carry-IN binary fast path (v_addc_co/subb_co/subbrev_co_u32). Same as
/// the _co glue but the per-lane carry-in is read from the SGPR-pair `src2`
/// (these forms have a src2 member) and expanded to a 0/1-per-lane vector;
/// carry-out goes to `sdst`. src2/sdst are SGPR operands (not gated).
template <typename Inst, typename CarryOp>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_binary_vop3_cin_simd(Inst &inst, Wavefront &wf,
                                                           CarryOp carry_op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const uint64_t cin_all = inst.src2.read_scalar64(wf);
  uint64_t carry_out = 0;
  const VgprStorage *r0 = simd_src_reg(inst.src0, wf);
  const VgprStorage *r1 = simd_src_reg(inst.src1, wf);
  const auto a_bcast = r0 ? util::native<T>{} : util::broadcast<T>(inst.src0.read_scalar(wf));
  const auto b_bcast = r1 ? util::native<T>{} : util::broadcast<T>(inst.src1.read_scalar(wf));
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const auto a = simd_load_or<T>(r0, base, a_bcast);
    const auto b = simd_load_or<T>(r1, base, b_bcast);
    const uint64_t cin_bits = (cin_all >> base) & chunk_full;
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
    carry_out = (carry_out & ~(chunk << base)) | ((carry_bits & chunk) << base);
  }
  inst.sdst.write_scalar64(wf, carry_out);
  return true;
}
template <typename Inst, typename CarryOp>
[[nodiscard]] bool try_execute_binary_vop3_cin_simd(Inst &, Wavefront &, CarryOp) {
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
[[nodiscard]] bool try_execute_vop3p_fma_mix_simd(Inst &, Wavefront &) {
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
[[nodiscard]] bool try_execute_vop3p_pk_binary_int_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3P packed-16 ternary integer SIMD fast path (3-source). Same default-
/// packing gate as the binary form: op_sel == 0, op_sel_hi == 3, and the
/// third source's high-half selector op_sel_hi_2 == 1 (per the scalar body
/// `sel2_hi = inst.inst_.op_sel_hi_2`, which is a single bit — value 1 picks
/// the high half for the high-output computation). For pk_mad_i16/u16 the
/// scalar bodies also do not apply neg/neg_hi/clamp; the SIMD path passes
/// through. Functor receives three u32 packed lane vectors and returns the
/// per-half-masked packed result.
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_ternary_int_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if (inst.inst_.op_sel != 0u || inst.inst_.op_sel_hi != 3u || inst.inst_.op_sel_hi_2 != 1u)
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
    const auto c = read_simd<T>(inst.src2, wf, base);
    const auto r = op(a, b, c);
    write_simd<T>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_ternary_int_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3P packed-16 floating-point binary SIMD fast path (pk_add/mul/min/
/// max_f16 family). Each 32-bit lane holds two f16 values; the SIMD path
/// widens to f32, applies per-half sign-flip from neg/neg_hi, runs the
/// functor in f32, narrows back to f16, and packs. Same default-packing
/// gate as the integer pk family. Scalar bodies for pk_*_f16 do NOT apply
/// clamp (verified inline pk_add_f16 at line 15109, pk_max_f16 at 15519),
/// so the SIMD path also ignores it.
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_binary_fp16_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  if (inst.inst_.op_sel != 0u || inst.inst_.op_sel_hi != 3u)
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  using F = util::native<float>;
  using U = util::native<uint32_t>;
  const U kSignBit(0x80000000u);
  const bool neg0_lo = inst.inst_.neg & 1u;
  const bool neg1_lo = inst.inst_.neg & 2u;
  const bool neg0_hi = inst.inst_.neg_hi & 1u;
  const bool neg1_hi = inst.inst_.neg_hi & 2u;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U raw0 = read_simd<uint32_t>(inst.src0, wf, base);
    const U raw1 = read_simd<uint32_t>(inst.src1, wf, base);
    F a_lo = util::f16_to_f32_simd(raw0 & 0xFFFFu);
    F a_hi = util::f16_to_f32_simd(raw0 >> 16);
    F b_lo = util::f16_to_f32_simd(raw1 & 0xFFFFu);
    F b_hi = util::f16_to_f32_simd(raw1 >> 16);
    if (neg0_lo)
      a_lo = std::bit_cast<F>(std::bit_cast<U>(a_lo) ^ kSignBit);
    if (neg1_lo)
      b_lo = std::bit_cast<F>(std::bit_cast<U>(b_lo) ^ kSignBit);
    if (neg0_hi)
      a_hi = std::bit_cast<F>(std::bit_cast<U>(a_hi) ^ kSignBit);
    if (neg1_hi)
      b_hi = std::bit_cast<F>(std::bit_cast<U>(b_hi) ^ kSignBit);
    const F r_lo = op(a_lo, b_lo);
    const F r_hi = op(a_hi, b_hi);
    const U h_lo = util::f32_to_f16_simd(r_lo);
    const U h_hi = util::f32_to_f16_simd(r_hi);
    const U packed = h_lo | (h_hi << 16);
    write_simd<uint32_t>(inst.vdst, wf, base, packed, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_binary_fp16_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3P packed-16 f16 ternary SIMD fast path (3-source pk_fma_f16). Same
/// default-packing gate as the integer ternary form (op_sel = 0,
/// op_sel_hi = 3, op_sel_hi_2 = 1). Per-source neg / neg_hi bits 0/1/2 for
/// src0/src1/src2 (verified pk_fma_f16 scalar at line 15300-15317). No
/// clamp on pk_fma_f16. NaN-payload divergence between stdx::fma and
/// std::fma accepted (same carve-out as fma_mix slice).
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_ternary_fp16_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if (inst.inst_.op_sel != 0u || inst.inst_.op_sel_hi != 3u || inst.inst_.op_sel_hi_2 != 1u)
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  using F = util::native<float>;
  using U = util::native<uint32_t>;
  const U kSignBit(0x80000000u);
  const bool neg0_lo = inst.inst_.neg & 1u;
  const bool neg1_lo = inst.inst_.neg & 2u;
  const bool neg2_lo = inst.inst_.neg & 4u;
  const bool neg0_hi = inst.inst_.neg_hi & 1u;
  const bool neg1_hi = inst.inst_.neg_hi & 2u;
  const bool neg2_hi = inst.inst_.neg_hi & 4u;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U raw0 = read_simd<uint32_t>(inst.src0, wf, base);
    const U raw1 = read_simd<uint32_t>(inst.src1, wf, base);
    const U raw2 = read_simd<uint32_t>(inst.src2, wf, base);
    F a_lo = util::f16_to_f32_simd(raw0 & 0xFFFFu);
    F a_hi = util::f16_to_f32_simd(raw0 >> 16);
    F b_lo = util::f16_to_f32_simd(raw1 & 0xFFFFu);
    F b_hi = util::f16_to_f32_simd(raw1 >> 16);
    F c_lo = util::f16_to_f32_simd(raw2 & 0xFFFFu);
    F c_hi = util::f16_to_f32_simd(raw2 >> 16);
    if (neg0_lo)
      a_lo = std::bit_cast<F>(std::bit_cast<U>(a_lo) ^ kSignBit);
    if (neg1_lo)
      b_lo = std::bit_cast<F>(std::bit_cast<U>(b_lo) ^ kSignBit);
    if (neg2_lo)
      c_lo = std::bit_cast<F>(std::bit_cast<U>(c_lo) ^ kSignBit);
    if (neg0_hi)
      a_hi = std::bit_cast<F>(std::bit_cast<U>(a_hi) ^ kSignBit);
    if (neg1_hi)
      b_hi = std::bit_cast<F>(std::bit_cast<U>(b_hi) ^ kSignBit);
    if (neg2_hi)
      c_hi = std::bit_cast<F>(std::bit_cast<U>(c_hi) ^ kSignBit);
    const F r_lo = op(a_lo, b_lo, c_lo);
    const F r_hi = op(a_hi, b_hi, c_hi);
    const U h_lo = util::f32_to_f16_simd(r_lo);
    const U h_hi = util::f32_to_f16_simd(r_hi);
    const U packed = h_lo | (h_hi << 16);
    write_simd<uint32_t>(inst.vdst, wf, base, packed, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_ternary_fp16_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3P packed-f32 binary fast path (v_pk_add_f32 / v_pk_mul_f32). In a VGPR
/// pair {N, N+1} register N is the LO f32 of every lane, N+1 the HI f32, so each
/// half is a native-width native<float> read of one register (read_pkf32_halves)
/// and the per-half arithmetic runs at full native width. Default-packing gate
/// (op_sel == 0, op_sel_hi == 3) bails to scalar otherwise — under default
/// packing the lo result comes from the lo halves and hi from the hi halves.
/// neg/neg_hi bits 0/1 sign-flip the respective half. No clamp on any pk_f32
/// scalar body. Non-VGPR sources splat the 32-bit operand into both halves.
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_binary_f32_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  if (inst.inst_.op_sel != 0u || inst.inst_.op_sel_hi != 3u)
    return false;
  constexpr std::size_t W = util::native_width_v<float>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const bool neg0_lo = inst.inst_.neg & 1u;
  const bool neg1_lo = inst.inst_.neg & 2u;
  const bool neg0_hi = inst.inst_.neg_hi & 1u;
  const bool neg1_hi = inst.inst_.neg_hi & 2u;
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const PkF32Halves a = read_pkf32_halves(inst.src0, wf, base);
    const PkF32Halves b = read_pkf32_halves(inst.src1, wf, base);
    const util::native<float> r_lo = op(pkf32_neg(a.lo, neg0_lo), pkf32_neg(b.lo, neg1_lo));
    const util::native<float> r_hi = op(pkf32_neg(a.hi, neg0_hi), pkf32_neg(b.hi, neg1_hi));
    rd64.lo->template simd_store<float>(base, r_lo, chunk);
    rd64.hi->template simd_store<float>(base, r_hi, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_binary_f32_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3P packed-f32 ternary fast path (v_pk_fma_f32). 3-source FMA per half;
/// same per-register native<float> read/write as the binary form. Default-
/// packing gate adds op_sel_hi_2 == 1 (the src2-hi select). neg/neg_hi bits
/// 0/1/2 sign-flip the respective half. No clamp. NaN-input payload divergence
/// between stdx::fma and std::fma accepted (same carve-out as the f16 pk ternary
/// / fma_mix slices).
template <typename Inst, typename Op>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_pk_ternary_f32_simd(Inst &inst, Wavefront &wf, Op op) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if (inst.inst_.op_sel != 0u || inst.inst_.op_sel_hi != 3u || inst.inst_.op_sel_hi_2 != 1u)
    return false;
  constexpr std::size_t W = util::native_width_v<float>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const bool neg0_lo = inst.inst_.neg & 1u;
  const bool neg1_lo = inst.inst_.neg & 2u;
  const bool neg2_lo = inst.inst_.neg & 4u;
  const bool neg0_hi = inst.inst_.neg_hi & 1u;
  const bool neg1_hi = inst.inst_.neg_hi & 2u;
  const bool neg2_hi = inst.inst_.neg_hi & 4u;
  const VgprStoragePair64 rd64 = simd_dst_reg64(inst.vdst, wf);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const PkF32Halves a = read_pkf32_halves(inst.src0, wf, base);
    const PkF32Halves b = read_pkf32_halves(inst.src1, wf, base);
    const PkF32Halves c = read_pkf32_halves(inst.src2, wf, base);
    const util::native<float> r_lo =
        op(pkf32_neg(a.lo, neg0_lo), pkf32_neg(b.lo, neg1_lo), pkf32_neg(c.lo, neg2_lo));
    const util::native<float> r_hi =
        op(pkf32_neg(a.hi, neg0_hi), pkf32_neg(b.hi, neg1_hi), pkf32_neg(c.hi, neg2_hi));
    rd64.lo->template simd_store<float>(base, r_lo, chunk);
    rd64.hi->template simd_store<float>(base, r_hi, chunk);
  }
  return true;
}

template <typename Inst, typename Op>
[[nodiscard]] bool try_execute_vop3p_pk_ternary_f32_simd(Inst &, Wavefront &, Op) {
  return false;
}

/// VOP3P v_pk_mov_b32 SIMD fast path. Default packing only (op_sel == 0,
/// op_sel_hi == 3): the result is a 64-bit pair `(src0_lo, src1_hi)`,
/// where the per-source pair is the consecutive {base, base+1} VGPRs
/// when the encoding points at the VGPR range. SGPR / literal sources
/// broadcast both halves identically (handled by read_simd64's
/// broadcast64 fallback). Reads each src as a 64-bit pair, picks the low
/// half of src0 and the high half of src1 via mask, packs, writes via
/// write_simd64. Functorless / fixed-op.
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_mov_b32_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.vdst.simd_capable())
    return false;
  if (inst.inst_.op_sel != 0u || inst.inst_.op_sel_hi != 3u)
    return false;
  constexpr std::size_t W = util::native_width64;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  using U64 = util::native<uint64_t>;
  const U64 kHiMask(0xFFFFFFFF00000000ULL);
  const U64 kLoMask(0x00000000FFFFFFFFULL);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U64 s0 = read_simd64<uint64_t>(inst.src0, wf, base);
    const U64 s1 = read_simd64<uint64_t>(inst.src1, wf, base);
    const U64 out = (s0 & kLoMask) | (s1 & kHiMask);
    write_simd64<uint64_t>(inst.vdst, wf, base, out, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_vop3p_mov_b32_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3P integer dot-product SIMD fast path (v_dot4_i32_i8 / v_dot4_u32_u8 /
/// v_dot8_i32_i4 / v_dot8_u32_u4 / v_dot2_i32_i16 / v_dot2_u32_u16). Unlike
/// the packed-16 family the destination is a single 32-bit lane (NOT packed)
/// and src2 is a per-lane accumulator; the dot reduction happens *within*
/// each lane, so the fast path vectorizes across lanes (each lane computes
/// its own reduction). ElemBits selects the sub-word width (16/8/4 -> 2/4/8
/// products per lane); Signed selects sign- vs zero-extension and, for the
/// signed forms when inst.clamp is set, a lower clamp to 0 — the scalar
/// std::clamp(sum, 0, INT_MAX) on an int32 sum is just max(sum, 0). The
/// unsigned scalar bodies have NO clamp branch, so the unsigned fast path
/// ignores the clamp bit entirely. Signed accumulation runs in the int32
/// domain (matching the scalar body and the pk_mad_i16 precedent); unsigned
/// in uint32. For the 16-bit forms op_sel / op_sel_hi pick the source halves,
/// so the fast path gates on the default packing (op_sel == 0, op_sel_hi == 3)
/// and bails otherwise; the 8/4-bit scalar bodies ignore op_sel so no gate is
/// needed there.
template <int ElemBits, bool Signed, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_dot_int_simd(Inst &inst, Wavefront &wf) {
  static_assert(ElemBits == 16 || ElemBits == 8 || ElemBits == 4, "dot ElemBits must be 16/8/4");
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if constexpr (ElemBits == 16) {
    if (inst.inst_.op_sel != 0u || inst.inst_.op_sel_hi != 3u)
      return false;
  }
  constexpr int N = 32 / ElemBits;
  constexpr uint32_t kElemMask = (ElemBits == 16) ? 0xFFFFu : (ElemBits == 8) ? 0xFFu : 0xFu;
  constexpr int kShift = 32 - ElemBits;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const bool clamp = inst.inst_.clamp;
  using U = util::native<uint32_t>;
  using I = util::native<int32_t>;
  const I kZeroI(0);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U raw0 = read_simd<uint32_t>(inst.src0, wf, base);
    const U raw1 = read_simd<uint32_t>(inst.src1, wf, base);
    const U acc = read_simd<uint32_t>(inst.src2, wf, base);
    if constexpr (Signed) {
      I sum = std::bit_cast<I>(acc);
      for (int i = 0; i < N; ++i) {
        const U ea = (raw0 >> (i * ElemBits)) & kElemMask;
        const U eb = (raw1 >> (i * ElemBits)) & kElemMask;
        const I a = (util::stdx::static_simd_cast<I>(ea) << kShift) >> kShift;
        const I b = (util::stdx::static_simd_cast<I>(eb) << kShift) >> kShift;
        sum += a * b;
      }
      if (clamp)
        util::stdx::where(sum < kZeroI, sum) = kZeroI;
      write_simd<uint32_t>(inst.vdst, wf, base, std::bit_cast<U>(sum), chunk);
    } else {
      U sum = acc;
      for (int i = 0; i < N; ++i) {
        const U ea = (raw0 >> (i * ElemBits)) & kElemMask;
        const U eb = (raw1 >> (i * ElemBits)) & kElemMask;
        sum += ea * eb;
      }
      write_simd<uint32_t>(inst.vdst, wf, base, sum, chunk);
    }
  }
  return true;
}

template <int ElemBits, bool Signed, typename Inst>
[[nodiscard]] bool try_execute_vop3p_dot_int_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3P v_dot2_f32_f16 SIMD fast path. Two f16 products plus an f32
/// accumulator collapse into a single f32 lane: result = a0*b0 + a1*b1 + acc
/// (plain `*` / `+`, left-to-right — matching the scalar, NOT a contracted
/// fma). op_sel / op_sel_hi pick the f16 halves of src0/src1 (gated to the
/// default packing op_sel == 0 && op_sel_hi == 3); neg / neg_hi flip the
/// src0/src1 product-operand signs and neg bit 2 flips the accumulator.
/// Optional clamp to [0, 1]. NaN-input payload divergence accepted (same
/// carve-out as the pk_fma slices).
template <typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_dot_f16_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if (inst.inst_.op_sel != 0u || inst.inst_.op_sel_hi != 3u)
    return false;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const bool clamp = inst.inst_.clamp;
  using F = util::native<float>;
  using U = util::native<uint32_t>;
  const F kZero(0.0f);
  const F kOne(1.0f);
  const U kSignBit(0x80000000u);
  const bool neg_a0 = inst.inst_.neg & 1u;
  const bool neg_b0 = inst.inst_.neg & 2u;
  const bool neg_acc = inst.inst_.neg & 4u;
  const bool neg_a1 = inst.inst_.neg_hi & 1u;
  const bool neg_b1 = inst.inst_.neg_hi & 2u;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U raw0 = read_simd<uint32_t>(inst.src0, wf, base);
    const U raw1 = read_simd<uint32_t>(inst.src1, wf, base);
    F acc = read_simd<float>(inst.src2, wf, base);
    F a0 = util::f16_to_f32_simd(raw0 & 0xFFFFu);
    F a1 = util::f16_to_f32_simd(raw0 >> 16);
    F b0 = util::f16_to_f32_simd(raw1 & 0xFFFFu);
    F b1 = util::f16_to_f32_simd(raw1 >> 16);
    if (neg_a0)
      a0 = std::bit_cast<F>(std::bit_cast<U>(a0) ^ kSignBit);
    if (neg_b0)
      b0 = std::bit_cast<F>(std::bit_cast<U>(b0) ^ kSignBit);
    if (neg_a1)
      a1 = std::bit_cast<F>(std::bit_cast<U>(a1) ^ kSignBit);
    if (neg_b1)
      b1 = std::bit_cast<F>(std::bit_cast<U>(b1) ^ kSignBit);
    if (neg_acc)
      acc = std::bit_cast<F>(std::bit_cast<U>(acc) ^ kSignBit);
    F r = a0 * b0 + a1 * b1 + acc;
    if (clamp) {
      util::stdx::where(r < kZero, r) = kZero;
      util::stdx::where(r > kOne, r) = kOne;
    }
    write_simd<float>(inst.vdst, wf, base, r, chunk);
  }
  return true;
}

template <typename Inst> [[nodiscard]] bool try_execute_vop3p_dot_f16_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP3P mixed-sign integer dot product (v_dot4_i32_iu8 / v_dot8_i32_iu4).
/// Same structure as try_execute_vop3p_dot_int_simd but the per-operand
/// signedness is chosen at RUNTIME from inst.neg (bit 0 -> src0 signed, bit 1
/// -> src1 signed) — hoisted out of the chunk loop. src2 is the int32
/// accumulator seed; clamp (when set) is the lower clamp to 0. The 8/4-bit
/// scalar bodies read no op_sel/neg_hi, so no gate.
template <int ElemBits, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_vop3p_dot_int_mixed_simd(Inst &inst, Wavefront &wf) {
  static_assert(ElemBits == 8 || ElemBits == 4, "iu dot ElemBits must be 8/4");
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.src1.simd_capable() ||
      !inst.src2.simd_capable() || !inst.vdst.simd_capable())
    return false;
  constexpr int N = 32 / ElemBits;
  constexpr uint32_t kElemMask = (ElemBits == 8) ? 0xFFu : 0xFu;
  constexpr int kShift = 32 - ElemBits;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  const bool clamp = inst.inst_.clamp;
  const bool src0_signed = (inst.inst_.neg & 0x1u) != 0;
  const bool src1_signed = (inst.inst_.neg & 0x2u) != 0;
  using U = util::native<uint32_t>;
  using I = util::native<int32_t>;
  const I kZeroI(0);
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U raw0 = read_simd<uint32_t>(inst.src0, wf, base);
    const U raw1 = read_simd<uint32_t>(inst.src1, wf, base);
    const U acc = read_simd<uint32_t>(inst.src2, wf, base);
    I sum = std::bit_cast<I>(acc);
    for (int i = 0; i < N; ++i) {
      const U ea = (raw0 >> (i * ElemBits)) & kElemMask;
      const U eb = (raw1 >> (i * ElemBits)) & kElemMask;
      // Sign-extend the low ElemBits when the operand is signed (same shift
      // trick as the signed dot path); plain zero-extended reinterpret when
      // unsigned. Sign choice is per operand, resolved at runtime.
      const I a = src0_signed ? ((util::stdx::static_simd_cast<I>(ea) << kShift) >> kShift)
                              : std::bit_cast<I>(ea);
      const I b = src1_signed ? ((util::stdx::static_simd_cast<I>(eb) << kShift) >> kShift)
                              : std::bit_cast<I>(eb);
      sum += a * b;
    }
    if (clamp)
      util::stdx::where(sum < kZeroI, sum) = kZeroI;
    write_simd<uint32_t>(inst.vdst, wf, base, std::bit_cast<U>(sum), chunk);
  }
  return true;
}

template <int ElemBits, typename Inst>
[[nodiscard]] bool try_execute_vop3p_dot_int_mixed_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP2/VOP3 dst-accumulate integer dot product (the "c" forms:
/// v_dot2c_i32_i16, v_dot4c_i32_i8, v_dot8c_i32_i4). The accumulator is the
/// DESTINATION register (inst.vdst), read as the accumulate source and written
/// as the result. VOP2 reads its 2nd source as inst.vsrc1, VOP3 as inst.src1
/// (selected by the Vop3 flag via if constexpr). All *c int forms are signed;
/// accumulation is int32-exact. The scalar bodies carry no op_sel/neg/clamp.
template <int ElemBits, bool Vop3, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_dotc_int_simd(Inst &inst, Wavefront &wf) {
  static_assert(ElemBits == 16 || ElemBits == 8 || ElemBits == 4, "dotc ElemBits must be 16/8/4");
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if constexpr (Vop3) {
    if (!inst.src1.simd_capable())
      return false;
  } else {
    if (!inst.vsrc1.simd_capable())
      return false;
  }
  constexpr int N = 32 / ElemBits;
  constexpr uint32_t kElemMask = (ElemBits == 16) ? 0xFFFFu : (ElemBits == 8) ? 0xFFu : 0xFu;
  constexpr int kShift = 32 - ElemBits;
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  using U = util::native<uint32_t>;
  using I = util::native<int32_t>;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U raw0 = read_simd<uint32_t>(inst.src0, wf, base);
    U raw1;
    if constexpr (Vop3)
      raw1 = read_simd<uint32_t>(inst.src1, wf, base);
    else
      raw1 = read_simd<uint32_t>(inst.vsrc1, wf, base);
    const U acc = read_simd<uint32_t>(inst.vdst, wf, base); // accumulator from dst
    I sum = std::bit_cast<I>(acc);
    for (int i = 0; i < N; ++i) {
      const U ea = (raw0 >> (i * ElemBits)) & kElemMask;
      const U eb = (raw1 >> (i * ElemBits)) & kElemMask;
      const I a = (util::stdx::static_simd_cast<I>(ea) << kShift) >> kShift;
      const I b = (util::stdx::static_simd_cast<I>(eb) << kShift) >> kShift;
      sum += a * b;
    }
    write_simd<uint32_t>(inst.vdst, wf, base, std::bit_cast<U>(sum), chunk);
  }
  return true;
}

template <int ElemBits, bool Vop3, typename Inst>
[[nodiscard]] bool try_execute_dotc_int_simd(Inst &, Wavefront &) {
  return false;
}

/// VOP2/VOP3 dst-accumulate f16 dot product (v_dot2c_f32_f16). Two f16 products
/// of src0 and src1/vsrc1 (widened via f16_to_f32_simd) plus the f32
/// accumulator read from the DESTINATION register. Bracketing matches the
/// scalar `facc += a0*b0 + a1*b1` exactly: acc + ((a0*b0)+(a1*b1)). No
/// op_sel/neg/clamp. NaN-payload divergence accepted (shared f16 carve-out).
template <bool Vop3, typename Inst>
  requires(util::has_stdx_simd)
[[nodiscard]] inline bool try_execute_dotc_f16_simd(Inst &inst, Wavefront &wf) {
  if (simd_force_scalar() || !inst.src0.simd_capable() || !inst.vdst.simd_capable())
    return false;
  if constexpr (Vop3) {
    if (!inst.src1.simd_capable())
      return false;
  } else {
    if (!inst.vsrc1.simd_capable())
      return false;
  }
  using T = uint32_t;
  constexpr std::size_t W = util::native_width_v<T>;
  const uint64_t chunk_full = util::mask<uint64_t>(static_cast<int>(W));
  const uint64_t exec = wf.exec();
  using F = util::native<float>;
  using U = util::native<uint32_t>;
  for (uint32_t base = 0; base < wf.wf_size(); base += static_cast<uint32_t>(W)) {
    const uint64_t chunk = (exec >> base) & chunk_full;
    if (chunk == 0)
      continue;
    const U raw0 = read_simd<uint32_t>(inst.src0, wf, base);
    U raw1;
    if constexpr (Vop3)
      raw1 = read_simd<uint32_t>(inst.src1, wf, base);
    else
      raw1 = read_simd<uint32_t>(inst.vsrc1, wf, base);
    const F acc = std::bit_cast<F>(read_simd<uint32_t>(inst.vdst, wf, base));
    const F a0 = util::f16_to_f32_simd(raw0 & 0xFFFFu);
    const F a1 = util::f16_to_f32_simd(raw0 >> 16);
    const F b0 = util::f16_to_f32_simd(raw1 & 0xFFFFu);
    const F b1 = util::f16_to_f32_simd(raw1 >> 16);
    const F r = acc + (a0 * b0 + a1 * b1);
    write_simd<uint32_t>(inst.vdst, wf, base, std::bit_cast<U>(r), chunk);
  }
  return true;
}

template <bool Vop3, typename Inst>
[[nodiscard]] bool try_execute_dotc_f16_simd(Inst &, Wavefront &) {
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

/// 64-bit-lane VOP2 binary counterpart (v_add/mul/max_num/min_num_f64). Lane
/// type fixed to double, read/written through the split lo/hi VGPR-pair path
/// (vsrc1 as the second source). Variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP2_BINARY_FP64(...)                                                    \
  if (::rocjitsu::amdgpu::try_execute_binary_vop2_f64_simd(inst, wf, __VA_ARGS__))                 \
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

/// VOP3 f64-source -> 32-bit-fp-dst cvt counterpart (src0 abs/neg + result
/// omod/clamp; for v_frexp_exp_i32_f64). Functor: native<double> -> narrow32<float>.
#define ROCJITSU_TRY_SIMD_CVT_VOP3_F64_TO_B32_FP(...)                                              \
  if (::rocjitsu::amdgpu::try_execute_cvt_vop3_f64_to_b32_fp_simd(inst, wf, __VA_ARGS__))          \
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

/// VOP3 f16 binary counterpart: same packed path as the integer form, but bails
/// to the (modifier-applying) scalar body when any abs/neg/omod/clamp field is
/// set. `T` is the 32-bit packed lane type; variadic in the functor.
#define ROCJITSU_TRY_SIMD_VOP3_BINARY_F16(T, ...)                                                  \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_f16_simd<T>(inst, wf, __VA_ARGS__))              \
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

/// VOP3 64-bit reverse-shift counterpart (v_lshlrev_b64 / v_lshrrev_b64 /
/// v_ashrrev_i64). src0 = 32-bit shift, src1 = 64-bit value; the functor takes
/// `(native<uint64_t> value, native<uint64_t> shift)`. Variadic.
#define ROCJITSU_TRY_SIMD_SHIFT64_VOP3(...)                                                        \
  if (::rocjitsu::amdgpu::try_execute_shift64_vop3_simd(inst, wf, __VA_ARGS__))                    \
  return

/// VOP3 v_lshl_add_u64 counterpart. Fixed op ((src0 << (src1 & 63)) + src2).
#define ROCJITSU_TRY_SIMD_LSHL_ADD_U64()                                                           \
  if (::rocjitsu::amdgpu::try_execute_lshl_add_u64_simd(inst, wf))                                 \
  return

/// VOP3 wide 32x32->64 multiply-add counterpart (v_mad_u64_u32 / v_mad_i64_i32).
/// The functor takes `(narrow32<uint32_t> s0, narrow32<uint32_t> s1,
/// native<uint64_t> c)`; variadic so its commas pass through.
#define ROCJITSU_TRY_SIMD_MAD_WIDE64_VOP3(...)                                                     \
  if (::rocjitsu::amdgpu::try_execute_mad_wide64_vop3_simd(inst, wf, __VA_ARGS__))                 \
  return

/// VOP3 carry-OUT counterpart (no carry-in; carry-out to SGPR sdst). Lane type
/// fixed to uint32_t; variadic in the SimdCarry functor.
#define ROCJITSU_TRY_SIMD_VOP3_CO(...)                                                             \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_co_simd(inst, wf, __VA_ARGS__))                  \
  return

/// VOP3 carry-IN counterpart (carry-in from SGPR src2, carry-out to SGPR sdst).
/// Lane type fixed to uint32_t; variadic in the SimdCarry functor.
#define ROCJITSU_TRY_SIMD_VOP3_CIN(...)                                                            \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_cin_simd(inst, wf, __VA_ARGS__))                 \
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

/// VOP3P packed-16 integer ternary probe (3-source pk_mad family).
#define ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_INT(...)                                                \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_ternary_int_simd(inst, wf, __VA_ARGS__))            \
  return

/// VOP3P packed-16 f16 binary probe. Functor takes (a, b) as f32 simd
/// vectors (already widened from f16 halves with neg applied) and returns
/// an f32 simd vector that is narrowed back to f16 inside the glue.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_FP16(...)                                                \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_binary_fp16_simd(inst, wf, __VA_ARGS__))            \
  return

/// VOP3P packed-16 f16 ternary probe (3-source pk_fma_f16).
#define ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_FP16(...)                                               \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_ternary_fp16_simd(inst, wf, __VA_ARGS__))           \
  return

/// VOP3P packed-f32 binary probe (v_pk_add_f32 / v_pk_mul_f32). Functor takes
/// (a, b) as narrow32<float> (neg-applied) and returns narrow32<float>.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_BINARY_F32(...)                                                 \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_binary_f32_simd(inst, wf, __VA_ARGS__))             \
  return

/// VOP3P packed-f32 ternary probe (v_pk_fma_f32). Functor takes (a, b, c) as
/// narrow32<float>; default-packing gate adds op_sel_hi_2 == 1.
#define ROCJITSU_TRY_SIMD_VOP3P_PK_TERNARY_F32(...)                                                \
  if (::rocjitsu::amdgpu::try_execute_vop3p_pk_ternary_f32_simd(inst, wf, __VA_ARGS__))            \
  return

/// VOP3P v_pk_mov_b32 probe. Functorless / fixed-op.
#define ROCJITSU_TRY_SIMD_VOP3P_MOV_B32()                                                          \
  if (::rocjitsu::amdgpu::try_execute_vop3p_mov_b32_simd(inst, wf))                                \
  return

/// VOP3P integer dot-product probe. Args: (ElemBits, Signed) — e.g.
/// (8, true) for v_dot4_i32_i8, (4, false) for v_dot8_u32_u4. Functorless.
#define ROCJITSU_TRY_SIMD_VOP3P_DOT_INT(...)                                                       \
  if (::rocjitsu::amdgpu::try_execute_vop3p_dot_int_simd<__VA_ARGS__>(inst, wf))                   \
  return

/// VOP3P v_dot2_f32_f16 probe. Functorless / fixed-op.
#define ROCJITSU_TRY_SIMD_VOP3P_DOT_F16()                                                          \
  if (::rocjitsu::amdgpu::try_execute_vop3p_dot_f16_simd(inst, wf))                                \
  return

/// VOP3P mixed-sign integer dot probe (v_dot4_i32_iu8 / v_dot8_i32_iu4). Arg:
/// ElemBits (8 or 4). Per-operand sign read at runtime from inst.neg.
#define ROCJITSU_TRY_SIMD_VOP3P_DOT_INT_MIXED(ElemBits)                                            \
  if (::rocjitsu::amdgpu::try_execute_vop3p_dot_int_mixed_simd<ElemBits>(inst, wf))                \
  return

/// VOP2/VOP3 dst-accumulate integer dot probe (the "c" forms). Args:
/// (ElemBits, Vop3) — e.g. (8, true) for v_dot4c_i32_i8_vop3.
#define ROCJITSU_TRY_SIMD_DOTC_INT(...)                                                            \
  if (::rocjitsu::amdgpu::try_execute_dotc_int_simd<__VA_ARGS__>(inst, wf))                        \
  return

/// VOP2/VOP3 dst-accumulate f16 dot probe (v_dot2c_f32_f16). Arg: Vop3 bool.
#define ROCJITSU_TRY_SIMD_DOTC_F16(...)                                                            \
  if (::rocjitsu::amdgpu::try_execute_dotc_f16_simd<__VA_ARGS__>(inst, wf))                        \
  return

#endif // ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
