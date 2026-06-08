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
  if (p) {
    constexpr auto W = static_cast<uint32_t>(util::native_width_v<T>);
    SimdAccess::notify_read(op, wf, lane_base, lane_base + W, 0xF);
    return util::load<T>(p);
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
  if (uint32_t *p = SimdAccess::dst_ptr(op, wf, lane_base)) {
    util::masked_store<T>(p, v, mask);
    return;
  }
  alignas(util::native<T>) uint32_t buf[W];
  util::blit_to_buffer<T>(buf, v);
  op.write_lane_chunk(wf, lane_base, static_cast<uint32_t>(W), buf, mask);
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

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_ISA_AMDGPU_SHARED_SIMD_GLUE_H_
