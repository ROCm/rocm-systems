// ARCH: gfx1250
//
// Lift fixture for `s_addk_i32` (renamed to `s_addk_co_i32` in
// gfx12+ assembly). The MC opcode is SOPK class `SOPK_32TIE` with
// a tied-to-def `$src0` — the MCInst carries three operands
// `[sdst, $src0 (tied), $simm16]`, and the decoder's
// `buildSrcMap` keeps the tied slot in srcMap (it's listed in
// `kKnownTiedIn` in `decode.cpp`). So for the handler in
// `handle_sopk.cpp`:
//
//   * `op.dst()`  -> sdst (def) AND the prior-dst value via
//                    `readReg32(op.dst())`.
//   * `op.src(0)` -> the tied `$src0` (same physical SGPR as
//                    sdst; reading it yields the prior-dst value
//                    again — redundant with `readReg32(op.dst())`).
//   * `op.src(1)` -> the `$simm16` immediate (the SGPR-relative
//                    16-bit constant we actually want to add).
//
// Regression guard. An earlier version of the handler read
// `op.src(0)` expecting the immediate — so `s_addk_co_i32 s0,
// 0x400` lifted to `add i32 %s0, %s0` (doubling) instead of
// `add i32 %s0, 1024`. For Triton-compiled `layer_norm` the
// compiler emits one `s_addk_co_i32 sN, 0x400` per `for off in
// range(0, N, BLOCK_SIZE)` loop as the counter increment; the
// doubled-0 counter never exits the loop and the kernel hangs
// forever. (Three such sites per kernel — one each for the mean
// reduction, variance reduction, and normalize-and-store pass —
// all broken identically.)
//
// The `S_MULK_I32` handler had the same latent bug (same TableGen
// class, same tied-def shape). No currently-corpus-tested kernel
// exercises `s_mulk_i32`, but the fix pattern is identical; this
// fixture covers `s_addk_i32` directly and the `mulk` arm is
// covered transitively by handler-shape identity.
//
// Kernel. Inline asm forces the exact `s_addk_co_i32` shape; the
// in-out SGPR constraint `+s` binds `%x` to a single 32-bit
// SGPR. We initialise from `blockIdx.x` (kernarg-derived, lives
// naturally in an SGPR) so hipcc materialises the source in the
// right bank, then the asm adds 0x400 (1024) and the result
// stores to a per-lane VGPR output.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_addk_i32_kernel(int *out) {
  int tid = (int)threadIdx.x;
  int x = (int)blockIdx.x;
  asm volatile(
      // Wave32 source: `s_addk_co_i32 sdst, simm16` where sdst is
      // both a source (tied $src0) and a destination. `+s` keeps
      // `%x` in the same SGPR for input and output — matching the
      // tied-in contract of the MC opcode.
      "s_addk_co_i32 %[x], 0x400\n\t"
      : [x] "+s"(x));
  out[tid] = x + tid;
}
