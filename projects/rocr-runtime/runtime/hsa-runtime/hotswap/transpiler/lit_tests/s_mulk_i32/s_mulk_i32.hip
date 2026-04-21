// ARCH: gfx1250
//
// Lit fixture for `s_mulk_i32` (SOPK_32TIE). Companion to
// `lit_tests/s_addk_i32/`; same tied-`$src0` operand-layout
// contract (MCInst = [sdst (def), $src0 (tied use), $simm16],
// `op.src(1)` is the immediate, `op.src(0)` aliases the tied
// SGPR). Handler in `handle_sopk.cpp` was fixed alongside
// `S_ADDK_I32` in the same commit to read `op.src(1)` as the
// immediate; no corpus kernel currently exercises `s_mulk_i32`,
// so this fixture is the dedicated regression guard for the mul
// arm (the `s_addk_i32` fixture guards the add arm).
//
// Without this fixture a future refactor that breaks the mul
// arm specifically (while leaving the add arm intact) passes
// every test in the repo. With it, `%mulk = mul i32 %..., 0x123`
// is the pinned contract and the absence of a second SSA
// operand (`mul i32 %x, %x`) regression-guards the
// doubling/squaring shape that the pre-fix bug would resurrect.
//
// Pinned immediate is 0x123 (not 0x400) deliberately — different
// from the sibling `s_addk_i32` fixture so a careless
// copy-paste of CHECK lines would fail noisily rather than
// match the wrong literal.

#include <hip/hip_runtime.h>

extern "C" __global__ void s_mulk_i32_kernel(int *out) {
  int tid = (int)threadIdx.x;
  int x = (int)blockIdx.x;
  asm volatile(
      "s_mulk_i32 %[x], 0x123\n\t"
      : [x] "+s"(x));
  out[tid] = x + tid;
}
