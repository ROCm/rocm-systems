// Canary for the `ds_swizzle_b32` P6 lift on the QUAD_PERM envelope
// — complements the existing `canary_ds_swizzle_swap1` (BITMASK_PERM
// / SWAP-1, offset=0x041F) by exercising a different top-level
// swizzle envelope that carries an independent imm-encoding contract
// in `handle_ds.cpp::DS_SWIZZLE_B32`.
//
// Coverage complementarity
// ========================
//
// The P6 handler enumerates four valid top-level envelopes, each with
// its own imm-encoding and wave-size-obliviousness argument:
//
//   BITMASK_PERM (top bit 0x0000 mask)   — and=AND_M, or=OR_M, xor=XOR_M
//                                         per-lane permutation.  Covered
//                                         by `canary_ds_swizzle_swap1`
//                                         (SWAP-1 is a BITMASK_PERM
//                                         sub-case with XOR=0x1F).
//   QUAD_PERM    (top byte  0x8000 mask) — four-element swizzle
//                                         inside each lane-quad.
//                                         **This canary.**
//   FFT_MODE     (top nibble 0xE000 mask) — 5-bit FFT shuffle.  No
//                                         canary; not hit by GPT-OSS
//                                         (handler comment documents
//                                         observed encoding
//                                         stability across targets).
//   ROTATE_MODE  (top nibble 0xC000 mask) — rotate-within-32-lanes.
//                                         Same status as FFT.
//
// QUAD_PERM is the most-commonly-hit second envelope in real
// corpora beyond BITMASK_PERM — Triton's vectorised reductions over
// width-4 sub-groups and GPT-OSS's bitmatrix routing pre-aggregation
// both emit it.  Authoring this canary:
//
//   (a) Validates `handle_ds.cpp`'s `dsSwizzleImm` decode path on
//       the top-byte-0x80 envelope independently of the top-bit-0
//       BITMASK_PERM path `canary_ds_swizzle_swap1` already
//       exercises.  A regression that mis-decodes the top-byte
//       (e.g. a sign-extension bug on the imm, or a shift that
//       drops the QUAD_PERM bit while preserving BITMASK_PERM's)
//       would fail here but pass the SWAP-1 canary.
//
//   (b) Validates the classifier (`dsSwizzleSafeForModRep`)
//       accepts the QUAD_PERM envelope under MODREP for the same
//       per-32-lane-half-independent reason the handler docstring
//       cites.  A regression that narrows the classifier to only
//       BITMASK_PERM would fail here.
//
//   (c) Complements the existing `Gfx1250Gpu.DsSwizzle` GPU test
//       (which uses SWAP-2 / offset=0x081F, a third BITMASK_PERM
//       sub-case) by adding the first QUAD_PERM end-to-end
//       regression gate in either the lit or compare_correctness
//       suites.
//
// Imm encoding
// ============
//
// QUAD_PERM imm: `0x8000 | (SEL3 << 6) | (SEL2 << 4) | (SEL1 << 2)
// | SEL0` — where `SELk` is the source-lane selector for quad-
// position `k`.  For lane L inside a quad, the swizzle reads from
// lane `(L & ~3) | Q[L & 3]`, where `Q[k] = SELk`.  The AMDGPU
// assembler's `swizzle(QUAD_PERM, A, B, C, D)` spelling encodes
// `A = SEL0, B = SEL1, C = SEL2, D = SEL3` (the assembler's first
// printed argument corresponds to quad-position 0 at the LOW
// two bits of the imm, NOT to SEL3 at the high two bits — a
// gotcha the initial version of this canary confused).
//
// This canary uses `swizzle(QUAD_PERM, 3, 2, 1, 0)` — the in-quad
// reversal: lane 0 reads lane 3, lane 1 reads lane 2, lane 2 reads
// lane 1, lane 3 reads lane 0, repeating every 4 lanes.  Encoded
// imm = 0x8000 | (0 << 6) | (1 << 4) | (2 << 2) | 3 = 0x801B.
//
// Gold semantics (CPU reference) = per-quad-reversal within each
// 4-lane group.  bit-5-preservation makes this identical on
// wave32 source and wave64 target, so all three modes (native /
// legacy / salmon) MUST match.
//
// Regression contract
// ===================
//
// * native / legacy / salmon `match` -> any drift to `WRONG`  ==
//   either hipcc's inline-asm bound the imm to a different
//   encoding (check disasm), the P6 imm-decoder regressed, the
//   classifier moved the QUAD_PERM envelope to refuse, or a
//   downstream projection broke the intrinsic-to-backend path.
// * any mode -> `EXIT=2`  ==  classifier refused this envelope.
//   Only intentional if a future ISA changes QUAD_PERM's
//   hardware semantics; otherwise a regression.
//
// See `handle_ds.cpp::DS_SWIZZLE_B32` block comment for the
// per-envelope argument and `hotswap/docs/gpt-oss-derisking.md`
// §7.2 for the corpus motivation.

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(64) void
canary_ds_swizzle_quad_perm(const float *in, float *out) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  float v = *(const volatile float *)(in + tid);
  // Same ISA-dispatched wait as `canary_ds_swizzle_swap1`: hipcc's
  // automatic scheduling across `asm volatile` has been observed to
  // omit the DS-counter wait on gfx942 (the store reads the pre-
  // swizzle VGPR and the permute silently disappears from the
  // output).  Bake it into the asm with gfx-appropriate mnemonic.
  // offset:0x801B = swizzle(QUAD_PERM, 3, 2, 1, 0) — in-quad
  // reversal (lane 0 reads lane 3, lane 1 reads lane 2, ...).
#if defined(__gfx12__) || defined(__gfx1250__) || defined(__gfx1200__) || \
    defined(__gfx1201__)
  asm volatile(
      "ds_swizzle_b32 %[v], %[v] offset:0x801b\n\t"
      "s_wait_dscnt 0"
      : [v] "+v"(v));
#else
  asm volatile(
      "ds_swizzle_b32 %[v], %[v] offset:0x801b\n\t"
      "s_waitcnt lgkmcnt(0)"
      : [v] "+v"(v));
#endif
  out[tid] = v;
}
