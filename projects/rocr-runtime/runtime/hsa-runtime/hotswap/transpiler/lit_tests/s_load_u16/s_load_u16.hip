// ARCH: gfx1250
//
// Fixture for the gfx12+ scalar narrow-load family
// (s_load_{u8,i8,u16,i16}). The kernel exercises an unsigned
// half-word scalar load through a runtime-materialised SGPR-pair
// base pointer — matching the shape every corpus site uses today
// (the failing kerneldex kernel is Triton's `_attn_fwd`, which
// loads a tensor-descriptor metadata field this way):
//
//     s_load_b64  s[N:N+1], s[0:1], <kernarg_offset>     ; ptr from kernarg
//     s_load_u16  s*,       s[N:N+1], 0x0                ; narrow scalar load
//
// === Producer pattern (inline asm) ===
//
// Inline asm is used rather than a plain `*scalar_ptr` read because
// the HIP/AMDGPU frontend aggressively routes even uniform-looking
// narrow loads through a VMEM `flat_load_u16` when the source
// pointer is reached through a pointer-to-pointer indirection (the
// aliasing potential kills the uniform-address proof that the
// scalarisation pass requires to pick `s_load_u16`). Only inline
// asm reliably pins the SCALAR encoding we need to exercise the
// raiser's narrow-SMEM handler.
//
// The "=s" output constraint forces the dest into an SGPR and the
// "s" input constraint forces `scalar_ptr` into an SGPR-pair whose
// index the register allocator picks — NOT s[0:1] (which holds the
// kernarg base pointer across the asm). This produces the
// non-kernarg-base shape the handler expects: `base.baseIdx > 1`,
// so the defensive kernarg-narrow-load refusal in handle_smem.cpp
// is NOT tripped.
//
// The store `out[threadIdx.x] = v` prevents DCE; using
// `unsigned short*` rather than `short*` pins the u16 (zero-extend)
// variant rather than the i16 (sign-extend) variant.
//
// === Sibling width/signedness coverage ===
//
// The u16 positive path here covers the other three narrow widths
// (u8, i8, i16) transitively — the only deltas at IR level are the
// element type (i8 vs i16), the Align immediate (1 vs 2), and the
// zext/sext final extension, all of which are decided inside the
// shared SemOp switch in handle_smem.cpp. No corpus kernel exercises
// the u8/i8/i16 variants today, so adding per-variant fixtures
// would violate the no-fallback design rule (no fixture without a
// producer).
//
// MODREP: irrelevant — a scalar SMEM load broadcasts identically
// across every wave-size, and the handler makes no wave-width-
// dependent decisions (the dest SGPR index comes from the decoded
// operand, the base pointer comes from readSGPR64).

#include <hip/hip_runtime.h>

extern "C" __global__ void s_load_u16_kernel(
    int *__restrict__ out,
    const unsigned short *__restrict__ scalar_ptr) {
  int v;
  // s_load_u16 emits the gfx12+ scalar halfword load against an SGPR
  // pair. The `s_wait_kmcnt` drains the SMEM fetch so the SGPR is
  // live before the v_mov the compiler inserts for the store path.
  asm volatile(
      "s_load_u16 %[dst], %[base], 0x0\n\t"
      "s_wait_kmcnt 0\n\t"
      : [dst] "=s"(v)
      : [base] "s"(scalar_ptr));
  out[threadIdx.x] = v;
}
