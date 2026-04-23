// Hardware probe for the Triton bitonic-merge cross-16 stage's
// compare semantics on gfx1250.
//
// Why this exists
// ===============
//
// Triton's `tl.sort` at (BLOCK_M=32, BLOCK_N=32) breaks under
// wave32→wave64 cross-widening, producing two SORTED HALVES
// instead of a fully merged sort (per
// `canary_tl_sort_fp32_deterministic` in
// `compare_correctness/kernels/triton/`).  The 4-instruction
// dance that implements the distance-16 compare-and-swap stage
// looks like this in the disassembled gfx1250 assembly:
//
//   v_dual_mov_b32 v3, v2 :: v_dual_mov_b32 v4, v2
//   v_permlane16_swap_b32 v3, v4
//   v_xor3_b32 v3, v3, v4, v2
//   v_and_b32 v4, 8, v0                    ; clobbers v4 for next stage
//   .long 0xd41b0002                       ; gfx1250-new single-dword
//                                          ; VOP3 compare; salmon raises
//                                          ; it as `fcmp ule v2, v3`
//   v_cndmask_b32_e32 v1, v2, v3, vcc_lo
//   s_xor_b32 vcc_lo, s2, vcc_lo
//   v_cndmask_b32_e32 v2, v2, v3, vcc_lo
//
// Per-lane value trace for the deterministic input v2 = L (lane
// id):
//
//   after mov:           v3 = L,             v4 = L
//   after swap:          v3 = L XOR 16,      v4 = L XOR 16
//                        (both hold partner's v2)
//   after xor3:          v3 = self XOR self = L,  v4 = L XOR 16
//                        (v3 "collapses back" to self; v4 untouched)
//   after v_and v4,8:    v4 = L & 8          (clobbered for next stage)
//
// At the compare site, salmon's decode says it reads v2 vs v3.
// BOTH are `L` (self), so the compare is self-vs-self = always
// `true` under any ordered predicate.  Then cndmask picks v3 (=
// self) regardless of the s_xor_b32 flip, so the "swap" is a
// no-op.  That explains the two-sorted-halves pattern exactly.
//
// BUT native gfx1250 produces a correct full sort.  So either:
//
//   (a) My read of `v_xor3_b32 v3, v3, v4, v2` → `v3 = self` is
//       wrong on gfx1250 (this instruction's semantic differs
//       from the gfx11 version I've been matching against).
//   (b) The `.long 0xd41b0002` compare's real operand set differs
//       from salmon's decode — it may be reading `v4`-before-
//       clobber, or using a gfx1250-new per-lane-id-biased form
//       that salmon's decoder doesn't model.
//
// This HIP kernel emits the exact sequence with known-distinct
// inputs for v3 and v4 so the hardware trace pins which.
//
// Probe design
// ============
//
// Per lane L, with `L` in [0, 32):
//
//   v2 = L                        (value identifier — "self")
//   v3 = 100 + L                  (DIFFERENT from v2 so the post-
//                                 xor3 value reveals whether the
//                                 xor3 canceled partner correctly)
//   v4 = 200 + L                  (DIFFERENT again)
//
// Run the 4-instruction sequence:
//
//   v_permlane16_swap_b32 v3, v4  ; swap
//   v_xor3_b32 v3, v3, v4, v2     ; xor3
//
// Then dump v2, v3, v4 to device memory for all 32 lanes.
//
// Expected under my model (cross-wired swap + xor3 semantics):
//
//   after swap:
//     new_v3[L] = src0_in[L XOR 16] = v4_in[L XOR 16] = 200 + (L XOR 16)
//     new_v4[L] = vdst_in[L XOR 16] = v3_in[L XOR 16] = 100 + (L XOR 16)
//   after xor3 (v3 = v3 ^ v4 ^ v2):
//     v3 = (200 + (L XOR 16)) ^ (100 + (L XOR 16)) ^ L
//        = arbitrary int (not reducing to anything clean)
//   v4 unchanged: v4 = 100 + (L XOR 16)
//
// The actual dump tells us:
//   - the POST-SWAP v3, v4 values (to pin the swap semantics
//     under DIFFERENT inputs)
//   - the POST-XOR3 v3 value (to pin whether xor3 reads pre- or
//     post-swap v3 — if it reads pre-swap, v3 ends up differently)
//
// Build:
//   hipcc --offload-arch=gfx1250 --genco \
//     test_data/gfx1250/bitonic_cross16_probe_kernel.hip \
//     -o test_data/gfx1250/bitonic_cross16_probe_gfx1250.hsaco
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler \
//     --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input=test_data/gfx1250/bitonic_cross16_probe_gfx1250.hsaco \
//     --output=test_data/gfx1250/bitonic_cross16_probe_gfx1250.hsaco \
//     --unbundle

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(32)
    void bitonic_cross16_probe_kernel(int *out_v2_final,
                                        int *out_v3_after_xor3,
                                        int *out_v4_after_swap) {
  int L = threadIdx.x;
  int v2 = L;                  // self identifier
  int v3 = 100 + L;            // distinct — reveals swap path
  int v4 = 200 + L;            // distinct — reveals swap path

  // The exact 4-instruction dance from Triton's tl.sort cross-16
  // stage, with inline asm.  We snapshot v4 BEFORE any clobber so
  // the dump reflects the post-swap (pre-xor3) state.
  asm volatile(
      "v_permlane16_swap_b32 %[v3], %[v4]\n\t"
      "v_xor3_b32 %[v3_out], %[v3], %[v4], %[v2]\n\t"
      : [v3] "+v"(v3), [v4] "+v"(v4), [v3_out] "=v"(v3)
      : [v2] "v"(v2));

  // Now v3 = post-xor3 value, v4 = post-swap value (pre-clobber
  // — we don't emit `v_and_b32 v4, 8, v0` here, so v4 still
  // holds the post-swap partner value).
  out_v2_final[L] = v2;
  out_v3_after_xor3[L] = v3;
  out_v4_after_swap[L] = v4;
}
