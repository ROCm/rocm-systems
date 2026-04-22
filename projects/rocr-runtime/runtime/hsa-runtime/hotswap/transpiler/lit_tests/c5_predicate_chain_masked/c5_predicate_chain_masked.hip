// ARCH: gfx1250
//
// Non-refusal sibling for the Class-5 "wave-size-sensitive predicate
// chain" narrow-O1 classifier (lit_tests/c5_predicate_chain_masked).
// Paired with lit_tests/c5_predicate_chain_tid, which pins the
// refusal path for the exact same compile-time-K predicate shape.
//
// The narrowing rule in
// `c5_predicate_chain_classifier.cpp` treats `and %v, K'` with
// `K' <= W_s - 1` as a "source-wave mask" that collapses replica-1's
// `tid` values back onto `[0, W_s)` before the predicate evaluates.
// That is the exact collapse the `tid AND (W_s-1)` rewrite would
// emit if §5 O2 landed; catching the masked shape here means the
// classifier is structurally ready to treat any future hand-emitted
// or rewrite-pass-emitted mask as safe.
//
// Soundness direction: masking BEFORE the icmp makes the predicate
// wave-size-oblivious (source wave 0's lane L and target replica-1's
// lane L+W_s both AND their tid by 31 before comparing, so they both
// see the same `tid & 31` in `[0, W_s)`). This is the §5.6.2
// `wave_id` lift's pattern and the SPE prelude's own `lane_id mod
// execBits` mask — both provably safe.
//
// The kernel uses the SAME compile-time K=15 as the refusal fixture;
// only the addition of `tid & 31` before the compare distinguishes
// them. If the non-refusal contract breaks (classifier regressed
// into refusing this shape), the classifier can no longer recognise
// the mask and will start refusing
// `lit_tests/c1_ttmp_wave_id_lift` and the SPE prelude itself.

#include <hip/hip_runtime.h>

extern "C" __global__ void c5_predicate_chain_masked_kernel(int *out) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  // Mask first, compare second. `tid & 31` (= tid AND (W_s - 1))
  // collapses any would-be-divergent replica-1 lane value onto
  // `[0, W_s)` before the predicate sees it. The classifier's
  // forward walk stops at the mask — downstream users of the masked
  // value are not tid-position-scoped anymore, and the compare
  // against compile-time K=15 evaluates consistently across
  // replicas.
  //
  // Forced via inline asm for the same reason the refusal fixture
  // does (hipcc would otherwise fold this pattern).
  int tid_masked = 0;
  asm volatile(
      "v_and_b32_e64 %[o], %[t], 31\n\t"
      : [o] "=v"(tid_masked)
      : [t] "v"(tid));
  int mask_cond = 0;
  asm volatile(
      "v_cmp_lt_u32_e64 %[m], %[t], 16\n\t"
      : [m] "=s"(mask_cond)
      : [t] "v"(tid_masked)
      : "scc");
  int val = tid;
  int chosen;
  asm volatile(
      "v_cndmask_b32_e64 %[o], -1, %[v], %[m]\n\t"
      : [o] "=v"(chosen)
      : [v] "v"(val), [m] "s"(mask_cond));
  out[gid] = chosen;
}
