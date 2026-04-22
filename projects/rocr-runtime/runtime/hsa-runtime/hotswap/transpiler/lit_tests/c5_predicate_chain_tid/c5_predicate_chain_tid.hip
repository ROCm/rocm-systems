// ARCH: gfx1250
//
// Fixture for the Class-5 "wave-size-sensitive predicate chain"
// obstruction (lit_tests/c5_predicate_chain_tid). Refusal sibling.
//
// Class 5 (hotswap/docs/modrep-predicate-chain.md §5 narrow-O1
// classifier): `llvm.amdgcn.workitem.id.x()` flows into an `icmp`
// whose other operand is a compile-time constant `K` in
// `(0, W_s - 1]`, and the chain from the intrinsic to the icmp is
// NOT AND-masked by `(W_s - 1)` first. This is a lane-position-
// scoped predicate (scan-stage guards like `tid >= 2^s`, half-wave
// broadcasts, quad-level masks). Under modulo-replication source
// wave 0's lane L and target replica-1's lane L+W_s share the same
// source EXEC bit but have different architectural `tid` values,
// so the predicate evaluates differently between replicas and the
// store-gating it controls commits to different slots. The
// classifier has no rewrite today — refuses loud.
//
// This fixture pins the REFUSAL path. The sibling
// `lit_tests/c5_predicate_chain_masked/` pins the non-refusal
// (masked) path; together they pin both sides of the classifier's
// soundness-not-completeness contract (see
// hotswap/docs/wave-size-translation.md §7).
//
// The canonical corpus surface is `compare_correctness`'
// `canary_bpermute_scan_fp32` Triton recipe (Kogge-Stone scan
// with per-stage guards `tid >= 2^s`, s ∈ {1, 2, 3, 4}; constants
// 1, 3, 7, 15 all `<= W_s - 1 = 31`). This fixture uses a single
// compile-time-K icmp-to-store chain to isolate the classifier
// against the full-Triton scan shape.
//
// Implementation note. HIP's codegen does not spontaneously emit
// a `tid < K` compare on a compile-time small K (any kernel
// expression like `if (tid < 16) ...` is typically fused into
// other control flow). We force the shape via inline asm: read
// `workitem.id.x()` into a VGPR, emit `v_cmp_lt_u32_e64` with an
// immediate RHS, and use the resulting SGPR mask to gate a store
// through `v_cndmask_b32` → `global_store_b32`. The compiled
// binary therefore contains the exact signature the narrow-O1
// classifier refuses on: `workitem.id.x()` → `icmp ult K=15, tid`
// → gate store.

#include <hip/hip_runtime.h>

extern "C" __global__ void c5_predicate_chain_tid_kernel(int *out) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  // Compile-time-constant predicate `tid < 16`: K = 15 is within
  // (0, W_s - 1 = 31] on the gfx1250 source, so the narrow-O1
  // classifier refuses this shape. Writing `out[gid] = tid` on
  // the "below-16" branch produces the standard lane-position-
  // scoped-store signature the classifier detects.
  //
  // The compare is forced via inline asm so hipcc does not elide
  // it (e.g. by moving the tid read into a predicate-free pure
  // arithmetic path). The `"=s"` output binds the SGPR mask to
  // `mask`; the subsequent v_cndmask gates the store under the
  // same mask.
  int val = tid;
  int mask_cond = 0;
  asm volatile(
      "v_cmp_lt_u32_e64 %[m], %[t], 16\n\t"
      : [m] "=s"(mask_cond)
      : [t] "v"(tid)
      : "scc");
  // Use the SGPR mask to select either `val` or -1, then store to
  // out[gid]. The store is unconditional at the IR level, but the
  // value it writes is gated by the lane-position-scoped predicate
  // — the raiser's post-mem2reg IR has the
  // `workitem.id.x() → icmp ult K=15 → select → store` chain that
  // the narrow-O1 classifier matches on.
  int chosen;
  asm volatile(
      "v_cndmask_b32_e64 %[o], -1, %[v], %[m]\n\t"
      : [o] "=v"(chosen)
      : [v] "v"(val), [m] "s"(mask_cond));
  out[gid] = chosen;
}
