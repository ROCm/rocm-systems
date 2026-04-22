// ARCH: gfx1250
//
// Fixture for the Class-5 phantom-lane sub-case
// (lit_tests/c5_predicate_chain_phantom_lane). Sibling of
// `c5_predicate_chain_tid` and `c5_predicate_chain_masked`, pinning the
// second refusal arm the classifier now carries: under WaveNative,
// when the kernel's `max_flat_workgroup_size` is BELOW the target
// wavefront width, phantom lanes are guaranteed for every launch and
// the WaveNative suppression (which relies on per-source-lane EXEC
// modeling) collapses.
//
// Structurally identical to `c5_predicate_chain_tid.hip` — same
// `workitem.id.x() → v_cmp_lt_u32 against small K → v_cndmask → store`
// shape that pins the narrow-O1 classifier's refusal rule — except
// for the `__launch_bounds__(32)` attribute. The 32-thread upper bound
// forces the AOT-compiled gfx1250 HSACO's `amd_kernel_code_t
// max_flat_workgroup_size` to 32, which is BELOW the gfx942 target
// wavefront width of 64 and triggers the phantom-lane refusal arm
// (c5_predicate_chain_classifier.cpp's `phantomLaneGuaranteed`
// predicate).
//
// Paired with `c5_predicate_chain_tid` which has the same icmp shape
// but NO launch_bounds → default max_flat_workgroup_size=1024
// (≥ target wave64), keeping the WaveNative suppression active on
// that fixture. The pair pins both WaveNative arms: phantom-lane
// refuse vs. phantom-lane-absent suppress.
//
// See `c5_predicate_chain_classifier.hpp`'s file-header docstring
// for the full rationale (phantom-lane is the soundness gap the
// `canary_bitmatrix_composite` recipe empirically falsified — this
// fixture is the minimal lit-level reproducer).

#include <hip/hip_runtime.h>

extern "C" __global__ __launch_bounds__(32) void
c5_predicate_chain_phantom_lane_kernel(int *out) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  int val = tid;
  int mask_cond = 0;
  asm volatile(
      "v_cmp_lt_u32_e64 %[m], %[t], 16\n\t"
      : [m] "=s"(mask_cond)
      : [t] "v"(tid)
      : "scc");
  int chosen;
  asm volatile(
      "v_cndmask_b32_e64 %[o], -1, %[v], %[m]\n\t"
      : [o] "=v"(chosen)
      : [v] "v"(val), [m] "s"(mask_cond));
  out[gid] = chosen;
}
