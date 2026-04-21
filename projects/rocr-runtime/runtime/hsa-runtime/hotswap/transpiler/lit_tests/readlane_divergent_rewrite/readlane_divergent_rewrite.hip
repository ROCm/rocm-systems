// ARCH: gfx1250
//
// Fixture for the cross-widen-divergent `v_readlane_b32` rewrite
// (lit_tests/readlane_divergent_rewrite). Pins the IR shape
// `rewrite_cross_lane_divergent.cpp` emits when the rewrite pass's
// forward use-chain classifier clears a readlane site for symmetric
// rewriting under `gfx1250 → gfx942` cross-widening.
//
// Companion to the writelane_divergent_rewrite fixture — the two
// together pin both halves of the cross-lane primitive rewrite
// catalogue (the only two intrinsics the pass acts on today). No
// WMMA in the kernel, so the `WaveIdLiftScalarized` classifier does
// NOT refuse it and the rewrite pass has a chance to run.

#include <hip/hip_runtime.h>

extern "C" __global__ void readlane_divergent_rewrite_kernel(int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // `v_readlane_b32 sDST, vSRC, <imm lane>` reads lane <imm> of a
  // divergent VGPR source and broadcasts it as a scalar. The VGPR
  // source here is the per-thread `tid`, which is divergent by
  // construction (derived from `amdgcn.workitem.id.x` through the
  // workgroup-local kernel prologue). Under cross-widening the
  // backend would insert an implicit readfirstlane tied to the
  // scalar output, collapsing per-source-wave state into one lane's
  // value. The rewrite replaces the call with
  //   `ds_bpermute(((lane_id & ~(W_s-1)) | lane_idx) << 2, src_div)`
  // so every source wave broadcasts only within its own lanes.
  int broadcast;
  asm volatile("v_readlane_b32 %[dst], %[vsrc], 0\n\t"
               : [dst] "=s"(broadcast)
               : [vsrc] "v"(tid));

  out[tid] = broadcast;
}
