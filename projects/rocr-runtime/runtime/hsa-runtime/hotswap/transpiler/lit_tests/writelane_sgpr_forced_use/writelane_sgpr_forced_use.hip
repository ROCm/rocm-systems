// ARCH: gfx1250
//
// Fixture for the use-chain classifier's SGPR-forced refusal path
// (lit_tests/writelane_sgpr_forced_use). Pins the half of
// `rewrite_cross_lane_divergent.cpp`'s forward use-chain walk that
// REFUSES the rewrite when a writelane / readlane result reaches an
// SGPR-constrained consumer — specifically, a `v_readfirstlane_b32`
// scalarisation downstream.
//
// Without this fixture the classifier has a silent failure mode: a
// future kernel that pairs a divergent writelane with an SGPR-forced
// consumer (scalar buffer load rsrc, `s_sendmsg` payload, explicit
// `v_readfirstlane_b32`, inline asm `"s"`, addrspace(4) load, …)
// would make the rewrite emit a `ds_bpermute` whose output the
// backend then re-collapses via `v_readfirstlane` at the SGPR
// boundary — recreating the original cross-wave-lane-id-leak miscompile.
// The classifier exists to catch exactly that shape.
//
// Shape.
//   (1) `s_bfe_u32 ttmp8, 0x50019` — canonical wave_id lift; produces
//       a divergent i32 via `handle_sop2.cpp`'s rescue.
//   (2) `v_writelane_b32` — consumes the divergent value and returns
//       an i32 carrying per-source-wave state.
//   (3) `v_readfirstlane_b32` on the writelane's OUTPUT — explicit
//       scalarisation marker. The classifier enumerates
//       `Intrinsic::amdgcn_readfirstlane` in `operandForcesSGPR`,
//       so the forward walk from the writelane's result hits it and
//       returns SGPRForced.
//
// Expected.
//   * Under `--enable-writelane-rewrite`, `raise_cli` FAILS with
//     `cross-wave-lane-id-leak` (the
//     `RaiseFailure::crossWaveRewriteOracleDisagreement` bucket)
//     and a detail naming the `readfirstlane` consumer.
//   * Under the flag-off path, the kernel raises normally (Phase
//     1.4.5 classifier matches `WaveIdLiftScalarized` only when WMMA
//     is also present; this fixture has no WMMA, so flag-off raises
//     cleanly — same contract as `writelane_divergent_rewrite`).

#include <hip/hip_runtime.h>

extern "C" __global__
void writelane_sgpr_forced_use_kernel(int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // (1) Divergent wave_id seed via canonical ttmp8 BFE rescue.
  int wave_id;
  asm volatile("s_bfe_u32 %[wid], ttmp8, 0x50019\n\t"
               : [wid] "=s"(wave_id));

  // (2) writelane — divergent scalar feed, VGPR result.
  int lane_slot;
  asm volatile("v_writelane_b32 %[out], %[val], 0\n\t"
               : [out] "=v"(lane_slot)
               : [val] "s"(wave_id));

  // (3) readfirstlane on the writelane's OUTPUT. Under cross-
  //     widening the rewrite would convert (2) to a per-lane
  //     `select`, but the `v_readfirstlane_b32` here forces the
  //     result back into an SGPR — re-introducing the collapse the
  //     rewrite was designed to avoid. The use-chain classifier in
  //     `rewrite_cross_lane_divergent.cpp::classifyForwardUseChain`
  //     must detect this and refuse.
  int sconsumer;
  asm volatile("v_readfirstlane_b32 %[dst], %[src]\n\t"
               : [dst] "=s"(sconsumer)
               : [src] "v"(lane_slot));

  // Keep both outputs live so front-end DCE does not strip the
  // writelane / readfirstlane chain before it reaches the compiled
  // code object.
  out[tid] = wave_id ^ sconsumer;
}
