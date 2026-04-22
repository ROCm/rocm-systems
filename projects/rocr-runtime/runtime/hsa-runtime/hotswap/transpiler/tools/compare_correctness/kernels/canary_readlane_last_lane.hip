// Canary for the `v_readlane_b32 sDST, vSRC, K` C1 leak pattern where
// K = W_s - 1 (the "broadcast from the last lane of the source wave"
// idiom).  Targets the exact cross-widening hazard the GPT-OSS
// derisking doc flags in §7.1 / §9.2 (C1 operand-range validator),
// with a deliberately chosen K that the *proposed* validator would
// NOT refuse (K < W_s) but that is nevertheless wave-size-dependent
// under modulo-replication.  Authored as a `.hip` rather than a
// Triton `.py` because Triton's reduction lowering does not expose
// a knob to force a specific lane constant — inline asm is the
// only way to pin the exact MCInst shape we need to probe.
//
// Motivation (from `hotswap/docs/gpt-oss-derisking.md` §7.1):
//
//   `bitmatrix_metadata_compute_stage1` in the GPT-OSS MoE router
//   path emits `v_readlane_b32 ×1` as part of its routing
//   sub-structure.  The §9.2 blocking-work list proposes a
//   raise-time operand-range validator that "loud-fails if the
//   readlane / writelane operand is ≥ W_s, passes otherwise".
//   That validator handles the easy case (K outside the source
//   wave's lane range) but not the harder case (K = W_s - 1, the
//   "last lane of the source wave" idiom that's common enough to
//   appear in reductions that broadcast their final result from
//   the butterfly's last surviving lane).  Under cross-widening
//   to wave64 with modulo-replication, a wave32 source kernel's
//   `readlane(v, 31)` reads target lane 31 — which on wave64
//   holds source thread 31's value, NOT source thread 63's value
//   (the second replica's "last lane").  All 64 target lanes
//   therefore receive source thread 31's data, silently losing
//   source wave 1's broadcast.
//
// Kernel shape:
//
//   * 64-thread workgroup, one workgroup.  On gfx1250 wave32 that
//     is 2 source waves of 32 each.
//   * Each lane seeds a VGPR with its 0..63 thread ID (stored via
//     volatile load so hipcc can't constant-fold the readlane
//     result across the workgroup boundary).
//   * `v_readlane_b32 s, v, 31` via inline asm binds the lane
//     constant to the raw instruction encoding — LLVM's inline-asm
//     matcher accepts `31` as a literal VSrc_b32 lane index and
//     emits the v_readlane_b32_e32 form directly.
//   * The SGPR result is moved back to a per-lane VGPR and written
//     to `out[tid]`.  Every lane in a wave writes the same scalar;
//     the per-lane write is how we verify the source's broadcast
//     semantics survived.
//
// Expected outputs and verdict contract (CpuReference gold):
//
//   CPU gold semantics = wave32 source:
//     * out[tid] for tid in  0..31 == in[31]  (source wave 0's lane 31)
//     * out[tid] for tid in 32..63 == in[63]  (source wave 1's lane 31)
//
// Observed verdicts (as of the canary's landing commit):
//
//   native : WRONG 32/64 on every shape.  hipcc compiles the
//     inline-asm `v_readlane_b32` against gfx942 wave64 verbatim;
//     target lane 31 holds source thread 31's value, so all 64
//     lanes receive in[31] and lanes 32..63 diverge from the
//     wave32-source gold.  This is the inherent cost of
//     re-emitting a wave32-semantics kernel against a wave64
//     target without any per-source-wave broadcast emulation —
//     not a hipcc bug.
//
//   legacy : WRONG 32/64 on every shape.  The legacy text-
//     transpiler copies the mnemonic across verbatim and so
//     inherits the same wave64-target-lane semantics as native.
//
//   salmon : match on every shape under the runtime's
//     WaveNativeProjection default.  (Re-testing the MODREP arm
//     requires the `--disable-wave-native` CLI flag to raise_cli —
//     there is no env-var override for the intercept library; see
//     `raiser.hpp`'s "No process-global env-var override" comment
//     for why.  `raise_cli` on the same HSACO + `--disable-wave-
//     native` also succeeds, so both projections exercise the
//     rewrite clean.)
//     The `rewrite_cross_lane_divergent.cpp` pass recognises the
//     lifted `llvm.amdgcn.readlane(val, 31)` as a cross-wave-
//     divergent site under cross-widening and rewrites it into
//     an on-target `ds_bpermute` that reads lane
//     `(lane_id & ~(W_s - 1)) | 31` — the "last lane of THIS
//     source-wave replica".  The resulting per-replica broadcast
//     matches the wave32-source gold bit-exactly.
//
// Regression contract:
//
//   This canary pins down two orthogonal signals at once:
//
//   1. salmon := match.  A drift from `match` to `WRONG 32/64`
//      would indicate the cross-wave-divergent readlane rewrite
//      in `rewrite_cross_lane_divergent.cpp` regressed — either
//      the classifier stopped flagging the site, or the rewrite
//      produced a wrong selector expression, or a downstream
//      projection (`wave_projection.cpp`) stopped threading
//      `cwd_lane_id` correctly.  The mismatch count is a stable
//      half-output-size signal, so the regression is loud.
//
//   2. native / legacy := WRONG 32/64.  If those modes ever
//      graduate to `match`, it means either hipcc started doing
//      source-wave-aware readlane emulation (unlikely; hipcc has
//      no notion of source wave) or the CpuReference drifted.  A
//      graduation-to-match on native/legacy is NOT a win — it
//      would mean the gold semantics diverged from the intended
//      wave32-source behaviour.  Investigate before relaxing.
//
//   3. salmon := EXIT=2.  Would mean the classifier moved from
//      "rewrite" to "refuse" for this site.  If that was
//      intentional (e.g. a tighter §7.1 C1 validator was
//      authored to refuse all wave-dependent lane constants),
//      retire this canary's `match` expectation and document the
//      refusal verdict; otherwise the refusal is a regression.
//
// See `hotswap/docs/gpt-oss-derisking.md` §7.1 for the original
// §9.2 item 5 "C1 operand-range validator" framing and the one
// GPT-OSS readlane site that motivated it.  Also consider pairing
// this canary with a companion `canary_writelane_last_lane` if a
// future corpus kernel surfaces the write-side variant (today the
// readlane side alone is enough to exercise the shared rewrite
// pass).

#include <hip/hip_runtime.h>

extern "C" __global__ void canary_readlane_last_lane(
    const float *in, float *out) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Volatile load keeps the per-lane seed live across any whole-
  // program constant-folding that hipcc might apply.  Without it
  // the compiler has been observed to fold across the inline asm
  // boundary when the seed is a statically-known index.
  float v = *(const volatile float *)(in + tid);
  float result;
  asm volatile(
      "v_readlane_b32 %[r], %[v], 31"
      : [r] "=s"(result)
      : [v] "v"(v));
  out[tid] = result;
}
