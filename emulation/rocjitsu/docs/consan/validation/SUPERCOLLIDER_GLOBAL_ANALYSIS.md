# SuperCollider global-memory support: feasibility and qualification

September 24, 2026. This is an implementation analysis, not a claim that ConSan
already detects global-memory races. Current Default and SuperCollider validation
colors and physical benchmark measurements are unchanged. The two motivating
workloads are `pytorch-scatter-reduce` and TokenSpeed Gluon BF16 MoE on gfx950.

## Conclusion

The paper's redundant-observation algorithm extends to global memory, but this
is a substantial extension of ConSan's semantic inventory and lowering, not an
additional delay setting or removal of the group-FLAT admission filter.

The main prerequisite is **knowing which accesses are ordinary, non-synchronizing
accesses**. In a compiler pass this distinction is available before lowering.
In an AMD binary it can be lost: the local compiler experiment below produced
identical instruction bodies for an ordinary load and a wave-scoped relaxed
atomic load on both gfx950 and gfx1201. An opcode-only rule cannot distinguish
them. Existing synchronization reconstruction helps, but cannot recover facts
that no longer exist in the binary.

Recommended first implementation: an opt-in global SuperCollider path with
hash-bound, per-site semantic provenance for reviewed workloads, followed by
compiler-produced provenance for general use. Preserve explicit unknown and
excluded coverage states. Do not advertise complete coverage by assuming every
unrecognized memory instruction is an ordinary access.

## What the original SuperCollider establishes

Read alongside the supplied [paper](/home/benoit/workspace/SuperCollider.pdf)
and [appendix](/home/benoit/workspace/SuperColliderAppendix.pdf):

- Sections 2.4 and 3.1–3.2 instrument weak loads and stores. Atomic, volatile,
  and other strong/synchronization accesses are excluded. A load is compared
  with a delayed coherent readback; a store's value is compared with a delayed
  readback. The original operation is retained.
- Section 3.3 adds an independent same-warp address-collision check for ordinary
  stores. This catches equal-value conflicting writes that readback cannot.
  Thread-private addresses must be filtered out.
- Section 3.4 checks both sides of asynchronous copies; it is a separate feature,
  not automatically provided by ordinary global load/store support.
- Section 3.5 perturbs warp/subwarp and block scheduling. Cross-block races can
  remain invisible when the conflicting blocks never execute concurrently.
- Appendix A's soundness argument assumes a weak original access and a coherent,
  non-synchronizing duplicate observation. Its simplified memory model is not
  itself a proof for AMD cache behavior or binary-level access classification.

The paper expressly allows false negatives. “No false negatives” can be a gate
for a specified set of injected faults and controlled schedules; it cannot be a
universal property of this value-change detector. Stable equal-value writes,
changes followed by restoration, and unobserved interleavings remain limitations.
No-false-positive requirements must include preservation of program outputs,
legal atomic accesses, and valid synchronization protocols.

## Evidence gathered on this host

Artifacts: `/home/benoit/workspace/consan-validation/sc-global-20260924/`.
These are feasibility and baseline runs without global detector instrumentation.

| Experiment | Result | What it establishes |
| --- | --- | --- |
| HIP global store/copy, compiled for gfx1201 and gfx950 | Exact oracle passed on physical gfx1201 and gfx950 functional emulator | Both execution paths are usable |
| Unchanged PyTorch scatter-reduce, BF16 and FP32, 1,024 inputs / 64 bins | Exact collision-count oracle passed on gfx1201 | Native workload remains available |
| Same PyTorch workload under gfx950 emulation | `hipErrorInvalidKernelFile` at `torch.ones`, before scatter-reduce; older installed wheel also rejected a setup image | Emulator/PyTorch baseline prerequisite remains unresolved; no detector conclusion |
| Pinned TokenSpeed production stage 1 + stage 2 under gfx950 emulation | CPU FP32 SwiGLU/weighted-expert oracle passed; maximum absolute error 0.00572324, permitted 0.04238554 | Actual production kernel code can run at a reduced correctness-test shape |
| Ordinary versus relaxed wave-scoped atomic load, HIP compiler output | Identical instruction bodies on both targets | Binary opcode classification alone cannot guarantee sound global instrumentation |

The TokenSpeed probe uses source revision
`a0cc3bb4d2b46e9995ae79a5ca1966aee00abc68` and
`tokenspeed-triton==3.8.10.post20260906`, matching the benchmark campaign's pins.
The local PyTorch is `2.14.0+rocm10.2.0a20260918`. Dependencies were installed into
an artifact-local directory, without replacing packages in the existing venvs.

Probe shape `(tokens, experts, hidden, intermediate, topk)` is
`(2, 4, 512, 256, 2)`, versus benchmark `(8, 256, 7168, 256, 8)`. It invokes
`invoke_stage1_warp_decode_gluon` and `invoke_stage2_warp_decode_gluon` directly.
It does **not** qualify the unchanged benchmark adapter, its weight-preparation
path, its conversion/copy kernels, or the original shape. An initial GPU-side
output cast hit the same PyTorch image-load problem; copying BF16 output to CPU
before casting allowed the production kernels' oracle to complete. No emulator
timing is a benchmark result.

### Access classification experiment

`classify.hip` and `classify-{gfx950,gfx1201}.s` retain the source and assembly.
`plain_load` uses `p[threadIdx.x]`; `wave_load` uses
`__hip_atomic_load(..., __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WAVEFRONT)`.
Their complete instruction bodies match on each architecture.

System-scoped relaxed/acquire probes do have distinguishing controls in these
examples: gfx950 uses `sc0 sc1`, and gfx1201 uses `scope:SCOPE_SYS`. The acquire
variants additionally invalidate caches after completion. This is useful evidence
for reconstruction, not a guarantee that cache bits uniquely encode source
atomicity. The [LLVM AMDGPU backend guide](https://llvm.org/docs/AMDGPUUsage.html)
documents architecture-specific memory-model lowering; the
[modifier reference](https://llvm.org/docs/AMDGPUModifierSyntax.html) documents
cache-policy controls. Coherent readback must follow the selected target's rules,
with completion waits and a scoped load that does not add an acquire/release
protocol or mutate the observed memory.

### PyTorch BF16 CAS-loop seed

The installed `torch/headeronly/cuda/Atomic.h` implements ROCm BF16 `gpuAtomicAdd`
through `AtomicFPOp<BFloat16>`. It reads the containing 32-bit word using a plain
load, computes an updated half, and retries `atomicCAS` until successful. The
initial read may observe a value that another update subsequently changes; CAS
validates the assumed value before committing. Packed halves share this word.

A detector report at that plain seed load is not evidence of a lost reduction
update. Conversely, a passing numeric oracle does not prove that a mixed
non-atomic/atomic source access satisfies the language memory model. Classify
this case explicitly before using the unmodified BF16 path as a universal
no-diagnostic oracle. Preferred remedies are a genuinely atomic seed load in the
workload/library, or a narrowly proven CAS-retry idiom policy with its exemption
reported. Never exempt arbitrary loads merely because they are near an atomic.
Confirm the exact generated site in the selected wheel before applying a policy.
