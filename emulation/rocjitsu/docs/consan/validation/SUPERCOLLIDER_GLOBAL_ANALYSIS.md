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

### A concrete false-positive trap

The colliding-store probe is stronger than the load-classification example.
`plain_collision_store` and `wave_collision_store` also have identical instruction
bodies on each target. The latter uses a relaxed wave-scoped atomic store.
With one wave, every lane may legally write its lane ID to the same location.

A standalone naive readback comparator, using an atomic system-scoped load,
reported **31/32 mismatches on gfx1201** and **63/64 on emulated gfx950** while
the atomic program's oracle passed. Those are expected differences in a legal
atomic program, not races. This is not a test of the current ConSan detector;
it demonstrates why an unqualified global-store replay extension would be wrong.
The checked-in `supercollider_global_semantics_probe.hip` preserves both the
compiler examples and this executable counterexample. Its successful exit
requires a valid atomic output and a positive naive-comparator mismatch count.

## Implementation map

Paths below are relative to `emulation/rocjitsu/`.

| Area | Current boundary | Required work |
| --- | --- | --- |
| Semantic inventory | `consan_analysis.inc` records native LDS, direct-to-LDS, and `flat_*` access sites; ordinary global instructions are counted but not ordinary access sites | Add explicit GLOBAL, buffer, and scalar-memory origins/forms; retain address expression, width, mask, value placement, cache controls, and semantic access role |
| Address spaces | `AccessAddressSpace::NonGroup` merges private and global; current FLAT admission rejects it | Distinguish global/private/group/unknown; runtime discrimination for generic FLAT where necessary; never treat private addresses as shared |
| Semantic admission | `consan_access_policy.cpp` reserves recognized synchronization sites and admits LDS/group-FLAT | Add a SuperCollider-specific global capability; reject or explicitly exclude atomic, volatile/MMIO, unknown-role, and unsupported forms; validate provenance against exact code bytes |
| Access lowering | `supercollider/consan_supercollider_flat.inc` is built around group-FLAT access/completion semantics | Separate target-specific coherent global probes with correct load/store completion, subword masking, original result preservation, full address capture, and cache policy |
| Resource planning | Existing owner/liveness analysis, SGPR/VGPR borrowing, spilling, descriptor updates | Reuse these mechanisms; budget saved 64-bit addresses and scalar operands, preserve EXEC/VCC/SCC, and test empty/partial waves and address/result overlap |
| Reports and validation | Existing site identity, report marker, coverage ledger and final patch validation | Carry global access kind/address/width, prove original access preserved exactly once, verify comparison/readback identity, count all admitted and excluded global sites |
| Scheduling | Existing timing perturbations and functional emulator scheduling | Test controlled interleavings and inter-workgroup residency; add block-order perturbation later if sensitivity requires it |

### Narrow first vertical slice

1. **Admission and negative tests first.** Define a per-site access-role contract
   with ordinary, atomic, volatile, private, and unknown states. A reviewed
   sidecar can bootstrap these two workloads, bound to the original code-object
   digest, instruction identity, and compiler/source provenance. A kernel-name
   allowlist is only a selection/performance control; it does not prove all
   accesses in the kernel are weak. Preserve existing synchronization reservations.
   For general binaries, compiler-emitted metadata is the reliable long-term path.
2. **Explicit GLOBAL vector accesses on gfx950 and gfx1201.** Support the measured
   load widths and stores, using target-specific coherent readback after local
   completion. Do not replay RMWs or implement readback by an atomic RMW. Retain
   the original weak load result for the application. Save the effective address
   before any destination register overwrites it. Compare bits, not floating-point
   equality, so stable NaNs do not become false reports.
3. **Scalar global loads and full workload accounting.** The TokenSpeed assembly
   uses SMEM for routing data as well as kernargs. Cover ordinary mutable-data
   scalar loads, or prove a specific input immutable and record that exclusion.
   Do not drop all SMEM as if it were private or constant. A vector-only result is
   explicitly partial until these sites are accounted for.
4. **Intra-wave ordinary-store collision checking.** Compare full 64-bit effective
   addresses and byte ranges across active lanes, respecting partial overlap and
   the configured same-value policy. This is independent of delayed value changes.
   Atomic stores must be excluded using semantic evidence, including the
   indistinguishable wave-scoped store case demonstrated above.
5. **Broaden forms separately.** Generic FLAT needs private/global discrimination;
   buffer accesses need descriptor addressing and exact out-of-bounds behavior.
   A discarded out-of-bounds store must not acquire a spurious readback mismatch.
   Formatted, packed, wide, and direct-to-LDS source observations need their own
   lowering/validation contracts. Direct-to-LDS destination checks alone do not
   cover races on the global source. Host/device and multi-GPU claims require
   separate coherence and lifetime validation.

For the local TokenSpeed reduction, stage 1 has three static
`global_load_dwordx4` sites and one `global_store_short`; stage 2 has three
`global_load_dwordx2` sites and one `global_store_dword`. These are **static
instruction counts, not dynamic coverage counts**. Both contain scalar loads,
and neither contains LDS instructions. Several loads overwrite their address
registers, e.g. `global_load_dwordx4 v[0:3], v[0:1], off`. Merely retargeting a
second load after the original would read through the newly loaded data. The
output layouts reduce across lanes and select output-owning lanes; a fault must
verify actual active writers. Do not assume two same-wave output owners exist
at a selected store. Test deterministic same-wave collisions in a separate
fixture when the production site only has one active writer per wave.

Keep this capability confined to SuperCollider. Enabling global accesses in the
Default engine would require a separate cross-workgroup memory/ownership model.

## Correctness qualification plan

Use the existing [validation rules](VALIDATION.md): exact site selection,
prospective faults, matching clean controls, reach witnesses, complete coverage,
GPU health, and independent numerical outcomes. Global support needs new faults;
weakening an atomic's ordering while leaving all accesses atomic is not a reliable
positive control for SuperCollider's weak-access detector.

| Test family | Clean / no-false-positive case | Positive case / expected detection |
| --- | --- | --- |
| Basic GLOBAL | Disjoint reads/writes, shared read-only inputs, separate kernels ordered on one stream | Controlled overlapping ordinary writes; weak read overlapping a writer |
| Atomics | Colliding FP32 atomic adds; wave/agent/system scoped atomic loads and stores | Reviewed removal of atomicity, replaced with ordinary load/add/store; actual lost-update race |
| BF16 scatter | Exact 16-per-bin oracle; diagnose and resolve the CAS seed policy; preserve adjacent packed halves | Non-atomic update fault with in-bounds collisions and an independent exact-count oracle |
| TokenSpeed | Both production stages versus CPU reference, valid routing IDs, unchanged clean output ownership | Alias real output-owning waves/programs to an overlapping in-bounds store range; add a controlled missing producer/consumer dependency |
| Register/address hazards | Loads overwriting address VGPRs, SGPR bases, offsets, masked lanes, EXEC=0, spills | Faulted address overlap still identifies the original site and byte range |
| Width and value | BF16 halves, bytes, 32/64/128-bit accesses, stable NaN bit patterns, disjoint neighboring subwords | Partial-overlap writes and distinct-value collisions; same-value stores handled by the explicit collision check |
| Memory/synchronization | Valid barrier and release/acquire publication; immutable scalar routing inputs; private FLAT | Missing publication with a witnessed value-changing interleaving; scalar input mutation where in scope |

For scatter-reduce, retain the untouched clean workload and CPU count oracle.
Audit the specific BF16 seed site before deciding whether a report is a detector
bug or a source-level memory-model issue. Do not hide it with a kernel-wide
suppression. Test packed-half preservation explicitly.

For TokenSpeed, restore the complete benchmark adapter's gfx950 baseline after
the PyTorch image-load issue is resolved, then inventory all four dispatches
identified in the physical campaign. Reuse the original native rocprofv3 allowlist
where the exact source/compiler/shape identity matches; otherwise record that a
new physical trace is unavailable and perform an explicitly emulated dispatch
inventory. The reduced two-stage probe is an implementation aid, not a replacement
for that gate. Keep the original shape's qualification separate from reduced
fixtures. Benchmark tables retain N/A until real hardware measurements exist.

Run deterministic emulator schedules for the controlled positive/negative tests,
then vary delays and workgroup scheduling. Record sanitizer detection and numeric
oracle independently: an output mismatch alone is not a race report, and a valid
output does not prove absence of races. For the finite test matrix require no
unexpected diagnoses, all expected detections in controlled cases, and unchanged
clean outputs. Statistical E2E qualification still uses its declared trial bar.
Emulator success proves behavior in that emulator; it does not prove all gfx950
cache/coherence behavior or physical scheduling sensitivity. Use gfx1201 hardware
for the corresponding real-cache tests while gfx950 hardware is unavailable.

## Reproduction

The two checked-in manual probes are:

- `tests/dbi/consan/device/supercollider_global_semantics_probe.hip`:
  compile with `hipcc -O2 --offload-arch=gfx1201 --offload-arch=gfx950` and run
  directly on this GPU or through the launcher below. Compile separately with
  `--cuda-device-only -O2 -S --offload-arch=TARGET` to inspect the access-classification
  examples. A positive naive mismatch count is the expected outcome.
- `tests/dbi/consan/consan_global_moe_probe.py --tokenspeed-dir CHECKOUT`:
  use a Python environment with PyTorch and the pinned `tokenspeed-triton` wheel.
  It directly imports the pinned production kernels, reports a CPU numerical
  oracle, and emits no performance measurements.

From the repository root, the launcher used here was:

```sh
/home/benoit/workspace/rocjitsu-rebase-20260923-build/tools/rocjitsu/rocjitsu \
  --config emulation/rocjitsu/configs/gfx950_mi355x_kmd.json -- PROGRAM ARGS
```

The current ROCm environment is recorded in
`/home/benoit/workspace/consan-validation/rdna4-20260923/env-current.sh`.
The artifact directory contains compiler assembly, source copies, emulator logs,
TokenSpeed's pinned checkout, isolated Python packages, and generated Triton
assembly/code objects. No production detector change was made during this analysis.
