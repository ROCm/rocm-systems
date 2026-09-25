# CDNA5 emulator revalidation — September 25, 2026

Campaign artifacts: `/home/benoit/workspace/consan-validation/cdna5-20260925/`.
Initial RocJITsu revision: `6aff016c3ce`, normal GCC build. Runtime and external
workloads follow [the audited setup](CDNA5_WORKLOAD_AUDIT_20260925.md).
The live qualification ledger is [STATUS_CDNA5.md](STATUS_CDNA5.md).

## Discovery prerequisite

The initial rocprofv3 kernel-trace attempt under the gfx1250 launcher aborted
before dispatch: canonicalizing `librocprofiler-sdk.so` returned ENOSYS.
The same profiler loads outside emulation with `/bin/true`. Evidence is in
`discovery/d128-block/probe.log` and `discovery/host-loader-check.log`.
The early legacy-stat failure was fixed in `3cc0f3ba80b`, and early open/close
handling in `a318db66e08`. Each passed a normal GCC rebuild and the interposer /
fork suite (70 passed, seven configuration-dependent skips).

Further tracing showed the profiler reading host topology and opening a host
render node before the emulator constructor. Do **not** pass those GPU ioctls
through. Prepending **both** profiler libraries with rocprofv3's `--preload`
corrects the constructor order:

```sh
--preload "$CONSAN_VALIDATION_ROCM_ROOT/lib/rocprofiler-sdk/librocprofiler-sdk-tool.so" \
          "$CONSAN_VALIDATION_ROCM_ROOT/lib/librocprofiler-sdk.so"
```

With this added to the rocprofv3 command inside the gfx1250 launcher, D128's
two numerical tests pass and kernel/agent CSVs are produced. Retained evidence:
`discovery/preload-both.log`, `discovery/d128-preload-both/`, and the generated
`discovery/d128-block.allowlist`. This is emulated dispatch discovery, not a
physical trace or performance measurement.

## Initial unfiltered clean control

`unfiltered-d128-block/` retains independent baseline, Default and SuperCollider
clean runs with all kernels eligible for instrumentation (no allowlist). All
three were accepted. Both engines patched 267/267 applicable access sites;
Default additionally patched 276/276 barriers. Static and dynamic completeness
passed with no unexpected diagnostics. These runs establish yellow cells;
reviewed prospective fault trials remain pending. Future filtered results will
use separate artifact roots and matching clean controls.

## Remaining hip-moi clean controls

`unfiltered-d128-pressure/`, `unfiltered-wmma-attention/`,
`unfiltered-streamk-arrival/`, and `unfiltered-tree-atomic-or/` each passed
baseline, Default, and SuperCollider clean qualification. Their cells are yellow
until prospective fault trials complete. The current D128 exact fault inventory
is retained under `unfiltered-inventory/d128-block/inventory/`; it is accepted
and has fresh code-object hashes and instruction addresses.

## Filtered campaign

All 21 non-Tensile validation IDs now have successful emulated kernel-dispatch
traces and generated allowlists under `allowlists/gfx1250/`. The profiler agent
CSV identifies gfx1250 (target version 120500), not the host gfx1201.
`discovery-batch/` retains standard SDK discovery; `discovery-pytorch-matched/`
retains PyTorch discovery. `discover.py` and `discover_pytorch.py` preserve the
exact commands and environment construction.

PyTorch must use its matching `_rocm_sdk_core/bin/rocprofv3`, `--rocm-root`
pointing at that package, and its versioned `librocprofiler-sdk-tool.so.1` and
`librocprofiler-sdk.so.1` in the preload pair. Put that package's `lib`,
`lib/llvm/lib`, and `lib/rocm_sysdeps/lib` first in the profiler's library search
path. Using the newer home-venv profiler with PyTorch's older LLVM bundle fails
with duplicate `PointerFlowAnalysisResult` registration. All eight PyTorch
operations successfully profile with the matched bundle.

Filtered clean results are under `filtered-clean/<id>/<profile>/`; each profile
has an independent baseline and matching coverage/oracle checks. Tensile uses
its host driver and an inner emulated client, with discovery performed by
`profile_tensile_client.sh` through `CONSAN_VALIDATION_TENSILE_WRAPPER`.
`discovery-tensile/` includes every selected shard, not just a canary shard.
Actual filtered Tensile clean results use `filtered-clean-tensile/`; the earlier
`filtered-clean/tensile-*` attempt stopped at argument parsing before execution.

D128's reviewed grouped K-publication fault is committed in the maintained
`consan_validation_faults_gfx1250.json`. Pristine disassembly and dry-run inventory
establish the current code hash and two physical publication barrier pairs.
`d128-fault-default/` runs eight precommitted trials per detector using that spec,
with gfx1250 emulator health discovery and dispatch smoke before/after each.

`torch.topk` Default hit the initial 30-second deadline during patching. The
otherwise identical `--timeout 180` retry passed baseline and clean qualification
in `topk-longer-clean/` (instrumented elapsed 50.98 s under campaign load).
The initial timeout remains preserved, and this is not a detector failure.

D128 delay-zero SuperCollider completed all eight admitted/reached trials with
healthy pre/post emulator probes and zero detections. The grouped publication
fault and expectations were fixed before the trials. Delay calibration remains
pending; no trial was replaced or omitted to improve the fraction.

## Issues uncovered by filtered clean runs

- `tree-atomic-or` Default completes its numerical test, but ConSan reports three
  cross-wave partial-store/final-consumer conflicts (return 89). Coverage is
  complete: accesses 32/32, barriers 6/6, atomics 2/2, fences 3/3. The bounded
  atomic-publication journal is currently enabled only for RDNA4 and CDNA4 in
  `consan_observation_policy.cpp` and `consan_sync.inc`; CDNA5 emits zero journal
  events. The missing CDNA5 port is the leading explanation, to be checked by
  regression and workload tests. Its native FLAT/GLOBAL format adds
  `scale_offset` to the RDNA4 layout, so blindly reusing the RDNA4 rewrite is
  insufficient. Earlier unfiltered clean success did not exercise a conflicting
  sampled access pair and is superseded by this failure.
- HipKittens naive BF16 aborts with `corrupted double-linked list` after both
  instrumented profiles. Static/dynamic coverage passes; baseline and profiler
  discovery succeed. The fault reproduces under GDB; retained logs are
  `hipkittens-teardown-gdb*.log`. This is a real clean execution failure, with
  root cause still under investigation.

D128's first Default matrix detected 1/8 admitted/reached faults; delay-zero
SuperCollider detected 0/8. Both matching filtered clean controls pass. The next
Default calibration is `high`, with a new matching clean control and the same
reviewed fault; artifacts are under `d128-high/`.

Tensor-descriptor add also reproduces the teardown heap corruption in both
profiles. HipKittens' full GDB stack reaches `KfdProcess::unmap_pages` while
freeing `LegacyPageTableEntry::host_extents`; this identifies where corruption is
detected, not yet where it originates. `hipkittens-teardown-gdb-full.log`
preserves the stack for follow-up.

`pytorch-scatter-reduce` returns zero and passes its numerical oracle in both
profiles, with no applicable LDS code objects (0/0 accesses). These are yellow
scope-limit cells, not detector failures. All three TP2 IDs now pass filtered
clean controls independently in each profile.

D128 Default qualifies at `high`: matching filtered clean passes and all eight
admitted/reached faults are detected, with complete coverage and healthy probes.
The qualification audit compares clean/fault provenance files and rehashes every
input. `default` had 1/8, so `high` is the lowest passing preset at or above
default. Evidence: `d128-high/`.

### Full-shard Tensile discovery completed

The discovery batch completed ten of twelve families across every shard.
`mxf8f4gemm-tdm` shard 2 and `sgemm-quick` shards 3–5 returned nonzero;
the mixed-format log records profiler finalization on SIGTERM at the client
deadline. Their allowlists are not admitted as complete. These two rows remain
orange while the interrupted shards are retried with longer deadlines.
Evidence: `cdna5-20260925/discovery-tensile/*/result.json` and command logs.

### CDNA5 publication journal activated and clean tree recovered

The CDNA5 target now enables publication modification capture, using its native
return-observation encoding (including the scaled-address bit). The out-of-tree
`tree-atomic-or` filtered Default clean run now passes with complete coverage:
`cdna5-20260925/publication-tree-clean/tree-atomic-or/clean/default/result.json`.
Fault qualification remains pending.

Normal GCC verification: all 779 `ConSan.*` host tests and all 67 gfx1250
atomic/ordering/publication device tests pass. Logs: `publication-host-tests.log`
and `publication-device-retest.log` under the campaign directory. The three
arrival/store/tree positive fixtures initially failed because their shared gfx1250
helper hid its LDS store in inline assembly, preventing compiler wait insertion.
Disassembly confirmed no producer `s_wait_dscnt`; the publication trace correctly
retained the producers without release evidence. Use the compiler-visible LDS
store already used by gfx950/gfx1201 for gfx1250 as well. The three positive and
three broken-address negative controls now pass (`publication-fixture-tests.log`).
A temporary expanded journal print was reverted after diagnosis.

The shared helper change also requires rebuilding and running the rest of the
gfx1250 device fixtures before final qualification. Earlier external results
retain their recorded hook provenance; final qualification must match the updated
hook. The HipKittens/TDM teardown corruption remains a separate open issue.

### Host OOM and campaign memory containment

At 13:11 on September 25 the host hit global OOM and the kernel killed a
VS Code process. The retained `cdna5-20260925/oom-kernel.log` records 581 Python
processes totaling approximately 51 GiB RSS and 13 Tensile clients. Four outer
workload workers, per-workload shard concurrency, and Tensile's unrestricted
internal generation pools compounded. The discovery retry was also concurrent.
The interrupted runs have no accepted results and must not qualify cells.

All subsequent campaign commands, including discovery and builds, must use:

```sh
emulation/rocjitsu/scripts/consan-bounded-run.sh COMMAND [ARGUMENTS...]
```

This wrapper places all invocations in the **same** user systemd slice, with
aggregate `MemoryMax=40G`, `MemoryHigh=36G`, and `MemorySwapMax=1G`.
The limit applies to the campaign, leaving the desktop outside its group.
Parent/child cgroup membership and the effective 42949672960-byte limit were
verified. A limit hit is an infrastructure failure, never detector evidence.
Tensile generation now defaults to two CPU workers per shard (`--cpu-threads`);
outer campaign workload batches are serialized. Do not overlap discovery with
large clean batches. The 38 Tensile validation tests pass with an assertion that
the bounded worker setting reaches the actual Tensile driver invocation.

### Tree producer-release fault qualification

The fresh, precommitted producer-release fault has eight admitted/reached
Default trials, two detected, with healthy before/after checks and complete
coverage (`tree-publication-fault/.../summary.json`). The clean journal repair
therefore establishes correctness but the default sampling preset is below the
6/8 bar for this workload. The table records 2/8 while calibration proceeds.

The first directories named `tree-high-clean` and `tree-high-fault` accidentally
used default: setting `RJ_CONSAN_PRESET` on the parent is stripped by this runner.
These directories provide **no high-preset evidence**. Use
`CONSAN_VALIDATION_DEFAULT_PRESET=high` for both phases; the corrected run uses
`tree-high-v2-clean` and `tree-high-v2-fault`. Verify each retained environment and
hook configuration before accepting preset results.

Bounded Tensile clean reruns use `bounded-clean-tensile`, one outer workload at a
time and two generation workers per shard, all inside the shared 40 GiB slice.
The observed campaign peak during this pass was approximately 5.3 GiB. The
current HipKittens debugger reproduction still aborts during page-table unmap;
its loaded emulator and ConSan libraries are the expected GCC build paths.

### Tree high evidence audit and further Tensile clean results

The correctly configured `tree-high-v2` clean passes and eight admitted/reached
fault trials all report the expected producer/consumer LDS conflicts. Clean/fault
provenance file sets match, all retained environments select high, and health
checks pass. However, fault reports have `incomplete_publication_pairs=3` and
`dynamic_complete=false`: weakening the producer release leaves its relaxed RMW
captured as an opaque modification. This invalidates the publication object for
analysis. Despite the runner summary accepting 8/8, the table remains yellow.
A prepared fix observes supported relaxed RMW transitions without inventing
release/acquire roles; it still needs compilation and regression verification.

Bounded Tensile MXF8 TDM passes clean in both modes across every shard. MXF4 TDM
Default rejects all six shards with `outcome=unsupported` before execution,
exit 92 inside the client (driver return 1). This is an orange transform blocker,
not an OOM or a completed clean run. Evidence is under `bounded-clean-tensile`.
HipKittens also corrupts the heap with SuperCollider trap reporting, ruling out
the automatic four-byte report allocation as a necessary trigger for that repro.

### Relaxed publication fix verified; tree high qualifies

Commit `d08b6bb984e` observes a supported relaxed RMW transition without claiming
release/acquire roles. Verification: 806 host tests, 67 gfx1250 atomic/publication
device tests, and 144 gfx950/gfx1201 emulator atomic/publication tests pass
(`relaxed-observation-*-tests.log`). The rebuilt hook's tree clean passes and
faults detect 8/8 at high, now with `analysis_complete=true` in every trial.
Clean/fault provenance file sets match; all eight trials are admitted/reached,
select high, and pass before/after health checks. Artifacts: `tree-relaxed-clean`
and `tree-relaxed-fault`. The Default cell is green. Default's earlier 2/8 result
predates this fix; repeat default on the new hook before calling high the lowest
passing preset for the final campaign.

Tensile MXF4 SuperCollider finishes all six shards numerically, but each reports
a SuperCollider mismatch. The cell is red pending diagnosis. Its Default
unsupported-transform blocker is distinct. D128 SuperCollider sleep=15 clean
passes, and its eight-trial calibration is running.

HipKittens still corrupts the heap with a copy-helper-only allowlist, zero patched
workload accesses and no applicable ConSan code object. This rules out inserted
workload probes as a necessary trigger. The hook/runtime lifecycle now deserves
priority; the failure is not evidence that the numerical kernel itself is wrong.
The diagnostic is `hipkittens-copy-only-diagnostic.log`.

D128 SuperCollider sleep=15 finished: clean passes, eight faults admitted/reached,
zero detected, zero numerical failures (`d128-sleep15-fault/.../summary.json`).
The cell remains yellow. Preloading the ConSan DSO without registering it as an
HSA tool completes HipKittens cleanly (`hipkittens-preload-only-diagnostic.log`),
whereas registering the tool with zero applicable workload instrumentation still
corrupts the heap. This isolates the trigger to active tool behavior rather than
merely loading the shared library; the precise corrupting operation is not yet
identified. Removing ConSan environment settings is not a disable test: the
hook defaults to Default instrumentation.

### Heap-corruption root cause: publication tracker static destruction

The publication dispatch isolation singleton owned a vector of pending records
but was destructed before later HIP exit callbacks. Its destructor freed the
vector; subsequent callback reset/submission reused the dead vector, overwriting
allocations in the same 32-byte size class (including page-table host extents).
A normal GCC exit-order reproducer fails with the old header (exit 3: a new
allocation's sentinel overwritten) and passes with the process-lifetime holder
(exit 0). The checked-in exit-order regression and 39 other publication tests
pass. Fix: `f63dd04b087`.

On the still-unmodified hook, a debugger intervention skipping **only**
`PublicationDispatchIsolation` destruction makes HipKittens exit normally:
`hipkittens-skip-publication-dtor.log`. Skipping all hook uninstall instead is
not a valid workaround: it leaves stale signal tracking and aborts elsewhere.
The lifetime fix retains the existing reset/uninstall behavior. The normal-hook
E2E confirmation is queued after the live Tensile workload finishes; do not
reclassify the red cells until that confirmation succeeds.

Tree Default requalification on the relaxed-observation fix finished 1/8 with a
matching clean pass (`tree-relaxed-default-*`), confirming high as the lowest
passing preset at or above default for that hook.

### Normal-hook confirmation of the lifetime repair

The rebuilt normal GCC hook completes HipKittens and PyTorch tensor-descriptor
add cleanly in **both** modes. All four maintained-runner results are accepted,
including baseline numerical checks and full applicable coverage; artifacts are
under `lifetime-clean`. Their status cells move from red to yellow pending fault
qualification. All 67 gfx1250 atomic/publication emulator tests pass with the
rebuilt hook (`publication-lifetime-device-tests.log`). No debugger intervention
is used for these accepted results. This confirms the static-lifetime fix resolves
both external heap-corruption reproducers.

Tensile F8 Default completed six shards; its 511/512/513 square-size shards hit
the 420-second client execution budget (one retained log has 10 of 12 expected
numeric rows). The full row is orange, with no shard coverage or oracle waived.
The bounded scheduler has resumed with the rebuilt hook.

### HipKittens fault requalification after lifetime repair

Fresh inventory retains ELF `9fe162c3afef62c6` and tile-publication split pair
`.text+0x09bc/0x09cc`. Source `gemm_naive.cpp` publishes the cooperatively loaded
A/B tiles before peer reads. Pristine ISA (`hipkittens-pristine.asm`, text VMA
0x1700) confirms the last tile store and restored EXEC before the pair, followed
by `ds_load_b128`; the later reuse barrier remains intact. The 64x64x32 launch
executes one workgroup and one reduction iteration. Both modes now predeclare
eight trials with minimum six detections and independent numerical outcomes.
The existing matching clean results are under `lifetime-clean`.

HipKittens Default now qualifies at **high, 8/8**, after default detected 3/8.
`hipkittens-high-clean` and `hipkittens-high-fault` have matching file provenance,
accepted clean correctness, complete analysis on all eight admitted/reached
faults, and healthy pre/post checks. Every trial records `RJ_CONSAN_PRESET=high`.
SuperCollider delay zero detects 0/8 (`hipkittens-sc-fault`); sleep=15 calibration
is running with its own matching clean pass.

HipKittens SuperCollider sleep=15 also finishes 0/8, with matching accepted
clean and eight complete, healthy, admitted/reached trials
(`hipkittens-sleep15-{clean,fault}`). The cell stays yellow.

PyTorch TDM fresh inventory is retained under `tdm-lifetime-inventory`. Review
of both cached Triton variants found wave-partitioned tensor transfers: the
one-CTA variant uses four 320-byte padded LDS regions, while the two-CTA
variant uses four 160-byte regions within each CTA. Their ordinary LDS accesses
use the corresponding wave region. A dropped workgroup barrier therefore needs
further dependency review before it can count as an injected inter-wave race;
no TDM fault has been selected or run from this inventory yet.

### WMMA attention publication fault refresh

The old helper-function fault identity is stale. Fresh inventory and pristine
ISA select the FastContextPolicy K/V publication instead: subgroup zero's K/V
stores precede the two split pairs at `.text+0x153a4/0x153bc` and
`0x153c0/0x153c4`; peer K loads follow. Both pairs implement one logical source
barrier and must be removed together. Initialization and later score/reuse
barriers remain. Artifacts: `wmma-lifetime-inventory`, `wmma-pristine.asm`.
The replacement spec predeclares eight trials per mode, minimum six detections.

Tree SuperCollider sleep=15 completes 0/8 detections with accepted matching clean,
eight admitted/reached trials, complete analysis and healthy pre/post checks.
Evidence: `tree-lifetime-sleep15-{clean,fault}`.

WMMA Default finishes 2/8 (`wmma-lifetime-default-{clean,fault}`), with
matching accepted clean, complete analysis, and eight healthy admitted/reached
trials. High calibration and SuperCollider sleep=15 are running.
Tensile F8 SuperCollider finishes all nine clean shards successfully under the
bounded scheduler (`bounded-clean-tensile/tensile-sk-f8gemm-quick/supercollider`);
its fault qualification remains pending.

WMMA SuperCollider sleep=15 finishes with zero detections despite **8/8
independent numerical oracle failures**. Clean correctness and complete
coverage pass on matching provenance; all fault trials are admitted/reached
and healthy (`wmma-lifetime-sleep15-{clean,fault}`). This is a demonstrated
miss on a manifested publication fault, not merely an unmanifested injected
race. It stays yellow under the shared detection-rate scale.

WMMA Default qualifies at high: 8/8 detections, matching accepted clean,
complete analysis and healthy admitted/reached trials
(`wmma-lifetime-high-{clean,fault}`). Default was 2/8, so high is the lowest
passing tested preset at or above default.

### D128 pressure publication fault refresh

The old helper identity is stale. Fresh `pressure-lifetime-inventory` and
`pressure-pristine.asm` select FullKvDoubleBuffered16Key FastContextPolicy's
cooperative K/V publication. All subgroups subsequently read fragments staged
by peers. The pair group `.text+0x3c608/0x3c624` and `0x3c628/0x3c64c`
is removed together, preserving surrounding reuse and reduction barriers.
The complete row still executes both full-KV and wide-key variants; this fault
qualifies the preselected full-KV publication. Eight trials per mode, minimum
six detections, are declared before running them.

### Tensile MXF4 Default rejection diagnosis

A bounded GDB replay of the retained client confirms rejection during automatic
preparation, before lowering (`mxf4-evidence-after-exec.log`). The typed evidence
plan is `InsufficientReportCapacity/PerBufferCeiling`: 152,288,376 bytes required
versus 134,217,728 allowed. Its inventory contains 2,448 access ranges, 544
barrier events and 15,376 atomic events, producing 17,824 watchpoint/sync slots
and 1,140,736 publication events. The existing environment setting cannot exceed
the same hard ceiling. Earlier placement-breakpoint runs reproduced exit 92 but
did not locate the cause; GDB breakpoints must be installed after launcher exec.
The pipeline now reports the reason, required bytes and ceiling when rejecting
an incomplete report plan. A regression checks the diagnostic and fail-closed
behavior. The active campaign still uses the unchanged hook until a safe rebuild
boundary. A capacity-policy change or smaller complete dispatch inventory still
needs evaluation; no coverage or evidence requirement has been relaxed.

Tensile HGEMM Default completed its full clean row under the bounded scheduler
(`bounded-clean-tensile/tensile-sk-hgemm-quick/default`); fault trials remain pending.

The normal GCC host build passes all 52 selected pipeline/evidence tests,
including the new capacity-rejection diagnostic regression
(`evidence-diagnostic-tests.log`). Only the host test target was rebuilt; live
E2E jobs retain their existing hook and emulator binaries.

D128 pressure finishes 1/8 at Default and 0/8 at SuperCollider sleep=15,
with matching accepted clean runs and complete analysis on all healthy,
admitted/reached trials (`pressure-lifetime-{default,sleep15}-{clean,fault}`).
High Default calibration follows; numerical fault outcomes did not fail.

### Stream-K arrival publication fault refresh

Fresh `streamk-lifetime-inventory` and line-annotated pristine ISA identify the
user counter RMW at `.text+0xd600` (ELF `46225079c03c98b7`, occurrence 104).
Each wave publishes its LDS WMMA partials through lane zero's increment; the
wave seeing old value one consumes both waves' partials. The reviewed mutation
removes the release edge at `0xd5f0/0xd5fc` and preserves acquire at `0xd610`.
The returned value reaches the source consumer branch at VMA `0xfd04/0xfd10`.
The two stale helper-function faults are replaced by this one selected fault,
with eight trials per mode and minimum six detections predeclared.

Tensile HGEMM SuperCollider also finishes its full clean row successfully;
both mode cells now await fault qualification.

D128 pressure qualifies at **high, 8/8** with matching accepted clean and
complete, healthy admitted/reached trials (`pressure-lifetime-high-*`).
Default was 1/8, making high the lowest passing preset at or above default.
Stream-K finishes 0/8 at Default and SuperCollider sleep=15, with matching
accepted clean, complete analysis and healthy admitted/reached trials
(`streamk-lifetime-{default,sleep15}-*`). High Default calibration follows.

Stream-K Default qualifies at high with 8/8 detections, matching accepted
clean, complete analysis and healthy admitted/reached trials
(`streamk-lifetime-high-*`). Default was 0/8. The release-weakening fault remains
observable with the acquire edge retained.

### Remaining workload inventory refresh

All 13 serial inventory runs under `remaining-lifetime-inventory` are accepted:
Qwen, TP1 prefill and decode/combined, all three TP2 IDs, CLIP, PyTorch cluster
load, mode, topk, sort, histc, and norm/softmax. The old Qwen/TP1/TP2 fault
site identities do not occur in these current inventories, so they cannot be
reused without review. TP2 decode and combined expose no sites in this barrier
inventory; that is not by itself evidence of absent applicable memory accesses.
The remaining rows need source/ISA dependency review and fault qualification.

### Qwen reduction publication fault refresh

The current Qwen ELF is `d6952ee5d5087587`. Pristine ISA for the first
`main$async_dispatch_0_reduction_5x1024_f32` reduction stores each wave's sum
through lane zero, then reads peer wave sums after `.text+0x23b0/0x23d4`.
This publication pair is the reviewed fault; the earlier barrier remains.
Artifacts: `qwen-pristine.asm` and `remaining-lifetime-inventory/qwen-prefill`.
Eight trials per mode, minimum six detections, are declared before outcomes.

### Explicit report-cap expansion

The planner and hook now retain a 128 MiB default while accepting an explicit
cap up to 256 MiB. Exact allocation and the 4 GiB per-process aggregate report
budget remain enforced. The maintained runner forwards
`CONSAN_VALIDATION_AUTO_REPORT_BUFFER_SIZE` identically to Default clean/fault
runs and records it; 208 Python tests pass. All 67 selected report-planning,
pipeline and evidence host tests pass (`report-cap-host-tests.log`), including
the observed MXF4 inventory: a 256 MiB cap admits all 152,288,376 required bytes
and all publication-event slots, while the unchanged default still rejects it.
This host test binary contains no HsaHooksUnitTest suite; hook/E2E confirmation
is pending the safe rebuild after the active Tensile workload completes.
