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

### Cluster-load/sync dependency review

Both maintained clean profiles pass (`filtered-clean/pytorch-cluster-load-sync`).
The fresh barrier ELF `3e2955ef8a5cb650` contains one `ds_store_b32 v1,v2` and
one `ds_load_b32 v1,v1`. Between them, the address VGPR is unchanged. It is
`4 * (32 * wave_id + lane_id)` masked to the CTA's 256-element LDS region;
each lane owns its slot. The source stores and reloads the same blocked layout
without redistribution. The companion clustered copy kernel only reads/writes
global memory, also without a producer/consumer handoff. Dropping the cluster
or workgroup barriers therefore does not create an inter-wave LDS race in
this workload. Its yellow cells now explain this instead of promising an
unreviewed barrier fault. No synthetic barrier-drop result is counted as a
false negative or detection qualification.

### TP1 prefill and decode publication fault refresh

Compiling the maintained TP1 MLIR/flags reproduces inventory ELF
`b8bf8d46b98911e4` exactly (`tp1-review.vmfb`, `tp1-pristine.asm`). Both attention
kernels publish a per-wave maximum at LDS address `672+12*wave_id`, then read
peer slots using `672+12*(lane_id&7)`. The reviewed publication pairs are
`.text+0x8efc/0x8f00` for prefill and `0x13870/0x13884` for decode. Only those
pairs are removed; earlier barriers and subsequent read-completion barriers
remain. The stale barrier-move specs are replaced with these source-independent
ISA-reviewed publication faults, each with eight predeclared trials per mode
and minimum six detections.

### Bounded sparse ML rerun and report-cap hook rebuild

The bounded Default `tensile-spmm-f8-ml` run completed with all three clients
hitting their 900 s deadline. The first two shards reported complete applicable
coverage; the third ended with seven applicable code objects but only six
coverage records. This is an incomplete timed-out clean qualification, not a
qualified detector miss. Evidence: `bounded-clean-tensile/tensile-spmm-f8-ml/`
`default/tensile-spmm-f8-ml/clean/default/result.json`.

After that child exited, the held batch boundary allowed the shared hook to
rebuild successfully with the tested explicit report-cap change. The scheduler
resumed, and Qwen qualification now uses that hook. MXF4 remains orange until
the queued full clean rerun with an explicit 256 MiB report cap completes.

### CLIP BF16 publication fault review

The maintained compiler invocation reproduces ELF `3a959c36a47bfed7`.
The selected batch matmul requires 64 threads in two 32-lane waves. Its LDS
store address is `2304+40*(thread_id>>1)+2*((8*thread_id)&8)`: thread 32
writes address 2944, which thread 16 reads after the publication barrier.
The spec removes only `.text+0x4b98/0x4bec`, preserving the final pair.
Evidence: `clip-pristine.asm`, `clip-notes.txt`, and the fresh CLIP inventory.
Eight trials per mode and a six-detection threshold are fixed before runs.

### TP1 manifest admission repair

The current-hook TP1 prefill clean run passes with complete 228/228 accesses
and 42/42 barriers. Its first fault launch was rejected before execution by a
stale gfx1250-only barrier-move manifest override. Removed that override so
the reviewed publication-drop spec uses the normal barrier-drop family. The
catalog test now loads every maintained gfx1250 fault through the real admission
parser; all 208 validation runner tests pass. No rejected launch is counted as
an admitted trial.

CLIP Default qualification (`clip-cap-default-{clean,fault}`) finishes at 3/8
detections. All trials are admitted/reached, with complete analysis and healthy
pre/post checks; clean passes and provenance files match. Next test `high`.

### TP2 publication fault refresh

Fresh compilation matches inventory ELF `6a4133b943b7fa3e`. The attention
maximum reduction writes `672+12*wave_id`, then reads
`672+12*(lane_id&7)` after `.text+0x90f4/0x90f8`. The spec removes that pair
and preserves entry and read-completion barriers. Replaced obsolete executable
specs tied to ELF `a78b10652108d8d6`; those historical results remain in git.
Evidence: `tp2-pristine.asm` and the fresh family inventory. Decode/combined
are supporting clean-oracle rows with no standalone admitted fault families;
the family publication fault runs prefill. Their empty inventories do not
establish absence of LDS accesses.

### Sparse ML SuperCollider transform failure

The bounded SuperCollider rerun terminates in about 27 seconds for each shard:
ConSan returns `outcome=invalid errors=1`, then rejects loading with strict
exit 92 (`transform-error`). No numerical result is produced, so this is a red
instrumentation failure, distinct from Default's 900 s timeout. The first
rejected input image is 2,260,856 bytes. Evidence: the three clean run logs and
`bounded-clean-tensile/tensile-spmm-f8-ml/supercollider/.../result.json`.

### MXF8/F4 TDM bounded Default result

Two of three Default clean shards pass in about 41 seconds, with complete
768-access/204-barrier/12-atomic/12-fence coverage. The third hits its 300 s
client deadline without a final analysis verdict. The cell remains orange,
now describing this measured timeout instead of the historical OOM. Evidence:
`bounded-clean-tensile/tensile-sk-mxf8f4gemm-tdm/default/`.

### SuperCollider subword store bank repair

The sparse ML invalid transform is `could not form sub-dword store difference`.
The new regression reproduces it for a gfx1250 byte store whose value is in
physical VGPR bank one. The subword XOR had encoded the physical register
number directly in a field limited to bank-local indices. It now selects the
source bank for Src0, keeps readback/destination in bank zero, and restores
bank zero before masking. The final zero comparison needs no guest bank.
The test checks byte/halfword low/high forms in all three nonzero banks and
restoration of the original guest mode. Before repair it fails with the exact
E2E error; afterward all 127 selected SuperCollider/check-trap host tests pass
(`subword-bank-before.log`, `subword-bank-after.log`). The table stays red
until the E2E clean run is verified.

### Sparse ML isolated-hook execution after the bank fix

Linked an isolated hook from the refreshed `rocjitsu_dbt_internal` archive;
the shared campaign hook is unchanged. The retained 16x16x64 client now exits
zero, its numerical rows pass, and coverage completes at 22,236/22,236 accesses.
It also reports `marker=1 mismatch=true`, exposing a second issue beyond the
fixed transform failure. The cell remains red with this newer diagnosis.
Evidence: `spmm-sc-subword-client.log` and `spmm-sc-subword.log`. The maintained
full three-shard baseline+clean rerun is running under `spmm-subword-clean/`.

### TP1 decode/combined Default qualification

Current-hook Default clean passes, but the reviewed maximum-publication fault
is detected in 0/8 admitted/reached trials. Every trial has complete analysis
and healthy pre/post checks, and clean/fault provenance files match. Evidence:
`tp1-decode-cap-default-{clean,fault}`. The emulator-only workload lock is
confirmed in each result, allowing independent CPU simulations to run in
parallel while physical GPU jobs retain the global lock. Next test `high`.

### PyTorch norm/softmax publication fault review

Retained native norm ELF `596cff9a6cd4b098` has a block-X reduction at
`.text+0x2ef5c8/0x2ef5e8`. The matched rocprofv3 trace records workgroup
64x8x1: each 64-thread row writes its partial, then threads 0..31 read
partials from threads 32..63 after this pair. This is a cross-wave dependency
and the 64-wide launch necessarily executes the first reduction iteration.
The spec removes this pair only and predeclares eight trials per mode.
Evidence: `norm-reduce-pristine.asm`, `norm-review-images`, and
`discovery-pytorch-matched/pytorch-norm-softmax/trace-0/hocher/373598_kernel_trace.csv`.

### MXF4 mismatch localized to the wrong LDS address bank

The report-word watchpoint catches solution 6 (MT256x256x256), wave 3, writing
marker one at runtime PC `0x7fef9adf2de0`. Its kernel descriptor is at
`0x7fef9ac19dc0`; the matching retained ELF descriptor is VMA `0x19dc0`, so
the report write is VMA `0x1f2de0`. The preceding instrumented sequence is:
original `ds_load_b128 v[550:553], v548` under bank mode `0x82`, then reset
to bank zero and replay `ds_load_b128 v[0:3], v36`. The first comparison
therefore compares data fetched using different address registers. This is a
ConSan false positive; the replay must preserve the original address bank
while keeping scratch destinations in bank zero. Evidence:
`mxf4-watch6-registers-gdb.log`, `mxf4-watch6-symbols.txt`, and
`mxf4-first-mismatch.asm`. A source repair and regression test are next.

### CPU contention and TP1 high timeout

TP1 prefill high clean passes. Five admitted/reached fault trials have no
detection; trial six exceeds its 60-second deadline before admission, so
this is not an eight-trial result. Host load exceeded 300 on 32 CPUs while
RAM remained below 28 GiB. The memory cap alone does not bound CPU contention.
The external `fault-cpu-steward.py` now admits at most two whole fault bundles,
holding other supervising CLIs between trials while already spawned children
finish normally. Its PID/state file records every held process, and its final
cleanup resumes held CLIs. No physical-GPU lock is changed. Retry affected
batches under lower contention before interpreting timeout results.

MXF8/F4 TDM SuperCollider also finishes with two of three clean shards passing
(768/768 applicable accesses each), while the third reaches the 300-second
client deadline without an analysis verdict. Its orange cell now reflects
this completed rerun. Evidence: `bounded-clean-tensile/`
`tensile-sk-mxf8f4gemm-tdm/supercollider/.../result.json`.

TP1 prefill Default finishes with 0/8 admitted/reached detections. This
corroborates the sampling shortfall seen at high, whose batch was truncated
by a timeout after five admitted trials; the cell retains the latest high
result while that retry is pending.

### CLIP Default qualified at high

The high clean baseline and instrumented run pass, and the reviewed tile
publication fault is detected in all 8/8 admitted/reached trials. Every trial
has complete analysis, healthy pre/post checks, and the high preset; clean and
fault provenance files match. Since Default was 3/8, high is the lowest
qualified preset at or above default. Evidence: `clip-cap-high-{clean,fault}`.

### Single-address LDS replay bank repair

The new regression reproduces incorrect replay bank handling for both loads
and stores in each nonzero address bank. The ordinary single-address replay
now selects the guest address bank through Src0, retains its destination in
the low scratch bank, and restores bank zero before comparison. Saved-address
and split-two-address paths are separate and still need their own high-bank
audit; this change does not claim to qualify those paths.

The regression fails before the repair. Afterward all 128 selected
SuperCollider/check-trap host tests pass (`address-bank-verified-tests.log`).
The isolated E2E hook will be linked under `address-hook/`, leaving both
shared and subword-fix hooks used by running jobs unchanged. MXF4 remains red
until its clean rerun completes without a mismatch.

### CLIP SuperCollider qualification complete

At sleep=15, the matching filtered clean run passes, but the reviewed
publication fault produces 0/8 detections. All eight trials are admitted and
reached, analysis is complete, pre/post health checks pass, and clean/fault
provenance files match. The cell stays yellow with the measured result.
Evidence: `clip-cap-sleep15-{clean,fault}`.

### MXF4 capacity rerun and remaining SuperCollider failures

`cdna5-20260925/mxf4-cap256-clean` completes all six baseline and Default
shards with return code zero, accepted numerical results, complete analysis,
and full applicable access/barrier/atomic/fence coverage. The explicit 256 MiB
report cap resolves the earlier capacity failure; fault qualification remains.
The isolated address-bank hook still reports a mismatch on MXF4 solution 6
(`mxf4-address-fix6-client.log`) despite numerical PASSED and complete coverage.
The first replay-address fix therefore does not yet qualify the clean workload.
`spmm-subword-clean` completes with failures: all three maintained shards reach
the 900 s client deadline and report clean SuperCollider mismatches. The
subword encoding fix removed the transform error but does not resolve this row.

### Qwen sleep=15 deadline

`qwen-cap-sleep15-clean/qwen-prefill/clean/supercollider/result.json` reports
exit 124 after 360 s with no final analysis verdict. Baseline passed in 70 s.
The driver correctly stopped before starting fault trials. Earlier delay-zero
clean evidence remains valid, but sleep=15 qualification needs a longer rerun
with less CPU contention; the table now exposes that blocker.

### Second MXF4 address-bank failure: saved address

The isolated first fix moved the first mismatch to patched ELF PC `0x1fa4d4`
(`mxf4-watch-address-gdb.log`, `mxf4-second-mismatch.asm`). Before the guest
`ds_load_b128 v[804:807], v549 offset:992`, the preserved-address move copied
low-bank `v37` into scratch `v4`. The replay therefore still read another LDS
address. Preserve Src0's guest bank while selecting low-bank destination for
this move, and then restore the complete guest mode for the original access.
The regression covers all three high banks and the preexisting low-address /
high-destination test still passes. `saved-address-verified.log` reports all
129 selected ConSan check/trap and SuperCollider host tests passing. The earlier
`saved-address-after.log` used overly strict instruction encoding expectations;
the final test accounts for the transition encoder clearing and setting the
retained source bank, which is semantically equivalent.

With the isolated `saved-address-hook`, the original retained MXF4 solution 6
now returns zero, numerically PASSED, marker zero / mismatch false, and complete
2448/2448 applicable access coverage (`mxf4-saved-address-fix6-client.log`).
Full six-shard qualification is running in `mxf4-saved-address-clean`; this
single-client diagnostic does not qualify the entire table cell.

### Qwen Default fault result

`qwen-cap-default-fault/qwen-prefill/faults/barrier-drop-reduction-publication/summary.json`
finishes eight admitted and reached trials with zero detections and zero oracle
manifestations. Each result has complete analysis and healthy before/after
checks. Default clean passed. The next `high` preset is now running as a fresh
clean/fault bundle in `qwen-cap-high-*`, retaining the same reviewed fault site.

### TP2 family Default fault result

`tp2-cap-default-fault/tp2-family/faults/barrier-drop-attention-max-publication/summary.json`
completes eight admitted/reached prefill trials: zero detections and zero oracle
manifestations. All eight have complete analysis and healthy before/after checks.
All three family clean workloads passed previously. The next preset is `high`;
decode/combined remain supporting clean runs, not independent injected faults.

### torch.mode publication fault review

The fresh native code dump `pytorch-torch-mode-review-images` and
`mode-pristine.asm` show the maintained 1x128 case initializes shared input
with 64 threads, confirmed by the matched rocprofv3 trace. Thread 16's first
sorting load reads thread 32's published value, crossing a wave32 boundary.
The two consecutive split barrier pairs at .text+0x2fcb8/0x2fcbc and
0x2fcc0/0x2fcc4 jointly protect this publication; dropping only one leaves
synchronization intact. The new `barrier-drop-input-publication-group` removes
both and retains later sorting barriers. Eight trials per mode and six required
detections are declared before outcomes. `mode-spec-tests.log`: 208 validation
runner tests pass, including admission of every gfx1250 fault specification.

The retained sparse ML client also passes numerically with marker zero after
both address-bank fixes (`spmm-sc-saved-address-client.log`); full three-shard
qualification remains required before changing its red cell.

### TP1 decode high result

`tp1-decode-cap-high-fault` completes eight admitted/reached trials with no
detections and no oracle manifestations, matching the earlier Default result.
Every trial has complete analysis and healthy before/after checks. The paired
high clean run passed. The next preset to test is `higher`.
