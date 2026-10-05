# ConSan RDNA3 (`gfx1100`) status

The physical results below use native `gfx1100` code objects on a Radeon PRO
W7900. For the six hip-moi rows, baseline, Default, and SuperCollider clean
runs passed with complete applicable coverage. The 15 fault campaigns covered
17 profile or operating-point outcomes and 136 admitted and reached trials.
Each outcome used eight trials;
the table records the selected profile controls, detection counts, and
lower-preset misses needed to interpret each grade.

## P0 qualification and sensitivity

Default and SuperCollider are detector profiles, not workload priorities. The
evaluated external P0 rows are one rocBLAS SGEMM workload and three PyTorch
workloads: mode, sort, and norm/softmax. Both profiles have clean controls for
all four rows. The seven rocBLAS fault matrices and the PyTorch operation-level
campaigns below are separate evidence sets. The P0 fixture and Qwen prefill
rows in the status table are separate workloads.

Fault qualification checks whether a detector reports a deliberately introduced,
reviewed synchronization error often enough to meet the qualification contract.
A passing clean run establishes output preservation and applicable coverage;
it does not establish sensitivity to a fault. Trials must prove that the
intended mutation was installed and reached, with complete evidence and healthy
pre/post probes. The campaigns below require at least six detections in eight
trials at each declared operating point.

Fresh native discovery, allowlists, and fault inventories were generated for
rocBLAS SGEMM, PyTorch mode, PyTorch sort, and PyTorch norm/softmax.
All 24 baseline/clean results passed across these four workloads, including
Default/default, Default/high, SuperCollider standard, and SuperCollider
sleep=15 controls. Applicable static coverage is recorded below. The rocBLAS
Default/default clean run collected zero access records; its high control
collected 456.

The rocBLAS fault removes the initial LDS publication barrier before peer waves
consume the published data. The fixed 64-cubed case dispatches one workgroup
of four wave32 waves. All seven eight-trial matrices below were declared before
execution and retained every miss. Each had a passing matching clean comparator;
the four additional comparators bring the total to 28 accepted clean results.
All 56 fault trials
were admitted and reached, had complete evidence, passed the numerical oracle,
and passed pre/post health probes. These are observed counts for this workload
and fault, not a general detection-rate guarantee.

| rocBLAS operating point | Detections | Runner exit | Interpretation |
| --- | ---: | ---: | --- |
| Default/default: workgroup/cell strides 256/256 | 2/8 | 1 | Below bar; six trials collected zero records |
| Default/high: workgroup/cell strides 16/16 | 8/8 | 0 | Qualified high preset |
| Default: explicit workgroup/cell strides 1/256 | 8/8 | 0 | Qualified explicit controls; standard preset remains below bar |
| SuperCollider sleep=15, delays on reads and writes | 0/8 | 1 | Below bar |
| SuperCollider sleep=15, delays on reads only | 0/8 | 1 | Below bar |
| SuperCollider sleep=127, delays on reads and writes | 0/8 | 1 | Below bar |
| SuperCollider sleep=127, delays on reads only | 0/8 | 1 | Below bar |

`Runner exit` is the validation runner's exit code for the entire eight-trial
fault campaign, not an individual workload's exit code. The runner returns 0
when the campaign is accepted and 1 when it is rejected. In this table, the
exit-1 campaigns missed the required six detections; all their trials were
admitted and reached, with passing numerical and pre/post health checks. An
exit of 1 here does not by itself indicate a workload crash.

High changes both workgroup and cell strides from 256 to 16, and retained 456
access records in each fault trial. It detected the fault 8/8 against the
predeclared 6/8 threshold. With workgroup stride 1 and cell stride still 256,
the same fault was detected 8/8 with 75 records per trial. Thus High observes
more, but this experiment does not require denser cell sampling to recover the
measured Default misses.

**Default miss mechanism.** gfx1100 prefers a literal dispatch identity;
the hook supplies the code-object reader handle. The
[workgroup gate](../../../lib/rocjitsu/src/rocjitsu/code/patch/consan/consan_runtime_workgroup_gate.cpp)
mixes its low six bits with workgroup coordinates and tests the result against
the workgroup stride and offset. The standard preset uses stride 256 and offset
zero. This SGEMM launches only workgroup (0,0,0): its two Default/default hits
had reader identity modulo 64 equal to zero, selected that workgroup, and each
retained 75 access records with 12 conflicts. The six misses had reader identity
modulo 64 equal to 16, failed the workgroup gate, and retained zero records.
The emitted gate was inspected to confirm the bypass. The kernel and installed
fault still ran; Default did not observe LDS accesses from the excluded group.
Cell sampling operates only within a selected workgroup, so changing its stride
alone cannot recover these misses. Selecting every workgroup with workgroup
stride 1 while retaining cell stride 256 produced 75 records and detections in
all eight trials, without increasing the eight banks per range. Whole-workgroup
exclusion explains these measured misses;
additional bank capacity or LDS opcode support is not needed to recover them.
Reader-allocation-dependent selection also makes cross-run detection counts
insufficient evidence of a sensitivity change without checking what was sampled.

**SuperCollider limit and remaining uncertainty.** The active replacement
kernel descriptor and disassembly were inspected, not just the retained
original text. The selected barrier is a NOP; the consumer retains an LDS load,
`s_sleep 15`, a duplicate load, a completion wait, and value comparisons.
All four SuperCollider configurations produced no mismatch and no numerical
failure.
The physical gates passed 10/10 with no skips, including
`ConSanGfx1100Physical.SuperColliderAllSupportedSites`, whose fixture requires
a positive mismatch marker. This supports a workload sensitivity limitation;
it does not independently prove every rocBLAS replay probe correct.
SuperCollider observes value instability, so an unordered access pair need not
produce a mismatch.
Per-wave timing and observed values are not retained by its sticky marker;
the exact interleaving responsible for this workload's misses remains unknown.

**Implementation decisions.** Existing independent workgroup/cell selectors
provide a measured Default workaround. A policy guaranteeing observation of
small grids, with reproducible selection independent of reader allocation, is
worth designing and benchmarking; automatically changing the standard preset
would affect overhead and needs separate evidence. Default sampling is the
current P0 investigation; SuperCollider sensitivity work is deferred. RDNA3
currently lacks
`sleep_wave` support in both the runner and
[replay delay emitter](../../../lib/rocjitsu/src/rocjitsu/code/patch/consan/targets/consan_supercollider_target_ops.cpp).
A bounded RDNA3 wave-dependent-delay prototype is a reasonable next
SuperCollider experiment, not an established fix: retain it only with matched
clean checks, instruction/state-preservation tests, measured detection gain,
and overhead measurements. Larger uniform delays and reads-only delays did not
improve detection in these trials. The proposed policy and delay support require
separate implementation and validation.

**PyTorch operation-level evidence.** These rows invoke PyTorch operations,
not entire models. Mode's primary compute kernel uses two wave32 waves; sort's
primary radix-sort kernel uses four, alongside preparation/copy dispatches.
The combined norm/softmax row invokes two distinct operations and kernels, so
its fault results must be reported separately. Native ISA review selected
publication barriers adjacent to LDS producer/consumer instructions for each
operation. These site-specific experiments do not establish sensitivity for
every barrier or every input shape.

All four operations passed native and Default/high clean controls. Mode has
244/244 applicable access and 51/51 barrier sites, sort 234/234 and 40/40,
norm alone 14/14 and 9/9, and softmax alone 8/8 and 6/6. Norm and softmax
also have accepted component-only clean controls. A zero profiler
`LDS_Block_Size` does not establish absence of LDS instructions. The combined
row's 22/22 access and 15/15 barrier sites are the sum of its two components.
The inventory runner stops after the first relevant code object, so an
inventory of the combined row alone exposes only norm's candidate barriers;
component-only inventory and ISA review were needed for softmax. Inventory
alone does not prove mutation installation or runtime reach.

The Default/high campaigns below used eight predeclared trials per selected
site and a 6/8 acceptance bar. Every listed trial admitted its mutation and
passed pre/post GPU health checks. A detector-owned runtime diagnostic proves
reach for every reported detection. `Diagnostic` is the count out of eight
planned trials, not a population-wide detection probability.

| PyTorch operation / selected fault | Default preset | Diagnostic | Runtime reach proved | Numeric oracle | Runner exit / result |
| --- | --- | ---: | ---: | --- | --- |
| `torch.mode`, adjacent LDS publication barriers | high | 8/8 | 8/8 | pass 8/8 | 0 / qualified |
| `torch.sort`, radix-sort LDS publication barrier | high | 8/8 | 8/8 | pass 8/8 | 0 / qualified |
| `torch.softmax`, LDS publication barrier | high | 8/8 | 8/8 | fail 8/8 under fault | 0 / qualified |
| `torch.linalg.vector_norm`, LDS publication barrier | high | 2/8 | 2/8 | pass 8/8 | 1 / below bar |
| `torch.linalg.vector_norm`, same barrier | higher | 5/8 | 5/8 | pass 8/8 | 1 / below bar |
| `torch.linalg.vector_norm`, same barrier | max | 5/8 | 5/8 | pass 8/8 | 1 / below bar |

The softmax fault changed the numeric output in all eight trials; the
detector's own diagnostic, rather than the failing oracle alone, counted as
detection. Runner exit 1 for norm means that the eight-trial campaign missed
the 6/8 qualification bar; it does not mean the workload crashed. Norm's
undetected trials have no independent runtime reach witness.
The mutation was installed, but installation and a passing numeric oracle do
not prove that the selected conditional barrier executed. At max, both
workgroup and cell sampling strides are one, yet only five trials produced
diagnostics. This rules out sparse sampling as a sufficient explanation for
that operating point. Scheduling or site reachability remains a hypothesis,
not an established cause. A dynamic fault-site execution witness or a
different reviewed site would distinguish them; the present evidence does not
justify a detector implementation change. The standard Default preset was
not fault-qualified for these PyTorch operations, and SuperCollider has clean
controls only.

An initial sort specification incorrectly requested a paired-barrier selector
for a singleton barrier. Admission rejected its only attempted trial before
the faulted workload ran; that rejected attempt is excluded from the table.
The corrected exact-site specification admitted all eight trials. No valid
fault trial was excluded from the counts.

This validation covers the stated clean controls,
fault matrices, and physical gates. It does not include a full project test
suite, GCC/UBSan lane, or performance benchmark.

## Reproduction and status

Reproduce these rows with the maintained
[validation runner](../../../tests/dbi/consan/consan_validation.py), the
[reviewed `gfx1100` fault specification](../../../tests/dbi/consan/consan_validation_faults_gfx1100.json),
and the linked allowlist procedure below. The runner's `manifest` and `explain`
commands provide the current target-native workload commands, profile
environment, correctness oracles, and fault policy.

Keep exact commands, exit statuses, toolchain versions, binary identities,
mutation offsets, and raw reports in the execution records. Regenerate native
inventories and review fault sites for the binaries under test. This status
document describes workload behavior and qualification limits independently
of a particular checkout, result directory, or revision.

The validation registry resolves all six hip-moi rows to target-native GFX11
fixtures and fails closed instead of substituting another architecture. A
profile is green only when its clean and reviewed fault campaigns meet the
shared qualification contract.

Start any new revalidation with
[rocprofv3-based allowlist discovery and application](VALIDATION.md#first-step-for-revalidation-generate-and-apply-kernel-allowlists).
Apply this before retrying recorded timeouts or raising deadlines; update their
status only after new runs provide evidence.

Physical evidence uses native `gfx1100` code objects on a matching GPU;
simulator prerequisites use RocJITsu `configs/gfx1100_w7900.json`. Exact counts
belong to the binaries used for qualification and must be refreshed after
relevant source, toolchain, workload, emulator, or runtime changes.

The [native publication tests](../../../tests/dbi/consan/hip_consan_rdna3_test.hip)
check output preservation with and without instrumentation. The aligned
32-bit store case must produce a complete publication trace. Unaligned stores
execute their original instruction because observation through atomic exchange
requires natural alignment; they must set the sticky dropped marker so the
decoder rejects the trace as incomplete. The tests keep the event count below
capacity so overflow detection cannot mask a missing dropped marker.

Shared [color scale](VALIDATION.md#status-colors): 🟩 qualified; 🟨 clean run
established, but fault qualification is pending/below bar or the workload is
outside detector scope; 🟧 clean qualification blocked by prerequisites,
unsupported applicable operations, incomplete coverage/evidence, or a timeout;
🟥 observed correctness or instrumentation failure. Empty/🩶 means unassessed
for this execution target; simulator prerequisites alone do not qualify hardware.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Physical compact gate | P0 | native two-wave LDS fixture (`ConSanGfx1100Physical.*`) | 🟨 exact clean output, visible records, zero diagnostics; reviewed conflict and broad campaign missing | 🟨 exact clean/all-sites rows and mutation containment; broad E2E fault campaign missing |
| Simulator prerequisite | P0 | native two-wave LDS fixture (`ConSanGfx1100Sim.*`) | 🩶 prerequisite passes | 🩶 prerequisite passes |
| Broad E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P0 | rocBLAS SGEMM 64³ (`rocblas-sgemm-square-64`) | 🟩 high (lowest passing): clean pass; access 232/232, barrier 5/5; initial-publication fault 8/8; `default` 2/8, six empty-report misses; explicit workgroup/cell strides 1/256 also 8/8 | 🟨 clean pass; access 232/232; sleep=15 and sleep=127, each with all-access or reads-only delay: four separate 0/8 batches (bar 6/8) |
| Broad E2E | P0 | PyTorch mode (`pytorch-torch-mode`) | 🟩 high: clean pass; access 244/244, barrier 51/51; selected publication fault detected and reached 8/8; standard preset unassessed | 🟨 clean pass; access 244/244; fault sensitivity unassessed |
| Broad E2E | P0 | PyTorch sort (`pytorch-torch-sort`) | 🟩 high: clean pass; access 234/234, barrier 40/40; selected publication fault detected and reached 8/8; standard preset unassessed | 🟨 clean pass; access 234/234; fault sensitivity unassessed |
| Broad E2E | P0 | PyTorch norm/softmax (`pytorch-norm-softmax`) | 🟨 combined clean pass; access 22/22, barrier 15/15. Component fault trials: softmax high 8/8 detected/reached; norm high 2/8, higher 5/8, max 5/8 diagnostics with miss reach unproved; combined row below bar | 🟨 combined clean pass; access 22/22; fault sensitivity unassessed |
| Broad E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 high (lowest passing): clean pass; access 278/278, barrier 138/138; grouped K-publication fault 8/8 (`default` 2/8) | 🟩 sleep=15: clean pass; access 278/278; grouped K-publication fault 8/8 |
| Broad E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 high (lowest passing): clean pass; access 503/503, barrier 36/36; grouped K/V-publication fault 8/8 (`default` 4/8) | 🟨 sleep=15: clean pass; access 503/503; grouped K/V-publication fault 0/8 (bar 6/8) |
| Broad E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 high (lowest passing): clean pass; access 115/115, barrier 18/18; grouped K/V-publication fault 8/8 (`default` 4/8) | 🟩 sleep=15: clean pass; access 115/115; grouped K/V-publication fault 8/8 |
| Broad E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 high: clean pass; access 14/14, barrier 1/1, atomic 1/1; release-order fault 8/8 | 🟨 delay=15: clean pass; access 14/14; release-order fault 0/8 (bar 6/8) |
| Broad E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 high: clean pass; access 18/18, barrier 1/1, atomic 2/2; producer release-order fault 8/8 | 🟨 delay=15: clean pass; access 18/18; producer release-order fault 0/8 (bar 6/8) |
| Broad E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟩 higher (lowest passing): clean pass; access 320/320, barrier 13/13; load-to-compute publication fault 8/8 (`default` 0/8, `high` 1/8) | 🟨 sleep=15: clean pass; access 320/320; load-to-compute publication fault 0/8 (bar 6/8) |
