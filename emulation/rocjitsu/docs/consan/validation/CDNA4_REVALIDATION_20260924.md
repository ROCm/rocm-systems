# CDNA4 revalidation, September 24, 2026

The physical gfx950 campaign follows the fault qualification rules in
[STATUS_RDNA4.md](STATUS_RDNA4.md). The current per-workload state is in
[STATUS_CDNA4.md](STATUS_CDNA4.md). All 50 cells have been regenerated: **20
qualified, 24 below the detection bar, two rejected clean controls, two
unavailable-fixture cells, and two outside LDS/FLAT coverage**. The 23 reviewed
fault families have 46 completed profile searches, including the two searches
that cannot qualify because their higher/max clean controls fail. The selected
evidence contains 135 complete eight-trial matrices (1,080 admitted/reached
trials). A passing clean run alone does not make a cell green.

## Qualification contract

Default tests `default`, `high`, `higher`, then `max`, stopping at the first
configuration with a passing matching clean run and at least six detections
in eight admitted and reached trials. SuperCollider independently tests
delay-zero, sleep=1 and sleep=15. An attempted wave-dependent sleep control
was rejected: that delay mode is implemented only for RDNA4. It contributes
no gfx950 qualification evidence. Every supported setting has its own clean
control. The numerical workload, exact reviewed
mutation and rocprofv3 allowlist remain fixed through each search. Failed
settings and original prospective expectations remain in separate artifacts.

Native discovery uses `rocprofv3 --kernel-trace --mangled-kernels
--output-format csv`, followed by `scripts/rocjitsu_consan_allowlist.py`.
The same list is passed through `CONSAN_VALIDATION_KERNEL_ALLOWLIST_DIR` to
clean, inventory and fault runs. Selected entries are checked against hook
load/instrumentation/dispatch records. Supporting TP2 decode and combined
commands retain their own exact clean oracles; the manifest assigns fault
qualification to `tp2-family`.

All physical work uses `/tmp/rocjitsu-consan-destructive-gpu.lock`. Each fault
trial retains both rocminfo and an uninstrumented native ordered-tile-handoff
smoke before and after execution. Numerical failure alone, a trap, timeout, admission failure or unhealthy GPU
is not counted as a detector success. Fault trials allow either numerical
outcome; an accepted detection requires an explicit ConSan diagnostic and
all admission, reach and health gates.

## Issues corrected during revalidation

- The bounded Tensile BF16 functional fixture completes in microseconds on
  this runtime. Its old 1 ms floor, derived from an older observed duration,
  rejected correct native and instrumented executions. Its duration floor is
  now zero, meaning **valid positive device timing is still mandatory**, while
  a correct functional run need not be slow. Performance workloads retain
  their duration requirements. Tests cover wrong output, zero/NaN device
  timing, the zero duration floor, and positive benchmark duration floors.
  Physical native, Default and SuperCollider clean rechecks pass.
- `explain` rejected `RJ_CONSAN_KERNEL_ALLOWLIST_FILE` as an unclassified
  setting. It is now classified as instrumentation selection; regression
  coverage exercises clean, inventory and fault environment auditing.
- Fault-command doctor/provenance probes initialized PyTorch on the GPU before
  the child trial took its lock. Those parent probes now acquire and release
  the same lock before child execution. A regression verifies exclusion and
  release on failure, including the child's ability to reacquire it.
- Nonzero wave-dependent sleep is RDNA4-only. The validation command now
  rejects unsupported targets and invalid sleep maxima before execution;
  regression tests cover the target and value restrictions.
- CPU fault-runner tests used the real GPU lock: the queued-containment test
  timed out, and the fake device-loss/quarantine test later waited behind
  physical validation. The test class now supplies a private lock shared by
  its fake child processes and quarantine probes. Exclusion and quarantine
  checks remain active without competing with physical GPU work.
- Top-k's first Default and SuperCollider fault attempts were rejected before
  execution: the reviewed two-singleton `s_barrier` group used an interface
  that admitted only groups of two signal/wait pairs. The grouped selector,
  transactional rewrite and final proof now admit complete full-barrier
  singletons within an explicitly selected group. Ordinary pair selection
  remains strict; partial signal/wait sequences remain rejected. A CDNA4
  regression verifies exact-one application, preservation of intervening
  instructions, duplicate/reversed selection rejection and incomplete-proof
  rejection. The runner also rejects missing group sequence identities early.
  Top-k and three hip-moi groups now pin both sequence identities from their
  original inventories without changing selected instruction sites. Physical
  retries use separate `*-grouped-v2` directories and matching clean controls;
  the rejected original attempts remain retained and do not qualify.
- Top-k's generic 30-second timeout expired during host transformation. Its
  new Default clean run alone spent about 25 seconds in inventory/patching.
  The first grouped Default fault stopped before installation; SuperCollider
  installed the mutation but timed out transforming a later object. Both
  post-run GPU health checks passed. Their timeout records remain retained,
  and the runner cleared each quarantine only after fresh locked native
  health checks. The gfx950 top-k bound is now 120 seconds; fresh
  `*-grouped-v3` clean/fault runs test this allowance without relaxing
  detection, coverage, oracle or health gates. Completed retry rows apply
  exactly one mutation and finish in about 56 seconds (Default) and 30 seconds
  (SuperCollider). The full batches qualify Default at `higher` with 8/8;
  SuperCollider remains below the bar at sleep=15 with 0/8.
- The first norm candidate was rejected **before any trial**: its native
  launch has X=64, which bypasses the wider-X reduction edge selected from the
  first object's static inventory. A targeted inventory of the later-loaded
  softmax object identifies its eight-wave partial-maximum publication
  barrier instead. The full 4x4096 softmax workload and common allowlist are
  unchanged. The rejected unrun candidate and replacement ISA review are both
  retained.
- The original 128-element `torch.mode` fixture launched a single wave64
  block on gfx950. Its sorting barrier trials cannot qualify inter-wave
  detection and are excluded from the ledger. The gfx950 fixture now requests
  256 values: native rocprofv3 shows `compute_mode<int,1024>` with 512 threads
  (eight waves), and the numerical baseline passes. A fresh two-kernel
  allowlist and static inventory select the stride-128 to stride-64 publication
  barrier at `.text+0x255dc`. The ISA review identifies a wave-0 key write read
  by wave 1, within the valid input range. Separate `*-mode-v2` campaigns retain
  this mutation and input across all presets/delays. RDNA4 keeps its prior
  input, and a regression checks the target-specific clean/fault commands.
- Building the external corpus required changing the local FP8 `4_wave.cu`
  copy of runtime `ROTATING_BUFFER_COUNT` from invalid `constexpr` to `const`.
  This dependency-only diff is retained in the initial campaign artifacts.

## Unresolved CDNA4 atomic-publication qualification

`streamk-arrival` and `tree-atomic-or` pass their Default `default` and `high`
clean controls, but detect 0/8 reviewed faults at both settings. Their `higher`
and `max` clean controls exit 89: numerical results and coverage pass, while
ConSan reports conflicts on LDS accesses protected by the source atomic
publication protocol. Both Default cells are therefore red. Their independent
SuperCollider sweeps pass clean controls but remain below the bar at 0/8.

The existing full publication journal is gated to RDNA4; CDNA4 uses the older
association model. A port was investigated using the exact pristine atomic
sites, native profiler allowlists, isolated hooks, a minimal probe family,
and TheRock rocGDB. The experiment found a scalar-address replay defect: an
instrumentation prelude borrowed the original wide atomic's address SGPRs and
replayed the guest before restoring them. An emulator regression reproduced
that defect, and restoring the scalar operands removed the physical aperture
fault. The experimental full Stream-K clean run then passed, but its fault
sweep remained 0/8 throughout. The full tree numerical result also passed,
while hip-moi's own metadata oracle failed; diagnostic output identifies a
producer missing from its acquired-epoch metadata. Its early-success retry
logic is a plausible timing-sensitive cause, not a proven repair.

**The publication port is not included in the working changes or qualifying
ledger evidence.** Full regression testing exposed 17 CPU/emulator failures and 13 physical
failures, including four RDNA4 emulator clean regressions, incomplete journal
evidence and a bounded journal overflow. An ordered manual tree probe also produced a false
positive. These results override the earlier passing host units and sparse
positive probes. The experiment is preserved as
`experimental-cdna4-publication-port.patch`, with immutable hooks, debugger
logs, code objects, the scalar replay regression and failed test XML under the
qualification artifact root. Its `*-publication-v2` fault batches are excluded.
No workload barrier was added and no oracle was disabled. The local
`atomic-publication-cdna4-investigation.md` records the detailed diagnosis;
[the RDNA4 investigation](ATOMIC_PUBLICATION_RDNA4_ANALYSIS.md) provides context.

## Regression checks

After withdrawing the experimental publication port, the final retained
changes pass the complete registered ConSan suite:

| Suite | Passed | Failures / skips |
| --- | ---: | ---: |
| CPU, including device tests on emulators | 3,319 | 0 / 0 |
| Physical gfx950, serialized | 436 | 0 / 0 |
| Complete hook unit binary | 285 | 0 / 0 |
| Python validation and allowlist tests | 377, plus 284 subtests | 0 / 0 |

CPU/emulator tests used 128 jobs. Physical tests used `ctest -j1` under the
shared GPU lock. Final native rocminfo and ordered-tile-handoff health checks
pass (`retained-fix-tests/final-gpu-health.log`). The final CTest inventory and logs/XML are in
`retained-fix-tests/`; `post-fix-python-full-clean-env.log` records the full
Python suite after the runner fixes. Python ran without campaign-specific
allowlist settings, which would interfere with cross-target unit fixtures.
The dated reproduction spec separately passes the runner's workload and
profile validation for 23 mutations and 46 eight-trial profiles.

The earlier `final-tests/` directory belongs to the **withdrawn experimental
port**, not the retained fixes; its 17 CPU/emulator and 13 physical failures
remain as rejection evidence. All hook binaries used by campaigns were copied
with checked hashes before rebuilding the reused original build directory.
The retained-fix regression hook has SHA-256
`f063e28a4fed126cedf7578e8019817bd767eb3be0b0a121d70ce4360b933715`.

The [dated reproduction spec](../../../tests/dbi/consan/consan_validation_faults_gfx950_20260924.json)
contains all 23 reviewed mutations and the selected eight-trial profile recipes.
It is schema-checked against the runner and retains the 6/8 requirement for
below-bar results. For the two rejected atomic-publication Default searches,
it retains `high`, the last setting with an admitted matching clean/fault
matrix; their higher/max clean failures remain recorded separately. The spec
is not a claim that all listed settings qualify. Reproduction requires the
pinned workload binaries and fresh full-workload rocprofv3 allowlists; exact
ISA selectors must be reviewed again after rebuilding a workload.

## Provenance and evidence

The source baseline is `rocm-systems`
`3722a18341ced1906143517fdea70ae823ffa068`, branch
`users/bjacob/sanitizers`, plus the recorded validation-script and documentation
changes. The hook binary has SHA-256
`ab0e7b0f614712ea8fd01238fdbc7229621da4d78577aca5838cdd84cf941299`.
Grouped retries use an isolated build with hook SHA-256
`e61d3a621f14ab8ebbd433726a091f2206b87905a695dd0526c68d3aeaf227ff`;
both qualification binaries are retained immutably before the final rebuild.
Each batch records its hook path and hash; the audit verifies the recorded
clean/fault hashes against the retained binary, rather than the mutable build
path. Experimental publication hooks are retained separately and excluded. The host GPU is an AMD Instinct MI350X, `gfx950:sramecc+:xnack-`.

ROCm comes exclusively from the active venv's TheRock `10.2.0a20260914` SDK.
No `/opt/rocm*` installation is used. PyTorch is
`2.13.0+rocm10.2.0a20260914`, Triton `3.8.0`, and IREE compiler/runtime `3.11.0`.
Tensile uses the locally built native client and matching YAML-enabled host
library, built against that SDK.

| Dependency | Revision |
| --- | --- |
| hip-moi | `f15bf1b96124445e1a8c3b57cbac7d56b9d8d0a9` |
| rocjitsu-test-corpus | `124bc1d14c240831ccc6c4e4b85772f6b8afef06` plus the build correction above |
| iree-test-suites | `49f46d6d4370e5aa0a6367751474e20c6c4e95c0` |
| rocm-libraries | `d3164197ed14cc6dda68f82b949fadf054735b54` |
| roofline/iree-regression-models Qwen inputs | `58e284509e0adab1d387ad1931e248d114c2cd22` |

The fetched public hip-moi checkout contains the RDNA4 Jakub fixture, but no
CDNA4 Jakub source or build target. Fetching the complete public history
found only the RDNA4 reference-bridge addition (`c015a45`); the active checkout
and workload builds stayed pinned. A follow-up search also checked the local
`rocjitsu-test-corpus` tree, its build/manifests and documentation, and all 28
fetched remote branch tips. No Jakub fixture was found there; the only Jakub
text matches were example developer paths. The validation catalog's
`_jakub_override` and CDNA registry explicitly resolve this row to
`hip-moi-build-gfx950-tests/tests/hip_moi_reference_cdna4_jakub_matmul`, not a
corpus executable. Search evidence is retained in `jakub-corpus-search.json`.
That prerequisite gap remains explicit; no RDNA4 binary or substitute workload
qualifies the CDNA4 cell.

Local artifact roots:

- `/home/benjacob/consan-validation-gfx950-20260924/`: original exact commands,
  native CSV traces, 26 generated allowlists, clean runs and coverage,
  inventories, initial fault reviews/trials, source pins and dependency diff.
- `/home/benjacob/consan-cdna4-qualification-20260924/`: prospective eight-trial
  specs and immutable snapshots; per-workload/configuration clean and fault
  results; pristine code objects and ISA reviews; timing-fix rechecks and tests.
  `native-allowlist-provenance.json` verifies exact equality against native
  rocprofv3 traces for all 26 selected fixtures, retaining hashes of each CSV
  and allowlist. It also retains the superseded single-wave mode generation.
  Tensile uses the direct native-client trace. All campaign hook binaries
  are preserved with checked hashes under `retained-hooks/`.
  `qualification-audit.json` checks completed batches for identical clean/fault
  instrumentation controls and allowlist hashes, matching workload commands
  and inputs, exact-one installation,
  complete/reached eight-trial matrices, matching clean coverage and GPU
  health. It retains effective preset/bank logs, spec/result hashes and oracle
  outcomes. Command comparison ignores diagnostic labels, Tensile artifact
  output directories, and the Sharktank fault-only option to record a numerical failure instead of aborting; clean
  runs still require the oracle to pass. Raw logs remain authoritative for launch-identity representation
  and diagnostic attribution; neither a 64-bit hardware-entry fingerprint nor
  a literal fallback implies injective global launch identity.

<!-- BEGIN PHYSICAL SWEEP RESULTS -->
## Completed physical sweep batches

Counts below are detections/admitted trials. Each completed batch also has
its matching clean control and health/coverage audit. Rejected attempts do
not contribute detections or qualify a setting. `v2` denotes the isolated
grouped-barrier hook; top-k `v3` additionally uses the 120-second bound.
`mode-v2` uses the corrected multi-wave mode fixture and fresh profiler allowlist.
Old single-wave mode batches and experimental publication-hook batches are excluded.
The two atomic-publication Default searches stop qualification at rejected
higher/max clean controls; these are red in the ledger.

| Workload | Default presets | SuperCollider settings |
| --- | --- | --- |
| `clip-bf16` | default: 2/8; high: 7/8 qualified | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `d128-block` | default: rejected; default v2: 0/8; high v2: 1/8; higher v2: 8/8 qualified | delay-zero: rejected; delay-zero v2: 0/8; sleep=1 v2: 1/8; sleep=15 v2: 6/8 qualified |
| `d128-pressure` | default v2: 0/8; high v2: 0/8; higher v2: 8/8 qualified | delay-zero v2: 0/8; sleep=1 v2: 0/8; sleep=15 v2: 0/8 |
| `hip-matmul-m128-n128-k128` | default: 0/8; high: 3/8; higher: 8/8 qualified | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `hip-streamk-simple-m256-n256-k256` | default: 1/8; high: 7/8 qualified | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `hip-streamk-two-tile-m256-n256-k256` | default: 2/8; high: 8/8 qualified | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `hipkittens-bf16fp32-16x32` | default: 0/8; high: 0/8; higher: 0/8; max: 0/8 | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `hipkittens-fp8fp32-4wave` | default: 0/8; high: 1/8; higher: 8/8 qualified | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `hipkittens-mxfp8-4wave` | default: 0/8; high: 1/8; higher: 8/8 qualified | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `pytorch-norm-softmax` | default: 0/8; high: 0/8; higher: 0/8; max: 0/8 | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `pytorch-torch-histc` | default: 0/8; high: 0/8; higher: 0/8; max: 0/8 | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `pytorch-torch-mode` | default: excluded: single-wave fixture; high: excluded: single-wave fixture; higher: excluded: single-wave fixture; max: excluded: single-wave fixture; default-mode-v2: 0/8; high-mode-v2: 1/8; higher-mode-v2: 8/8 qualified | delay-zero: excluded: single-wave fixture; sleep=1: excluded: single-wave fixture; sleep=15: excluded: single-wave fixture; sleep=1-mode-v2: 0/8; sleep=15-mode-v2: 1/8; delay-zero-mode-v2: 0/8 |
| `pytorch-torch-sort` | default: 0/8; high: 4/8; higher: 7/8 qualified | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `pytorch-torch-topk` | default: rejected; default v2: rejected; default v3: 0/8; high v3: 2/8; higher v3: 8/8 qualified | delay-zero: rejected; delay-zero v2: rejected; delay-zero v3: 0/8; sleep=1 v3: 0/8; sleep=15 v3: 0/8 |
| `qwen-prefill` | default: 1/8; high: 2/8; higher: 8/8 qualified | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `rocblas-sgemm-square-64` | default: 0/8; high: 0/8; higher: 8/8 qualified | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `streamk-arrival` | default: 0/8; high: 0/8; default-publication-v2: excluded: experimental publication hook withdrawn; high-publication-v2: excluded: experimental publication hook withdrawn; higher-publication-v2: excluded: experimental publication hook withdrawn; max-publication-v2: excluded: experimental publication hook withdrawn | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `tensile-gfx950-lds-positive` | default: 0/8; high: 0/8; higher: 0/8; max: 0/8 | delay-zero: 8/8 qualified |
| `tp1-decode-combined` | default: 0/8; high: 0/8; higher: 8/8 qualified | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `tp1-prefill` | default: 0/8; high: 1/8; higher: 8/8 qualified | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `tp2-family` | default: 0/8; high: 3/8; higher: 8/8 qualified | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `tree-atomic-or` | default: 0/8; high: 0/8 | delay-zero: 0/8; sleep=1: 0/8; sleep=15: 0/8 |
| `wmma-attention` | default v2: 0/8; high v2: 0/8; higher v2: 8/8 qualified | delay-zero v2: 0/8; sleep=1 v2: 0/8; sleep=15 v2: 8/8 qualified |
<!-- END PHYSICAL SWEEP RESULTS -->
