# gfx1201 ConSan empirical cost and detection study

Tracking: `bd-2wsf`

Status: complete physical-gfx1201 performance and detection campaigns have
been analyzed with a checked-in provenance-verifying reader. The recommendation
below is final for this frozen study; the required single local review is
complete.

This study measures the practical cost and detection value of the four ConSan
engines on physical `gfx1201`. Its final output is a recommendation for the
ordinary supported instrumentation surface. The recommendation must follow the
data; it must not be used to choose which measurements to collect.

The existing [RDNA4 status ledger](STATUS_RDNA4.md) remains useful historical
and qualification evidence. Numbers enter this study only when their raw
artifacts satisfy the protocol below at one frozen source and workload
checkpoint. In particular, cold process ratios, warm kernel ratios, and
simulator timings are not interchangeable. Only GPU-timestamp-derived device
measurements qualify for the performance recommendation; CPU/process elapsed
time remains a secondary diagnostic in the raw artifacts.

## Decision questions

For each engine and supported semantic family, determine:

1. Does the clean workload remain correct and completely instrumented?
2. What are the transformation, first-operation, warm-kernel, code-size,
   resource, and runtime-memory costs?
3. Which reviewed, executed real-kernel faults produce an attributed ConSan
   diagnosis, and at what observed rate?
4. Does the engine provide unique useful coverage, or is it dominated by
   another engine with lower cost and lower implementation complexity?
5. Should the engine or semantic family be the default, remain opt-in, be
   narrowed to a smaller contract, or be retired from ordinary support?

A timeout, crash, trap, reset, or incorrect output is not a ConSan detection.
Those outcomes are recorded separately from an attributed diagnostic.

## Engines under study

| Engine | Ordinary mode | Intended evidence |
|---|---|---|
| SuperCollider | `supercollider` | Perturbation plus redundant-access/read-back mismatch evidence |
| MOI Record/Replay | `record-replay` | Bounded device history plus inspectable host replay |
| MOI Sampled | `sampled` | Statistical causal-window evidence under ordinary automatic defaults |
| MOI Inline Shadow | `inline-shadow` | Device-side shadowing with attributed diagnostics |

All ordinary measurements use the documented `standard-v1` behavior. A site
cap, kernel filter, manually selected register, forced spill, reduced report
size, synchronization override, or sampling-residue override disqualifies an
ordinary-support result. Experimental parameter sweeps may be reported in a
separate sensitivity section but cannot replace the default row.

## Workload corpus

The final corpus must contain at least four independent source families and
must include:

- production-shaped kernels from `~/projects/sanitizers/rdna4_matmul`;
- representative attention, WMMA, and/or atomic kernels from
  `~/projects/sanitizers/third_party/hip-moi`;
- native llama.cpp kernels with an independent CPU oracle; and
- at least one generated or library kernel family from Triton, AITER, or
  Composable Kernel.

Prefer admitting two generated/library families when they can be built and run
without weakening the protocol. Focused microbenchmarks may explain a result,
but they do not count toward the four-family real-workload requirement.

Initial candidates are deliberately broader than the final admitted set:

| Source family | Candidate workload | Primary stress | Oracle | Fault route | Status |
|---|---|---|---|---|---|
| rdna4_matmul | Production FP16/FP8 WMMA matmul | High-rate LDS staging and barriers | Exact small-shape plus sampled 4096³ host reference | Manual racy variants and automatic final-ISA mutation | FP8 admitted for Sampled; other engine failures retained |
| hip-moi | D128/WMMA attention | LDS handoff, barriers, register pressure | Existing exact reference | Automatic barrier mutation; manual source control if needed | D128 block admitted for all four engines |
| hip-moi | Stream-K or tree atomic reduction | Atomics, fences, LDS coordination | Existing exact reference | Automatic order/scope mutation | Arrival counter admitted except Inline Shadow |
| llama.cpp | Quantized matvec | Real quantized LDS accesses and barrier behavior | CPU result comparison | Automatic reviewed mutation | Admitted except Record/Replay |
| llama.cpp | RMS normalization | Small production reduction | CPU result comparison | Automatic mutation after an effective site is identified | Candidate |
| Triton/PyTorch | Compiled or split online softmax | Generated LDS/barrier code | CPU-derived tensor comparison | Automatic reviewed mutation | Compiled softmax admitted except Record/Replay |
| AITER or CK | One production attention, GEMM, reduction, or collective kernel | Library-generated high-performance code | Library reference or CPU comparison | Inventory-driven automatic mutation | Candidate |

An admitted workload must have a bounded production-relevant launch shape, an
independent correctness oracle, a stable baseline command, source and binary
identity, and at least one ConSan-visible supported site. Tiny shapes used only
for compilation or smoke tests are not performance evidence.

### Frozen corpus checkpoint

Corpus admission was completed on 2026-08-02 in
`artifacts/bd-2wsf-corpus-admission-833029ddd0-20260802`. The artifact is a
clean-qualification and inventory checkpoint, not performance evidence. Every
native baseline passed its independent oracle, and every selected workload had
a complete exact-site inventory. No kernel filter, site cap, forced register,
or coverage relaxation was used. The set satisfies the four-family requirement
with rdna4_matmul, hip-moi, native llama.cpp, and generated Triton kernels.

The source and runtime lock is:

| Component | Identity |
|---|---|
| RocJITsu / validator | `833029ddd01960957d257b7b39602b2a3d701c74` |
| ConSan hook | SHA-256 `b72109fb27a9f2edd49dac504e3469b536613dae2b34eb8b5be7961a180a6a6d` |
| rdna4_matmul | `03431b8af28be2c3cad828bec77dfe8836d4a555`; production executable SHA-256 `56ee1d612f91b746c4a0f029b7aab5cfc969be01ccb22a064806d7019ec2b608` |
| hip-moi | `0fcd57def36188500a19abaf3b6bca3a6e773034`; D128 executable SHA-256 `0751f77dfcb4d1529acdfd5c37b6f3691e19cfa7a28bbef90e81f4ada621da4e`; Stream-K executable SHA-256 `9378b6b8a6b953f336bc71cfb597c702c6aea0123c4e6c9fb01b901266d29960` |
| rocjitsu-test-corpus / llama.cpp | `5f1e7f57a8d502294e94e5b960aec71964f1b79d`; runner SHA-256 `1fbab008c93c0ee54f48406753876d42b1b94a356ea27cd5f9c385c9a8556ba4`; ggml SHA-256 values: core `7268f0290a73a5a8cef8cf0d32d32490c706bba33d3686d64ce7a0230a77cd71`, base `b0fb7be4d4f0a1b694ad94a02abcedb7754dc1a7879f0e97f0bd353caddf5f10`, CPU `9c5d52ed0842d4e9db93c4e7f030e6efddd0988037c884b67850a06a783bd8bd`, HIP `b7026216d5355efefc54be22c0ac23a96bb2008a2484b755f437d56dab94dabd` |
| PyTorch / Triton | PyTorch `2.14.0a0+rocm7.15.0a20260721`, HIP `7.15.0`, Triton `3.8.0` |

The admitted launch and oracle set is:

| Workload | Frozen launch and oracle | Code-object fingerprint | Inventory |
|---|---|---|---|
| `rdna4-matmul-fp8-production` | Exact `256x128x128` plus two sampled host-reference tiles at `4096³` | `fnv1a64:2ea902aa8104f911` | 227 LDS-access sites and 48 barrier sites in 24 sequences |
| `llama-rdna4-mul-mat-vec-q` | One token, embedding 1024, independent CPU binary output, absolute tolerance `0.02` | `fnv1a64:de2f8cc9883dcdb8` | Fault inventory: 88 barrier sites in 44 sequences; clean coverage: 462 access sites |
| `d128-block` | One 64-thread workgroup, sequence 32, D128/V128, exact and sampled host-reference contexts | `fnv1a64:cebac378f1e018d9` | 8 barrier sites in 4 sequences |
| `streamk-arrival` | One 64-thread workgroup, one `16x16` tile, two K16 partials, exact host-expected output | `fnv1a64:6a20802924b6e784` | 34 atomic sites with order and scope mutation identities |
| `pytorch-rdna4-compiled-softmax` | Full-graph compiled FP32 softmax at `[128, 256]`, CPU `allclose` oracle | `fnv1a64:46555795a0b96f12` | 6 barrier sites in 3 sequences |

Clean pair admission at that checkpoint is:

| Workload | SuperCollider | Record/Replay | Sampled | Inline Shadow |
|---|---:|---:|---:|---:|
| FP8 production matmul | rejected: 75/227 accesses | rejected: dynamic history incomplete | admitted: 227/227 accesses, 48/48 barriers | rejected: 215/227 accesses |
| llama.cpp quantized matvec | admitted: 462/462 accesses | rejected: transform status 4112 | admitted: 462/462 accesses, 88/88 barriers | admitted: 462/462 accesses, 44/44 barriers |
| hip-moi D128 attention block | admitted: 12/12 accesses | admitted: 12/12 accesses, 8/8 barriers | admitted: 12/12 accesses, 8/8 barriers | admitted: 12/12 accesses, 4/4 barriers |
| hip-moi Stream-K arrival counter | admitted: 4/4 accesses | admitted: 4/4 accesses, 15/15 atomics, 8/8 barriers, 16/16 fences | admitted: 4/4 accesses, 15/15 atomics, 8/8 barriers | rejected: 0/4 accesses and 0/15 atomics |
| PyTorch/Triton compiled softmax | admitted: 4/4 accesses | rejected: 512 clean access-conflict diagnostics | admitted: 4/4 accesses, 6/6 barriers | admitted: 4/4 accesses, 3/3 barriers |

These rejections are part of the empirical result. Performance and detection
collection must not hide them by substituting a smaller workload or a relaxed
engine configuration. A pair may enter a later campaign only after a real
implementation fix and a fresh clean-source checkpoint.

The final performance and detection campaigns re-ran every admission gate at
RocJITsu commit `2a4af5a44ad555d594b45a8d34d45f6983322e64`. Native GPU
timing support is pinned at hip-moi commit
`72a675dd671d2ef2e8592c858427d8f9a71bb300` and rocjitsu-test-corpus
commit `a4c21378d144b4f49a53e6084306c0d3c40240bb`; the rdna4_matmul
source remains `03431b8af28be2c3cad828bec77dfe8836d4a555`. Those timing
changes use the existing launch and oracle paths rather than reduced or
test-specific kernels.

The external rdna4_matmul project is
[`kuhar/sanitizer-strategy`](https://github.com/kuhar/sanitizer-strategy) at
commit `03431b8af28be2c3cad828bec77dfe8836d4a555`. The checked-in empirical
manifest pins that source identity, and the report reader rejects artifacts
whose recorded `rdna4_matmul` source head differs. A reproducible checkout and
production-only build starts with:

```sh
git clone https://github.com/kuhar/sanitizer-strategy.git rdna4_matmul
git -C rdna4_matmul checkout 03431b8af28be2c3cad828bec77dfe8836d4a555
VENV=/path/to/therock-venv PRODUCTION_ONLY=1 \
  rdna4_matmul/build_and_test.sh --help
```

`build_and_test.sh` is the executable interface between this validator and the
external project. The validator supplies the selected production kernel,
oracle shape, benchmark duration or fixed iteration count, device-timing
request, and output mode through the environment assembled in
`consan_rdna4_matmul_validation.py`; that module is the frozen definition of
the interface for this study.

The `rdna4_matmul` source-level instrumentation catalog is not the workload
binary: loading that catalog would make RocJITsu transform hundreds of
unexecuted experimental kernels. Build the project-owned production-only
target from a clean source checkpoint instead:

```sh
VENV=/path/to/therock-venv PRODUCTION_ONLY=1 \
  /path/to/rdna4_matmul/build_and_test.sh --help
export CONSAN_VALIDATION_RDNA4_MATMUL_DIR=/path/to/rdna4_matmul
export CONSAN_VALIDATION_LLAMA_BUILD_DIR=/path/to/llama/build
export CONSAN_VALIDATION_LLVM_READELF=/path/to/llvm-readelf
```

The llama setting names the CMake build root containing
`cases/llama.cpp/<runner>` and `third_party/llama.cpp/ggml/src`. When it is
set, that root is authoritative: a missing runner or shared-library closure is
a validation error, with no fallback to another build. Both pinned and
auto-discovered builds prepend the recorded ggml directories to
`LD_LIBRARY_PATH`, and provenance verifies with the dynamic loader that the
mapped ggml files are the files it hashes.

This target contains the ordinary uninstrumented FP16 and FP8 production
kernels and the same host oracles. RocJITsu still analyzes every kernel in the
loaded production object; no RocJITsu kernel filter or coverage relaxation is
used. Clean and cold rows run the exact small-shape oracle plus a two-tile
sampled host oracle at 4096³ without the benchmark loop. Warm rows use the
harness's device events, fixed warmups, and at least 250 ms of timed dispatches.

## Frozen provenance

Every campaign records, at minimum:

- the RocJITsu commit and dirty-tree state;
- the hook binary SHA-256 and build configuration;
- every workload repository commit, submodule state, and local patch;
- the exact code-object hash and kernel names;
- the TheRock SDK, compiler, HIP runtime, kernel driver, and firmware identity;
- framework and kernel-generator package versions and module paths for generated
  workloads;
- the selected KFD node, PCI identity, `gfx1201` target, and visible-device
  environment;
- all environment settings and exact commands;
- inputs, launch dimensions, warmups, timing iterations, process count, and
  randomization seed; and
- GPU temperature, clocks, and competing-process checks when those readings
  are available without changing the machine configuration.

Optional machine observations record explicit unavailable or failed-probe
states. Required workload identities, including PyTorch and Triton package
identity and the llama dynamic-loader closure, fail the campaign when they
cannot be recorded. Dynamic observations are captured once when the campaign
is created and preserved by `--resume`; stable source, binary, machine, and
toolchain identity fields must still match on every resumed invocation.

The checkpoint artifact must be read with the validator commit named in its
lock table. A provenance-schema change requires a fresh artifact root and a
new clean-source checkpoint; the validator rejects an attempt to resume an old
schema with a diagnostic that directs the operator to create a new root.

Use a new artifact root after any source, hook, binary, input, methodology, or
failed-preparation change. Never merge samples from different checkpoints into
one summary row.

### Artifact retention and reproduction

The eleven final artifact roots named by `consan_empirical_gfx1201.json` and
listed in the generated results file are retained under
`/home/jakub/rocjitsu/sanitizers/artifacts` in the study workspace. Together
they currently occupy 1.2 GiB. They contain the raw rows, code objects,
provenance, campaign summaries, and device-health observations required for a
fresh `--check`; the checked-in generated report is not a substitute for those
raw artifacts.

These roots are workspace evidence, not repository contents, and no durable
shared archive exists at this checkpoint. They must remain intact through
landing and any decision based on this study. Before that workspace is cleaned,
all eleven roots must be copied without renaming to a durable project-selected
store. Until such a copy exists, another checkout can reproduce the procedure
from the pinned sources but cannot independently re-verify the exact retained
samples. Any later campaign uses a new root and updates the manifest instead of
overwriting these results.

## Clean admission gate

Before timing or fault trials, every workload/engine pair must:

1. pass the independent workload oracle;
2. report an applicable code object and the expected kernel identity;
3. patch every admitted supported site with no hidden coverage limiter;
4. report static, dynamic, and analytical completeness;
5. emit no unexpected ConSan diagnostic or overflow; and
6. terminate within the predeclared bound while leaving the device healthy.

Pairs that fail admission remain explicit results. They are not dropped from
the denominator and are not timed as if they represented the engine's ordinary
behavior.

## Performance protocol

Performance collection never injects a fault. Native and instrumented rows use
the same workload binary, input, launch shape, and correctness oracle.

Record three distinct costs where the workload supports them:

1. **Transformation and cold first-operation cost:** process start through the
   first accepted operation, with transformation time and workload execution
   time separated when the hook exposes both.
2. **Warm end-to-end cost:** host-observed latency after transformation and
   workload warmup, including required ConSan host processing.
3. **Warm device-kernel cost:** device timing for the workload kernel sequence,
   excluding process startup and one-time transformation.

Only the third cost is a qualifying performance metric. Transformation,
first-operation, host, and process elapsed measurements remain useful
operational diagnostics, but they do not enter overhead comparisons or the
recommendation.

If bounded detector state makes repeated dispatches invalid, collect one
device-timed operation per fresh process and report it as an explicit
single-dispatch exception. Do not generalize that ratio to steady-state
throughput.

The default final collection uses at least ten independent process rounds. A
round brackets the instrumented modes with native baseline-before and
baseline-after samples; engine order is randomized from a recorded seed. A
warm-capable workload has separate cold and warm schedules in each round. Each
warm sample performs an untimed correctness-preserving warmup and enough timed
iterations to exceed 250 ms in aggregate, unless the workload's state contract
requires fresh processes. The native calibration fixes one iteration count for
all engines in that campaign.

Qualifying timing comes from native HIP events in the project harness, native
HIP events around the exact GTest launch path, or PyTorch HIP events. The
compiled-softmax calibration discards the compile transient, retains 100 GPU
event samples, and sizes the common timed window from the fastest retained
sample with 25% headroom. Stream-K is the sole exception to the 250 ms
aggregate: repeated launches exceed ordinary ConSan dynamic-evidence capacity,
so each row contains one real dispatch with a recorded 0.5 ms minimum and no
warmup.

A project-owned self-timed harness may satisfy the same 250 ms requirement
internally. Its native calibration reports the exact launch count and measured
aggregate; every instrumented row reuses that count and rejects an aggregate
below 250 ms. The campaign records one aggregate device estimate per round,
does not multiply the harness's own iteration loop, and uses separate
oracle-only processes for the cold metric.

Baseline drift is evaluated separately for every metric using the absolute
difference divided by the mean of the two bracketing baselines. A metric sample
is rejected when either baseline fails its oracle or drift exceeds 5 percent;
other stable metrics from the same round remain usable. The campaign passes
only after every reported engine/metric cell has at least ten accepted rounds.

For each metric, retain every raw sample. Report the median paired slowdown,
interquartile range, and a 95-percent bootstrap confidence interval. The paired
baseline for a mode is interpolated from the two bracketing native samples by
execution position. Also report absolute latency or throughput; a ratio alone
can make a tiny kernel look disproportionately important.

Collect these structural costs beside timing:

- original and patched code-object bytes and ratio;
- transformation latency and patch counts by semantic family;
- original and patched SGPR, VGPR, AccVGPR, LDS, and private-segment metadata;
- spills, relay/island counts, and unsupported or failed placements; and
- instrumentation-owned host and device allocations, including high-water
  marks and overflow state.

The checked-in runner exposes this contract through `study`. For example:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target gfx1201 study \
  --workload pytorch-rdna4-compiled-softmax --profile all \
  --rounds 10 --seed 20260802 \
  --timeout 120 \
  --artifact-root "$CONSAN_ARTIFACT_ROOT"
```

The artifact root contains frozen provenance, clean admission rows with
original and patched code objects, an uninstrumented timing calibration,
deterministically scheduled cold and warm round directories, raw host/device
samples, and an atomically updated `campaign.json`. Passing `--resume` reuses
complete matching rows; an interrupted row is preserved with an
`incomplete-N` suffix before it is retried. A changed source, binary, command,
or campaign configuration requires a new artifact root.

With `--profile all`, admission is evaluated independently for every engine.
The campaign retains each rejection and times every admitted pair; a rejected
pair is never allowed to suppress evidence for the other engines or enter the
timing schedule. The campaign summary separates requested, timed, and rejected
profiles.

Campaigns are collected per workload today. Performance collection is tracked
by `bd-2wsf.3`, fault collection by `bd-2wsf.4`, and the cross-workload reader
that verifies shared provenance and renders the recommendation tables by
`bd-2wsf.6`; manual transcription is not an accepted final reporting path.

## Detection protocol

The fault corpus combines two mechanisms:

- exact final-ISA mutations selected by the existing ConSan inventory and
  fault runner; and
- source-level manual faults in directly owned HIP kernels when they provide a
  clearer, independently reviewable race and oracle.

Every fault has a precommitted identity, semantic class, applicability by
engine, expected independent-oracle effect, and allowed diagnostic outcome. An
automatic mutation records the code-object hash, kernel, instruction offset,
original bytes, replacement bytes, and occurrence. A manual fault records the
source diff and disassembled binary delta.

A fault becomes an admitted row only after an instrumentation-independent
argument establishes the missing ordering and both conflicting endpoints are
shown to execute. That evidence may be a direct source/dataflow proof backed by
the final ISA, an access witness from the owned HIP harness, or a native
fault-only control that manifests the expected oracle failure. An oracle
failure is not required in every trial: detecting a real race before it changes
the numeric result is useful. Oracle-manifestation rate is therefore reported
separately from detection rate. Unreached mutations do not count as detector
misses; reached but schedule-masked faults remain an explicitly labeled class.

The final fault set contains at least one admitted fault from every admitted
source family. Across the corpus it must exercise uncoordinated LDS accesses,
barrier ordering, and every atomic or fence family proposed for ordinary
support. Prefer two independent workloads for a semantic family before making
a broad keep or retire recommendation.

Use fresh contained processes. Report attempted, mutation-admitted, and reached
counts separately. A reached trial has complete installation evidence plus a
detector-owned diagnostic, an independent-oracle manifestation, a runtime
access counter, or a reviewed final-ISA proof that the selected instruction is
unconditionally executed by the completed workload. Process completion alone
is not a reach witness; timeouts without a direct witness do not enter the rate
denominator. A pilot may
use ten executions, but a final detection estimate uses at least 30 independent reached trials per applicable
fault/engine pair; report hits/trials and a Wilson 95-percent confidence
interval. Extend a predeclared row to 100 trials when its 30-trial interval is
too wide to distinguish the recommendation categories. A deterministic claim
requires every final trial to diagnose the fault. Record the seed and ordinary
automatic sampling parameters for every trial.

Each trial records these outcomes separately:

| Outcome | Meaning |
|---|---|
| Fault reached | The reviewed mutation executed |
| Oracle failed | The workload independently exposed incorrect behavior |
| ConSan diagnosis | An attributed diagnostic matched the reviewed fault |
| Contained failure | The process failed without an attributed diagnosis |
| Timeout | The process exceeded its fixed bound |
| Device health | The post-trial health probe passed or failed |

Detection yield is computed only over admitted, reached trials. Oracle
manifestation uses the same denominator and is reported independently. False
positives are measured by the clean repeated-process campaign and reported next
to detection yield.

## Implementation-complexity protocol

Complexity is not reduced to one source-line count. For each engine, record:

- engine-specific production and test code, translation units, and public
  settings;
- shared versus target-specific analysis and emission paths;
- instrumentation phases, relocation/relay requirements, register allocation,
  spilling, and descriptor changes;
- persistent and per-dispatch ABI state, report formats, replay/shadow models,
  and memory bounds;
- distinct overflow, incompleteness, containment, and cleanup paths;
- supported semantic families and typed exclusions;
- target-porting and regression-test obligations; and
- known failure modes from the admitted workloads.

Mechanical counts are retained as raw data. Qualitative maintenance risk is
reported as low, medium, or high with a source-grounded explanation; it is not
summed into a synthetic score.

## Results

No cell is populated from recollection. The complete generated tables, sample
counts, confidence intervals, structural measurements, mechanical complexity
inventory, and artifact names are in
[GFX1201_EMPIRICAL_RESULTS.md](GFX1201_EMPIRICAL_RESULTS.md). The checked-in
reader discovers the exact artifacts named by `consan_empirical_gfx1201.json`,
verifies their shared source, hook, runtime, toolchain, selected-device, and
fault-spec provenance, and then renders that file. Regenerate and verify it
from the workspace root with:

```sh
python3 rocm-systems/emulation/rocjitsu/tests/dbi/consan/consan_empirical_report.py \
  --artifact-base artifacts \
  --output rocm-systems/emulation/rocjitsu/docs/consan/GFX1201_EMPIRICAL_RESULTS.md
python3 rocm-systems/emulation/rocjitsu/tests/dbi/consan/consan_empirical_report.py \
  --artifact-base artifacts \
  --check rocm-systems/emulation/rocjitsu/docs/consan/GFX1201_EMPIRICAL_RESULTS.md
```

The final fault specification is
`consan_validation_faults_gfx1201_empirical.json`, byte-for-byte identical to
the reviewed specification materialized before all 522 final trials and now
checked in. Its SHA-256 is
`ac56252db119e9625fa9d7f90d7014a3c492c082e9e128f175d95dff2addada2`.

### Performance and resource summary

- **Sampled is the only engine admitted on all five workloads.** SuperCollider
  admitted four, Inline Shadow three, and Record/Replay two. Rejected pairs
  remain explicit evidence: FP8 undercoverage affected SuperCollider and Inline
  Shadow; Record/Replay had incomplete FP8 history, failed to transform llama,
  and emitted 512 clean access-conflict diagnoses on softmax; Inline Shadow did
  not cover the Stream-K accesses or atomics.
- **GPU overhead is workload- and engine-dependent.** Sampled slowed the
  production FP8 loop by 4.72x, llama by 1.47x, D128 by 1.04x, and generated
  softmax by 2.47x. SuperCollider measured 2.99x on llama, 0.99x on D128, and
  2.46x on softmax. Inline Shadow measured 3.87x, 1.40x, and 3.32x on those
  same workloads. Record/Replay measured 4.14x on D128.
- **Stream-K is not practical under the current ordinary implementation.** Its
  constrained single-dispatch GPU slowdowns were 506.53x for SuperCollider,
  632.83x for Sampled, and 703.19x for Record/Replay. These are valid native
  event measurements, but the one-dispatch evidence-capacity constraint means
  they are not steady-state throughput estimates. CPU/process timing is
  retained only in the raw artifacts and is excluded from these conclusions.
- **State and object costs separate the engines more clearly.** SuperCollider
  used no report allocation in admitted rows and grew modified objects by
  1.01x to 1.57x. Sampled used 0.01 MiB to 0.66 MiB of peak live report memory
  and grew objects by 1.13x to 3.83x. Inline Shadow reached 48.02 MiB of report
  memory on D128 and softmax and grew the tiny softmax object by 7.80x.
  Record/Replay used 18.66 MiB to 57.61 MiB on its two admitted workloads.
  No accepted row spilled registers, and the retained code-object metadata had
  no nonzero SGPR, VGPR, LDS, private-segment, or spill-count delta. The roughly
  4.6 GiB transform peak is a shared reserved virtual-address budget, not
  resident memory.

### Detection summary

- **Inline Shadow was deterministic for every admitted barrier case:** 90/90
  diagnoses across llama, D128, and generated softmax, with a 95% lower bound
  of 88.6% within each 30-trial row. It diagnosed all 30 softmax trials even
  though the independent output oracle passed every time.
- **Sampled traded breadth and low state for workload-dependent yield.** It
  diagnosed 21/32 llama trials (65.6%, 95% interval 48.3% to 79.6%), 8/32 D128
  trials (25.0%, 13.3% to 42.1%), and 1/32 softmax trials (3.1%, 0.6% to
  15.7%). It diagnosed 0/32 FP8 trials, where the injected barrier drop also
  produced no oracle manifestation, and 0/64 Stream-K atomic order/scope
  trials.
- **Record/Replay supplied unique synchronization evidence.** It diagnosed
  30/30 D128 barrier trials and all 60/60 Stream-K order/scope trials. None of
  the Stream-K output oracles failed, so those 60 diagnoses are evidence not
  obtainable from the workload result alone.
- **SuperCollider diagnosed none of the admitted final faults:** 0/90 barrier
  trials and 0/60 atomic order/scope trials. The strongest miss was D128, where
  the independent oracle failed in 30/30 trials while SuperCollider remained
  silent.
- Every applicable final trial was admitted and reached, every before/after
  device probe passed, and every clean admitted campaign produced zero
  unexpected diagnoses. Non-applicable engine/workload pairs were never
  counted as misses.

### Complexity summary

- The shared ConSan production directories contain 65 text/build files and
  81,531 lines. These shared analysis, placement, ABI, hook, configuration, and
  report paths dominate the maintenance floor for every MOI engine.
- SuperCollider has 7,334 lines in four dedicated emitter files. It is
  comparatively stateless but carries substantial target-instruction emission
  surface and supplied no detection value in the final corpus: **medium risk**.
- Record/Replay has only 2,065 lines in four dedicated files, but that count
  excludes its dependence on the shared MOI placement/report stack. Its device
  history, dispatch directory, replay, saturation, provenance-repair, cleanup,
  and completeness states make it **high risk** despite the smaller dedicated
  count.
- Sampled has 4,085 dedicated lines across access, sync, and host causal-window
  files. It has the broadest admission and smallest MOI reports, but its offset,
  stride, publication, and confidence requirements create a continuing
  statistical-test obligation: **medium-high risk**.
- Inline Shadow has 10,159 dedicated lines across five model, atomic, and
  emission files, the largest engine-owned surface. Combined with its report
  allocation and placement pressure, its deterministic detection comes at
  **high risk**.

## Empirical recommendation

The recommendation is to make **Sampled the default gfx1201 barrier/LDS triage
mode**, while being explicit that a single run is only a sample. It is the only
mode that cleanly admitted the entire corpus, cost 1.04x to 4.72x on the four
repeatable GPU-timed workloads, and found real barrier defects in native llama,
owned HIP, and generated Triton kernels. Qualification and user guidance should
prescribe a bounded offset sweep when confidence matters. This is not a general
atomic/fence contract: both Stream-K order/scope faults were 0/32 and the
single-dispatch Sampled slowdown was 632.83x. Ordinary Sampled support should
therefore exclude atomic order/scope claims until new evidence justifies them.

Keep **Inline Shadow as an opt-in deterministic barrier/LDS diagnostic** only
after its normal clean-coverage gate passes. Its 90/90 detection record is the
best barrier evidence in the study, including diagnoses before output
corruption. It should not claim ordinary FP8 or Stream-K support at this
checkpoint, and its 48 MiB report allocation and largest dedicated code surface
make it a poor default.

Narrow **Record/Replay to an expert opt-in synchronization mode**, with atomic
order and scope as its primary unique contract. Its 60/60 Stream-K result is
the only evidence for those two fault classes, so retiring it would lose useful
coverage. It should not be advertised as broadly supported until the FP8
history, llama transformation, and clean softmax-diagnostic failures are fixed.
Its 703.19x constrained Stream-K GPU slowdown, 4.14x D128 slowdown, and
report/ABI burden rule it out as the default.

Remove **SuperCollider from the ordinary race-detection surface** and retain it
only as an explicitly experimental perturbation probe while deciding whether
to delete it. Its lower memory and object costs do not compensate for 0/150
diagnoses, including 0/30 on an always-manifesting D128 defect. A future attempt
to restore ordinary support needs a separately reviewed fault class that
demonstrates unique detections; low overhead alone is insufficient.

### Limits of the conclusion

This is a physical-gfx1201 result for five frozen workloads and six exact-site
fault cases. It does not establish the same performance or support contract on
other targets. The barrier cases create real uncoordinated LDS execution, but
the study does not cover every global-memory race pattern. The FP8 miss is
qualified by zero oracle manifestations in all 32 trials. Sampled intervals are
specific to the standard-v1 offset sweep, and a wider workload or fault corpus
may change its yield. All implementation changes that follow from this
recommendation require ordinary target-regression qualification.

## Recommendation rule

The final recommendation uses a Pareto argument, not an unweighted average.

- **Default:** clean and complete across the required corpus, low false-positive
  risk, useful detection on important real faults, and acceptable ordinary
  overhead and maintenance burden.
- **Supported opt-in:** supplies unique useful evidence but has materially
  higher cost, nondeterminism, or operational complexity.
- **Narrowed:** useful only for specific semantic families, workload scales, or
  diagnostic workflows that can be stated as a precise supported contract.
- **Retire from ordinary support:** dominated by another engine in detection
  and cost, or unable to meet correctness/completeness requirements without
  workload-specific controls.

An expensive engine is not rejected solely for being expensive if it detects a
high-value class that no cheaper engine finds. Conversely, a cheap engine is
not retained solely for low overhead if it produces no actionable evidence.
The report must state uncertainty and preserve qualified misses rather than
silently tuning a mode until it detects the selected fault.

## Implementation follow-ups

The empirical recommendation is deliberately separated from implementation
work. The following issues are children of the existing cross-target ConSan
support epic and carry the conclusions forward without changing this frozen
evidence set:

- `bd-2sjm.7`: encode the empirical gfx1201 supported-mode and semantic-family
  policy in user-facing configuration, documentation, and qualification tests;
- `bd-2sjm.8`: make Stream-K evidence scale across repeated dispatches before
  claiming ordinary throughput or expanding its supported surface;
- `bd-2sjm.9`: fix the FP8, llama, softmax, and Stream-K admission gaps before
  broadening the corresponding engines; and
- `bd-2sjm.10`: remove SuperCollider from ordinary race detection while
  preserving an explicitly experimental path only if it remains useful.

Future target studies should reuse the paired GPU-timestamp protocol and move
target-specific policy into the manifest as those studies are scheduled. A
recurring unattended regression gate needs durable artifact storage, stable
physical runners, and bounded workload shapes before it can replace this
point-in-time qualification.

## Execution order

1. Freeze this methodology and the artifact schema (`bd-2wsf.1`).
2. Admit and pin the real-workload corpus (`bd-2wsf.2`).
3. Collect clean performance and resource evidence (`bd-2wsf.3`).
4. Collect reviewed-fault detection evidence (`bd-2wsf.4`).
5. Audit implementation complexity in parallel (`bd-2wsf.5`).
6. Publish the recommendation and create any implementation follow-ups only
   after review (`bd-2wsf.6`).

The existing `consan_validation.py` runner and artifact contracts are the
starting point. Extend them when this protocol needs additional raw metrics;
do not create a second runner that can drift from clean/fault qualification.
