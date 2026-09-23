# ConSan benchmarking

This is the reproducible runbook for measuring ConSan. Benchmarking is separate
from correctness qualification in [`validation/`](../validation/VALIDATION.md):
a benchmark result cannot promote a validation cell, and validation status does
not carry performance claims.

The active performance targets are [gfx950](STATUS_GFX950.md) and
[gfx1201](STATUS_GFX1201.md). Benchmark ledgers name concrete targets because
performance does not generalize across every product in an architecture family.
The other `STATUS_*.md` files are placeholders, not required targets.
The prepared MI350X campaign invocation and evidence layout are in
[`GFX950.md`](GFX950.md).

The most recently rerun campaigns in these ledgers provide the authoritative
end-to-end performance measurements for their recorded configurations. Results
apply to the measured source revision; they do not automatically qualify later
changes. Interpret preset labels using each artifact's resolved controls and
source revision; a label reused after a policy change does not imply identical
sampling. Machine-local artifact paths identify the original evidence location
and are not portable download links.

## Measurement contract

The runner executes a native baseline, Default Mode with the default preset,
Default Mode with `RJ_CONSAN_PRESET=high`, and SuperCollider on one physical
GPU. GPU work must be serialized: do not run two cells concurrently, and use
`-j1` if the surrounding test driver has a job-count option. Each cell runs in a
fresh process and performs the same bounded, synchronized operation twice.
Every admitted cell must:

- use exactly the same operation and inputs natively and in every mode;
- pass an independent numerical or output oracle;
- use the generated exact-name kernel allowlist for every instrumented run;
- pass ConSan's static site-instrumentation audit; and
- preserve enough provenance to reproduce the executable, software stack,
  target, workload configuration, and selected kernels.

The long-term suite-latency target is 30 minutes on a warm machine with one
supported GPU and at least 16 GiB of device memory. This is currently an
aspiration, not an admission threshold: do not distort initial workloads merely
to meet it. The runner records actual suite time so later corpus trimming can be
evidence-based.

The core consists of production-shaped Aorta operations. Small generator- or
library-specific GEMMs may supplement it when no portable end-to-end Aorta path
can prove that ISA source. Such probes must obey the same baseline, oracle,
allowlist, audit, and provenance rules and must be labelled as kernel workloads,
not end-to-end models.

## Prerequisites and invocation

Keep Aorta external to RocJITsu and point the runner at a named checkout. Use a
ROCm Python environment that contains the Aorta dependencies, a ConSan hook
built from the RocJITsu revision being measured, and `rocprofv3` from the same
ROCm distribution as the runtime under test. Generator-specific cells may also
need Gluon and `hipblaslt-bench` from that distribution.

The runner's required inputs can be supplied by options or their corresponding
environment variables. A representative invocation from the RocJITsu source
tree is:

```sh
python3 tests/dbi/consan/consan_benchmark.py \
  --target gfx1201 \
  --aorta-dir "$HOME/workspace/aorta" \
  --python /path/to/rocm-python \
  --hook /path/to/librocjitsu_dbi_hooks.so \
  --rocprofv3 /path/to/rocprofv3 \
  --hipblaslt-bench /path/to/hipblaslt-bench \
  --output-dir /tmp/consan-benchmark-gfx1201 \
  --status docs/consan/benchmark/STATUS_GFX1201.md \
  --timeout 600
```

`CONSAN_BENCHMARK_AORTA_DIR`, `CONSAN_BENCHMARK_PYTHON`,
`CONSAN_BENCHMARK_HOOK`, `CONSAN_BENCHMARK_ROCPROFV3`, and
`CONSAN_BENCHMARK_HIPBLASLT_BENCH` are the environment equivalents. Omit the
hipBLASLt argument only when the selected target corpus has no hipBLASLt cell.
Use `--workload ID` repeatedly for a focused run. Use `--resume` after an
interruption; a cell is reused only when its complete input fingerprint matches.

Before accepting results on a new host, confirm that all binaries and Python
libraries resolve to the intended local ROCm installation and that the reported
GPU target matches `--target`. Do not substitute an emulator for the physical
benchmark GPU.

## Exact two-pass allowlist workflow

Allowlist discovery is a native profiling pass, not a textual scan of binaries
and not a hand-written guess. For each workload the runner must perform the
same two-pass workflow documented in
[`USAGE.md`](../USAGE.md#generate-and-use-a-kernel-allowlist):

1. Run the exact workload natively under `rocprofv3 --kernel-trace`.
2. Convert the resulting trace directory with
   `rocjitsu_consan_allowlist.py`.
3. Reject an empty or malformed inventory.
4. Pass the generated file unchanged to every ConSan mode through
   `RJ_CONSAN_KERNEL_ALLOWLIST_FILE`.

In expanded form, the discovery operation is (replace `./workload its-arguments`
with the exact native command):

```sh
rocprofv3 --kernel-trace --output-format csv \
  --output-directory "$artifact_dir/kernel-profile" -- \
  ./workload its-arguments

rocjitsu_consan_allowlist.py \
  --output "$artifact_dir/kernel-allowlist.txt" \
  "$artifact_dir/kernel-profile"
```

The benchmark runner automates these commands and records both the trace-derived
inventory and final allowlist. A framework profiler may provide supplementary
diagnostics, but it is not the canonical allowlist source. If ordinary lazy
loading causes the measured operation to dispatch a kernel absent from the
discovery pass, fix the workload's deterministic setup or repeat discovery;
never silently append a guessed name.

## Per-workload execution order

For each workload, use this order:

1. `rocprofv3` native kernel-inventory pass and allowlist conversion;
2. two native timing processes, each containing Run1 and Run2, used immediately
   to establish a separate median for each run ordinal;
3. Default Mode (`RJ_CONSAN_PRESET=default`), Default Mode
   (`RJ_CONSAN_PRESET=high`), and SuperCollider, updating the target status row
   after each completed configuration; and
4. one final native timing sample used only as a post-validation drift check.

The final sample does not silently redefine the denominator after the modes
have run. Compare it with the initial native median and flag material drift in
the machine-readable result. If drift makes the measurements unreliable,
invalidate and rerun the workload rather than selecting the more favorable
baseline.

## Startup, runtime, and coverage

PyTorch and other lazy stacks may load GPU code during the measured operation,
so API-level timestamps alone cannot separate instrumentation from execution.
The ConSan hook maintains a monotonic process clock covering the union of wall
time intervals spent preparing selected replacement code objects and loading
and binding their replacements. Concurrent transforms are therefore not
double-counted.

Each target status row begins with matching uninstrumented Startup and Run
values, then reports two values per benchmark variant, in the order Default
Mode, Default Mode (`high`), and SuperCollider. The two Default Mode variants
explicitly select `RJ_CONSAN_PRESET=default` and `RJ_CONSAN_PRESET=high`,
respectively; `high` is a preset, not a separate sanitizer mode. Inherited
expert sampling controls are cleared for all variants.

The default and high presets use workgroup/LDS-cell strides of 256/256 and
16/16, respectively. SuperCollider runs with the preset unset.

Each pair contains:

- **Startup (seconds)**: instrumentation time plus the measured first operation.
  PyTorch/Gluon measure the synchronized host operation, including load/bind,
  first-dispatch warm-up, and selected evidence checkpoints. hipBLASLt instead
  exposes GPU event times: its Startup includes instrumentation plus the first
  device-timed batch, but excludes host client setup, loading outside the hook's
  timer, and host numerical validation. It is not an end-to-end cold-start time;
  capturing a first-batch host completion timestamp remains a runner improvement.
- **Run (seconds and ratio)**: the absolute latency of the second identical
  synchronized operation, followed by that latency divided by the median
  second-operation time from the two initial native processes.

The first operation is both a correctness-checked warm-up and part of the
reported Startup latency. For the synchronized host measurements, its host
evidence analysis is included there as well. For ConSan, the harness uses
`RJ_CONSAN_EPOCH_ANALYSIS=manual` and keeps an explicit analysis window open
from immediately before Run1 through its final synchronization. This includes
every internal synchronized epoch belonging to Run1. Opaque subprocess
payloads that cannot call the window API select their first automatic report
epoch with `nth:1` and use a direct device timer for steady execution. Run2
remains fully instrumented, but its quiescent report epochs are validated and
recycled without snapshot, decode, analysis, or rendering. The resulting Run
value measures the sustainable device instrumentation path rather than
repeatedly charging a user-selected host analysis. SuperCollider intentionally
keeps its lifetime-sticky marker and needs no selection window. The two
operations must use identical inputs and execution shape; workloads with
evolving state must restore it before each operation. The Run ratio uses only
matching second-operation ordinals.

Use a workload's direct device timer when it exposes one reliably (for example,
the hipBLASLt event time); otherwise use synchronized host time and subtract
the hook's overlapping preparation time. Status values use three significant
digits. Absolute latencies, clock snapshots, drift, coverage evidence, memory,
kernel inventory, and provenance remain in JSON artifacts rather than the
human status table.

The coverage audit is enabled by default. A mode result is admissible only when
static analysis completed and every selected supported site was patched.
Dynamic reports are still decoded at the selected Run1 checkpoints and retained
in the JSON and log artifacts, but reported workload races or exhaustion of a
bounded dynamic evidence table do not reject a performance cell: numerical
correctness and static instrumentation completeness are the benchmark gates,
while exhaustive race-analysis qualification belongs to validation.
The opt-out exists for investigation but is intentionally not advertised as a
normal benchmark path. Prior measurements found its cost negligible, so status
tables do not carry separate audit-on/off columns.

A workload with no applicable LDS or synchronization sites is **not
applicable**, not an accepted instrumentation measurement. The runner records
this outcome only when every code-object inventory completed with zero sites,
without expert limits or incomplete objects, and every exact-name allowlisted
kernel was loaded and dispatched. Missing dispatch evidence, missing audit
records, and incomplete instrumentation still fail closed. Each mode is run and
checked independently, including the numerical oracle. These cells retain raw
observations and provenance in checkpoints, but show `N/A (no applicable sites)`
in the table and have no sanitizer latency or overhead-ratio summary. This
distinction matters for small MoE shapes that select register-only wave GEMVs.

## Artifacts and result checkpointing

Never treat terminal output as the only record. The output directory contains a
log and fingerprinted JSON checkpoint for every inventory, native, and mode
cell, the trace-derived allowlist, and a final `summary.json`. A checkpoint is
written only after the process exits successfully, its oracle passes, and (for
an instrumented audited cell) coverage is accepted or inapplicability is proven
as described above. Inapplicable checkpoints explicitly retain
`coverage.accepted=false` and `coverage.applicable=false`. Keep partial logs for
timeouts and failures.

Each checkpoint retains the RocJITsu commit and dirty state as provenance.
Its reuse fingerprint covers the actual source tree (including staged,
unstaged, and untracked code), Aorta identity for Aorta payloads, hook and runner
hashes, Python support/payload hashes, installed package versions and RECORD
hashes, runtime paths, target, controlled environment, exact command, allowlist,
profiler, and the applicable external benchmark executable. Markdown-only
ledger updates and committing unchanged code do not change the executable
fingerprint. Code changes do. Fingerprints are per workload, so changing the
selected subset cannot itself invalidate a completed cell. Old checkpoints
without these identities must be rerun; they are not migrated by assumption.
The status file is a concise projection of accepted artifacts, not an
independent source of truth.

## Current corpus

The portable Aorta core is deliberately small and ISA-diverse:

| Workload | Source | Distinct behavior |
| --- | --- | --- |
| Synthetic dense prefill, 32-token prompt | Aorta PyTorch eager inference | Prefill attention, normalization, embedding, reductions, and dense GEMMs. |
| Synthetic dense decode, one continuous-batch tick | Aorta PyTorch eager inference | Decode-shaped small-M operations and launch-heavy execution. This is synthetic decode, not Qwen decode. |
| Synthetic four-expert top-1 MoE prefill, 16-token prompt | Aorta PyTorch eager inference | Routing, masks, indexed gather/write, GLU experts, and sparse-expert shapes. |
| Verified shared-memory round trip | Gluon JIT kernel probe | A portable, directly attributable Gluon-generated code object with genuine LDS load/store sites. |
| Verified FP16 GEMM | local hipBLASLt/Tensile kernel probe | A dispatched library-selected Tensile solution, captured by exact kernel name. |

The last two are kernel workloads, not end-to-end Aorta models. They close a
generator-provenance gap without pretending that PyTorch's runtime-selected GEMM
provider proves Gluon or Tensile coverage.

The Aorta adapter checks every model forward against a CPU FP32 copy of the
actual BF16 weights and identical input tokens, outside the timed interval.
It requires exact greedy-token agreement, relative L2 logit error at most 1%,
and maximum absolute error at most 1% of the reference logit's peak magnitude.
These normalized checks accommodate BF16 rounding around zero while testing
the complete output tensor; finite values and repeatable checksums alone do
not establish correctness. Native gfx950 reconnaissance found approximately
0.54% relative L2 error against FP32 for these three shapes, with all greedy
tokens equal. The bound remains fixed across native and instrumented runs.

Aorta's TokenSpeed recipes provide real Qwen and additional Gluon/Triton/Torch
paths, but the pinned stack currently accepts gfx950/gfx1250 rather than
gfx1201. TokenSpeed may therefore be included on gfx950 and omitted on gfx1201.
More generally, the target corpus is a target-specific union: record an
external-framework workload as not selected when it does not work naturally on
that GPU instead of turning a difficult port into a prerequisite.

All workloads must fit a 16 GiB GPU without multi-GPU sharding. Prefer distinct
generators and execution shapes over redundant model-size variants. Revisit the
corpus when an added workload supplies a new instruction, synchronization,
addressing, dispatch, or generator family at acceptable cost.

The gfx950 corpus adds eight TokenSpeed rows: medium-M and large-M BF16 Gluon
GEMM, Gluon attention prefill and decode, Qwen3-0.6B prefill and cached decode,
Triton FP8 block-scaled GEMM, and Gluon BF16 MoE. These use local installations,
not Aorta's container ROCm stack. Import TokenSpeed before initializing HSA:
its Proton registration can replace legacy interception if imported later.

For Qwen, pass `--qwen-model /path/to/local/Qwen3-0.6B` (or set
`CONSAN_BENCHMARK_QWEN_MODEL`) consistently across focused/resumed invocations.
Checkpoint files are content-hashed. The operations use TokenSpeed's production
BF16 model runner, MHA backend, and a 1024-token cache on one GPU with graphs
and prefix reuse disabled. Prefill processes the fixed six-token prompt
`[9707,11,358,1079,264,3465]`; cached decode processes its first generated token
at position six. Both operations include metadata preparation and greedy token
selection, excluding scheduler and HTTP overhead. Prefill rewrites every live
prefix position; decode rewrites the same next position against the unchanged
prefix on each run. Decode cache population happens during setup. A separate
Hugging Face CPU FP32 eager model computes the exact greedy-token output oracle
in each process. These are complete model operations, not serving throughput.

## Bringing up another physical target

To populate a new `STATUS_<TARGET>.md`:

1. build RocJITsu and the ConSan hook for that target from the intended commit;
2. assemble one coherent local ROCm runtime, profiler, Python environment, and
   optional library benchmark executable;
3. start with one native inventory/oracle run per candidate and omit only the
   workloads that reject the target or require substantial porting;
4. run the serialized full matrix with default coverage auditing, preserving
   artifacts and updating partial status after every mode;
5. inspect final native drift, failed/unused allowlist entries, coverage
   verdicts, target identity, and numerical results before admitting cells; and
6. commit the concise status table together with any runner, payload, test, or
   documentation changes needed to make the procedure reproducible.

When benchmarking exposes a ConSan correctness bug, stop admitting performance
numbers for that path, fix the bug, add a regression test, rerun the relevant
nonphysical tests, rebuild the hook, and restart the invalidated physical cells.
Do not normalize a failure into target-specific benchmark procedure.

No number from former validation or empirical campaigns is admitted into these
status ledgers. Those runs used different corpora and timing contracts and are
not comparable.
