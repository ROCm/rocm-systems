# ConSan Validation

This guide explains the reproducible experiment behind the per-target support
ledgers: [gfx942](STATUS_CDNA3.md), [gfx950](STATUS_CDNA4.md),
[gfx1201](STATUS_RDNA4.md), and [gfx1250](STATUS_GFX1250.md). The executable authority is
[`consan_validation.py`](../../tests/dbi/consan/consan_validation.py): it owns the
workload manifest, instrumentation profiles, commands, timeouts, knob hygiene,
coverage gates, overhead calculation, and fault-containment policy. Prefer a
script change with tests over copying another command into this document.

[USAGE.md](USAGE.md) remains the complete setting reference,
[SPILLING.md](SPILLING.md) explains ConSan's resource-policy integration, the
reusable backend is documented in
[AMDGPU register spilling](../spilling.md), and
the target-specific status files above are the published result ledgers.

The status table began as a cumulative ledger: its rows were promoted at
different frozen checkpoints and some predated today's stronger completeness
and overhead wording. These scripts define one-tip requalification; they do
not manufacture old colors. The 2026-07-16 campaign demonstrated that rule by
temporarily demoting five cells when stronger evidence contradicted them, then
restoring them only after focused fixes and exact-tip reruns.

The latest gfx1201 Record/Replay audit uses executable commit `5af82ade33` and
hook SHA-256
`0edfe1985a2ee4512b65185a6ad625fe1160e0d080be5edd160d2a12b75dd82b`.
It accepts all 19 clean rows and all 57 baseline-before, Record/Replay, and
baseline-after overhead rows.  The reviewed-fault campaign accepts 17/19 rows;
the two corrected fault-sensitivity gaps and repeated-trial evidence are
published in [STATUS_RDNA4.md](STATUS_RDNA4.md).  Unrounded ratios, raw
commands, and source/hook provenance remain in the generated artifacts.

## Workspace contract

Set one root containing the external projects and build outputs:

```sh
export CONSAN_VALIDATION_WORKSPACE_DIR=/path/to/workspace
export CONSAN_VALIDATION_TARGET=gfx1201
```

The runner expects these paths beneath that root:

```text
iree-test-suites/
iree-test-suites-build/
hip-moi/
rocjitsu-test-corpus/
rocjitsu-test-corpus-build/
rocjitsu-build/
```

For hip-moi rows, the runner requires the exact target-resolved executable
reported by `manifest` and `doctor`; it does not require a generic
`hip-moi-build/` alias. Current manifests use `hip-moi-build/` for gfx1201 and
gfx950, `hip-moi-build-gfx942-tests/` for gfx942, and
`hip-moi-build-gfx1250-tests/` for gfx1250.

For compatibility with the original gfx1201 workspace, it also recognizes
`rocjitsu-main-gpu-build/` as the rocJITsu build name. The hook must be at
`lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so` inside that build.

`iree-run-module`, `iree-benchmark-module`, and `rocminfo` are resolved from
`PATH`; IREE is not vendored or found through a machine-specific build path.
`rocminfo` is required by mixed and non-PyTorch campaigns.  A PyTorch-only
doctor or row instead verifies the target through its stronger in-process
numeric dispatch, device-architecture, and exact-hook mapping probe, so a
prebuilt-wheel setup does not need a separate `rocminfo` installation.
When the launcher Python is not the workload environment, select the exact
interpreters explicitly:

```sh
export CONSAN_VALIDATION_SHARKTANK_PYTHON=/path/to/iree-venv/bin/python
export CONSAN_VALIDATION_PYTORCH_PYTHON=/path/to/pytorch-venv/bin/python
export CONSAN_VALIDATION_TENSILE_PYTHON=/path/to/tensile-venv/bin/python
```

The Sharktank interpreter must import the IREE Python bindings plus `pytest`,
`numpy`, and `ml_dtypes`; the other two variables are needed only when those
workload families are selected.

PyTorch validation deliberately uses a separate, prebuilt-wheel interpreter;
the workspace `pytorch/` checkout is for workload discovery and source
provenance, not for building PyTorch.  Point the runner at that interpreter:

```sh
python3 -m venv "$CONSAN_VALIDATION_WORKSPACE_DIR/consan-pytorch-venv"
"$CONSAN_VALIDATION_WORKSPACE_DIR/consan-pytorch-venv/bin/python" -m pip \
  install 'torch==2.14.0.dev20260720+rocm7.1' 'numpy==2.5.1' \
  --index-url https://download.pytorch.org/whl/nightly/rocm7.1
export CONSAN_VALIDATION_PYTORCH_PYTHON="$CONSAN_VALIDATION_WORKSPACE_DIR/consan-pytorch-venv/bin/python"
```

The final export is optional when the environment uses that standard workspace
path: the runner discovers
`$CONSAN_VALIDATION_WORKSPACE_DIR/consan-pytorch-venv/bin/python`
automatically.  Set `CONSAN_VALIDATION_PYTORCH_PYTHON` only to select a
different prebuilt-wheel environment.  The doctor imports both `torch` and
`triton`, performs a numeric GPU dispatch, checks the reported gfx target, and
verifies from the process mappings that PyTorch's HSA runtime loaded the exact
ConSan hook selected from the workspace.  It records the runtime versions and
device identity.  Thus an existing but unusable environment, a wrong device,
or a wheel runtime that silently skips `HSA_TOOLS_LIB` fails before a
validation row is accepted.

Freeze the exact wheel and bundled Triton versions in the campaign artifacts.
The example index matches the current ROCm 7.1 validation stack; choose an
official prebuilt wheel compatible with the machine's runtime and target when
reproducing elsewhere.  The doctor requires this interpreter only when the
selected target manifest contains a PyTorch workload.

Official PyTorch ROCm wheels and the native llama.cpp clients use a modern HSA
runtime. After successful rocprofiler registration that runtime does not consult legacy
`HSA_TOOLS_LIB` tooling unless `HSA_TOOLS_ROCPROFILER_V1_TOOLS=1` is present.
The runner therefore sets and audits that variable automatically for
instrumented PyTorch and llama.cpp rows and removes it from baseline rows. It is classified
as `runtime-plumbing`, never as workload tuning; users should not have to
discover or manually preserve it.  The doctor tests this behavior with a real
dispatch.  Its linkage-only canary uses a deliberately nonmatching internal
kernel filter to avoid instrumenting PyTorch's large bundled kernel object;
that filter is never inherited by validation rows, whose ordinary environment
and complete-coverage gates remain unchanged.

Run the preflight before GPU work:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" doctor
```

The doctor reports every missing checkout, artifact, workload executable,
hook, and tool.  When a PyTorch workload is selected, it also performs one
small numeric GPU dispatch and verifies that the process loaded the exact
ConSan hook.  Other workload kinds remain filesystem/tooling-only preflight.

### Native llama.cpp corpus clients

The gfx1201 manifest includes `llama-rdna4-mul-mat-vec-q` and
`llama-rdna4-rms-norm` from `rocjitsu-test-corpus`. Build its kernels project
out of source at
`$CONSAN_VALIDATION_WORKSPACE_DIR/rocjitsu-test-corpus-build/kernels/gfx1201`
with `CMAKE_HIP_ARCHITECTURES=gfx1201`, `KERNEL_CORPUS_ENABLE_ALL=OFF`, and
`KERNEL_CORPUS_ENABLE_LLAMA_HIP=ON`. The resulting executables are expected at
`cases/llama.cpp/llama_cpp_mul_mat_vec_q` and
`cases/llama.cpp/llama_cpp_rms_norm` beneath that directory. The runner also
recognizes the target's pytest build artifact while migrating an existing
workspace, but the stable out-of-source path is the reproducible contract.

These clients require HIP, rocBLAS, and hipBLAS. For a source-built TheRock
workspace, configure TheRock with `THEROCK_ENABLE_BLAS=ON`, build its focused
`hipBLAS+stage` target, use `TheRock-build/dist/rocm` as `ROCM_PATH`, and add
the hipBLAS and hipBLAS-common `dist` prefixes to the corpus
`CMAKE_PREFIX_PATH`. This is build/runtime enablement, not an instrumentation
knob. The exact local paths and retained gfx1201 setup evidence are recorded in
[STATUS_RDNA4.md](STATUS_RDNA4.md).

The corpus executable's `--validate` option selects its CPU backend; it does
not validate a GPU execution. The checked-in
[`consan_llama_validation.py`](../../tests/dbi/consan/consan_llama_validation.py)
therefore launches two processes for every repetition: an instrumented GPU
process without `--validate`, and an uninstrumented CPU process with
`--validate` after scrubbing all ConSan/HSA-tool settings. Both write binary
F32 results. RMSNorm uses the corpus's compact 128-element shape and requires
a maximum absolute error of `1e-5`. Quantized matvec uses a still-compact but
fault-sensitive 1,024-element embedding: the corpus's 128-element setup smoke
schedule-masks its synchronization fault. The 1,024-element clean GPU/CPU
baseline requires a maximum absolute error of `2e-2`; its independently
reviewed exact barrier deletion exceeds that bound by roughly 40 orders of
magnitude. The wrapper emits the GPU timing and oracle result as JSON for the
normal paired-overhead machinery.

## Inspecting the executable contract

The checked-in prose does not duplicate commands or fault expectations that
can drift away from the runner. `manifest` gives the compact matrix, while
`explain` derives an audit view from the same command, environment, and fault
policy functions used during execution:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" manifest

python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" manifest --json > manifest.json

python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" explain \
  --workload all --profile all
```

The JSON contains all north-star workloads, four canonical profiles, admitted
fault families, row-declared timeouts, forbidden exploratory controls, and the maximum
GPU parallelism. Review it before starting a campaign and retain it with the
results.

For an exact, machine-readable pre-run audit, include the reviewed fault spec:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" explain \
  --workload all --profile all \
  --spec /path/to/reviewed-faults.json --json > validation-audit.json
```

For every selected workload, `validation-audit.json` contains:

- the exact argument arrays used for clean correctness and overhead, plus the
  top-level validator invocation template and process/repetition count;
- every harness-supplied setting for each profile, with inherited shell
  settings excluded and each setting classified, plus the ordinary runtime
  defaults on which the command deliberately relies;
- every admitted injection from the reviewed spec, including its exact machine
  selector and fault payload command;
- the precommitted detector and independent-oracle outcome for each flavor;
- the human-readable diagnostic requirement, deterministic or statistical
  trial count, trial overrides, policy overrides and unsets; and
- the fully merged effective setting set for every fault trial.

The human form intentionally summarizes long identities. `--json` is the
authority for exact selectors and per-trial environments. Actual `result.json`
files remain the post-run authority for what executed.

Settings are classified in a totally explicit way:

| Category | Meaning | Usability interpretation |
|---|---|---|
| `runtime-plumbing` | Locates the hook or target | Required setup, not coverage tuning |
| `instrumentation-selection` | Selects flavor or engine; can explicitly override synchronization defaults | Ordinary MOI automatically enables barrier and atomic tracking |
| `acceptance-assertion` | Turns missing records, incomplete patching, unexpected diagnostics, or overflow into failure | Cannot help a row pass |
| `workload-tuning` | Changes a workload-specific operating point | Marked `usability_exception: true` |
| `fault-injection` | Selects an exact deliberate mutation | Experiment control, not an ordinary user setting |
| `fault-containment` | Serializes destructive GPU work | Experiment safety control |

The audit also repeats the forbidden ordinary controls. A qualifying clean row
must not use a site cap, kernel filter, manually selected temporary register,
scratch register, MOI metadata register, or force-spill setting. Consequently,
a reviewer can distinguish instrumentation selection from settings that would
artificially make a difficult workload smaller.

The top-level `usability_audit` makes that conclusion queryable. It lists any
forbidden coverage-limiting control that actually appeared, workload-specific
tuning, automatic event-family defaults, explicit expert overrides, and
fault-only acceptance guards relaxed by a reviewed policy. An empty
`coverage_limiting_controls_present` is evidence that the clean profiles did
not qualify by narrowing patch coverage. An empty
`explicit_event_family_overrides` confirms that qualification relied on the
ordinary MOI defaults: both barrier and atomic tracking are enabled without a
user setting.

Without `--spec`, `explain` shows inventory templates and prints
`REVIEW_REQUIRED` instead of pretending that detector outcomes have been
chosen. Historical gfx1201 evidence can be inspected, but never executed, with:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target gfx1201 explain --workload all --profile all \
  --spec emulation/rocjitsu/tests/dbi/consan/consan_validation_faults_gfx1201_reference.json \
  --allow-reference --json > gfx1201-historical-audit.json
```

`--allow-reference` only permits read-only explanation. The `fault` subcommand
continues to reject the cumulative reference file.

The current gfx1201 manifest covers Qwen3-0.6B prefill; native
PyTorch/Inductor compact and split online-softmax clients, collision-heavy
scatter-reduce, Qwen-vocabulary top-k, causal SDPA, and histogram;
native llama.cpp quantized matvec and RMSNorm; Sharktank TP1 prefill and
decode/combined, TP2, and CLIP BF16; and the hip-moi D128, WMMA, Stream-K,
tree-atomic-OR, and Jakub workloads. The profile IDs are `supercollider`,
`record-replay`, `sampled`, and `inline-shadow`.

## Clean correctness and coverage

Choose a new artifact root. A row directory must not already exist:

```sh
export CONSAN_ARTIFACT_ROOT="$CONSAN_VALIDATION_WORKSPACE_DIR/consan-validation/run-001"

python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" run \
  --workload qwen-prefill --profile all --phase clean \
  --include-baseline --artifact-root "$CONSAN_ARTIFACT_ROOT"
```

Replace `qwen-prefill` with any ID printed by `manifest`. Each process starts
after removing inherited `HSA_TOOLS_LIB` and every `RJ_CONSAN_*` setting. The
runner then applies only the named canonical profile. Every instrumented
profile records its explicit `RJ_CONSAN_MODE` and strict completeness policy,
so provenance does not depend on selection defaults. This
prevents a coverage-limiting setting, kernel filter, explicit temporary register,
force-spill control, or stale sampling setting from silently qualifying a cell.
If a strict transform cannot be installed, the hook emits a typed
`ConSan load rejection` record and terminates that contained workload process
with exit code 92. The runner preserves the rejection fields in `result.json`;
it does not misreport the result as a missing teardown verdict or allow a HIP
client to launch through a null symbol after ignoring the HSA load error.

An instrumented clean row is accepted only when:

- the workload's independent numerical or semantic oracle passes;
- ConSan reports an applicable code object and no dynamic-incomplete result;
- every supported access, barrier, atomic, and fence site is patched;
- MOI emits no unexpected clean diagnostic and no forbidden overflow; and
- every repeated process satisfies the same coverage gate.

No clean profile currently has a workload-specific tuning exception. In
particular, Qwen Sampled relies on the ordinary `standard-v1` runtime profile:
stride 16,384 and offset zero are automatic runtime defaults, not environment
settings supplied by the validation harness. `explain --json` records those
values under `implicit_runtime_defaults`, while `workload_specific_tuning`
remains empty.

The ordinary process deadline is 30 seconds. The Qwen-vocabulary top-k row
declares 120 seconds because its complete Record/Replay transform patches
63,474 accesses and 7,100 barriers and takes about 90 seconds on this machine;
that execution bound does not change instrumentation coverage or semantics.
An explicit `--timeout` overrides the manifest only for diagnosis.

## Correct-workload overhead

Overhead is measured without fault injection:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" run \
  --workload tp1-prefill --profile all --phase overhead \
  --include-baseline --artifact-root "$CONSAN_ARTIFACT_ROOT"
```

With `--include-baseline`, the execution order is baseline-before, the selected
instrumented profiles, then baseline-after. `summary.json` reports the paired
baseline as the mean of the two baseline medians, each mode-specific slowdown,
and the maximum mode ratio used for the support-table cell:

```text
slowdown = instrumented median / paired baseline median
```

The active `gfx950` and `gfx1250` campaigns use one benchmark repetition per
process for Qwen, Sharktank, PyTorch, and Tensile.  These results qualify the
order of magnitude rather than providing statistically smoothed performance
numbers.  Sharktank still performs its untimed warmup, while CLIP and hip-moi
rows use fresh processes as declared by the manifest. Raw samples, commands,
complete controlled environment, hook hash, source revisions, and unrounded
ratios remain in the artifact tree.

## Fault inventory and reviewed specs

Fault identities contain code-object hashes, kernel names, PCs, mnemonics, and
occurrences. They are intentionally not hard-coded into the portable manifest.
Regenerate them after changing a target, compiler, workload, or binary:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" inventory \
  --workload tp1-prefill --artifact-root "$CONSAN_ARTIFACT_ROOT"
```

This runs each admitted fault family separately with
`RJ_CONSAN_FAULT_DRY_RUN=1`. Family-specific analysis is enabled, but no site
identity is selected and no mutation is applied. Inventory is a static-analysis
operation: after observing a relevant fault site and the matching code object's
coverage record, the collector deliberately stops the process instead of
waiting for the unmodified workload to execute. A timeout before that matching
record is a failure. It retains each raw log as
`command-<family>.log` and records per-family plus deduplicated aggregate
sites, synchronization sequences, and barrier destinations in
`inventory.json`. It also creates
`fault-spec.template.json` for the workload's admitted fault families.

Review the inventory before mutation. In the copied spec:

1. replace every `REPLACE_FROM_INVENTORY` value with an exact compatible
   identity;
2. precommit `detector` as `detected`, `not_detected`, or `statistical` for
   every profile; a statistical policy also sets `minimum_detections`;
3. precommit the workload `oracle` as `pass`, `fail`, or `any`;
4. add a `trials` list under a profile when a statistical campaign requires
   predeclared overrides, such as Qwen Sampled offsets; and
5. set top-level `review_required` to `false` only after that review.

A profile may instead have `"disposition": "not-applicable"` when the
inventory proves the fault family is semantically absent. This is recorded as
typed N/A and performs no mutation. Do not choose a different site or expected
outcome after observing a run.

Profile policy can also contain an `environment` object or an `unset` list.
These are retained as part of the experiment, not hidden shell tuning. The
gfx1201 policy uses `RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1` for deterministic
Inline catches and unsets the clean overflow guard only for the two accepted
fault rows whose useful diagnostic is intentionally retained in a bounded
buffer with disclosed duplicate-report drops.

The cumulative gfx1201 ledger is checked in as
`consan_validation_faults_gfx1201_reference.json`. It records the historical
selectors, deterministic qualified misses/detections, oracle outcomes, and
32-offset Qwen Sampled campaign behind the current table. That historical
fault campaign explicitly retains stride 256 in its profile policy; this does
not alter the untuned clean profile. It is reference data, not a runnable
default: its rows came from different frozen checkpoints, and a later analyzer
may deliberately reject an old association. Copy the relevant values into the
generated template only after a fresh inventory proves that every identity and
expectation still applies. The runner refuses the reference-only file itself;
exact-one mutation makes stale reviewed identities fail closed.

Example reviewed profile policy:

```json
{
  "sampled": {
    "detector": "statistical",
    "minimum_detections": 1,
    "oracle": "any",
    "trials": [
      {"RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET": "0"},
      {"RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET": "1"}
    ]
  }
}
```

## Contained fault execution

Fault runs are serialized and require an explicit destructive acknowledgement:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" fault \
  --workload tp1-prefill --profile all \
  --spec /path/to/reviewed-tp1-faults.json --fault barrier-drop \
  --artifact-root "$CONSAN_ARTIFACT_ROOT" --allow-destructive
```

The top-level runner delegates each trial to
[`consan_fault_runner.py`](../../tests/dbi/consan/consan_fault_runner.py). That runner
creates a process group, holds the global destructive-GPU lock, enforces the
deadline, captures original/patched code objects, runs a device-discovery
command plus a target-dispatch smoke before and after, and quarantines the
artifact root if health fails.  The defaults are `rocminfo` and the portable
workload smoke, with a 30-second probe deadline.  Slow software devices may
set a larger retained `--health-timeout`.  Environments where either default
cannot terminate may pass
the paired `--health-command-json` and `--smoke-command-json` overrides.  Both
exact commands are retained in every row manifest and replayed verbatim; the
smoke must still execute target code and check an independent result.  A
target runtime that emits that result but cannot complete process teardown may
use `consan_marker_smoke.py` as the smoke command.  The adapter requires an
exact caller-specified success marker and only then terminates the smoke's
process group; it does not apply to the workload or mutation command.

Promotion requires all of the following to match the reviewed spec:

- mutation accounting is exactly `requested=1 planned=1 applied=1`;
- the detector result is the precommitted deterministic value, or a
  statistical campaign reaches its precommitted minimum detection count;
- the independent oracle matches its precommitted result when not `any`;
- the process does not time out; and
- both device-health gates pass.

A timeout, trap, crash, output mismatch, or GPU reset is not a ConSan
detection. The fault runner retains these as distinct execution outcomes.
GoogleTest assertions, IREE expected-output checks, and the Sharktank wrapper
are converted into explicit oracle evidence rather than inferred from ConSan.

## Campaign discipline and porting

Run no more than four GPU jobs concurrently. Fault rows are always serialized.
Use a new artifact root after any source, hook, workload, manifest, settings,
or reviewed fault-spec change. Never merge exploratory output into an accepted
campaign.

For a new gfx architecture:

1. implement and test its instruction builder, branch forms, waits, descriptor
   growth, register allocation, and spill backend;
2. build target-native versions of every applicable workload;
3. run `doctor`, retain `manifest --json`, and pass clean rows;
4. inventory every fault family and review new target-specific specs;
5. run overhead and contained fault rows against one frozen commit; and
6. update [STATUS_RDNA4.md](STATUS_RDNA4.md) only from the generated results.

Do not copy gfx1201 coverage denominators, machine-code identities, or timing
factors to another target. A port is credible when another engineer can use
the same scripts to distinguish an ISA-backend failure, a spill/resource
failure, a workload-oracle failure, a stale fault selector, a detector miss,
and infrastructure/device loss.

## Testing the validation scripts

The orchestration and containment logic has CPU-only unit coverage:

```sh
cd emulation/rocjitsu/tests/dbi/consan
python3 -m unittest \
  test_consan_coverage_gate.py \
  test_consan_fault_runner.py \
  test_consan_run_provenance.py \
  test_consan_validation.py
```

These tests do not qualify a GPU cell. They protect the executable protocol:
environment scrubbing, profile isolation, workload commands, overhead math,
identity inventory, fault-spec validation, and workload-oracle parsing.
