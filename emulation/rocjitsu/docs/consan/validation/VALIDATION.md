# ConSan validation

This guide describes the maintained validation interfaces for ConSan. The
executable authority for external-workload campaigns is
[`consan_validation.py`](../../../tests/dbi/consan/consan_validation.py); its
manifest owns workload commands, timeouts, profile environments, correctness
oracles, and fault policy. Use `--help`, `manifest`, and `explain` instead of
copying workload-specific commands into this document.

The target ledgers record qualification state for
[CDNA3 / gfx942](STATUS_CDNA3.md), [CDNA4 / gfx950](STATUS_CDNA4.md),
[RDNA3 / gfx1100](STATUS_RDNA3.md), [RDNA4 / gfx1201](STATUS_RDNA4.md), and
[CDNA5 / gfx1250](STATUS_GFX1250.md). A ledger is not a substitute for rerunning
the gates after a relevant source, toolchain, workload, or runtime change.

## First step for revalidation: generate and apply kernel allowlists

Start every new external-workload revalidation with the rocprofv3-based
[allowlist procedure in USAGE.md](../USAGE.md#generate-and-use-a-kernel-allowlist),
as already automated by the
[benchmark runner](../benchmark/BENCHMARK.md#exact-two-pass-allowlist-workflow).
This substantially reduced benchmark workload time by avoiding instrumentation
of unrelated library kernels. Apply it before retrying historical validation
timeouts or increasing their deadlines. It may resolve many of those timeouts;
only new runs can establish which ones.

For each workload:

1. Resolve the exact native command and inputs with the runner's `explain`
   interface. Profile that workload without ConSan using
   `rocprofv3 --kernel-trace --output-format csv` from the matching ROCm stack.
   Include setup, warm-up, and every execution stage to be validated.
2. Convert the trace with `rocjitsu_consan_allowlist.py`. Reject empty or malformed
   inventories; retain the trace and generated exact-name list with the new
   campaign artifacts.
3. Apply the generated file through `RJ_CONSAN_KERNEL_ALLOWLIST_FILE` to every
   instrumented profile for that workload, with `RJ_CONSAN_KERNEL_ALLOWLIST`
   unset. Use the same list across modes. Regenerate it when the workload,
   inputs, execution paths, target, or software stack changes.
4. Verify the effective child environment and the hook's
   `ConSan kernel allowlist entry` records: selected kernels should be loaded,
   instrumented, and dispatched. If discovery missed an executed path, repeat
   discovery with that path included. Unlisted kernels are unchecked, and a
   shared helper requires all its reachable kernel entries to be selected.
   A native trace may name only one descriptor for code shared by several
   aliased kernels (as in rocPRIM). If owner filtering excludes those sites,
   inspect the pristine code object's ownership inventory and add the exact
   names of every owner of the selected shared sites to a separate expanded
   list. Retain the native list, code-object hash, added names, and reason for
   expansion. Use the expanded list for both modes and recheck coverage; do not
   infer completeness from a successful numerical oracle alone.
5. Rerun the clean qualification and applicable fault trials with the existing
   correctness, coverage, completeness, and containment checks. Record new
   results and provenance before revising a timeout row or its status.

**Runner integration:** the runner scrubs inherited `RJ_CONSAN_*` variables.
Store each generated file at `ALLOWLIST_DIR/TARGET/WORKLOAD_ID.txt` and set
`CONSAN_VALIDATION_KERNEL_ALLOWLIST_DIR=ALLOWLIST_DIR`. The runner explicitly
passes the matching file to clean, inventory, and fault children and records
its hash in provenance. Missing, empty, or malformed selected files fail the
run. Native baselines remain uninstrumented. Discovery still uses the linked
rocprofv3 procedure; the runner does not generate traces automatically.
If matching native profiling is unavailable, record that prerequisite gap.

Historical ledgers retain the results of their original configurations. Their
timeouts are priorities for revalidation with generated allowlists, not evidence
that those retries will still time out or that the rows are already resolved.

## Explicit sampling configurations

For a separate Default-engine sampling investigation, set
`CONSAN_VALIDATION_DEFAULT_PRESET=low|default|high|higher|max`. The runner passes
this as `RJ_CONSAN_PRESET` to Default clean and fault children and captures it in
the effective environment/provenance. It does not affect native baselines or
SuperCollider. With no override, the standard profile remains the contract.
Run a matching clean comparator and precommit fault expectations for a changed
preset. Label the preset explicitly in the ledger; a denser configuration's
success cannot qualify the standard profile or erase its detection misses.

Default-engine presets apply to Default only. SuperCollider already defaults to
runtime stride 1 and rejects independent workgroup/cell selectors; its replay
and perturbation evidence must be assessed separately from Default event counts.

For the September 23 RDNA4 calibration, qualify the named Default preset
against at least **6 detections in 8 admitted/reached trials** (75% observed
rate), with all clean correctness, applicable coverage, completeness and health
checks passing. Precommit the preset and threshold before running. Select independently for each workload, in ascending order:
`default`, `high`, `higher`, `max`. If `default` passes, stop; there is no
need to test `low`. Choose the lowest passing preset at or above `default`
and name it in the cell. Reuse matching recorded evidence; while lower presets
are being tested, show only the lowest already-qualified preset in the cell.
Keep unsuccessful presets and search progress in the campaign artifacts. Green in
the status table qualifies that recorded configuration. The observed rate is
not a lower confidence bound; retain counts and intervals in campaign artifacts.

SuperCollider delay matrices must use `RJ_CONSAN_SC_DELAY`. The earlier
September 23 matrices used an unrecognized variable and therefore ran at delay
zero; their artifact snapshots retain that evidence. Revalidate with the
correct variable before claiming coverage of multiple delay settings.
Set `CONSAN_VALIDATION_SC_DELAY` and `CONSAN_VALIDATION_SC_DELAY_MODE` to
apply explicit delay controls to SuperCollider clean comparators as well as
fault runs. Native and Default runs ignore these selectors. Every distinct
delay configuration contributing detections must have a passing clean
comparator with the same controls; a delay-zero clean run cannot qualify a
nonzero-delay fault result.

For workloads that intentionally use identical concurrent LDS stores, an
explicit policy investigation can set
`CONSAN_VALIDATION_ALLOW_PROVABLY_SAME_VALUE_WRITE_RACES=1`. The runner passes
`RJ_CONSAN_ALLOW_PROVABLY_SAME_VALUE_WRITE_RACES=1` to instrumented clean and
fault runs, records it in their environments, and leaves native baselines
uninstrumented. This suppresses only statically proven same-instruction,
same-value stores within a wave; cross-wave publication conflicts remain
checked. Require matching clean and fault runs under this policy and label it
in the cell. A successful numerical oracle alone does not qualify the policy.

## Validation layers

ConSan uses four complementary layers:

1. Host unit tests exercise parsing, inventory, policy, planning, native
   emission, placement, spilling, report models, final validation, hook
   configuration, and validation-script behavior.
2. Target-native simulator tests execute real code objects for all five
   supported targets through RocJITsu.
3. Physical tests execute the matching target-native fixtures on hardware and
   finish with an uninstrumented health check.
4. External-workload campaigns qualify unmodified production-shaped programs
   and reviewed fault injection.

Performance measurement is outside this contract. Use the separate
[benchmark procedure](../benchmark/BENCHMARK.md); benchmark evidence never
promotes a correctness cell.

Simulator success proves behavior under the emulator. It does not promote a
physical cell. Physical and destructive fault tests are serialized.

## Checked-in CTest gates

The `consan` label includes the maintained host and device contracts. The
`consan-device` label identifies target-native executable tests; those tests
also carry `simulator` or `physical` as applicable.

Build the test dependencies before running CTest:

```sh
cmake --build /path/to/rocjitsu-build -j16
```

Run the labeled nonphysical ConSan gate with bounded host parallelism.
Also select the ConSan unit suites by name and run the hook unit binary:
not every discovered GoogleTest case carries the `consan` label.

```sh
ctest --test-dir /path/to/rocjitsu-build \
  -L consan -LE physical --output-on-failure -j16
ctest --test-dir /path/to/rocjitsu-build \
  -R '^ConSan' -LE 'physical|consan-device' --output-on-failure -j16
/path/to/rocjitsu-build/tests/hsa_hooks_unit_test
```

Run the simulator device matrix explicitly when isolating that layer:

```sh
ctest --test-dir /path/to/rocjitsu-build \
  -L consan-device -L simulator --output-on-failure -j16
```

On a host with the matching GPU, run the physical device matrix serially:

```sh
ctest --test-dir /path/to/rocjitsu-build \
  -L consan-device -L physical --output-on-failure -j1
```

These fixtures use the production hook and transform path. They check portable
semantic outcomes—correct output, applicability, coverage, completeness,
diagnostic policy, and device health—without pinning patch counts, register
numbers, code-cave choices, or native instruction sequences.

The common matrix covers native LDS and group-FLAT access, workgroup
synchronization, selected atomic/fence ordering, multiple execution owners,
multidimensional and repeated dispatch identity, private spilling and dynamic
stacks, code-object lifecycle, graph replay, and high-pressure placement.
Target-specific fixtures cover only semantic forms admitted by the generated
[capability contract](../CAPABILITIES.md), including CDNA5 cluster operations.

CDNA3 and CDNA4 can additionally run the target-native hip-moi simulator corpus
when their build directories are supplied at configure time:

```sh
cmake -S /path/to/rocm-systems/emulation/rocjitsu \
  -B /path/to/rocjitsu-build \
  -DRJ_CONSAN_GFX942_HIP_MOI_BUILD_DIR=/path/to/hip-moi-build-gfx942-tests \
  -DRJ_CONSAN_GFX950_HIP_MOI_BUILD_DIR=/path/to/hip-moi-build-gfx950-tests

ctest --test-dir /path/to/rocjitsu-build \
  -R '^ConSanGfx(942|950)HipMoiSim\.' --output-on-failure -j1
```

## Building against TheRock

Use one coherent SDK for configuration, compilation, and execution. For an
installed TheRock development package:

```sh
# Set this to the installed development package directory in your environment.
ROCM_SDK=/path/to/site-packages/_rocm_sdk_devel

cmake -S /path/to/rocm-systems/emulation/rocjitsu \
  -B /path/to/rocjitsu-build -G Ninja \
  -DCMAKE_C_COMPILER="$ROCM_SDK/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$ROCM_SDK/llvm/bin/clang++" \
  -DROCM_PATH="$ROCM_SDK"

cmake --build /path/to/rocjitsu-build -j16

export LD_LIBRARY_PATH="$ROCM_SDK/lib:$ROCM_SDK/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ctest --test-dir /path/to/rocjitsu-build \
  -L consan -LE physical --output-on-failure -j16
ctest --test-dir /path/to/rocjitsu-build \
  -R '^ConSan' -LE 'physical|consan-device' --output-on-failure -j16
/path/to/rocjitsu-build/tests/hsa_hooks_unit_test
```

For a source-built TheRock tree, use its `dist/rocm` directory consistently.
Do not mix it with `/opt/rocm` compilers or libraries.

## External-workload runner

Set a workspace containing RocJITsu and whichever external projects the chosen
manifest rows require:

```sh
export CONSAN_VALIDATION_WORKSPACE_DIR=/path/to/workspace
export CONSAN_VALIDATION_TARGET=gfx1201
```

The runner recognizes `rocm-systems/` and `TheRock/rocm-systems/` source
layouts. Its common workspace names are:

```text
iree-test-suites/
iree-test-suites-build/
hip-moi/
rocjitsu-test-corpus/
rocjitsu-test-corpus-build/
rocjitsu-build/
```

Set `CONSAN_VALIDATION_HOOK=/absolute/path/to/librocjitsu_dbi_hooks.so` to
select a freshly built hook outside the default `rocjitsu-build` directory.

Additional paths are workload-dependent and are reported by `doctor`. IREE
command-line tools and `rocminfo` are resolved from `PATH`. Workload-specific
Python interpreters may be selected with:

```sh
export CONSAN_VALIDATION_SHARKTANK_PYTHON=/path/to/python
export CONSAN_VALIDATION_PYTORCH_PYTHON=/path/to/python
export CONSAN_VALIDATION_TENSILE_PYTHON=/path/to/python
```

The runner exposes these subcommands:

| Command | Purpose |
| --- | --- |
| `doctor` | Validate the selected workspace, tools, artifacts, runtime target, and hook mapping. |
| `manifest` | Print the target's executable workload matrix. |
| `prepare` | Build a canonical generated artifact; currently used for `qwen-prefill`. |
| `explain` | Expand workload commands, profile settings, implicit defaults, and reviewed fault policy without executing the workload. |
| `run` | Execute clean correctness and coverage rows. |
| `inventory` | Discover target- and binary-specific fault sites without mutation. |
| `fault` | Execute one reviewed fault specification with containment and health checks. |

Discover the exact current choices through the executable interface:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py --help
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" manifest
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" doctor --workload all
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" explain \
  --workload qwen-prefill --profile all
```

For an emulated target, pass the same JSON argv prefix to `doctor`, `run`,
`inventory`, and `fault`:

```sh
--launcher-json '["rocjitsu", "--config", "gfx1250_mi455x.json", "--"]'
```

The prefix is retained in the artifacts. The exact config name is deployment
specific; use one whose target matches `--target`.

### Preparing Qwen

The Qwen row requires a generated VMFB with recorded compiler and input
provenance:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" prepare --workload qwen-prefill
```

`doctor` rejects a missing or stale preparation manifest. Do not substitute an
untracked VMFB or weaken the output oracle.

### Clean qualification

Complete [allowlist discovery and runner integration](#first-step-for-revalidation-generate-and-apply-kernel-allowlists)
before these runs.

Use a new artifact root for each source, binary, runtime, settings, or manifest
state:

```sh
export CONSAN_ARTIFACT_ROOT="$CONSAN_VALIDATION_WORKSPACE_DIR/consan-validation/run-001"

python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" run \
  --workload qwen-prefill --profile all --phase clean \
  --include-baseline --artifact-root "$CONSAN_ARTIFACT_ROOT"
```

The runner scrubs inherited `HSA_TOOLS_LIB` and `RJ_CONSAN_*` variables before
constructing each profile. A clean instrumented row is accepted only when the
independent workload oracle passes, the expected code object is applicable,
all selected supported sites are instrumented, static and dynamic evidence is
complete, no forbidden overflow or unexpected diagnostic appears, and the
process completes within its manifest deadline.

Strict load rejection is a typed outcome with exit code 92. The runner retains
its reason rather than converting it into a missing teardown verdict.

`--timeout` is a diagnostic override. Changing it changes the execution
contract and requires a new artifact root. A missing workload/profile pair is
an incomplete campaign, not an omitted result.

### Fault inventory and review

Fault identities include the code-object hash, kernel, PC, mnemonic, and
occurrence. Rediscover them after any target, compiler, source, or binary
change:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" inventory \
  --workload tp1-prefill --artifact-root "$CONSAN_ARTIFACT_ROOT"
```

Inventory sets dry-run fault controls and applies no mutation. Review the
generated inventory and copy its template before editing. A reviewed spec must
replace every placeholder, precommit the expected detector outcome and
independent oracle for every applicable profile, declare any statistical trial
matrix, and set `review_required` to false. Do not choose a different site or
expected result after observing a live trial. The runner snapshots the exact
spec bytes it loads into `fault-spec.snapshot.json` and hashes those bytes in
the summary. Resuming a root with a different saved spec is rejected; use a new
root for changed expectations or trial settings.

A trial that fails mutation admission or produces no result stops the remaining
trials for that profile. The summary retains planned and attempted counts and
marks the incomplete batch rejected. Detector misses from admitted trials still
run the full precommitted matrix. Diagnose the admission failure before starting
a new batch.

### Contained fault execution

Fault runs are destructive experiments. Run them one at a time and acknowledge
that explicitly:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" fault \
  --workload tp1-prefill --profile all \
  --spec /path/to/reviewed-faults.json --fault barrier-drop \
  --artifact-root "$CONSAN_ARTIFACT_ROOT" --allow-destructive
```

The runner holds the global destructive-GPU lock, uses a separate process
group, enforces the deadline, and runs discovery plus a target-dispatch smoke
before and after each trial. `--health-command-json` and
`--smoke-command-json` can replace both commands when the defaults are not
usable; the exact replacements are retained.

A qualifying applied trial requires exactly-one planning and installation,
complete per-reader and per-process reservation evidence, a matching reviewed
detector result, a matching independent oracle when required, normal bounded
completion, and healthy pre/post probes. A timeout, signal, trap, wrong output,
or device reset is not a ConSan detection.

## Evidence and campaign discipline

Keep source state, dirty-tree state, toolchain/runtime identity, executable and
code-object hashes, commands, environment, workload inputs, timeouts, target
identity, raw output, coverage, report completeness, and health results
together under the artifact root. Do not merge exploratory and accepted roots
or reuse a root after its stable contract changes.

The hardware-entry ConSan path uses a 64-bit fingerprint of queue pointer and
absolute queue-local dispatch ID. It distinguishes the tested simultaneous
queues and ring-slot reuse but is not an injective encoding of the full pair:
collisions and queue-address reuse remain possible. ConSan also has a weaker
literal fallback under scalar pressure. Qualification must report which
representation was used; runtime trust cannot upgrade either representation
into exact global launch identity.

Record resolved workgroup/cell strides and offsets, achieved bank geometry,
FLAT provenance policy, owner/dispatch operating point, and host-epoch selection
alongside the preset name. Names alone do not preserve meaning across policy
changes. Interpret completeness against the
[implemented assumptions](../DESIGN.md#heuristics-and-their-failure-directions);
it does not independently validate those assumptions.

Fault qualification additionally separates mutation attempted, installed, and
reached. Process completion alone is not proof that the selected instruction
executed. Preserve oracle manifestations, ConSan diagnostics, timeouts,
signals, and health failures as different outcomes.

## Testing the validation runner

The runner's orchestration and containment behavior has CPU-only unit tests:

```sh
cd emulation/rocjitsu/tests/dbi/consan
python3 -m unittest \
  test_consan_coverage_gate.py \
  test_consan_fault_runner.py \
  test_consan_run_provenance.py \
  test_consan_tensile_validation.py \
  test_consan_validation.py
```

These tests validate environment scrubbing, manifest/profile isolation,
provenance, workload commands and oracles, coverage gates, identity inventory,
fault-spec validation, reservation accounting, and health containment. They do
not qualify a simulator or physical target cell.
