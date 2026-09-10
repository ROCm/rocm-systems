# ConSan validation

This guide describes the maintained validation interfaces for ConSan. The
executable authority for external-workload campaigns is
[`consan_validation.py`](../../tests/dbi/consan/consan_validation.py); its
manifest owns workload commands, timeouts, profile environments, correctness
oracles, and fault policy. Use `--help`, `manifest`, and `explain` instead of
copying workload-specific commands into this document.

The target ledgers record qualification state for
[CDNA3 / gfx942](STATUS_CDNA3.md), [CDNA4 / gfx950](STATUS_CDNA4.md),
[RDNA3 / gfx1100](STATUS_RDNA3.md), [RDNA4 / gfx1201](STATUS_RDNA4.md), and
[CDNA5 / gfx1250](STATUS_GFX1250.md). A ledger is not a substitute for rerunning
the gates after a relevant source, toolchain, workload, or runtime change.

## Validation layers

ConSan uses four complementary layers:

1. Host unit tests exercise parsing, inventory, policy, planning, native
   emission, placement, spilling, report models, final validation, hook
   configuration, and validation-script behavior.
2. Target-native simulator tests execute real code objects for all five
   supported targets through RocJITsu.
3. Physical tests execute the matching target-native fixtures on hardware and
   finish with an uninstrumented health check.
4. External-workload campaigns qualify unmodified production-shaped programs,
   overhead, and reviewed fault injection.

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

Run the complete nonphysical ConSan gate with bounded host parallelism:

```sh
ctest --test-dir /path/to/rocjitsu-build \
  -L consan -LE physical --output-on-failure -j16
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
[capability contract](CAPABILITIES.md), including CDNA5 cluster operations.

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
ROCM_SDK="$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_devel"

cmake -S /path/to/rocm-systems/emulation/rocjitsu \
  -B /path/to/rocjitsu-build -G Ninja \
  -DCMAKE_C_COMPILER="$ROCM_SDK/lib/llvm/bin/clang" \
  -DCMAKE_CXX_COMPILER="$ROCM_SDK/lib/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER="$ROCM_SDK/bin/hipcc" \
  -DROCM_PATH="$ROCM_SDK"

LD_LIBRARY_PATH="$ROCM_SDK/lib:$ROCM_SDK/lib/rocm_sysdeps/lib" \
  ctest --test-dir /path/to/rocjitsu-build \
  -L consan -LE physical --output-on-failure -j16
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
| `run` | Execute `clean` correctness/coverage rows or `overhead` rows. |
| `study` | Run the reproducible physical-gfx1201 empirical timing protocol. |
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

### Overhead

Run overhead without fault injection:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" run \
  --workload tp1-prefill --profile all --phase overhead \
  --include-baseline --artifact-root "$CONSAN_ARTIFACT_ROOT"
```

With `--include-baseline`, the order is baseline-before, selected profiles,
then baseline-after. `summary.json` reports raw samples, paired baseline, and
mode ratios. Do not compare cold first-operation ratios with warm steady-state
ratios; the manifest and result artifacts identify which protocol a row uses.
For statistically controlled performance studies, follow
[EMPIRICAL_METHODOLOGY.md](EMPIRICAL_METHODOLOGY.md) and use `study`.

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
expected result after observing a live trial.

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

The hardware-entry MOI path uses a 64-bit fingerprint of queue pointer and
absolute queue-local dispatch ID. It distinguishes the tested simultaneous
queues and ring-slot reuse but is not an injective encoding of the full pair:
collisions and queue-address reuse remain possible. Sampled also has a weaker
literal fallback under scalar pressure. Qualification must report which
representation was used; runtime trust cannot upgrade either representation
into exact global launch identity.

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
provenance, workload commands and oracles, coverage gates, overhead math,
identity inventory, fault-spec validation, reservation accounting, and health
containment. They do not qualify a simulator or physical target cell.
