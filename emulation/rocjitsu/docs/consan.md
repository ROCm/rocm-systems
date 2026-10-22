# ConSan: design and use

ConSan investigates memory conflicts in compiled AMD GPU kernels using native
ISA instrumentation. It inserts probes into a code object and diagnoses the
accesses observed when that instrumented kernel runs. Working on native
instructions makes the analysis follow the memory operations and active lanes
after source compilation, without requiring the kernel's source as input.

The implementation separates preparation, execution/capture, and diagnosis.
Prepared code can be inspected before an application loads it; the application
process supplies execution resources and saves observations; a separate tool
analyzes that saved evidence after the run.

The execution/capture runtime is a standalone ROCr HSA tools plugin,
`libhsa_consan.so`, loaded into the application through `HSA_TOOLS_LIB`. It
intercepts code-object loading and dispatch, supplies runtime bindings and
storage, and saves observations.

Currently, ConSan implements one bounded analysis of LDS, the memory shared
within a workgroup. It does not provide general GPU program coverage.

The simulator also provides a separate [race detector](race-detector.md) that
tracks in-flight memory events during simulated execution. ConSan obtains
observations from inserted native ISA probes and diagnoses saved captures.
Their evidence and supported operation sets differ, including when ConSan's
instrumented kernel runs through the simulator.

## Design

The three-stage split has three purposes:

- **Offline preparation:** Move program analysis, probe lowering, and rewriting
  out of application startup, so transformed code and resource requirements can
  be inspected before the application runs.
- **Execution and capture:** Reuse a prepared artifact for matching original
  inputs. The runtime supplies values known only at launch time and handles
  execution and capture.
- **Offline diagnosis:** Replay saved evidence in an independent analyzer.
  Diagnosis can evolve separately from application execution, within the saved
  evidence's contract.

```text
             Original code object + target/entry
                              |
                              v
+------------------------------------------------------------+
| 1. Offline preparation (compiler)                          |
|                                                            |
| Decode/qualify -> probe/resource planning                  |
|                -> rewrite/relocate/validate                |
+------------------------------------------------------------+
                              |
                  prepared.hsaco + facts.bin
                              |
                              v
+------------------------------------------------------------+
| 2. Execution and capture (HSA plugin + GPU)                |
|                                                            |
| Application loads original -> match/verify/load prepared   |
| Application dispatch -> bind wrapper/context/report        |
| Prepared kernel + probes -> observations/loss/status       |
| Completion/visibility -> forward application completion    |
| Owning host snapshot -> writer -> explicit finish          |
+------------------------------------------------------------+
                              |
                     Closed capture file
                              |
                              v
+------------------------------------------------------------+
| 3. Offline diagnosis (analyzer)                            |
|                                                            |
| Decode/qualify -> analysis checks -> render                |
+------------------------------------------------------------+
                              |
                     Findings + coverage
```

### Offline preparation

The compiler qualifies the exact original image and entry, plans probes and
resources, and rewrites code while preserving the original operations. Its
artifact directory contains:

- `prepared.hsaco`: the instrumented code object.
- `facts.bin`: image identities and static instruction facts, including each
  observation site's read/write kind and width.
- `ready`: a marker published after the artifact files close successfully.

### Execution and capture

The plugin matches the application's original object to the artifact and loads
the prepared replacement. At dispatch, the runtime supplies launch bindings and
report storage; GPU probes publish site identities, effective LDS addresses,
widths, and lane masks. An independent host snapshot lets GPU report storage be
reused while capture writing continues. Explicit finish drains these handoffs
and closes the capture.

The runtime neither compiles nor diagnoses races. Its transport completion does
not synchronize guest LDS accesses.

Applications integrating the plugin must check completion values and finish
status. The plugin tracks signals successfully created through the hooked
`hsa_signal_create` or, when provided by the runtime, `hsa_amd_signal_create`
after plugin installation. Dispatch completions must be fresh, dedicated signals
initialized to `1`. Signals created before installation, imported from another
process, or obtained through other APIs or by bypassing the hooks are untracked.
They cannot be used as dispatch completions and receive no failure notification.

When the plugin rejects a dispatch before submission, it records failure and
stores `-1` to its tracked application completion signal, unless pending work
already owns that signal. Rejected batches also notify tracked AND/OR barrier
completions with `-1`, preserving signals owned by pending dispatches. Once the
runtime has failed, it discards later AND/OR barriers with the same notification
rules. Healthy individual barriers and barrier-only batches pass through. Other
packet types are discarded without completion notification, even if their
signals are tracked. Clients must use bounded waits: a discarded barrier with an
untracked signal can retain its initial value, and a dependency requiring `0`
is not satisfied by the failure value `-1`.

The failure notification releases a waiter for a value less than `1`, but does
not mean that the kernel ran. Accept only exactly `0` as normal completion, and
check it before reading kernel output. A timeout or any other completion value
is an execution failure.
The [example runner](../tests/patch/consan/consan_hsa_runner.cpp) enforces these checks.

Keep resources alive until their ownership handoffs have drained. The runner
uses a successful `rj_consan_drain()` before releasing each completed dispatch's
signal. Call `rj_consan_finish()` while HSA and the remaining dispatch resources
are still alive, and check for return `0` before tearing them down or calling
`hsa_shut_down`. A nonzero finish status means the capture was not successfully
saved, even if earlier kernels completed normally. The plugin
refuses HSA shutdown until finish reaches its terminal state. Shutdown acceptance
does not establish capture success; the application must retain the finish
status. A failed finish does not promise recovery from work still in flight.
Unexpected C++ exceptions terminate the process and do not promise recovery,
cleanup or completion of outstanding work.

### Offline diagnosis

The analyzer validates the closed capture, joins GPU observations to static
instruction facts, runs the checks, and emits JSON findings and coverage. The
capture owns the evidence needed for replay; analysis needs neither the
original code-object files nor live HSA state.

## Available analysis

`LdsRaceDetection` is the only implemented analysis. It checks ordinary native
32-bit LDS reads/writes at naturally aligned addresses, using the same
observations for two checks:

| Check | What it reports |
| --- | --- |
| Cross-wave conflict | Address-group records from different waves access the same LDS word, with at least one write. Read/read pairs do not conflict. |
| Same-instruction ordinary-store collision | Multiple lanes of one store observation write the same LDS word. |

### Analysis and admission

Preparation accepts a limited instruction/access subset without relevant guest
synchronization; unsupported forms, barriers, and atomics are rejected. Exact
conditions are defined by [compiler qualification](../lib/rocjitsu/src/rocjitsu/code/patch/consan/lds_race.cpp)
and [runtime admission](../lib/rocjitsu/src/rocjitsu/hooks/consan/hsa_plugin.cpp).

The input code object must contain exactly one kernel, matching the requested
entry. Selecting an entry does not exclude sibling kernels from this requirement;
provide a code object containing only the selected kernel.

Recording is bounded and does not provide a full execution history. Broader
access forms, synchronization analysis, and application integration require
extensions.

## How to use

This example runs a **gfx1250 wave32** assembly fixture with 64 threads in one
workgroup through the rocJitsu simulator; no physical gfx1250 GPU is required.
The AMDHSA code-object-v5 image, entry, target, and wave mode must match the
simulator configuration. A runner option does not change an image's wave mode.

The [controlled HSA runner](../tests/patch/consan/consan_hsa_runner.cpp) supplies this
fixture's argument bindings, controlled queue, nonoverlapping one-workgroup
launch, and explicit finish protocol. It is the example application, not a
general application launcher; other integrations must meet the runtime
admission conditions linked above.

The [assembly fixture](../tests/patch/consan/fixtures/native_lds.s) has this logic
for the selected `native-1.hsaco`. This is a logical sketch, not compilable HIP
input:

```text
one workgroup, two wave32 waves
wave 0, lane 0: write a 32-bit value to LDS byte offset 0
wave 1, lane 0: read 32 bits from LDS byte offset 0
no workgroup synchronization; inter-wave execution order is unspecified
```

The racing read has no required value; the finding is based on the observed
conflicting accesses.

### Build and setup

Follow [building.md](building.md) for general prerequisites and build setup.
Use a configured build from this checkout, with its ROCm SDK environment active.
This simulator example requires Linux, HSA headers/runtime, and the AMD
assembler/linker for the original assembly fixture. The phase components are:

- **Offline preparation:** `rj_consan_prepare`, enabled by
  `RJ_BUILD_CONSAN_COMPILER=ON`.
- **Execution and capture:** `libhsa_consan.so`, enabled by
  `RJ_BUILD_CONSAN_RUNTIME=ON`. The example uses `consan_hsa_runner` as its
  application and `rocjitsu` to launch it in the simulator.
- **Offline diagnosis:** `rj_consan_analyze`, enabled by
  `RJ_BUILD_CONSAN_ANALYZER=ON`.

Enable `BUILD_TESTING=ON` for the runner and fixtures. Use a full build to also
obtain the simulator launcher, its `librocjitsu.so` runtime, and the target
fixture. Preserve the build's compiler, SDK, and sanitizer settings. Run from
the repository root, selecting your configured directory for `consan_build`:

```bash
consan_source="$PWD/emulation/rocjitsu"
consan_build="$consan_source/build"
cmake -S "$consan_source" -B "$consan_build" \
  -DBUILD_TESTING=ON \
  -DRJ_BUILD_CONSAN_COMPILER=ON \
  -DRJ_BUILD_CONSAN_RUNTIME=ON \
  -DRJ_BUILD_CONSAN_ANALYZER=ON
cmake --build "$consan_build" --parallel 4
```

Set the paths and plugin environment in the same shell. This selects the
target-specific assembly image and a new directory for artifacts and capture;
the launcher passes the plugin settings to the runner.

```bash
consan_prepare="$consan_build/tools/rj_consan_prepare"
consan_analyze="$consan_build/tools/rj_consan_analyze"
consan_launcher="$consan_build/tools/rocjitsu/rocjitsu"
consan_runner="$consan_build/tests/patch/consan/consan_hsa_runner"
consan_plugin="$consan_build/lib/rocjitsu/src/rocjitsu/hooks/consan/libhsa_consan.so"
consan_original="$consan_build/tests/patch/consan/fixtures/targets/gfx1250/wave32/native-1.hsaco"
consan_config="$consan_source/configs/gfx1250_mi455x.json"
consan_example=$(mktemp -d "${TMPDIR:-/tmp}/consan-example.XXXXXX")

export HSA_TOOLS_LIB="$consan_plugin"
export RJ_CONSAN_PLUGIN_PATH="$consan_plugin"
export RJ_CONSAN_ARTIFACT_DIR="$consan_example/artifact"
export RJ_CONSAN_CAPTURE_PATH="$consan_example/capture.bin"
```

### 1. Offline preparation

```bash
"$consan_prepare" \
  "$consan_original" gfx1250 lds_producer_consumer "$RJ_CONSAN_ARTIFACT_DIR"
```

The artifact destination must be new. The runtime verifies that this artifact
matches the exact original image and qualified entry.

### 2. Execution and capture

```bash
"$consan_launcher" --config "$consan_config" -- \
  "$consan_runner" \
  "$consan_original" lds_producer_consumer 1 observe \
  --target gfx1250 --wave-size 32 --threads 64
```

The runner loads the **original** image; the plugin supplies the replacement.
The ConSan plugin produces the capture inside this simulated application run.
The runner calls [rj_consan_finish()](../lib/rocjitsu/src/rocjitsu/hooks/consan/api.h)
before HSA teardown; no separate finish command is needed. Successful execution
and capture return runner exit 0 and publish `RJ_CONSAN_CAPTURE_PATH`. Existing
capture files are not overwritten.

### 3. Offline diagnosis

```bash
"$consan_analyze" "$RJ_CONSAN_CAPTURE_PATH" --examples 1
```

The analyzer writes JSON to stdout. For this race example, exit 1 means a
conflict was found; exit 0 means no findings were observed. The result includes:

```json
{
  "verdict": "findings",
  "target": 1250,
  "wave_size": 32,
  "cross_wave_pairs": 1,
  "collision_groups": 0,
  "coverage_complete": true
}
```

Static facts identify the write/read sites; captured observations place them
in different waves at LDS offset 0, each with lane 0 participating. Joining
those facts produces one conflicting record pair, `cross_wave_pairs: 1`, with
no same-instruction store collision. The example includes original instruction
positions and lane masks. `--examples 1` limits displayed examples per check per
execution without changing counts or coverage.

Findings describe conflicts among valid observations; coverage describes how
fully the admitted scope was observed. Findings can coexist with incomplete
coverage. Zero findings, even with `coverage_complete: true`, does not prove
that the application is race-free or that another run has the same result.
Always inspect coverage independently of the verdict and exit code.

The closed capture can be analyzed again without images or application state.
Other normal nonzero analyzer statuses indicate input/output failures. See the
[analyzer CLI](../tools/rj_consan_analyze.cpp) and
[analysis results](../lib/rocjitsu/src/rocjitsu/code/consan/analysis/lds_race.h)
for option, status, and result details.

Entry names must be valid UTF-8 for JSON output. Preparation rejects malformed
names before publishing an artifact. The analyzer checks saved captures and
rejects malformed names with input-error exit 2 before writing JSON. Binary
capture retains the original name bytes. ASCII and valid UTF-8 names keep their
original spelling.

## Extending ConSan

ConSan currently shares binary-instrumentation building blocks with general
DBI: code-object patching, the entry prologue, kernarg wrappers, descriptor
scanning, instruction builders, and text-layout helpers. Its LDS lowerer does
not use `Instrumentor` or `TrampolineBuilder`. It owns instruction placement,
branch rebinding, probe register reservation, and inline probe emission for a
closed, acyclic CFG without probe calls. The general instrumentor currently
admits only `BeforeInst` sites, limits its call-free inline path to a nop, and
does not automatically grow SGPR allocations; it cannot directly express this
lowering's entry, access, and exit probes.

A common binary-instrumentation layer shared by ConSan and general DBI is a
possible future direction. Evaluating it requires common contracts for
instruction placement and branch relocation, register allocation and resource
growth, guest-state preservation, and target-specific hazards. LDS operation
qualification and observation semantics would remain with ConSan. The current
reuse boundary is a starting point for that discussion, not a commitment to
move ConSan into the existing instrumentor or to introduce another framework.

A new analysis primarily extends **compiler facts/probes and offline diagnosis**
together. Define its question, scope, required evidence, and behavior under loss:

- **Compiler:** select operations and lower probes while preserving guest state
  and executing each original operation once. Reuse rocJitsu's code-object
  parsing, ISA decoders/builders, and ELF rewriting/relocation through the
  [shared DBI machinery](dbi-design.md).
- **Offline analyzer:** qualify the saved evidence and implement checks,
  results, and rendering. Start from the current
  [LDS analyzer](../lib/rocjitsu/src/rocjitsu/code/consan/analysis/lds_race.cpp).
- **Runtime/capture:** reuse artifact matching, launch bindings, owning
  snapshots, and capture writing. Extend these services only when new host
  facts, storage, or lifetimes require it; diagnosis stays offline.

When new evidence is needed, evolve the [saved contract](../lib/rocjitsu/src/rocjitsu/code/consan/contract.h),
codecs, and validators with their producers and consumers. Current input,
contract, and result types are LDS-specific, so a second analysis must adapt
those boundaries together.

Use focused producer/transport, runtime, and analyzer checks, plus the
three-process controls in [consan_e2e.py](../tests/patch/consan/consan_e2e.py), which cover
a finding case and a defined-behavior preservation control. Every admitted
target/wave mode uses the same integration scenarios in the simulator and on
matching physical hardware. Required simulator cases must execute without skips;
physical cases skip when the requested HSA agent is unavailable. Runtime-stage
scenarios remain available without building the analyzer. Keep simulator and
physical-target evidence distinct.
