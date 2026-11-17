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
             Original code object + target/selected entries
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

The compiler inventories the exact original image, qualifies each selected entry,
and plans its probes, bootstrap and resources. One transactional rewrite produces
a coherent replacement that preserves unselected native entries. Its
artifact directory contains:

- `prepared.hsaco`: the instrumented code object.
- `facts.bin`: exact image and selection identities, an owning entry table with
  selected/native dispositions and ABI/resources, and each selected entry's
  instruction facts including observation read/write kind and width. Native rows
  retain their reachable code ranges for byte preservation checks; entry-address
  order does not determine ownership of intervening instructions.
- `ready`: a marker published after the artifact files close successfully.

The preparation tool embeds probe bodies compiled at build time for each
supported target/wave mode and lowers calls through the shared DBI framework.
Register counts and kernarg size in the prepared descriptor and AMDHSA note
are updated together by the shared code-object editor. Missing fields or integer
encodings too narrow for the requested sizes reject preparation before artifact
publication. ConSan keeps its own target, argument, and launch admission checks.

The compiled probes use [`DeviceWaveState`](../lib/rocjitsu/src/rocjitsu/code/patch/consan/wave_state.h)
for counters owned by each wave. State persists in runtime-owned memory across
probe calls until dispatch completion, so repeated execution does not depend on
values surviving in implicit registers.

### Execution and capture

The plugin matches the application's original object to the artifact and loads
the prepared replacement. At dispatch, the runtime supplies launch bindings and
report storage; GPU probes publish site identities, effective LDS addresses,
widths, and lane masks. An independent host snapshot lets GPU report storage be
reused while capture writing continues. Explicit finish drains these handoffs
and closes the capture.

A module binds every image entry after successful freeze. The actual kernel object
selects an entry binding; selected dispatches use that entry's original argument
ABI and prepared resource plan. Known unselected entries retain their original
packet, resources and completion signal and are native/uncovered. Unknown kernel
objects are rejected. One module, a controlled queue, nonoverlap and dedicated
completion signals remain required, including for native siblings.
Admission recognizes a native completion already visible to the application even
if the observer has not yet processed it. The observer retains its signal lease
until its wait returns, so callers still drain before releasing that signal.

The capture header owns the immutable image/entry facts. Each selected execution
references its original descriptor offset within that exact image. Guest probe
entry IDs remain local zero; host EntryKeys distinguish entries. Report generation
increases across all uses of a slot, including a change of entry. Native dispatches
are omitted from the selected-execution capture and require separate application
accounting. No cross-dispatch LDS candidate is formed.

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

#### Application infrastructure profile

`RJ_CONSAN_APPLICATION_PROFILE=1` enables the bounded application adapter.
`RJ_CONSAN_NATIVE_IMAGES` is a colon-separated list of exact original code-object
files whose entries remain native and uncovered. The runtime verifies their
complete ELF/metadata inventory and matches both bytes and digest before loading
them. These images do not require a prepared artifact. An optional selected
artifact still requires `RJ_CONSAN_ARTIFACT_DIR` and
`RJ_CONSAN_SELECTION_PROFILE`, together with `RJ_CONSAN_CAPTURE_PATH` for its
capture. Native-only execution does not require a capture path; native execution
cannot substitute for a selected artifact's capture.

This adapter imposes no additional limit on the number of live system-memory
compute queues; queue creation remains subject to ROCR and system resources.
Queues created through `hsa_amd_queue_create` before image loading are supported.
Queue priority and CU mask are retained. Freeze resolves every entry before
publishing bindings, and kernel objects must be distinct across the admitted
modules. Native ABI checks permit compiler hidden arguments, constant pointers,
nonzero private segments, and descriptor register allocations larger than their
metadata requirement.
Selected-entry qualification retains its stricter register and ISA requirements.

The native packet profile accepts standard dispatches and AMD extended format 3
dispatches with original completion signal zero. ROCR callback batches are split
in their original order, up to the shared 1024-execution budget. Hardware workgroup
limits and required workgroup shapes are checked; the metadata flat bound is
retained as inventory rather than imposing the selected detector's launch bound
on native packets. Only the null completion field changes, to a fresh runtime-owned
signal. Header fences, dependency signal, arguments, resources, cluster geometry
and performance hints are preserved. No barrier or dependency edge is added.
The private completion publication has a cost; exact timing equivalence is not
claimed. Nonzero native dispatch completions and extended format 4 are unsupported.

Multiple launches may remain pending. The observer retires each private signal
only after observing exactly zero, without reading, resetting or forwarding an
application signal. Original barriers pass through while admission is healthy;
queue profiling calls use the original API. Queue and executable destruction wait
for their own launches. Drain waits for the submitted prefix even if a later
launch completes first.
Coalesced batches apply the execution budget to dispatches individually, so a
trailing barrier does not consume an execution slot. If a dispatch exceeds that
budget, following tracked barrier completions receive failure notification while
pending dispatch ownership is retained.

Native-only finish drains and joins the observer without creating a selected
capture. Success means native infrastructure completion, with no ConSan probe or
LDS-analysis coverage. A configured selected artifact that never starts capture
fails finish. In application mode, shutdown automatically finishes before releasing
the final tracked HSA reference. An intermediate `hsa_shut_down` leaves admission
open for clients that still hold a reference. Tracking starts with the installing
`hsa_init` reference and counts successful initialization and shutdown calls
through the installed API table; cached calls that bypass its hooks are outside
this contract. Explicit finish remains available for retaining its status.
An application must treat tool-install failure as failure: ROCR may continue
running the original program after rejecting a tool.

Runtime ownership ends at `OnUnload`, rather than at C++ static destruction, so
earlier-registered HIP exit handlers can still destroy executables through the
hooks. Deletion is deferred through an enclosing shutdown call. Both Linux plugin
variants use `NODELETE` to keep hook code mapped when ROCR closes the tool inside
that call; Runtime resources are still released, but the DSO mapping remains for
the process lifetime.

#### Execution trace for application transport

Use an explicit profile to instrument entry and terminal boundaries without
performing LDS analysis:

```bash
"$consan_prepare" original.hsaco gfx1250 new-artifact-directory \
  --profile execution-trace --entry kernel_a --entry kernel_b
export RJ_CONSAN_APPLICATION_PROFILE=1
export RJ_CONSAN_ARTIFACT_DIR=/absolute/path/to/new-artifact-directory
export RJ_CONSAN_SELECTION_PROFILE=<printed-selection-profile>
export RJ_CONSAN_NATIVE_IMAGES=/absolute/path/to/exact-runtime-helper.hsaco
```

Keep the selected original image out of `RJ_CONSAN_NATIVE_IMAGES`: exact native
matches deliberately remain native. The new profile has a distinct options and
selection digest. Existing artifacts/captures must be regenerated after a producer
contract change; runtime and offline decoding reject mismatched contracts.

This profile admits gfx1250 wave32 selected entries with zero private segment,
a closed direct CFG, recognizable terminal instructions, an enabled original
workgroup-X input, and the shared SBI preservation/resource checks. Natural
loops and zero LDS are supported. Calls, indirect control flow, guest PC
introspection and VGPR deallocation remain outside this profile. Entry/exit boundary mappings are
separate from LDS access sites; a trace manifest has no LDS sites.

Selected dispatches require complete 1D workgroups, up to 512 groups, and retain
the original required workgroup shape and metadata flat bound. Extended-format-3
selected packets require unit cluster sizes in every dimension: TTMP9 supplies
a cluster coordinate for nonunit clusters and cannot identify a unique report
row. Native dispatches retain support for nonunit clusters. A GPU probe uses
snapshots of original VGPR0 lane zero and TTMP9 to identify local wave and
workgroup. Every expected row must contain its own serial, entry, flattened row,
workgroup, local wave and exactly one entry plus one exit observation. The host
initializes these counters to zero; it does not synthesize probe execution.

Application trace accepts original kernargs only when pointer-info proves a
known HSA allocation, a nonnull equal host/agent base, and the full argument
extent. CPU-mapped coarse allocations are allowed for this pre-publication copy;
unknown, locked, inaccessible or aliased pointers are rejected. The runtime-owned
wrapper, context and report still use a fine-grained CPU pool. Original arguments
and the original guest pointer are saved before forwarding the dispatch, so
completion never requires rereading application-owned kernargs.

Selected packets retain their representation, dependency signal, cluster
geometry, performance hints and barrier bit. They replace kernargs/completion and
strengthen acquire/release fences to SYSTEM for report visibility. ConSan adds
no barrier packet, dependency edge or per-dispatch host wait. This visibility
requirement and instrumentation overhead mean timing equivalence is not claimed.
The initial application profile requires null dispatch completions; no nonexistent
application signal is forwarded. Saved completion evidence certifies visibility,
without claiming signal forwarding or single-queue/nonoverlap restrictions.

The observer copies each completed report into independent owning host bytes
before permitting storage reuse. A serial-keyed queue lets observation and copy
continue while the writer emits captures in submission order. Application report
slots are bounded by the execution and cumulative capture-byte budgets; legacy
LDS admission retains its 16-slot limit. Queue/executable teardown waits through
its launches' independent copy, and finish waits for the submitted prefix and
successful End/close/publication.

`rj_consan_analyze` dispatches trace captures to a separate transport analyzer.
Its JSON says `scope: execution_trace` and `lds_analysis: not_performed`, with
expected/completed rows, validated entry/exit event counts, per-execution gaps and
native-uncovered entries. It emits no LDS verdict or race counters. Trace exit
codes are 0 for complete observed transport, 1 for transport gaps, 2 for invalid
input, and 3 for output failure; `--examples` remains an LDS-only option and does
not change trace diagnostics. An empty execution list cannot certify transport.

Registered `ConSanExecutionTraceSim.Groups1`, `.Groups16` and `.Groups512` controls
execute the original whole A/B/C/A fixture with original CPU output oracles and
native mixing, then replay in a fresh process after removing test artifacts.
Their application's following completion barrier is part of the fixture; the
ConSan interceptor adds none. These controls exercise simulator execution,
including TTMP workgroup attribution, and do not establish physical-GPU coverage.

### Offline diagnosis

The analyzer validates the closed capture, joins GPU observations to static
instruction facts, runs the checks, and emits JSON findings and coverage. The
capture owns the evidence needed for replay; analysis needs neither the
original code-object files nor live HSA state.

A prepared image's selected entries share one qualified scope; native entries
have no analysis facts. Mixed selected LDS/trace scopes and mixed barrier/loop
scopes are rejected before publication. The shared contract retains LDS scopes
1 (no synchronization), 2 (workgroup barrier), and 3 (bounded barrier loop), with
execution trace assigned scope 4. Readers interpret each scope's row fields
independently; trace counters are never barrier completion evidence.

## Available analysis

`LdsRaceDetection` is the only implemented analysis. It checks ordinary native
LDS reads/writes and admitted relaxed LDS atomic accesses at naturally aligned addresses, using the same observations
for two checks. All supported targets admit ordinary single-range b8, b16, b32,
b64, b96, and b128: gfx942/gfx950 wave64, gfx1100/gfx1201 wave32 or wave64, and
gfx1250 wave32.

| Check | What it reports |
| --- | --- |
| Cross-wave conflict | Address-group records from different waves access overlapping half-open byte ranges, with at least one ordinary write or an ordinary read paired with an atomic. Atomic/atomic and read/read pairs are excluded. |
| Same-instruction ordinary-store collision | Multiple lanes of one store observation write the same LDS range. |

### Analysis and admission

Preparation accepts a limited instruction/access subset with no synchronization,
or qualified full-workgroup barriers on the supported target/wave profiles.
On every supported target and wave size, relaxed native `ds_add_u32` and
`ds_add_f32` record four-byte Atomic accesses. `ds_add_u64` records eight bytes.
The native `ds_add_f64` also records eight bytes on gfx942, gfx950 and gfx1250.
The gfx1100 and gfx1201 instruction sets have no native FP64 LDS atomic add.
Returning `ds_cmpst_rtn_b32` / `ds_cmpstore_rtn_b32` contributes one four-byte
Atomic access on both compare outcomes and preserves the guest destination.
If the comparison succeeds, the instruction writes the replacement value; if it
fails, LDS remains unchanged. Either outcome can produce a conflict report with
an unordered ordinary load of the same bytes from another wave.
The probe captures the effective address and guest EXEC before the original RMW.
The report alone therefore cannot tell whether a compare-store exchange succeeded.
This observation contributes no ordering edge. Other atomic and barrier forms
require their own qualification. Exact
conditions are defined by [compiler qualification](../lib/rocjitsu/src/rocjitsu/code/patch/consan/lds_race.cpp)
and [runtime admission](../lib/rocjitsu/src/rocjitsu/hooks/consan/hsa_plugin.cpp).

Entries in one image must have disjoint, closed scopes and the same target/wave
mode. Shared code, cross-entry control flow, calls and unknown ownership reject
preparation. Each selected entry has its own original ABI, bootstrap, resources
and static site mapping. Unselected native siblings remain in the whole image;
this does not provide arbitrary native queue or HIP application integration.
The multi-entry simulator fixture qualifies gfx1250 wave32; single-entry coverage
on other modes does not establish their multi-entry execution qualification.

The subword forms are byte/halfword stores and whole-register signed or
unsigned byte/halfword loads. D16 loads place a zero- or sign-extended byte
(`ds_load_u8_d16`, `ds_load_i8_d16`) or a halfword (`ds_load_u16_d16`) in
the low half of a VGPR. Their `_hi` variants select the high half. In active
lanes the unselected half is preserved on gfx1100/gfx1201 and cleared on the
SRAM-ECC gfx942/gfx950/gfx1250 profiles. Inactive lanes keep the entire VGPR.
`ds_store_b8_d16_hi` and `ds_store_b16_d16_hi` store bits 23:16 or 31:16
of the source VGPR, respectively. The LDS range is one byte for byte forms
and two bytes for halfword forms.

The wider forms are
`ds_store_b32/b64/b96/b128` or `ds_load_b32/b64/b96/b128` on gfx1100/gfx1201/gfx1250.
gfx942/gfx950 use the corresponding `ds_write_*` and `ds_read_*` names.
Access probes capture addresses before an aliased load can overwrite them, record actual byte
widths and byte immediates, and preserve every guest result word. For b96 the
range spans three dwords. Its effective start must be 16-byte aligned so the
range is independent of the launch alignment mode. FLAT forms remain unsupported. The minimum fixed LDS extent remains four bytes.

Two-address native LDS loads and stores support 32- and 64-bit elements.
One guest instruction produces one occurrence and shares the four sampled
address groups across both elements. Each group records the pre-instruction
base and lane mask. Saved site facts retain both encoded offsets in operand
order. Replay adds each offset times the element size to the recorded base,
with an additional factor of 64 for stride-64 forms. For example, a b32
operation with offsets 3 and 5 covers bytes [base + 12, base + 16) and
[base + 20, base + 24). With equal offsets, gfx942/gfx950 coalesce the access
and a store uses DATA0.

The following matrix transpose reads gather LDS data from physical source lanes
and distribute the result across destination lanes. Each source lane contributes
one contiguous range starting at its pre-instruction address VGPR plus the byte
immediate. Only the returned elements undergo the matrix permutation. The
before-instruction probe also covers an address VGPR aliased by the destination.

| Element bits | gfx950, wave64 | gfx1250, wave32 |
| --- | --- | --- |
| 4 | `ds_read_b64_tr_b4` (8 bytes) | `ds_load_tr4_b64` (8 bytes) |
| 6 | `ds_read_b96_tr_b6` (12 bytes) | `ds_load_tr6_b96` (12 bytes) |
| 8 | `ds_read_b64_tr_b8` (8 bytes) | `ds_load_tr8_b64` (8 bytes) |
| 16 | `ds_read_b64_tr_b16` (8 bytes) | `ds_load_tr16_b128` (16 bytes) |

On gfx950, a defined active transpose requires full EXEC and the B64
destination starts at an even VGPR. The simulator issues no requests at
empty EXEC. Replay rejects a nonempty partial EXEC and reports
`partial_exec_lds_transpose`. The capture profile conservatively requires a
16-byte-aligned effective start for packed B96. [CDNA4 ISA, section 11.4](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)

On gfx1250, all 32 physical source lanes request data even at partial or empty
EXEC and in a short final wave. Each transpose record preserves the guest
EXEC separately from the source lanes which actually requested the recorded
range. Finding examples expose `mask_role` as `transpose_source_lanes` and
`anchor_exec` as the original mask. Physical source lanes may exceed the
number of logical workitems in the last wave. These reads share the four
address-group retention limit with ordinary operations. [CDNA5 ISA, section 11.2.4](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf)

Access widths are measured in bytes. `NativeLdsAccess::width` and
`SiteFacts::width` default to zero, meaning unspecified and invalid. Producers
must explicitly supply 1, 2, 4, 8, 12, or 16; zero never implies b32 or a zero-length
access. Manifest admission rejects zero, and a committed observation must match
its site's width. An all-zero uncommitted report slot still represents vacant
storage, not an access.

Range admission checks bounds by subtraction before calculating ends. Diagnosis
sweeps checked ranges, retaining grouped counts for identical ranges, excluding
Read/Read and same-wave pairs, and treating adjacent ranges as disjoint. Natural
alignment and the admitted widths bound active range shapes. Widths share the
same four-address-group quota, loss rules, exact collision masks, and qualified
barrier certificates. The compatibility token/options reject older b32-only
producer interpretations and target-specific width restrictions.

### Workgroups and synchronization

LDS belongs to a workgroup. Observations are kept separate by run, dispatch,
workgroup and wave, so equal LDS offsets in different workgroups cannot form
conflicts. Each workgroup's synchronization evidence is also independent.
All supported target/wave profiles admit up to sixteen complete 1D workgroups,
including partial final waves. Multi-group launches require the original
kernel's workgroup-X input, which entry snapshots preserve before guest
instructions can overwrite it.

Barrier qualification recognizes an original LDS drain followed by a
full-workgroup barrier: `s_barrier` on gfx942/gfx950/gfx1100, or a matching
signal/wait pair on gfx1201/gfx1250. It requires at least two waves per workgroup.
Legacy barriers require an original `s_waitcnt` with a zero LGKM threshold
immediately before `s_barrier`. Preparation rejects sequences with an intervening
instruction, including a global store or `s_waitcnt_vscnt`, even if an earlier
wait drained LDS.
Split-barrier drains may use `s_wait_dscnt` or a combined load/DS or store/DS
wait, provided the DS threshold is zero. The additional load/store threshold
may be nonzero.
In acyclic flow, every path to an exit must pass through each barrier group.
A barrier loop may instead use one basic block with a conditional self-backedge
and a forward exit, with all barrier groups inside that block. A preheader may
bypass the loop for zero iterations. These restrictions give every completed
iteration the same static barrier order without assuming uniform trip counts.

A static LDS instruction can execute many times. Within a wave, its site and
dynamic access sequence identify one execution; all address groups from that
execution share the sequence. A separate phase counts completed guest barrier
waits. For example, a producer/consumer loop needs two kinds of ordering:

```text
producer store(i) -> B0(i) -> consumer load(i) -> B1(i) -> producer store(i+1)
```

B0 publishes the producer's value before the consumer reads it. B1 lets the
consumer finish before the producer reuses the location. Distinguishing each
barrier's occurrence prevents a completion from another iteration from being
used as evidence for either relationship.

The analyzer derives ordering from matching barrier completions supplied by
every required wave, including waves with no retained LDS accesses. Probes
record completion only after the unchanged guest wait; probe execution and
numeric phase differences alone cannot establish synchronization. Access
positions must agree with both the qualified program and retained completions.

For a conflicting cross-wave pair, the result is:

- **Ordered:** a common completion places one access before the other.
- **Unordered:** both accesses lie in the same closed synchronization interval.
  Matching adjacent completions bound that interval; entry and a fully validated
  exit can supply the first and last boundaries. In the no-synchronization
  profile, qualification itself rules out ordering between valid endpoints.
- **Unresolved:** missing or contradictory evidence prevents either conclusion.
  A contradictory access position cannot be certified using another boundary.

### Bounded recording and coverage

Recording retains representatives within fixed storage. In repeated profiles,
each wave and site has four immutable access banks. An address group selects
`ctz(mask) % 4`, using its lowest active lane; the first record claims that bank.
Later attempts count as loss even if another bank is empty. Thus four banks do
not guarantee four iterations: repeated lane-0 accesses all select bank zero.
Other sites retain their own capacity. A probe also examines at most four
distinct addresses per instruction.

Each wave separately retains its first sixteen barrier completions. Later
barriers continue executing and contribute synchronization loss without
overwriting earlier witnesses. Sequence, phase and loss counters saturate
rather than wrap; a saturated access phase has unknown ordering position.
The [saved contract](../lib/rocjitsu/src/rocjitsu/code/consan/contract.h) defines
the shared record layout and capacity rules.

Loss limits coverage to the retained observations and synchronization intervals.
It does not revoke a conclusion whose required evidence remains valid: an
early closed interval can still confirm an unordered pair after later journal
loss. Same-instruction store collisions use exact address and lane-mask evidence
independently of barrier position. Capacity loss is reported separately from
malformed-evidence integrity issues.

The analyzer keeps ordered, confirmed unordered, and unresolved pairs separate.
Only confirmed conflicts contribute to the findings exit status; incomplete
coverage alone does not. Always inspect `coverage_complete` alongside findings.
Zero findings with missing evidence does not establish that unobserved accesses
were safe.

## How to use

This example runs a **gfx1250 wave32** assembly fixture with 64 threads in one
workgroup through the rocjitsu simulator; no physical gfx1250 GPU is required.
The AMDHSA code-object-v5 image, entry, target, and wave mode must match the
simulator configuration. A runner option does not change an image's wave mode.

The [controlled HSA runner](../tests/patch/consan/consan_hsa_runner.cpp) supplies this
fixture's argument bindings, controlled queue, nonoverlapping bounded 1D
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
assembler/linker for the original assembly fixture. Preparation additionally
requires `amdclang++` and `clang-offload-bundler` to build its callable probes.
The phase components are:

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
consan_preparation=$("$consan_prepare" \
  "$consan_original" gfx1250 "$RJ_CONSAN_ARTIFACT_DIR" --entry lds_producer_consumer)
export RJ_CONSAN_SELECTION_PROFILE=$(printf '%s\n' "$consan_preparation" | \
  sed -n 's/^selection_profile=//p')
```

The artifact destination must be new. The runtime verifies that this artifact
matches the exact original image and the independently configured selection
profile. `RJ_CONSAN_SELECTION_PROFILE` must be the 64 hexadecimal digits printed
by the intended preparation command. Missing/malformed or different profile
configuration rejects loading; the runtime does not infer a variant by symbol
name or copy its expected profile from the artifact it is checking. The old
single-entry positional CLI remains an explicit convenience that emits the same
new image-facts format.

The saved producer contract changed with image entry routing. Regenerate prepared
artifacts and captures together; historical facts/captures are not accepted as
compatible merely because guest records still use local entry ID zero.

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
  "ordered_pairs": 0,
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

## Multiple disjoint entries

Use repeated `--entry` options to select entries in one whole image:

```bash
"$consan_prepare" original.hsaco gfx1250 new-artifact-directory --entry mk_a --entry mk_b
```

A/B share one prepared ELF and one image table; an unselected C retains its native
ABI/resources. Keep the printed selection profile with the exact original and
prepared identities in the launch configuration. The HSA runtime's argument
alignment query is distinct from the AMDHSA metadata requirement: the pinned
ROCR code-object-v5 loader reports 16-byte alignment even for metadata alignment
8. Replacement preserves the HSA query rather than projecting the metadata value.

The registered `ConSanMultiEntrySim.SelectedAB` integration control runs the
[whole A/B/C assembly fixture](../tests/patch/consan/fixtures/multi_entry.s) with
one real load, A→B→C→A dispatches, different A/B argument/resource demands,
independent guarded CPU output oracles, selected facts/observation attribution,
and native accounting. It removes original/prepared artifacts before fresh-process
analysis. `ConSanMultiEntrySim.WrongProfile` checks that a different configured
profile cannot publish a successful capture. Run these with the required test
dependencies built; they do not establish toy Llama or physical-GPU support.

The analyzer reports an image entry/disposition table and each selected
execution's EntryKey/name. Decoder checks reject unknown/native keys and
inconsistent facts/extents. Local guest ID zero does not independently prove
entry provenance if a fabricated key selects another entry with an identical
layout; correct attribution depends on the runtime's validated kernel-object
binding. Checksums preserve recorded bytes, without providing a security
attestation of guest identity.

## Extending ConSan

Preparation uses the shared `Instrumentor` and `TrampolineBuilder` with
separately compiled, reusable HIP probe functions.

DBI owns dispatch entry and terminal placement, operand capture, parallel
argument copies, call envelopes, state preservation, resource growth, and emitted
site attribution. The runtime explicitly supplies the
kernarg wrapper required by the DBI entry prologue. ConSan retains native LDS
qualification, address grouping, observation fields, and diagnosis. Ordering and
loss counters use DBI's explicit memory-backed wave state; their locations stay
in ConSan's existing report rows. No hidden register state carries values from
one probe invocation to the next.

ConSan selects DBI's `DeclaredRegisters` preservation policy. It restores
originally declared GPRs that the probe or call envelope overwrites, including
values liveness marks dead. This gives ConSan a stronger transparency contract
and reduces its dependence on liveness precision. The LDS checks do not
intrinsically require this policy: `LiveRegisters` preserves live values and
can need fewer backups, but changing ConSan's policy would need execution validation of
that narrower contract. Merely preparing a dense input under `LiveRegisters`
does not establish that its instrumented execution is correct.

The conservative choice can require more registers and save/restore
instructions, and larger trampolines can exhaust branch reach sooner. ConSan
currently accepts those costs rather than selecting a different policy when
an input exceeds the resource or branch-distance limits.

The callable path uses register backups instead of introducing a private frame.
It requires each selected kernel to have enumerable register uses, zero VGPR banks at gfx1250
call sites, a bank-preserving probe body, and enough registers for the combined
probe and preservation plan. An input exceeding that plan is rejected during
preparation. Entry stubs and per-site trampolines use SOPP branches, whose
signed 16-bit dword displacement reaches approximately 128 KiB in either
direction. Large original text or dense instrumentation can exceed that reach
even within the input size and instruction budgets; preparation rejects these
inputs before publishing an artifact. Streaming transport, arbitrary placement,
and general MODE save/restore remain outside the current implementation.

Enabling `RJ_BUILD_CONSAN_COMPILER` requires ROCm's `amdclang++` and
`clang-offload-bundler`, including when `BUILD_TESTING=OFF`: CMake compiles the
HIP probe functions and embeds a standalone code object for each supported
target/wave mode. A missing tool is a configuration error. These are build-time
dependencies of preparation; the runtime and analyzer neither build that
catalogue nor require the device compiler.

A new analysis primarily extends **compiler facts/probes and offline diagnosis**
together. Define its question, scope, required evidence, and behavior under loss:

- **Compiler:** select operations and lower probes while preserving guest state
  and executing each original operation once. Reuse rocjitsu's code-object
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

Barrier tests check publication and reuse independently by removing each
barrier and checking the resulting unordered pair. Output preservation alone
does not show that a wait blocked an early consumer, so the
[simulator controls](../tests/patch/consan/bootstrap_sim_test.cpp) also delay the
producer and compare synchronized and unsynchronized execution. Removing only
the prepared guest wait while retaining its callback checks that instrumentation
does not substitute for guest synchronization. These scheduling controls provide
simulator evidence; physical hardware may schedule unsynchronized accesses
differently.

## Fault injection

The independent offline `rj_code_mutate` CLI and its fault-injection library can
remove a qualified complete barrier from an AMDGPU code object. They use
Instrumentor's instruction-replacement path and do not require the ConSan
compiler, runtime, or analyzer. See the [command-line usage](fault-injection.md#command-line-usage)
and [Fault injection](fault-injection.md) for the library interface, current
features, target/wave coverage, and limitations.

Treat the mutant as a new original image: perform a **fresh ConSan prepare**,
then execution/capture → analyze. Do not reuse pristine prepared artifacts or
capture bindings. Drop uses branch-to-NOP trampolines, so emitted ELF file
offsets and CFG block boundaries can differ from the pristine image. Preparation
must discover the mutant's entries and instructions. Preparation may reject a
target or program profile that the mutation library admits.

Fresh preparation currently requires a **single-kernel code object**. A
sibling-preserving multi-kernel mutant cannot use this path. Remaining barriers
must satisfy ConSan's qualified gfx1250 wave32 workgroup-barrier profile; other
barrier forms remain unsupported. The mutant must also satisfy ConSan's other
target, entry, instruction, native-LDS and resource restrictions, enforced by
[input qualification](../lib/rocjitsu/src/rocjitsu/code/patch/consan/input.cpp) and
the [LDS classifier](../lib/rocjitsu/src/rocjitsu/code/patch/consan/lds_race.cpp).
Acyclic trampoline returns can be admitted. Apart from ConSan's qualified
conditional single-block barrier loop, CFG cycles remain rejected. Native LDS
access PCs must increase along each path within an iteration; the qualified
loop uses repeated epochs. Mutation admission and detector admission remain
independent decisions.

For this example, reuse the paths and plugin environment from
[Build and setup](#build-and-setup). Replace `original.hsaco` and `my_kernel`
with a `gfx1250` workload's input and exact entry containing an admitted
complete barrier:

```bash
consan_mutant="$consan_example/mutant.hsaco"
"$consan_build/tools/rj_code_mutate" \
  --input original.hsaco --kernel my_kernel --kind DropBarrier \
  --ordinal 0 --output "$consan_mutant"
export RJ_CONSAN_ARTIFACT_DIR="$consan_example/mutant-artifact"
export RJ_CONSAN_CAPTURE_PATH="$consan_example/mutant-capture.bin"
"$consan_prepare" \
  "$consan_mutant" gfx1250 my_kernel "$RJ_CONSAN_ARTIFACT_DIR"
```

If preparation admits the mutant, continue with [execution and capture](#2-execution-and-capture)
using the workload's launch code to load `consan_mutant` and its matching entry.
Keep the new artifact and capture bindings above, then run the existing
[offline diagnosis](#3-offline-diagnosis) command. The plugin's original image
is now the mutant; it must not bind the pristine image or its prepared artifact.

Mutation success reports emission after input qualification. It establishes
neither execution nor detector qualification: removing a barrier does not prove
that a race occurs, that a particular execution observes it, or that ConSan
reports it. Qualify those stages separately using the normal capture and analysis
workflow.
