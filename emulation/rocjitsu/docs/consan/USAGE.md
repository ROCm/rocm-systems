# ConSan usage

ConSan instruments final AMD GPU code objects through the rocJITsu HSA-tools
hook. It has native support for `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and
`gfx1250` and does not translate code objects between GPU architectures.

ConSan reads the active workgroup-LDS capacity from the runtime agent. It does
not hard-code a target-family LDS size; simulator and offline tests use the selected
RocJITsu JSON configuration as their source of truth.

Use [TUTORIAL.md](TUTORIAL.md) for a short walkthrough,
[FLAVORS.md](FLAVORS.md) for a conceptual device/deferred/host comparison,
[SPILLING.md](SPILLING.md) for ConSan register-resource policy and
[AMDGPU register spilling](../spilling.md) for the reusable backend.

## Build and load the hook

Build in a rocJITsu CMake build directory, never in the source tree:

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_dbi_hooks
```

The hook is normally located at:

```sh
export CONSAN_HOOK="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
```

Load it into any HSA application with:

```sh
env \
  HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_LOG=1 \
  ./application
```

Loading the hook is itself the activation action and defaults to MOI
Sampled; no separate enable variable is required. `RJ_CONSAN_LOG=1` is
optional.

Every valid code object on a waitcheck-supported target that is not excluded
by the kernel allowlist is checked for missing AMDGPU waits at load time before
ConSan allocates runtime state or runs its DBI transform. Hazards and analysis
failures are reported first, then ConSan still instruments the suspect code so
it can diagnose its memory-ordering behavior. This preflight is part of the
ConSan hook and is not controlled by
`ROCJITSU_WAITCHECK`, `ROCJITSU_WAITCHECK_MODE`, or
`ROCJITSU_WAITCHECK_FAIL`; ordinary ConSan runs do not load the separate
waitcheck HSA tool.

## Ordinary flavors and engines

ConSan exposes two top-level flavors. MOI contains three engines:

For the execution model behind these short descriptions, see the side-by-side
comparison in [FLAVORS.md](FLAVORS.md).

| Selection | Ordinary `standard-v1` behavior | Primary tradeoff |
| --- | --- | --- |
| `RJ_CONSAN_MODE=record-replay` | Instrument all admitted supported access, barrier, atomic, and fence sites; allocate an inventory-sized report; replay visible records on the host. | Expert synchronization engine with clear reference/debug semantics, but only a bounded dynamic snapshot. |
| `RJ_CONSAN_MODE=inline-shadow` | Publish exact-shadow cells and bounded diagnostics on the GPU; track admitted barriers and atomics. | Strongest supported-form attribution, with higher overhead. |
| default or `RJ_CONSAN_MODE=sampled` | Patch all admitted supported sites; use automatic runtime stride 256 and offset zero; retain bounded sampled causal windows and synchronization metadata. | Default engine: bounded retained state, lower overhead, and probabilistic detection. |
| `RJ_CONSAN_MODE=supercollider` | Duplicate/read-back supported LDS accesses, delay, compare, and set an automatically allocated non-trapping mismatch marker. | Complementary value-instability diagnostic; it does not attribute a happens-before edge or exact racing pair. |

`RJ_CONSAN_MODE` defaults to `sampled` when unset or empty. To retain the
previous default behavior, explicitly set `RJ_CONSAN_MODE=record-replay`.
Spelling the mode explicitly also makes saved commands self-describing.

Record/Replay's complete static-site instrumentation is not an exhaustive
dynamic trace. The ordinary automatic layout uses a report-wide dispatch
directory and a report-wide access-identity table with 2× open-addressing
headroom. The access table is sized from the admitted logical ranges and
adaptive dispatch/owner diversity factors; records retain the full dispatch,
static site, three-dimensional workgroup, and wave-owner identity. Either
table reaching its bounded 1,024-probe limit is a typed, report-wide
dynamic-incomplete saturation signal; it is never silent cross-identity reuse.
Caller-owned size-derived buffers retain single-bank behavior with exact
dispatch/workgroup qualification. Both automatic and caller-owned layouts
capture the complete 32-bit `(workgroup_x, workgroup_y, workgroup_z)` tuple at
kernel entry; later probes never assume descriptor SGPRs or RDNA launch TTMPs
still contain launch values. A clean replay remains inconclusive.

Ordinary runs do not need a register number, report-buffer size, barrier
switch, atomic switch, or sampling setting. The hook logs
`moi_profile=standard-v1` and whether a value came from the standard settings or
an expert override.

## Minimal commands

Default engine, Sampled:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_LOG=1 \
  ./application
```

Inline Shadow:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_MODE=inline-shadow \
  RJ_CONSAN_LOG=1 \
  ./application
```

Record/Replay (explicit opt-in):

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_MODE=record-replay \
  RJ_CONSAN_LOG=1 \
  ./application
```

SuperCollider:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_MODE=supercollider \
  RJ_CONSAN_LOG=1 \
  ./application
```

For focused tests, require an effective transform and report collection:

```sh
export RJ_CONSAN_POLICY=strict
```

Strict policy does not make race diagnostics fatal. Expert validation harnesses
with a predeclared expected result can add one of these assertions:

```sh
export RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1   # known-correct MOI control
export RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1  # predeclared positive MOI control
```

Do not enable both diagnostic guards. Strict policy can be too restrictive for
a broad application that loads helper code objects with no admitted sites. A
strict code-object rejection terminates at the loader boundary with the typed
`ConSan load rejection` diagnostic and exit code 92. This prevents HIP clients
that ignore an HSA load error from retaining a null kernel symbol and crashing
later during launch. Explicit `RJ_CONSAN_FAIL_CLOSED=1` under the default policy
continues to return the HSA error to callers that correctly handle it.

## Core controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_MODE=record-replay|inline-shadow|sampled|supercollider` | `sampled` | Select the analysis. Loading the hook activates ConSan. |
| `RJ_CONSAN_POLICY=default|strict` | `default` | `strict` defaults fail-closed and require-patch guards to true; for MOI it also defaults automatic-record and forbid-overflow guards to true. It does not require complete static coverage or make race diagnostics fatal. A load-time rejection terminates with exit code 92. |
| `RJ_CONSAN_LOG=N` | disabled | Enable compact logs at `1`; larger values add inventory detail. |
| `RJ_CONSAN_FAIL_CLOSED=0|1` | `0` | Reject unsupported/invalid transformation outcomes instead of loading the original. |
| `RJ_CONSAN_REQUIRE_PATCH=0|1` | `0` | Reject an applicable code object when no real access/barrier/atomic/fence instrumentation patch is emitted. Prologues and metadata-only changes do not satisfy it. |
| `RJ_CONSAN_KERNEL_ALLOWLIST=name[,name...]` | unset (all entries) | Instrument only the named HSA kernel entries. Names are exact, comma-separated, and may optionally include the `.kd` suffix. |
| `RJ_CONSAN_KERNEL_ALLOWLIST_FILE=path` | unset | Read exact HSA kernel entry names one per line. This is mutually exclusive with `RJ_CONSAN_KERNEL_ALLOWLIST` and supports demangled names containing commas. |
| `RJ_CONSAN_FLAT_PROVENANCE=likely|strict` | `likely` | Admit proven `Group` plus heuristic `MaybeGroup` flat LDS sites, or only proven `Group` sites. |
| `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES=N` | `939524096` (896 MiB) | Bound total additional ELF bytes, including alignment padding, across one code-object transformation. `N` is unsigned decimal; zero permits only no-growth rewrites. |
| `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT=N` | unset (absolute policy applies) | Use an alternative bound relative to the original code object: total additional ELF bytes may not exceed `floor(original input bytes * N / 100)`. `N` is unsigned decimal in `0..4294967295`; values above 100 intentionally allow growth larger than the original image. |
| `RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES=N` | unset (unlimited) | Bound the sum of conservative major-image reservations across code objects currently being transformed. Each reservation is the maximum of the explicitly modeled incremental-patch, composite-patch, and final-validation ownership phases described below. All inventory and retry passes for one reader share it. `N` is unsigned decimal; zero rejects every nonempty transform. A rejected fail-open load bypasses transformation, runs the original object, and makes the final analysis verdict incomplete. Fail-closed mode and `RJ_CONSAN_REQUIRE_PATCH=1` instead reject the load because applicability is not yet known. |
| `RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES=N` | unset (unlimited) | Bound the live aggregate of full retained replacement images. `N` is unsigned decimal; zero rejects every nonempty replacement, including a no-growth rewrite. Bytes are released when a replacement load fails or its executable is destroyed. Retained ownership and its charge survive hook unload/reload because unload does not quiesce runtime loads or invalidate existing executables. |
| `RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES=N` | unset (unlimited) | Additionally bound the live aggregate of alignment-inclusive ELF growth across retained replacement code objects in this process. `N` is unsigned decimal; zero permits only no-growth replacements. Growth is released when a replacement load fails or its executable is destroyed and, like full-image ownership, survives hook unload/reload. In fail-open mode, an over-budget code object runs uninstrumented and makes the final analysis verdict incomplete; use fail-closed mode when every applicable object must be instrumented. |
| `RJ_CONSAN_MAX_PATCHES=N` | `65536` | Expert static patch cap. Setting it explicitly makes omitted admitted sites an intentional coverage limit; ordinary runs use the implementation-sized all-supported allowance. |
| `RJ_CONSAN_DUMP_DIR=PATH` | unset | Write original and transformed `.hsaco` objects for inspection. |

## Generate and use a kernel allowlist

Use an allowlist to instrument the kernels your workload actually dispatches.
This is especially useful with large library binaries: ConSan can skip unrelated
code objects before decoding instructions or planning patches, reducing startup
latency and transformation memory use.

### 1. Profile the workload without ConSan

Use `rocprofv3` from the same ROCm installation as the application. Run the same
command, inputs, shapes, and execution stages that you will run under ConSan.
Include any setup or warm-up that selects or generates kernels.

Create a fresh trace directory so traces from another run cannot enter the list:

```sh
profile_dir="$(mktemp -d "$PWD/consan-profile.XXXXXX")"

env -u HSA_TOOLS_LIB \
  rocprofv3 --kernel-trace --output-format csv \
  --output-directory "$profile_dir" -- ./application
```

Removing `HSA_TOOLS_LIB` from this command's environment prevents an exported
ConSan hook setting from instrumenting the discovery pass.

### 2. Convert the trace into exact kernel names

```sh
rocjitsu_consan_allowlist.py \
  --output "$PWD/consan-kernels.txt" "$profile_dir"
```

The converter accepts trace CSV files or output directories. It writes each
unique dispatched kernel name on its own line and reports how many names it
wrote. It rejects missing or malformed traces and traces with no GPU dispatches;
resolve that error before proceeding.

Keep this file with the trace. Regenerate it when the workload or software stack
changes which kernels run. If a later run takes an unprofiled path, profile that
path too; do not append guessed kernel names.

### 3. Run ConSan with the generated file

Use the same application command as in step 1. `CONSAN_HOOK` is the hook path
set in [Build and load the hook](#build-and-load-the-hook).

```sh
env -u RJ_CONSAN_KERNEL_ALLOWLIST \
  HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_MODE=record-replay \
  RJ_CONSAN_LOG=1 \
  RJ_CONSAN_KERNEL_ALLOWLIST_FILE="$PWD/consan-kernels.txt" \
  ./application 2> "$PWD/consan.log"
```

Change `RJ_CONSAN_MODE` to use another analysis mode. Reuse the same generated
file when comparing modes. The command clears the inline allowlist variable
because the inline and file controls are mutually exclusive.

### 4. Check that the selected kernels were exercised

At normal unload, look in `consan.log` for one `ConSan kernel allowlist entry`
record per requested name. Check its `loaded`, `instrumented`, `dispatches`,
`visible_records`, and `status` fields:

| Observation | What to check next |
| --- | --- |
| `status=not-loaded` | Confirm that the ConSan run used the profiled workload and configuration. |
| `status=loaded-not-instrumented` | Inspect the instrumentation diagnostics. This entry contributed no supported instrumentation. |
| `status=instrumented-not-dispatched` | Exercise the expected workload path, or regenerate the list for the path you actually intend to test. |
| `dispatches` is nonzero but `visible_records=0` | Review the MOI zero-record diagnostic and runtime sampling settings before drawing a coverage conclusion. |

For each code object with no matching entry, the hook also logs
`outcome=skipped reason=no-matching-entry` immediately. These skip records remain
available even if the process does not reach normal unload.

### Select a small list by hand

Names match exact HSA kernel entries; the `.kd` suffix is optional. For example:

```sh
unset RJ_CONSAN_KERNEL_ALLOWLIST_FILE
export RJ_CONSAN_KERNEL_ALLOWLIST=attention_fwd,attention_bwd.kd
```

This selects `attention_fwd` and `attention_bwd`, but not `attention_fwd_debug`.
Use the file form for long or generated lists and for demangled names containing
commas.

### Understand the scope of the checks

- Code objects with no matching entry load unchanged. The hook checks their
  bounded ELF kernel-symbol index before waitcheck or ConSan semantic inventory;
  it skips instruction decoding, inventory construction, transform-memory
  reservation, and patch planning for definitively unmatched objects.
- A malformed or unsupported symbol index cannot prove that no entry matches.
  Such an object follows the ordinary conservative analysis path.
- Kernel-local sites belonging to unlisted entries remain untouched. A site in
  a shared helper is instrumented only if **every** kernel entry that can reach
  it is selected. Selecting only some owners can therefore reduce static
  coverage; include all owners if you need that helper instrumented.

## Bound instrumentation memory

The three process controls are independent. The concurrent-transform control is
acquired before semantic inventory. It is a conservative admission unit for
major ELF and parser storage, not a strict RSS limit: allocator bookkeeping,
other non-image analysis state, and unrelated process memory are outside the
model.

Let `I` be the original input size and `M` be `I` plus the configured per-object
maximum growth. The modeled phases are `I + 12*M` for an ordinary incremental
patch, `I + 13*M` while independently validated composite mutation storage
remains live, and `9*I + 10*M` during final validation.

Each parser has eight major-image units: its image, up to two units for section
objects and their vector slots, bounded payload, bounded section names, and up
to three units for compact section classification plus symbol- and
kernel-metadata-derived state.

That final budget charges each newly retained role for its copied name plus
conservative aggregate kernel/function record and container state; roles sharing
one logical symbol name share overlapping transient-state charges. It also
charges retained kernel metadata map entries before insertion and vector
capacity beyond the requested entry count. Section classification is one bit per
section and keeps symbol lookup linear in the ELF section and symbol counts.
Classification, symbol, and metadata state share that three-unit budget rather
than receiving separate allowances, so metadata can reject an object already at
the symbol boundary. This tightens parser admission without changing the
eight-unit parser coefficient or the `12*M`, `13*M`, and `9*I + 10*M` phase
coefficients.

Section headers, transient symbol-name characters, and metadata names are views
into the image rather than duplicate owning collections. Repeated kernel names
within one AMDGPU metadata note retain the final record, while an earlier note
keeps precedence over later notes, matching the parser's existing merge rules.
Metadata note walking separately charges both planned payload passes against
four image-sized units of work, so overlapping or repeated program-header
references cannot multiply parser time without limit. Reserved-range ELF symbol
section indices do not resolve against oversized directly encoded section
tables.

Admission uses the largest phase value and reports the governing phase and
coefficients. The parser rejects aggregate copied section payload or
section-name bytes larger than its backing image, and conservatively charged
symbol- and metadata-derived state larger than its three-unit budget.

Deployments with a tuned concurrent-transform ceiling should derive it from the
reported phase and current coefficients. The patcher preallocates every file
insertion, commits same-size rewrites directly, moves every emitted image, and
avoids a separate padding buffer so vector growth cannot add an unmodelled
geometric full-image allocation.

The two retained- image controls are charged together after transformation, when
the exact replacement size is known: one counts the full image and the other
counts only its growth delta. Admission and ownership are one transaction, so
failure of either retained budget commits neither charge. Failed
replacement-reader creation or loading releases every local storage owner before
refunding the retained charge or invoking a fallback loader.

Unload starts a new peak- reporting interval without releasing live transform
charges or retained replacement ownership; the latter remains until an
executable destruction observed while the hook is active, or process exit. New
HSA API calls made after the tool's `OnUnload` callback and before a synthetic
reinstall are outside the hook lifetime and cannot be reconciled on reload.
Teardown reports the live and peak values for all three controls, including
baseline runs where the ceilings are unlimited.

The absolute and percentage growth variables are mutually exclusive. A
growth-policy rejection reports the exact alignment-inclusive bytes required,
the effective total limit, and the selected policy. Successive ConSan stages
share that one original-image budget rather than receiving independent limits.

## SuperCollider controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_SC_REPORT_MODE=auto|trap` | `auto` | `auto` owns a non-trapping sticky marker per relevant code object. `trap` is an expert process-disrupting mode. |
| `RJ_CONSAN_REPORT_BUFFER=0xADDR` | unset | Use a caller-owned device-visible 32-bit marker instead of automatic allocation. |
| `RJ_CONSAN_REPORT_MARKER=N` | `1` | Value written on mismatch. |
| `RJ_CONSAN_DELAY=N` | `0` | Delay parameter between the guest access and duplicate/read-back. |
| `RJ_CONSAN_DELAY_MODE=nop|sleep|sleep_var` | `nop` | Select `s_nop`, `s_sleep`, or `s_sleep_var` delay lowering. |
| `RJ_CONSAN_DELAY_VAR_SSRC=N` | `106` | Scalar source encoding used by `sleep_var`. |
| `RJ_CONSAN_CHECK_TRAP_MODE=all|lds|flat` | `all` | Restrict SuperCollider to native DS or admitted flat LDS sites for debugging. |

The automatic marker reports that at least one duplicated/read-back value
differed. It does not identify an address, lane, value, or happens-before
violation. A race-free program can advance another wave between the original
and repeated access, so compare repeated known-correct and suspect runs.

SuperCollider timing perturbation is a validation-only composition mechanism,
not part of an ordinary detection run:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_SC_PERTURB_KIND=none|barrier|atomic` | `none` | Select one synchronization family to delay. |
| `RJ_CONSAN_SC_PERTURB_EDGE=release|acquire` | `release` | Select the ordering edge of an atomic perturbation. |
| `RJ_CONSAN_SC_PERTURB_IDENTITY=IDENTITY` | unset | Select the reviewed semantic sequence identity. |
| `RJ_CONSAN_SC_PERTURB_INDEX=N` | `0` | Zero-based diagnostic selector when an identity is not supplied. Prefer identity. |
| `RJ_CONSAN_SC_PERTURB_MAX=N` | `1` | Bound the number of selected perturbations. |
| `RJ_CONSAN_SC_PERTURB_SLEEP=N` | `1` | Sleep immediate used by the perturbation. |
| `RJ_CONSAN_SC_PERTURB_REQUIRED_COUNT=N` | `0` | Require exactly this many selected perturbations when nonzero. |

As with fault injection, inventory and review an exact identity before using a
live perturbation in a qualification campaign.

## MOI report buffers

With no caller-owned buffer, the hook first inventories the final transformed
code and computes the exact report layout required by the selected ordinary
engine:

- Record/Replay reserves admitted static access ranges and enabled barrier,
  atomic, fence, and diagnostic regions.
- Sampled reserves admitted logical ranges, bounded window banks,
  synchronization metadata, and pending-acquire state.
- Inline Shadow reserves one versioned exact-shadow slot per four-byte cell in
  the maximum declared LDS span of the owners plus the enabled diagnostic and
  ordering regions.

The automatic allocator requests the exact planned bytes. It never silently
shrinks site coverage or disables an event kind to fit. Sampled and Inline
Shadow retain a 128 MiB per-buffer ceiling. Record/Replay permits up to 1 GiB
per buffer so generated libraries retain bounded dispatch and owner diversity
across benchmark loops; aggregate live automatic-report memory remains bounded
at 4 GiB per process. Arithmetic overflow, a ceiling violation, or allocation
failure is a typed incomplete outcome.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=N` | Engine ceiling (1 GiB for Record/Replay; 128 MiB otherwise) | Expert cap for HSA-tool-owned allocation; ordinary inventory still requests exact bytes below the cap. `0` disables automatic allocation. Dynamic access append requires an explicit finite cap. |
| `RJ_CONSAN_MOI_REPORT_BUFFER=0xADDR` | unset | Caller-owned device-visible report buffer. |
| `RJ_CONSAN_MOI_REPORT_BUFFER_SIZE=N` | `0` | Size of the caller-owned buffer; layout requirements depend on the engine and enabled event families. |
| `RJ_CONSAN_MOI_REQUIRE_RECORDS=0|1` | `0` | At unload, require some visible auto-buffer access, synchronization, shadow, or sampled evidence. |
| `RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=0|1` | `0` | Require at least one Inline, replay, or sampled diagnostic/conflict. |
| `RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=0|1` | `0` | Require zero diagnostics/conflicts. |
| `RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT=0|1` | `0` | Record/Replay-only positive guard. |
| `RJ_CONSAN_MOI_FORBID_OVERFLOW=0|1` | `0` | Fail if evidence was truly dropped. Sampled bounded saturation is reported separately from loss. |

The unload summary reports required and allocated bytes, per-region capacities,
current and peak live bytes, private spill growth, Inline LDS shadow size,
Sampled banks, saturation, undercoverage, overflow, and drops.

### Repeated synchronized work

Repeated synchronized work normally requires no ConSan-specific API. Automatic
MOI reports are bounded epoch storage: ConSan tracks completion signals for
instrumented dispatches, or for ordered barriers that cover signal-less
dispatches, and observes ordinary HSA waits. Once all tracked instrumented work
is quiescent, it analyzes and recycles the current report epoch automatically.
Normal HIP and PyTorch launch-and-synchronize loops therefore keep working as
their repetition count grows.

By default, the automatic transaction snapshots and analyzes every live report
before it clears any of them. It preserves each allocation address, layout,
reader, and generation embedded in instrumented code. Record/Replay access and
dispatch tables, Sampled causal windows/publication state, and Inline Shadow
ownership, ordering, and diagnostic state are all epoch-local and recycled.
Final unload combines every analyzed epoch with the last live epoch, so earlier
conflicts, diagnostics, saturation, and evidence are neither forgotten nor
counted twice.

For a long repeated workload where analyzing every iteration is unnecessary,
`RJ_CONSAN_MOI_EPOCH_ANALYSIS` selects which synchronized epochs receive the
full host snapshot/decode/analyze pipeline:

| Value | Selected synchronized epochs |
| --- | --- |
| `every` | Every epoch (default). |
| `nth:N` | Only epoch `N`. |
| `periodic:N` | Epochs `N`, `2N`, `3N`, and so on. |
| `periodic:N:OFFSET` | `OFFSET`, then every `N` epochs; `1 <= OFFSET <= N`. |
| `manual` | Epochs completed while an explicit analysis window is open. |

Unselected epochs are still instrumented on the GPU. At a proven quiescence
boundary, ConSan validates and resets their bounded report storage without
copying, decoding, analyzing, or rendering it. Consequently, diagnostics in an
unselected epoch do not contribute to the final verdict. The default remains
`every` when complete per-iteration analysis is required.

Manual selection is intended for harnesses that know the semantic operation
boundaries. After setting `RJ_CONSAN_MOI_EPOCH_ANALYSIS=manual`, resolve and
call these exports from the loaded `HSA_TOOLS_LIB`:

```c
uint32_t rj_dbi_consan_begin_epoch_analysis_window();
uint32_t rj_dbi_consan_end_epoch_analysis_window();
```

Open the window before submitting the first operation to analyze and close it
after its final synchronization. Every synchronized epoch completed within the
window is analyzed. Nesting, closing an unopened window, or calling either API
under a non-manual policy returns status `3`; `0` means success, `1` means the
hook is inactive, and `2` means the selected mode is not an MOI engine.

#### Expert fallback for externally synchronized work

A custom runtime may submit an instrumented dispatch without a trackable
completion signal, or may establish quiescence through a mechanism that ConSan
cannot observe. ConSan conservatively disables automatic recycling for that
epoch rather than guessing that it is safe. Such a runtime may explicitly
checkpoint after it has independently established device-wide quiescence:

```c
// First synchronize every queue that can write ConSan report state.
hipDeviceSynchronize();

// Resolve this exported symbol from the loaded HSA_TOOLS_LIB and call it.
uint32_t status = rj_dbi_consan_checkpoint_after_device_synchronize();
```

The synchronization is a mandatory precondition, not an operation performed by
the API. Calling while any queue can still write a report is invalid. The
operation is all-or-nothing across live reports: a snapshot failure leaves all
device evidence untouched and retryable. Status values are `0` for a completed
MOI checkpoint, `1` when the hook is inactive, `2` for the intentional
SuperCollider no-op, and `3` when a report snapshot or decode failed. Treat any
status other than `0` or the expected SuperCollider `2` as an error.

SuperCollider does not use the MOI report layout. Its mismatch marker is
lifetime-sticky and capacity-independent, so status `2` leaves it unchanged.

## Sampled presets

`RJ_CONSAN_MOI_SAMPLED_PRESET=low|default|high|max` selects a performance/coverage
trade-off for Sampled. Unset, empty, and `default` preserve the existing behavior.
Explicit nonempty presets require the Sampled engine; invalid names are rejected.
Names are case-insensitive.

| Preset | Workgroup stride | LDS-cell stride | Intended use |
| --- | ---: | ---: | --- |
| `low` | 1024 | 1024 | Try reducing recording overhead on large workloads, accepting more misses. |
| `default` | 256 | 256 | Existing standard settings, unchanged. |
| `high` | 1 | 4 | Small/minimized repros: retain every workgroup and one in four four-byte LDS cells. |
| `max` | 1 | 1 | Remove workgroup and cell filtering when `high` misses an issue. |

For example:

```sh
HSA_TOOLS_LIB="$CONSAN_HOOK" RJ_CONSAN_MOI_SAMPLED_PRESET=high ./application
```

Every preset keeps all eligible static sites, offset zero, automatic banks (up
to eight per logical range), the 128 MiB report ceiling, synchronization tracking,
every-epoch analysis, and the existing diagnostic-example limits. These are
selector defaults, not a percentage of all dynamic accesses or guaranteed
performance ratios. The `high` setting covers the aligned-address minimized
intra-wave and cross-wave store repros; it can miss races on other cell residues.
`max` removes that cell filter but retains bounded windows and the existing
limitations of the analysis. It is not exhaustive tracing.

Explicit knobs override preset defaults. Independent workgroup/cell settings
replace only the specified fields; unspecified fields inherit the preset.
Either legacy `RUNTIME_SAMPLE_*` variable selects coupled operation instead:
the preset's workgroup stride is the fallback for both selectors. Mixing explicit
legacy and independent selectors remains an error. Offsets must fit the resolved
stride. All other knobs retain their existing meanings and can also be overridden.
`RJ_CONSAN_LOG=1` shows the preset and the resolved configuration.

## MOI event and sampling controls

Defaults below assume the `default` preset.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_MOI_TRACK_BARRIERS=0|1` | `1` for every MOI engine | Track admitted barrier events. Explicit `0` is an expert compatibility override. |
| `RJ_CONSAN_MOI_TRACK_ATOMICS=0|1` | `1` for every MOI engine | Track admitted atomic/fence ordering evidence. Explicit `0` is an expert compatibility override. |
| `RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=0|1` | `0` | Record/Replay per-lane dynamic append. This is bounded expert tracing, not an exhaustive ordinary contract. |
| `RJ_CONSAN_MOI_SAMPLE_STRIDE=N` | `1` | Sampled static site stride; this removes nonselected sites and therefore limits declared coverage. |
| `RJ_CONSAN_MOI_SAMPLE_OFFSET=M` | `0` | Static residue, smaller than the static stride. |
| `RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE=N` | `65,536` for Record/Replay; `256` for Sampled; `1` for Inline Shadow | Expert power-of-two runtime stride in `1..16777216`; leaves all eligible static sites patched. |
| `RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET=M` | `0` | Expert runtime residue smaller than the runtime stride. |
| `RJ_CONSAN_MOI_WORKGROUP_SAMPLE_STRIDE=N` | `256` | Sampled-only independent workgroup stride; power of two in `1..16777216`. Applies consistently to access and synchronization probes. |
| `RJ_CONSAN_MOI_WORKGROUP_SAMPLE_OFFSET=M` | `0` | Residue smaller than the workgroup stride. |
| `RJ_CONSAN_MOI_CELL_SAMPLE_STRIDE=N` | `256` | Sampled-only independent LDS-cell stride; power of two in `1..16777216`. Selects `(byte_address >> 2) % stride`, keeping all waves that touch a selected cell. |
| `RJ_CONSAN_MOI_CELL_SAMPLE_OFFSET=M` | `0` | Residue smaller than the cell stride. |
| `RJ_CONSAN_MOI_SAMPLED_BANKS=N` | `0` (auto) | Requested maximum banks per logical range: `0`, `1`, `2`, `4`, or `8`. Auto requests eight; report memory limits may reduce the achieved geometry. Bank routing is unchanged. |
| `RJ_CONSAN_MOI_SAMPLED_CHECK=0|1` | `0` | Enable the lower-fidelity immediate adjacent-range GPU check in addition to host scanning. |
| `RJ_CONSAN_MOI_SAMPLED_CONFLICT_LIMIT=N` | `8` | Maximum distinct conflict examples retained per Sampled report (`0..1024`). Zero suppresses examples, not analysis or conflict counting. |
| `RJ_CONSAN_MOI_SAMPLED_TOTAL_CONFLICT_LIMIT=N` | `64` | Maximum Sampled conflict examples across all reports, executables, and epochs of one hook session (`0..65536`). |

With the `default` preset and no independent selector variables, Sampled keeps
the legacy coupled runtime stride/offset for both workgroups and LDS cells. Setting any
`WORKGROUP_SAMPLE_*` or `CELL_SAMPLE_*` variable enables independent selection:
unspecified strides inherit the preset (256 with `default`) and offsets default to zero. Mixing these variables
with either legacy `RUNTIME_SAMPLE_*` variable is rejected, even if the values
agree. For example, workgroup stride 1 and cell stride 256 selects every
workgroup while retaining only the chosen LDS-cell residue. Synchronization
selection follows the workgroup setting, never the cell setting.

With `RJ_CONSAN_LOG=1`, the Sampled configuration line includes both resolved
selectors, static-site selection, requested banks, report ceiling, analysis
policy, and example limits. Per-object planning and static mapping lines show
the achieved slots and bank geometry (`effective_banks_min`/`effective_banks_max`
for admitted static mappings); a request is not a capacity guarantee.


Sampled conflict examples contain the code-object fingerprint, both original
instruction offsets, dispatch/workgroup identity, owners, access kinds, and LDS
byte ranges. An unavailable instruction mapping is printed as `unavailable`,
which is distinct from a mapped offset of zero. Attribution does not depend on
the separate 64-entry watchpoint detail listing.

Sampled also reports single-instruction write collisions among lanes when it
can prove one exact LDS byte interval for all participating lanes. The current
proof recognizes plain constant/scalar broadcasts and VGPR copies within one
basic block, invalidates clobbered registers, and discards facts on EXEC changes
and CFG edges. Only ordinary single-range native LDS forms qualify; FLAT,
multi-range accesses, lane-permuting moves, and gfx1250 selectable VGPR banks
remain unsupported for exact masks. Objects that write register mode through
`s_setreg*` or `s_set_gpr_idx*` also receive no exact masks; relative VGPR moves
clear local proof facts. These restrictions affect lane evidence,
not the existing cross-wave access checking.

For a qualifying retained access, `first_lanes` and `second_lanes` show exact
participating masks. Otherwise that side prints `unavailable`. A single-group
write diagnostic splits one mask into its first lane and remaining lanes; it
counts one conflict example rather than enumerating every lane pair. Read-only
and atomic groups do not create these ordinary write/write diagnostics.
Identical addresses and values do not make concurrent non-atomic writes race-free.
By default, Sampled and Record/Replay diagnose same-instruction lane collisions,
including stores whose address and every data word are proven uniform.

`RJ_CONSAN_MOI_ALLOW_PROVABLY_SAME_VALUE_WRITE_RACES=1` explicitly opts into suppressing this
common pattern. The default is `0`; sampling presets do not enable it. The opt-in
applies only to one ordinary native LDS store when static analysis proves both
an identical address and identical data across its participating lanes. The proof
checks every stored data register using local scalar/constant broadcasts and
copies, and is discarded on clobbers, EXEC changes, or CFG boundaries. Unknown or
differing values retain the diagnostic. This is a diagnostic suppression policy,
not a claim that equal-value writes are synchronized.

The store evidence remains available for cross-wave checking, including conflicts
with reads and identical-valued writes from other waves. The proof adds no GPU
value capture or comparison and does not enlarge report records. Record/Replay
also supports this proof on gfx1250 objects without selectable-bank transitions
or indirect transfers; unrelated WAVE_MODE writes that do not touch VGPR_MSB are
allowed. Sampled's gfx1250 exact-lane limitation remains unchanged. Existing
sampling still determines whether any evidence is retained.

Report ABI version 13 adds an eight-byte exact mask to each causal window,
protected by the same publication claim and report generation as the access.
Automatic Sampled layouts no longer reserve unused device diagnostic records;
diagnostics are built on the host. Direct callers must size their buffers with
the current layout helpers. A fixed byte budget determines capacity using the
new record size; use the reported achieved bank count when comparing runs.

Examples are deduplicated within each report by the two sites (or slot indices
when unmapped), execution identity, owners, epochs, access kinds, and byte ranges.
Traversal order determines which examples are retained. The total
`sampled_conflicts` counts cross-wave evidence pairs plus one conflict per
retained exact group with colliding lanes, including repeated examples. `sampled_conflict_examples` counts retained examples and
`sampled_conflict_pairs_without_example` counts the remaining pairs, including
duplicates and pairs omitted by either limit. Reaching an output limit does not
stop analysis or change the diagnostic guards' verdict.

Record/Replay runtime selection retains whole workgroups. Its per-probe gate
mixes the exact x/y/z workgroup coordinates and the CDNA5 cluster coordinate
when present, but deliberately does not rotate selection by dispatch identity.
Offset zero therefore includes workgroup zero across repeated launches.
Dispatch identity remains part of each retained record and host-replay key.

Sampled uses the same dispatch/workgroup vocabulary. Its workgroup selector
includes dispatch identity, so offset zero does not guarantee selecting
workgroup zero, and repeated processes can select different workgroups at the
same offset. A fixed offset schedule reproduces selector settings, not dispatch
identities. Workgroup stride 1 removes this source of selection misses. When entry-captured
identity and scalar resources permit, a fast gate skips the shared access body
for an unselected workgroup; private-identity and compact-spill operating points
use an in-body fallback. Selected workgroups additionally mix owner, epoch,
persistent sequence, static site, and LDS-cell identity into bounded causal
windows. Each logical range receives as many as eight immutable windows when
capacity permits. A later valid identity after every bank fills is
`sampled_saturated_windows`; malformed publication or true evidence loss uses
separate counters and makes the analysis incomplete.

Sampled diagnostics attribute the retained instruction, wave owner and byte
range. Exact masks and intra-wave diagnostics are limited to the locally proven
uniform-address cases described above. Other windows retain representative
accesses; dense sampling or additional banks do not recover missing lane provenance.

A retained conflict prints `ambiguous` when overlapping static mappings name
different original instructions for its slot. Missing mappings print
`unavailable`; neither case is presented as instruction offset zero.

### Repeatable offset sweeps

The [Sampled sweep runner](../../tests/dbi/consan/consan_sampled_sweep.py) runs an
unchanged command with a fixed, bounded schedule. For example, from
`emulation/rocjitsu`:

```sh
python3 tests/dbi/consan/consan_sampled_sweep.py \
  --hook /path/to/librocjitsu_dbi_hooks.so --output /tmp/sampled-investigation \
  --workgroup-stride 256 --workgroup-offsets 0,1 \
  --cell-stride 256 --cell-offsets 0,1,2,3 --banks 0 --run-budget 8 \
  --timeout 180 -- ./workload its-arguments
```

The workgroup offsets are the outer loop and cell offsets the inner loop, in
supplied order. The budget selects a prefix of their Cartesian product; the
manifest records both requested and scheduled counts. The runner replaces
inherited runtime selectors, enables Sampled logging and fail-closed patching,
and preserves other controls, including static selection, report ceilings,
epoch policy and diagnostic assertions. Use a fresh output directory. Inherit
the same SDK/library environment used to build and run the workload.

Each run keeps its complete log, effective configuration, report geometry,
retention/saturation/mapping counters, completeness verdicts and diagnosed
pairs. `summary.json` aggregates references to already diagnosed pairs; it
never joins accesses from different executions. Missing hook evidence, a
nonzero exit or a timeout stops the sweep and remains a failure in the summary.
A run with no diagnostic remains an observed miss/clean result to interpret
against the workload's independent fault and correctness oracle. The tool does
not certify race freedom or change expected fault outcomes. Denser selectors,
more banks and additional runs request more investigation work; their elapsed
cost is recorded per run and is not an unchanged-default overhead measurement.

The `sampled-selection` CTest label exercises an explicit production-rate
positive, an unselected address, an ordered control and independent-offset
cases. These are separate from the ordinary dense semantic device matrix.

## Resource overrides

Register selection is automatic. Access, barrier, atomic, and diagnostic
probes use dead VGPRs, fresh descriptor-backed windows, or spill-preserved
windows as appropriate. Scalar windows are placed above decoded and metadata
ownership and preserve EXEC, VCC, and SCC. These variables are expert/debug
overrides, not ordinary setup:

- `RJ_CONSAN_TMP_VGPR`
- `RJ_CONSAN_MOI_EXEC_SAVE_SGPR`
- `RJ_CONSAN_MOI_OWNER_VGPR`
- `RJ_CONSAN_MOI_EPOCH_VGPR`
- `RJ_CONSAN_MOI_OWNER_SGPR`
- `RJ_CONSAN_MOI_OWNER_SOURCE=automatic|auto|workitem_id|hw_id` (defaults to
  `automatic`; resolves to resident-wave `hw_id` for Inline Shadow and to
  entry-captured `workitem_id` for Record/Replay and Sampled)
- `RJ_CONSAN_MOI_INIT_OWNER_EPOCH`

Overrides remain subject to ownership, alignment, liveness, overlap, and
descriptor checks. They cannot force an unsafe plan. For Record/Replay,
`RJ_CONSAN_MOI_OWNER_VGPR` and `RJ_CONSAN_MOI_EPOCH_VGPR` are one paired
override: setting only one is rejected as unsupported instead of silently
dropping instrumentation. Inline Shadow rejects an explicit `workitem_id`
owner because `workitem_id_x` alone cannot distinguish resident waves in
arbitrary multidimensional workgroups. Its `hw_id` path is supported on
gfx942, gfx950, gfx1100, gfx1201, and gfx1250. See
[SPILLING.md](SPILLING.md).

## Malformed-input guard

`RJ_CONSAN_ABORT_UNMATCHED_BARRIER_WAIT=1` is an opt-in destructive containment
guard. It replaces only a statically unique immediate wait which belongs to no
bounded same-owner barrier sequence with a terminal instruction and records an
`inline-malformed-barrier-abort` patch. Dynamic, ambiguous, and paired waits
are untouched. It is not enabled by ordinary flavors or engines and is not a
race diagnostic. See [MALFORMED_INPUT.md](MALFORMED_INPUT.md).

## Fault injection

Fault controls mutate final native code before ConSan instruments the result.
The public raw workflow has two steps:

1. enable one fault family with `RJ_CONSAN_FAULT_DRY_RUN=1` and
   `RJ_CONSAN_LOG=1`, then review the emitted `ConSan fault site` and `ConSan
   fault plan` records; and
2. repeat the run without `DRY_RUN`, select the reviewed
   `RJ_CONSAN_FAULT_SITE_IDENTITY`, and set
   `RJ_CONSAN_FAULT_REQUIRE_EXACTLY_ONE=1`.

The final `ConSan fault summary` must report
`requested=1 planned=1 applied=1`. Identities belong to the exact native binary
and must be rediscovered after a relevant rebuild. Prefer identities over
numeric index selectors.

Supported mutation families are:

| Family | Enabling control | Additional selection |
| --- | --- | --- |
| Drop a barrier | `RJ_CONSAN_FAULT_DROP_BARRIER=1` | Logical barriers may require exact sequence and companion identities. |
| Move a barrier | `RJ_CONSAN_FAULT_MOVE_BARRIER=1` | Set the direction and an exact suitable destination identity. |
| Change barrier ID/scope or participants | `RJ_CONSAN_FAULT_MUTATE_BARRIER_ID_SCOPE=1` or `RJ_CONSAN_FAULT_MUTATE_BARRIER_PARTICIPANTS=1` | Set the exact sequence and target ID/scope, count, or mask required by the selected form. |
| Change an atomic address | `RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS=1` | Declare a positive, four-byte-aligned, signed-24-bit `RJ_CONSAN_FAULT_ATOMIC_VALID_ADDRESS_DELTA` backed by valid padded storage. |
| Change an LDS address register | `RJ_CONSAN_FAULT_LDS_WRONG_ADDRESS=1` | Set `RJ_CONSAN_FAULT_LDS_ADDRESS_VGPR` to a distinct, already allocated and workload-initialized VGPR selected from the production kernel. |
| Weaken atomic ordering or scope | `RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER=1` or `RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE=1` | Order weakening can select `release`, `acquire`, or `any` through `RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE`. |
| Change an ordinary access | `RJ_CONSAN_FAULT_ORDINARY_WRONG_ADDRESS=1`, `RJ_CONSAN_FAULT_ORDINARY_WEAKEN_ORDER=1`, or `RJ_CONSAN_FAULT_ORDINARY_WEAKEN_SCOPE=1` | Wrong-address injection requires explicitly padded valid storage and its aligned delta. |

Barrier sequence, companion, and destination identities are printed by the
dry-run inventory when the selected family needs them. If an exact site occurs
in more than one code-object load, `RJ_CONSAN_FAULT_LOAD_OCCURRENCE=N` selects
the one-based occurrence for a live exactly-one run.

The complete fault-selection controls are:

| Variable | Use |
| --- | --- |
| `RJ_CONSAN_FAULT_SITE_IDENTITY` | Exact primary identity printed by dry run. |
| `RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY` | Exact logical barrier sequence containing the primary. |
| `RJ_CONSAN_FAULT_BARRIER_COMPANION_SITE_IDENTITY` and `RJ_CONSAN_FAULT_BARRIER_COMPANION_SEQUENCE_IDENTITY` | Paired identities required for grouped barrier drops; setting only one is invalid. |
| `RJ_CONSAN_FAULT_BARRIER_DESTINATION_IDENTITY` | Reviewed destination for a barrier move. |
| `RJ_CONSAN_FAULT_BARRIER_MOVE_DIRECTION=legacy-marker|earlier|later` | Direction/form of the move; new campaigns should use an explicit reviewed direction. |
| `RJ_CONSAN_FAULT_BARRIER_TARGET_ID` | New barrier ID/scope value; required by ID/scope mutation. |
| `RJ_CONSAN_FAULT_BARRIER_TARGET_PARTICIPANT_COUNT` or `RJ_CONSAN_FAULT_BARRIER_TARGET_PARTICIPANT_MASK` | New participant contract for the selected target form. |
| `RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE=any|release|acquire` | Ordering edge weakened by the atomic-order family. |
| `RJ_CONSAN_FAULT_ATOMIC_VALID_ADDRESS_DELTA` | Required positive, four-byte-aligned, signed-24-bit delta into explicitly valid padded storage. |
| `RJ_CONSAN_FAULT_ORDINARY_VALID_ADDRESS_DELTA` | Required positive aligned signed-24-bit delta into explicitly valid storage. |
| `RJ_CONSAN_FAULT_LDS_ADDRESS_VGPR` | Distinct allocated and initialized 8-bit VGPR index for LDS wrong-address mutation. |
| `RJ_CONSAN_FAULT_LOAD_OCCURRENCE` | Positive one-based matching reader load for an exactly-one run. |
| `RJ_CONSAN_FAULT_RESERVATION_TIMEOUT_MS` | Process-wide exactly-one reservation wait; default `30000`. |

`RJ_CONSAN_FAULT_BARRIER_INDEX`, `RJ_CONSAN_FAULT_ATOMIC_INDEX`,
`RJ_CONSAN_FAULT_LDS_INDEX`, and `RJ_CONSAN_FAULT_ORDINARY_INDEX` are accepted
zero-based diagnostic selectors. They are less stable than semantic identities
and should not be committed as campaign provenance.

Three explicitly destructive proof overrides exist for carefully reviewed
malformed controls:

- `RJ_CONSAN_FAULT_ALLOW_DESTRUCTIVE_INCOMPLETE_BARRIER_DROP`;
- `RJ_CONSAN_FAULT_ALLOW_COMPLETING_CONDITIONAL_BARRIER_MOVE`; and
- `RJ_CONSAN_FAULT_ALLOW_DESTRUCTIVE_DIVERGENT_BARRIER_MOVE`.

They relax specific mutation-safety proofs, not ConSan instrumentation safety.
Do not use them for ordinary detection or as a shortcut around a dry-run
review.

Variables named `RJ_CONSAN_TEST_*` and the low-level `RJ_CONSAN_PROBE_*`
mechanism controls are test interfaces, not supported user configuration, and
are intentionally omitted from this reference.

The LDS wrong-address family rewrites only the decoded DS address operand. It
does not grow a kernel's VGPR allocation or synthesize a register value: doing
either would turn a controlled workload mutation into undefined input. The
replacement must fit every execution owner's descriptor allocation, and a
checked-in control should record how production code initializes that register.

Live exactly-one mutations are reserved process-wide so concurrent code-object
loads cannot install the same fault twice. A contender waits up to 30000 ms by
default, matching the ordinary validation-process deadline, then loads without
the fault and reports
`reservation=contention-timeout`. Set
`RJ_CONSAN_FAULT_RESERVATION_TIMEOUT_MS` to a positive millisecond deadline
when a large transform or loader needs a longer bounded wait.

At process teardown, an exactly-one run emits `ConSan fault reservation
summary` with process-wide counts for `reserved`,
`mutation_already_installed`, `contention_timeout`, and
`reentrant_contention`. The fault runner retains these under
`mutation.reservation`, together with per-reader `not-requested` records. A
zero-attempt summary therefore means that no reader planned the selected
mutation; it is not interchangeable with a timeout or same-thread reentry.
`mutation-already-installed` is an expected bounded outcome when a later load
matches a mutation already installed elsewhere in the process. Timeout and
reentry counts reject campaign qualification even if another reader installed
one mutation successfully. Inspect `mutation.reservation.processes` to find
the affected process and increase the timeout only when the recorded transform
or loader window legitimately needs it. The required per-reader summary,
installation record, and process summary are emitted independently of
`RJ_CONSAN_LOG`. If a process initializes and shuts down the HSA tool more than
once, the parser aggregates its complete lifetime summaries under the same
process record and retains `summary_records`.

Reservation qualification requires a clean hook teardown. A process that
terminates through `_Exit`, abort, or another path that bypasses `OnUnload`
cannot attest complete process accounting; its retained reservation evidence
is invalid and the run must be repeated. Pair summaries report this as
`reservation_evidence_invalid`, separately from `fault_not_applied` and
`reservation_contended`. A workload/profile with no applicable mutation site
remains `unsupported`.

Fault injection is intentionally disruptive. Apply one mutation at a time,
use an external timeout, and check device health before and after the run.
Program corruption and a ConSan diagnostic are separate outcomes; a timeout,
signal, or lost device is not a detection.

## Coverage and diagnostics

At `RJ_CONSAN_LOG=1`, the important records are:

```text
ConSan patch end ... outcome=... patches=... modified=...
ConSan summary ... patches=... modified=...
ConSan coverage ... flavor=... engine=... access=... barrier=... atomic=... fence=...
ConSan analysis verdict ... static_complete=... dynamic_complete=...
ConSan MOI report memory required_bytes=... allocated_bytes=... peak_live_bytes=...
```

`RJ_CONSAN_LOG=3` additionally emits the machine-readable, per-site records
needed to reconcile those aggregate counts:

```text
ConSan coverage_site ... kind=... disposition=... outcome=... reason=... lowering_reason=... resource_reason=...
ConSan proof patch ... kind=... anchor=... trampoline=...
```

Detailed post-transform records retain their line-oriented format, but the
hook writes them in bounded batches. Large objects therefore do not pay one
global lock and multiple unbuffered writes for every site or patch.

For each event kind, the aggregate accounting is:

```text
discovered = supported + unsupported
supported = selected + expert_limit_omitted
selected = patched + resource_failed + placement_or_lowering_failed
```

An unsupported-only object remains applicable and incomplete. MOI emits one
typed `coverage_site` row per relevant final-code site. SuperCollider exposes
aggregate coverage but does not manufacture MOI per-site dispositions.

Interpret outcomes independently:

- `modified=true` proves replacement bytes were loaded, not that a race was
  found.
- Passing application correctness checks show that instrumentation preserved
  the result for that run, not that the program is race-free.
- An application output mismatch shows that a fault changed behavior, not that
  ConSan diagnosed it.
- A timeout, signal, or GPU reset is not a detection.
- A Sampled clean run or statistical miss is inconclusive about race freedom.

## Current boundaries

- Native instrumentation is architecture-specific; the code object must target
  a supported GPU (`gfx942`, `gfx950`, `gfx1100`, `gfx1201`, or `gfx1250`).
- ConSan is LDS/shared-memory focused. Selected atomics/fences provide ordering
  evidence; they are not general global-memory race instrumentation.
- Flat/generic LDS classification is conservative. `MaybeGroup` is heuristic;
  use `strict` when precision matters more than recall.
- SuperCollider reports redundant-access instability, not causality.
- Record/Replay is a bounded snapshot unless dynamic append is explicitly
  enabled, and dynamic append is still bounded by its finite report.
- Sampled is probabilistic and can miss races.
- Inline Shadow has bounded diagnostics and supported-form semantics, not
  unbounded tracing or complete ISA coverage.
- Ordinary VGPR spilling is target-specific; general SGPR and AccVGPR spilling
  are not available. Unsupported ownership or resource shapes fail explicitly.
