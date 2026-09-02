# ConSan usage

ConSan instruments final AMD GPU code objects through the rocJITsu HSA-tools
hook. It has native support for `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and
`gfx1250` and does not translate code objects between GPU architectures.

ConSan reads the active workgroup-LDS capacity from the runtime agent. It does
not hard-code a gfx942 LDS size; simulator and offline tests use the selected
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
Record/Replay; no separate enable variable is required. `RJ_CONSAN_LOG=1` is
optional.

Every valid code object on a waitcheck-supported target is checked for missing
AMDGPU waits at load time before ConSan allocates runtime state or runs its DBI
transform. Hazards and analysis failures are reported first, then ConSan still
instruments the suspect code so it can diagnose its memory-ordering behavior.
This preflight is part of the ConSan hook and is not controlled by
`ROCJITSU_WAITCHECK`, `ROCJITSU_WAITCHECK_MODE`, or
`ROCJITSU_WAITCHECK_FAIL`; ordinary ConSan runs do not load the separate
waitcheck HSA tool.

## Ordinary flavors and engines

ConSan exposes two top-level flavors. MOI contains three engines:

For the execution model behind these short descriptions, see the side-by-side
comparison in [FLAVORS.md](FLAVORS.md).

| Selection | Ordinary `standard-v1` behavior | Primary tradeoff |
| --- | --- | --- |
| default or `RJ_CONSAN_MODE=record-replay` | Instrument all admitted supported access, barrier, atomic, and fence sites; allocate an inventory-sized report; replay visible records on the host. | Mechanical default and expert synchronization engine with clear reference/debug semantics, but only a bounded dynamic snapshot. |
| `RJ_CONSAN_MODE=inline-shadow` | Publish exact-shadow cells and bounded diagnostics on the GPU; track admitted barriers and atomics. | Strongest supported-form attribution, with higher overhead. |
| `RJ_CONSAN_MODE=sampled` | Patch all admitted supported sites; use automatic runtime stride 16,384 and offset zero; retain bounded sampled causal windows and synchronization metadata. | Bounded retained state and probabilistic detection. |
| `RJ_CONSAN_MODE=supercollider` | Duplicate/read-back supported LDS accesses, delay, compare, and set an automatically allocated non-trapping mismatch marker. | Complementary value-instability diagnostic; it does not attribute a happens-before edge or exact racing pair. |

`RJ_CONSAN_MODE` defaults to `record-replay`; spelling it explicitly can make a
saved command self-describing.

Record/Replay's complete static-site instrumentation is not an exhaustive
dynamic trace. The ordinary automatic layout uses a report-wide dispatch
directory and a report-wide access-identity table with 2× open-addressing
headroom. The access table is sized from the admitted logical ranges and
adaptive dispatch/owner diversity factors; records retain the full dispatch,
static site, three-dimensional workgroup, and wave-owner identity. Either
table reaching its bounded 256-probe limit is a typed, report-wide
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

Current mechanical default, Record/Replay:

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

Sampled:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_MODE=sampled \
  RJ_CONSAN_LOG=1 \
  ./application
```

On gfx1201, the [empirical study](GFX1201_EMPIRICAL_STUDY.md) recommends
Sampled for ordinary barrier/LDS triage, with multiple runtime offsets when
confidence matters. Record/Replay remains the coded default until the public
supported-mode policy is updated.

SuperCollider:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_MODE=supercollider \
  RJ_CONSAN_LOG=1 \
  ./application
```

For focused tests, require complete instrumentation and report collection:

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
| `RJ_CONSAN_MODE=record-replay|inline-shadow|sampled|supercollider` | `record-replay` | Select the analysis. Loading the hook activates ConSan. |
| `RJ_CONSAN_POLICY=default|strict` | `default` | `strict` rejects unsupported or incomplete instrumentation, requires real patches and MOI evidence, and rejects report overflow. A load-time rejection terminates with exit code 92; race diagnostics remain nonfatal. |
| `RJ_CONSAN_LOG=N` | disabled | Enable compact logs at `1`; larger values add inventory detail. |
| `RJ_CONSAN_FAIL_CLOSED=0|1` | `0` | Reject unsupported/invalid transformation outcomes instead of loading the original. |
| `RJ_CONSAN_REQUIRE_PATCH=0|1` | `0` | Reject an applicable code object when no real access/barrier/atomic/fence instrumentation patch is emitted. Prologues and metadata-only changes do not satisfy it. |
| `RJ_CONSAN_KERNEL_ALLOWLIST=name[,name...]` | unset (all entries) | Instrument only the named HSA kernel entries. Names are exact, comma-separated, and may optionally include the `.kd` suffix. |
| `RJ_CONSAN_FLAT_PROVENANCE=likely|strict` | `likely` | Admit proven `Group` plus heuristic `MaybeGroup` flat LDS sites, or only proven `Group` sites. |
| `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES=N` | `419430400` (400 MiB) | Bound total additional ELF bytes, including alignment padding, across one code-object transformation. `N` is unsigned decimal; zero permits only no-growth rewrites. |
| `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT=N` | unset (absolute policy applies) | Use an alternative bound relative to the original code object: total additional ELF bytes may not exceed `floor(original input bytes * N / 100)`. `N` is unsigned decimal in `0..4294967295`; values above 100 intentionally allow growth larger than the original image. |
| `RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES=N` | unset (unlimited) | Bound the sum of conservative major-image reservations across code objects currently being transformed. Each reservation is the maximum of the explicitly modeled incremental-patch, composite-patch, and final-validation ownership phases described below. All inventory and retry passes for one reader share it. `N` is unsigned decimal; zero rejects every nonempty transform. A rejected fail-open load bypasses transformation, runs the original object, and makes the final analysis verdict incomplete. Fail-closed mode and `RJ_CONSAN_REQUIRE_PATCH=1` instead reject the load because applicability is not yet known. |
| `RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES=N` | unset (unlimited) | Bound the live aggregate of full retained replacement images. `N` is unsigned decimal; zero rejects every nonempty replacement, including a no-growth rewrite. Bytes are released when a replacement load fails or its executable is destroyed. Retained ownership and its charge survive hook unload/reload because unload does not quiesce runtime loads or invalidate existing executables. |
| `RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES=N` | unset (unlimited) | Additionally bound the live aggregate of alignment-inclusive ELF growth across retained replacement code objects in this process. `N` is unsigned decimal; zero permits only no-growth replacements. Growth is released when a replacement load fails or its executable is destroyed and, like full-image ownership, survives hook unload/reload. In fail-open mode, an over-budget code object runs uninstrumented and makes the final analysis verdict incomplete; use fail-closed mode when every applicable object must be instrumented. |
| `RJ_CONSAN_DUMP_DIR=PATH` | unset | Write original and transformed `.hsaco` objects for inspection. |

The kernel allowlist is a production scoping control, not a substring filter.
For example, `RJ_CONSAN_KERNEL_ALLOWLIST=attention_fwd,attention_bwd.kd`
selects exactly those two entries and does not select `attention_fwd_debug`.
Before waitcheck or ConSan semantic inventory, the hook checks only the bounded
ELF kernel-symbol index. Code objects containing no matching entry are loaded
unchanged without decoding executable instructions, constructing inventory,
reserving transform memory, or running patch planning. A malformed or
unsupported symbol index is not treated as proof of absence: it falls through
to the ordinary conservative analysis path. The hook logs an immediate
`outcome=skipped reason=no-matching-entry` record for every definitively
unmatched object, so scoping remains observable even if the process does not
reach normal unload reporting.
Kernel-local sites owned by unlisted entries remain untouched. A physical site
in a shared helper is instrumented only when every kernel entry that can reach
it is allowlisted; this avoids silently instrumenting an unselected dispatch.
Consequently, selecting only one owner of a shared helper can intentionally
report less static coverage than selecting all of its owners.

At unload, ConSan emits one `ConSan kernel allowlist entry` record per requested
name. Its `loaded`, `instrumented`, `dispatches`, and `visible_records` fields,
plus the typed `status`, distinguish an entry that was never loaded, one that
loaded but had no supported instrumentation, one that was instrumented but
never dispatched, and one that dispatched yet produced no visible evidence.
The last state should be interpreted together with the ordinary MOI zero-record
diagnostic and runtime sampling settings.

The three process controls are independent. The concurrent-transform control is
acquired before semantic inventory. It is a conservative admission unit for
major ELF and parser storage, not a strict RSS limit: allocator bookkeeping,
other non-image analysis state, and unrelated process memory are outside the
model. Let `I` be the original input size and `M` be `I` plus the configured
per-object maximum growth. The modeled phases are `I + 12*M` for an ordinary
incremental patch, `I + 13*M` while independently validated composite mutation
storage remains live, and `9*I + 10*M` during final validation. Each parser has
eight major-image units: its image, up to two units for section objects and
their vector slots, bounded payload, bounded section names, and up to three
units for compact section classification plus symbol- and
kernel-metadata-derived state. That final budget charges each newly retained
role for its copied name plus conservative aggregate kernel/function record and
container state; roles sharing one logical symbol name share overlapping
transient-state charges. It also charges retained kernel metadata map entries
before insertion and vector capacity beyond the requested entry count. Section
classification is one bit per section and keeps symbol lookup linear in the ELF
section and symbol counts. Classification, symbol, and metadata state share
that three-unit budget rather than receiving separate allowances, so metadata
can reject an object already at the symbol boundary. This tightens parser
admission without changing the eight-unit parser coefficient or the `12*M`,
`13*M`, and `9*I + 10*M` phase coefficients.
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
symbol- and metadata-derived state larger than its three-unit budget. These
coefficients supersede the earlier `I + 10*M`, `I + 11*M`, and
`7*I + 8*M` model, which did not fully account for section-object, vector-slot,
or dense per-symbol state. Deployments with a tuned concurrent-transform
ceiling should rederive it from the current coefficients. The patcher
preallocates every file insertion, commits same-size rewrites directly, moves
every emitted image, and avoids a separate padding buffer so vector growth
cannot add an unmodelled geometric full-image allocation. The two retained-
image controls are charged together after transformation, when the exact
replacement size is known: one counts the full image and the other counts only
its growth delta. Admission and ownership are one transaction, so failure of
either retained budget commits neither charge. Failed replacement-reader
creation or loading releases every local storage owner before refunding the
retained charge or invoking a fallback loader. Unload starts a new peak-
reporting interval without releasing live transform charges or retained
replacement ownership; the latter remains until an executable destruction
observed while the hook is active, or process exit. New HSA API calls made
after the tool's `OnUnload` callback and before a synthetic reinstall are
outside the hook lifetime and cannot be reconciled on reload. Teardown reports
the live and peak values for all three controls, including baseline runs where
the ceilings are unlimited.

For one transition, the old `RJ_CONSAN_FLAVOR`, `RJ_CONSAN_MOI_ENGINE`, and
`RJ_CONSAN_MOI_BACKEND` variables remain accepted with deprecation warnings.
Do not combine old selection variables with `RJ_CONSAN_MODE`.
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
Shadow retain a 128 MiB per-buffer ceiling. Record/Replay permits up to 512 MiB
per buffer so generated libraries retain bounded dispatch and owner diversity
across benchmark loops; aggregate live automatic-report memory remains bounded
at 1 GiB per process. Arithmetic overflow, a ceiling violation, or allocation
failure is a typed incomplete outcome.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=N` | Engine ceiling (512 MiB for Record/Replay; 128 MiB otherwise) | Expert cap for HSA-tool-owned allocation; ordinary inventory still requests exact bytes below the cap. `0` disables automatic allocation. Dynamic access append requires an explicit finite cap. |
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

## MOI event and sampling controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_MOI_TRACK_BARRIERS=0|1` | `1` for every MOI engine | Track admitted barrier events. Explicit `0` is an expert compatibility override. |
| `RJ_CONSAN_MOI_TRACK_ATOMICS=0|1` | `1` for every MOI engine | Track admitted atomic/fence ordering evidence. Explicit `0` is an expert compatibility override. |
| `RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=0|1` | `0` | Record/Replay per-lane dynamic append. This is bounded expert tracing, not an exhaustive ordinary contract. |
| `RJ_CONSAN_MOI_SAMPLE_STRIDE=N` | `1` | Sampled static site stride; this removes nonselected sites and therefore limits declared coverage. |
| `RJ_CONSAN_MOI_SAMPLE_OFFSET=M` | `0` | Static residue, smaller than the static stride. |
| `RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE=N` | `65,536` for Record/Replay; `16,384` for Sampled | Expert power-of-two runtime stride in `1..16777216`; leaves all eligible static sites patched. |
| `RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET=M` | `0` | Expert runtime residue smaller than the runtime stride. |
| `RJ_CONSAN_MOI_SAMPLED_CHECK=0|1` | `0` | Enable the lower-fidelity immediate adjacent-range GPU check in addition to host scanning. |

Record/Replay runtime selection retains whole workgroups. It mixes the exact
x/y/z workgroup coordinates and the gfx1250 cluster coordinate when present,
but not dispatch identity; offset zero therefore always includes workgroup
zero and selects the same workgroup coordinates across repeated launches.
Dispatch identity remains part of each retained record and host-replay key.

Sampled runtime selection mixes hardware dispatch identity, workgroup
coordinates, wave owner, epoch, persistent per-wave sequence, static site, and
LDS address. Each logical range receives as many as eight immutable windows
when capacity permits. A later valid identity after every bank fills is
`sampled_saturated_windows`; malformed publication or true evidence loss uses
separate counters and makes the analysis incomplete.

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
and must be rediscovered after a relevant rebuild. Prefer identities over the
legacy numeric index selectors.

Supported mutation families are:

| Family | Enabling control | Additional selection |
| --- | --- | --- |
| Drop a barrier | `RJ_CONSAN_FAULT_DROP_BARRIER=1` | Logical barriers may require exact sequence and companion identities. |
| Move a barrier | `RJ_CONSAN_FAULT_MOVE_BARRIER=1` | Set the direction and an exact suitable destination identity. |
| Change barrier ID/scope or participants | `RJ_CONSAN_FAULT_MUTATE_BARRIER_ID_SCOPE=1` or `RJ_CONSAN_FAULT_MUTATE_BARRIER_PARTICIPANTS=1` | Set the exact sequence and target ID/scope, count, or mask required by the selected form. |
| Change an atomic address | `RJ_CONSAN_FAULT_ATOMIC_WRONG_ADDRESS=1` | Declare a nonzero aligned `RJ_CONSAN_FAULT_ATOMIC_VALID_ADDRESS_DELTA` backed by valid padded storage. |
| Change an LDS address register | `RJ_CONSAN_FAULT_LDS_WRONG_ADDRESS=1` | Set `RJ_CONSAN_FAULT_LDS_ADDRESS_VGPR` to a distinct, already allocated and workload-initialized VGPR selected from the production kernel. gfx1250 two-address DS forms are excluded until their split relocation has an exact proof. |
| Weaken atomic ordering or scope | `RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER=1` or `RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE=1` | Order weakening can select `release`, `acquire`, or `any` through `RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE`. |
| Change an ordinary access | `RJ_CONSAN_FAULT_ORDINARY_WRONG_ADDRESS=1`, `RJ_CONSAN_FAULT_ORDINARY_WEAKEN_ORDER=1`, or `RJ_CONSAN_FAULT_ORDINARY_WEAKEN_SCOPE=1` | Wrong-address injection requires explicitly padded valid storage and its aligned delta. |

Barrier sequence, companion, and destination identities are printed by the
dry-run inventory when the selected family needs them. If an exact site occurs
in more than one code-object load, `RJ_CONSAN_FAULT_LOAD_OCCURRENCE=N` selects
the one-based occurrence for a live exactly-one run.

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
remains `unsupported` even when historical reservation evidence is absent.

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
