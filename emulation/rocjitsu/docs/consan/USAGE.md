# ConSan usage

ConSan instruments final AMD GPU code objects through the rocJITsu HSA-tools
hook. It has native support for `gfx942`, `gfx950`, `gfx1201`, and `gfx1250`
and does not translate code objects between GPU architectures.

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

Loading the hook enables ConSan and defaults to MOI Record/Replay.
`RJ_CONSAN_LOG=1` is optional.

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
| default or `RJ_CONSAN_MODE=record-replay` | Instrument all admitted supported access, barrier, atomic, and fence sites; allocate an inventory-sized report; replay visible records on the host. | **Recommended starting engine:** clear reference/debug semantics, but only a bounded dynamic snapshot. |
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

Recommended default, Record/Replay:

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
| `RJ_CONSAN_FLAT_PROVENANCE=likely|strict` | `likely` | Admit proven `Group` plus heuristic `MaybeGroup` flat LDS sites, or only proven `Group` sites. |
| `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES=N` | `201326592` (192 MiB) | Bound total additional ELF bytes, including alignment padding, across one code-object transformation. `N` is unsigned decimal; zero permits only no-growth rewrites. |
| `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT=N` | unset (absolute policy applies) | Use an alternative bound relative to the original code object: total additional ELF bytes may not exceed `floor(original input bytes * N / 100)`. `N` is unsigned decimal in `0..4294967295`; values above 100 intentionally allow growth larger than the original image. |
| `RJ_CONSAN_DUMP_DIR=PATH` | unset | Write original and transformed `.hsaco` objects for inspection. |

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
shrinks site coverage or disables an event kind to fit. The hard ceilings are
128 MiB per automatic buffer and 256 MiB live automatic-report memory per
process. Arithmetic overflow, a ceiling violation, or allocation failure is a
typed incomplete outcome.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=N` | 128 MiB ceiling | Expert cap for HSA-tool-owned allocation; ordinary inventory still requests exact bytes below the cap. `0` disables automatic allocation. Dynamic access append requires an explicit finite cap. |
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
| `RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE=N` | `16,384` for Sampled | Expert power-of-two runtime stride in `1..16777216`; leaves all eligible static sites patched. |
| `RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET=M` | `0` | Expert runtime residue smaller than the runtime stride. |
| `RJ_CONSAN_MOI_SAMPLED_CHECK=0|1` | `0` | Enable the lower-fidelity immediate adjacent-range GPU check in addition to host scanning. |

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
- `RJ_CONSAN_MOI_OWNER_SOURCE=workitem_id|hw_id` (defaults to `workitem_id`)
- `RJ_CONSAN_MOI_INIT_OWNER_EPOCH`

Overrides remain subject to ownership, alignment, liveness, overlap, and
descriptor checks. They cannot force an unsafe plan. For Record/Replay,
`RJ_CONSAN_MOI_OWNER_VGPR` and `RJ_CONSAN_MOI_EPOCH_VGPR` are one paired
override: setting only one is rejected as unsupported instead of silently
dropping instrumentation. See
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
| Weaken atomic ordering or scope | `RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER=1` or `RJ_CONSAN_FAULT_ATOMIC_WEAKEN_SCOPE=1` | Order weakening can select `release`, `acquire`, or `any` through `RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE`. |
| Change an ordinary access | `RJ_CONSAN_FAULT_ORDINARY_WRONG_ADDRESS=1`, `RJ_CONSAN_FAULT_ORDINARY_WEAKEN_ORDER=1`, or `RJ_CONSAN_FAULT_ORDINARY_WEAKEN_SCOPE=1` | Wrong-address injection requires explicitly padded valid storage and its aligned delta. |

Barrier sequence, companion, and destination identities are printed by the
dry-run inventory when the selected family needs them. If an exact site occurs
in more than one code-object load, `RJ_CONSAN_FAULT_LOAD_OCCURRENCE=N` selects
the one-based occurrence for a live exactly-one run.

Live exactly-one mutations are reserved process-wide so concurrent code-object
loads cannot install the same fault twice. A contender waits up to 30000 ms by
default, matching the ordinary validation-process deadline, then loads without
the fault and reports
`reservation=contention-timeout`. Set
`RJ_CONSAN_FAULT_RESERVATION_TIMEOUT_MS` to a positive millisecond deadline
when a large transform or loader needs a longer bounded wait.

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
ConSan coverage_site ... kind=... disposition=... outcome=... reason=... lowering_reason=... resource_reason=...
ConSan analysis verdict ... static_complete=... dynamic_complete=...
ConSan MOI report memory required_bytes=... allocated_bytes=... peak_live_bytes=...
```

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
  a supported GPU (`gfx942`, `gfx950`, `gfx1201`, or `gfx1250`).
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
