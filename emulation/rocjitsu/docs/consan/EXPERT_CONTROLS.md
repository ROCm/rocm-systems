# ConSan expert controls

Start with [USAGE.md](USAGE.md) for everyday runs, presets, and kernel
allowlists. This reference covers explicit overrides, resource limits, and
validation interfaces. Defaults apply unless a preset or explicit override
changes them.

- [Shared controls](#shared-controls): policy, memory limits, register overrides,
  and validation.
- [Default mode](#default-mode): event tracking, sampling, diagnostics, and reports.
- [SuperCollider controls](#supercollider-controls): mismatch reporting, delays,
  and perturbation.

## Shared controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_POLICY=default\|strict` | `default` | `strict` defaults fail-closed and require-patch guards to true; for the default mode it also defaults automatic-record and forbid-overflow guards to true. It does not require complete static coverage or make race diagnostics fatal. Configuration rejection or a load-time code-object rejection terminates with exit code 92. |
| `RJ_CONSAN_FAIL_CLOSED=0\|1` | `0` | Reject unsupported/invalid transformation outcomes instead of loading the original. |
| `RJ_CONSAN_REQUIRE_PATCH=0\|1` | `0` | Reject an applicable code object when no real access/barrier/atomic/fence instrumentation patch is emitted. Prologues and metadata-only changes do not satisfy it. |
| `RJ_CONSAN_FLAT_PROVENANCE=likely\|strict` | `likely` | Admit proven `Group` plus heuristic `MaybeGroup` flat LDS sites, or only proven `Group` sites. |
| `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_BYTES=N` | `939524096` (896 MiB) | Bound total additional ELF bytes, including alignment padding, across one code-object transformation. `N` is unsigned decimal; zero permits only no-growth rewrites. |
| `RJ_CONSAN_MAX_PATCHED_IMAGE_GROWTH_PERCENT=N` | unset (absolute policy applies) | Use an alternative bound relative to the original code object: total additional ELF bytes may not exceed `floor(original input bytes * N / 100)`. `N` is unsigned decimal in `0..4294967295`; values above 100 intentionally allow growth larger than the original image. |
| `RJ_CONSAN_MAX_PROCESS_CONCURRENT_TRANSFORM_BYTES=N` | unset (unlimited) | Bound the sum of conservative major-image reservations across code objects currently being transformed. Each reservation is the maximum of the explicitly modeled incremental-patch, composite-patch, and final-validation ownership phases described below. All inventory and retry passes for one reader share it. `N` is unsigned decimal; zero rejects every nonempty transform. A rejected fail-open load bypasses transformation, runs the original object, and makes the final analysis verdict incomplete. Fail-closed mode and `RJ_CONSAN_REQUIRE_PATCH=1` instead reject the load because applicability is not yet known. |
| `RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_BYTES=N` | unset (unlimited) | Bound the live aggregate of full retained replacement images. `N` is unsigned decimal; zero rejects every nonempty replacement, including a no-growth rewrite. Bytes are released when a replacement load fails or its executable is destroyed. Retained ownership and its charge survive hook unload/reload because unload does not quiesce runtime loads or invalidate existing executables. |
| `RJ_CONSAN_MAX_PROCESS_PATCHED_IMAGE_GROWTH_BYTES=N` | unset (unlimited) | Additionally bound the live aggregate of alignment-inclusive ELF growth across retained replacement code objects in this process. `N` is unsigned decimal; zero permits only no-growth replacements. Growth is released when a replacement load fails or its executable is destroyed and, like full-image ownership, survives hook unload/reload. In fail-open mode, an over-budget code object runs uninstrumented and makes the final analysis verdict incomplete; use fail-closed mode when every applicable object must be instrumented. |
| `RJ_CONSAN_MAX_PATCHES=N` | `65536` | Expert static patch cap. Setting it explicitly makes omitted admitted sites an intentional coverage limit; ordinary runs use the implementation-sized all-supported allowance. |
| `RJ_CONSAN_DUMP_DIR=PATH` | unset | Write original and transformed `.hsaco` objects for inspection. |

### Strict policy and validation assertions

For focused tests, require an effective transform and report collection:

```sh
export RJ_CONSAN_POLICY=strict
```

Strict policy does not make race diagnostics fatal. Expert validation harnesses
with a predeclared expected result can add one of these assertions:

```sh
export RJ_CONSAN_FORBID_DIAGNOSTICS=1   # known-correct ConSan control
export RJ_CONSAN_REQUIRE_DIAGNOSTICS=1  # predeclared positive ConSan control
```

With a valid `RJ_CONSAN_POLICY=strict` setting, invalid ConSan configuration
also terminates during hook initialization with exit code 92 and a
`ConSan configuration rejected` diagnostic. Under the default policy,
configuration rejection warns that ConSan is not installed and the run is
unchecked, then allows the application to continue.

Do not enable both diagnostic guards. Strict policy can be too restrictive for
a broad application that loads helper code objects with no admitted sites. A
strict code-object rejection terminates at the loader boundary with the typed
`ConSan load rejection` diagnostic and exit code 92. This prevents HIP clients
that ignore an HSA load error from retaining a null kernel symbol and crashing
later during launch. Explicit `RJ_CONSAN_FAIL_CLOSED=1` under the default policy
continues to return the HSA error to callers that correctly handle it.

### Bound instrumentation memory

For large library binaries, first use a
[kernel allowlist](USAGE.md#generate-and-use-a-kernel-allowlist) to avoid
transforming code the workload does not use.

Set the byte limits in [Shared controls](#shared-controls) when you need to
bound transformation or retained-image memory. These are admission budgets,
not a strict process RSS limit. A rejected object runs uninstrumented under
fail-open policy and makes coverage incomplete; use fail-closed policy if that
is unacceptable. The absolute and percentage per-object growth limits are
mutually exclusive.

Use the hook's reported live/peak usage and rejection diagnostics when tuning
a limit. The accounting formulas and ownership lifetimes are documented in
[DESIGN.md](DESIGN.md#transform-memory-accounting).

### Resource overrides

Register selection is automatic. Access, barrier, atomic, and diagnostic
probes use dead VGPRs, fresh descriptor-backed windows, or spill-preserved
windows as appropriate. Scalar windows are placed above decoded and metadata
ownership and preserve EXEC, VCC, and SCC. These variables are expert/debug
overrides, not ordinary setup:

- `RJ_CONSAN_TMP_VGPR`
- `RJ_CONSAN_EXEC_SAVE_SGPR`
- `RJ_CONSAN_OWNER_VGPR`
- `RJ_CONSAN_EPOCH_VGPR`
- `RJ_CONSAN_OWNER_SGPR`
- `RJ_CONSAN_OWNER_SOURCE=automatic|auto|workitem_id|hw_id` (defaults to
  `automatic`, which resolves to entry-captured `workitem_id`)
- `RJ_CONSAN_INIT_OWNER_EPOCH`

Overrides remain subject to ownership, alignment, liveness, overlap, and
kernel-descriptor checks. Owner and epoch VGPRs are a paired override; setting
only one is unsupported.

ConSan's workitem-derived owner uses `workitem_id_x`, so it cannot distinguish
waves separated only in the y/z dimensions of a workgroup. For access-only
multidimensional checks, select `hw_id`; use `RJ_CONSAN_WATCHPOINT_BANKS=1024`
to retain sparse resident-wave IDs without low-bit bank collisions. This
increases report allocation. Hardware-ID owners do not support ConSan's
barrier/atomic ordering path. See [SPILLING.md](SPILLING.md).

### Malformed-input guard

`RJ_CONSAN_ABORT_UNMATCHED_BARRIER_WAIT=1` is an opt-in destructive containment
guard. It replaces only a statically unique immediate wait which belongs to no
bounded same-owner barrier sequence with a terminal instruction and records an
`inline-malformed-barrier-abort` patch. Dynamic, ambiguous, and paired waits
are untouched. It is not enabled by ordinary modes and is not a
race diagnostic. See [MALFORMED_INPUT.md](MALFORMED_INPUT.md).

### Fault injection

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
| `RJ_CONSAN_FAULT_BARRIER_MOVE_DIRECTION=legacy-marker\|earlier\|later` | Direction/form of the move; new campaigns should use an explicit reviewed direction. |
| `RJ_CONSAN_FAULT_BARRIER_TARGET_ID` | New barrier ID/scope value; required by ID/scope mutation. |
| `RJ_CONSAN_FAULT_BARRIER_TARGET_PARTICIPANT_COUNT` or `RJ_CONSAN_FAULT_BARRIER_TARGET_PARTICIPANT_MASK` | New participant contract for the selected target form. |
| `RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE=any\|release\|acquire` | Ordering edge weakened by the atomic-order family. |
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

## Default mode

### ConSan event and sampling controls

These controls apply to the default mode. Defaults below assume the `default`
preset.

Explicit knobs override preset defaults. Independent workgroup/cell settings
replace only the specified fields; unspecified fields inherit the preset.
Either legacy `RUNTIME_SAMPLE_*` variable selects coupled operation instead:
the preset's workgroup stride is the fallback for both selectors. Mixing explicit
legacy and independent selectors remains an error. Offsets must fit the resolved
stride. All other knobs retain their existing meanings and can also be overridden.
`RJ_CONSAN_LOG=1` shows the preset and the resolved configuration.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_TRACK_BARRIERS=0\|1` | `1` | Track admitted barrier events. Explicit `0` is an expert compatibility override. |
| `RJ_CONSAN_TRACK_ATOMICS=0\|1` | `1` | Track admitted atomic/fence ordering evidence. Explicit `0` is an expert compatibility override. |
| `RJ_CONSAN_SAMPLE_STRIDE=N` | `1` | ConSan static site stride; this removes nonselected sites and therefore limits declared coverage. |
| `RJ_CONSAN_SAMPLE_OFFSET=M` | `0` | Static residue, smaller than the static stride. |
| `RJ_CONSAN_RUNTIME_SAMPLE_STRIDE=N` | `256` | Expert power-of-two runtime stride in `1..16777216`; leaves all eligible static sites patched. |
| `RJ_CONSAN_RUNTIME_SAMPLE_OFFSET=M` | `0` | Expert runtime residue smaller than the runtime stride. |
| `RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE=N` | `256` | Independent workgroup stride; power of two in `1..16777216`. Applies consistently to access and synchronization probes. |
| `RJ_CONSAN_WORKGROUP_SAMPLE_OFFSET=M` | `0` | Residue smaller than the workgroup stride. |
| `RJ_CONSAN_CELL_SAMPLE_STRIDE=N` | `256` | Independent LDS-cell stride; power of two in `1..16777216`. Selects `(byte_address >> 2) % stride`, keeping all waves that touch a selected cell. |
| `RJ_CONSAN_CELL_SAMPLE_OFFSET=M` | `0` | Residue smaller than the cell stride. |
| `RJ_CONSAN_WATCHPOINT_BANKS=N` | `0` (auto) | Requested maximum banks per logical range: `0` or a power of two through `1024`. Auto requests eight; report memory limits may reduce the achieved geometry. Bank routing is unchanged. |
| `RJ_CONSAN_DEVICE_CONFLICT_CHECK=0\|1` | `0` | Enable the lower-fidelity immediate adjacent-range GPU check in addition to host scanning. |
| `RJ_CONSAN_CONFLICT_LIMIT=N` | `8` | Maximum distinct conflict examples retained per ConSan report (`0..1024`). Zero suppresses examples, not analysis or conflict counting. |
| `RJ_CONSAN_TOTAL_CONFLICT_LIMIT=N` | `64` | Maximum ConSan conflict examples across all reports, executables, and epochs of one hook session (`0..65536`). |

With the `default` preset and no independent selector variables, ConSan keeps
the legacy coupled runtime stride/offset for both workgroups and LDS cells. Setting any
`WORKGROUP_SAMPLE_*` or `CELL_SAMPLE_*` variable enables independent selection:
unspecified strides inherit the preset (256 with `default`) and offsets default to zero. Mixing these variables
with either legacy `RUNTIME_SAMPLE_*` variable is rejected, even if the values
agree. For example, workgroup stride 1 and cell stride 256 selects every
workgroup while retaining only the chosen LDS-cell residue. Synchronization
selection follows the workgroup setting, never the cell setting.

With `RJ_CONSAN_LOG=1`, the ConSan configuration line includes both resolved
selectors, static-site selection, requested banks, report ceiling, analysis
policy, and example limits. Per-object planning and static mapping lines show
the achieved slots and bank geometry (`effective_banks_min`/`effective_banks_max`
for admitted static mappings); a request is not a capacity guarantee.


ConSan conflict examples contain the code-object fingerprint, both original
instruction offsets, dispatch/workgroup identity, owners, access kinds, and LDS
byte ranges. An unavailable instruction mapping is printed as `unavailable`,
which is distinct from a mapped offset of zero. Attribution does not depend on
the separate 64-entry watchpoint detail listing.

ConSan also reports single-instruction write collisions among lanes when it
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
By default, ConSan diagnoses same-instruction lane collisions,
including stores whose address and every data word are proven uniform.

`RJ_CONSAN_ALLOW_PROVABLY_SAME_VALUE_WRITE_RACES=1` explicitly opts into suppressing this
common pattern. The default is `0`; sampling presets do not enable it. The opt-in
applies only to one ordinary native LDS store when static analysis proves both
an identical address and identical data across its participating lanes. The proof
checks every stored data register using local scalar/constant broadcasts and
copies, and is discarded on clobbers, EXEC changes, or CFG boundaries. Unknown or
differing values retain the diagnostic. This is a diagnostic suppression policy,
not a claim that equal-value writes are synchronized.

The store evidence remains available for cross-wave checking, including conflicts
with reads and identical-valued writes from other waves. The proof adds no GPU
value capture or comparison and does not enlarge report records. ConSan's
gfx1250 exact-lane limitation remains unchanged. Sampling still determines
whether evidence is retained.

Each causal window contains an exact lane mask protected by the same publication
claim and report generation as the access. Diagnostics are built on the host.
Direct callers must size buffers with the current layout helpers; a fixed byte
budget determines capacity using the record size. Use the reported achieved bank
count when comparing runs.

Examples are deduplicated within each report by the two sites (or slot indices
when unmapped), execution identity, owners, epochs, access kinds, and byte ranges.
Traversal order determines which examples are retained. The total
`conflicts` counts cross-wave evidence pairs plus one conflict per
retained exact group with colliding lanes, including repeated examples. `conflict_examples` counts retained examples and
`conflict_pairs_without_example` counts the remaining pairs, including
duplicates and pairs omitted by either limit. Reaching an output limit does not
stop analysis or change the diagnostic guards' verdict.

ConSan retains workgroup coordinates and a dispatch fingerprint, not an exact
globally unique launch identity. Its workgroup selector
includes dispatch identity, so offset zero does not guarantee selecting
workgroup zero, and repeated processes can select different workgroups at the
same offset. A fixed offset schedule reproduces selector settings, not dispatch
identities. Workgroup stride 1 removes this source of selection misses. When entry-captured
identity and scalar resources permit, a fast gate skips the shared access body
for an unselected workgroup; private-identity and compact-spill operating points
use an in-body fallback. The cell selector uses the starting LDS byte address
shifted right by two; it does not select every cell touched by a wide access.
Each static logical range has a separate table of immutable banks. The bank
hash mixes dispatch fingerprint, workgroup coordinates, cluster-workgroup ID,
and wave owner. It does not mix the access address, epoch, or a dynamic sequence.

A publisher tries one bank; there is no search for another empty bank and no
replacement of an older representative. A different retained window identity
at that bank increments `saturated_windows`, even if other banks remain empty.
The repeated-window check does not compare the packed access address or owner,
so this counter is not a count of every discarded access. Saturation is an
expected retention limit and does not itself make the verdict incomplete.
Malformed publication and true evidence loss use separate counters and can
make the analysis incomplete. See the [retention design](DESIGN.md#selection-and-retention)
for the resulting detection blind spots.

ConSan diagnostics attribute the retained instruction, wave owner and byte
range. Exact masks and intra-wave diagnostics are limited to the locally proven
uniform-address cases described above. Other windows retain representative
accesses; dense sampling or additional banks do not recover missing lane provenance.

A retained conflict prints `ambiguous` when overlapping static mappings name
different original instructions for its slot. Missing mappings print
`unavailable`; neither case is presented as instruction offset zero.

#### Repeatable offset sweeps

The [ConSan sweep runner](../../tests/dbi/consan/consan_sweep.py) runs an
unchanged command with a fixed, bounded schedule. For example, from
`emulation/rocjitsu`:

```sh
python3 tests/dbi/consan/consan_sweep.py \
  --hook /path/to/librocjitsu_dbi_hooks.so --output /tmp/consan-investigation \
  --workgroup-stride 256 --workgroup-offsets 0,1 \
  --cell-stride 256 --cell-offsets 0,1,2,3 --banks 0 --run-budget 8 \
  --timeout 180 -- ./workload its-arguments
```

The workgroup offsets are the outer loop and cell offsets the inner loop, in
supplied order. The budget selects a prefix of their Cartesian product; the
manifest records both requested and scheduled counts. The runner replaces
inherited runtime selectors, enables ConSan logging and fail-closed patching,
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

The `sampling-selection` CTest label exercises an explicit production-rate
positive, an unselected address, an ordered control and independent-offset
cases. These are separate from the ordinary dense semantic device matrix.

### ConSan report buffers

With no caller-owned buffer, the hook inventories the code and plans the
ConSan report: admitted logical ranges, bounded window banks, synchronization
metadata, and pending-acquire state.

Automatic planning starts with eight banks per logical range, or the explicit
requested bank count, and can repeatedly halve that count to fit the per-buffer
ceiling, down to one. This preserves the static ranges and reserved sync slots;
the achieved bank count, not the requested count, describes retention capacity.
The automatic allocator then requests the exact planned bytes. It never silently
shrinks site coverage or disables an event kind to fit. The per-buffer ceiling
is 128 MiB, and aggregate live automatic-report memory is bounded at 4 GiB per
process. Arithmetic overflow, a ceiling violation, or allocation failure is a
typed incomplete outcome.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_AUTO_REPORT_BUFFER_SIZE=N` | 128 MiB | Expert cap for HSA-tool-owned allocation; ordinary inventory still requests exact bytes below the cap. `0` disables automatic allocation. |
| `RJ_CONSAN_REPORT_BUFFER=0xADDR` | unset | Caller-owned device-visible report buffer. |
| `RJ_CONSAN_REPORT_BUFFER_SIZE=N` | `0` | Size of the caller-owned buffer; layout requirements depend on the enabled event families. |
| `RJ_CONSAN_REQUIRE_RECORDS=0\|1` | `0` | At unload, require at least one visible auto-buffer ConSan watchpoint; standalone synchronization metadata does not satisfy this check. |
| `RJ_CONSAN_REQUIRE_DIAGNOSTICS=0\|1` | `0` | Require at least one ConSan conflict. |
| `RJ_CONSAN_FORBID_DIAGNOSTICS=0\|1` | `0` | Require zero diagnostics/conflicts. |
| `RJ_CONSAN_FORBID_OVERFLOW=0\|1` | `0` | Fail if evidence was truly dropped. ConSan bounded saturation is reported separately from loss. |

The unload summary reports required and allocated bytes, per-region capacities,
current and peak live bytes, private spill growth, ConSan banks, saturation, undercoverage, overflow, and drops.

#### Repeated synchronized work

Repeated synchronized work normally requires no ConSan-specific API. Automatic
ConSan reports are bounded epoch storage: ConSan tracks completion signals for
instrumented dispatches, or for ordered barriers that cover signal-less
dispatches, and observes ordinary HSA waits. Once all tracked instrumented work
is quiescent, it analyzes and recycles the current report epoch automatically.
Normal HIP and PyTorch launch-and-synchronize loops therefore keep working as
their repetition count grows.

By default, the automatic transaction snapshots and analyzes every live report
before it clears any of them. It preserves each allocation address, layout,
reader, and generation embedded in instrumented code. ConSan causal windows,
publication state, and synchronization metadata are epoch-local and recycled.
Final unload combines every analyzed epoch with the last live epoch, so earlier
conflicts, diagnostics, saturation, and evidence are neither forgotten nor
counted twice.

These host report epochs are distinct from the per-wave barrier epoch inside a
dispatch. The latter currently saturates at 1023 without making the trust verdict
incomplete; see [the design limitation](DESIGN.md#identity-and-barrier-epochs).
Recycling completed dispatches does not repair exhaustion inside a long kernel.

For a long repeated workload where analyzing every iteration is unnecessary,
`RJ_CONSAN_EPOCH_ANALYSIS` selects which synchronized epochs receive the
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
boundaries. After setting `RJ_CONSAN_EPOCH_ANALYSIS=manual`, resolve and
call these exports from the loaded `HSA_TOOLS_LIB`:

```c
uint32_t rj_dbi_consan_begin_epoch_analysis_window();
uint32_t rj_dbi_consan_end_epoch_analysis_window();
```

Open the window before submitting the first operation to analyze and close it
after its final synchronization. Every synchronized epoch completed within the
window is analyzed. Nesting, closing an unopened window, or calling either API
under a non-manual policy returns status `3`; `0` means success, `1` means the
hook is inactive, and `2` means the selected mode is not the default mode.

##### Expert fallback for externally synchronized work

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
ConSan checkpoint, `1` when the hook is inactive, `2` for the intentional
SuperCollider no-op, and `3` when a report snapshot or decode failed. Treat any
status other than `0` or the expected SuperCollider `2` as an error.

SuperCollider does not use the ConSan report layout. Its mismatch marker is
lifetime-sticky and capacity-independent, so status `2` leaves it unchanged.

## SuperCollider controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_SC_REPORT_MODE=auto\|trap` | `auto` | `auto` owns a non-trapping sticky marker per relevant code object. `trap` is an expert process-disrupting mode. |
| `RJ_CONSAN_SC_REPORT_BUFFER=0xADDR` | unset | Use a caller-owned device-visible 32-bit marker instead of automatic allocation. |
| `RJ_CONSAN_SC_REPORT_MARKER=N` | `1` | Value written on mismatch. |
| `RJ_CONSAN_SC_DELAY=N` | `0` | Delay parameter between the guest access and duplicate/read-back. |
| `RJ_CONSAN_SC_DELAY_MODE=nop\|sleep\|sleep_var` | `nop` | Select `s_nop`, `s_sleep`, or `s_sleep_var` delay lowering. |
| `RJ_CONSAN_SC_DELAY_VAR_SSRC=N` | `106` | Scalar source encoding used by `sleep_var`. |
| `RJ_CONSAN_CHECK_TRAP_MODE=all\|lds\|flat` | `all` | Restrict SuperCollider to native DS or admitted flat LDS sites for debugging. |

The automatic marker reports that at least one duplicated/read-back value
differed. It does not identify an address, lane, value, or happens-before
violation. A race-free program can advance another wave between the original
and repeated access, so compare repeated known-correct and suspect runs.

SuperCollider timing perturbation is a validation-only composition mechanism,
not part of an ordinary detection run:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_SC_PERTURB_KIND=none\|barrier\|atomic` | `none` | Select one synchronization family to delay. |
| `RJ_CONSAN_SC_PERTURB_EDGE=release\|acquire` | `release` | Select the ordering edge of an atomic perturbation. |
| `RJ_CONSAN_SC_PERTURB_IDENTITY=IDENTITY` | unset | Select the reviewed semantic sequence identity. |
| `RJ_CONSAN_SC_PERTURB_INDEX=N` | `0` | Zero-based diagnostic selector when an identity is not supplied. Prefer identity. |
| `RJ_CONSAN_SC_PERTURB_MAX=N` | `1` | Bound the number of selected perturbations. |
| `RJ_CONSAN_SC_PERTURB_SLEEP=N` | `1` | Sleep immediate used by the perturbation. |
| `RJ_CONSAN_SC_PERTURB_REQUIRED_COUNT=N` | `0` | Require exactly this many selected perturbations when nonzero. |

As with fault injection, inventory and review an exact identity before using a
live perturbation in a qualification campaign.
