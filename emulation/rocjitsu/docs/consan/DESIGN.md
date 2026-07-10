# ConSan Design

ConSan is the rocJITsu DBI sanitizer effort for AMD LDS/shared-memory race
instrumentation. It works by intercepting GPU code-object loads through the HSA
tools interface, inspecting final native machine code, and loading a patched
replacement code object when instrumentation is possible.

The current implementation and tests are centered on the local RDNA4 /
`gfx1201` GPU because that is the hardware available in this workspace. That is
an incidental validation focus, not the intended product boundary. The intended
native-instrumentation target set is `gfx942`, `gfx950`, `gfx1201`, and
`gfx1250`. ConSan is not meant to translate kernels between GPU ISAs; it should
patch the final code object for the architecture that will actually run.

This document is present-facing. It describes what the current code does, why it
does it, and which parts are deliberately prototype-shaped rather than the
intended destination.

## Current Status At A Glance

| Area | Current code | Intended direction |
| --- | --- | --- |
| Interception | HSA-tools hook via `HSA_TOOLS_LIB`. | Keep HSA-tools as the main path. |
| Architecture | Native RDNA4 / `gfx1201` implementation and validation. | Generalize the native patching path for `gfx942`, `gfx950`, `gfx1201`, and `gfx1250`; do not translate between targets. |
| Public flavor switch | `RJ_CONSAN_FLAVOR=supercollider|moi`. | Keep the two top-level flavors. |
| SuperCollider | Usable redundant LDS/likely-group-flat check/trap mode. | Keep as a simple perturbation/value-check sanitizer mode. |
| MOI `record_replay` | DBI records plus host-side replay diagnostics. | Keep as reference/debug engine and oracle for inline work. |
| MOI `inline_shadow` | Narrow direct exact-shadow engine for native dword LDS, barriers, and one-slot atomic ordering controls. | Make this the exact low-volume GPU-side sanitizer. |
| MOI `sampled` | Direct sampled entry publication plus host-side sampled conflict scan. | Make this the low-overhead sanitizer option with real runtime sampling. |
| MOI broad operation | `record_replay` and `sampled` can run useful broad compatibility sweeps with prototype knobs; `inline_shadow` remains targeted. | Make `RJ_CONSAN_FLAVOR=moi` plus an engine choice usable over the standard corpus without per-kernel register, owner, epoch, or buffer tuning. |
| Registers | Mix of conservative automatic selection, descriptor growth, and explicit env knobs. | Centralize scratch allocation and reuse existing spill/cave machinery. |
| Diagnostics | Useful test guards and compact summaries; inline diagnostics are still sparse. | Structured, bounded diagnostics suitable for team use. |
| Flat/generic LDS | Conservative `Group`/`MaybeGroup` heuristic. | Harden provenance and address normalization before broadening coverage. |

## Source Map

Primary files:

- `lib/rocjitsu/src/rocjitsu/hooks/rj_hsa_dbi_hooks.cpp`
  - HSA-tools hook.
  - Environment parsing.
  - HSA-tool-owned MOI report-buffer allocation.
  - Teardown summaries and demo guards.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan.h`
  - Public ConSan patching options and result structures.
  - Flavor, MOI engine, delay mode, owner-source enums.
  - Decoded native DS, flat, barrier, fence, atomic, and MOI candidate records.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan.cpp`
  - Code-object inventory.
  - SuperCollider check/trap patching.
  - Shared candidate selection, patch placement, descriptor edits, flat
    provenance, and fault-injection plumbing.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan_moi.h`
  - MOI report ABI.
  - Exact-shadow, sampled, diagnostic, access, barrier, and atomic record
    layouts.
  - Host-side exact-shadow and sampled semantic helpers.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan_moi.cpp`
  - MOI DBI probe generation.
  - Record/replay access, barrier, and atomic probes.
  - Direct sampled probes.
  - Inline-shadow probes and inline atomic ordering prototype.
- `lib/rocjitsu/src/rocjitsu/code/patch/instruction_builder.*`
  - Small instruction encoders used by injected probes. Many current ConSan
    encoders are RDNA4-specific and need architecture-parametric equivalents
    as the target matrix expands.
- `lib/rocjitsu/src/rocjitsu/code/patch/trampoline_builder.*`,
  `kernel_text_layout.*`, `code_object_patcher.*`, `spill_manager.*`
  - Reusable patch-placement and DBT utilities that ConSan should lean on more
    heavily as probe size grows.

Test anchors:

- `tests/patch/consan_test.cpp`
  - Unit and synthetic patch-shape coverage.
  - Host-side MOI exact-shadow, sampled, barrier, and atomic semantic coverage.
- `tests/hip/`
  - Focused live GPU controls for ConSan instrumentation modes.
- IREE e2e tests in an external HIP-enabled IREE build directory.
  - Compatibility and patchability coverage for real kernels, currently
    exercised locally on RDNA4 / `gfx1201`.

## Public Mode Model

ConSan has a two-level public mode switch.

Top-level flavor:

```sh
RJ_CONSAN_FLAVOR=supercollider
RJ_CONSAN_FLAVOR=moi
```

MOI engine, used only with `RJ_CONSAN_FLAVOR=moi`:

```sh
RJ_CONSAN_MOI_ENGINE=record_replay
RJ_CONSAN_MOI_ENGINE=inline_shadow
RJ_CONSAN_MOI_ENGINE=sampled
```

Legacy `RJ_CONSAN_MOI_BACKEND` aliases still exist for compatibility:

- `context` maps to `record_replay`.
- `sampled_watchpoint` maps to `sampled`.

The intended naming is now:

- `supercollider`: simple perturbation plus redundant-access checking.
- `moi`: structured memory-order instrumentation.
- `record_replay`: exact host-side reference/debug MOI engine.
- `inline_shadow`: exact GPU-side MOI engine.
- `sampled`: low-overhead, lower-fidelity MOI engine.

## MOI Broad Enablement Gap

MOI has real DBI instrumentation today, but it is not yet a broad "turn it on
over everything" mode in the same sense as SuperCollider. SuperCollider still
has rough edges, but the typical command line is close to one flavor choice and
one trap/report choice. MOI currently needs more engine-specific resource
configuration and has narrower coverage in its exact inline path.

The gap is not one feature. It is a set of concrete blockers:

- **Scratch register allocation.** MOI probes still use explicit knobs such as
  `RJ_CONSAN_TMP_VGPR`, `RJ_CONSAN_MOI_EXEC_SAVE_SGPR`,
  `RJ_CONSAN_MOI_OWNER_SGPR`, `RJ_CONSAN_MOI_OWNER_VGPR`, and
  `RJ_CONSAN_MOI_EPOCH_VGPR` for important paths. Broad operation needs a
  single scratch-planning policy that can find free registers when possible,
  grow descriptors when legal, and spill/fill when necessary. The first
  implementation reference is Kunwar Grover's
  `origin/users/Groverkss/text-relocation-land` branch, which is confirmed to
  contain initial VGPR spilling support.
- **Owner and epoch state.** `inline_shadow` needs an owner and epoch live at
  instrumented accesses. Today those values can be initialized by prologue code,
  but the registers are manually chosen and the owner derivation is still
  architecture- and layout-sensitive. Broad operation needs automatic
  owner/epoch placement, a robust owner policy for arbitrary workgroup shapes,
  and clear fallback behavior when that policy is not available.
- **Report-buffer capacity.** MOI can use HSA-tool-owned auto report buffers,
  which is the right direction for applications such as IREE. Current broad
  tests still pass explicit sizes. Broad operation needs per-engine default
  sizing, capacity planning from the number of patched sites where possible,
  and visible overflow diagnostics so "buffer too small" does not look like "no
  races."
- **Engine profiles.** `record_replay`, `inline_shadow`, and `sampled` have
  different resource needs and diagnostic meanings. Broad operation should not
  require users to memorize prototype recipes. Each engine needs a documented
  default profile, with explicit override knobs retained for debugging.
- **Instruction coverage.** `record_replay` and `sampled` can cover more of
  the current broad IREE compatibility corpus, while `inline_shadow` is still
  mostly native dword LDS. Exact inline operation needs multi-cell native DS
  coverage, a policy for d16 accesses, and eventually likely-group flat/VFLAT
  coverage after provenance and address normalization are hardened.
- **Patch placement and text growth.** Broad MOI instrumentation inserts larger
  probes than SuperCollider in several paths. Coverage should not depend on
  hand-selected compact kernels. ConSan needs shared, tested placement through
  inline padding, local caves, appended caves, and text-relocation utilities.
- **Flat/generic LDS provenance.** Compilers can emit flat accesses for source
  `__shared__` memory. Broad operation needs logs and policy that distinguish
  strongly proven group memory from heuristic `MaybeGroup` memory, especially
  before inline-shadow writes exact shadow entries from flat addresses.
- **Barrier and atomic ordering.** Barriers and atomics are ordering evidence
  for LDS races, not a global-memory sanitizer. Broad operation needs barrier
  and selected atomic semantics that match the `record_replay` oracle closely
  enough for the MVP corpus.
- **Diagnostics.** `inline_shadow` currently emits a compact first-conflict
  diagnostic. Team-facing broad operation needs bounded diagnostics that include
  instruction offsets, access kinds, owners, epoch, LDS byte range, lane/EXEC
  information when practical, and overflow signals.
- **Runtime sampling policy.** `sampled` currently does deterministic static
  site throttling plus host-side scan. To become the broad low-overhead engine,
  it needs runtime sampling/generation policy and eventually in-kernel sampled
  conflict checks.
- **Architecture dispatch.** Local validation is on `gfx1201`, but the intended
  target set is `gfx942`, `gfx950`, `gfx1201`, and `gfx1250`. Broad operation
  needs ISA-specific capability checks and encoders instead of implicit RDNA4
  assumptions.
- **Test parity.** MOI must pass the same style of corpus as SuperCollider:
  focused rocJITsu tests, hip-moi semantic controls, IREE TileAndFuse, broader
  IREE e2e tests, and rocjitsu-test-corpus cases. Current local evidence is
  useful but not parity: broad IREE compatibility has passed for
  `record_replay` and `sampled`, while a broad `inline_shadow` sweep exposed a
  timeout/hang and remains a targeted mode.

The intended operational target is:

```sh
HSA_TOOLS_LIB=/path/to/librocjitsu_dbi_hooks.so \
RJ_CONSAN_FLAVOR=moi \
RJ_CONSAN_MOI_ENGINE=record_replay|inline_shadow|sampled \
ctest ...
```

Engine choice should remain explicit. The gap to close is the extra prototype
configuration currently needed after that choice.

## Interception And Code-Object Flow

The HSA hook wraps the code-object load path. For each memory-backed code
object, it:

1. Refreshes runtime configuration from environment variables.
2. Reads the original code-object bytes.
3. Runs ConSan inventory and patch planning.
4. Emits modified bytes when the selected mode can patch at least one site.
5. Loads the replacement memory-backed code object.
6. Logs a compact summary when `RJ_CONSAN_LOG` is enabled.

This path does not translate the program to a different ISA. The target remains
the original native code object for the GPU that will execute the kernel.

`RJ_CONSAN_REQUIRE_PATCH=1` is the main non-vacuity guard. It rejects a code
object when ConSan finds supported candidates for the selected mode but cannot
patch any such candidate. It still lets unsupported code objects load normally.

`RJ_CONSAN_DUMP_DIR=/path` writes original and patched code objects for
inspection.

## Shared DBI Constraints

ConSan is a post-register-allocation binary patcher. That imposes constraints
that a compiler pass would not have:

- It must decode final machine instructions.
- It must find or create patch placement.
- It must preserve architectural state such as VCC and EXEC.
- It must avoid clobbering live SGPRs/VGPRs.
- It must update AMDHSA kernel descriptors when increasing register allocation.
- It must infer flat/generic address-space provenance from machine-code
  dataflow, not from compiler IR address spaces.

Current placement mechanisms:

- inline replacement when a site has enough trailing `s_nop 0` padding;
- branch to an uncovered local NOP cave when a compact site has a reachable
  cave;
- appended `.text` cave when the object shape is simple enough and branch range
  constraints are satisfied.

Current register policy:

- SuperCollider probes have the most mature automatic scratch selection.
- Some paths can grow descriptor VGPR allocation as fallback.
- MOI engines still rely heavily on explicit env knobs for scratch, owner,
  epoch, and EXEC-save registers.
- Descriptor growth does not prove liveness. It only makes the register
  allocation legal in the kernel descriptor.
- There is no general ConSan spilling policy yet.

Kunwar Grover's `origin/users/Groverkss/text-relocation-land` branch is now the
confirmed first reference for spilling work. It contains initial VGPR-only
spilling support plus related text-relocation mechanics. ConSan should study,
reuse, or lightly adapt that implementation before adding new local machinery.
If ConSan needs SGPR spilling before rocJITsu has a shared implementation, the
right near-term response is a minimal ConSan-local SGPR spill/fill path for the
specific probe shape that needs it, not a comprehensive spill allocator.

This code should be treated as prototype integration work. Other rocJITsu work
is expected to make spilling more comprehensive, so ConSan should avoid
over-engineering interfaces that may be replaced by shared DBI infrastructure.

This register policy is the biggest gap between current implementation and the
intended architecture.

## SuperCollider Flavor

### Purpose

The SuperCollider flavor is the simplest usable sanitizer mode. It preserves
the original memory access, inserts a delay, repeats or reads back the same LDS
address, compares values, and reports a mismatch.

The motivating paper describes the redundant-read idea as issuing "a redundant
read to the same address" after delay. ConSan applies that shape after register
allocation, directly to final native code. The current implementation of this
shape is RDNA4 / `gfx1201`-centric.

### Current Algorithm

For a load:

```text
original LDS or likely-group-flat load
delay
duplicate load into scratch VGPRs
wait for the duplicate access
compare original destination VGPRs with scratch VGPRs
report mismatch
restore VCC if needed
return to original fallthrough
```

For a store:

```text
original LDS or likely-group-flat store
delay
synthesized readback into scratch VGPRs
wait for the readback
compare original store data with scratch VGPRs
report mismatch
restore VCC if needed
return to original fallthrough
```

Delay modes:

- `RJ_CONSAN_DELAY_MODE=nop`: emit `RJ_CONSAN_DELAY` copies of `s_nop 0`.
- `RJ_CONSAN_DELAY_MODE=sleep`: emit one `s_sleep N` when delay is nonzero.
- `RJ_CONSAN_DELAY_MODE=sleep_var`: emit one `s_sleep_var` from
  `RJ_CONSAN_DELAY_VAR_SSRC` when delay is nonzero.

Reporting:

- Default: `s_trap 0`.
- Optional: `RJ_CONSAN_REPORT_BUFFER=0xADDR` writes
  `RJ_CONSAN_REPORT_MARKER` to one caller-owned device-visible word and
  continues.

The marker-buffer ABI is intentionally small. It proves a non-trapping mismatch
path exists, but it does not record PC, lane, LDS address, values, or counts.

### Current Instruction Coverage

Native LDS check/trap supports:

- `ds_load_b32`
- `ds_load_b64`
- `ds_load_b128`
- `ds_load_2addr_b32`
- `ds_load_2addr_b64`
- `ds_load_2addr_stride64_b32`
- `ds_load_2addr_stride64_b64`
- `ds_load_u16_d16`
- `ds_load_u16_d16_hi`
- `ds_store_b32`
- `ds_store_b64`
- `ds_store_b128`

Likely group/LDS flat check/trap supports RDNA4 12-byte VFLAT:

- `flat_load_b32`
- `flat_load_b64`
- `flat_load_b128`
- `flat_store_b32`
- `flat_store_b64`
- `flat_store_b128`

Current exclusions:

- ordinary global-memory instrumentation;
- unsupported flat widths such as b8, b16, and b96;
- arbitrary flat accesses with unknown provenance;
- atomics as SuperCollider duplicate-access checks;
- async copies;
- same-value lost-update checks within one wave;
- structured race reports.

### Flat/VFLAT Rationale

Flat support is in scope because real compiled HIP helper code can access LDS
through flat/generic pointers. Source-level `__shared__` does not guarantee
that final machine code will use `ds_*` instructions. Once optimization has
materialized a generic pointer, the final instruction selector may emit
`flat_load_*` or `flat_store_*`; the pointer value decides whether the access
reaches the LDS aperture.

ConSan therefore classifies flat sites using a machine-code provenance tracker.
The public hints are:

- `Group`
- `Private`
- `MaybeGroup`
- `MaybePrivate`
- `Global`
- `Unknown`

Only `Group` and `MaybeGroup` are currently instrumentable. `MaybeGroup` is an
MVP heuristic: it means the observed dataflow is consistent with a pointer
derived from `src_shared_base`. It is not a formal proof for arbitrary
binaries. The intended direction is to harden this provenance before relying on
flat coverage for broad team-facing diagnostics.

### Where Current Code Is Prototype-Shaped

The current SuperCollider flavor represents our intention in these ways:

- HSA-level DBI interception is the right mechanism.
- Redundant LDS access plus delay is the intended sanitizer shape.
- Native DS and likely-group-flat coverage are both relevant.
- Non-vacuity guards and dump support are expected to stay.

It does not yet represent the destination in these ways:

- delay is deterministic or scalar-source based, not a robust randomized
  sampling policy;
- marker-buffer reporting is a smoke-test ABI, not final diagnostics;
- flat `MaybeGroup` classification is heuristic;
- register pressure is handled conservatively but not through a general spill
  policy.

## MOI Flavor

### Purpose

MOI is the structured race-detection flavor. It models accesses, owners, epochs,
barriers, and selected atomic ordering events. It is the path intended to grow
into higher-quality diagnostics than SuperCollider can provide.

MOI is still LDS-focused for this MVP. Global memory is intentionally out of
scope except where flat atomics are used as ordering events for LDS.

### Shared Semantic Model

MOI records or computes:

- access kind: read or write;
- LDS byte offset and byte count;
- 4-byte LDS shadow-cell range;
- workgroup identity;
- owner identity within a workgroup;
- epoch/order state;
- source instruction offset;
- selected synchronization and atomic events.

Exact conflict predicate, in simplified terms:

- same workgroup;
- overlapping LDS cell/range;
- at least one write;
- different owner;
- same unordered epoch, unless atomic/barrier semantics establish ordering.

Record/replay is the semantic reference. Inline-shadow and sampled should match
it where they claim exactness, and document lower fidelity where they do not.

### Report Buffer ABI

The MOI report buffer starts with `ConSanMoiReportHeader` and then engine-
specific sections. The header includes counts, capacities, dropped-record
signals, and offsets/capacities for:

- access records;
- barrier records;
- atomic records;
- diagnostics;
- exact-shadow entries;
- inline atomic release slots;
- sampled watchpoints.

Report-buffer sources:

- `RJ_CONSAN_MOI_REPORT_BUFFER=0xADDR` and
  `RJ_CONSAN_MOI_REPORT_BUFFER_SIZE=N`: caller-supplied buffer.
- `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=N`: HSA tool allocates one buffer per
  patched code object and summarizes it at teardown.

The auto-buffer path is the practical path for IREE and other applications that
cannot add a sanitizer kernel argument.

Demo guards:

- `RJ_CONSAN_MOI_REQUIRE_RECORDS=1`: fail at teardown if no auto buffer has
  visible access, barrier, atomic, diagnostic, exact-shadow, or sampled data.
- `RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1`: fail if no diagnostic/conflict signal
  is observed.
- `RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1`: fail if any diagnostic/conflict signal
  is observed.
- `RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT=1`: stricter `record_replay` guard
  that requires host replay to emit a conflict.

### Record/Replay Engine

`RJ_CONSAN_MOI_ENGINE=record_replay` is the reference/debug engine.

Current implementation:

- Patches selected native DS and likely-group-flat accesses.
- Emits `ConSanMoiAccessRecord` entries.
- Supports static per-site slots by default.
- Supports dynamic per-lane append with
  `RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1`.
- Records dynamic event indexes.
- Can patch supported RDNA4 barrier sites when
  `RJ_CONSAN_MOI_TRACK_BARRIERS=1`.
- Can record a narrow RDNA4 no-SADDR `flat_atomic*` subset when
  `RJ_CONSAN_MOI_TRACK_ATOMICS=1`.
- Replays visible records on the host into exact-shadow diagnostics.
- Coalesces contiguous same-workgroup barrier-arrival runs into one logical
  epoch advance.
- Models selected release/acquire atomic ordering on the host.

Important current simplifications:

- Static access-record slots overwrite on repeated execution of the same site.
- Dynamic access append requires `RJ_CONSAN_MOI_EXEC_SAVE_SGPR`.
- Dynamic append can consume records quickly because it writes per active lane.
- Some candidates are skipped near compiler-generated EXEC-mask regions until
  control-flow and liveness handling are stronger.
- Atomic DBI support is narrow and intended as LDS ordering evidence, not
  global-memory race detection.

Intended role:

- Keep `record_replay` as the oracle and debug mode.
- Use it to validate inline-shadow and sampled semantics.
- Do not optimize it at the expense of clarity.

### Inline-Shadow Engine

`RJ_CONSAN_MOI_ENGINE=inline_shadow` is the intended exact GPU-side engine. It
updates/checks shadow state during kernel execution instead of logging every
access for host replay.

Current implementation:

- Uses the shared MOI report buffer.
- Requires enough exact-shadow entries to cover the full 64 KiB LDS space at
  one 4-byte cell per entry.
- Instruments native dword LDS `ds_store_b32`/`ds_write_b32` and
  `ds_load_b32`/`ds_read_b32`.
- Uses `flat_atomic_swap_b64` to publish a packed exact-shadow word and obtain
  the prior word atomically.
- Reports a conflict when the prior entry is non-empty, from a different owner,
  in the same epoch, and not read/read.
- Can initialize owner/epoch VGPRs at kernel entry with
  `RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1`.
- Can increment an epoch VGPR after supported barrier sites with
  `RJ_CONSAN_MOI_TRACK_BARRIERS=1`.
- Can use `RJ_CONSAN_MOI_OWNER_SOURCE=hw_id` to initialize owner from RDNA4
  `HW_ID1` low bits through a scalar temporary.
- Has a one-slot inline atomic release/acquire prototype for a narrow no-SADDR
  `flat_atomic*` subset.

Current diagnostic shape:

- writes `diagnostic_count=1`;
- overwrites the first diagnostic record;
- records kind, backend, generation, owners, access kinds, instruction offsets,
  and epoch when configured;
- does not yet fill all range and lane-mask fields;
- intentionally favors a basic report over a complicated ABI.

Important current simplifications:

- Only native dword LDS accesses are covered.
- Owner, epoch, scratch VGPRs, and SGPR temporaries are mostly explicit knobs.
- `hw_id` owner source makes owner wave-uniform but still requires a manually
  selected scalar temporary.
- The atomic ordering path has one release slot.
- Diagnostics are first-conflict style, not rich multi-record reporting.

Intended role:

- Become the exact low-volume sanitizer.
- Cover wider LDS forms and likely-group-flat LDS after provenance is stronger.
- Use automatic scratch/spill policy instead of manual register knobs.
- Emit structured bounded diagnostics that are useful without reading raw logs.

### Sampled Engine

`RJ_CONSAN_MOI_ENGINE=sampled` is the lower-overhead, lower-fidelity engine.

Current implementation:

- Patches selected access sites.
- Writes compact 64-bit sampled watchpoint entries directly from DBI probes.
- Uses one static sampled slot per patched site.
- Packs valid/consumed bits, access kind, owner, epoch, generation, and LDS
  cell range.
- Uses generation zero in direct DBI mode.
- Selects static sites with `RJ_CONSAN_MOI_SAMPLE_STRIDE` and
  `RJ_CONSAN_MOI_SAMPLE_OFFSET`.
- Scans sampled entries on the host at HSA-tool teardown and reports sampled
  conflict counts.
- Keeps host-side sampled publish/replay helpers as semantic references.

Important current simplifications:

- Static-site selection is not runtime probabilistic sampling.
- There is no in-kernel sampled conflict checker.
- Clean sampled output is inconclusive.
- Owner/epoch values are not masked before packing; current tests rely on
  values fitting the prototype 10-bit fields.

Intended role:

- Provide a low-overhead mode that can be run more broadly than exact
  instrumentation.
- Add runtime sampling policy and generation management.
- Add in-kernel sampled conflict checks once the table policy is settled.
- Continue documenting that sampled mode can miss races.

## Owner And Workgroup Identity

MOI separates:

- workgroup identity: `(workgroup_x, workgroup_y, workgroup_z)`;
- owner identity: the logical peer inside a workgroup used by conflict checks.

Current workgroup identity is stronger than current owner identity. Access and
barrier probes can read RDNA4 launch TTMP payload fields and record 3D
workgroup coordinates, so host replay avoids cross-workgroup comparisons.

Current owner options:

- implicit workitem estimate:
  `workitem_id_x >> log2(wavefront_size)`;
- explicit owner VGPR with `RJ_CONSAN_MOI_OWNER_VGPR`;
- prologue initialization with `RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1`;
- `RJ_CONSAN_MOI_OWNER_SOURCE=workitem_id` default for prologue init;
- `RJ_CONSAN_MOI_OWNER_SOURCE=hw_id` for RDNA4 `HW_ID1` low bits, requiring
  `RJ_CONSAN_MOI_OWNER_SGPR`.

The `workitem_id` estimate is adequate for current 1D two-wave controls. It is
not a complete owner derivation for arbitrary 2D/3D local invocation layouts.

The `hw_id` source is useful for targeted inline-shadow experiments because it
is wave-uniform and does not depend on local invocation dimensionality. It is
not the final policy because it still requires manual SGPR selection and does
not perform liveness or spilling.

Intended direction:

- Keep 3D workgroup identity.
- Use a robust owner derivation that does not require user-selected registers.
- Preserve `hw_id` as a useful low-level source where appropriate.
- Integrate owner/epoch state with the future scratch/spill policy.

## Barrier And Atomic Semantics

Barriers:

- `record_replay` appends barrier-arrival records and host replay coalesces
  contiguous same-workgroup arrivals into logical epoch advances.
- `inline_shadow` can trampoline supported barriers, execute the original
  barrier, and increment an epoch VGPR after the barrier.
- `sampled` currently does not use barrier records in direct sampled checking.

Atomics:

- MOI treats atomics as ordering events for LDS, not as global-memory race
  checks.
- `record_replay` has host-side release/acquire modeling and a narrow DBI
  atomic-record path.
- `inline_shadow` has a one-slot release/acquire address-matching prototype.
- `sampled` has no atomic ordering implementation yet.

Current atomic support is intentionally narrow. The next goal is semantic
confidence on LDS ordering controls, not broad atomic opcode coverage.

## Why The Current Code Is Not The Final Design

The code intentionally contains several prototype mechanisms:

- manual register knobs for MOI probes;
- descriptor growth without general liveness/spilling proof;
- explicit report-buffer sizes in broad MOI test recipes;
- engine-specific resource recipes that users currently need to know;
- `MaybeGroup` flat LDS provenance;
- static per-site record and sampled slots;
- one-slot inline atomic release state;
- sparse inline diagnostic records;
- `inline_shadow` coverage and stability limited to targeted tests rather than
  the full broad corpus;
- deterministic or scalar-source delay instead of randomized sampling policy;
- IREE correctness tests used as patchability/non-corruption evidence, not race
  reports.

These are acceptable prototype choices because they prove the core DBI
building blocks:

- HSA-level interception works.
- Final native RDNA4 / `gfx1201` code can be decoded and patched.
- Native DS and likely-group-flat sites can be found.
- Compact IREE kernels can be patched through local or appended caves.
- HSA-tool-owned report buffers make MOI usable without application ABI
  changes.
- Host replay, inline exact shadow, and sampled publication can all observe
  DBI-written state.

They should not be mistaken for the target architecture. The target is a small
set of well-defined sanitizer modes with automatic resource management,
defensible LDS classification, clear diagnostics, and a stable test matrix.

## Validation Evidence

Current evidence categories:

- Focused unit tests for instruction encoding, patch planning, report layout,
  exact-shadow replay, sampled replay, and atomic/barrier semantics.
- Focused rocJITsu HIP GPU tests for SuperCollider, record/replay,
  inline-shadow, sampled, barriers, atomics, and owner-source controls.
- IREE RDNA4 / `gfx1201` TileAndFuse matmul tests under SuperCollider and MOI
  modes.
- Broader IREE e2e inventory under SuperCollider patch-required mode.
- Broader IREE e2e compatibility sweeps under MOI `record_replay` and
  `sampled` with prototype resource knobs.
- A broader IREE `inline_shadow` sweep is not yet clean: targeted TileAndFuse
  tests pass, but a broad sweep exposed a timeout/hang, so `inline_shadow` is
  not yet a blanket corpus mode.

What a passing IREE test means:

- With `RJ_CONSAN_REQUIRE_PATCH=1`, supported code objects did not silently
  skip patching.
- With `RJ_CONSAN_MOI_REQUIRE_RECORDS=1`, at least one auto report buffer
  showed visible ConSan state.
- Passing correctness means the instrumentation did not corrupt the tested
  known-correct workload.

What it does not mean:

- It does not prove that IREE had a race.
- It does not prove sampled mode saw every race.
- It does not prove flat `MaybeGroup` is formally correct for arbitrary code.
- It does not prove manually selected registers are generally safe.

## Immediate Engineering Gaps

The highest-leverage gap is register and spill policy. Most remaining feature
work needs larger probes or more persistent state. Without a better scratch
policy, each new probe risks adding another manual env knob or relying on
descriptor growth as if it were liveness proof.

Before implementing new spilling machinery, start from Kunwar Grover's
`text-relocation-land` branch:

- `origin/users/Groverkss/text-relocation-land`

That branch is confirmed to contain initial VGPR-only spilling support. The
first ConSan task is to understand its data structures, spill/fill placement
rules, and interaction with text relocation, then reuse or adapt the smallest
useful subset. `origin/users/Groverkss/dbt-tooling` and
`origin/users/Groverkss/dbt_interposer` remain useful background references,
but `text-relocation-land` is the primary branch for spill implementation.

If ConSan needs SGPR spilling, assume it is not already covered there. Implement
only the minimal SGPR support needed for the current probe family, keep it
isolated, and prefer deleting or replacing it when shared rocJITsu spilling
lands.

After scratch/spill policy, the next operational gap is MOI defaulting: engine
profiles, auto report-buffer sizing, clear failure modes, and a test matrix
that proves each engine can run without ad hoc per-test register choices. That
work is tracked in `PLAN.md` as the path from targeted MOI instrumentation to
broad MOI operation.
