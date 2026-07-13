# ConSan Design

ConSan is the rocJITsu DBI sanitizer effort for AMD LDS/shared-memory race
instrumentation. It works by intercepting GPU code-object loads through the HSA
tools interface, inspecting final native machine code, and loading a patched
replacement code object when instrumentation is possible.

The current implementation has native paths for RDNA4 / `gfx1201` and CDNA4 /
`gfx950`. The gfx1201 profile has completed its broad qualification; gfx950 has
completed focused and selected-workload qualification. Its broad SuperCollider
inventory is classified, while the three broad MOI profiles remain in progress.
The intended native-instrumentation target set also includes
`gfx942` and `gfx1250`. ConSan does not translate kernels between GPU ISAs; it
patches the final code object for the architecture that will actually run.

This document is present-facing. It describes what the current code does, why it
does it, and which parts are deliberately prototype-shaped rather than the
intended destination. [SPILLING.md](SPILLING.md) is the focused guide to the R1
register allocation, private-layout, ownership, and spill transaction.

## Current Status At A Glance

| Area | Current code | Intended direction |
| --- | --- | --- |
| Interception | HSA-tools hook via `HSA_TOOLS_LIB`. | Keep HSA-tools as the main path. |
| Architecture | Native RDNA4 / `gfx1201` and CDNA4 / `gfx950` implementation. gfx950 focused and selected qualification is complete; broad SuperCollider is classified and broad MOI qualification is in progress. | Generalize the same native patching model to `gfx942` and `gfx1250`; do not translate between targets. |
| Public flavor switch | `RJ_CONSAN_FLAVOR=supercollider|moi`. | Keep the two top-level flavors. |
| SuperCollider | Usable redundant LDS/likely-group-flat check/trap mode. | Keep as a simple perturbation/value-check sanitizer mode. |
| MOI `record_replay` | DBI records plus host-side replay diagnostics. | Keep as reference/debug engine and oracle for inline work. |
| MOI `inline_shadow` | Direct exact-shadow engine for decoded native LDS cell ranges, admitted group-flat forms, barriers, and one address-scoped atomic ordering slot per workgroup partition. | Broaden proven instruction/order coverage while retaining exact supported-form semantics. |
| MOI `sampled` | Runtime-qualified sampled entry publication, host-side conflict scan, and opt-in immediate check. | Improve sampling fidelity and overhead measurement without presenting clean samples as proof of race freedom. |
| MOI broad operation | All three `standard-v1` engines complete the 209-test gfx1201 IREE compatibility sweep without register-number or buffer-size configuration. On gfx950 all three pass the guarded 10-test selected tier; broad MOI qualification is tracked per engine. Tier0 supplies guarded semantic evidence on both targets. | Expand instruction and architecture breadth without weakening explicit unsupported-site behavior. |
| Registers | Owner-scoped plans select one per-site scratch assignment for record/replay, sampled, and inline-shadow access, barrier, and atomic probes, including helpers shared by multiple kernels. Live victim windows use target-native gfx1201 or gfx950 private scratch (including zero-private kernels through dispatch rewriting), with one common layout for every owner. gfx950's default `workitem_id` identity normally uses five persistent VGPRs; an `ACCUM_OFFSET` overlap selects a fresh five-SGPR wave-state window. If no such scalar window exists, that default path fails visibly rather than silently changing owner semantics. Private owner/epoch state is selected separately when forced or when preserving a non-workitem owner source. Scalar paths preserve EXEC, VCC, and SCC. | Keep explicit register variables as debug overrides and replace target-local machinery when shared rocJITsu infrastructure matures. |
| Diagnostics | Bounded inline and sampled diagnostics plus resource, overflow, and unsupported-site summaries; compact shadow words limit prior-lane detail. | Preserve bounded output while improving precision and presentation. |
| Flat/generic LDS | Explicit `likely`/`strict` admission policy over `Group`/`MaybeGroup`, with target-specific RDNA4 and CDNA4 group-flat address contracts. CDNA4 emission is limited to strongly classified generic-segment, zero-offset dword/dwordx2/dwordx4 forms. | Extend proven provenance conservatively as compiler code shapes and native targets broaden. |

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
  - Small architecture-dispatched instruction encoders used by injected
    probes, including the current RDNA4 and CDNA4 probe primitives.
- `lib/rocjitsu/src/rocjitsu/code/patch/trampoline_builder.*`,
  `kernel_text_layout.*`, `code_object_patcher.*`, `spill_manager.*`
  - Reusable patch-placement and DBT utilities used by all ConSan engines.
    `spill_manager.*` also emits transactional
    gfx1201 and gfx950 B32 VGPR save/restore batches, performs kernel-local fixed-private
    descriptor growth, and keeps the matching AMDGPU MessagePack metadata
    coherent in place.

Test anchors:

- `tests/patch/consan_test.cpp`
  - Unit and synthetic patch-shape coverage.
  - Host-side MOI exact-shadow, sampled, barrier, and atomic semantic coverage.
- `tests/dbi/`
  - Focused live GPU controls for ConSan instrumentation modes.
- IREE e2e tests in an external HIP-enabled IREE build directory.
  - Compatibility and patchability coverage for real kernels on RDNA4 /
    `gfx1201` and CDNA4 / `gfx950`.

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

## MOI Broad Operation And Boundaries

On gfx1201, MOI is now a broad "turn it on" mode in the same operational sense
as SuperCollider: select the flavor and one explicit engine, and the
`standard-v1` profile supplies resources and report buffers. All three engines
pass the same selected and broad IREE tiers as SuperCollider. This is an
operational and compatibility claim, not a claim that every loaded instruction
is instrumented or that every engine has identical precision.

The implementation boundary is:

- **Scratch register allocation.** Record/replay, sampled, and inline-shadow
  access, barrier, and atomic probes choose per-site dead or fresh
  descriptor-backed VGPR windows and spill allowed live windows when needed.
  Shared helpers use one plan valid for every owning kernel. Scalar and
  persistent state are automatic too. The first implementation reference is
  Kunwar Grover's
  `origin/users/Groverkss/text-relocation-land` branch, whose reusable
  contribution is the initial VGPR spill-slot allocator.
- **Patch-budget ranking.** When automatic scalar identity is active, all three
  access engines stably rank candidates by the shared resource outcome:
  explicit/dead, descriptor growth, spill, then unsupported. Equal outcomes
  retain program order. `RJ_CONSAN_MAX_PATCHES` therefore selects the best
  available sites without changing forced-spill semantics.
- **Owner and epoch state.** `inline_shadow` automatically uses a persistent
  descriptor-backed representation, scalar wave state when gfx950 ordinary
  VGPR placement would cross `ACCUM_OFFSET`, or derived-owner/private-epoch
  state when forced or when needed to preserve a non-workitem owner source.
  Exhausting the default gfx950 five-SGPR path is an explicit failure, not a
  private-state fallback. The packed identity contract is intentionally
  bounded and architecture-sensitive.
- **Report-buffer capacity.** MOI can use HSA-tool-owned auto report buffers,
  which is the right direction for applications such as IREE. The HSA hook
  defaults to 64 KiB for record/replay and sampled and 256 KiB for inline
  shadow, allocates only after a code object inventories relevant sites, and
  reports dropped records unconditionally. Explicit size and zero-disable
  overrides remain available.
- **Engine profiles.** `record_replay`, `inline_shadow`, and `sampled` use the
  documented `standard-v1` profiles: lazy per-engine buffers, automatic
  resources, conservative one-site composition, and explicit extensions for
  broader patch counts, dynamic records, ordering probes, and immediate
  sampled checking. Startup logs name the profile.
- **Instruction coverage.** Inline shadow handles supported native multi-cell
  DS ranges and admitted zero-offset group-flat forms through the same shadow
  publisher. Unsupported d16, encoding, or provenance shapes remain visible
  skips rather than speculative instrumentation.
- **Patch placement and text growth.** All MOI engines and SuperCollider use a
  shared transactional planner for inline padding, local caves, and appended
  caves. A failed plan does not leave partial descriptor/text mutations.
- **Flat/generic LDS provenance.** Compilers can emit flat accesses for source
  `__shared__` memory. Inventory distinguishes proven `Group`, heuristic
  `MaybeGroup`, private, and unknown forms. `strict` admits only proven group;
  `likely` also admits likely group, with exclusions reported explicitly.
- **Barrier and atomic ordering.** Barriers and selected atomics are ordering
  evidence for LDS races, not global-memory checks. Record/replay is the
  semantic oracle; inline ordering deliberately remains a narrow one-slot
  prototype, and sampled mode does not consume barrier epochs.
- **Diagnostics.** `inline_shadow` emits bounded first-N diagnostics with
  instruction offsets, access kinds, owners, epoch, LDS byte ranges, the
  current conflict EXEC mask, and visible overflow. The prior writer's lane
  mask is unavailable in the compact exact-shadow word and is reported as
  unknown/zero.
- **Runtime sampling policy.** `sampled` combines deterministic runtime
  generation policy with host scanning and an opt-in immediate in-kernel
  conflict check. Static site selection still bounds patch composition.
- **Architecture dispatch.** Native capability dispatch and focused validation
  now cover `gfx1201` and `gfx950`. Their encodings, register geometry, waits,
  and spill limits are selected explicitly. `gfx942` and `gfx1250` remain
  future targets rather than inheriting assumptions from either implemented
  architecture.
- **Test parity.** At checkpoint `bd297cdbfc`, the focused
  ConSan/resource/placement filter covers 263 tests. The gfx950 matrix also
  covers 49 hip-moi controls, 10 selected
  IREE tests per profile, and a 259-test broad IREE inventory. The broader
  rocjitsu-test-corpus has
  separate local dependency limitations documented in `LOCAL_TESTING.md`.

The intended operational target is:

```sh
HSA_TOOLS_LIB="$RJ_HOOK" \
RJ_CONSAN_FLAVOR=moi \
RJ_CONSAN_MOI_ENGINE=record_replay|inline_shadow|sampled \
ctest ...
```

Engine choice remains explicit because the engines trade precision, overhead,
and diagnostic shape. Advanced knobs remain available, but ordinary corpus
runs do not require register or buffer selection.

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

`DbiPatchPlacementPlanner` is the shared transactional allocator for these
choices. It records explicit anchor/body/return mappings, reserves the return
branch as part of every cave, and leaves its state unchanged on overlap or
branch-range failure. All MOI access, barrier, and atomic probe families use it,
as do SuperCollider's native LDS and likely-group flat check/trap paths. The
native and flat passes preserve their explicit text/file-coordinate mapping
when they compose, and appended-cave emission rejects stale planned offsets.

Current register policy:

- SuperCollider probes have the most mature automatic scratch selection.
- Static MOI record/replay and sampled access probes use a read-only,
  kernel-scoped CFG/liveness plan. Each site first searches dead VGPRs within
  its current descriptor allocation, then a fresh range above all guest
  references, growing only the owning descriptor when needed.
- Symbol-backed code ranges exclude alignment padding from CFG decoding, so
  the same planning path works on normal multi-kernel HIP code objects.
- Direct-kernel and reachable shared-helper static record/replay and sampled
  sites consume a typed spill-required outcome. They use an appended cave
  containing the spill save, derived-owner setup, original access,
  conservative LDS wait, instrumentation, spill restore, and return.
- Spill plans grow both the owning descriptor and the named kernel's AMDGPU
  MessagePack private size. The HSA hook also associates that requirement with
  the loaded kernel object and rewrites its AQL dispatch packet, so a compiled
  private size of zero can become nonzero without relying on the runtime's
  original symbol metadata.
- Dynamic record, barrier-record, inline diagnostic, and inline atomic acquire
  paths allocate fresh descriptor-backed scalar windows. Scalar VCC snapshots
  make restoration independent of active lanes; SCC is captured before the
  probe and restored last. Explicit SGPR knobs remain debug overrides.
- When no explicit inline-shadow owner/epoch pair is supplied, ConSan first
  places a dedicated pair above guest references and the selected scratch
  window, replans scratch with that pair forbidden, and injects a kernel-entry
  initializer. If no pair fits, it derives owner per access and keeps epoch in
  a persistent private dword. The epoch slot precedes an independently aligned
  ephemeral spill zone shared by access, barrier, and entry-prologue leases.
  Shared helpers use one representation for every reachable owner; private
  workitem-derived ownership additionally requires the owners to agree on wave
  size.
- Target-dispatched spill backends allocate stable slots through
  `SpillManager`. gfx1201 emits address-free VSCRATCH B32 operations with split
  store/load waits. gfx950 emits eight-byte address-free FLAT_SCRATCH dword
  operations and drains `VM_CNT` before both clobber and guest reuse. Both grow
  only the selected descriptor's fixed private segment. Dynamic-stack kernels
  are detected from compiler-emitted symbols and rejected.
- Static record/replay, sampled, and descriptor-full inline-shadow access
  probes consume that backend for direct kernels and reachable shared helpers.
  A shared spill starts above the maximum original private extent and grows
  every owner to the same required size. Dynamic stacks and unresolved
  indirect ownership remain outside the current resource slice. No SGPR spill
  backend is present because the current tier can fail safely when all 106
  normal SGPRs are occupied.

Kunwar Grover's `origin/users/Groverkss/text-relocation-land` branch was the
first reference for spilling work. Its directly reusable piece is the
`SpillManager` slot allocator: it provides aligned, stable per-lane offsets but
does not emit gfx1201 or gfx950 scratch instructions or update descriptors.
ConSan reuses that allocator and supplies the small target-specific backends
around it.
If ConSan needs SGPR spilling before rocJITsu has a shared implementation, the
right near-term response is a minimal ConSan-local SGPR spill/fill path for the
specific probe shape that needs it, not a comprehensive spill allocator.

This code should be treated as prototype integration work. Other rocJITsu work
is expected to make spilling more comprehensive, so ConSan should avoid
over-engineering interfaces that may be replaced by shared DBI infrastructure.

Barrier/atomic VGPR patchers now consume the same plan, and bounded HSA logs
report explicit, dead, descriptor-growth, spill, and unsupported outcomes plus
planned and emitted spill bytes.

## gfx950 / CDNA4 Native Contract

gfx950 is wave64. Injected paths therefore treat `EXEC` and `VCC` as full
64-bit architectural state. Scalar save windows preserve both halves of VCC;
the RDNA4 SuperCollider path needs only `VCC_LO` for its compare shape,
but that shortcut is not valid for CDNA4. Probe address arithmetic uses a
VCC-neutral CDNA4 form rather than silently clobbering guest condition state.

The admitted native DS set is the normalized B32/B64/B128 and supported d16 or
two-address load/store set listed below. Other DS encodings, including atomics,
permute/swizzle, transpose, reserved GDS forms, and forms whose operands cannot
be normalized are inventory-only with a typed reason. The admitted CDNA4
group-FLAT set is deliberately smaller: strongly proven generic-segment,
zero-offset dword, dwordx2, and dwordx4 loads and stores.

Waits are also target-specific. Native LDS completion drains `LGKM_CNT`.
General FLAT loads, stores, and shadow atomics drain both `VM_CNT` and
`LGKM_CNT`; `s_barrier` does not imply either wait. Address-free FLAT_SCRATCH
save/fill drains `VM_CNT` at the save-before-clobber and
restore-before-guest-use boundaries. No gfx12 `s_wait_loadcnt`,
`s_wait_storecnt`, or `s_wait_dscnt` encoding is reused on gfx950.

CDNA4 descriptor geometry has an additional ordinary-VGPR hazard:
`COMPUTE_PGM_RSRC3.ACCUM_OFFSET` establishes the AccVGPR alias boundary.
Fresh or persistent ordinary VGPR windows must not cross a nonzero boundary.
For the default `workitem_id` owner source, when the normal five-VGPR identity
window would cross it, ConSan uses a fresh five-SGPR wave-state window
containing owner, epoch, and X/Y/Z workgroup coordinates. If no such window is
available, instrumentation fails visibly. Private owner/epoch state is a
distinct representation selected when forced or when preserving a
non-workitem owner source; it is not a fallback after five-SGPR exhaustion.
AccVGPR spilling itself is not implemented.

Finally, an AMDHSA descriptor with kernarg preloading can expose two hardware
entry paths exactly 256 bytes apart. Entry redirection preserves both paths:
the generated owner/epoch or private-state prologue has a matched stub for each
entry and returns each one to its corresponding original path. Treating such a
kernel as having only the descriptor's base entry can branch into unrelated
code and is invalid.

## SuperCollider Flavor

### Purpose

The SuperCollider flavor is the simplest usable sanitizer mode. It preserves
the original memory access, inserts a delay, repeats or reads back the same LDS
address, compares values, and reports a mismatch.

The motivating paper describes the redundant-read idea as issuing "a redundant
read to the same address" after delay. ConSan applies that shape after register
allocation, directly to final native code. The implementation emits the shape
natively for both RDNA4 / `gfx1201` and CDNA4 / `gfx950`.

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

Likely group/LDS flat check/trap supports the admitted RDNA4 12-byte VFLAT and
CDNA4 eight-byte generic FLAT forms:

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

`Group` means that both 32-bit halves were coherently traced from
`src_shared_base`. `MaybeGroup` means only a component, select, or arithmetic
chain remains consistent with that origin; it is a heuristic, not a proof for
arbitrary binaries. `RJ_CONSAN_FLAT_PROVENANCE=likely` (the default) admits
both classifications. `strict` admits only `Group`, for demonstrations and
qualification runs that prefer precision over flat-site recall. Inventory and
verbose site logs retain the classifications independently of this selection
policy, and skipped-candidate warnings count strict-policy exclusions.

For an admitted flat group pointer in `v[addr:addr+1]`, ConSan's LDS
normalization contract is: `v[addr]` is the unsigned byte offset within the LDS
aperture and `v[addr+1]` is provenance evidence only. Static VFLAT/FLAT offset
bytes are added to the low word before rounding the byte interval to 4-byte
shadow cells. The high word must never be mixed into the shadow index. CDNA4
currently admits only strongly classified, generic-segment, zero-offset B32,
B64, and B128 forms. Sites whose encoding or provenance cannot satisfy the
target's contract remain unpatched.

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
- SuperCollider uses the shared resource/placement machinery where its probe
  shape requires it; the reusable gfx1201 spill backend is currently driven by
  the larger MOI probes rather than claiming general multi-target spilling.

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
- With no caller buffer or size override, the HSA tool allocates a per-engine
  default buffer only for code objects with relevant MOI sites and summarizes
  it at teardown.
- `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=N`: override the default; explicit
  zero disables auto allocation.

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
- `RJ_CONSAN_MOI_FORBID_OVERFLOW=1`: fail at teardown if an auto buffer dropped
  access, barrier, atomic, or diagnostic records. Overflow is always printed to
  stderr even without this guard.

### Record/Replay Engine

`RJ_CONSAN_MOI_ENGINE=record_replay` is the reference/debug engine.

Current implementation:

- Patches selected native DS and likely-group-flat accesses.
- Emits `ConSanMoiAccessRecord` entries.
- Supports static per-site slots by default.
- Supports dynamic per-lane append with
  `RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1`.
- Records dynamic event indexes.
- Can patch supported RDNA4 and CDNA4 barrier sites when
  `RJ_CONSAN_MOI_TRACK_BARRIERS=1`.
- Can record a narrow target-proven no-SADDR `flat_atomic*` subset when
  `RJ_CONSAN_MOI_TRACK_ATOMICS=1`.
- Replays visible records on the host into exact-shadow diagnostics.
- Coalesces contiguous same-workgroup barrier-arrival runs into one logical
  epoch advance.
- Models selected release/acquire atomic ordering on the host.

Important current simplifications:

- Static access-record slots overwrite on repeated execution of the same site.
- Dynamic access append automatically allocates its EXEC/VCC/SCC scalar window;
  `RJ_CONSAN_MOI_EXEC_SAVE_SGPR` is an optional debug override.
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
- Instruments decoded native scalar, B64, B128, d16, and two-address LDS
  loads/stores, publishing every rounded 4-byte cell in each access range.
- Instruments supported zero-offset flat/VFLAT loads and stores admitted by
  the configured F1 provenance policy. The low address VGPR is normalized as
  the LDS byte offset and feeds the same cell-range publisher as native DS.
- Uses `flat_atomic_swap_b64` to publish a packed exact-shadow word and obtain
  the prior word atomically.
- Partitions exact shadow by the bounded workgroup coordinate
  `p = x + Ex * (y + Ey * z)`. Every partition owns a complete 16,384-entry
  table for the 64-KiB LDS address space; distinct in-range workgroups cannot
  compare against one another.
- Checks X/Y/Z against the configured extents before calculating or touching a
  partition address. An out-of-range workgroup skips exact-shadow and atomic
  state and increments the typed partition-overflow counter.
- Reports a conflict when the prior entry is non-empty, from a different owner,
  in the same epoch, and not read/read.
- Can initialize owner/epoch VGPRs at kernel entry with
  `RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1`.
- Can increment an epoch VGPR after supported barrier sites with
  `RJ_CONSAN_MOI_TRACK_BARRIERS=1`.
- Can use `RJ_CONSAN_MOI_OWNER_SOURCE=hw_id` to initialize owner from RDNA4
  `HW_ID1` low bits through a scalar temporary. This source is unavailable on
  CDNA4; gfx950 uses the logical owner initialized at entry.
- Has one inline atomic release/acquire slot per workgroup partition for a
  narrow no-SADDR `flat_atomic*` subset. Release and acquire use the same
  mixed-radix partition as exact shadow.
- Validates explicit and HSA-tool-owned buffer layouts against the configured
  partition count before patch execution. Each partition requires the complete
  exact-shadow table and an atomic slot; insufficient capacity is a visible
  configuration/resource failure rather than aliased storage.

Current diagnostic shape:

- atomically reserves one slot per conflicting wave and writes up to the
  configured diagnostic capacity;
- records kind, backend, generation, owners, access kinds, instruction offsets,
  and epoch when configured;
- records both LDS ranges and the current conflict EXEC mask; the prior lane
  mask remains unknown because it is not present in the exact-shadow word;
- reports count-over-capacity as dropped diagnostics.

Important current simplifications:

- Native byte/d16 accesses conservatively cover their rounded 4-byte cell;
  byte-precise masks are not represented.
- Direct-kernel owner, epoch, scratch VGPRs, and SGPR temporaries are automatic;
  explicit register variables remain debug overrides.
- `hw_id` owner source makes owner wave-uniform and automatically receives a
  fresh scalar temporary when the kernel has capacity.
- The atomic ordering path has one release slot per configured workgroup
  partition.
- Diagnostics are bounded first-N records, not an unbounded trace.

Intended role:

- Become the exact low-volume sanitizer.
- Extend flat/VFLAT encoding coverage beyond the current supported forms while
  preserving the explicit provenance policy.
- Use automatic scratch/spill policy instead of manual register knobs.
- Emit structured bounded diagnostics that are useful without reading raw logs.

### Sampled Engine

`RJ_CONSAN_MOI_ENGINE=sampled` is the lower-overhead, lower-fidelity engine.

Current implementation:

- Patches selected access sites.
- Writes compact 64-bit sampled watchpoint entries directly from DBI probes.
- Uses one static sampled slot per patched site.
- Partitions sampled storage by the bounded workgroup coordinate
  `p = x + Ex * (y + Ey * z)`. Each partition owns
  `floor(capacity / (Ex * Ey * Ez))` consecutive site slots, so distinct
  in-range workgroups never alias.
- The public `RJ_CONSAN_MOI_WORKGROUP_EXTENT_X/Y/Z` knobs default to `1`.
  For `2,2,2`, partition indices are `0..7`; a non-vacuous run needs at least
  eight report slots per patched site and must dispatch only coordinates in
  that bounded cube. Auto buffers publish the derived slots-per-partition in
  the report ABI. A caller-owned sampled buffer must initialize the same ABI-v2
  header field and confirm it with
  `RJ_CONSAN_MOI_REPORT_SLOTS_PER_PARTITION`.
- Packs valid/consumed bits, access kind, owner, epoch, generation, and LDS
  cell range.
- Uses generation zero in direct DBI mode.
- Selects static sites with `RJ_CONSAN_MOI_SAMPLE_STRIDE` and
  `RJ_CONSAN_MOI_SAMPLE_OFFSET`.
- Can leave every eligible static site patched while deterministically
  selecting runtime waves with `RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE` and
  `RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET`. The power-of-two policy compares the
  wave owner against `owner & (stride - 1)` and preserves VCC in an
  automatically allocated scalar pair.
- Auto-buffer probes publish the buffer generation in every sampled entry.
  Host replay ignores entries from older generations, scans the active entries
  at HSA-tool teardown, and reports sampled conflict counts.
- With `RJ_CONSAN_MOI_SAMPLED_CHECK=1`, site `i` checks the immediately
  preceding site slot in the same workgroup partition before publishing.
  Matching valid generation, epoch, owner inequality, conflicting access
  kinds, and exact cell range increment the report header's sampled
  immediate-conflict counter on the GPU. The HSA summary and diagnostic guards
  consume that counter without waiting for host pairwise replay.
- Bounds checks precede addressing. An X/Y/Z coordinate outside its configured
  extent skips publication without aliasing another partition and increments
  the typed partition-overflow counter. Host replay also refuses to compare
  entries from different partitions.
- Keeps host-side sampled publish/replay helpers as semantic references.

Important current simplifications:

- Runtime selection is deterministic owner-stride sampling, not probabilistic
  or temporally varying sampling within one wave.
- The in-kernel checker compares one adjacent slot and exact ranges rather than
  scanning the table or testing all overlapping ranges. Its counter is an
  immediate signal, not a structured full diagnostic record.
- Clean sampled output is inconclusive.
- Owner/epoch values are masked to the prototype 10-bit fields before packing.

Intended role:

- Provide a low-overhead mode that can be run more broadly than exact
  instrumentation.
- Add runtime sampling policy and generation management.
- Broaden the current adjacent-slot in-kernel check only when a stronger table
  policy has a measured benefit.
- Continue documenting that sampled mode can miss races.

## Owner And Workgroup Identity

MOI separates:

- workgroup identity: `(workgroup_x, workgroup_y, workgroup_z)`;
- owner identity: the logical peer inside a workgroup used by conflict checks.

Current workgroup identity is stronger than current owner identity. Access and
barrier probes record target-native 3D workgroup coordinates, so host replay
avoids cross-workgroup comparisons. gfx950 entry instrumentation guarantees a
stable X/Y/Z snapshot even when the original descriptor omitted one or all
workgroup-ID inputs: it enables the missing entry ABI inputs, snapshots all
three coordinates into the chosen persistent VGPR or five-SGPR representation,
and restores the guest-visible entry SGPR layout before returning to original
code. Kernarg-preload dual entries receive matching stubs.

Current owner options:

- implicit workitem estimate:
  `workitem_id_x >> log2(wavefront_size)`;
- explicit owner VGPR with `RJ_CONSAN_MOI_OWNER_VGPR`;
- prologue initialization with `RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1`;
- `RJ_CONSAN_MOI_OWNER_SOURCE=workitem_id` default for prologue init;
- `RJ_CONSAN_MOI_OWNER_SOURCE=hw_id` for RDNA4 `HW_ID1` low bits; an explicit
  `RJ_CONSAN_MOI_OWNER_SGPR` is only a debug override. This option is not
  available on gfx950.

The `workitem_id` estimate is adequate for current 1D two-wave controls. It is
not a complete owner derivation for arbitrary 2D/3D local invocation layouts.

The `hw_id` source is useful for targeted inline-shadow experiments because it
is wave-uniform and does not depend on local invocation dimensionality. Its
temporary is chosen above all guest scalar references and descriptor-backed;
full-SGPR kernels fail visibly rather than borrowing an unproven register.

Intended direction:

- Keep 3D workgroup identity.
- Use a robust owner derivation that does not require user-selected registers.
- Preserve `hw_id` as a useful low-level source where appropriate.
- Owner/epoch state is integrated with the R1 scratch/spill policy; broaden the
  identity encoding only when a concrete workload exceeds its packed bounds.

## Barrier And Atomic Semantics

Barriers:

- `record_replay` appends barrier-arrival records and host replay coalesces
  contiguous same-workgroup arrivals into logical epoch advances.
- `inline_shadow` can trampoline supported barriers, execute the original
- `inline_shadow` can trampoline supported barriers, execute the original
  barrier, and increment an epoch VGPR after the barrier. Exact-shadow packing
  masks that monotonically incremented value to 10 bits, so long-running
  kernels use epochs modulo 1024 without corrupting neighboring metadata
  fields. A conflict separated by exactly 1024 barrier epochs can therefore be
  conservatively reported as unordered.
- `sampled` currently does not use barrier records in direct sampled checking.

Atomics:

- MOI treats atomics as ordering events for LDS, not as global-memory race
  checks.
- `record_replay` has host-side release/acquire modeling and a narrow DBI
  atomic-record path.
- `inline_shadow` has one release/acquire address-matching slot per workgroup
  partition.
- `sampled` has no atomic ordering implementation yet.

Current atomic support is intentionally narrow. Broader opcode coverage should
follow concrete semantic controls rather than being inferred from compatibility
runs.

## Why The Current Code Is Not The Final Design

The code intentionally contains several prototype mechanisms:

- optional manual register debug overrides;
- target-local gfx1201 and gfx950 ordinary-VGPR spill backends rather than
  general register-class spilling;
- conservative versioned engine profiles with advanced extensions kept opt-in;
- `MaybeGroup` flat LDS provenance;
- static per-site record and sampled slots;
- one address-scoped inline atomic release slot per workgroup partition;
- `inline_shadow` feature/non-vacuity coverage limited to targeted tests even
  though its broad compatibility sweep is now clean;
- deterministic or scalar-source delay instead of randomized sampling policy;
- IREE correctness tests used as patchability/non-corruption evidence, not race
  reports.

These are acceptable prototype choices because they prove the core DBI
building blocks:

- HSA-level interception works.
- Final native RDNA4 / `gfx1201` and CDNA4 / `gfx950` code can be decoded and
  patched for the explicitly supported forms.
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
- IREE RDNA4 / `gfx1201` and CDNA4 / `gfx950` selected workloads under
  SuperCollider and MOI modes.
- Broader IREE e2e inventories under target-aware profiles. The current gfx950
  SuperCollider checkpoint is 257 ordinary passes out of 259; the other two
  rows are typed `s_trap 0` sanitizer mismatch outcomes, with no corruption,
  replacement-object loader failure, or timeout.
- Broader 209-test IREE e2e compatibility sweeps under all three MOI engines
  without register-number configuration.
- Guarded TileAndFuse, scan, and softmax subsets under all three engines;
  inline shadow remains narrower in supported access forms despite the clean
  compatibility sweep.

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
- It does not prove unsupported instruction families are instrumented.

## Remaining Engineering Boundary

The resource path is in place: non-spill allocation, gfx1201/gfx950 spill-backed
access/barrier/atomic probes, zero-to-nonzero dispatch scratch,
persistent-state fallbacks, scalar/special-state policy, compatible
shared-function assignments, and bounded outcome summaries.

Spill reconnaissance started from Kunwar Grover's `text-relocation-land`
branch:

- `origin/users/Groverkss/text-relocation-land`

That branch contributes the `SpillManager` slot allocator but no target spill
instructions or spill/fill placement. ConSan now reuses that allocator beneath
its gfx1201 and gfx950 B32 backends and the MOI probe integrations.
`origin/users/Groverkss/dbt-tooling` and
`origin/users/Groverkss/dbt_interposer` remain useful background references;
later work is broader shared infrastructure and target coverage rather than
another ConSan allocator.

If ConSan needs SGPR spilling, assume it is not already covered there. Implement
only the minimal SGPR support needed for the current probe family, keep it
isolated, and prefer deleting or replacing it when shared rocJITsu spilling
lands.

The gfx1201 profile-by-tier qualification and broad-turn-on acceptance are
complete. gfx950 focused and selected-workload qualification is complete, and
its broad SuperCollider inventory is classified; record/replay, sampled, and
inline-shadow broad qualification remain separate in-progress gates. Native
work for gfx942 and gfx1250 remains deferred.
