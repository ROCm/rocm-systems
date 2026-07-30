# ConSan design

ConSan is rocJITsu's dynamic binary instrumentation (DBI) sanitizer for AMD
LDS/shared-memory concurrency bugs. Its MOI flavor is Memory-Ordering
Instrumentation. ConSan intercepts GPU code-object loads through the HSA tools
interface, inspects final native machine code, and loads a validated patched
replacement when the selected flavor and engine can instrument it.

ConSan has target-specific native instrumentation for `gfx942`, `gfx950`,
`gfx1201`, and `gfx1250`. It does not translate kernels between GPU ISAs; it
patches the final code object for the architecture that will actually run.
Workgroup-LDS capacity comes from the runtime agent. The simulator JSON is the
source of truth for offline execution, so no target-specific LDS size is baked
into gfx942 probes.

This document describes the current implementation, its invariants, and its
explicit limitations. [FLAVORS.md](FLAVORS.md) gives a phase-by-phase
conceptual comparison of the SuperCollider flavor and three MOI engines,
including the device, deferred, and host responsibilities.
[SPILLING.md](SPILLING.md) describes ConSan's register selection, ownership,
and runtime integration; the reusable RocJitsu backend is documented separately
in [AMDGPU register spilling](../spilling.md).

## Current status at a glance

| Area | Implemented today | Boundary or direction |
| --- | --- | --- |
| Interception | HSA-tools hook via `HSA_TOOLS_LIB`. | Keep HSA-tools as the main path. |
| Architecture | Target-specific native instrumentation for `gfx942`, `gfx950`, `gfx1201`, and `gfx1250`. | Keep target capabilities and encoders explicit; do not translate between targets. |
| Public selection | Loading the hook selects MOI Record/Replay. `RJ_CONSAN_MODE` selects alternatives and `RJ_CONSAN_POLICY` selects default or strict completeness checks. | Keep flavor and engine as implementation concepts beneath a small public interface. |
| SuperCollider | Delayed redundant LDS and admitted group-flat observations with an automatic mismatch marker, including gfx1250 dynamic-stack group-FLAT probes under full register pressure. | Keep as a complementary perturbation/value-instability flavor. |
| MOI Record/Replay | Bounded device records plus host replay. This is the recommended starting engine. | Preserve clear reference/debug semantics while making snapshot limits explicit. |
| MOI Inline Shadow | Immediate supported-form shadow checks for admitted LDS accesses, barriers, and selected atomic ordering. | Broaden proven instruction and ordering coverage without weakening typed exclusions. |
| MOI Sampled | Deterministically selected causal windows plus deferred host analysis. | Improve statistical sensitivity while keeping clean runs explicitly inconclusive. |
| Ordinary operation | The `standard-v1` settings select all admitted supported sites and allocate registers and reports automatically. | Expand instruction and architecture breadth without introducing program-specific setup. |
| Registers | Owner-scoped liveness plans use dead or fresh registers and target-specific VGPR spilling where required; special state is preserved explicitly. | Extend reusable spilling only for concrete register classes and target needs. |
| Diagnostics | Bounded inline and sampled diagnostics plus resource, overflow, and unsupported-site summaries; compact shadow words limit prior-lane detail. | Preserve bounded output while improving precision and presentation. |
| Flat/generic LDS | Explicit `likely`/`strict` admission policy over `Group`/`MaybeGroup`, with target-specific normalized group-flat address contracts. | Extend proven provenance conservatively as compiler code shapes broaden. |

## Source map

ConSan is contained under focused subdirectories wherever the code is specific
to the sanitizer. Shared files retain only reusable binary-patching machinery
or small integration calls. Paths in this section are relative to
`emulation/rocjitsu/`:

```text
lib/rocjitsu/src/rocjitsu/
├── code/patch/
│   ├── consan/                         # ConSan analysis and transformations
│   ├── instruction_sequence.{h,cpp}    # Reusable instruction composition
│   ├── *_instrumentation_builder.h     # Target-specific probe sequences
│   └── spill_manager.{h,cpp}           # Reusable register spilling
└── hooks/
    └── consan/                         # HSA/DBI runtime integration
tests/
├── consan/                             # Focused CMake test registration
├── dbi/consan/                         # HSA/DBI fixtures
├── patch/consan/                       # Feature-split transformation tests
└── fuzz/                               # Placement and transformation fuzzers
docs/consan/                            # User and implementation documents
```

Primary files:

- `lib/rocjitsu/src/rocjitsu/code/patch/consan/CMakeLists.txt`
  - Adds the focused ConSan transformation sources to `rocjitsu_code`; the
    shared code manifest only enters this subdirectory.
- `lib/rocjitsu/src/rocjitsu/hooks/consan/rj_hsa_dbi_hook_config.cpp`,
  `rj_hsa_dbi_hook_moi_report.cpp`, `rj_hsa_dbi_hooks.cpp`
  - Environment parsing and typed configuration.
  - HSA-tool-owned MOI report-buffer allocation and teardown summaries.
  - HSA interception, code-object transformation, and dispatch integration.
  - Private shared declarations live in `rj_hsa_dbi_hook_internal.h`.
- `lib/rocjitsu/src/rocjitsu/hooks/consan/rj_hsa_dbi_replay_provenance.h`,
  `rj_hsa_dbi_sampled_sync.h`
  - Focused runtime helpers for replay provenance and sampled causal state.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan/consan.h`
  - Compatibility umbrella for feature-split public option, code-object,
    resource, fault/synchronization, and result type fragments.
  - Flavor, MOI engine, delay mode, owner-source enums.
  - Decoded native DS, flat, barrier, fence, atomic, and MOI candidate records.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan/consan.cpp`
  - Thin transformation orchestrator. Ordered `consan_*.inc` implementation
    fragments isolate code-object analysis, placement, synchronization
    analysis, fault injection, SuperCollider LDS/flat lowering, composition,
    and final validation while retaining one private translation-unit
    boundary.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan/consan_types.cpp`
  - Public flavor, outcome, disposition, and parser utilities kept separate
    from the transformation core.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan/consan_moi.h`
  - Compatibility umbrella for feature-split MOI type, report-layout,
    Record/Replay, Inline Shadow, and Sampled model fragments.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan/consan_moi_abi.h`
  - Report layouts shared by injected GPU code and the host-side reader:
    exact-shadow, sampled, diagnostic, access, barrier, atomic, and fence
    records.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan/consan_moi.cpp`
  - Thin MOI orchestrator. Ordered `consan_moi_*.inc` implementation fragments
    isolate candidate discovery, placement, shared emission, prologues,
    Record/Replay, Inline Shadow, Sampled, barrier, and atomic lowering.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan/consan_moi_model.cpp`
  - Host-side Record/Replay compaction and replay semantics.
  - Sampled metadata, publication, causal-window, and watchpoint models.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan/consan_moi_report_plan.cpp`
  - Inventory-derived MOI report-buffer layouts and capacity planning.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan/consan_resource.*`
  - Per-owner register requests, descriptor growth, and spill-plan selection.
- `lib/rocjitsu/src/rocjitsu/code/patch/instruction_sequence.*`
  - Reusable checked instruction-sequence composition used by probe builders.
- `lib/rocjitsu/src/rocjitsu/code/patch/{cdna3,cdna4,gfx1250,rdna4}_instrumentation_builder.h`
  - Target-specific instruction encoders used by injected probes, isolated
    from the architecture-generic `instruction_builder.*` surface.
- `lib/rocjitsu/src/rocjitsu/code/patch/trampoline_builder.*`,
  `kernel_text_layout.*`, `code_object_patcher.*`, `spill_manager.*`
  - Reusable patch-placement and DBT utilities used by all ConSan engines.
    `spill_manager.*` also emits transactional target-specific B32 VGPR
    save/restore batches, performs kernel-local fixed-private descriptor
    growth, and deliberately leaves non-authoritative AMDGPU MessagePack notes
    untouched.

Test anchors:

- `tests/consan/CMakeLists.txt`
  - ConSan unit, fuzz, HIP-binary, and live-GPU test registration. The shared
    test manifest only invokes the focused registration helpers.
- `tests/patch/{cdna3,cdna4,gfx1250,rdna4}_instrumentation_builder_test.cpp`
  - Exact encoding and rejection coverage for the target-specific builders.
- `tests/patch/consan/`
  - Shared synthetic ELF fixtures plus feature-centric core, analysis,
    resource, fault-injection, SuperCollider, Record/Replay, Inline Shadow, and
    Sampled unit tests.
- `tests/dbi/consan/`
  - ConSan HSA/DBI fixtures and focused runtime contract tests. Generic DBI
    fixtures remain in the parent `tests/dbi/` directory.

## Public mode and policy model

Loading the HSA hook activates ConSan. The ordinary default is MOI
Record/Replay. Alternatives use one variable:

```sh
RJ_CONSAN_MODE=record-replay
RJ_CONSAN_MODE=inline-shadow
RJ_CONSAN_MODE=sampled
RJ_CONSAN_MODE=supercollider
```

Focused validation can require complete instrumentation and evidence:

```sh
RJ_CONSAN_POLICY=strict
```

Strict policy rejects unsupported or incomplete instrumentation, requires real
patches and MOI evidence, and rejects overflow. It does not make race
diagnostics fatal. The former `RJ_CONSAN_FLAVOR`, `RJ_CONSAN_MOI_ENGINE`, and
`RJ_CONSAN_MOI_BACKEND` inputs remain deprecated transition aliases.

The public terminology is:

- `supercollider`: simple perturbation plus redundant-access checking.
- `moi`: structured memory-order instrumentation.
- `record_replay`: bounded device recording plus host-side reference/debug
  replay. It is the recommended starting engine.
- `inline_shadow`: immediate supported-form GPU-side shadow checking.
- `sampled`: statistical MOI with bounded retained causal windows.

## Ordinary MOI operation and boundaries

MOI has the same broad "turn it on" operation as SuperCollider. Enabling ConSan
selects MOI Record/Replay by default; the versioned `standard-v1` settings
supply resources, reports, and synchronization tracking. This does not imply
that every loaded instruction is supported or that every engine has identical
sensitivity.

The implementation boundary is:

- **Scratch register allocation.** Record/Replay, Sampled, and Inline Shadow
  access, barrier, and atomic probes choose per-site dead or fresh
  descriptor-backed VGPR windows and spill allowed live windows when needed.
  Shared helpers use one plan valid for every owning kernel. Scalar and
  persistent state are automatic too. The reusable allocation and gfx1201
  save/restore backends are documented
  in [AMDGPU register spilling](../spilling.md).
- **Owner and epoch state.** `inline_shadow` automatically uses a persistent
  descriptor-backed pair or derived-owner/private-epoch state. The packed
  identity contract is intentionally bounded and architecture-sensitive.
- **Report-buffer capacity.** MOI can use HSA-tool-owned auto report buffers,
  which is the ordinary path for applications. After final-code
  inventory, the hook computes an exact engine-specific layout and allocates
  the required bytes under a 128 MiB per-buffer and 256 MiB process ceiling.
  It reports saturation, undercoverage, overflow, and dropped evidence
  separately. Expert cap, caller-buffer, and zero-disable overrides remain.
- **Versioned ordinary settings.** `record_replay`, `inline_shadow`, and
  `sampled` use `standard-v1`: all admitted supported sites,
  inventory-derived reports, automatic resources, and barrier/atomic tracking.
  Dynamic records and immediate sampled checking remain explicit expert
  extensions. Startup logs identify the settings version and override source.
- **Instruction coverage.** Inline Shadow handles supported native multi-cell
  DS ranges and admitted zero-offset group-flat forms through the same shadow
  publisher, including the admitted d16 forms. Unsupported encodings,
  address shapes, and provenance remain visible typed exclusions rather than
  speculative instrumentation.
- **Patch placement and text growth.** All MOI engines and SuperCollider use a
  shared transactional planner for inline padding, local caves, and appended
  caves. A failed plan does not leave partial descriptor/text mutations.
- **Flat/generic LDS provenance.** Compilers can emit flat accesses for source
  `__shared__` memory. Inventory distinguishes proven `Group`, heuristic
  `MaybeGroup`, private, and unknown forms. `strict` admits only proven `Group`;
  `likely` also admits `MaybeGroup`, with exclusions reported explicitly.
- **Barrier and atomic ordering.** Barriers and selected atomics are ordering
  evidence for LDS races, not global-memory checks. Record/Replay is the
  semantic oracle; Inline uses bounded address-scoped release/acquire state;
  Sampled attaches supported barrier, atomic, and fence evidence to its causal
  windows. Every ordinary MOI engine enables admitted barriers and atomics.
- **Diagnostics.** `inline_shadow` emits bounded first-N diagnostics with
  instruction offsets, access kinds, owners, epoch, LDS byte ranges, the
  current conflict EXEC mask, and visible overflow. The prior writer's lane
  mask is unavailable in the compact exact-shadow word and is reported as
  unknown/zero.
- **Runtime sampling policy.** `sampled` leaves every admitted static site
  patched under ordinary settings, then applies deterministic runtime
  selection before deferred host scanning. An immediate adjacent-range
  in-kernel check remains an expert extension.
- **Architecture dispatch.** `gfx942`, `gfx950`, `gfx1201`, and `gfx1250` use explicit
  ISA-specific capability checks and encoders rather than implicit RDNA4
  assumptions.

The recommended ordinary invocation is:

```sh
env HSA_TOOLS_LIB=/path/to/librocjitsu_dbi_hooks.so \
  ./application
```

Use `inline_shadow` or `sampled` only when their different sensitivity,
retained-state, and overhead trade-offs are desired. Ordinary runs do not
require site, register, report-buffer, barrier, atomic, or sampling selection.

### Resource-plan fallback telemetry

The HSA hook emits a `ConSan MOI resources` aggregate and one
`ConSan MOI resource-alternative` record for every structurally distinct
fallback attempt. These records are qualification evidence and are visible at
`RJ_CONSAN_LOG=1`.

The `attempt` field starts at zero for each resource plan and follows stable
planning order: attempts retained from the initial plan, the outer
spill-backed recovery, then attempts nested inside that recovery. The kinds
are:

- `guest_operand_overlap_spill`: retries allocation with the admitted guest
  operand overlap and a spill-backed save/restore window.
- `spill_backed_operand_recovery`: retries a clobbering Inline Shadow load with
  its compact address-recovery transaction shape.

Each attempt has exactly one outcome:

- `selected`: the final resource plan uses this attempt. At most one attempt
  per plan is selected.
- `rejected`: the attempt could not produce the required resource plan.
- `superseded`: the attempt worked, but a later outer attempt replaced it.
- `contributed`: the nested attempt enabled the selected outer attempt.
- `vetoed`: the attempt worked locally, but a later whole-plan constraint
  rejected it.

The aggregate attempt count equals the sum of these five outcome counts. The
fault runner validates the detailed vocabulary, chronological attempt numbers,
per-outcome totals, and the at-most-one-selected invariant against the
aggregate. It retains at most 4096 detailed records while still validating all
records, and reports the total, truncated count, completeness, and any parse
error in the `resource_plan_alternative_*` metrics.

## Interception and code-object flow

The HSA hook wraps the code-object load path. For each memory-backed code
object, it:

1. Refreshes runtime configuration from environment variables.
2. Reads the original code-object bytes.
3. Runs ConSan inventory and patch planning.
4. Emits modified bytes when the selected flavor and engine can patch at least
   one site.
5. Loads the replacement memory-backed code object.
6. Logs a compact summary when `RJ_CONSAN_LOG` is enabled.

This path does not translate the program to a different ISA. The target remains
the original native code object for the GPU that will execute the kernel.

`RJ_CONSAN_REQUIRE_PATCH=1` is the main non-vacuity guard. It rejects a code
object when ConSan finds supported candidates for the selected flavor and
engine but cannot patch any such candidate. It still lets unsupported code
objects load normally.

`RJ_CONSAN_DUMP_DIR=/path` writes original and patched code objects for
inspection.

## Shared DBI constraints

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

- All ordinary flavors and engines select resources automatically.
  SuperCollider's smaller probes often fit dead or fresh registers, while MOI
  also uses spill-preserved victim windows when necessary.
- Static MOI Record/Replay and Sampled access probes use a read-only,
  kernel-scoped CFG/liveness plan. Each site first searches dead VGPRs within
  its current descriptor allocation, then a fresh range above all guest
  references, growing only the owning descriptor when needed.
- Symbol-backed code ranges exclude alignment padding from CFG decoding, so
  the same planning path works on normal multi-kernel HIP code objects.
- Direct-kernel and reachable shared-helper static Record/Replay and Sampled
  sites consume a typed spill-required outcome. They use an appended cave
  containing the spill save, derived-owner setup, original access,
  conservative LDS wait, instrumentation, spill restore, and return.
- Spill plans grow the owning kernel descriptor. The HSA hook also associates
  the absolute requirement and any site-local dynamic-frame addend with the
  loaded kernel object. It rewrites the AQL dispatch packet to the greater of
  the descriptor minimum and the launch-selected private depth plus the
  maximum site-local frame. Thus a compiled private size of zero can become
  nonzero without treating the descriptor or offline JSON as the source of
  truth for a runtime-configurable stack. ROCR consumes those descriptor and
  packet fields; ConSan does not read or rewrite the duplicated MessagePack
  private-size entry.
- Dynamic record, barrier-record, inline diagnostic, and inline atomic acquire
  paths allocate fresh descriptor-backed scalar windows. Scalar VCC snapshots
  make restoration independent of active lanes; SCC is captured before the
  probe and restored last. Explicit SGPR knobs remain debug overrides.
- When no explicit Inline Shadow owner/epoch pair is supplied, ConSan first
  places a dedicated pair above guest references and the selected scratch
  window, replans scratch with that pair forbidden, and injects a kernel-entry
  initializer. If no pair fits, it derives owner per access and keeps epoch in
  a persistent private dword. The epoch slot precedes an independently aligned
  ephemeral spill zone shared by access, barrier, and entry-prologue leases.
  Shared helpers use one representation for every reachable owner; private
  workitem-derived ownership additionally requires the owners to agree on wave
  size.
- The target-specific spill backends allocate stable slots through
  `SpillManager`, emit address-free `scratch_store/load_b32` batches with
  conservative split waits, and grows only the selected descriptor's fixed
  private segment. For the supported compiler dynamic-stack convention,
  Inline, Record/Replay, and Sampled probes on CDNA3/CDNA4 and RDNA4-family
  targets can instead create a site-local frame, preserve the caller frame and
  SCC, spill the borrowed VGPR window, and grow dispatch backing by the maximum
  added depth. Allocation excludes the backend's named stack-top and
  frame-base registers rather than assuming a numeric preserved-SGPR range.
- On gfx1250, a saturated SuperCollider group-FLAT probe bootstraps the same
  kind of site-local frame without assuming a dead or permanently preserved
  SGPR window. It saves the full VGPR victim window first, uses four saved
  VGPRs as transient reservoirs for the live VCC-save pair, frame base, and
  SCC, then restores all guest state. Shared helpers use this recipe only when
  every owner is dynamic; mixed owner stacks produce a typed rejection when
  the spill recipe is actually needed.
- Static Record/Replay, Sampled, and descriptor-full Inline Shadow access
  probes consume those backends for direct kernels and reachable shared helpers.
  A shared spill starts above the maximum original private extent and grows
  every owner to the same required size. Mixed fixed/dynamic shared ownership,
  unsupported engine/target recipes, and unresolved indirect ownership remain
  unsupported. Dynamic full-VGPR RDNA4 Sampled owners require a persistent
  scalar tuple proven untouched over each complete owner scope plus an
  entry-local dead VGPR pair; failure of either proof is typed and
  transactional. No general SGPR spill backend is present.

The generic allocator, target-specific spill sequences, descriptor helper,
implementation provenance, and backend tests now live outside ConSan and are
documented in [AMDGPU register spilling](../spilling.md). ConSan owns the
victim-selection, multi-owner, code-placement, logging, and dispatch policies
described in [SPILLING.md](SPILLING.md).

If ConSan needs SGPR spilling before rocJITsu has a shared implementation, the
right near-term response is a minimal ConSan-local SGPR spill/fill path for the
specific probe shape that needs it, not a comprehensive spill allocator.

This resource path is deliberately narrower than general register-class
spilling. New support should extend the reusable DBI infrastructure only when
a concrete probe and target require it.

Barrier/atomic VGPR patchers now consume the same plan, and bounded HSA logs
report explicit, dead, descriptor-growth, spill, and unsupported outcomes plus
planned and emitted spill bytes.

### Coverage disposition and lowering ledger

MOI coverage is derived from one per-final-code-site ledger rather than from
the candidates or patches that survive filtering. Each record first retains a
semantic `disposition` (`not_applicable`, `supported`, or `unsupported`) and a
stable semantic `reason`. A second, independent lowering layer is finalized
after register planning and patch emission:

- semantic exclusions retain `NotApplicable` or `Unsupported` and never become
  resource or placement failures;
- a supported site with an unsupported register plan becomes
  `ResourceFailed`, with category `UnsupportedResourcePlan` and the exact
  `ConSanRegisterPlanReason`;
- a supported site with no emitted patch becomes
  `PlacementOrLoweringFailed`, with `InstrumentationPatchMissing` rather than
  a free-form warning;
- an emitted site becomes `Patched`. A Sampled barrier body covering a typed
  multi-event sequence marks every exact member event patched, not only the
  branch anchor.

For MOI, the hook consumes these durable outcomes symmetrically for access,
barrier, atomic, and fence counts. For each kind it enforces the accounting
shape

```text
discovered = supported + unsupported
supported = selected + expert_limit_omitted
selected = patched + resource_failed + placement_or_lowering_failed
```

and completeness requires every failure or omission term to be zero. Relevant
unsupported-only objects therefore remain applicable and incomplete. The hook
emits one `coverage_site` record for every semantically relevant site,
retaining kind, semantic disposition/reason, lowering outcome/reason, detailed
register reason, container ownership, kernel/function scope, text offset, and
mnemonic. Only `NotApplicable` records are omitted, preventing unrelated
instructions from adding noise while preserving exact patched eligibility.

The user-facing vocabulary and exact names are listed in
[USAGE.md](USAGE.md#coverage-and-diagnostics).

The aggregate record carries explicit `flavor` and `engine` identity.
SuperCollider derives access totals from its LDS/flat inventory and emitted
patches, but does not manufacture MOI lowering dispositions. Its aggregate
accounting remains strict; only MOI readers participate in exact
`coverage_site` cardinality reconciliation.

## SuperCollider flavor

### Purpose

SuperCollider is the simplest flavor. It preserves
the original memory access, inserts a delay, repeats or reads back the same LDS
address, compares values, and reports a mismatch.

The NVIDIA Research paper [“SuperCollider: Scalable and Effective Data Race
Detection for
CUDA”](https://research.nvidia.com/publication/2026-06_supercollider-scalable-and-effective-data-race-detection-cuda)
by Stephenson et al. (PLDI 2026) describes the redundant-read idea as issuing
"a redundant read to the same address" after delay. ConSan implements that core
technique after register allocation, directly in AMD final native code. Its
binary rewriting, reporting path, supported memory spaces, and current semantic
contract are ConSan-specific rather than a wholesale port of the CUDA system.
Target-specific decoding and lowering are selected explicitly by architecture.

### Current algorithm

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

- Default `RJ_CONSAN_SC_REPORT_MODE=auto`: the hook allocates and zeroes one
  device-visible sticky marker for each relevant code object, patches its
  address, reads and summarizes it at teardown, and frees it. Allocation
  failure records incomplete analysis and loads original code under fail-open
  behavior; fail-closed or require-patch policy rejects it. It never silently
  falls back to a trap.
- Explicit `RJ_CONSAN_REPORT_BUFFER=0xADDR`: write
  `RJ_CONSAN_REPORT_MARKER` to one caller-owned device-visible word and
  continue.
- Expert `RJ_CONSAN_SC_REPORT_MODE=trap`: execute `s_trap 0` on mismatch.

The marker-buffer ABI is intentionally small. It provides a non-trapping
mismatch path, but it does not record PC, lane, LDS address, values, or counts.
The marker is measured redundant-access instability, not a causal race
diagnostic. A race-free program can legitimately advance another wave between
the original and repeated access. Users should therefore interpret this
channel separately from MOI diagnostics and compare repeated known-correct and
suspect runs.

### Fault and perturbation composition

Barrier and atomic fault injection compose with SuperCollider perturbation as a
staged transaction. ConSan inventories the pristine image and retains the exact
selected candidate, sequence, anchor, container, and descriptor owner in a
private internal plan. It validates the mutation, instruments that staged
image, and then validates the complete pristine-to-output transformation. The
internal carrier is not a public option and cannot be supplied by host controls.
If either stage is unsupported or invalid, ConSan rolls back the mutation and
returns no replacement image.

Barrier moves translate the selected edge into the owned whole-pair trampoline;
barrier drops fail closed if they destroy that edge. Atomic address and scope
faults normally mutate the atomic member while perturbation remains anchored on
an outer cache edge. Order weakening can remove the exact cache operation that
is also the perturbation anchor. In that overlap case the perturbation
trampoline carries only the mutation's NOP replacement and validation rejects
any attempt to resurrect the removed cache operation. Partial atomic overlaps
are rejected.

The proof establishes exact identity, ownership, byte accounting, branch
placement, mutation retention, and rollback. SuperCollider may expose value
instability caused by a weakened sequence; it does not claim exact
happens-before reconstruction.

### Current instruction coverage

The normative target-by-engine form list is
[CAPABILITIES.md](CAPABILITIES.md). Native LDS check/trap covers the common
single- and dual-range forms plus the target-specific subword, 96-bit, and
transpose extensions named there. Likely group/LDS FLAT check/trap covers the
target-specific admitted 16, 32, 64, and 128-bit VFLAT encodings.

Current exclusions:

- ordinary global-memory instrumentation;
- unsupported group-FLAT widths such as b8 and b96;
- arbitrary flat accesses with unknown provenance;
- atomics as SuperCollider duplicate-access checks;
- async copies;
- same-value lost-update checks within one wave;
- structured race reports.

### Flat/VFLAT rationale

Flat support is in scope because real compiled HIP helper code can access LDS
through flat/generic pointers. Source-level `__shared__` does not guarantee
that final machine code will use `ds_*` instructions. Once optimization has
materialized a generic pointer, the final instruction selector may emit
`flat_load_*` or `flat_store_*`; the pointer value decides whether the access
reaches the LDS aperture.

ConSan therefore classifies flat sites using a machine-code provenance tracker.
The reported classifications are:

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
both classifications. `strict` admits only `Group` for investigations that
prefer provenance precision over flat-site recall. Inventory and verbose site
logs retain the classifications independently of this selection policy, and
skipped-candidate warnings count strict-policy exclusions.

For an admitted flat group pointer, ConSan's target-specific LDS normalization
contract identifies the unsigned byte offset within the LDS aperture and keeps
the remaining address bits as provenance evidence only. Static VFLAT `ioffset`
bytes are added before rounding the byte interval to 4-byte shadow cells. Sites
whose encoding or provenance cannot satisfy the target's contract remain
unpatched.

### Current limitations

- Delay is deterministic or scalar-source based rather than randomized.
- The automatic marker reports only that some redundant observation changed;
  it does not retain the address, lane, values, count, or causal peer.
- `MaybeGroup` flat provenance is heuristic.
- Instruction lowering and VGPR spilling use target-specific backends and are
  exercised primarily by the larger MOI probes.

## MOI flavor

### Purpose

MOI is the structured race-detection flavor. It models accesses, owners,
epochs, barriers, and selected atomic ordering events. Within its admitted
forms and retained capacities, it can provide causal attribution that
SuperCollider's value-instability marker cannot.

MOI is LDS-focused in the current design. Global memory is intentionally out
of scope except where selected atomics/fences provide ordering evidence for
LDS communication.

### Shared semantic model

MOI records or computes:

- access kind: read or write;
- LDS byte offset and byte count;
- 4-byte LDS shadow-cell range;
- workgroup identity;
- owner identity within a workgroup;
- epoch/order state;
- source instruction offset;
- selected synchronization and atomic events.

The supported-form conflict predicate is, in simplified terms:

- same workgroup;
- overlapping LDS cell/range;
- at least one write;
- different owner;
- same unordered epoch, unless atomic/barrier semantics establish ordering.

Record/Replay is the semantic reference. Inline Shadow and Sampled should match
it where they claim the same semantics and document lower fidelity where they
do not.

### Report-buffer ABI

The MOI report buffer starts with `ConSanMoiReportHeader` and then engine-
specific sections. The header includes counts, capacities, dropped-record
signals, and offsets/capacities for:

- access records;
- barrier records;
- atomic records;
- diagnostics;
- exact-shadow entries;
- inline atomic release slots;
- inline causal snapshots and acquired-epoch tokens;
- sampled watchpoints.

Report-buffer sources:

- `RJ_CONSAN_MOI_REPORT_BUFFER=0xADDR` and
  `RJ_CONSAN_MOI_REPORT_BUFFER_SIZE=N`: caller-supplied buffer.
- With no caller buffer, the HSA tool inventories relevant MOI sites, plans the
  exact engine layout, allocates those bytes below the configured ceilings,
  and summarizes it at teardown.
- `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=N`: expert allocation cap; the ordinary
  planner still requests exact inventory-sized bytes below it. Explicit zero
  disables auto allocation. Dynamic access append requires an explicit finite
  cap because its execution count is not statically predictable.

The auto-buffer path is the practical choice for applications that cannot add
a sanitizer kernel argument.

#### Bounded-memory policy

Ordinary `standard-v1` allocation is governed by an exact, checked plan derived
from final-code inventory rather than an engine-wide default:

| Engine | Inventory-derived report requirement |
| --- | --- |
| Record/Replay | Header/alignment, a report-wide dispatch directory with 2× open-addressing headroom, a report-wide access-identity table sized from the admitted logical ranges and adaptive dispatch/owner diversity factors with the same headroom, and exact capacities for every enabled barrier, atomic, fence, and finite diagnostic region. Exhaustive per-lane dynamic append is excluded from the ordinary completeness contract. |
| Sampled | Header/alignment plus the admitted logical ranges, each range's configured bounded dynamic-window bank, paired synchronization metadata, pending-acquire state, and finite diagnostics. |
| Inline Shadow | Header/alignment plus 16 dispatch banks, each with one versioned exact-shadow slot for every four-byte cell in the maximum declared LDS span of the owning kernels, finite diagnostics, and only the release, snapshot, and acquired-token tables required by enabled ordering instrumentation. |

All additions, multiplications, alignments, and conversions are checked. One
automatic report buffer may require at most **128 MiB**, and the sum of live
automatic report buffers in a process may be at most **256 MiB**. These are
hard safety ceilings rather than allocation quanta. The allocator reserves the
exact planned bytes below them and accounts the reservation against the
process ceiling before exposing its device address.

Arithmetic overflow, either ceiling, or allocation failure yields the typed
`insufficient_report_capacity` result with a stable subreason and required,
available, and live-byte values. It is a static incomplete verdict: ordinary
operation neither truncates the admitted inventory nor silently disables an
event kind or window to fit. Explicit expert sizes are subject to the same
ceilings for now. An explicit zero continues to disable automatic allocation.

`RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1` remains a bounded expert experiment.
No static estimator can infer its dynamic per-lane execution volume, and an
allocation below the safety ceiling is not a completeness claim. Its capacity,
visible records, drops, and `dynamic_complete` verdict are always retained.

The memory ledger retains, per code object and in process aggregates: engine
and settings version, admitted inventory, required and allocated bytes,
required and allocated capacity for every ABI region, current and peak live
auto-report bytes, allocation/free outcome, descriptor LDS growth,
private-spill growth, and every saturation, undercoverage, overflow, and
dropped-evidence counter. This ledger is part of the runtime summary, not
verbose-only debugging output.

Runtime self-checks:

- `RJ_CONSAN_MOI_REQUIRE_RECORDS=1`: fail at teardown if no auto buffer has
  visible access, barrier, atomic, diagnostic, exact-shadow, or sampled data.
- `RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1`: fail if no diagnostic/conflict signal
  is observed.
- `RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1`: fail if any diagnostic/conflict signal
  is observed.
- `RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT=1`: stricter `record_replay` guard
  that requires host replay to emit a conflict.
- `RJ_CONSAN_MOI_FORBID_OVERFLOW=1`: fail at teardown if an auto buffer dropped
  access, barrier, atomic, or diagnostic records, or if a Record/Replay
  dispatch/owner bank saturated. Overflow and saturation are always printed to
  stderr even without this guard.

### Record/Replay engine

`RJ_CONSAN_MODE=record-replay` is the recommended starting engine and the
reference model for the other MOI engines.

Current implementation:

- Patches every admitted supported native DS and likely-group-flat access under
  ordinary settings.
- Emits `ConSanMoiAccessRecord` entries.
- Supports bounded report-wide dispatch and access-identity tables by default.
- Supports dynamic per-lane append with
  `RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1`.
- Records dynamic event indexes.
- Patches supported target barrier sites by default; explicit
  `RJ_CONSAN_MOI_TRACK_BARRIERS=0` is an expert compatibility override.
- Records the target's admitted `flat_atomic*` subset by default; explicit
  `RJ_CONSAN_MOI_TRACK_ATOMICS=0` disables it for focused bring-up.
- Replays visible records on the host into exact-shadow diagnostics.
- Coalesces contiguous same-workgroup barrier-arrival runs into one logical
  epoch advance.
- Models selected release/acquire atomic ordering on the host.

Important current simplifications:

- An automatic ABI-v12 layout uses a report-wide 2,048-slot dispatch directory,
  providing 2× open-addressing headroom for 1,024 anticipated hardware
  dispatch identities. A separate report-wide access table is sized from the
  number of logical access ranges, 16×16 default dispatch/owner diversity
  factors, and the same 2× headroom, then rounded to a power of two. For large
  code objects the fitter halves both diversity factors together until the
  complete admitted inventory fits the unchanged 128 MiB per-buffer ceiling.
  These factors size anticipated concurrency; they are not hard-coded
  dispatch or owner buckets. Hot sites can use capacity left idle by cold
  sites.
- Both tables use bounded triangular probing. The dispatch directory qualifies
  the complete 64-bit hardware dispatch identity. Each 80-byte access record
  is keyed by its directory slot, an explicit static logical-range token, all
  three workgroup coordinates, and the canonical wave owner. A 64-bit
  compare-and-swap claims the access slot; the payload is written before an
  atomic `access_kind` commit consumed by host replay. A repeated publisher
  leaves the immutable record alone only after observing that commit and
  matching the full identity.
- A probe is capped at 256 slots independently of the allocated table size.
  Exhausting either probe sets both the general saturation bit and a typed
  dispatch-directory or access-table bit. Saturation is deliberately
  report-wide and fail-closed: later automatic captures bypass the tables,
  teardown makes dynamic completeness false, and no workload-specific fallback
  silently reuses an unrelated slot.
- Zero is the unpublished claim-token sentinel. The reversible token encoding
  has one dispatch-ID preimage for zero; that identity is rejected through the
  same fail-closed saturation path rather than silently publishing an
  ambiguous record.
- Caller-owned size-derived layouts remain single-bank first-light buffers.
  They are bounded and do not gain automatic multi-dispatch retention, but
  occupied-slot reuse is now qualified by exact dispatch and workgroup
  identity rather than silently accepting a different workgroup. They capture
  the same exact three-coordinate workgroup tuple at kernel entry as automatic
  layouts; single-bank describes retention, not a weaker identity lifetime.
- Dynamic access append automatically allocates its EXEC/VCC/SCC scalar window;
  `RJ_CONSAN_MOI_EXEC_SAVE_SGPR` is an optional debug override.
- Dynamic append can consume records quickly because it writes per active lane.
- Some candidates are skipped near compiler-generated EXEC-mask regions until
  control-flow and liveness handling are stronger.
- Atomic DBI support is narrow and serves as LDS ordering evidence, not
  global-memory race detection.

Design role:

- Provide the clearest inspectable semantics with modest program overhead.
- Serve as the reference model for Inline Shadow and Sampled behavior.
- Report bounded-snapshot limits honestly: each dispatch/owner bank retains
  first-light evidence rather than execution history, so a clean replay is not
  proof of race freedom.
- Preserve clarity rather than optimizing away the reference semantics.

### Inline Shadow engine

`RJ_CONSAN_MODE=inline-shadow` is the immediate supported-form GPU-side
engine. It updates and checks shadow state during kernel execution instead of
logging each access for host replay.

Current implementation:

- Uses the shared MOI report buffer.
- Uses an inventory-sized ABI-v12 layout with 16 dispatch-selected banks, each
  containing one versioned slot per four-byte cell in the maximum declared LDS
  span of the owning kernels, plus only the required finite diagnostic and
  ordering regions. Banking keeps unrelated concurrent dispatches off the same
  hot slot in the external table. A caller-owned partial shadow cannot qualify
  merely because one execution touches less LDS than the owner declares.
- Instruments decoded native scalar, B64, B128, d16, and two-address LDS
  loads/stores, publishing every rounded 4-byte cell in each access range.
- Instruments supported zero-offset flat/VFLAT loads and stores admitted by
  the configured flat-provenance policy. The low address VGPR is normalized as
  the LDS byte offset and feeds the same cell-range publisher as native DS.
- Uses a versioned compare/exchange transaction to publish external exact
  shadow state and obtain one stable prior entry. Native LDS paths use a
  workgroup-local 64-bit exchange when a local mirror fits.
- Reports a conflict when the prior entry is non-empty, from a different owner,
  in the same epoch, and not read/read.
- Automatically initializes persistent owner/epoch VGPRs at each owning kernel
  entry. When shared atomic helpers need launch identity, it also persists an
  exact nonzero workgroup key there instead of rereading guest-reusable launch
  SGPRs at a late call site.
- Increments an epoch VGPR after supported barrier sites by default.
- Represents supported atomic release/acquire ordering with bounded
  direct-mapped release, causal-snapshot, and pair-scoped acquired-token
  tables. A publisher atomically replaces any stable even release slot;
  acquire lookup imports an edge only after exact dispatch, workgroup, and
  address qualification. Same-wave releases of one object are coalesced behind
  the winning version transaction, while true simultaneous slot collisions
  remain observable as coverage loss.

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
- Entry-captured `workitem_id` is the ordinary owner source. Inline Shadow
  prefers persistent descriptor-backed owner/epoch VGPRs because owner state
  is consumed at every hot access; private scratch is the capacity fallback.
- `hw_id` remains an expert owner source. It is wave-uniform and automatically
  receives a fresh scalar temporary when the kernel has capacity.
- Atomic ordering metadata is finite and direct-mapped; exhausted contention
  retries or a simultaneous collision between distinct objects makes the
  dynamic-completeness verdict false.
- Diagnostics are bounded first-N records, not an unbounded trace.

Design role:

- Provide the strongest current supported-form immediate attribution without
  retaining a full event trace.
- Extend flat/VFLAT encoding coverage beyond the current supported forms while
  preserving the explicit provenance policy.
- Use automatic scratch/spill policy instead of manual register knobs.
- Emit structured bounded diagnostics that are useful without reading raw logs.

### Sampled engine

`RJ_CONSAN_MODE=sampled` is the statistical engine. It retains selected
causal windows instead of attempting to preserve every dynamic event.

Current implementation:

- Uses the ordinary `standard-v1` runtime sampling policy (stride 16,384,
  offset zero) without program-specific settings. The environment variables
  remain expert overrides for controlled statistical campaigns.
- Leaves every admitted supported static access site patched under ordinary
  settings.
- Writes compact 64-bit sampled watchpoint entries directly from DBI probes.
- Assigns one logical sampled range per decoded access range. When runtime
  sampling is enabled and report capacity permits, each logical range receives
  a power-of-two bank of as many as eight immutable dynamic windows. Access
  and barrier probes choose the same bank from dispatch/workgroup identity.
- Packs valid/consumed bits, access kind, owner, epoch, generation, and LDS
  cell range.
- Uses generation zero in direct DBI mode.
- Supports expert static-site subsampling with `RJ_CONSAN_MOI_SAMPLE_STRIDE`
  and `RJ_CONSAN_MOI_SAMPLE_OFFSET`; ordinary settings select every admitted
  static site.
- Can leave every eligible static site patched while deterministically
  selecting runtime accesses with `RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE` and
  `RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET`. The power-of-two policy mixes hardware
  dispatch ID, workgroup, wave, epoch, persistent per-wave sequence, site, and
  address through a strong finalizer before comparing the selected residue; it
  preserves VCC in an automatically allocated scalar pair.
- Treats a repeated claim for the exact stored dispatch/workgroup/epoch/site
  causal identity as the same retained sample. A different identity arriving
  after every immutable bank is occupied is explicit bounded saturation, not
  a malformed record or publication-loss drop. True drops remain separate and
  make the analysis incomplete.
- Publishes typed barrier synchronization metadata into the bank selected by
  the same dynamic identity, so host replay can classify sampled causal
  windows without joining unrelated dispatches.
- Auto-buffer probes publish the buffer generation in every sampled entry.
  Host replay ignores entries from older generations, scans the active entries
  at HSA-tool teardown, and reports sampled conflict counts.
- With `RJ_CONSAN_MOI_SAMPLED_CHECK=1`, logical range `i` checks the
  corresponding bank of the immediately preceding logical range before
  publishing. Matching valid generation and epoch, owner inequality,
  conflicting access kinds, and exact cell range increment the report header's
  sampled immediate-conflict counter on the GPU. The HSA summary and diagnostic
  guards consume that counter without waiting for host pairwise replay.
- Keeps host-side sampled publish/replay helpers as semantic references.

Important current simplifications:

- Runtime selection is deterministic for the full dynamic identity but varies
  over a wave's persistent access sequence. It is statistical coverage, not a
  deterministic detection guarantee.
- The in-kernel checker compares one adjacent logical range/bank and exact
  ranges rather than scanning the table or testing all overlapping ranges. Its
  counter is an immediate signal, not a structured full diagnostic record.
- There is no in-kernel whole-table sampled conflict checker.
- Clean sampled output is inconclusive.
- Owner/epoch values are masked to the current compact 10-bit fields before
  packing.

Design role:

- Provide bounded statistical evidence without retaining an exhaustive event
  history.
- Keep the automatic runtime policy usable without program-specific tuning
  and characterize its detection rate on real faults.
- Improve table-wide and overlapping-range checking without turning the
  ordinary engine into an exhaustive trace.
- Continue documenting that the Sampled engine can miss races.

## Owner and workgroup identity

MOI separates:

- workgroup identity: `(workgroup_x, workgroup_y, workgroup_z)`;
- owner identity: the logical peer inside a workgroup used by conflict checks.

Current workgroup identity is stronger than current owner identity. Every
Record/Replay layout captures the exact 32-bit
`(workgroup_x, workgroup_y, workgroup_z)` tuple at kernel entry and publishes
those three components directly. It therefore has no Inline Shadow packing
limit. Caller-owned layouts retain one first-light bank but use the same exact
tuple and entry-state lifetime contract. Inline Shadow separately captures the
compact key required by its shadow-cell representation.

On CDNA3/CDNA4, ConSan enables the complete x/y/z system-SGPR launch payload
for every patched Record/Replay owner before execution. The entry prologue
captures that full payload, then reconstructs the guest's original compact
system-SGPR suffix (including workgroup-info when present) before returning to
guest code. An absent descriptor dimension is therefore neither assumed to be
zero nor allowed to change the guest ABI. RDNA4-family patching likewise
enables and captures its full launch payload explicitly.

This is an entry-state lifetime invariant, not an engine-specific
optimization: a probe that can execute after arbitrary guest code may consume
an ABI entry value only from state ConSan captured persistently at entry.
Descriptor-selected entry SGPRs, RDNA launch TTMP payload fields, and the
entry workitem-ID VGPR are ordinary guest-reusable state after the prologue.
Record/Replay workgroup identity, Inline Shadow workgroup identity, and the
ordinary MOI owner source all follow this rule. SuperCollider is deliberately
different: it duplicates one instruction at a site and compares the immediate
result there; it carries no dispatch-wide identity from one probe to another.

The ordinary owner source for every MOI engine is
`RJ_CONSAN_MOI_OWNER_SOURCE=workitem_id`. ConSan captures it at kernel entry,
before `v0` becomes reusable guest state, and derives the current estimate as
`workitem_id_x >> log2(wavefront_size)`. Inline Shadow stores the result in its
automatically allocated persistent owner state when possible.

Expert/debug alternatives are:

- an explicit owner/epoch VGPR pair through `RJ_CONSAN_MOI_OWNER_VGPR` and
  `RJ_CONSAN_MOI_EPOCH_VGPR` (Record/Replay rejects either one in isolation);
- explicit prologue initialization through
  `RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1`; and
- `RJ_CONSAN_MOI_OWNER_SOURCE=hw_id`, which uses RDNA4 `HW_ID1` low bits. An
  explicit `RJ_CONSAN_MOI_OWNER_SGPR` remains a debug override.

The `workitem_id` estimate is adequate when captured at entry for current 1D
two-wave controls. It is not a complete owner derivation for arbitrary 2D/3D
local invocation layouts, and `v0` cannot be treated as workitem identity after
entry because it is ordinary guest state by then.

The `hw_id` source is useful for targeted experiments because it is
wave-uniform and does not depend on local invocation dimensionality. It is not
the ordinary Inline Shadow operating point: deriving it in every hot probe is
materially more expensive than entry-initialized persistent state. Its
temporary is chosen above all guest scalar references and descriptor-backed;
full-SGPR kernels fail visibly rather than borrowing an unproven register.

Current boundary and direction:

- Keep 3D workgroup identity.
- Use a robust owner derivation that does not require user-selected registers.
- Preserve `hw_id` as a useful low-level source where appropriate.
- Owner/epoch state is integrated with the common scratch/spill policy. Broaden
  the identity encoding only when a concrete program exceeds its packed
  bounds.

## Barrier and atomic semantics

Barriers:

- `record_replay` appends barrier-arrival records and host replay coalesces
  contiguous same-workgroup arrivals into logical epoch advances.
- `inline_shadow` can trampoline supported barriers, execute the original
  barrier, and increment an epoch VGPR after the barrier. Exact-shadow packing
  masks that monotonically incremented value to 10 bits, so long-running
  kernels use epochs modulo 1024 without corrupting neighboring metadata
  fields. A conflict separated by exactly 1024 barrier epochs can therefore be
  conservatively reported as unordered.
- `sampled` publishes typed barrier synchronization metadata into the same
  dynamically selected bank as the causal access window. Host scanning uses
  it to avoid joining unrelated dispatch/workgroup/epoch identities.

Atomics:

- MOI treats atomics as ordering events for LDS, not as global-memory race
  checks.
- `record_replay` has host-side release/acquire modeling and a narrow DBI
  atomic-record path.
- `inline_shadow` has bounded address-scoped release/acquire metadata.
- `sampled` publishes admitted atomic/fence synchronization evidence into its
  selected causal windows.

Current atomic support is intentionally narrow. Broader opcode coverage should
follow concrete semantic controls rather than being inferred from compatibility
runs.

## Current boundaries

The current implementation deliberately retains several bounded or
target-specific mechanisms:

- optional manual register debug overrides;
- target-specific ordinary-VGPR spill backends rather than general
  register-class spilling;
- conservative versioned ordinary settings with advanced extensions kept
  opt-in;
- `MaybeGroup` flat LDS provenance;
- static Record/Replay snapshots and bounded Sampled window banks;
- finite direct-mapped Inline atomic ordering tables;
- bounded Inline diagnostics and a deliberately narrow admitted atomic
  vocabulary;
- deterministic or scalar-source delay instead of a randomized perturbation
  schedule.

The direction remains a small set of well-defined flavor/engine choices with
automatic resource management, defensible LDS classification, and clear
bounded diagnostics.

## Remaining engineering boundary

The register-resource path is in place: non-spill allocation, gfx1201
spill-backed access/barrier/atomic probes, zero-to-nonzero dispatch scratch,
persistent-state fallbacks, scalar/special-state policy, compatible
shared-function assignments, and bounded outcome summaries.

The reusable backend and its relationship to Kunwar Grover's
`users/Groverkss/text-relocation-land` work are documented in
[AMDGPU register spilling](../spilling.md). ConSan integrates that backend with
Record/Replay, Sampled, and Inline Shadow probes; later work is broader target
coverage rather than another ConSan allocator.

If ConSan needs SGPR spilling, assume it is not already covered there. Implement
only the minimal SGPR support needed for the current probe family, keep it
isolated, and prefer deleting or replacing it when shared rocJITsu spilling
lands.

## Barrier-mutation safety boundary

Barrier mutation is gated by the typed
`consan_barrier_mutation_qualification` table in `consan_options.h.inc`.
Encoder availability alone does not make a mutation semantically safe.

Cross-block whole-barrier movement also has an explicit typed CFG contract.
`CompletingStructuredDiamond` is the non-destructive conditional control: the
destination lies in a two-successor guard, the original pair begins the common
reconverged block, and each distinct acyclic arm has exactly the guard as its
predecessor and the source as its successor. This preserves one barrier-pair
execution for every traversal that reached the original source. Final
validation rebuilds the pristine CFG and rederives the complete contract.

`DestructiveStructuredExecDiamond` is deliberately separate. It places the
pair in one EXEC-narrowed optional arm, requires the destructive opt-in, and is
eligible only under explicit destructive containment. Neither option is
accepted for the other contract, and arbitrary cross-block or cyclic
placements remain rejected.

The gfx1201 completing-ID mutation is unsupported by the current encoding
contract. `-2` is a trap barrier, `-3` and `-4` are cluster barriers, and the
positive named-barrier completion lifecycle is a gfx1250 instruction family.
ConSan must not substitute one of those forms for another.

Participant mutation deliberately has a narrower encoding contract than
lifecycle ID retargeting. The locally authoritative LLVM AMDGPU definitions
model `llvm.amdgcn.s.barrier.init(ptr, i32 memberCnt)`, and instruction
selection packs the named barrier ID into M0 bits 5:0 and the member count into
M0 bits 21:16. ConSan therefore admits a count rewrite only when
`s_barrier_init m0` is immediately preceded by a literal `s_mov_b32 m0`, the
reserved bits are zero, and both the named ID and member count are in their
valid encoded ranges. It changes only bits 21:16 and validates that the setup
instruction, barrier adjacency, ID, and reserved bits remain unchanged.

Dynamic M0 construction cannot establish the stored count without data-flow
analysis. Immediate barrier-init encodings carry the barrier ID rather than a
member count, and the verified encoding contains no participant-mask field.
Those cases consequently produce an explicit typed `Unsupported` result.
