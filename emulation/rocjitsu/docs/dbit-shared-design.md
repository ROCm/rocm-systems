# DBI/DBT Shared Design

## Purpose

RocJITsu has two code-transformation arms:

- dynamic binary translation (DBT) moves a guest kernel to a target ISA and may
  change every instruction, kernel resource declaration, and code address;
- dynamic binary instrumentation (DBI) preserves the kernel ISA and most of its
  layout while redirecting selected instructions through injected code.

They operate on the same AMDGPU objects and need many of the same mechanisms,
but they do not have the same transformation contract. This document records
the sharing boundary in the current tree, explains intentional non-sharing,
and gives guidance for extending DBI without copying mature DBT machinery or
coupling the two orchestrators.

This is a description of the branch as it exists, not a claim that everything
under `code/patch/` is a finished common API. In particular, that directory
currently contains shared services, DBI-owned services, and DBT-owned services.

## The common transformation stack

The useful shared design is a stack, not a common DBI/DBT base class:

```text
                 DBI Instrumentor             DBT BinaryTranslator
                  site-oriented                kernel-scope-oriented
                         |                              |
          +--------------+------------------------------+--------------+
          |              common transformation mechanisms              |
          | CFG and indirect recovery | liveness and register model     |
          | instruction construction  | text layout/relocation helpers  |
          | ELF and descriptor mutation                               |
          +-------------------------------------------------------------+
                         |                              |
                       Decoder / Instruction / AmdGpuCodeObject
```

The two top-level pipelines should remain separate. Sharing is strongest below
the policy layer, where an operation can be expressed without knowing whether
its producer was an instrumentation point or a translation rule.

Both arms follow the same safety pattern:

1. read and decode an immutable source object;
2. derive analysis facts and a transformation plan;
3. construct bytes and resolve layout before committing them;
4. mutate a private ELF image through `CodeObjectPatcher`;
5. return a new byte vector, leaving the input object unchanged.

DBI explicitly preflights all sites before replacing `.text`. DBT builds each
kernel in a local buffer, records relocation work, and returns the original
image on fatal translation errors. This transactional shape is a shared design
principle even though the intermediate plans are intentionally different.

## What is shared today

### Code-object and instruction model

`AmdGpuCodeObject`, `Decoder`, `Instruction`, and `BasicBlock` are the common
front end. Both arms decode the source ISA rather than treating `.text` as an
untyped word array.

`BasicBlock::build` is more than a block splitter. It incorporates direct CFG
edges and the facts produced by `analysis/indirect_branch_discovery`. DBT uses
the full graph to form relocatable kernel scopes, validate calls and returns,
and repair indirect transfers. DBI currently uses decoded blocks to find an
exact anchor and passes all decoded blocks to one liveness analysis. The
implementation marks this as temporary: kernel terminators prevent false
fallthrough today, but DBI still needs to form the containing kernel scope per
anchor as DBT does. Future block-entry, block-exit, and predicate-selected
instrumentation should extend the common CFG use rather than introduce a
DBI-specific graph.

The graph is deliberately context-neutral. DBT supplies kernel-specific call
and return relationships to liveness as `ScopedCfgEdge`s instead of installing
them globally in `BasicBlock`. This matters to DBI as soon as probes or
instrumentation selection need call-sensitive liveness: the shared graph may be
reused, while each orchestrator remains responsible for defining its scope and
extra edges.

### Register representation, def/use, and liveness

The register-analysis chain is shared end to end:

```text
Instruction
    -> InstDefUse
    -> RegisterRef / RegisterSet
    -> LivenessAnalysis over KernelBlockScope
```

`RegisterRef` and `RegisterSet` are ISA-independent representations of ordinary
SGPR, VGPR, and AccVGPR lanes. `InstDefUse` adds explicit operands and modeled
implicit operands. `LivenessAnalysis` performs kernel-scoped backward dataflow
and provides live-before queries and dead-register searches.

DBT uses liveness at semantic expansion sites to allocate temporary registers,
and configures it with destination encoding limits, optional gfx1250 VGPR-bank
state, selected instruction snapshots, and scoped call/return edges. DBI uses
the same live-before facts at an anchor to select a probe-call target pair and
to compute

```text
instrumentation clobbers = probe clobbers | trampoline-builder clobbers
spill set                = live before anchor & instrumentation clobbers
```

The current limits are shared limits, not DBI or DBT exceptions: special state
such as EXEC, VCC, SCC, M0, FLAT_SCRATCH, and TTMP is not part of
`RegisterSet` liveness. DBI therefore performs a separate, currently
best-effort probe scan for explicit special-register operands and display-name
fallbacks. It rejects detected EXEC, VCC, M0, and FLAT_SCRATCH use and preserves
SCC in the call envelope; implicit special-state effects that the decoder and
def/use layer do not expose remain a gap. DBT rules that manipulate special
state use rule-specific policy. Neither arm should infer safety for special
state from an empty ordinary-register live set.

Private-segment allocation is shared at a narrower boundary than
`SpillManager`. `PrivateSegmentCursor`, currently declared beside
`SpillManager`, supplies common aligned range arithmetic. DBI's `SpillManager`
layers stable register-to-slot identity and a DBI spill zone on that cursor.
DBT's `SemanticSpillFrame` layers short-lived semantic spill ranges on the same
cursor when a lowering cannot obtain dead registers. The register mapping,
frame lifetime, emitted spill sequence, and resource-accounting policy remain
producer-specific. `SpillManager` is implemented and directly tested, but the
current `Instrumentor` does not instantiate it: every non-empty spill set still
fails closed.

### ISA-parameterized instruction construction

`code/patch/instruction_builder.*` is the common handwritten instruction
construction layer. It centralizes:

- selection of generation-specific AMDGPU opcodes;
- construction of common scalar control and arithmetic instructions;
- scalar operand encodings; and
- PC-relative calculations and canonical PC-building sequences.

The clearest shared invariant is
`compute_sopp_branch_simm16(branch_pc, target)`: both DBI trampolines and DBT
relocation use the AMDGPU rule `(target - (branch_pc + 4)) / 4` and the same
range checks. Builders must be used instead of copying an opcode merely because
the SOPP bit layout is stable; for example, `s_branch` differs between GFX9 and
GFX12 families.

The builder is used by DBI's `TrampolineBuilder`, DBT's kernel descriptor
prologues, semantic lowerings, wait-count and hazard handling, and common text
layout. It is appropriately transformation-neutral. New generally useful
instruction constructors should land here (or in the generated ISA builders it
wraps), while constructors meaningful to only one lowering can remain local.

There is some incomplete consolidation. Several DBT semantic files still have
local `build_s_mov_b32_lit` or `build_s_mov_b64` helpers, and
`TrampolineBuilder` duplicates free-SGPR searches available in
`LivenessAnalysis`. These are candidates for small neutral APIs, but only after
their constraints are made explicit; superficially similar helpers sometimes
differ in architecture, allocation ceiling, alignment, or unavailable-set
semantics.

### Text placement and control-flow repair

`code/patch/kernel_text_layout.*` is shared in intent and mostly DBT-driven in
current use. Its neutral types and operations cover:

- appending words and architecture-correct padding;
- placement and rebasing of a kernel-local body;
- descriptor-visible entry stubs and preload-entry residue constraints;
- source-to-target block placement;
- direct branch windows, long branches, and branch islands; and
- recovered indirect transfer and PC-builder fixups.

DBT needs all of this because translated instructions change size and every
kernel scope is relocated as a unit. DBI currently uses only `append_words`; it
keeps original bytes in place and appends probe bodies and trampolines to a
local cave. Consequently, its only control-flow repair is the branch from an
anchor to a trampoline and the branch back.

DBI should reuse more of this layer when its requirements cross the current
threshold. Examples include branch islands for large kernels, descriptor entry
prologues, or relocating a whole instrumented kernel because in-place anchors
are no longer sufficient. It should not adopt `KernelTextLayout` preemptively:
a source-to-target block map is unnecessary overhead for an in-place patch, and
DBT-specific fixup records do not describe a probe call.

The neutral plan types in this file demonstrate the right direction.
`KernelEntryLayoutPlan` intentionally carries placement facts rather than a
`KdTranslation`; `append_relocated_kernel_text` returns placement results while
leaving diagnostics and descriptor policy to its caller.

### ELF mutation

`CodeObjectPatcher` is the shared commit layer. Both arms use it to copy an
input image, expose `.text`, replace that section while maintaining ELF
structure, and emit the result. `replace_text` handles section and segment size
changes, shifts later contents, updates affected symbols and relocations, and
preserves load alignment.

The caller, not the core of `replace_text`, decides what the new text means. DBI
supplies the old text with patched anchors plus appended probe/trampoline bytes.
DBT supplies
newly placed kernel bodies and exact `TextOffsetRelocation` and
`PcRelativeDataRelocation` maps. The patcher interprets ELF coordinates, not
instructions or instrumentation points. Other `CodeObjectPatcher` operations
are intentionally AMDHSA ABI-aware, including descriptor entry redirection.

This boundary is not completely clean today. `CodeObjectPatcher` includes
operations taking DBT's `KdTranslation`, and its implementation includes
`dbt/kernel_descriptor_translator.h`. The lower-level operations
`patch_kernel_descriptor` and `redirect_kernel_entry` are neutral; applying a
DBT translation plan and materializing DBT sidecars are convenience operations
with an upward dependency. Future shared work should move policy-to-bytes
adaptation out of the patcher or replace `KdTranslation` parameters with a
small descriptor mutation plan owned by the patch layer. DBI should not depend
on `KdTranslation` merely to update SGPR, scratch, or entry fields.

### Utilities currently housed in `code/patch`

The directory name describes code mutation broadly; it does not establish that
every file is used by both arms. Current ownership is:

| Area | Current owner/use | Sharing status |
| --- | --- | --- |
| `code_object_patcher` | DBI and DBT | Shared core with DBT-specific convenience APIs |
| `instruction_builder` | DBI and DBT | Shared |
| `kernel_text_layout` | DBT; small utility use by DBI | Neutral mechanism, mostly DBT-consumed |
| `spill_manager` | DBI-owned/tested register slots, not yet wired into `Instrumentor`; DBT uses `PrivateSegmentCursor` | Shared range primitive plus DBI policy |
| `instrumentor`, `trampoline_builder` | DBI | Deliberately DBI-specific |
| `probe_callable`, `probe_symbol`, `probe_clobber` | DBI | Deliberately DBI-specific |
| `kernarg_extension`, `sidecar_metadata` | DBT virtual LDS/runtime | Neutral serialization forms, currently DBT-specific feature policy |
| `error_report` | patch helpers | Small shared patch-layer utility |

`kernarg_extension` may become useful to DBI when a probe needs hidden dispatch
arguments, but DBI should reuse the layout and serialization protocol only
after defining runtime ownership of those arguments. A shared file location is
not by itself a supported DBI contract.

## What is deliberately not shared

### Orchestration and units of transformation

DBI's unit is an instrumentation site. It validates an anchor, snapshots one
4- or 8-byte instruction, plans a trampoline, and patches the anchor in place.
Multiple sites share copied probe bodies but otherwise remain independently
planned until the all-or-nothing commit.

DBT's unit is a kernel scope rooted at a descriptor entry. It may duplicate
shared helpers across scopes, translate every instruction, change instruction
sizes, create normal and sidecar variants, and rebuild all control-flow
addresses in that scope. The scope is deliberately the relocation unit because
a flat `.text` delta cannot describe instruction expansion or shared helpers.

A common `BinaryTransformer` base class would erase these important
differences. Reuse should occur in analyzers, plans with narrow semantics,
emitters, and commit services. The orchestrators should compose those services
directly.

### Byte generation

`TrampolineBuilder` is DBI-specific even though it uses common instruction and
branch builders. Its contract assumes an original anchor that will be replaced
by `s_branch`, an optional relocated original instruction, ordered before/after
items, and a branch back to the original stream. Probe-call planning also knows
the probe calling convention, link pair, SCC preservation envelope, and
probe-body clobbers.

The copied probe itself is deliberately constrained rather than treated as a
small DBT relocation unit. `ProbeCallable` requires a self-contained body with
no ELF relocations, rejects calls and private/scratch accesses, and requires a
return through the calling-convention link pair (`s_setpc_b64 s[30:31]` for the
implemented convention). These restrictions explain why probe discovery and
clobber summaries are DBI policy, and why DBT's general relocation machinery
cannot simply be assumed for a copied probe.

DBT's encoding translator and semantic translator solve a different problem.
They legalize a guest instruction for a host ISA and emit it at a new cursor;
they do not return to the source stream. Generated legalization and encoding
tables must remain DBT-owned and generated from `lib/python/amdisa/codegen`.
DBI should use the common builders for injected target-ISA instructions, not
route instrumentation through DBT legalization unless it actually accepts a
guest-ISA probe requiring translation.

DBT's wait-count translation and hazard tracking are also not general-purpose
post-emission schedulers. Wait-count conversion is guest-to-host semantic
policy, while the current hazard tracker is narrowly integrated with GFX12
semantic expansion. DBI-injected sequences must obey target ordering and hazard
rules, but should reuse these components only after a neutral scheduling
contract is extracted for a demonstrated need.

### Resource policy

Both arms consume liveness, `RegisterSet`, and spill layout, but allocation
policy is intentionally separate.

DBI must preserve all state observable after the probe and reconcile three
parties: the interrupted kernel, the trampoline envelope, and the probe calling
convention. Today it fails if the fixed link pair is live, if required SGPRs are
not available, if detected special state cannot be preserved, or if the
ordinary spill set is non-empty. Descriptor checking is incomplete: DBI proves
that the fixed link pair `s[30:31]` is allocated, but selects the target pair and
SCC temporary low-first under a conservative cross-family bound rather than the
kernel's actual SGPR count. This is presently safe only because requiring
`s[30:31]` implies the lower selected SGPRs are allocated; the allocation API
still needs an explicit descriptor bound.

DBT replaces an instruction and owns the destination kernel ABI. It can grow
descriptor register counts, reserve persistent scratch, select dead temporaries
under destination encoding limits, or emit rule-specific spills. It also has
target-specific concepts such as AccVGPR bases, virtual-LDS pointer state, and
long-branch scratch SGPRs.

The reusable output of analysis is “these registers are live” or “this slot has
this offset,” not “this transformation may grow the descriptor.” The latter is
orchestrator policy.

### Descriptor, metadata, and runtime policy

DBT must change the ELF machine identity and translate descriptor ABI fields to
the host. It owns target wave-size validation, entry prologues, register/private
resource feedback, skipped-kernel behavior, virtual-LDS sidecars, kernarg
extension metadata, and dispatch selection between variants.

DBI normally preserves the machine identity and descriptor entry. Descriptor
changes are needed only when instrumentation adds resources, scratch, or hidden
arguments. Those changes must preserve the source ABI rather than perform a
guest-to-host conversion. DBI should reuse neutral descriptor field mutation
and metadata serialization, while keeping its resource negotiation and failure
policy separate from `KernelDescriptorTranslator`.

### Runtime integration and transform order

Runtime integration is currently asymmetric. DBT has a load-time HSA hook,
guest KFD discovery/forwarding, and sidecar registries. DBI is directly usable
as a code-object transformation and in simulated/hardware tests, while its
general real-ROCR tools-layer integration remains future work. Those runtime
surfaces should not be treated as shared merely because both transformations
occur before device code is loaded.

No combined DBI-plus-DBT pipeline is implemented. If one is added, ordering must
be explicit: translating first and instrumenting the host-ISA artifact is the
natural default because DBT can change the ISA, instruction offsets, CFG, and
descriptor resources. Instrumenting a guest artifact first would require DBT to
recognize and translate injected probes and trampolines. Source-level DBI
requests applied after DBT would also need a retained source-to-target mapping;
raw source offsets cannot be reused as host offsets.

### Diagnostics and partial-success contracts

DBT has typed `TranslationDiagnostic`s, supports diagnostic continuation, and
can optionally replace failed kernels with non-dispatchable stubs while
translating independent kernels. DBI currently returns string errors/warnings
and is all-or-nothing across queued sites.

These user-visible contracts should not be unified accidentally. A common
low-level failure category is useful where a shared mechanism must distinguish
invalid layout from a resource limit, as `TextRelocationResult` does. The
orchestrator must translate that failure into its own diagnostic and atomicity
policy.

## Reuse guidance for DBI development

Before adding DBI-local machinery, classify the need by the fact it produces:

1. If it decodes instructions, builds CFG structure, discovers indirect
   targets, computes def/use or liveness, use `code/` and `analysis/`.
2. If it represents ordinary registers or private-memory slots, extend
   `RegisterSet` or `SpillManager` without embedding probe or translation
   policy.
3. If it constructs an architecture-dependent instruction or computes a
   generally applicable PC-relative value, use or extend
   `instruction_builder`.
4. If it places a kernel, preserves an entry residue, or repairs relocated
   control flow, look first at `kernel_text_layout` and extract a smaller
   neutral operation if the whole DBT layout is not appropriate.
5. If it mutates ELF structure, symbols, relocations, descriptors, or sections,
   use `CodeObjectPatcher`; pass it byte-level mutation plans rather than DBI
   objects.
6. If it selects instrumentation sites, defines probe ABI, preserves interrupted
   state, or decides whether a failed site aborts the artifact, keep it in DBI.

Conversely, DBT may reuse DBI-incubated mechanisms when their contracts become
neutral. `PrivateSegmentCursor` is already an example. Probe symbol resolution and
probe clobber summaries are not currently neutral because they assume a copied
callable probe and its return convention.

## Recommended convergence work

The following work would improve the boundary without merging the pipelines.
The ordering reflects dependency value rather than an implementation schedule.

### 1. Form a real DBI kernel scope per anchor

Replace DBI's current all-decoded-block liveness input with the blocks reachable
from the descriptor entry containing each anchor, including the appropriate
scoped call/return edges. Grouping sites by kernel can retain analysis reuse.
The current absence of false cross-kernel fallthrough is a narrow safety
property, not fulfillment of the `LivenessAnalysis` kernel-scope contract.

### 2. Make patch-layer dependency direction explicit

Split `CodeObjectPatcher`'s neutral ELF operations from adapters that consume
`KdTranslation`. Define small patch-owned plans for descriptor bytes, entry
redirection, and appended loaded records. DBT should convert its translation
plan to those forms before commit. This lets DBI update descriptor resources
without importing DBT policy and makes `code/patch` a true lower layer.

### 3. Extract set-based dead-register selection

The duplication called out in `TrampolineBuilder` should become a small utility
that finds an aligned free run in a `RegisterSet` under explicit lower/upper
bounds. `LivenessAnalysis::find_free_*` can delegate after obtaining
`live_before(inst)`; DBI can call the same primitive with a composed unavailable
set. Bounds must be supplied by the caller so destination encodability, actual
descriptor allocation, and conservative cross-family limits are not conflated.
For DBI this closes a present low-first allocation assumption, not merely a
code-duplication concern.

### 4. Model special architectural state deliberately

DBI's probe preservation and some DBT semantic rules both need EXEC, VCC, SCC,
M0, and scratch-related state. Do not silently add them to the existing bitsets:
their widths, aliases, implicit effects, and save/restore rules differ from
ordinary registers. First extend `InstDefUse` with a separate special-state
fact set and make the decoder/def-use layer produce conservative explicit and
implicit effects, then decide which facts participate in dataflow. EXEC-masked
definitions require particular care. Both arms can consume the facts while
retaining their own preservation policy.

### 5. Share descriptor resource mutation, not translation policy

Introduce an ABI-versioned descriptor resource view and a checked mutation plan
for SGPR/VGPR counts, private segment size, and entry offset. DBT can populate it
from target translation; DBI can populate it from probe negotiation and
`SpillManager`. Wave conversion, virtual LDS, sidecar selection, and skipped
kernels remain DBT policy.

### 6. Extract range extension when DBI needs it

DBI currently fails when either trampoline branch exceeds SOPP `simm16`.
DBT already implements patch windows, long PC builders, and SGPR-free branch
islands, but the current APIs assume a relocated kernel body and, for some
forms, descriptor-grown scratch SGPRs. A future DBI layout stage should extract
and request a generic transfer plan instead of calling the existing kernel
layout API or copying its algorithms. It must plan reachable island/cave
placement, account for scratch and clobber effects, and preserve the fixed-size
anchor.

### 7. Keep shared failures typed and local

Add machine-readable failure enums to shared mechanisms when callers need to
make different policy decisions. Avoid imposing DBT's diagnostic taxonomy on
DBI or changing DBI's atomicity merely to reuse an emitter. Human-readable
messages belong at the orchestrator boundary or accompany, rather than replace,
typed failure data.

## Design invariants

Changes in either arm should converge on and preserve these invariants. Current
DBI's all-block liveness scope is the explicitly documented exception to the
kernel-scope invariant:

- The input `AmdGpuCodeObject` is immutable; output is a new ELF image.
- No shared emitter mutates ELF state while it is still validating or planning.
- Architecture-dependent opcodes come from generated builders or
  `instruction_builder`, not copied numeric encodings.
- PC-relative math has one implementation per transfer form and is range
  checked before bytes are committed.
- Analyses are scoped to one kernel and do not accidentally follow CFG edges
  into another descriptor entry.
- Ordinary register liveness is not treated as proof about special state.
- ELF mutation consumes final coordinates or explicit relocation maps; it does
  not infer transformation intent.
- A producer that relocates instructions fails closed on relocation places in
  `.text` and on references to `.text` whose addend/symbol form it cannot remap.
- Descriptor-visible resource changes are checked against the emitted code and
  committed atomically with it.
- Generated DBT legalizations, decoders, and translators are changed through
  the Python generator, never edited as a shortcut for DBI.
- A utility is shared because its contract is transformation-neutral, not
  because it happens to reside under `code/patch`.

## Testing the boundary

Shared mechanisms need direct unit tests independent of either orchestrator:

- instruction construction and branch boundary cases across architectures;
- register-set operations, def/use facts, scoped liveness, and allocation
  limits;
- spill layout alignment, stability, and transactional failure;
- kernel placement and relocation with short, long, conditional, indirect, and
  preload-entry cases; and
- ELF replacement with moved symbols, relocation references, descriptors, and
  appended loaded/non-allocated data. Relocation tests should distinguish
  supported RELA/symbol and relative forms from in-`.text` places, REL implicit
  addends, section symbols, and named-symbol nonzero addends that DBT rejects.

DBI tests should then cover anchor validation, probe ABI/clobbers, preservation,
multi-site cave layout, and all-or-nothing commit. DBT tests should cover
legalization, semantic expansion, kernel-scope duplication, descriptor
translation, control-flow repair, sidecars, and dispatchability. At least one
integration test in each arm should exercise every shared mechanism it relies
on; unit coverage alone will not expose a mismatch between resource planning,
final placement, and descriptor mutation.

## Summary

DBI and DBT share a substrate, not a transformation algorithm. The mature DBT
path already provides reusable decoding, CFG recovery, liveness, the register
model and private-segment range allocation, instruction construction,
relocation, and ELF mutation. DBI
should consume those mechanisms as its probes become more capable. It should
not inherit DBT's kernel translation, descriptor conversion, metadata policy,
or diagnostics simply to avoid writing a thin adapter.

The desired dependency direction is:

```text
DBI policy ----+
               +--> neutral analysis / planning / emission / ELF mutation
DBT policy ----+
```

Keeping policy above that seam allows DBI to catch up quickly while preserving
the very different correctness contracts of instrumentation and translation.
