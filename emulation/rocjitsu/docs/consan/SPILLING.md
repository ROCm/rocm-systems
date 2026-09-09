# ConSan register allocation and spilling

ConSan inserts native probes into kernels whose compiler allocation is already
fixed. This document describes how it proves temporary and persistent register
choices, handles code shared by several kernels, and connects private spill
storage to runtime dispatch. The reusable save/restore backend is documented in
[AMDGPU register spilling](../spilling.md).

## Resource-planning contract

A convenient register number is not safe evidence. The guest may use it, an
instruction operand may overlap it, or a shared helper may be reached by
kernels with different allocations and private layouts. Every admitted probe
therefore receives a typed resource plan before native emission.

The planner considers these alternatives:

| Alternative | Contract |
| --- | --- |
| Explicit override | Use the requested debug window only if ordinary ownership, liveness, operand, persistent-state, encoding, and descriptor checks all pass. |
| Dead window | Reuse registers within the current allocation that are dead at the insertion point for every owner. |
| Fresh window | Select registers above all guest references and grow every owning descriptor consistently. |
| Spill-backed window | Borrow an eligible live VGPR window, save per-lane values to an owner-compatible private layout, emit the probe, and restore before guest execution resumes. |
| Rejected | Publish a typed resource failure; do not let placement or emission improvise another semantic plan. |

The ordinary architectural VGPR namespace has 256 registers. ConSan does not
provide a general compiler-style SGPR or AccVGPR spill stack. It does provide
mode-specific bounded preservation of selected scalar windows by moving scalar
values through spill-managed VGPRs and private memory. Indirect-router PC,
key, call-return, EXEC, VCC, and SCC roles that must remain live outside that
window still require a proved dead or fresh assignment.

Resource planning returns the complete choice. Emission consumes it; it does
not repeat liveness analysis, choose a new scratch base, or silently fall back
to an unsafe register.

## Execution ownership

Program inventory associates each physical site with every kernel entry that
can execute it. For a direct kernel body this is normally one owner. For shared
helper text, one byte sequence and one resource assignment must work for all
owners.

The all-owner plan uses:

- the union of live-before sets;
- the smallest existing allocation when considering an in-allocation window;
- the largest decoded and metadata-backed register reference before fresh
  growth;
- compatible wave-size and stack conventions;
- the maximum original private extent as the base of a shared fixed layout;
  and
- one native instruction sequence for the shared bytes.

Every reachable owner receives the resulting descriptor/private requirement;
unrelated kernels do not. An unresolved indirect edge or incomplete owner set
is a missing-ownership rejection. An explicit override cannot reinterpret it
as “owned by every descriptor.”

Persistent MOI state follows the same rule. Shared text must have one
owner/epoch/workgroup representation valid for all owners, whether that is a
VGPR tuple, scalar tuple, or fixed private layout.

## Persistent and transient state

MOI separates state that survives between probes from state borrowed only for
one probe:

- persistent owner, epoch, workgroup, and dispatch identity;
- access/barrier/atomic scratch VGPRs;
- transient scalar publication and indirect-routing state; and
- saved guest state required by a spill transaction.

Mode policy selects the required shapes. Record/Replay and Sampled normally
initialize owner/epoch state at entry and derive the owner from captured
work-item identity. Inline Shadow defaults to resident-wave hardware identity
and does not initialize an owner/epoch VGPR pair unless the selected operating
point needs one. An explicit `workitem_id` owner is rejected by Inline Shadow
because `workitem_id_x` alone is not a unique resident-wave identity for
arbitrary multidimensional workgroups.

When fixed private storage is used, ConSan appends a stable DBI-owned region
after the maximum guest per-lane extent:

```text
guest private | alignment | persistent mode state | transient spill leases
```

Persistent offsets are planned first. Access, prologue, barrier, atomic, and
router leases start after that prefix. Every owner of shared spilled text uses
the same offsets and grows to the same required minimum. A lease cannot overlap
persistent state or another simultaneously live lease.

Inline Shadow prefers descriptor-backed VGPR owner/epoch state when safe. Its
fixed-stack fallback places persistent epoch and entry-captured owner in
private memory. Record/Replay and Sampled can use their own proved persistent
scalar or vector layouts at high pressure. These are mode-owned choices over a
shared layout mechanism, not target-specific copies of the allocator.

## Dynamic-stack kernels

The supported compiler convention exposes stack top in `s32` and current frame
base in `s33`. A dynamic-stack spill cannot use an absolute offset from the
descriptor: the launch-selected frame is runtime state. The spill backend
therefore creates a site-local frame relative to the incoming stack top,
preserves the incoming frame and condition codes, and restores everything
before returning to guest code.

The MOI mode registry owns the dynamic-stack policy. All current ConSan target
profiles provide the normalized backend used by Record/Replay and Sampled;
both require every owner of shared spilled text to use the dynamic convention.
Inline Shadow supplies its mode-level dynamic recipe and does not require that
blanket all-owner rule for every operating point. Actual admission still
depends on target encodability, owner proof, scalar bootstrap state, and the
selected probe's complete resource demand.

SuperCollider plans its redundant-access windows separately but uses the same
underlying dynamic-frame helpers when pressure requires borrowing. Native-LDS
and group-FLAT probes preserve their exact VCC/SCC/return-PC needs, and a mixed
fixed/dynamic shared-owner set is rejected when no single save/restore recipe
is valid.

For every accepted dynamic patch, the static result records a maximum
site-local frame addend. The descriptor records an absolute private minimum.
Immediately before dispatch the HSA hook computes the required packet value
from:

```text
max(descriptor minimum, launch-selected private bytes + maximum site-local addend)
```

Alternative probes use a maximum, not a sum, because their site-local frames
cannot be active simultaneously. The typed per-kernel
`ConSanDispatchRequirements` carries this fact from validated lowering to the
loaded symbol; the hook does not infer it from patch names.

## Scalar state and dispatch identity

Scalar planning starts above all decoded and metadata-backed guest ownership
and respects target-reserved ranges. Special architectural registers and
implicit dynamic-stack roles are excluded even when they are absent from the
explicit instruction operands.

Dispatch identity is planned as part of each mode's scalar ABI. A target
profile describes how identity can be captured; a mode decides whether it
needs identity and which typed fallback is semantically valid. Sampled and
Inline Shadow can use a frozen report identity at operating points where their
mode policy permits it. This can remove persistent pair pressure but does not
solve transient scratch or router allocation.

Automatic hardware identity is queue-aware: descriptor planning enables both
the AMDHSA queue-pointer and absolute dispatch-ID preloads when they are not
already present, then restores every displaced guest preload from an explicit
source map. The persistent 64-bit value is presently a fingerprint, not an
injective encoding of the full pair; see [VALIDATION.md](VALIDATION.md) for the
residual collision and queue-lifetime limitation.

Sampled's mode-owned literal fallback remains available when the hardware
identity pair overlaps guest scalar state. It is not queue-aware, so a clean
run at that operating point does not establish exact multi-launch separation.
This limitation belongs to Sampled's evidence semantics and is not repaired or
hidden by common placement, target emission, final validation, or runtime trust.

SuperCollider's indirect route reserves disjoint return-PC and condition-code
state and validates its entry/return encodings. MOI common planning similarly
retains a single scalar-routing state that access, synchronization, prologue,
and spill plans project without recomputing different ABI snapshots.

## Descriptor and runtime transaction

Register allocation is not complete until code-object and runtime effects
agree:

1. Analyze owners, liveness, stack convention, and descriptor state.
2. Resolve persistent state, scratch windows, private layout, and native
   save/restore encodings without changing the input.
3. Join the requirements of all probes and shared owners.
4. Grow only affected kernel descriptors and enable private storage when
   required.
5. Commit descriptor and text changes through `CodeObjectPatcher`.
6. Independently validate the final image and its resource effects.
7. Bind per-kernel requirements by symbol name after replacement loading.
8. Adjust the AQL private-segment size for dynamic dispatches.

AMDGPU metadata notes remain analysis inputs; the kernel descriptor and typed
dispatch requirement are the runtime authority. When text growth moves ELF
contents, ConSan resolves the active descriptor by stable kernel identity
rather than retaining a stale byte offset.

## Failure model

Resource failures remain distinct from semantic unsupported forms and
placement failures. Common typed causes include:

- incomplete or unresolved execution ownership;
- no register assignment valid for every owner;
- overlap with a guest operand, persistent state, or live value;
- descriptor or private-size overflow;
- an unsupported or mixed dynamic-stack owner shape;
- incompatible owner wave sizes;
- missing scalar bootstrap/router state;
- target encoding failure; and
- a decoded semantic form with no native lowering.

Logs report the chosen allocation source, scratch range, spill/private bytes,
site kind, owner set, and rejection reason. An uninstrumented resource failure
therefore cannot be mistaken for a clean instrumented execution.

## Source map

- `code/patch/consan/consan_resource.*` owns common resource alternatives and
  typed outcomes.
- `consan_moi_probe_planning.cpp`, `consan_moi_placement.cpp`, and
  `consan_moi_mode_planning.*` join common mechanics with mode policy.
- `code/patch/consan/modes/<mode>/` owns mode-local scratch, persistent-state,
  scalar-ABI, and fallback choices.
- `code/patch/consan/targets/` owns target profiles and special native state.
- `code/patch/spill_manager.*` owns reusable spill layouts and save/restore
  construction.
- `hooks/consan/` owns symbol binding and dispatch-packet adjustment.

See [DESIGN.md](DESIGN.md) for the complete component graph and [USAGE.md](USAGE.md)
for the expert resource overrides.
