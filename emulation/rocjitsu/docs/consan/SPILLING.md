# ConSan Register Allocation and Spilling

This document describes how ConSan selects temporary registers, proves that a
choice is valid for every owning kernel, and integrates RocJitsu's spill
backend with code-object loading and dispatch.

The reusable allocator, target-specific save/restore sequences, descriptor
helper, support boundary, and provenance are documented in
[AMDGPU register spilling](../spilling.md). This guide covers only the
ConSan-specific policy layered on that backend.

## Allocation policy

Native DBI probes run inside an already allocated kernel. Choosing a convenient
register number globally is unsafe: the guest may use it, and shared helper
text may be reached by kernels with different register and private-segment
layouts.

For each ConSan probe request, the resource planner tries these outcomes in
order:

| Outcome | Meaning |
|---|---|
| Explicit override | Use a requested debug register window only when it does not overlap instruction operands, persistent state, or live guest values. |
| Liveness-dead | Reuse a window inside the current descriptor allocation that is dead before the instrumented instruction. |
| Descriptor growth | Allocate a fresh window above every guest reference and grow each owning descriptor to cover it. |
| Spill required | Borrow an allowed live window, save it to per-lane private scratch, run the probe, and restore it before guest execution resumes. |
| Unsupported | Do not patch; retain a typed reason such as missing ownership, forbidden overlap, a full scalar file, unsupported dynamic-stack use, or an unencodable private layout. |

Explicit environment variables are encoding and debugging controls. They do
not bypass liveness, ownership, or overlap checks, and ordinary runs do not
require them.

The ordinary-VGPR architectural limit is 256. ConSan does not implement a
general SGPR spill stack. It does have a narrower private-memory recipe
available to selected Record/Replay, Sampled, and InlineShadow probes. The
recipe preserves a fixed transient scalar window: each borrowed uniform SGPR
is copied through one spill-managed VGPR, stored per lane, reloaded, and
restored with `v_readfirstlane_b32`. The planner selects that recipe only when
it can also find owner-compatible indirect-router PC, key, call-return, and SCC
state.
Those router registers are needed outside the saved window and must therefore
be dead or fresh at every routed site. A probe fails explicitly when neither
the bounded recipe nor a safe ordinary window exists.

Hardware dispatch identity normally occupies a persistent SGPR pair. If a
RDNA4-family InlineShadow or Sampled owner has no such pair, emitted probes can
use the report's frozen literal dispatch identity instead. This removes
persistent scalar pressure but does not solve transient or indirect-router
allocation.

SuperCollider's indirect path reserves disjoint scalar state above the owning
kernel's maximum referenced SGPR: VCC preservation, an SCC save, and an aligned
PC pair. It captures SCC before the cave and restores it after return-PC
arithmetic. Final validation checks ownership, allocation, disjointness, and
the exact entry and return encodings.

## Site ownership

ConSan decodes symbol-backed ranges into basic blocks, builds a CFG scope from
each kernel entry, and runs liveness within that scope. Call edges associate a
helper block with every kernel that can reach it.

For a direct-kernel site there is one owner. For shared text, one plan must be
valid for all owners. It uses:

- the union of owner live-before sets, so a dead window is dead for every
  caller;
- the smallest current VGPR allocation, so an in-allocation window exists in
  every caller;
- the largest referenced VGPR and SGPR indices, so fresh growth is above all
  guest state;
- the maximum original private size as the base for a common spill layout; and
- one register assignment and instruction sequence for the shared bytes.

Every descriptor that can reach the helper receives the required register and
private extent. Unrelated descriptors remain unchanged. Unreachable or
unresolved indirect text is a missing-owner result; even an explicit override
does not guess that every descriptor owns it.

Persistent InlineShadow state follows the same rule. Shared text either uses
one owner/epoch VGPR representation for every owner or one common private-state
layout. Work-item-ID-derived private ownership additionally requires all owners
to agree on wave size.

## ConSan private layout

The shared backend appends stable spill slots after the maximum original
per-lane private extent. ConSan may reserve a persistent prefix inside that
DBI-owned region before allocating ephemeral spill slots:

```text
guest private | alignment | persistent epoch/owner state | spill leases
```

InlineShadow prefers two descriptor-backed VGPRs for owner and epoch state,
even when that requires safe descriptor growth. Keeping hot-path state in
registers is cheaper than reloading it at every access. If the pair cannot fit,
the persistent epoch dword precedes the entry-captured owner dword in private
memory. Access, prologue, barrier, and atomic spill leases start above that
prefix so temporary state cannot overlap persistent state.

For a shared spill, every owner uses the same slot offsets and grows to the same
required private extent. Mixed fixed/dynamic ownership and unknown owners fail
closed.

## Dynamic-stack integration

The supported dynamic-stack path uses a target-specific compiler convention in
which `s32` is the stack top and `s33` is the current frame base. A spill-backed
probe uses the shared backend's site-local dynamic frame only after ownership
analysis establishes that this recipe applies. InlineShadow supports the
recipe on every admitted target. Record/Replay and Sampled support it on
CDNA3/CDNA4 and the RDNA4 family when every owner of the spilled site uses a
dynamic stack.

ConSan supplies safe scalar save registers, preserves SCC and the incoming
frame, borrows the VGPR window, and restores all state before resuming guest
code. The engine-specific scalar window reserves a frame-base save slot, and
automatic scalar allocation excludes the backend's named stack-top and
frame-base registers individually, whose implicit roles need not appear as
decoded operands. This path applies to access, atomic, and synchronization
probes that receive a spill-backed resource plan. A full-VGPR RDNA4 Sampled
owner keeps persistent owner/epoch state in a scalar tuple only after
whole-owner CFG/reference analysis proves the tuple untouched; its entry
initializer separately borrows an entry-local dead VGPR pair. If either proof
is unavailable, the dynamic owner fails closed instead of using fixed-offset
private state.
The owner-scope reference summary and scalar-tuple admissibility proof are the
intended shared foundation for future general SGPR spilling and mixed
fixed/dynamic ownership; those follow-ups must preserve the same fail-closed
contract rather than derive a weaker placement rule.

The gfx1250 SuperCollider group-FLAT path also supports full register pressure.
It first saves the complete borrowed VGPR window relative to the incoming stack
top, then uses four already-preserved VGPRs as transient reservoirs for the
live VCC-save SGPR pair, incoming frame base, and SCC. The SGPR pair is exactly
the 64-bit VCC preservation shape required by this probe; it is not a
hard-coded preserved SGPR range. A shared helper may use this recipe when all
owners are dynamic-stack kernels. A mixed fixed/dynamic owner set receives the
typed `mixed_stack_owner_spill_rejections` outcome only when simultaneous
register pressure actually requires a shared spill recipe.

The descriptor records an absolute private-segment minimum. Separately, each
dynamic patch records its site-local frame depth. Immediately before dispatch,
the HSA hook computes:

```text
max(descriptor minimum, launch private bytes + maximum site-local frame depth)
```

The launch value remains authoritative for the runtime-configurable stack
depth; alternative site-local frames use their maximum rather than a sum
because they cannot be active simultaneously. The emitted scratch accesses
remain relative to the runtime frame. Fixed-offset private owner/epoch state
cannot be used by a dynamic-stack owner; the planner instead selects
owner-compatible VGPR state or, on CDNA, a scalar persistent tuple.

## Code-object and dispatch transaction

ConSan coordinates the static spill helpers with its HSA runtime integration:

1. Plan ownership, registers, private layout, and encodings without mutating
   the code object.
2. Grow only owning kernel descriptors and enable private storage when a
   zero-private kernel first needs it.
3. Leave AMDGPU MessagePack notes untouched; ROCR does not use their duplicated
   private-size entries as runtime authority.
4. Commit descriptor and text changes through `CodeObjectPatcher`.
5. Associate the private requirement with the loaded kernel object.
6. Rewrite the AQL dispatch packet's private-segment size before submission.

When text growth moves later ELF contents, ConSan resolves the active
descriptor by kernel name instead of retaining a stale pre-growth file offset.
A failed plan does not intentionally leave a partially instrumented image for
loading.

Separately from spill storage, ConSan currently combines the compiler-emitted
`.sgpr_count` analysis hint with decoded references before placing dispatch
state or transient EXEC/VCC/SCC windows. It never uses that hint to reduce a
decoded bound, and it does not rewrite the note. Replacing all analysis-note
inputs with descriptor- and instruction-derived facts is a distinct follow-up
from removing private-size mutation.

## Current ConSan support

| Probe family | Current resource path |
|---|---|
| SuperCollider group-FLAT probes | Dead and fresh windows on admitted targets; gfx1250 full-pressure dynamic-stack owners can use the borrowed-pair site-local frame. |
| Record/Replay access probes | Dead, fresh-growth, and spill-backed VGPR windows; dynamic-stack spill on CDNA3/CDNA4 and the RDNA4 family. |
| Sampled access probes | Dead, fresh-growth, and spill-backed VGPR windows; dynamic-stack spill on CDNA3/CDNA4 and the RDNA4 family. |
| InlineShadow access probes | Dead, fresh-growth, and spill-backed VGPR windows on every admitted target; fixed-stack owners may use the private-epoch fallback. |
| Reachable shared helpers | One all-owner-compatible dead, fresh, or common spill plan; a dynamic spill requires every owner to be dynamic. |
| Persistent owner/epoch/key state | Owner-compatible VGPR tuples, fixed-stack private state where supported, or proven owner-scope scalar tuples for dynamic full-VGPR CDNA/RDNA4 owners. |
| Private-epoch entry/barrier temporaries | Saved and restored through the target-specific private path. |
| Transient scalar state | Component-local dead/fresh windows, plus bounded private-memory preservation for supported Record/Replay, Sampled, and InlineShadow probes. Indirect-router scalars remain dead/fresh. |
| Barrier and atomic VGPR temporaries | Dead, fresh-growth, and spill-backed common plans, including the engine/target dynamic-stack matrix above. |
| General SGPR or AccVGPR spilling | Not implemented. |
| Dynamic-stack kernels | InlineShadow, Record/Replay, and Sampled on all admitted CDNA3/CDNA4/RDNA3/RDNA4-family targets; gfx1250 SuperCollider group-FLAT full-pressure spill; unsupported engine/target or mixed-owner spill recipes fail closed. |
| Unresolved indirect ownership | Not instrumented. |

This is narrower than a compiler register allocator. It is the semantically
safe resource path needed by the implemented probes while keeping the shared
backend replaceable and independently reusable.

## Failure reporting

ConSan preserves backend and policy failures as typed outcomes. Important
examples include:

- missing ownership or unresolved indirect control flow;
- no assignment valid across all owners;
- overlap with an instruction operand or persistent state;
- invalid descriptor or private-segment growth;
- dynamic-stack use outside the supported recipe;
- incompatible owner wave sizes;
- branch, cave, or relocated-prefix placement failure;
- unsupported target or register class; and
- decoded access forms without an instrumentation lowering, reported as
  `unsupported_mnemonic`.

The HSA log distinguishes explicit, dead, descriptor-growth, spill, and
unsupported plans. It reports planned and emitted spill bytes, site kind,
typed reason, and a bounded owner list. A resource rejection is therefore
distinguishable from a successfully instrumented run that found no race.

## ConSan source map

- `lib/rocjitsu/src/rocjitsu/code/patch/consan/consan_resource.*`: request,
  allocation-source, and typed-failure policy.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan/consan_moi.cpp` and its feature
  fragments: owner-scoped planning and probe integration.
- `lib/rocjitsu/src/rocjitsu/hooks/consan/`: load interception, resource
  reporting, kernel-object association, and dispatch private-size rewriting.

For the backend API and its provenance, return to
[AMDGPU register spilling](../spilling.md). For the overall sanitizer, continue
with [DESIGN.md](DESIGN.md); for user-facing controls, see [USAGE.md](USAGE.md).
