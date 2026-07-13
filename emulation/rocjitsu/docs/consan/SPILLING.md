# ConSan Register Allocation and Spilling

This document describes the register-resource path delivered by ConSan R1:
how a probe obtains temporary registers, when it spills, how private scratch is
laid out, which kernel descriptors must change, and where the implementation
deliberately fails closed. It is the focused companion to [DESIGN.md](DESIGN.md);
[PLAN.md](PLAN.md) tracks later feature, operational, and qualification work.

The first complete version now covers the committed R1A-R1H implementation on
gfx1201 and the target-dispatched CDNA4 spill path on gfx950.
Access, barrier, and atomic VGPR paths use the same policy, and the HSA log
reports bounded per-source outcomes plus planned and emitted spill bytes.
Later DAG nodes may extend the implementation, but they do not reopen the R1
resource contract described here.

## Credit and provenance

ConSan's spilling work started from Kunwar Grover's
`origin/users/Groverkss/text-relocation-land` branch. That branch established
the implementation direction we followed: request semantic scratch registers,
use liveness when possible, give borrowed registers stable per-lane spill
slots, and bracket instrumentation with save and restore code. In particular,
ConSan's `SpillManager` is based on the slot-allocation and spill-frame ideas
from Kunwar's branch.

Kunwar's concrete save/restore emitter targets CDNA3, so it could not be reused
as machine code on ConSan's first live target. The following pieces were added
for the ConSan/gfx1201 integration, followed by its gfx950 port:

- the RDNA4 address-free `scratch_store_b32` / `scratch_load_b32` emitter and
  its wait-counter sequences;
- the CDNA4 eight-byte address-free `scratch_store_dword` /
  `scratch_load_dword` emitter and its `VM_CNT` waits;
- kernel- and shared-owner CFG/liveness planning in the DBI patch path;
- descriptor, AMDGPU metadata, and dispatch-private-segment coordination;
- zero-private-segment dispatch support;
- common-layout handling for helpers reached by more than one kernel;
- transactional patch integration and the focused synthetic/HIP test matrix.

The branch was a design and implementation starting point, not a claim that
its CDNA3 backend or broader binary-translation changes were imported
wholesale. The related `origin/users/Groverkss/dbt-tooling` and
`origin/users/Groverkss/dbt_interposer` branches remain useful background, but
they are not prerequisites for the current ConSan path.

## Why instrumentation needs this

Native DBI probes run inside an already allocated GPU kernel. They need VGPRs
for addresses, record slots, shadow values, and temporary arithmetic, and
some probes also need SGPRs to preserve EXEC, VCC, and SCC. Choosing a
convenient register number globally is unsafe: the guest kernel may use it,
and a helper function may be reached by kernels with different register and
private-segment configurations.

R1 therefore treats a register window as a planned resource. A site is patched
only when the planner can prove one assignment correct for every kernel that
can execute those instruction bytes.

## Allocation policy

For each probe request, the allocator tries these outcomes in order:

| Outcome | Meaning |
| --- | --- |
| Explicit override | Use the requested debug register window, but only if it does not overlap the instruction's operands, persistent state, or a live guest value. |
| Liveness-dead | Reuse a window inside the current descriptor allocation that is dead before the instrumented instruction. |
| Descriptor growth | Allocate a fresh window above every guest reference and grow each owning descriptor to cover it. |
| Spill required | Borrow an allowed live window, save it to per-lane private scratch, run the probe, and restore it before guest execution resumes. |
| Unsupported | Do not patch; retain a typed reason such as missing ownership, invalid descriptor, forbidden overlap, full scalar file, dynamic stack, or unencodable private layout. |

Explicit environment variables remain useful for encoding tests and debugging,
but they do not bypass liveness or overlap checks.

The current ordinary-VGPR architectural limit is 256. On CDNA4, however, a
nonzero descriptor `ACCUM_OFFSET` also marks the AccVGPR alias boundary; an
ordinary scratch or persistent window may not cross into that range. SGPR
preservation uses fresh descriptor-backed scalar windows; there is
intentionally no general SGPR spill stack. gfx1201 exposes 106 normal SGPRs,
whereas gfx950 exposes `s0-s101` before FLAT_SCRATCH and other architectural
state. A probe that cannot obtain a complete target-valid scalar window is
skipped with a bounded diagnostic.

## Site and owner model

The planner decodes symbol-backed code ranges into basic blocks, constructs a
kernel CFG scope from each kernel entry, and runs liveness over that scope.
Call edges associate a helper block with every kernel that can reach it.

For a direct-kernel site, there is one owner. For shared text, the plan uses:

- the union of all owners' live-before sets, so a dead window is dead for
  every caller;
- the smallest current VGPR allocation, so an in-allocation window exists in
  every caller;
- the largest referenced VGPR/SGPR index, so fresh growth is above every
  caller's guest state;
- the maximum original private-segment size as the base for a common spill
  layout;
- one register assignment and one instruction sequence for the shared bytes.

Every descriptor that can reach the patched helper receives the required
register/private extent. Unrelated descriptors remain unchanged. Unreachable
or unresolved indirect helper text is a missing-owner result; even an explicit
register override does not guess that every descriptor owns it.

Persistent inline-shadow state follows the same ownership rule. A shared
helper either uses one owner/epoch VGPR representation for every owner or one
common private-epoch layout. Workitem-ID-derived private ownership additionally
requires all owners to agree on wave size, because wave32 and wave64 use
different owner shifts.

## Per-lane private layout

`SpillManager` appends a DBI-owned zone to the kernel's existing per-lane
private segment:

```text
0                                      original_private_bytes
+-----------------------------------------------------------+
| guest/compiler private segment                            |
+-----------------------------------------------------------+
                         align up to 16 bytes
                         |
                         v
                         +--------+--------+--------+
DBI spill zone           | slot 0 | slot 1 |  ...   |
                         +--------+--------+--------+
                           4 B      4 B
```

Each ordinary VGPR receives one 4-byte slot per lane. Consecutive VGPR windows
receive consecutive slots. Allocation is stable and idempotent for the same
register, and a failed multi-slot reservation leaves the manager unchanged.

For shared text, the DBI zone begins at
`align_up(max(owner.original_private_bytes), 16)`. The immediate offsets in the
shared save/fill instructions are therefore valid for every owner, and all
owners grow to the same sufficient extent.

When inline shadow falls back to a private epoch, the persistent epoch dword
is placed first. Ephemeral access/prologue/barrier spill leases begin in a
separately aligned zone above it, preventing persistent and temporary state
from overlapping.

On gfx950, the address-free FLAT_SCRATCH encoding has a signed 13-bit byte
offset. ConSan uses only non-negative, dword-aligned offsets, so the last
encodable slot starts at 4092 and the complete per-lane private extent is
limited to 4096 bytes. Planning rejects a larger guest-plus-DBI layout before
emission. This tighter CDNA4 limit is part of the private scratch transaction,
not merely an instruction-builder restriction.

## Target-specific save and restore sequences

The backends support ordinary B32 VGPR windows on RDNA4/gfx1201 and
CDNA4/gfx950. For each register they emit an address-free per-lane scratch
store on save and the matching scratch load on restore. gfx1201 uses its
one-word VSCRATCH forms and separate zero-threshold store/load waits. gfx950
uses two-word FLAT_SCRATCH forms and `s_waitcnt vmcnt(0)` after both the store
batch and load batch.

Conceptually:

```text
scratch_store_b32 vN, private_offset_N
...
wait until scratch stores complete

instrumentation may clobber vN...

scratch_load_b32 vN, private_offset_N
...
wait until scratch loads complete
resume guest code
```

The waits are conservative. They ensure the borrowed value is saved before
the probe clobbers it and restored before guest code consumes it. The sequence
runs under the current EXEC mask and does not itself change EXEC. Probes that
narrow EXEC must restore the guest mask before the borrowed window is finally
returned.

These scratch waits are distinct from the other CDNA4 memory contracts. Native
DS completion drains `LGKM_CNT`; general FLAT operations drain both `VM_CNT`
and `LGKM_CNT`. None of the gfx12 `s_wait_storecnt`, `s_wait_loadcnt`, or
`s_wait_dscnt` encodings is used as a gfx950 substitute.

## Descriptor, metadata, and dispatch transaction

A spill is not correct merely because the load/store instructions encode. The
runtime must allocate enough private backing for every affected dispatch.
ConSan coordinates four pieces:

1. Plan register and private requirements without mutating the code object.
2. Grow only owning kernel descriptors, enabling the private segment when a
   zero-private kernel first needs spill space.
3. Keep the named kernel's AMDGPU MessagePack private size coherent when an
   in-place update is representable.
4. Have the HSA hook associate the requirement with the loaded kernel object
   and rewrite the AQL dispatch packet's private-segment size when needed.

Descriptor changes and text replacement are prepared through
`CodeObjectPatcher`. When `.text` growth shifts later ELF sections, subsequent
stages resolve the active descriptor by kernel name rather than writing
through a stale pre-growth file offset. A failed plan or unencodable update
does not intentionally leave a partially instrumented object for loading.

Two CDNA4 descriptor details participate in the same transaction. First,
ordinary VGPR allocation and persistent identity placement must remain below a
nonzero `COMPUTE_PGM_RSRC3.ACCUM_OFFSET`; ConSan does not spill AccVGPRs.
Second, kernarg preloading can expose two hardware entries exactly 256 bytes
apart. Owner/epoch and private-state redirection emit a corresponding pair of
entry stubs and return each to its matching original path. A single redirected
stub is not a valid substitute for both firmware entry paths.

## Current support boundary

The committed R1 boundary is:

| Resource or probe family | Current state |
| --- | --- |
| Static record/replay access probes | Dead, fresh-growth, and spill-backed VGPR windows. |
| Sampled access probes | Dead, fresh-growth, and spill-backed VGPR windows. |
| Inline-shadow access probes | Dead, fresh-growth, and spill-backed VGPR windows, including private-epoch fallback. |
| Reachable shared access helpers | One all-owner-compatible dead, fresh, or common spill plan. |
| Private-epoch entry/barrier temporaries | Saved and restored through the target-dispatched gfx1201/gfx950 private path. |
| Dynamic record, barrier, diagnostic, and atomic scalar state | Automatic descriptor-backed SGPR windows with EXEC/VCC/SCC preservation; gfx950 preserves full wave64 VCC while RDNA4 SuperCollider compare paths need `VCC_LO`. |
| Barrier and atomic VGPR temporaries | Dead, fresh-growth, and spill-backed common plans; explicit register variables are debug overrides. |
| SGPR spilling | Not implemented; full-file pressure fails closed. |
| Dynamic-stack kernels | Spill growth rejected until the stack convention is proven compatible. |
| AccVGPR spilling | Not implemented. |
| Unresolved indirect ownership | Not instrumented. |
| Native targets | gfx1201 and gfx950 B32 spill emission; other targets remain unsupported. |

This is deliberately narrower than a general compiler register allocator. R1
provides the minimum semantically safe DBI resource path needed by current MOI
probes while keeping the target-specific backend replaceable by future shared
rocJITsu infrastructure.

## Failure boundaries

The resource path refuses instrumentation when any required invariant is not
proven. Important examples include:

- no owning kernel scope or unresolved indirect ownership;
- no one assignment valid across all owners;
- overlap with an instruction operand or persistent owner/epoch state;
- invalid or unencodable descriptor growth;
- private-segment size or scratch-immediate overflow;
- a compiler dynamic-stack marker on an owning kernel;
- incompatible shared private-owner wave sizes;
- branch/cave placement failure;
- malformed or non-growable in-place AMDGPU metadata;
- unsupported architecture or register class.

These outcomes are distinguishable from "no candidate", "successfully
instrumented and no race found", and an actual race diagnostic. The HSA log
summarizes explicit, dead, descriptor-growth, spill, and unsupported plans,
planned/emitted spill bytes, site kind, typed reason, and a bounded owner-name
list. Patch, visible-record, and diagnostic guards cover the corresponding
runtime distinctions.

## Tests and evidence

The focused CPU/synthetic suite is:

```sh
./emulation/rocjitsu/build/tests/rocjitsu_tests \
  --gtest_filter='ConSanResourcePlan.*:ConSanMoi.*:SpillManager.*:InstructionBuilder.*'
```

The current focused ConSan result is 241/241. It covers allocation precedence,
forbidden ranges, rollback, gfx1201 and gfx950 encodings and waits,
descriptor/metadata growth, zero-private kernels, dynamic-stack rejection,
shared-owner dead/fresh/spill layouts, mixed-wave rejection, persistent state,
barrier/atomic rollout, bounded summaries, and staged text-growth descriptor
updates.

The live target-aware tier includes forced-spill tests whose victim VGPRs
remain live across instrumentation:

```sh
ctest -j8 --output-on-failure \
  -R '^(ConSanSpillHipTest|ConSanInlineShadowTest|ConSanMoiHipTest)\.'
```

The established gfx1201 resource/behavior result is 29/29. gfx950 focused spill
and engine tiers also pass; its broad IREE qualification remains in progress.
Notable controls include record/replay and sampled forced-spill preservation,
zero-to-nonzero private dispatch backing, private-epoch barriers, inline
diagnostics, and atomic handoff behavior, all without register-number
configuration. The established gfx1201 independent hip-moi control suite is
189/189, and all three engines complete its 209-test IREE compatibility sweep.
See
[LOCAL_TESTING.md](LOCAL_TESTING.md) for workspace paths and the broader test
ladder.

## Source map

- `lib/rocjitsu/src/rocjitsu/code/patch/consan_resource.*`: register request,
  allocation-source, and typed failure policy.
- `lib/rocjitsu/src/rocjitsu/code/patch/spill_manager.*`: stable private slots,
  gfx1201/gfx950 save/fill sequences, descriptor growth, and metadata growth.
- `lib/rocjitsu/src/rocjitsu/code/patch/consan_moi.cpp`: owner-scoped planning
  and probe integration.
- `lib/rocjitsu/src/rocjitsu/code/patch/code_object_patcher.*`: transactional
  descriptor/text changes and relocation-aware emission.
- `lib/rocjitsu/src/rocjitsu/hooks/rj_hsa_dbi_hooks.cpp`: load interception,
  resource logging, kernel-object association, and dispatch private-size
  rewriting.
- `tests/patch/consan_test.cpp`: synthetic plan, encoding, patch-shape, and
  descriptor-isolation coverage.
- `tests/dbi/hip_consan_moi_test.hip`: live semantic and spill-preservation
  tests.

For the overall sanitizer architecture, continue with [DESIGN.md](DESIGN.md).
For user-facing controls, see [USAGE.md](USAGE.md) and
[TUTORIAL.md](TUTORIAL.md). For the dependency DAG and R1H acceptance criteria,
see [PLAN.md](PLAN.md).
