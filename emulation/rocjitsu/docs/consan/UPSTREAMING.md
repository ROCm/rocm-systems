# Preparing ConSan for upstream review

This is the live plan for dividing `shared/rocjitsu/sanitizers` into thematic,
reviewable changes based on **`origin/develop`**. Update it in place as decisions
and chunk boundaries change. It describes current thinking, not a history of
previous plans or a collection of test results.

The integrated reference for the first decomposition is the local tag
`consan/upstreaming-baseline-20260929`. Its upstream base is `origin/develop`
commit `952951abbf5c267f73497dc39351fd793128e7b5`; the reference includes the
sanitizer branch and integration repairs.

The next step is to construct `shared/rocjitsu/sanitizers2` in a separate
worktree, with a linear history and one commit per prospective PR at each
checkpoint. Creating the stack does not imply submitting its contents for
review. Keep all work local; do not push.

## What makes a useful chunk

A chunk should fulfill a human-understandable purpose, for example:

> Let instrumentation preserve registers in a kernel that originally allocated
> no scratch.

Include the implementation, tests, build integration, and relevant documentation
for that purpose. A chunk must build and fulfill its stated purpose without a
later commit repairing or completing it. It may depend on earlier chunks:
self-contained does not mean independent of the rest of the stack.

Prefer semantic boundaries to directory boundaries. A report layout and its
decoder can belong together even though their source files live in different
components. Conversely, “all gfx1250 changes” combines instruction semantics,
resource preservation, and tensor-DMA observation, which deserve distinct
reviews and DBI compatibility discussions.

For every chunk, maintain:

- A stable identifier, such as `consan-barrier-epochs`, that survives rewritten
  commit hashes.
- Its purpose and the behavior it adds or corrects.
- Its actual prerequisites, rather than every preceding commit in the stack.
- Included and excluded scope.
- Its DBI position: aligned with the design, filling an implementation gap, or
  proposing an architectural extension.
- Verification requirements that demonstrate its purpose.

Keep chunk identifiers in commit messages and maintain the current inventory
and decisions in this document or linked planning documents. Keep execution
logs and test-result files outside Git.

## Initial families of chunks

These are starting categories for discovery, not a final commit list. Each
family can produce several PR-sized commits. The branch contains substantial
work outside ConSan itself, including waitcheck, emulator support, generated
code, and test infrastructure; all of it needs accounting.

| Family | Examples of useful chunks | DBI discussion |
| --- | --- | --- |
| Independent correctness fixes | Reject invalid scratch alignment; repair a particular instruction's decoding or execution. | Usually shared infrastructure with no new architectural contract. |
| Instruction and program analysis | Recover memory operands, register liveness, helper ownership, or indirect control flow. | Which facts should DBI expose to clients? |
| Register and scratch preservation | Enable scratch from zero; support dynamic private frames; preserve target-specific special state. | Framework ownership, resource limits, preservation guarantees. |
| Safe code-object rewriting | Grow text transactionally; repair branches and metadata; retain original-site mappings. | Shared DBI/DBT placement and attribution. |
| ConSan observation planning | Identify LDS accesses and barriers; distinguish unsupported sites from failed lowering. | Client semantics versus generic site selection. |
| ConSan evidence protocol and analysis | Define bounded access records and execution identities; decode and compare retained observations. | Bounded device state versus host streaming. |
| Native ConSan instrumentation | Publish LDS evidence; maintain barrier epochs; preserve the surrounding kernel. | Native emission versus compiled calls; persistent instrumentation state. |
| Runtime ownership and binding | Allocate reports; retain replacement images; track completion and safe recycling. | Common hook pipeline, variants, concurrent dispatches. |
| Synchronization beyond barriers | Observe a supported atomic/fence protocol, including compare-exchange failure. | Dynamic results, placement, and synchronization evidence. |
| Additional access forms and targets | Recover group-FLAT addresses; observe gfx1250 tensor-DMA accesses. | Target semantics and reusable framework facilities. |
| Fault injection and qualification tools | Remove a selected barrier for a detector test; generate workload allowlists. | Explicit transformation versus observation. |
| SuperCollider | Redundant observations and mismatch reporting. | Paired operations and device-side comparison. |
| Waitcheck | Hazard analysis, target support, CLI/API, and runtime integration. | A separate client; avoid making all of waitcheck a ConSan prerequisite. |
| Other supporting work | Emulator fixes, test infrastructure, and changes outside rocjitsu. | Establish actual dependencies and identify independently useful work. |

Start by extracting three representative chunks: a small independent
correctness fix, a meaningful shared resource or rewriting capability, and a
ConSan-specific contract that warrants a DBI discussion. Use those extractions
to improve the boundaries before committing to the complete decomposition.

Check the current upstream implementation before proposing each chunk. For
example, generic probe-call drains and all-lane spilling already exist;
additional ConSan preservation mechanisms should not be presented as inventing
those safeguards. Likewise, reuse existing sidecar and kernarg facilities.

## Git representation

Keep `shared/rocjitsu/sanitizers` as the source/reference branch. Build the new
stack in a separate worktree:

```text
recorded origin/develop base
    |
    A  Fix scratch allocation alignment
    |
    B  Enable instrumentation scratch from zero
    |
    C  Preserve original-site mappings through text rewriting
    |
    ...
    |
shared/rocjitsu/sanitizers2
```

One commit per prospective PR is the checkpoint format, not a restriction on
daily work. Make frequent local commits and use fixups while developing. Fold
those changes into their owning thematic commits at checkpoints. If changing B
requires adaptations in D and F, update all three rather than leave an
unexplained repair commit at the tip.

Use ordinary Git initially. Add branch pointers at individual chunk boundaries
when useful for review; avoid maintaining dozens of moving pointers before
there is a need. Give substantial rewrites immutable local checkpoint tags and
record their base commits. Do not rely only on the reflog to preserve a week's
worth of decomposition work.

## Build the stack incrementally

The temporary working representation can include an explicitly unsliced
remainder:

```text
base -- A -- B -- WIP: remaining unsliced changes
```

After extracting another theme:

```text
base -- A -- B -- C -- WIP: smaller remaining changes
```

The remainder is not a prospective PR. It preserves the complete implementation
while the meaningful prefix grows. At each extraction verify both:

1. The prefix through the new chunk builds and passes its relevant tests.
2. The complete stack reproduces the agreed reference tree, except for
   explicitly recorded intentional changes.

The first check catches a chunk that secretly depends on a later declaration,
build rule, or runtime behavior. The second catches functionality accidentally
lost during extraction. A passing test run at the full tip does not establish
that intermediate commits are self-contained.

## Define the reference tree precisely

The initial target is the sanitizer functionality integrated onto a pinned
`origin/develop`, preserving newer upstream work and resolving conflicts.
Copying the old sanitizer tip onto a newer base could otherwise revert upstream
changes.

Retain that integrated reference with an immutable local tag. The first
reconstruction should reproduce its tree. Account explicitly for any files
intentionally excluded from the deliverable; do not silently omit inconvenient
changes or checked-in artifacts during slicing.

Initially separate reorganization from architecture changes. First extract, for
example, bounded ConSan evidence as it exists. Discuss its DBI contract in
isolation. Then deliberately revise that chunk and its dependents if a different
contract is agreed. This makes an intentional design change distinguishable
from an extraction mistake.

Preserve working capabilities during decomposition. A narrow first detector PR
can admit only selected kernels, but unsupported cases must be reported
explicitly, and the remaining capabilities stay available at the full tip.

## Iteration and upstream updates

Once `sanitizers2` is the working branch, make it the normal development
destination. Avoid independently evolving both sanitizer branches. Port any
necessary fixes from the reference branch deliberately, recording their place
in the thematic stack.

For an upstream update:

1. Checkpoint the old stack and record its old base.
2. Fetch `origin/develop` and pin the new base SHA.
3. Replay the stack onto that new base, resolving changes in their owning chunks.
4. Compare the old and new series with `git range-diff` and inspect any changed
   behavior or dropped commit.
5. Build and test affected prefixes and the integrated result. Update the
   recorded base and dependency inventory.

Use explicit fetch and rebase rather than relying on the branch's tracking
configuration. Schematic commands, with SHA placeholders replaced:

```text
git fetch origin develop
git rebase --onto <new-origin-develop-sha> <recorded-old-base-sha> shared/rocjitsu/sanitizers2
```

The stack itself stays linear. The preparation merge on the old sanitizer
branch does not establish a merge-based update policy for `sanitizers2`.

Revisit the inventory after rebasing: if upstream has absorbed a chunk, remove
or reduce it rather than retain a duplicate. `range-diff` helps compare versions
of a commit series but does not replace tests. Git documents
[interactive rebase](https://git-scm.com/docs/git-rebase) and
[range-diff](https://git-scm.com/docs/git-range-diff).

## Verification policy

Each prospective PR owns tests for its purpose, with useful coverage at that
prefix. Resource and rewriting changes also exercise existing DBI/DBT users.
Detector changes check preserved application output, actual evidence production,
and detection on known faulty cases; an empty report is not sufficient success.
Retain current semantics such as full 32-bit barrier epochs when slicing.

Qualify substantial integrated checkpoints with the rocjitsu test suite,
including ConSan on emulator and physical gfx1201. Revisit representative
validation and benchmarks when changes to waits, evidence publication, or
runtime ownership could affect detection or performance. Parallelize CPU and
emulator work within the overall 40 GiB memory budget; serialize physical GPU
tests using the shared lock. Keep test results and execution logs outside Git.
