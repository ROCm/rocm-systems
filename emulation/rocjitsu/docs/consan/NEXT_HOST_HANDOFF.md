# ConSan production-reimplementation handoff

This document is the restart state for moving the active ConSan refactoring
from the gfx950 host that was retired on 2026-08-28 to another machine. It is
deliberately more operational than the design documents: a new Codex agent
should be able to reconstruct the branch, build, test topology, current design
judgment, and next action without relying on conversation history.

## Read this first

1. **Do not start from the current remote branch until the old host's commits
   have been pushed.** At the source snapshot described here, local
   `shared/rocjitsu/sanitizers` was 240 commits ahead of
   `origin/shared/rocjitsu/sanitizers`. The last code commit was
   `28a35a910e06648f22813ddfddd43eb848b0634e`; this handoff document is committed
   immediately after it.
2. The active task is to complete Stages 7 through 10 in
   [IMPLEMENTATION_SIZE_STUDY.md](IMPLEMENTATION_SIZE_STUDY.md), not to begin a
   new design effort. Stages 0 through 6 are complete. Stage 7 is in progress.
3. The full Stage 7 ConSan gate has **not** completed at the current code
   revision. It was intentionally terminated when this host-removal handoff
   became urgent. Focused Stage 7 tests were green; that is not a substitute
   for the required full gate.
4. Use RocJitsu itself for all emulation. The earlier FFM-based gfx1250 path is
   obsolete and must not be revived.
5. Do not disable ccache to work around a permission problem. Check that its
   directory is writable before building and report a permission problem
   immediately.

## Active objective and working rules

Carry out the plan under **“Where to turn attention: incremental refactoring
plan”** in [IMPLEMENTATION_SIZE_STUDY.md](IMPLEMENTATION_SIZE_STUDY.md).

The explicit completion requirements are:

- create at least one git commit for every Stage 0 through 10;
- run every checked-in ConSan test, including device tests, at least once per
  stage;
- run focused host or device subsets more often when they are informative, but
  batch compatible edits so testing does not dominate the work;
- preserve observable behavior aligned with ConSan's goals and keep the
  checked-in tests green;
- if a test unnecessarily entrenches an incidental prototype implementation
  trait rather than the behavioral contract, explain and repair the test
  instead of preserving the accidental trait;
- perform component-sized, fully incremental cutovers. There is no big-switch
  rewrite: old state or helpers are deleted as each narrow replacement becomes
  authoritative;
- seek substantial design improvement and code deletion. There is an explicit
  mandate to change ConSan data structures and implementation design, not just
  rename or locally tidy the prototype;
- do not distort the code merely to reduce a line counter. Prefer low-hanging
  deletion and clean boundaries that enable deeper later simplification;
- do not spend time on E2E revalidation during this implementation-size plan.
  The checked-in host and device suites are the iteration gate; E2E remains a
  later qualification gate;
- new enums and types receive generous explanatory comments, and all
  reasonably testable unit behavior receives direct host-unit tests;
- add a regression test whenever a defect is found and fixed. Transport a
  behavioral test across other gfx families when the underlying contract is
  cross-cutting rather than target-exclusive;
- format modified C/C++ sources with `clang-format` before committing;
- use 64-way build and test parallelism on a similarly wide replacement host,
  but do not encode that machine-local preference in shared project files; and
- avoid `rm` commands in this workflow. They previously caused interactive
  permission prompts and blocked unattended progress.

The new tests created before this refactoring are the primary behavioral asset.
Device tests pair a correct workload, requiring exact results and no ConSan
diagnostic, with an adjacent incorrect workload requiring the intended
diagnostic. Do not weaken these pairs merely to make a refactor pass.

## Git state at handoff

| Item | Value at the source snapshot |
| --- | --- |
| Repository | `/home/ossci/xx/TheRock/rocm-systems` |
| Local branch | `shared/rocjitsu/sanitizers` |
| Configured upstream | `origin/shared/rocjitsu/sanitizers` |
| Last code commit | `28a35a910e06648f22813ddfddd43eb848b0634e` |
| Remote branch before transfer | `97c1640b2f6529ef59a6dcf1068243107e190d09` |
| Ahead/behind before this handoff commit | 240 ahead, 0 behind |
| Latest develop merge | `6f03a8a3e4` (`Merge origin/develop into shared/rocjitsu/sanitizers`) |
| Local `origin/develop` and merge base | `e36e73e19cacc47bf496c0871170f9096dba5521` |
| Worktree before adding this document | Clean |

The branch intentionally tracks `origin/shared/rocjitsu/sanitizers`; do not
rename it back to a user branch. Develop updates have been incorporated with
merge commits, not by rebasing the shared branch.

After committing this document, transfer the work with a normal non-force
push:

```sh
git push origin \
  shared/rocjitsu/sanitizers:shared/rocjitsu/sanitizers
```

Then verify that the remote points to the committed handoff tip:

```sh
git ls-remote origin refs/heads/shared/rocjitsu/sanitizers
git rev-parse HEAD
```

On the new host, do not infer success from a branch name alone. Fetch and
verify the last code commit explicitly:

```sh
git fetch origin shared/rocjitsu/sanitizers
git switch --track origin/shared/rocjitsu/sanitizers
git merge-base --is-ancestor \
  28a35a910e06648f22813ddfddd43eb848b0634e HEAD
git status --short --branch
```

If a same-named local branch already exists, switch to it, confirm its upstream,
and fast-forward it normally; do not force-reset a worktree with unknown local
changes.

## Stage ledger

The detailed exit evidence through Stage 6 is authoritative in
[IMPLEMENTATION_SIZE_STUDY.md](IMPLEMENTATION_SIZE_STUDY.md) and
[REIMPLEMENTATION_MAP.md](REIMPLEMENTATION_MAP.md). This is the concise restart
index.

| Stage | State | Boundary or exit commit | Full checked-in gate evidence |
| ---: | --- | --- | --- |
| 0 | Complete | `bd75474457` | 5,245/5,245; 3,501 device; 593 physical gfx950; 578 seconds wall |
| 1 | Foundation complete; cross-cutting target gate remains active through Stage 8 | `7f85506d23` | 5,245/5,245; 568.71 seconds wall |
| 2 | Complete | `ed9859a144` | All 5,247 ultimately passed; one physical timeout passed alone immediately; nonphysical 4,654/4,654 |
| 3 | Complete | `453485e7ab` | 5,271/5,271; 3,501 device; 593 physical; 574.58 seconds wall |
| 4 | Complete | `2cbbf16f2f` | All 5,276 ultimately passed; one teardown timeout passed alone immediately; 618.61 seconds wall |
| 5 | Complete | implementation `2dd3e217e9`, evidence commit `36e61c997f` | 5,279/5,279; 560.13 seconds wall |
| 6 | Complete | `54d410cb1e` | 5,282/5,282; 3,501 device; 593 physical; 566.36 seconds wall |
| 7 | **In progress** | `46db9ca6a2`, `3d361d3691`, `28a35a910e` | Focused gates green; full 5,283-test gate interrupted and must be rerun |
| 8 | Not started | — | Required after the Stage 8 commit(s) |
| 9 | Not started | — | Required after the Stage 9 commit(s) |
| 10 | Not started | — | Required final audit and gate |

The increasing totals reflect direct regression tests added by the stages. The
current build registers 5,283 tests selected by `ConSan|consan`; 3,501 carry
the `consan-device` label, including 593 physical gfx950 rows and 2,908
nonphysical device rows.

## Exact Stage 7 state

Stage 7 deletes flavor-private copies of mechanics that now have shared
planning, placement, resource, target-emission, synchronization, evidence, and
mutation contracts. Flavor-specific semantics must remain explicit:

- InlineShadow owns its exact shadow representation, cell/epoch/token state
  transitions, and exact evidence protocol;
- Sampled owns statistical selection, sampling windows, and sampled evidence
  semantics;
- Record/Replay owns replay-event meaning, payload, causal stream, and host
  replay/trust behavior;
- shared code owns mechanics whose invariant is genuinely identical across
  those flavors. Similar-looking code with different evidence meaning must not
  be merged merely for textual deduplication.

### Stage 7 commits already made

`46db9ca6a2` (`rocjitsu: consolidate ConSan flavor mechanics`) did the first
mechanical tranche:

- shared candidate ordering and synchronization-reservation inventory;
- removed flavor-private host rediscovery/filtering;
- moved common scalar-spill mechanics out of InlineShadow;
- reused target-profile facts and shared synchronization inventory ranges; and
- added common access-application helpers.

It changed production by 259 additions and 374 deletions. A focused 509-test
cross-flavor gate passed.

`3d361d3691` (`rocjitsu: share ConSan access lowering mechanics`) consolidated
the common appended-access path:

- common descriptor requirements and appended-text initialization;
- common dense patch grouping, dense dispatcher emission, and dense entry-host
  emission;
- common pre-reserved/local-NOP/anchor-relocation planning through
  `MoiAccessEntryIslandPlan` and `plan_moi_access_entry_island`;
- Record/Replay and Sampled access resource planning through
  `plan_moi_probe_resources`;
- typed SCC restoration through `MoiEncodedSccRestoreRequest`, including the
  bit-test-only case; and
- direct host coverage in `moi_common_test.cpp`.

A subtle but important distinction was retained in the common dense-host
helper: InlineShadow restores SCC from its indirect-jump assignment, whereas
Record/Replay and Sampled use the common special-state assignment. The helper
accepts this semantic choice explicitly through `use_indirect_jump_scc_save`;
do not collapse the distinction by accident. The same focused 509-test gate
passed after this correction.

`28a35a910e` (`rocjitsu: use ConSan target facts in flavor lowering`) replaced
remaining product helper decisions in the touched InlineShadow and Sampled
paths with already-resolved target-profile facts. A focused 51-test
cross-target gate covering VGPR-bank, runtime-workgroup, and Sampled barrier
behavior passed.

Relative to the Stage 6 revision `54d410cb1e`, the three Stage 7 commits make
780 production additions and 905 production deletions: a net reduction of 125
physical implementation lines. The current 79-file production scope has
95,118 physical lines and 90,700 nonblank lines, compared with 95,243 and
90,835 at Stage 6. This is useful consolidation, but line reduction alone does
not prove the Stage 7 exit criteria.

### The interrupted full gate

The following command was started from `/home/ossci/xx/rocjitsu-build` after
`28a35a910e`:

```sh
ctest -j64 --output-on-failure -R 'ConSan|consan'
```

It was terminated on request after approximately 213 seconds because the host
was being removed. There was no final CTest result, and no conclusion should be
drawn from the partial run. No source edit was made after the focused tests or
during the partial gate.

### Remaining Stage 7 judgment and work

Do not mechanically declare Stage 7 complete just because its three tranches
are green. Audit the strict exit text in
[IMPLEMENTATION_SIZE_STUDY.md](IMPLEMENTATION_SIZE_STUDY.md#stage-7-delete-residual-flavor-private-mechanics).
In particular:

1. `consan_moi_record_planning.inc` still contains
   `bind_moi_record_event_options`, explicitly documented as the remaining
   compatibility boundary between `MoiPlannedRecordEvent` and native emitters
   that still consume broad mutable `MoiOptions`. It is used by atomic, fence,
   and barrier Record/Replay emission. Decide whether Stage 7 can replace it
   with a genuinely narrow typed emission input. Merely moving the same broad
   mutation into a method or storing an entire `MoiOptions` copy in the plan is
   not progress.
2. Some flavor-owned target mechanics remain. Examples at the source snapshot
   include `consan_arch_has_s_call_i64` in
   `consan_moi_inline_shadow_emission.inc`,
   `consan_is_capability_arch` in `consan_moi_record_atomic.inc`, and several
   direct `ConSanEncodingFamily` decisions in InlineShadow, Inline atomic, and
   Sampled synchronization emission. A target-profile fact used as an ABI fact
   may be legitimate; a flavor-private native-selection branch is not. Apply
   the Stage 1 taxonomy rather than blindly eliminating all target vocabulary.
3. Re-read the three large flavor bodies after the consolidation. Current
   physical sizes are approximately 8,647 lines across the three InlineShadow
   files, 4,360 across the two Sampled files, and 1,908 across the three
   Record/Replay files. Confirm that the residual size differences can be
   explained by the unique semantics above, not private inventory,
   allocation, ABI, placement, or common emission loops.
4. Add a short **Stage 7 exit evidence** subsection to
   [IMPLEMENTATION_SIZE_STUDY.md](IMPLEMENTATION_SIZE_STUDY.md). State each
   flavor's surviving responsibility, deletions and size delta, focused test
   evidence, the full-gate result, and any deliberately retained target-owned
   boundary. Do not claim a boundary is gone if an adapter still embodies it.
5. Build, run an appropriate focused gate after any further edit, then run the
   entire checked-in ConSan gate once at the final Stage 7 revision. Commit the
   exit evidence. Stage 7 already has more than the required one commit, but it
   is not complete until the design and test exit criteria are true.

## Stages 8 through 10

These stages have not started. Follow their authoritative text rather than
treating this summary as a replacement.

### Stage 8: SuperCollider

Transport the production boundaries to SuperCollider. Its current principal
files are about 1,065 physical lines of common code, 1,956 FLAT lines, and
4,285 LDS lines. Reuse shared access classification, inventory, descriptor
growth, resource planning, placement, and target operations only where the
semantic contract matches. Preserve SuperCollider's distinct race-detection
semantics. Compare FLAT and LDS by semantic operation and target requirement,
not instruction spelling. The exit test is that remaining SuperCollider code
is explainable as sanitizer semantics or unavoidable target lowering rather
than historical pipeline duplication.

Stage 1's target-boundary gate remains active through this stage. Stage 8 must
have at least one identifiable commit and one complete checked-in ConSan gate.

### Stage 9: hook and DBI boundary

Narrow the hook-facing surface without building a local imitation of future
DBI. Make the hook consume production request/result/report-layout contracts,
separate report interpretation from generic HSA lifecycle and code-object
coordination, and create an explicit keep/delete map. Document lifecycle,
ownership, before/after instrumentation, and host-processing responsibilities
that DBI will eventually replace.

Relevant DBI decisions already received from its owner:

- near-term DBI processing is host-only, so Record/Replay is the first and most
  important migration target;
- other engines may follow later as on-device processing support evolves;
- production ConSan remains in-tree in RocJitsu;
- DBI intends to support both before and after instrumentation;
- continue using `HSA_TOOLS_LIB` for now; DBI will eventually own the hook;
- translated-code instrumentation is deferred until a concrete use case; and
- test placement can remain under current ConSan ownership for now.

The hook layer is intentionally not a cosmetic cleanup target while DBI's own
design is evolving. Delete code only when a production contract or an
established DBI contract owns the responsibility.

### Stage 10: validation and final deletion audit

Make validation consume authoritative plans, operating points, and target
facts. Retain independent validation of emitted bytes, ABI conformance, bounds,
and behavioral invariants; delete only mirror reconstruction made redundant by
the typed design. Search systematically for legacy projections, aliases,
duplicate target facts, temporary adapters, write-only state, and stale design
language. Update [DESIGN.md](DESIGN.md), [PRODUCTION_DESIGN.md](PRODUCTION_DESIGN.md),
[IMPLEMENTATION_SIZE_STUDY.md](IMPLEMENTATION_SIZE_STUDY.md), and the Stage 0
ledger to describe the implementation that actually exists.

Run the complete host and device gate at the final revision. If physical
gfx950 is present, include its serialized tier. Audit every explicit Stage
0–10 requirement and commit rather than treating green tests alone as proof of
completion.

## Reconstructing the build on the new host

Do not copy the old CMake build directory. It contains absolute paths and is
host-specific. Configure a fresh Ninja build against the ROCm runtime from the
new host's own `TheRock/build`.

The old host used:

| Purpose | Old-host path |
| --- | --- |
| TheRock checkout | `/home/ossci/xx/TheRock` |
| ROCm source repository | `/home/ossci/xx/TheRock/rocm-systems` |
| TheRock build | `/home/ossci/xx/TheRock/build` |
| Built ROCm prefix/runtime | `/home/ossci/xx/TheRock/build/dist/rocm` |
| RocJitsu CMake build | `/home/ossci/xx/rocjitsu-build` |
| gfx950 hip-moi artifact tree | `/home/ossci/xx/hip-moi-build-gfx950-tests` |
| gfx1250 hip-moi artifact tree | `/home/ossci/xx/hip-moi-build-gfx1250-tests` |

The old CMake configuration used Ninja, `RelWithDebInfo`, LLVM 21.1.8 clang,
`BUILD_TESTING=ON`, `RJ_ENABLE_HIP_TESTS=ON`, and both C and C++ compiler
launchers set to `ccache`. `CMAKE_PREFIX_PATH`, `ROCM_PATH`, and
`CMAKE_INSTALL_PREFIX` all pointed at the in-workspace TheRock ROCm prefix (and
its `lib/rocm_sysdeps` directory). It also named the two hip-moi artifact trees
above. The gfx942 hip-moi path was empty in this particular build.

A corresponding fresh configuration, with paths adapted to the replacement
workspace, is:

```sh
cmake -S "$WORKSPACE/TheRock/rocm-systems/emulation/rocjitsu" \
  -B "$WORKSPACE/rocjitsu-build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER="$LLVM_ROOT/bin/clang" \
  -DCMAKE_CXX_COMPILER="$LLVM_ROOT/bin/clang++" \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_PREFIX_PATH="$WORKSPACE/TheRock/build/dist/rocm;$WORKSPACE/TheRock/build/dist/rocm/lib/rocm_sysdeps" \
  -DCMAKE_INSTALL_PREFIX="$WORKSPACE/TheRock/build/dist/rocm" \
  -DROCM_PATH="$WORKSPACE/TheRock/build/dist/rocm" \
  -DBUILD_TESTING=ON \
  -DRJ_ENABLE_HIP_TESTS=ON \
  -DRJ_CONSAN_GFX950_HIP_MOI_BUILD_DIR="$WORKSPACE/hip-moi-build-gfx950-tests" \
  -DRJ_CONSAN_GFX1250_HIP_MOI_BUILD_DIR="$WORKSPACE/hip-moi-build-gfx1250-tests"
```

`WORKSPACE` and `LLVM_ROOT` above are explanatory placeholders, not shared
project settings. Omit an external hip-moi argument if that exact artifact tree
does not yet exist; CMake deliberately rejects a stale or wrongly named path.
Recreate the missing artifact before claiming the same full test inventory.
[VALIDATION.md](VALIDATION.md#workspace-contract) documents the target-specific
hip-moi tree contract.

Build with:

```sh
cmake --build "$WORKSPACE/rocjitsu-build" -j64
```

The old host had `libzstd-dev`, zlib development files, and `jq` installed;
these had been needed during the clean rebuild and test work. Install equivalent
development packages on the new image if configuration reports them missing.

### ccache prerequisite

On the old host, `CCACHE_DIR=/tmp/ccache`, with a 40 GiB limit, and ccache was
active for both C and C++. On the new host, check it before configuring or
building:

```sh
ccache -p
ccache -s
```

Confirm that the reported cache directory exists and is writable by the
current user. If it is outside the writable workspace or produces a permission
error, stop and tell the user immediately. Do **not** remove the compiler
launcher flags and do not set ccache off as a workaround.

### ROCm runtime and physical GPU prerequisite

Use the ROCm runtime built by this workspace, not a system ROCm installation.
Before physical testing, put the built prefix first and inspect the detected
agents, for example:

```sh
export PATH="$WORKSPACE/TheRock/build/dist/rocm/bin:$PATH"
export LD_LIBRARY_PATH="$WORKSPACE/TheRock/build/dist/rocm/lib:$WORKSPACE/TheRock/build/dist/rocm/lib/rocm_sysdeps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
rocminfo | rg 'gfx[0-9]+'
```

The checked-in CMake currently auto-registers the physical gfx950 tier only
when a gfx950 agent is present. That behavior was reviewed and intentionally
left unchanged. A host with no gfx950 will therefore have fewer tests; do not
hardwire a fake physical gfx950 assumption. Record the new inventory and run
all applicable physical rows if the replacement host exposes another
supported physical tier.

## Test commands and expected inventory

Always enumerate first on a fresh build because hardware and optional hip-moi
artifacts affect registration:

```sh
ctest --test-dir "$WORKSPACE/rocjitsu-build" -N -R 'ConSan|consan'
ctest --test-dir "$WORKSPACE/rocjitsu-build" -N -L consan-device -LE physical
ctest --test-dir "$WORKSPACE/rocjitsu-build" -N -L physical -R 'ConSan|consan'
```

At the handoff snapshot the corresponding counts were:

- all checked-in ConSan tests: **5,283**;
- nonphysical ConSan tests: **4,690**;
- nonphysical `consan-device` rows: **2,908**;
- physical gfx950 rows: **593**; and
- all device rows: **3,501**.

The five RocJitsu-emulated architecture families are gfx942 (CDNA3), gfx950
(CDNA4), gfx1250 (CDNA5), gfx1100 (RDNA3), and gfx1201 (RDNA4). gfx950 is also
repeated on physical hardware when available.

Fast device-only iteration without physical serialization:

```sh
ctest --test-dir "$WORKSPACE/rocjitsu-build" \
  -j64 --output-on-failure -L consan-device -LE physical
```

All nonphysical ConSan tests:

```sh
ctest --test-dir "$WORKSPACE/rocjitsu-build" \
  -j64 --output-on-failure -R 'ConSan|consan' -LE physical
```

Physical tier only, when registered:

```sh
ctest --test-dir "$WORKSPACE/rocjitsu-build" \
  -j64 --output-on-failure -L physical -R 'ConSan|consan'
```

The physical tests share the target-scoped CTest resource lock
`consan_physical_gfx950`; `-j64` is correct because CTest serializes only the
tests contending for that GPU while unrelated work remains parallel.

The required full per-stage gate is:

```sh
ctest --test-dir "$WORKSPACE/rocjitsu-build" \
  -j64 --output-on-failure -R 'ConSan|consan'
```

On the old wide host this typically took about 9.3–10.3 minutes with all 593
serialized physical rows. The emulator-only device suite is much faster. Use
focused CTest regular expressions or direct GTest filters during an edit
tranche, verify the selected count with `ctest -N`, and reserve the full gate
for the stage checkpoint or a high-risk cross-cutting change.

## Important source and document map

Read these in order when resuming:

1. [IMPLEMENTATION_SIZE_STUDY.md](IMPLEMENTATION_SIZE_STUDY.md), especially the
   incremental plan and Stage 7–10 exit criteria.
2. [REIMPLEMENTATION_MAP.md](REIMPLEMENTATION_MAP.md), which contains the Stage
   0 responsibility map, target-reference ledger, end-state boundaries,
   cut sequence, and initial deletion opportunity estimate.
3. [PRODUCTION_DESIGN.md](PRODUCTION_DESIGN.md), the detailed production design
   and incremental migration record.
4. [DESIGN.md](DESIGN.md), the higher-level component and dependency view.
5. [PLAN.md](PLAN.md), for the wider production-migration constraints and DBI
   timeline.
6. [PRODUCTION_BASELINE.md](PRODUCTION_BASELINE.md) and
   [PLAN_DEVICE_TESTS.md](PLAN_DEVICE_TESTS.md), for the behavioral contracts
   that refactoring must preserve.

Current Stage 7 implementation concentration:

- `consan_moi_access_apply.inc`: shared appended-access mechanics introduced by
  the current stage;
- `consan_moi_placement.inc` and `consan_moi_probe_planning.inc`: shared
  resource, placement, and probe plans consumed by flavor paths;
- `consan_moi_inline_shadow.inc`, `consan_moi_inline_shadow_emission.inc`, and
  `consan_moi_inline_atomic.inc`: InlineShadow planning and exact-shadow
  emission;
- `consan_moi_sampled_access.inc` and `consan_moi_sampled_sync.inc`: Sampled
  policy and access/synchronization lowering;
- `consan_moi_record_replay.inc`, `consan_moi_record_atomic.inc`, and
  `consan_moi_record_planning.inc`: Record/Replay access and synchronization
  path, including the remaining event-options compatibility seam; and
- `tests/patch/consan/moi_common_test.cpp`: direct coverage for the latest
  shared access/SCC plans.

## First-session checklist on the replacement host

1. Verify the remote branch contains `28a35a910e` and this handoff commit.
2. Check out local `shared/rocjitsu/sanitizers` tracking the origin branch.
3. Fetch origin and record whether `origin/develop` advanced. If it did, merge
   it with a merge commit only after establishing a green baseline; do not
   rebase this shared branch.
4. Verify ccache and report any permission failure instead of disabling it.
5. Verify the in-workspace ROCm runtime and enumerate any physical GPU.
6. Rebuild RocJitsu from a fresh CMake directory; do not copy the old absolute-
   path build tree.
7. Enumerate the ConSan tests and compare optional/hardware-dependent deltas
   with the 5,283/4,690/2,908/593 snapshot above.
8. Run the focused Stage 7 host and emulator subsets. Investigate any new-host
   difference before changing code.
9. Complete the remaining Stage 7 boundary audit and any justified cleanup.
10. Run the full applicable ConSan gate at the final Stage 7 revision, record
    exit evidence, commit it, and only then begin Stage 8.

The source-host worktree had no hidden uncommitted implementation work. The
only in-flight activity was the terminated full CTest run, so there is no
patch, generated source, or test result outside git that must be recovered.
