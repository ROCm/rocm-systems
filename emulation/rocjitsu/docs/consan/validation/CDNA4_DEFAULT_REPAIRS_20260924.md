# CDNA4 Default repairs, September 24, 2026

This follow-up repairs the Default atomic-publication failures from the [initial
revalidation](CDNA4_REVALIDATION_20260924.md). Current results and work in progress are
tracked in [STATUS_CDNA4.md](STATUS_CDNA4.md).

## Repairs

CDNA4 now uses the bounded publication journal already used on RDNA4. It records
observed and written values for supported 32-bit add, OR, CAS and release-store
operations, while retaining other modifications as opaque events. Unsupported or
incomplete evidence does not establish ordering. The port includes:

- Native CDNA4 FLAT/GLOBAL encodings, including SC1, non-returning add/OR,
  unsigned 12-bit FLAT displacements and preservation of address/cache fields.
- Restoration of borrowed scalar address operands before guest replay. An
  earlier experimental port replayed a wide metadata atomic with a clobbered
  scalar base; the new emulator regression exercises that alias explicitly.
- A register-based journal capacity comparison supported on CDNA4, and a
  bounded allocation of 64 publication records per causal slot. Overflow
  remains a completeness failure.
- Recognition of LDS completion at the actual communication instruction,
  including combined VM/LDS waits after a cache operation, while retaining
  the existing RDNA4 proof before the cache prefix.
- Early resource planning for publication modifications, so all required
  modifications can receive instrumentation resources.

The device regression fixtures also needed valid LDS release sequences: compiler-visible
LDS stores on gfx950, and an LDS-zero wait in the inline release helpers for FLAT,
VGLOBAL and full-bank Stream-K. Incorrect controls retain their distinct addresses or
absent publication edges. The global atomic order test now requires a journal-ordered
pair on gfx950 instead of a legacy sync record.

The CDNA release-order fault injector previously removed the cache operation but
preserved the combined VM/LDS-zero wait. That remaining wait independently orders LDS
publication and explains why removing the cache operation need not produce an LDS race.
The repaired mutation removes both parts as one logical fault. A regression verifies the
exact changed bytes, reinventory without the LDS release proof, preservation of the RMW,
and preservation of the release boundary for an acquire-only mutation. VM-only CDNA
sequences retain their existing behavior. Previously recorded misses are not rewritten.

## Fresh Stream-K arrival qualification

The workload binary, native rocprofv3 allowlist, selected atomic add at `.text+0x88f4`,
and 6/8 detection threshold remain fixed. These are new prospective trials with the
repaired release-order mutation:

| Default preset | Matching clean | Detections / admitted trials |
| --- | --- | --- |
| default | Pass | 0/8 |
| high | Pass | 1/8 |
| higher | Pass | 8/8 |

`higher` is the lowest passing preset. All eight qualifying trials have complete
applicable coverage, exact-one mutation installation, reach evidence and healthy
pre/post GPU probes. Three earlier candidate `max` clean runs also passed; they are
supporting clean evidence, not a separate fault qualification.

## Tree atomic-cache repair

Before the hip-moi repair, `max` produced correct numerical output and no ConSan
conflicts for tree atomic-OR, but the independent hip-moi consistency check failed.
That control could not qualify. The final-hook recheck in
`clean-v4/tree-atomic-or/` confirms 12 ordered publication pairs, zero conflicts and
complete applicable coverage.

Post-kernel snapshots in `tree-debug/` establish a concrete hip-moi cache bug:
all four authoritative release records are ready, but failed instrumented runs
have cache masks `0xb` or `0xe` rather than `0xf`. Their missing acquired epoch
matches the omitted producer. The native control and a passing instrumented run
have complete masks and acquired epochs. Cache insertion has bounded retries;
previously, an address hit incorrectly bypassed table lookup even when insertion
had omitted a producer.

The candidate hip-moi repair imports cached producers, then searches the
underlying table for every producer not successfully imported. Two deterministic
regressions (partial and empty masks) fail before this change and pass afterward;
a negative control confirms cache bits cannot invent release records. All four
atomic fast-path tests pass. Fresh native rocprofv3 profiling generated a new
three-kernel allowlist, and all three `max` clean controls pass with the repaired
hip-moi library (`tree-fixed/clean-{0,1,2}/`). The rebuilt producer release OR
is `.text+0x1988c`, occurrence 208, in `fnv1a64:304fa2ce12cd1d95`; the final
acquire/release OR stays at `.text+0x8a24`. Fresh ISA review and prospective
qualification review are retained in `tree-fixed/`. Fresh `default`, `high`, and
`higher` batches have passing matching clean controls and detect 0/8, 1/8, and
8/8 respectively. **`higher` is the lowest passing preset and tree is green.**
All trials are admitted/reached, coverage is complete, and pre/post GPU probes
are healthy. The three matrices are retained in
`tree-atomic-or/{default,high,higher}-cache-repair-v1/`.
The existing SuperCollider `sleep=15` setting was also rerun on this same repaired
binary: its clean control passes and fault detection remains 0/8, with all eight
trials admitted/reached and healthy. It remains yellow. This evidence is in
`tree-atomic-or/sc-sleep15-cache-repair-v1/`. The final audit covers all 23
repair-campaign batches and reports zero errors. No hip-moi oracle, workload synchronization, or numerical check has
been weakened. The repair is committed locally in hip-moi as `346343f`; its
full gfx950 CTest suite passes 184/184 (`tree-fixed/ctest-native-wave-r1.xml`).
Generic two-subgroup atomic tests were changed to use native wave sizes because
their fixed 32-lane split could deadlock inside one CDNA wave64. The initial
CTest runtime-path rejection and the stopped old-fixture deadlock are retained;
GPU discovery and a dispatch smoke test passed before releasing the lock.
The D128 block, D128 pressure, MFMA attention, and Stream-K arrival executables
remain byte-for-byte unchanged (`tree-fixed/rebuilt-fixture-hashes.json`). Evidence: `tree-debug/regression-{before,after}.log` and
`tree-debug/{native,max-0,max-1,max-2}.log`.

## Remaining sampling searches

The fixed reviewed mutations were also tested at `max` with 64, 128, 256, and 512
banks, each with its own clean comparator and prospective 6/8 contract. HipKittens BF16
has completed all four settings: clean passes, 0/8 detections at each. Tensile has
completed all four bank counts with the same result, as have histc and
norm/softmax. All sixteen bank-search batches have passing clean controls, eight
admitted/reached trials, complete coverage, healthy GPU probes, and zero
detections. The audit of these sixteen batches plus the three Stream-K arrival
presets reports no gate or provenance errors (`qualification-audit.json`).

Larger bank counts only address competition between retained wave/workgroup
representatives. The [selection and retention
design](../DESIGN.md#selection-and-retention) hashes dispatch, workgroup and wave
identity, rather than access address or loop iteration, and retains the first successful
representative. Therefore `max` and larger bank counts do not imply a complete dynamic
access history. The observed misses do not by themselves identify the precise lost pair;
changing retention needs separate correctness and capacity qualification.

For Tensile, the 256-bank fault log at the selected store `.text+0x142c`
retains `[0,16)`, `[64,80)`, `[128,144)`, and `[192,208)` for wave owners 0–3.
Those representative intervals are disjoint. The extracted records and log hash
are in `tensile-retention-observation.json`; they show that extra banks did not
add other lanes' addresses at this site. This is a retention limitation, not an
admission or instrumentation failure.

## Evidence and verification

Artifacts are retained under `/home/benjacob/consan-default-repairs-20260924/`. The
final candidate hook is retained with SHA-256
`59ce0ac2fbea2e2e31d83b822485cf765f6fb1f4b1bbdfc5821463be8d2619d9` (`hook-v4.json`).
Qualification is in `streamk-arrival/{default,high,higher}-publication-v4-r1/`, with
prospective specifications, matching controls and every trial's logs and health
evidence. An initial retained-hook path failed the runner's build-directory prerequisite
before executing a workload; that attempt remains in `default-publication-v4/`. The
runner now infers a build-directory prerequisite only for the canonical hook suffix,
allowing explicit retained-hook paths without an indexing crash. Regression tests cover
both retained and canonical paths.

The full CPU/emulator suite passes 3,325 tests (`cpu-v4.xml`), the hook suite passes 285
tests (`hooks-v4.xml`), and all 436 physical tests pass with the final fault-injector
change (`physical-v4.xml`). All 25 available in-scope Default clean rechecks are
complete: 24 passed immediately; the repaired tree now also passes three fresh
clean controls with its native allowlist. The retained-hook
runner fix passes all 207 validation unit tests (`python-validation-r1.log`). All watchpoint-bank searches are complete; results are recorded in the live
ledger.

The first general torch.mode recheck used the older fixture's allowlist and is excluded
from current evidence even though the runner accepted its static coverage. The selected
recheck is `clean-v4-mode-correct-allowlist/`, which uses the current fixture's retained
native trace-derived list and passes.

All physical work uses `/tmp/rocjitsu-consan-destructive-gpu.lock`, the venv TheRock
ROCm stack, and full native rocprofv3 allowlists (regenerated for the repaired tree). Only the tree fixture binary changed among the hip-moi validation rows; its
original and repaired binaries, fresh native trace, allowlist, source patch,
and commit pin are retained. The other four hip-moi binaries are unchanged.
## Lane-retention follow-up

The owner-only bank mapping makes every selected lane in a wave contend for
one immutable representative. Tensile's wrong-address store uses the thread
serial as its byte address, so 16-byte stores from neighboring lanes overlap.
The old mapping retains wave-leading intervals `[0,16)`, `[64,80)`,
`[128,144)`, and `[192,208)` and misses the actual overlapping pairs even with
512 banks.

The candidate partitions CDNA4 tables of at least 64 banks into eight lane
groups per wave bucket when no atomic-publication journal is present. Access
and barrier probes use the same mapping; atomic attachment remains unchanged.
The bank allocation stays fixed. This trades wave/workgroup capacity for lane
diversity and does not provide a complete dynamic history. The candidate now
retains `[56,72)` and `[64,80)` and reports the genuine cross-wave write/write
conflict at the unchanged injected store `.text+0x142c`.

The matching 64-bank Tensile clean control passes with complete access/barrier
coverage. All eight admitted/reached fault trials detect the injected race,
using the original reviewed mutation and full native allowlist. Tensile is
qualified at `max` plus 64 banks. The completed-matrix audit reports zero errors.
The lane-retention repair is committed locally as `a84ec42d79`. Regression checks pass: 3,325 existing
CPU/emulator tests, the new full/sparse-EXEC lane-bank regression, 436 physical
gfx950 tests, and 285 hook tests. Artifacts are `cpu-lanes-v1.xml`,
`lane-retention-unit.log`, `physical-lanes-v1.xml`, `hooks-lanes-v1.xml`, and
`tensile-gfx950-lds-positive/max-lanes-v1-banks-64/` under the repair root.
The frozen candidate hook is recorded in `hook-lanes-v1.json`.

The follow-up scales lane groups with the bank budget: 64/128/256/512 banks
retain up to 8/16/32/64 lane representatives per wave, with at least eight wave
buckets. This addresses softmax's peer reads in lanes 0–7, which an eight-group
partition can still collapse onto lane 0. A first refinement exposed systematic
aliasing of sparse wave IDs in the smaller owner table; folding high owner bits
restores both two- and three-dimensional cross-wave emulator regressions.
The corrected candidate passes all 3,325 existing CPU/emulator tests, the expanded
lane-group regression, and all 436 physical tests (`cpu-lanes-v3.xml`,
`lane-retention-v3-unit.log`, `lane-retention-v3-sparse-owner.log`, and
`physical-lanes-v3.xml`). Its immutable hook is pinned in `hook-lanes-v3.json`.
Fresh external clean rechecks and prospective fault matrices are running. The
superseded comparison controllers stop after their current eight-trial batches;
all completed evidence remains retained, including the rejected intermediate
candidate's two emulator failures.
