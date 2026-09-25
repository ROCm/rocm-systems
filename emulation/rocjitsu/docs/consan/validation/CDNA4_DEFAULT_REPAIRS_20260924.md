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

## Remaining tree failure

At `max`, the candidate produces correct numerical output and no ConSan conflicts for
tree atomic-OR. The independent hip-moi consistency check still fails, so the cell
remains red and cannot qualify on that clean control. The final-hook recheck in
`clean-v4/tree-atomic-or/` confirms 12 ordered publication pairs, zero conflicts and
complete applicable coverage.

Inspection of hip-moi's `context.hpp` shows that release-RMW metadata is published after
the hardware atomic. Its acquiring RMW retries stop once any producer metadata is found,
even when another producer's metadata is still pending. The earlier diagnostic build
reported missing producer-to-consumer ordering while ConSan's complete journal ordered
the LDS pairs. This is a suspected timing defect in the independent oracle, not
justification for ignoring it. No hip-moi oracle, workload synchronization or numerical
check has been weakened. A robust repair needs metadata tied to the atomic's observed
value; merely increasing a retry delay would not establish that contract.

## Remaining sampling searches

The fixed reviewed mutations are also being tested at `max` with 64, 128, 256, and 512
banks, each with its own clean comparator and prospective 6/8 contract. HipKittens BF16
has completed all four settings: clean passes, 0/8 detections at each. Tensile has
completed all four bank counts with the same result; histc and norm/softmax
are still running.

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
complete: 24 pass and tree remains rejected by its independent oracle. The retained-hook
runner fix passes all 207 validation unit tests (`python-validation-r1.log`). The
remaining watchpoint-bank searches are in progress; results are tracked in the live
ledger.

The first general torch.mode recheck used the older fixture's allowlist and is excluded
from current evidence even though the runner accepted its static coverage. The selected
recheck is `clean-v4-mode-correct-allowlist/`, which uses the current fixture's retained
native trace-derived list and passes.

All physical work uses `/tmp/rocjitsu-consan-destructive-gpu.lock`, the venv TheRock
ROCm stack, and the unchanged full native rocprofv3 allowlists. Source fixtures and
binaries used by the external validation rows are unchanged.