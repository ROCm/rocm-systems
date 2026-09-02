# CUID specification workspace

## Layout

- **`specs/cuid/`**: the **published specification**, as it stands. Written down
  from "Persistent platform component identification for SW tools" (S1) at
  version 84. This is the baseline: it records what the page says, including
  where the page contradicts itself. Those contradictions are marked
  `Recorded contradiction` / `Recorded defect` / `Recorded gap` and are *not*
  resolved here, because a baseline that quietly fixed things would stop being a
  baseline.

- **`changes/`**: deltas against that baseline, each with a proposal, a design
  note, tasks, and spec deltas.

## Two tiers

Not a product name and not for a manual; a working distinction the corpus needs
because the requirements differ between them and nothing named the split.

**Tier 1** — the `amdgpu` driver publishes `cuid_primary`, `cuid_secondary`,
`cuid_seed` and `cuid_temporary`. The kernel is the producer, the library and
`amd-smi` report its value verbatim, and the agreement between the two layers
is the property worth testing.

**Tier 2** — the driver publishes nothing and `libamdcuid` is the only
producer. It reads the record store, and computes from the PCIe Device Serial
Number where the store is empty.

**Tier 2 is what ships first.** CUID reaches customers inside AMD SMI one
release before the kernel series lands, so for that release every node is tier
2, and a node stays tier 2 for as long as it runs a kernel without the series.
Tier 1 is the later state, not the normal one.

What that costs, and what has to hold:

| | tier 1 | tier 2 |
|---|---|---|
| Producer | the driver | this library |
| Unprivileged read | always: `cuid_secondary` is 0444 | only once the record store exists, because deriving reads PCIe configuration space |
| Store required | no | yes, for an unprivileged caller |
| Who fills the store | anyone; it is redundant | the first privileged lookup, which now records what it computed |
| `source` reports | `DRIVER` | `STORE`, or `LIBRARY` before anything is recorded |
| Survives a driver reload | **no** — see `CONFLICTS.md` O3 | yes; nothing per-device is held in the kernel |
| Survives an OS reinstall | yes | yes, where a real serial exists; an auxiliary value does not, and says so through bit 117 |

The tier-2 obligations were not met until measured. On a driver built without
`cuid_sysfs_init()`, an unprivileged caller could not read a CUID at all and a
privileged one recomputed on every call because nothing was recorded. Both are
fixed; `tests/QA_PLAN.md` makes tier 2 the first cell of the matrix for the
same reason.

Reading them the wrong way round is the mistake to avoid: a result obtained on
a tier-1 node says nothing about tier 2, because the driver's value stands in
front of every code path tier 2 depends on. Two defects hid there.

## The changes

| Change | Layer | State |
|---|---|---|
| `amend-published-cuid-spec` | The published pages | Specified, not applied; needs Confluence write access |
| `add-cuid-kernel-interface` | `amdgpu` driver | Implemented and verified on two W6800s |
| `pin-cuid-cross-layer-contract` | Cross-layer format, keys, vectors | Implemented in both trees |
| `integrate-cuid-into-amdsmi` | `amd-smi` API, CLI, Python | Implemented |

`amend-published-cuid-spec` is the one with no code. Everything in it is already
implemented by the other three; it exists because the page still describes
something a conforming producer cannot build.

`pin-cuid-cross-layer-contract` is authoritative for any value the layers must
agree on. Where another change states one of those values, that one governs.

## Why the baseline is recorded verbatim

Writing the page down as-is is what made three contradictions countable rather
than anecdotal:

- bit 117 is claimed by UnitID part 2 in the Primary table and by the Auxiliary
  Value Identifier in the Derived table;
- the derived hash slot is 45 bits wide, labelled 46, inside prose that says
  110;
- the auxiliary input structure gives Format 17 bits and Machine ID 127, so the
  ranges total 256 while neither field is the size its own description needs.

Each had already reached shipping code, and each was found by hand, months
apart, by comparing code to code.

## Conformance vectors

The normative worked examples live in
`changes/pin-cuid-cross-layer-contract/specs/cuid/conformance-vectors/spec.md`
and exist as a generated artifact shared byte-for-byte between the kernel tree
(`tools/testing/selftests/amdgpu/cuid_vectors.txt`) and the library
(`projects/cuid/tests/vectors/cuid_vectors.txt`).

Each tree checks its own copy against the generator, so a hand-edited vector
fails a build: in this tree that is `cuid_vectors.py --check`, run by the
`vectors-drift-check` job in `.github/workflows/cuid-workflow.yml`, and the
library asserts every vector in `cuidtstUnprivileged.ConformanceVectors`.

Be precise about what that buys. `cuid_vectors.py --check` compares the table
against the generator **sitting beside it**, so it catches a hand-edited table
and nothing else. The only check that compares the two trees is
`check_vectors_drift.py` in the kernel tree, which sha256-compares both files
against `$AMDCUID_LIBRARY_TREE`; it runs on the kernel side only, and only when
that variable is set. Nothing in this repository notices if the kernel's copy of
the generator diverges from this one. See the TODO on the `vectors-drift-check`
job.

## Checking the corpus

`check_conflict_register.py` is a bookkeeping check over `CONFLICTS.md`: markers
in `specs/` have rows, rows have non-placeholder resolutions, and a short list of
named constants is stated consistently across the change dirs. It does not read
the specs for meaning and does not verify that any resolution was implemented. A
clean run is not a claim that the four proposals agree. It was passing while
`add-cuid-kernel-interface` specified an at-most-32 seed against a kernel and two
sibling proposals that all said exactly-32; the constant cross-check was added
because of that.

Each change dir carries a `.openspec.yaml` with an explicit `status:`.
