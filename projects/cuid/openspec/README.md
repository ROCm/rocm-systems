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
