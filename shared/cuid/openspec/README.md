# CUID specification workspace

## Layout

* `specs/cuid/` records version 84 of "Persistent platform component
  identification for SW tools" (S1), including its contradictions. Markers
  `Recorded contradiction`, `Recorded defect` and `Recorded gap` point to
  resolutions; they do not amend the baseline text.
* `changes/` contains proposals, design notes, task lists and spec deltas.
* `CONFLICTS.md` maps the baseline's defects to their resolutions and records
  cross-layer decisions and open questions.

## Lookup configurations

**Driver-published:** the driver publishes `cuid_primary` and `cuid_unit_id`.
Root reads the driver's primary CUID.

**Library-computed:** the driver publishes nothing. A whole GPU gets a
temporary CUID built by the library.

There is no node key, so in both configurations every derived CUID is
temporary, with source `LIBRARY`, and a partition gets none. Test both
configurations. See `../tests/QA_PLAN.md` for current counts and untested
configurations.

## Change status

| Change | Layer | State |
|---|---|---|
| `amend-published-cuid-spec` | Published pages | Specified; not yet applied to the published document |
| `add-cuid-kernel-interface` | `amdgpu` driver | Implemented; whole-GPU testing on two W6800s |
| `pin-cuid-cross-layer-contract` | Format, keys, vectors | Implemented in both trees |
| `integrate-cuid-into-amdsmi` | API, CLI, bindings | Implemented; its daemon and record-store key sources are superseded by `split-identity-from-key-store`, and seed provisioning waits for the key store; release artifact checks remain in the QA plan |
| `split-identity-from-key-store` | Driver identity without a key, partition UnitID, temporary CUIDs | Library and amd-smi implemented; kernel and partition hardware testing remain |

`amend-published-cuid-spec` contains documentation corrections, not code.
`pin-cuid-cross-layer-contract` defines shared constants and format values.

## Conformance vectors

The normative worked examples live in
`changes/pin-cuid-cross-layer-contract/specs/cuid/conformance-vectors/spec.md`
and as the generated table `shared/cuid/tests/vectors/cuid_vectors.txt`, which
is the source of truth.

`cuid_vectors.py --check` compares the table against its generator;
`cuidtstUnprivileged.ConformanceVectors` asserts every row. The
`vectors-drift-check` job in `.github/workflows/cuid-workflow.yml` runs the
generator check.

The kernel has no copy of the table. A KUnit known-answer test in amdgpu
asserts a subset of its rows; nothing compares that subset with this table
automatically.

## Corpus checks

`check_conflict_register.py` checks that baseline markers have register rows,
rows name resolutions, and selected constants agree across changes. It does not
validate the meaning of requirements or prove their implementation. Each change
directory carries a `.openspec.yaml` status; task lists record remaining work.
