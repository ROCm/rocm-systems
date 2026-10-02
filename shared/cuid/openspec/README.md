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

**Driver-published:** the driver publishes CUID attributes. The library reads
the driver's derived value and amd-smi reports source `DRIVER`. A driver that
publishes only `cuid_primary` and `cuid_unit_id` leaves a whole GPU keyed like
a CPU and a partition without a CUID.

**Library-computed:** the driver publishes nothing. A whole GPU gets a temporary CUID and a
partition gets none. CPU, NIC, NPU and platform CUIDs are derived by root with
the node key, and are temporary for everyone else. The reported source is
`LIBRARY`.

Test both configurations; `../tests/QA_PLAN.md` has the test matrix and its
known gaps. amdgpu holds the node key only in memory, so derived
values need the key set again after every reboot or driver reload; temporary
values depend on node-local inputs instead.

## Change status

| Change | Layer | State |
|---|---|---|
| `amend-published-cuid-spec` | Published pages | Specified; not yet applied to the published document |
| `add-cuid-kernel-interface` | `amdgpu` driver | Implemented; whole-GPU testing on two W6800s |
| `pin-cuid-cross-layer-contract` | Format, keys, vectors | Implemented in both trees |
| `integrate-cuid-into-amdsmi` | API, CLI, bindings | Implemented; its daemon and record-store key sources are superseded by `split-identity-from-key-store` and `add-volatile-node-key`; release artifact checks remain in the QA plan |
| `split-identity-from-key-store` | Driver identity without a key, partition UnitID, temporary CUIDs | Implemented; kernel tested on two W6800s and on 8x MI350X in SPX, DPX, QPX and CPX |
| `add-volatile-node-key` | In-memory node key set through `cuid_seed` | Implemented; kernel, sanitizer and MI350X checks done |
| `adopt-uefi-key-store` | Node key kept in a UEFI variable | Proposed as the next step after `add-volatile-node-key`; needs an efivarfs change |

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
