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
the driver's derived value and amd-smi reports source `DRIVER`.

**Library-computed:** the driver publishes nothing. A whole GPU gets a temporary CUID and a
partition gets none. CPU, NIC, NPU and platform CUIDs are derived by root with
the node key, and are temporary for everyone else. The reported source is
`LIBRARY`.

Test both configurations. See `../tests/QA_PLAN.md` for current counts and
untested configurations. The node key lives in the `AmdCuidKey` UEFI variable,
so a driver reload or a reinstallation keeps derived values; temporary values
depend on node-local inputs instead.

## Change status

| Change | Layer | State |
|---|---|---|
| `amend-published-cuid-spec` | Published pages | Specified; not yet applied to the published document |
| `add-cuid-kernel-interface` | `amdgpu` driver | Implemented; whole-GPU testing on two W6800s |
| `pin-cuid-cross-layer-contract` | Format, keys, vectors | Implemented in both trees |
| `integrate-cuid-into-amdsmi` | API, CLI, bindings | Implemented; its daemon and record-store key sources are superseded by `adopt-uefi-key-store`; release artifact checks remain in the QA plan |
| `adopt-uefi-key-store` | UEFI node key, partition UnitID, temporary CUIDs | Implemented in the kernel and the library; the OVMF harness remains |

`amend-published-cuid-spec` contains documentation corrections, not code.
`pin-cuid-cross-layer-contract` defines shared constants and format values.

## Conformance vectors

The normative worked examples live in
`changes/pin-cuid-cross-layer-contract/specs/cuid/conformance-vectors/spec.md`
and exist as a generated artifact shared byte-for-byte between the kernel tree
(`tools/testing/selftests/amdgpu/cuid_vectors.txt`) and the library
(`shared/cuid/tests/vectors/cuid_vectors.txt`).

The generators beside both tables are byte-identical.
`cuid_vectors.py --check` compares it against the local generator;
`cuidtstUnprivileged.ConformanceVectors` asserts every row. The
`vectors-drift-check` job in `.github/workflows/cuid-workflow.yml` runs the
local generator check only; it does not compare repositories.

The kernel's `check_vectors_drift.py` compares both files with the library tree
when `AMDCUID_LIBRARY_TREE` is set. That is the only cross-tree check.

## Corpus checks

`check_conflict_register.py` checks that baseline markers have register rows,
rows name resolutions, and selected constants agree across changes. It does not
validate the meaning of requirements or prove their implementation. Each change
directory carries a `.openspec.yaml` status; task lists record remaining work.
