# ConSan semantic capability matrix

This document is the normative supported-form contract for ConSan on
`gfx942`, `gfx950`, `gfx1201`, and `gfx1250`. It describes equivalent memory
and synchronization semantics, not identical ISA mnemonic sets. The
target-specific status ledgers record workload qualification separately.

A form marked **supported** is decoded into the shared semantic inventory and
has a lowering path for the named engine. Register pressure, report capacity,
placement, or bounded runtime-state exhaustion can still make a particular
site dynamically incomplete. Those outcomes are reported explicitly; they do
not turn an unsupported or incomplete run into a clean result.

## Engine contract

| Semantic form | SuperCollider | Record/Replay | Sampled | Inline Shadow |
| --- | --- | --- | --- | --- |
| Ordinary native LDS read/write | Redundant observation and mismatch marker | Bounded access record and host replay | Selected causal window | Exact supported-cell shadow update |
| Proven or likely-group FLAT read/write | Same, after target-specific LDS-offset normalization | Same as native LDS | Same as native LDS | Same as native LDS |
| Workgroup barrier | Mutation may compose with an access probe; no ordering claim | Barrier-arrival record and host epoch coalescing | Qualified barrier metadata in the selected causal bank | Execute the original barrier, then advance the device epoch |
| Cluster barrier | Mutation only; no ordering claim | Supported on `gfx1250` | Supported on `gfx1250` | Supported on `gfx1250` |
| Ordered atomic RMW | Mutation may compose with an access probe; no ordering claim | Addressed release/acquire record and host replay | Qualified ordering metadata in the selected causal bank | Bounded address-scoped release/acquire transaction |
| Ordered compare-exchange | Mutation only; no ordering claim | Supported when the dynamic outcome is available | Supported when the dynamic outcome is available | Supported when the dynamic outcome is available |
| Ordinary communication plus cache/fence sequence | Mutation only; no ordering claim | Dedicated addressed fence record and host replay | Not a standalone form; ordering must be associated with an admitted atomic sequence | Not a standalone form; ordering must be associated with an admitted atomic sequence |
| Relaxed LDS RMW used as an LDS access | Redundant observation is not claimed for atomics | Access evidence where the target decoder exposes an admitted LDS RMW | Access evidence where admitted | Exact access evidence where admitted |

SuperCollider deliberately does not implement a happens-before model.
Record/Replay, Sampled, and Inline Shadow share the semantic inventory but have
different bounded evidence models. Engine differences in this table are
therefore intentional and are not target-parity gaps.

## Cross-target equivalent forms

| Semantic form | `gfx942` / CDNA3 | `gfx950` / CDNA4 | `gfx1201` / RDNA4 | `gfx1250` |
| --- | --- | --- | --- | --- |
| Native LDS single-range read/write | 8, 16, 32, 64, and 128-bit admitted forms | 8, 16, 32, 64, and 128-bit admitted forms | 16, 32, 64, and 128-bit admitted forms | 8, 16, 32, 64, and 128-bit admitted forms; 96-bit load extension (SuperCollider also admits the store readback) |
| Native LDS dual-range read/write | 32/64-bit adjacent and stride-64 forms | 32/64-bit adjacent and stride-64 forms | 32/64-bit adjacent and stride-64 forms | 32/64-bit adjacent and stride-64 forms |
| Native LDS transpose read | Target-native admitted transpose forms | Target-native admitted transpose forms | No equivalent claimed | Target-native admitted transpose forms |
| Group-FLAT read/write | 16, 32, 64, and 128-bit forms | 16, 32, 64, and 128-bit forms | 16, 32, 64, and 128-bit forms | 16, 32, 64, and 128-bit forms, including the encoded signed immediate |
| Full workgroup barrier | Singleton `s_barrier` | Singleton `s_barrier` | Qualified signal/wait sequence | Qualified signal/wait sequence |
| Cluster-scope barrier | No target form claimed | No target form claimed | No target form claimed | Qualified cluster signal/wait sequence |
| 32-bit ordered FLAT atomic | Cache-associated device/system release, acquire, and acquire-release | Cache-associated device/system release, acquire, and acquire-release | Device/system release, acquire, and acquire-release | Device/system release, acquire, and acquire-release |
| Ordered global/buffer atomic address forms | FLAT address semantics only | FLAT address semantics only | VGLOBAL and buffer-resource materialization | VGLOBAL and buffer-resource materialization |
| Ordered LDS atomic | Relaxed LDS RMW is access-only | Relaxed LDS RMW is access-only | No ordered LDS form claimed | Ordered LDS token form plus relaxed access-only forms |
| Addressed ordinary-memory fence | Cache-associated release/acquire sequence | Cache-associated release/acquire sequence | Global/buffer communication sequence | Global/buffer communication sequence |

The width lists name semantic byte ranges admitted by ConSan, not every alias
spelling in an ISA manual. For example, CDNA `ds_read_*`/`ds_write_*`, RDNA
`ds_load_*`/`ds_store_*`, and gfx1250 VDS spellings enter the same native-LDS
range model.

## Typed exclusions

These exclusions are part of the contract:

- ordinary global memory is not race-checked; global or buffer addresses are
  used only as synchronization identities for admitted ordering sequences;
- arbitrary FLAT accesses without group provenance are not treated as LDS;
- group-FLAT 8-bit and 96-bit ordinary accesses have no cross-target claim;
- `gfx1201` has no claimed native-LDS 8-bit or 96-bit ordinary form;
- native transpose and 96-bit forms are target extensions, not parity
  requirements;
- barrier lifecycle operations with dynamic participant state are inventoried
  but unsupported unless they form one of the qualified workgroup or cluster
  sequences above;
- wave-scope atomics are not cross-owner synchronization and are typed
  not-applicable;
- atomics with missing scope/order/address metadata, unsupported width, or an
  unavailable compare-exchange outcome are typed unsupported;
- standalone ordinary-memory fence replay is Record/Replay-only;
- SuperCollider atomics and barriers are fault-injection composition points,
  not causal race evidence; and
- async copies are outside the current access contract.

Target-specific absence is not reported as a lowering failure. Relevant sites
receive a stable semantic disposition (`not_applicable`, `supported`, or
`unsupported`) before resource planning, and supported sites retain an
independent lowering outcome.

## Evidence map

The host tests exercise the same production inventory and lowerers used by the
HSA hook:

- access decoding and normalization:
  `analysis_test.cpp`, `moi_record_replay_test.cpp`, and
  `supercollider_test.cpp`, including CDNA subword/transpose/dual-range,
  gfx1250 subword/transpose/96-bit, and group-FLAT D16 cases;
- all-engine access placement:
  `moi_engine_conformance_test.cpp`, `moi_sampled_test.cpp`, and
  `moi_inline_shadow_test.cpp`;
- barrier semantics:
  `moi_record_replay_test.cpp`, `moi_sampled_test.cpp`, and
  `moi_inline_shadow_test.cpp`, including CDNA singleton, RDNA split, and
  gfx1250 cluster cases;
- atomic and fence semantics:
  `moi_record_replay_model_test.cpp`, `moi_record_replay_test.cpp`,
  `moi_sampled_test.cpp`, and `moi_inline_atomic_test.cpp`; and
- the four-target Inline release transaction:
  `ConSanMoi.SupportedTargetsInlineAtomicReleaseCarriesClaimedPredecessor`.

The registered execution gates use native code objects for each target:

- physical `gfx1201`: `ConSanMoiHipTest.*` and `ConSanInlineShadowTest.*`;
- simulated `gfx1250`: `ConSanGfx1250Sim.*`;
- simulated `gfx950`: `ConSanGfx950Sim.*` and the opted-in
  `ConSanGfx950HipMoiSim.*` corpus; and
- simulated `gfx942`: `ConSanGfx942Sim.*` and the opted-in
  `ConSanGfx942HipMoiSim.*` corpus.

Simulator and offline LDS capacity comes from the selected RocJITsu JSON.
Runtime instrumentation always uses the LDS capacity supplied by the active
runtime agent.
