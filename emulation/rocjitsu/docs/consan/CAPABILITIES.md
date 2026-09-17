# ConSan semantic capability matrix

This document is the normative supported-form contract for ConSan on
`gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250`. It describes
equivalent memory and synchronization semantics, not identical ISA mnemonic sets. The
target-specific status ledgers record workload qualification separately.

The following compact projection is generated from the typed contract in
`consan_capability_contract.h`. The host-side
`ConSan.CapabilityManifestMatchesDocumentation` gate rejects drift in target,
mode, or semantic-form availability without parsing the surrounding prose or
copying target-native mnemonic lists. "Associated only" means the form is not
standalone evidence for that mode; it contributes ordering metadata only to
an admitted atomic sequence. Forms without a parenthesized qualifier are
supported under the mode semantics described below. "Unsupported" records
a mode-specific lowering gap for a form admitted by the target family.

<!-- BEGIN GENERATED CONSAN CAPABILITY CONTRACT -->
| Target | Mode | Access | Barrier | Atomic | Fence |
| --- | --- | --- | --- | --- | --- |
| `gfx942` | SuperCollider | native LDS<br>group FLAT | workgroup (mutation only) | ordered FLAT (mutation only)<br>ordered VGLOBAL (mutation only) | addressed ordinary (mutation only) |
| `gfx942` | ConSan | native LDS<br>group FLAT | workgroup | ordered FLAT<br>ordered VGLOBAL<br>relaxed LDS RMW (access only) | addressed ordinary (associated only) |
| `gfx950` | SuperCollider | native LDS<br>group FLAT | workgroup (mutation only) | ordered FLAT (mutation only)<br>ordered VGLOBAL (mutation only) | addressed ordinary (mutation only) |
| `gfx950` | ConSan | native LDS<br>group FLAT | workgroup | ordered FLAT<br>ordered VGLOBAL<br>relaxed LDS RMW (access only) | addressed ordinary (associated only) |
| `gfx1100` | SuperCollider | native LDS<br>group FLAT | workgroup (mutation only) | ordered FLAT (mutation only)<br>ordered VGLOBAL (mutation only) | addressed ordinary (mutation only) |
| `gfx1100` | ConSan | native LDS<br>group FLAT | workgroup | ordered FLAT<br>ordered VGLOBAL | addressed ordinary (associated only) |
| `gfx1201` | SuperCollider | native LDS<br>group FLAT | workgroup (mutation only) | ordered FLAT (mutation only)<br>ordered VGLOBAL (mutation only) | addressed ordinary (mutation only) |
| `gfx1201` | ConSan | native LDS<br>group FLAT | workgroup | ordered FLAT<br>ordered VGLOBAL | addressed ordinary (associated only) |
| `gfx1250` | SuperCollider | native LDS<br>group FLAT | workgroup (mutation only)<br>cluster (mutation only) | ordered FLAT (mutation only)<br>ordered VGLOBAL (mutation only)<br>ordered LDS (mutation only) | addressed ordinary (mutation only) |
| `gfx1250` | ConSan | native LDS<br>group FLAT | workgroup<br>cluster | ordered FLAT<br>ordered VGLOBAL<br>ordered LDS<br>relaxed LDS RMW (access only) | addressed ordinary (associated only) |
<!-- END GENERATED CONSAN CAPABILITY CONTRACT -->

A form marked **supported** is decoded into the shared semantic inventory and
has a lowering path for the named mode. Register pressure, report capacity,
placement, or bounded runtime-state exhaustion can still make a particular
site dynamically incomplete. Those outcomes are reported explicitly; they do
not turn an unsupported or incomplete run into a clean result.

## Mode contract

| Semantic form | SuperCollider | ConSan |
| --- | --- | --- |
| Ordinary native LDS read/write | Redundant observation and mismatch marker | Selected causal window |
| Proven or likely-group FLAT read/write | Same, after target-specific LDS-offset normalization | Same as native LDS |
| Workgroup barrier | Mutation may compose with an access probe; no ordering claim | Qualified barrier metadata in the selected causal bank |
| Cluster barrier | Mutation only; no ordering claim | Supported on `gfx1250` |
| Ordered atomic RMW | Mutation may compose with an access probe; no ordering claim | Qualified ordering metadata in the selected causal bank |
| Ordered compare-exchange | Mutation only; no ordering claim | Supported when the dynamic outcome is available |
| Ordinary communication plus cache/fence sequence | Mutation only; no ordering claim | Qualified addressed communication sequence associated with a selected access window |
| Relaxed LDS RMW used as an LDS access | Redundant observation is not claimed for atomics | Access evidence where admitted |

SuperCollider deliberately does not implement a happens-before model.
ConSan analyzes bounded causal evidence. Both modes share the semantic
inventory, but their detection guarantees differ; this is not a target-parity gap.

## Cross-target equivalent forms

| Semantic form | `gfx942` / CDNA3 | `gfx950` / CDNA4 | `gfx1100` / RDNA3 | `gfx1201` / RDNA4 | `gfx1250` / CDNA5 |
| --- | --- | --- | --- | --- | --- |
| Native LDS single-range read/write | 8, 16, 32, 64, 96, and 128-bit admitted forms | 8, 16, 32, 64, 96, and 128-bit admitted forms | 8, 16, 32, 64, 96, and 128-bit admitted forms | 16, 32, 64, 96, and 128-bit admitted forms | 8, 16, 32, 64, and 128-bit admitted forms; 96-bit load extension (SuperCollider also admits the store readback) |
| Native LDS dual-range read/write | 32/64-bit adjacent and stride-64 forms | 32/64-bit adjacent and stride-64 forms | 32/64-bit adjacent and stride-64 forms | 32/64-bit adjacent and stride-64 forms | 32/64-bit adjacent and stride-64 forms |
| Native LDS transpose read | Target-native admitted transpose forms | Target-native admitted transpose forms | No equivalent claimed | No equivalent claimed | Target-native admitted transpose forms |
| Group-FLAT read/write | 16, 32, 64, and 128-bit forms | 16, 32, 64, and 128-bit forms | 16, 32, 64, and 128-bit forms | 16, 32, 64, and 128-bit forms | 16, 32, 64, and 128-bit forms, including the encoded signed immediate |
| Full workgroup barrier | Singleton `s_barrier` | Singleton `s_barrier` | Singleton `s_barrier` | Qualified signal/wait sequence | Qualified signal/wait sequence |
| Cluster-scope barrier | No target form claimed | No target form claimed | No target form claimed | No target form claimed | Qualified cluster signal/wait sequence |
| 32-bit ordered FLAT atomic | Cache-associated device/system release, acquire, and acquire-release | Cache-associated device/system release, acquire, and acquire-release | Compiler-qualified device-scoped acquire; release is not inferred | Device/system release, acquire, and acquire-release | Device/system release, acquire, and acquire-release |
| Ordered VGLOBAL atomic address forms | VGLOBAL vector-only and scalar-base materialization | VGLOBAL vector-only and scalar-base materialization | VGLOBAL vector-only and scalar-base materialization | VGLOBAL vector-only and scalar-base materialization | VGLOBAL vector-only and scalar-base materialization |
| Ordered LDS atomic | Relaxed LDS RMW is access-only | Relaxed LDS RMW is access-only | No ordered LDS form claimed | No ordered LDS form claimed | Ordered LDS token form plus relaxed access-only forms |
| Addressed ordinary-memory fence | Cache-associated release/acquire sequence | Cache-associated release/acquire sequence | Compiler-qualified acquire cache sequence; no release claim | Global/buffer communication sequence | Global/buffer communication sequence |

The width lists name semantic byte ranges admitted by ConSan, not every alias
spelling in an ISA manual. For example, CDNA `ds_read_*`/`ds_write_*`, RDNA
`ds_load_*`/`ds_store_*`, and gfx1250 VDS spellings enter the same native-LDS
range model.

## ConSan owner and lane coverage

The default owner is derived from entry `workitem_id_x`; it does not distinguish
waves separated only in a workgroup's y/z dimensions. Access-only checks can use
`RJ_CONSAN_OWNER_SOURCE=hw_id`. Sparse resident-wave IDs may require more
retention banks; the multidimensional device checks explicitly request 1,024.
Hardware-ID owners do not support ConSan's barrier/atomic ordering path.

Uniform-address instructions can retain exact lane masks on supported targets.
This is not a general per-lane trace: the CDNA second-address-group payload can
miss a same-instruction collision when the representative address is elsewhere.
The second-address-group diagnostic test is qualified on gfx1201; its success
must not be interpreted as a cross-target guarantee.

## Typed exclusions

These exclusions are part of the contract:

- ordinary global memory is not race-checked; global or buffer addresses are
  used only as synchronization identities for admitted ordering sequences;
- arbitrary FLAT accesses without group provenance are not treated as LDS;
- group-FLAT 8-bit and 96-bit ordinary accesses have no cross-target claim;
- `gfx1201` has no claimed native-LDS 8-bit ordinary form, and `gfx1100` has no
  claimed native transpose-read form;
- native transpose and 96-bit forms are target extensions, not parity
  requirements;
- barrier lifecycle operations with dynamic participant state are inventoried
  but unsupported unless they form one of the qualified workgroup or cluster
  sequences above;
- wave-scope atomics are not cross-owner synchronization and are typed
  not-applicable;
- atomics with missing scope/order/address metadata, unsupported width, or an
  unavailable compare-exchange outcome are typed unsupported;
- ordinary-memory fences contribute associated ordering metadata, not a
  standalone event trace;
- SuperCollider atomics and barriers are fault-injection composition points,
  not causal race evidence; and
- async copies are outside the current access contract.

Target-specific absence is not reported as a lowering failure. Relevant sites
receive a stable semantic disposition (`not_applicable`, `supported`, or
`unsupported`) before resource planning, and supported sites retain an
independent lowering outcome.

## Evidence map

Host tests exercise the production inventory, policy, and lowerers used by the
HSA hook:

- access decoding and normalization: `access_classifier_test.cpp`,
  `analysis_test.cpp`, `probe_lowering_test.cpp`, and `supercollider_test.cpp`;
- target contracts and disposition tables: `capability_contract_test.cpp`;
- barrier sequencing and completion: `barrier_policy_test.cpp` and
  `probe_lowering_test.cpp`;
- atomic and addressed communication policy: `atomic_classifier_test.cpp`,
  `atomic_fence_policy_test.cpp`, and `probe_lowering_test.cpp`;
- register preservation and private frames: `spill_manager_test.cpp`,
  `common_test.cpp`, and `lowering_plan_test.cpp`;
- evidence sizing, lifecycle, and analysis: `evidence_requirements_test.cpp`,
  `consan_report_plan_test.cpp`, and `hsa_hooks_unit_test.cpp`.

[VALIDATION.md](validation/VALIDATION.md) describes the native and simulator
execution procedures. The target-specific validation ledgers record which
workloads have actually been qualified; a host test does not establish physical
GPU execution coverage.

Simulator and offline LDS capacity comes from the selected RocJITsu JSON.
Runtime instrumentation uses the capacity supplied by the active runtime agent.
