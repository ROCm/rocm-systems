<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# HIP Tests Restructuring Roadmap (Proposal)

This document is a **proposal**. It records what the test inventory shows today,
the rules a future restructuring must follow, and the workstreams that would move
the suite toward a clearer tier model. It does **not** describe current policy and
it does **not** retire, move, or weaken any test. Current, enforced behavior lives
in [`TESTING.md`](TESTING.md); the facts this roadmap reasons over live in
[`test_inventory.yaml`](test_inventory.yaml) and
[`test_coverage_model.yaml`](test_coverage_model.yaml); the rationale for existing
guards lives in [`TESTING_INCIDENTS.md`](TESTING_INCIDENTS.md).

**Separation of concerns.**

| Question | Authoritative source |
|---|---|
| What is validated and gated today? | [`TESTING.md`](TESTING.md) |
| What suites exist and where are they placed? | [`test_inventory.yaml`](test_inventory.yaml) |
| What coverage is intended, and where are the gaps? | [`test_coverage_model.yaml`](test_coverage_model.yaml) |
| Why does a non-obvious guard or skip exist? | [`TESTING_INCIDENTS.md`](TESTING_INCIDENTS.md) |
| What should change, under what criteria? | This document (proposal) |

No statement in this roadmap should be read as an enforced gate. Where an entry
proposes a move, it names the inventory record and the condition that must be met
first; nothing here authorizes editing source trees.

## 1. Inventory findings

These findings summarize the current catalog of 45 inventory records. Each is
grounded in [`test_inventory.yaml`](test_inventory.yaml) and the source evidence
those records cite; none asserts work that has already happened.

### 1.1 Contract coverage is separately governed

The contract tier (16 `contract-*` records such as
[`contract-memory`, `contract-graph`, `contract-interop`](test_inventory.yaml)) is
governed by a single path-filtered workflow rather than by device
execution alone. Its one `contract-coverage` job runs two static checks — a
coverage drift checker and a generated test-plan staleness check — on every PR that
touches the HIP headers, the contract tier, or the contract tooling; the workflow's
`paths:` filter scopes the job, so the checks are not always-on and a docs-only PR
does not trigger them, and they require each public HIP API to have either a
contract test or an allowlisted rationale. The checks are workflow-defined; whether
either blocks merge is governed by branch-protection settings external to this
repository.

- Records: `static-contract-coverage`, `static-test-plan` (both `kind: contract`,
  `gating: trusted`, no GPU required).
- Evidence: [`catch/contract/tools/check_contract_coverage.py`](catch/contract/tools/check_contract_coverage.py),
  [`catch/contract/uncovered_apis.txt`](catch/contract/uncovered_apis.txt),
  [`catch/tools/gen_test_plan.py`](catch/tools/gen_test_plan.py),
  [`CONTRACT_COVERAGE.md`](CONTRACT_COVERAGE.md), and the in-repo workflow
  [`../../.github/workflows/hip-contract-coverage.yml`](../../.github/workflows/hip-contract-coverage.yml).
- Rationale preserved in [`TESTING_INCIDENTS.md`](TESTING_INCIDENTS.md) `TI-004`.

Implication for restructuring: contract records are the portable API baseline and
should be treated as a governed interface, not merged into functional sweeps. Any
move touching a `contract-*` record must keep the drift and test-plan checks green.

### 1.2 Unit/functional coverage is broad and historically distributed

The functional tier under [`catch/unit/`](catch/unit/CMakeLists.txt) is the largest
part of the catalog (the `unit-*` records) and spans memory, stream/event, graph,
kernel launch, device/context, occupancy, texture/surface, RTC, module/loader, VMM,
error handling, cooperative/multi-GPU, interop/graphics, compiler, and
multithread-sync domains. Many of these records carry `disposition: review` because
a contract record pins the portable invariant for the same domain while the
functional record still owns the deeper, sweep-sized behavior.

- Example pairs: `contract-memory` (`keep`) alongside `unit-memory` (`review`);
  `contract-graph` (`keep`) alongside `unit-graph` (`review`).
- Coverage model reflects this as `gap_status: covered_dual_tier` for domains such
  as `memory.allocation_and_integrity` and `graph.lifecycle_and_error_paths` in
  [`test_coverage_model.yaml`](test_coverage_model.yaml).

Implication: "contract exists" does not mean "functional is redundant." A `review`
disposition is an explicit request for owner classification, not a scheduled
deletion.

### 1.3 Integration responsibilities cross several directories

There is no separate top-level integration build option today. Integration-style
behavior is distributed across interop, multi-GPU, multiprocess, RTC, loader, and
graph areas.

- Records: `integration-multiprocess` (`review`), `unit-cooperative-multigpu`
  (`review`), `unit-interop-graphics` (`review`), plus `contract-multi-device-launch`
  (`review`).
- Coverage model: `multi_gpu.peer_and_cooperative_launch` carries
  `gap_status: distributed_classification_open`; `interop.import_and_ipc` carries
  `gap_status: environment_gated`.
- Matches the current-state description in [`TESTING.md`](TESTING.md) ("Integration
  testing") and the first known-gap row there.

### 1.4 Performance and stress are separate build modes, off by default

- Records: `performance-api`, `performance-scenarios` (built only with
  `BUILD_PERF_TESTS`), `stress-suite` (built only with `BUILD_STRESS_TESTS`).
- Evidence: [`catch/performance/CMakeLists.txt`](catch/performance/CMakeLists.txt),
  [`catch/stress/CMakeLists.txt`](catch/stress/CMakeLists.txt).
- These run on demand or on a dedicated cadence, so restructuring them does not
  affect the per-PR gate set and must not be conflated with contract/functional
  reorganization.

### 1.5 Restructured domains coexist with legacy sources

Several domains already have a contract-tier record and a legacy functional record
active at the same time. The coexistence is deliberate and is the main reason this
is a roadmap rather than a completed migration. The `review` dispositions in
[`test_inventory.yaml`](test_inventory.yaml) carry `related_restructuring:
TEST_RESTRUCTURING.md`, pointing back here for the classification decision.

## 2. Migration and retirement decision rules

These rules bind every proposal in this document. They exist so a reorganization
cannot quietly drop coverage or history (see [`TESTING_INCIDENTS.md`](TESTING_INCIDENTS.md)
for the kind of guard that looks removable but is not).

1. **Full migration requires oracle and prerequisite preservation.** A migration may
   be marked *full* only when the replacement cases preserve the original record's
   `oracle` and its meaningful `prerequisites` (capability gates, platform gates,
   device-count requirements). A replacement that checks a weaker oracle, or that
   drops a capability/skip guard, is not a full migration.
2. **Partial migrations keep legacy cases active.** If replacement coverage is
   narrower than the legacy record, the legacy source stays active and the record
   stays `disposition: review` (or `migrate` with an explicit partial note). Legacy
   cases are not removed on the strength of a same-named contract test.
3. **Retirement is a reviewed proposal, never an implicit side effect.** A `retire`
   record is invalid unless it names: replacement inventory IDs; evidence that the
   replacement is equivalent or stronger coverage; CI evidence that the replacement
   actually runs on the relevant cadence; and recorded owner approval. Absent any of
   these, the correct disposition is `review`.
4. **Unresolved classification is `review`, not a move.** When it is unclear which
   tier a case belongs to, the record stays `review` and the open question is
   recorded (see Section 4). Ambiguity is never resolved by silently relocating
   source.
5. **Governed interfaces stay green.** Any change touching a `contract-*` record
   must keep the coverage drift and test-plan staleness checks passing
   (`static-contract-coverage`, `static-test-plan`; `TI-004`).

## 3. Workstreams

Five workstreams structure the proposed work. Each names concrete outputs and
measurable acceptance criteria. None is claimed complete; acceptance criteria are
the bar a future change must clear, not a status report.

### WS-1: Inventory and evidence hardening

- **Outputs:** Complete, deduplicated records in
  [`test_inventory.yaml`](test_inventory.yaml); every record's `evidence` paths
  resolvable; every `history` reference pointing at a real `TI-NNN` account in
  [`TESTING_INCIDENTS.md`](TESTING_INCIDENTS.md).
- **Acceptance criteria:**
  - Every inventory `id` is unique and every `disposition` is within the declared
    vocabulary.
  - Every `evidence.path` resolves to a file in the repository, and every
    `source.paths` entry resolves to an existing file or directory (as appropriate,
    since a suite's source is often a directory), each unless explicitly marked
    `external:`.
  - Every `history` / `incidents` reference resolves to a defined incident ID.
  - The deterministic documentation checker passes with zero reference errors.

### WS-2: Domain consolidation (contract vs functional boundary)

- **Outputs:** For each dual-tier domain, a recorded decision that either keeps both
  tiers with distinct oracles or proposes a concrete `migrate`/`merge_into` with a
  target ID; applied first to the `review` records in memory, stream/event, graph,
  kernel launch, device/context, occupancy, texture/surface, module/loader, VMM, and
  error handling.
- **Acceptance criteria:**
  - Each consolidated domain has exactly one record per tier with a non-overlapping,
    stated oracle, or a documented reason both tiers must remain.
  - No `merge_into` names a missing target ID.
  - No consolidation weakens a contract check (Rule 5) or drops a capability guard
    (Rule 1).

### WS-3: Coverage-gap closure

- **Outputs:** Tests or explicit accepted-gap entries for the classes flagged in
  [`test_coverage_model.yaml`](test_coverage_model.yaml) as `no_contract_tier`
  (`kernel_launch.device_side_correctness`, `rtc.runtime_compilation`) and other
  non-`covered_dual_tier` classes.
- **Acceptance criteria:**
  - Each prioritized coverage class has either a new `current_inventory` record or a
    recorded, owner-reviewed decision that the gap is accepted.
  - Pairwise checks listed in the coverage model that are marked priority are either
    exercised by a named record or explicitly deferred with a reason.
  - No class is reported as closed without a concrete inventory ID backing it.

### WS-4: CI and cadence alignment

- **Outputs:** A mapping from inventory `cadence`/`gating` to the actual CI lanes,
  distinguishing in-repo gates from `external:` lanes; wiring proposals for
  `HIP_TEST_LEVEL` cadence selection.
- **Acceptance criteria:**
  - Every record's claimed `cadence` and `gating` is backed by an in-repo workflow
    reference or marked `external:` with a pointer to where that lane is defined.
  - The contract drift and test-plan lanes remain `trusted` and path-filtered
    checks in the single `contract-coverage` workflow job (the workflow `paths:`
    filter restricts the job to PRs touching the HIP headers, the contract tier, or
    the contract tooling; docs-only changes do not trigger them); any
    merge-required status is governed by branch protection external to this
    repository.
  - The `HIP_TEST_LEVEL` parameterize-vs-select distinction (`TI-005`) is preserved
    in any cadence wiring; a label filter alone must not silently change workload
    parameters.

### WS-5: Retirement and cleanup

- **Outputs:** For any proposed `retire`, a complete retirement dossier meeting
  Rule 3; removal of stale allowlist entries only where the coverage checker confirms
  they are stale.
- **Acceptance criteria:**
  - No source is deleted without a `retire` record naming replacement IDs, equivalent
    or stronger coverage evidence, CI evidence, and recorded owner approval.
  - Allowlist edits keep [`catch/contract/uncovered_apis.txt`](catch/contract/uncovered_apis.txt)
    truthful; the coverage checker passes afterward.
  - Every retirement preserves the relevant incident account in
    [`TESTING_INCIDENTS.md`](TESTING_INCIDENTS.md) rather than erasing the history.

## 4. Open item: multi-GPU / multi-device classification

This is recorded as an **open classification question**, not a decision. No source
is moved by this documentation change.

Cooperative-group and multi-device launch cases are, by execution context,
integration / multi-GPU work, but their source currently lives in unit directories.
The inventory already flags this and points back here:

- [`unit-cooperative-multigpu`](test_inventory.yaml) (`disposition: review`):
  cooperative and peer launch cases live in unit directories but are
  integration/multi-GPU by execution context.
- [`contract-multi-device-launch`](test_inventory.yaml) (`disposition: review`):
  the contract case is portable and stays active, but its multi-GPU
  execution-context classification needs explicit metadata.
- [`integration-multiprocess`](test_inventory.yaml) (`disposition: review`):
  multiprocess cases are integration by nature but live outside a dedicated tier.
- Coverage model: `multi_gpu.peer_and_cooperative_launch` carries
  `gap_status: distributed_classification_open` in
  [`test_coverage_model.yaml`](test_coverage_model.yaml).

**Proposed resolution (pending owner review):** add explicit integration / multi-GPU
`execution_context` metadata to these records so the coverage model can audit
multi-device coverage independently, **even though their source remains in a unit
directory**. Relocating the source files is explicitly out of scope for this
documentation change and would require its own reviewed migration under the Section 2
rules. Until an owner decides, these records stay `review`.

## 5. How this roadmap is maintained

- Update a record's `disposition` here and in [`test_inventory.yaml`](test_inventory.yaml)
  in the same change, with evidence, when a classification is decided.
- Keep this document limited to proposals and criteria; move any newly enforced
  policy into [`TESTING.md`](TESTING.md) and any newly understood guard history into
  [`TESTING_INCIDENTS.md`](TESTING_INCIDENTS.md).
- Validate links and whitespace (`git diff --check`) and run the documentation
  checker before merging any change to this file.
