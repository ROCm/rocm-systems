<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# HIP Test Control Rationale and Incidents

This document records *why* the HIP test suite carries certain non-obvious test
controls — capability skips, error-state resets, teardown guards, coverage
gates, level semantics, and architecture guards. These controls look like dead
code, suppressed failures, or redundant ceremony to a reader who does not know
the behavior that motivated them, and that reader is the one most likely to
remove them. Each entry below explains the control so it survives review.

Scope and provenance. Every claim here is grounded in in-repository evidence:
source under `catch/`, the authoring guide
[`catch/contract/AUTHORING.md`](catch/contract/AUTHORING.md), the coverage
tracker [`CONTRACT_COVERAGE.md`](CONTRACT_COVERAGE.md), the strategy document
[`TESTING.md`](TESTING.md), the suite inventory
[`test_inventory.yaml`](test_inventory.yaml), and the in-repo CI workflow. Where
a control's rationale is a convention rather than a single landed change, the
originating commit is cited when `git log` locates it and is marked **historical
commit not located** otherwise, rather than guessed. No workspace-local test logs
or external trackers are treated as evidence here.

Incident IDs are stable (`TI-NNN`). The `history:` fields in
[`test_inventory.yaml`](test_inventory.yaml) reference these IDs.

Each incident below records the same labelled fields so a reviewer can see, at a
glance, what went wrong and whether the control may be removed:

- **Finding** — the failure or hazard that motivated the control.
- **Control** — the test-suite mechanism that addresses it.
- **Protects** — the signal or behavior the control keeps trustworthy.
- **Origin** — where the control was introduced (commit/PR when located).
- **Current status** — whether the control is active and how widely it is used.
- **Retirement condition** — the only condition under which it may be removed.

## TI-001: Capability-gated skips (accepted-or-unsupported and image gates)

**Finding.** Contract cases assert portable guarantees that must hold on *any
conforming backend*, but some guarantees are only observable where a runtime path
or hardware capability is present. On a host that legitimately lacks the
capability — for example a WSL2 iGPU without image support — a well-formed call
returning `hipErrorNotSupported` was being reported as a contract violation for
behavior the API never promised there.

**Control.** A well-formed contract call that returns `hipErrorNotSupported` is
treated as a *capability skip*, not a failure:
`if (status == hipErrorNotSupported) HIP_SKIP_TEST("...");`. Texture, array, and
image cases additionally open with `CHECK_IMAGE_SUPPORT;` so they skip cleanly on
hosts without image support (for example a WSL2 iGPU) and still run on
image-capable datacenter GPUs.

**Protects.** The trustworthiness of the contract signal across the heterogeneous
fleet the suite runs on: a capability skip records that a path was not exercised
here without turning an unsupported environment into a false failure.

**Rationale.** The contract tier asserts portable guarantees that must hold on
*any conforming backend*, but some guarantees are only observable where a runtime
path or hardware capability is present. Without the skip, an environment that
legitimately lacks the capability would report a contract violation for behavior
the API never promised there. Treating a documented `hipErrorNotSupported` (or a
missing image capability) as a skip keeps the signal trustworthy across the
heterogeneous fleet the suite runs on.

**Evidence.**
- Convention documented in [`catch/contract/AUTHORING.md`](catch/contract/AUTHORING.md)
  ("Treat `hipErrorNotSupported` as a capability skip" and the
  `CHECK_IMAGE_SUPPORT` gate).
- Live example of the unsupported-path probe-then-skip helpers in
  [`catch/contract/graph_mem_nodes/test_hip_graph_mem_nodes_contract.cc`](catch/contract/graph_mem_nodes/test_hip_graph_mem_nodes_contract.cc)
  (`TryTrimGraphMemory`, `TryAddMemAllocNode`, and `HIP_SKIP_TEST` calls).
- Capability-gated interop/IPC import paths are classified in
  [`test_inventory.yaml`](test_inventory.yaml) under `contract-interop`, whose
  prerequisites list an external producer or graphics environment.
- `CHECK_IMAGE_SUPPORT` is used across 11 contract source files.

**Origin.** The capability-skip convention and the `CHECK_IMAGE_SUPPORT`
gate landed with the contract suite in `0d90b30fb6c`
(`test(hip): add public-API semantic contract-test suite (#8827)`), refined by
`d8274c37f85` (`test(hip): address contract test follow-up feedback (#9702)`).

**Current status.** Active convention across the contract tier; the
`CHECK_IMAGE_SUPPORT` gate is used across 11 contract source files and the
accepted-or-unsupported skip is applied wherever a capability path may be absent.

**Retirement condition.** Only when every supported backend and host in the fleet
guarantees the capability in question, making the skip unreachable. Until then
the skip must stay; removing it reintroduces false failures on capability-poor
hosts.

**Inventory records.** `contract-interop` (`TI-001`).

## TI-002: Sticky-error reset after intentional rejection

**Finding.** HIP keeps a per-thread sticky error. A rejection left uncleared
poisoned the next test's `HIP_CHECK(hipGetLastError())`, which surfaced as a
failure in a *different* test when a Catch2 executable ran multiple cases in one
shared process, frequently only on the NVIDIA backend. Direct execution uses
declaration order by default; callers may request a different order explicitly.

**Control.** After any call a test *expects* to fail, the test clears the
thread-global sticky error with `(void)hipGetLastError();`.

**Protects.** Isolation between cases that share one Catch2 process: it keeps an
intentional rejection from leaking into a later, unrelated case and producing an
order-dependent, hard-to-localize failure.

**Rationale.** HIP keeps a per-thread sticky error. A rejection left uncleared
poisons the next test's `HIP_CHECK(hipGetLastError())`, which surfaces as a
failure in a *different* test when direct Catch2 execution runs multiple cases in
one shared process. Direct execution uses declaration order by default, unless
the caller passes an explicit order option, and the failure is frequently seen
only on the NVIDIA backend. The reset is deliberately placed at each
intentional-rejection site so the hazard never crosses a test boundary. A
maintainer who sees a lone `(void)hipGetLastError()` and removes it as a no-op
reintroduces an order-dependent, hard-to-localize cross-test failure.

**Evidence.**
- Convention documented in [`catch/contract/AUTHORING.md`](catch/contract/AUTHORING.md)
  ("Clear the sticky error after every intentional rejection"), including the
  diagnostic hint: if a test passes alone but fails in-suite, suspect this.
- The reset appears in 29 contract source files; e.g.
  [`catch/contract/vmm_handle/test_hip_vmm_handle_contract.cc`](catch/contract/vmm_handle/test_hip_vmm_handle_contract.cc)
  and
  [`catch/contract/mem_map_array/test_hip_mem_map_array_contract.cc`](catch/contract/mem_map_array/test_hip_mem_map_array_contract.cc).
- Recognized as a recurring hazard in [`test_inventory.yaml`](test_inventory.yaml)
  under `contract-error`.

**Origin.** The convention is part of the contract suite and its authoring
guide (`0d90b30fb6c`, #8827). A single prior change that first established the
sticky-error-reset rule separately from the suite introduction was **not located**
in `git log` (the `(void)hipGetLastError()` string predates the contract tier in
unrelated CLR code, so a `-S` search does not isolate the convention's origin).

**Current status.** Active; the reset appears in 29 contract source files and is
mandated for every intentional-rejection site by the authoring guide.

**Retirement condition.** Only if HIP stops keeping a per-thread sticky error, or
the harness runs each case in its own process so a leaked error cannot cross a
test boundary. Until then the reset must stay.

**Inventory records.** `contract-error` (`TI-002`).

## TI-003: ContractCleanup exception-safe teardown

**Finding.** Contract tests use raw HIP handles (allocations, streams, events,
graphs, modules, arrays, VMM mappings, pools) and a failing `REQUIRE`/`HIP_CHECK`
throws and unwinds the stack. Teardown written as plain statements after the
assertions was skipped on that path, leaking the resource. A Catch2 test
executable run directly executes multiple cases in one shared process, in
declaration order by default unless the caller passes an explicit order option,
so a leaked resource persists into and destabilizes later cases. In contrast,
CTest `catch_discover_tests` starts one process per registered case; the suite
must remain correct under both run modes.

**Control.** Contract tests release every acquired resource through a
scope-exit guard, `hip::contract::ContractCleanup`, declared *before* the
resources and fed `cleanup.Add(...)` immediately after each successful
allocation. Handles are captured **by value**, not by reference.

**Protects.** Resource safety under exception unwinding: every acquired handle is
released even when an assertion throws, so a failing case cannot leak state into
the cases that share a directly-run Catch2 test executable's process (where cases
are not CTest-isolated) and destabilize later cases.

**Rationale.** Contract tests use raw HIP handles (allocations, streams, events,
graphs, modules, arrays, VMM mappings, pools) and a failing `REQUIRE`/`HIP_CHECK`
throws and unwinds the stack. Teardown written as plain statements after the
assertions is skipped on that path, leaking the resource; when a Catch2 test
executable is run directly its cases share one process and run in declaration
order by default unless the caller passes an explicit order option, so the leak
persists into and destabilizes later cases. In contrast, CTest
`catch_discover_tests` runs each case in its own process. The guard runs the
registered actions in reverse order during unwinding, giving correct RAII without
a bespoke wrapper per handle type. The by-value capture requirement is not
stylistic: because the guard is declared before the resources, a `[&]` capture
would read a dangling local at scope exit.

**Evidence.**
- Implementation and the full rationale comment in
  [`catch/include/contract_cleanup.hh`](catch/include/contract_cleanup.hh).
- Mandated by [`catch/contract/AUTHORING.md`](catch/contract/AUTHORING.md)
  ("Clean up every resource with `ContractCleanup`"), with
  `mem_batch_copy_3d/test_hip_mem_batch_copy_3d_contract.cc` named as the
  exemplar.

**Origin.** `0d90b30fb6c`
(`test(hip): add public-API semantic contract-test suite (#8827)`).

**Current status.** Active suite-wide convention mandated by the authoring guide;
it underlies the resource-safety claims of the contract tier.

**Retirement condition.** Only if contract tests stop using raw HIP handles, or
the harness provides equivalent exception-safe RAII automatically for every
handle type. Until then the guard must stay.

**Inventory records.** The guard is a suite-wide convention rather than a single
inventory record; it underlies the resource-safety claims of the contract tier.

## TI-004: Contract coverage and test-plan drift checks

**Finding.** Without a gate, a new or changed public HIP API could land without
either portable contract coverage or a reviewed justification for its absence,
turning "we forgot to test this" into a silent omission; the uncovered-API
allowlist could accumulate stale excuses; and `catch/TEST_PLAN.md` could drift out
of sync with the `@asserts:` tags in the contract sources.

**Control.** A single path-filtered workflow runs one `contract-coverage` job that
governs the contract tier through two static checks. They are **not** always-on:
the workflow's `paths:` filter restricts the job to pull requests that touch the
HIP headers, the contract tier, or the contract tooling, so a docs-only PR does not
trigger it. Both checks are workflow-defined and path-filtered; whether either
blocks merge depends on branch-protection settings external to this repository and
is not asserted here.
1. A coverage drift checker,
   [`catch/contract/tools/check_contract_coverage.py`](catch/contract/tools/check_contract_coverage.py),
   fails a PR when a public HIP API declared in `hip_runtime_api.h` has neither a
   contract test nor an entry in
   [`catch/contract/uncovered_apis.txt`](catch/contract/uncovered_apis.txt) with
   a reason. It also fails on *stale* allowlist entries so the allowlist stays
   truthful, and it verifies the numeric snapshot block in
   [`CONTRACT_COVERAGE.md`](CONTRACT_COVERAGE.md).
2. A test-plan staleness check (`catch/tools/gen_test_plan.py --check`) fails when
   `catch/TEST_PLAN.md` is out of date with the `// @asserts:` tags in the
   contract sources.

**Protects.** The contract tier as a governed public-API interface: it keeps new
public APIs from landing untested and keeps the generated plan and allowlist
honest.

**Rationale.** The check exists so a new or changed public API cannot land without
either portable contract coverage or an explicit, reviewed justification for its
absence. It converts "we forgot to test this" from a silent omission into a
self-documenting CI failure on the relevant paths, and keeps the
allowlist from accumulating stale excuses. The snapshot (2026-11-05) records 97.4%
approximate declared API-name coverage with 13 intentionally uncovered,
allowlisted APIs.

**Evidence.**
- Checker, allowlist, and philosophy in
  [`CONTRACT_COVERAGE.md`](CONTRACT_COVERAGE.md) and
  [`catch/contract/AUTHORING.md`](catch/contract/AUTHORING.md).
- Defined in-repo by
  [`../../.github/workflows/hip-contract-coverage.yml`](../../.github/workflows/hip-contract-coverage.yml),
  whose single `contract-coverage` job runs both checks and whose `paths:` filter
  scopes them to HIP-header, contract-tier, and contract-tooling changes.
- Classified in [`test_inventory.yaml`](test_inventory.yaml) as
  `static-contract-coverage` (trusted, path-filtered, no GPU required) and
  `static-test-plan`.

**Origin.** The checker, allowlist, workflow, and authoring guide landed
with the contract suite in `0d90b30fb6c` (#8827); follow-up refinement in
`d8274c37f85` (#9702). A later dispatch-table fix for external-semaphore graph
node APIs touched the allowlist in `666d066f964`
(`fix(clr): set dispatch-table entries for external-semaphore graph node APIs (#15085)`).

**Current status.** Active in-repo as two static checks in the single
`contract-coverage` job of
[`../../.github/workflows/hip-contract-coverage.yml`](../../.github/workflows/hip-contract-coverage.yml),
path-filtered; whether either blocks merge is governed by branch-protection
settings external to this repository and is not asserted here. Classified in the
inventory as `static-contract-coverage` and `static-test-plan`.

**Retirement condition.** Only if public-API contract coverage and test-plan
freshness are guaranteed by another mechanism. Until then both checks are
retained.

**Inventory records.** `static-contract-coverage` (`TI-004`).

## TI-005: Test-level semantics (parameterization vs selection)

**Finding.** The two level knobs are independent and easy to conflate. Filtering
with a label alone (`ctest -L quick`) selects the quick cases but still runs them
at the default `level_2` parameters — so a run intended to be a fast sanity check
silently executes release-validation sizes, and a run intended to be exhaustive
silently under-tests.

**Control.** `HIP_TEST_LEVEL=level_N` sets workload *parameters* (sizes,
iteration counts; default `level_2`) and never filters. A Catch2 tag such as
`[level_0]` or `ctest -L level_0` *selects* cases and never changes parameters.
To both select and scale a run, set both, e.g.
`HIP_TEST_LEVEL=level_0 ctest -L level_0`. Named categories (`quick`, `standard`,
`comprehensive`, `full`) map to level sets, and parameters come from the highest
selected level.

**Protects.** Correctly scoped CI lanes and honest conclusions: documenting the
split prevents mis-scoped runs and misleading "it passed quickly" results.

**Rationale.** The two knobs are independent and easy to conflate. Filtering with
a label alone (`ctest -L quick`) selects the quick cases but still runs them at
the default `level_2` parameters — so a run intended to be a fast sanity check
silently executes release-validation sizes, and a run intended to be exhaustive
silently under-tests. Documenting the split prevents mis-scoped CI lanes and
misleading "it passed quickly" conclusions.

**Evidence.**
- Behavior specified in [`TESTING.md`](TESTING.md) ("Development workflow":
  `HIP_TEST_LEVEL` only sets parameters; a label filter only selects).
- Category-to-level mapping and the same caveat (a ctest label filter never
  reaches the binary, so it leaves parameters at `level_2`) in
  [`catch/config/configs/test_categories.yaml`](catch/config/configs/test_categories.yaml).
- Label-generation model (`parse_config.py` emits one `[level_N]` tag per level
  and one category tag; `catch_discover_tests(ADD_TAGS_AS_LABELS)` turns tags into
  ctest labels) documented in [`test_inventory.yaml`](test_inventory.yaml)
  (`ctest_label_model`).

**Origin.** The level/category parameterization model is documented with
the testing strategy in `0d90b30fb6c` (#8827); the category definitions predate it
in `catch/config/configs/test_categories.yaml`. A single commit that first
introduced the parameterize-vs-select distinction itself was **not located** in
`git log`.

**Current status.** Documented behavior, reflected in
[`catch/config/configs/test_categories.yaml`](catch/config/configs/test_categories.yaml)
and the `ctest_label_model` block of [`test_inventory.yaml`](test_inventory.yaml);
no single suite record owns it.

**Retirement condition.** Only if the harness unifies selection and
parameterization into a single knob so the two can no longer be set
inconsistently. Until then the distinction must stay documented.

**Inventory records.** Documented structurally in the `ctest_label_model` and
`ctest_label` fields of [`test_inventory.yaml`](test_inventory.yaml); no single
suite record owns it.

## TI-006: Architecture and runtime guards

**Finding.** Backend-divergent or CUDA-removed APIs failed to build on the backend
where the path does not exist; `gl_interop` pulled in `GL/gl.h`, which headless
CUDA nodes lack (defect reference `EXSWHTEC-246`); and the graph memory-free
contract case keeps `hipGraphAddMemFreeNode` coverage structural-only rather than
instantiating or launching its graph. Repository evidence establishes that
structural guard, but does not locate or verify the originating crash rationale.
An exact account of an uncatchable local SIGSEGV on launch is therefore unverified
rationale, not a repository-evidenced fact (see Origin).

**Control.** Backend-divergent or CUDA-removed APIs are guarded at compile time
with `#if HT_AMD` (1 on AMD, 0 on NVIDIA), compiling to an empty binary on the
backend where the path does not exist rather than failing to build. Build-level
gates exclude paths an environment cannot support — `gl_interop` is built only
under `HIP_PLATFORM STREQUAL "amd"`. The graph memory-free case is kept as a
structural-only assertion: the node is constructed and its parameters are
round-tripped through `hipGraphMemFreeNodeGetParams`, but the graph is never
instantiated or launched. This preserves the repository-evidenced structural
coverage without asserting an unverified local-SIGSEGV rationale as fact.

**Protects.** Portability of the contract build and run across backends and
headless hosts: it keeps CUDA nodes green without weakening AMD coverage, while
retaining the graph memory-free node's observable construction and parameter
round-trip coverage without treating an unverified crash account as established
repository evidence.

**Rationale.** A contract test must be *portable*, which sometimes means "compile
to nothing" on a backend that removed the API, and sometimes means "do not even
build" a path whose dependencies are absent on the target host. `gl_interop` is
AMD-gated because the NVIDIA header path pulls in `GL/gl.h`, which headless CUDA
nodes lack (defect reference `EXSWHTEC-246`); gating at build time keeps those
nodes green without weakening AMD coverage. For the graph memory-free node, the
repository-evidenced structural assertion — construct the node and round-trip its
parameters without instantiating or launching the graph — preserves an observable
portable invariant. The exact local-SIGSEGV account that may have motivated this
scope is not located or verified in repository evidence, so no claim is made that
node construction is safe, that launch is the hazard, or that a SIGSEGV would
propagate to sibling cases. `#if HT_AMD` is used across 44 contract source files.

**Evidence.**
- `#if HT_AMD` and `BACKEND-DIFF:` conventions documented in
  [`catch/contract/AUTHORING.md`](catch/contract/AUTHORING.md).
- `gl_interop` AMD build gate and the `EXSWHTEC-246` defect note in
  [`catch/unit/CMakeLists.txt`](catch/unit/CMakeLists.txt).
- Graph memory-free node structural-only guard (node constructed and parameters
  round-tripped via `hipGraphMemFreeNodeGetParams`, graph never launched) in
  [`catch/contract/graph_mem_nodes/test_hip_graph_mem_nodes_contract.cc`](catch/contract/graph_mem_nodes/test_hip_graph_mem_nodes_contract.cc).
- Classified in [`test_inventory.yaml`](test_inventory.yaml) under
  `contract-graph` (repository-evidenced structural-only graph memory-free-node
  coverage; originating crash rationale unverified) and `unit-interop-graphics`
  (gl_interop AMD-gated).

**Origin.** The `HT_AMD` guarding convention and the structural-only
graph-mem-node guard landed with the contract suite in `0d90b30fb6c` (#8827); that
structural-only guard is the repository evidence for this control. The exact local
SIGSEGV account proposed as the reason for keeping the memory-free node structural
is **not located or verified** as a landed change or other in-repository record,
and remains unverified rationale rather than a repository-evidenced event. The
`gl_interop` AMD build gate lives in `catch/unit/CMakeLists.txt`, whose
testing-framework scaffold originates in `abab64b35a1`; the specific change that
disabled `gl_interop` on NVIDIA under `EXSWHTEC-246` was **not located** as a
distinct commit in `git log`.

**Current status.** Active; `#if HT_AMD` is used across 44 contract source files,
the `gl_interop` build gate is in place, and the graph-mem-node structural guard
remains.

**Retirement condition.** Evaluated per guard: a guard may be removed only when
the guarded API exists on all supported backends, the absent header dependency is
gone, or the graph memory-free case has a verified rationale and broader execution
coverage that supersedes the current structural-only assertion.

**Inventory records.** `contract-graph` (`TI-006`), `unit-interop-graphics`
(`TI-006`).
