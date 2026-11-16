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

## TI-001: Capability-gated skips (accepted-or-unsupported and image gates)

**Control.** A well-formed contract call that returns `hipErrorNotSupported` is
treated as a *capability skip*, not a failure:
`if (status == hipErrorNotSupported) HIP_SKIP_TEST("...");`. Texture, array, and
image cases additionally open with `CHECK_IMAGE_SUPPORT;` so they skip cleanly on
hosts without image support (for example a WSL2 iGPU) and still run on
image-capable datacenter GPUs.

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

**Introduced in.** The capability-skip convention and the `CHECK_IMAGE_SUPPORT`
gate landed with the contract suite in `0d90b30fb6c`
(`test(hip): add public-API semantic contract-test suite (#8827)`), refined by
`d8274c37f85` (`test(hip): address contract test follow-up feedback (#9702)`).

**Inventory records.** `contract-interop` (`TI-001`).

## TI-002: Sticky-error reset after intentional rejection

**Control.** After any call a test *expects* to fail, the test clears the
thread-global sticky error with `(void)hipGetLastError();`.

**Rationale.** HIP keeps a per-thread sticky error. A rejection left uncleared
poisons the next test's `HIP_CHECK(hipGetLastError())`, which surfaces as a
failure in a *different* test — one that only appears in full-suite, randomized
order (Catch2 shares one process) and frequently only on the NVIDIA backend. The
reset is deliberately placed at each intentional-rejection site so the hazard
never crosses a test boundary. A maintainer who sees a lone `(void)hipGetLastError()`
and removes it as a no-op reintroduces an order-dependent, hard-to-localize
cross-test failure.

**Evidence.**
- Convention documented in [`catch/contract/AUTHORING.md`](catch/contract/AUTHORING.md)
  ("Clear the sticky error after every intentional rejection"), including the
  diagnostic hint: if a test passes alone but fails in-suite, suspect this.
- The reset appears in 38 contract source files; e.g.
  [`catch/contract/vmm_handle/test_hip_vmm_handle_contract.cc`](catch/contract/vmm_handle/test_hip_vmm_handle_contract.cc)
  and
  [`catch/contract/mem_map_array/test_hip_mem_map_array_contract.cc`](catch/contract/mem_map_array/test_hip_mem_map_array_contract.cc).
- Recognized as a recurring hazard in [`test_inventory.yaml`](test_inventory.yaml)
  under `contract-error`.

**Introduced in.** The convention is part of the contract suite and its authoring
guide (`0d90b30fb6c`, #8827). A single prior change that first established the
sticky-error-reset rule separately from the suite introduction was **not located**
in `git log` (the `(void)hipGetLastError()` string predates the contract tier in
unrelated CLR code, so a `-S` search does not isolate the convention's origin).

**Inventory records.** `contract-error` (`TI-002`).

## TI-003: ContractCleanup exception-safe teardown

**Control.** Contract tests release every acquired resource through a
scope-exit guard, `hip::contract::ContractCleanup`, declared *before* the
resources and fed `cleanup.Add(...)` immediately after each successful
allocation. Handles are captured **by value**, not by reference.

**Rationale.** Contract tests use raw HIP handles (allocations, streams, events,
graphs, modules, arrays, VMM mappings, pools) and a failing `REQUIRE`/`HIP_CHECK`
throws and unwinds the stack. Teardown written as plain statements after the
assertions is skipped on that path, leaking the resource into Catch2's shared,
randomized-order process and destabilizing later tests. The guard runs the
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

**Introduced in.** `0d90b30fb6c`
(`test(hip): add public-API semantic contract-test suite (#8827)`).

**Inventory records.** The guard is a suite-wide convention rather than a single
inventory record; it underlies the resource-safety claims of the contract tier.

## TI-004: Contract coverage and test-plan drift gates

**Control.** Two static, always-on PR gates govern the contract tier:
1. A coverage drift checker,
   [`catch/contract/tools/check_contract_coverage.py`](catch/contract/tools/check_contract_coverage.py),
   fails a PR when a public HIP API declared in `hip_runtime_api.h` has neither a
   contract test nor an entry in
   [`catch/contract/uncovered_apis.txt`](catch/contract/uncovered_apis.txt) with
   a reason. It also fails on *stale* allowlist entries so the allowlist stays
   truthful, and it verifies the numeric snapshot block in
   [`CONTRACT_COVERAGE.md`](CONTRACT_COVERAGE.md).
2. A test-plan staleness gate (`catch/tools/gen_test_plan.py --check`) fails when
   `catch/TEST_PLAN.md` is out of date with the `// @asserts:` tags in the
   contract sources.

**Rationale.** The gate exists so a new or changed public API cannot land without
either portable contract coverage or an explicit, reviewed justification for its
absence. It converts "we forgot to test this" from a silent omission into a
blocking, self-documenting CI failure, and keeps the allowlist from accumulating
stale excuses. The snapshot (2026-11-05) records 97.4% approximate declared
API-name coverage with 13 intentionally uncovered, allowlisted APIs.

**Evidence.**
- Checker, allowlist, and philosophy in
  [`CONTRACT_COVERAGE.md`](CONTRACT_COVERAGE.md) and
  [`catch/contract/AUTHORING.md`](catch/contract/AUTHORING.md).
- Enforced in-repo by
  [`../../.github/workflows/hip-contract-coverage.yml`](../../.github/workflows/hip-contract-coverage.yml).
- Classified in [`test_inventory.yaml`](test_inventory.yaml) as
  `static-contract-coverage` (trusted, always-on, no GPU required) and
  `static-test-plan`.

**Introduced in.** The checker, allowlist, workflow, and authoring guide landed
with the contract suite in `0d90b30fb6c` (#8827); follow-up refinement in
`d8274c37f85` (#9702). A later dispatch-table fix for external-semaphore graph
node APIs touched the allowlist in `666d066f964`
(`fix(clr): set dispatch-table entries for external-semaphore graph node APIs (#15085)`).

**Inventory records.** `static-contract-coverage` (`TI-004`).

## TI-005: Test-level semantics (parameterization vs selection)

**Control.** `HIP_TEST_LEVEL=level_N` sets workload *parameters* (sizes,
iteration counts; default `level_2`) and never filters. A Catch2 tag such as
`[level_0]` or `ctest -L level_0` *selects* cases and never changes parameters.
To both select and scale a run, set both, e.g.
`HIP_TEST_LEVEL=level_0 ctest -L level_0`. Named categories (`quick`, `standard`,
`comprehensive`, `full`) map to level sets, and parameters come from the highest
selected level.

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

**Introduced in.** The level/category parameterization model is documented with
the testing strategy in `0d90b30fb6c` (#8827); the category definitions predate it
in `catch/config/configs/test_categories.yaml`. A single commit that first
introduced the parameterize-vs-select distinction itself was **not located** in
`git log`.

**Inventory records.** Documented structurally in the `ctest_label_model` and
`ctest_label` fields of [`test_inventory.yaml`](test_inventory.yaml); no single
suite record owns it.

## TI-006: Architecture and runtime guards

**Control.** Backend-divergent or CUDA-removed APIs are guarded at compile time
with `#if HT_AMD` (1 on AMD, 0 on NVIDIA), compiling to an empty binary on the
backend where the path does not exist rather than failing to build. Build-level
gates exclude paths an environment cannot support — `gl_interop` is built only
under `HIP_PLATFORM STREQUAL "amd"`. Runtime paths with a known local crash are
reduced to structural assertions guarded by an unsupported-capability check
instead of being removed.

**Rationale.** A contract test must be *portable*, which sometimes means "compile
to nothing" on a backend that removed the API, and sometimes means "do not even
build" a path whose dependencies are absent on the target host. `gl_interop` is
AMD-gated because the NVIDIA header path pulls in `GL/gl.h`, which headless CUDA
nodes lack (defect reference `EXSWHTEC-246`); gating at build time keeps those
nodes green without weakening AMD coverage. For graph memory-free node paths that
have exhibited local segfaults, keeping a structural, capability-skipped
assertion preserves the portable invariant check while avoiding a crash that would
take down the shared Catch2 process. `#if HT_AMD` is used across 44 contract
source files.

**Evidence.**
- `#if HT_AMD` and `BACKEND-DIFF:` conventions documented in
  [`catch/contract/AUTHORING.md`](catch/contract/AUTHORING.md).
- `gl_interop` AMD build gate and the `EXSWHTEC-246` defect note in
  [`catch/unit/CMakeLists.txt`](catch/unit/CMakeLists.txt).
- Graph memory-free node structural/skip guard in
  [`catch/contract/graph_mem_nodes/test_hip_graph_mem_nodes_contract.cc`](catch/contract/graph_mem_nodes/test_hip_graph_mem_nodes_contract.cc).
- Classified in [`test_inventory.yaml`](test_inventory.yaml) under
  `contract-graph` (graph node paths guarded against known local segfaults) and
  `unit-interop-graphics` (gl_interop AMD-gated).

**Introduced in.** The `HT_AMD` guarding convention and the graph-mem-node guard
landed with the contract suite in `0d90b30fb6c` (#8827). The `gl_interop` AMD
build gate lives in `catch/unit/CMakeLists.txt`, whose testing-framework scaffold
originates in `abab64b35a1`; the specific change that disabled `gl_interop` on
NVIDIA under `EXSWHTEC-246` was **not located** as a distinct commit in `git log`.

**Inventory records.** `contract-graph` (`TI-006`), `unit-interop-graphics`
(`TI-006`).
