<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# HIP Runtime Testing Strategy

This document describes how the HIP runtime test suite is validated today, which signals gate changes, and where the remaining gaps are. It follows the ROCm-wide testing strategy template and describes the current state rather than an aspirational target. A documented gap is one that can be discussed, prioritized, and closed.

## Component overview

hip-tests validates the public HIP API contract and observable runtime behavior for the HIP runtime, HIPRTC, compiler-facing paths used by runtime tests, and loader/module behavior. It is built against a ROCm/HIP installation and runs primarily on real AMD GPU hardware because the most important runtime behavior is only observable through a device, driver, and runtime stack.

This document does not define testing strategy for ROCm math libraries, communication libraries, profiler internals, or application frameworks. Those components own their own testing strategy documents.

**Key architectural constraint that shapes testing:** HIP runtime behavior is mostly hardware- and driver-observable, so the highest-confidence tests are device-executed contract, functional, integration, and stress tests rather than CPU-only unit tests.

## Companion documents

`TESTING.md` is the current-state strategy: it describes the behavior that is validated today, the signals that gate changes, and the gaps that remain. It is deliberately concise. The detailed, evidence-backed facts live in companion documents so this guide does not drift into a generated case list:

- [`test_inventory.yaml`](test_inventory.yaml) records **what exists** — the meaningful test groups, their source paths, build options, cadence, gating, and disposition, each tied to repository evidence.
- [`test_coverage_model.yaml`](test_coverage_model.yaml) records the **intended coverage dimensions** — risk-weighted coverage classes across API surface, behavior, execution context, environment, and cadence, each mapped to the inventory records that cover it.
- [`TESTING_INCIDENTS.md`](TESTING_INCIDENTS.md) preserves the **historical rationale** for non-obvious guards, skips, quarantines, and validation controls, so a control is not removed before the failure it prevents is understood.
- [`TEST_RESTRUCTURING.md`](TEST_RESTRUCTURING.md) holds **proposals** — evidence-backed restructuring findings, migration criteria, and workstreams. It describes what is proposed, not what is enforced today.
- The generated [`catch/TEST_PLAN.md`](catch/TEST_PLAN.md) is the derived per-case assertion inventory.

### Evidence boundaries

A few boundaries keep these documents honest and prevent over-claiming:

- `catch/TEST_PLAN.md` is **generated** from the `@asserts:` annotations by `catch/tools/gen_test_plan.py`. It is not a hand-maintained duplicate inventory: cite it as evidence, never edit it by hand, and do not re-list its cases in this guide or in `test_inventory.yaml`.
- CI lanes that run outside this repository — notably the TheRock build and HIP runtime test jobs — are **external** to hip-tests. The inventory marks them `external:` and links them to the evidence that does live here. A single static, build-free workflow runs in-repo and is **path-filtered** rather than always-on: `.github/workflows/hip-contract-coverage.yml`, whose one `contract-coverage` job runs two static checks — a contract-coverage check and a test-plan check (scoped by the workflow's `paths:` filter to pull requests touching the HIP headers, the contract tier, or the contract tooling). These checks are workflow-defined and path-filtered; whether either is merge-required is governed by branch-protection settings that are external to this repository and not asserted here. A change outside the workflow's filtered paths does not trigger it; in particular a docs-only change does not trigger the contract-coverage workflow. The testing-documentation set additionally has a dedicated consistency checker, `tools/check_testing_docs.py` with unit tests under `tools/tests/`, that validates these documents as a connected set (schema, vocabularies, disposition/retirement targets, repository-path pointers, incident references, and Markdown links). It is a **recommended local validation** to run before pushing changes to this documentation set; it is **not currently enforced by a CI gate** in this repository. Wiring it into CI is a planned integration, and this document will be updated to describe it as an enforced gate only once such a workflow exists.
- An unsupported-capability skip (`HIP_SKIP_TEST`, YAML `unsupported:`) is **signal-preserving behavior, not passing coverage**. A cleanly skipped case records that a path was not exercised here; it does not count as a validated contract.

## Development workflow

All device and runtime tests — the contract, unit/functional, and performance tiers that exercise the HIP runtime on a GPU — are C++ built on Catch2 and discovered and run through CTest. (The in-repo documentation/governance tooling, notably `tools/check_testing_docs.py`, is a standalone Python check with its own `unittest` suite under `tools/tests/`; it is build-free and is not part of the Catch2/CTest harness.) Test executables are grouped by CMake `BUILD_*` options and by CTest labels generated from Catch2/YAML tags. Workload size is parameterized by level. `HIP_TEST_LEVEL=level_N` only sets the parameters (default `level_2`) and never filters; a Catch2 filter such as `[level_0]` or `ctest -L level_0` only selects tests and never changes the parameters. Set both to combine them, e.g. `HIP_TEST_LEVEL=level_0 ctest -L level_0`.

| You changed | Run this before pushing | Needs GPU |
|---|---|---|
| Public HIP API behavior | Contract tests for the touched API plus the related functional test label | Yes |
| Functional behavior in a feature area | The matching `catch/unit/` feature-area CTest label or executable | Yes |
| YAML tags, levels, or config generation | Config validation and a focused CTest label using the generated tags | Usually yes |
| Build or test harness logic | Configure, build, and run a small representative label such as `contract` or the touched feature label | Usually yes |
| Performance-sensitive runtime path | Relevant functional tests plus the matching performance scenario when available | Yes |
| Testing-documentation set (these docs, `test_inventory.yaml`, `test_coverage_model.yaml`, `tools/`) | `python tools/check_testing_docs.py` plus its unit tests `python -m unittest tools/tests/test_check_testing_docs.py` — run these locally as a recommended validation; not currently enforced by a CI gate | No |
| Documentation-only change | Markdown/link checks and review against the current test framework | No |

## Testing strategy and layers

The suite lives under `projects/hip-tests/catch/`. Each layer is gated by a CMake `BUILD_*` option and surfaced through CTest labels derived from Catch2/YAML tags via `catch_discover_tests(... ADD_TAGS_AS_LABELS ...)`.

| Layer | Location | Build status / cadence | Primary signal |
|---|---|---|---|
| Contract | `catch/contract/` | ON through `BUILD_CONTRACT_TESTS` | Small public-API invariants and coverage drift |
| Unit / functional | `catch/unit/` plus related focused suites | ON through `BUILD_UNIT_TESTS` | Per-API and per-feature behavior |
| Integration | Currently distributed across unit, interop, multi-GPU, multiprocess, RTC, loader, and graph areas | No separate top-level build option today; runs as a PR or on-demand subset depending on platform and capability | Runtime interaction with the ROCm stack and real devices |
| Performance | `catch/performance/` | OFF by default through `BUILD_PERF_TESTS` | Hot-path throughput and latency measurements |
| Stress | `catch/stress/` | OFF by default through `BUILD_STRESS_TESTS` | Long-running and high-load behavior |

### Contract testing

Contract tests live in `catch/contract/` and are built ON by default through `BUILD_CONTRACT_TESTS`. Each case pins the small, portable, device-only semantic guarantee of a public HIP API: successful round-trips, accepted-or-unsupported capability probes, invalid-input rejection, and sticky-error behavior. Cases carry an `@asserts: API - invariant` annotation directly above the case. `catch/tools/gen_test_plan.py` compiles these annotations into `catch/TEST_PLAN.md`, an inventory of what each case asserts. Every declared public API is either covered by a contract test or listed with a rationale in `catch/contract/uncovered_apis.txt`. The repository-root workflow `.github/workflows/hip-contract-coverage.yml` is path-filtered: its single `contract-coverage` job runs two static checks — the coverage checker and the generated test-plan staleness check — on pull requests that touch the HIP headers, the contract tier, or the contract tooling, so new public HIP APIs do not land without a contract test or an explicit allowlist entry. The checks are workflow-defined and path-filtered, not always-on — a docs-only change does not trigger them; whether either is merge-required depends on branch-protection settings external to this repository.

### Unit and functional testing

Unit and functional tests live under `catch/unit/` feature-area directories and are built ON by default through `BUILD_UNIT_TESTS`. They validate functional correctness of individual HIP APIs and features across many feature areas, including memory, stream, graph, module, texture, cooperative groups, peer-to-peer, RTC, and virtual memory: arguments accepted or rejected, values returned, and observable side effects such as data movement and synchronization. Tests use the in-tree `HIP_CHECK` and `HIP_ASSERT` helpers and are discovered with `catch_discover_tests`, so under CTest each registered case runs in its own process; running a Catch2 test executable directly instead shares a single process across the cases it holds, which is the shared-process hazard that case-level isolation such as the sticky-error reset guards against (see [TI-002](TESTING_INCIDENTS.md#ti-002-sticky-error-reset-after-intentional-rejection)). Workload size is parameterized by level. `HIP_TEST_LEVEL=level_N` sets the parameters and a `[level_N]` Catch2 filter or `ctest -L level_N` selects the tests, independently of each other; `catch/config/configs/definitions.yaml` holds the generated per-level parameters and `catch/README.md` is the canonical reference for level semantics.

### Integration testing

Integration coverage is currently distributed across the unit, interop, multi-GPU, multiprocess, RTC, loader, and graph areas rather than a separate top-level tier. It exercises the runtime, compiler-facing paths, HIPRTC, and loader together against a full ROCm/HIP install on real hardware: cross-component interop validated against graphics and external producers, multi-GPU and peer paths that skip cleanly on single-GPU hosts, and an architecture matrix spanning supported CDNA and RDNA families on Linux and Windows. Some groups and cases are platform- or capability-gated, so the matrix describes where the suite runs rather than guaranteeing every case runs on every platform.

### Performance and benchmarking

Performance tests live in `catch/performance/` and are OFF by default through `BUILD_PERF_TESTS`. They measure throughput and latency of hot-path APIs and scenarios such as memcpy, memset, kernel launch, stream, event, and mempool paths on the same Catch2/CTest harness, reusing the per-level workload parameters in `definitions.yaml`. Because they are not built by default, they run on demand or on a dedicated cadence rather than gating every PR. Automated universal per-PR performance regression gating is not currently enforced by this component document; regression baselining is handled by the performance-tracking process that consumes these tests.

### Stress and system-style validation

Stress tests live in `catch/stress/` and are OFF by default through `BUILD_STRESS_TESTS`. They cover long-running and high-load soak behavior and are run on demand or on a periodic/scheduled cadence rather than per PR. End-to-end, full-stack system-style scenarios are today expressed through the integration and stress areas rather than a distinct top-level system tier.

## Pre-submit and CI gates

### Validation gates and ownership

| Signal | Classification | Blocks merge? | Notes |
|---|---|---|---|
| TheRock build and HIP runtime test jobs | Trusted gate | Yes, when required by the PR policy | Builds ROCm packages and runs the configured HIP test set. |
| Contract API coverage drift check (a static check in the single `contract-coverage` job of `hip-contract-coverage.yml`) | Trusted, path-filtered workflow check | Workflow-defined and path-filtered; whether it is merge-required depends on branch-protection settings external to this repository (unknown here) | Path-filtered (workflow `paths:` filter), not always-on; a docs-only PR does not trigger it. Requires new public APIs to have a contract test or an explicit allowlist rationale. |
| Generated test-plan staleness check (the second static check in the same `contract-coverage` job) | Trusted, path-filtered workflow check | Workflow-defined and path-filtered; branch-protection/merge-required status external to this repository (unknown here) | Path-filtered, not always-on. Keeps `catch/TEST_PLAN.md` synchronized with `@asserts:` annotations. |
| Formatting and PR-description checks | Trusted gate | Yes, where configured | Maintains source and review hygiene. |
| Performance and stress sweeps | Informational or scheduled gate, depending on workflow | Not universally on every PR | Used for longer-running validation and regression investigation. |
| Flaky or quarantined cases | Unstable-flaky | No until fixed | Must have an owner, issue, and removal plan before becoming trusted signal. |

The contract API coverage drift check and the generated test-plan staleness check are not two independent workflows: they are two static checks run by the single `contract-coverage` job in `.github/workflows/hip-contract-coverage.yml`, under one shared `paths:` filter. Listing them on separate rows reflects the two distinct signals, not two separately configured required gates.

The testing-documentation set (these Markdown docs, the inventory/coverage YAML, and `tools/`) has a dedicated consistency checker, `tools/check_testing_docs.py` with unit tests under `tools/tests/`, but it is **not** in the table above because it is not an enforced CI gate today: it is a recommended local validation to run before pushing changes to this documentation set, and wiring it into CI is a planned integration rather than a current merge-blocking check. It is listed here, and in the development-workflow table above, so contributors can run it without implying an enforcement that does not yet exist.

### PR test classification

When you change public HIP API behavior, run the contract tests for the touched API together with the related functional test label. When you change functional behavior in a feature area, run the matching `catch/unit/` feature-area CTest label or executable. Config, tag, or level changes should be validated with config validation plus a focused label using the generated tags. A code-changing PR is expected to include an accompanying test file, and a change to a public API needs at least one test covering the new or changed behavior, including at least one invalid-input path where the API defines one.

### Flaky, disabled, and known-bug policy

A flaky or quarantined case is treated as unstable signal: it does not block merge, but it must have an owner, a tracking issue, and a removal or fix plan before it can be promoted back to trusted signal. Use `HIP_SKIP_TEST` for a genuine capability gap such as an unsupported device or runtime path, too few GPUs, or no image support, so the case skips cleanly instead of failing. Never leave a test that crashes the process. Any test that intentionally triggers a HIP error must consume the sticky thread-local error with `(void)hipGetLastError()` before returning so it does not leak into later cases in the same process.

Disabling a test case should be rare and explicit. It is appropriate when a test cannot execute on a platform or architecture because a required capability, OS facility, graphics environment, or runtime mode is absent; when a known product or infrastructure defect has a tracking issue; or when keeping the test enabled would crash the process and hide later signal. Prefer runtime capability checks plus `HIP_SKIP_TEST` for environment-dependent behavior, and use the YAML skip fields only when a case should be hidden for a specific platform, architecture, sanitizer mode, or tracked known issue. The `disabled:` and `unsupported:` fields share the same syntax and produce the same skip; they differ only in intent: `disabled:` is a temporary skip for a regression or tracked bug that is expected to be re-enabled, while `unsupported:` is a permanent skip where the platform, architecture, or backend cannot support the case. Do not disable a test only to make CI green. Every skipped case must record why it is skipped, and where it is tracked when the entry points at a bug, in the `reason:` field of the mapping form (`{targets: [...], reason: "..."}`). This is enforced in CI by `check_config.py`: any skip section with a non-empty `targets` list must carry a non-empty `reason`, and a `disabled:` reason must also cite a real Jira reference from a known project key (AIRUNTIME, ROCM, SWDEV, EXSWHTEC, AIRADSW), e.g. `AIRUNTIME-2892`; an `unsupported:` reason needs no Jira reference. To disable a case on every build, add `all` to its `disabled:` targets (targets are additive, so existing entries can stay); `amd` and `nvidia` cover every OS of that platform in either field. 'all' is not permitted to be used with unsupported, just remove the test; `check_config.py` rejects it.

## Coverage

Contract coverage is tracked statically: every declared public HIP API is either covered by a contract test or listed with a rationale in `catch/contract/uncovered_apis.txt`. The inventory of what each case asserts is generated into `catch/TEST_PLAN.md` by `catch/tools/gen_test_plan.py` from the `@asserts:` annotations, and the repository-root `.github/workflows/hip-contract-coverage.yml` runs both the coverage drift check and the test-plan staleness check as two static checks in its single `contract-coverage` job, path-filtered (scoped by its `paths:` filter to HIP-header, contract-tier, and contract-tooling changes rather than running on every PR); these checks are workflow-defined, and any merge-required status is governed by branch-protection settings external to this repository. Functional coverage is organized per feature area under `catch/unit/`, with the expectation that every public HIP API a change touches has at least one functional test exercising the changed behavior. Device-side code coverage is not fully captured by host coverage tools.

Code coverage and test coverage are different signals. Code coverage measures which host-side lines or branches executed. Test coverage measures which public API contracts, configurations, capabilities, and regression scenarios were intentionally validated. HIP runtime confidence depends heavily on test coverage across real hardware and runtime configurations, not only on line coverage.

### Inventory-informed coverage view

The tables below summarize the inventory and coverage model at a glance. Each row points at an inventory record id in [`test_inventory.yaml`](test_inventory.yaml) or a coverage class id in [`test_coverage_model.yaml`](test_coverage_model.yaml) rather than restating source-file lists; consult those files for the authoritative paths, build options, and evidence. They are a reading aid, not a replacement for the companion documents.

**Current test kinds**

| Kind | Representative inventory records | Build option | Default cadence / gating |
|---|---|---|---|
| Contract | `contract-memory`, `contract-stream-event`, `contract-graph`, `contract-device`, `contract-runtime-services` (and the rest of the contract tier) | `BUILD_CONTRACT_TESTS` (ON) | PR / trusted |
| Unit / functional | `unit-memory`, `unit-graph`, `unit-kernel-launch`, `unit-device-libraries`, `unit-rtc` (and the rest of the legacy unit tier) | `BUILD_UNIT_TESTS` (ON) | PR / trusted |
| Integration (distributed) | `integration-multiprocess`, `unit-cooperative-multigpu`, `unit-interop-graphics` | Built via `BUILD_UNIT_TESTS`; no separate top-level tier today | PR or on-demand, platform/capability-gated |
| Performance | `performance-api`, `performance-scenarios` | `BUILD_PERF_TESTS` (OFF) | On-demand / informational |
| Stress | `stress-suite` | `BUILD_STRESS_TESTS` (OFF) | On-demand / informational |
| Static governance (workflow) | `static-contract-coverage`, `static-test-plan` | n/a (two Python checks in one workflow job, no GPU/build) | PR (path-filtered) / trusted, in-repo |
| Static governance (build-time) | `static-config-check`, `static-sources-check` | `ENABLE_YAML_TAGS` (ON); `check_hip_tests_config` / `check_hip_tests_sources` custom targets | Runs during any build; config check gates (fails the build), sources check warns by default (fails only under `THEROCK_REQUIRE_HIP_YAML_ENTRIES`) |

**Major behavioral domains**

| Coverage class | Tiers present | Inventory records | Status |
|---|---|---|---|
| `memory.allocation_and_integrity` | contract + functional | `contract-memory`, `unit-memory`, `unit-oob` | covered_dual_tier |
| `stream_event.ordering_and_synchronization` | contract + functional | `contract-stream-event`, `unit-stream-event`, `unit-multithread-sync` | covered_dual_tier |
| `graph.lifecycle_and_error_paths` | contract + functional | `contract-graph`, `unit-graph` | covered_dual_tier |
| `kernel_launch.configuration_and_execution` | contract + functional | `contract-kernel-launch`, `unit-kernel-launch` | covered_dual_tier |
| `runtime.error_reporting` | contract + functional | `contract-error`, `unit-error-handling` | covered_dual_tier |
| `rtc.runtime_compilation` | functional only | `unit-rtc` | no_contract_tier |
| `multi_gpu.peer_and_cooperative_launch` | contract + functional | `contract-peer`, `contract-multi-device-launch`, `unit-cooperative-multigpu` | distributed_classification_open |

**Actual cadence and gating**

| Cadence / gating | Inventory records | Notes |
|---|---|---|
| PR, trusted | Contract tier, legacy unit/functional tier, `integration-multiprocess`, `static-contract-coverage`, `static-test-plan`, `static-config-check` | Device suites run through external TheRock jobs; the `static-contract-coverage`/`static-test-plan` checks run in-repo as two checks in the single path-filtered `contract-coverage` workflow job (scoped to HIP-header, contract-tier, and contract-tooling changes, not docs-only PRs); `static-config-check` is instead a build-time hard gate — the `check_hip_tests_config` `ENABLE_YAML_TAGS` custom target that fails any build with a malformed YAML config — not a workflow job; merge-required status for all of these is governed by branch protection external to this repository |
| Build-time, informational | `static-sources-check` | The `check_hip_tests_sources` `ENABLE_YAML_TAGS` custom target runs on every build and only warns when a source Catch2 case has no YAML config entry; it escalates to a hard build failure solely when `THEROCK_REQUIRE_HIP_YAML_ENTRIES` is set, and is not a per-PR workflow gate |
| On-demand, informational | `performance-api`, `performance-scenarios`, `stress-suite`, `unit-interop-graphics` | Opt-in build modes or environment-gated lanes, not per-PR gates |
| Capability- or environment-gated | `contract-memory-pool-vmm`, `unit-vmm`, `contract-peer`, `unit-cooperative-multigpu`, `unit-interop-graphics` | Skip cleanly when the driver capability, device topology, or graphics environment is absent |

**Coverage gaps from the model**

| Coverage class | `gap_status` | What it means |
|---|---|---|
| `kernel_launch.device_side_correctness`, `rtc.runtime_compilation` | no_contract_tier | Device-side primitive and HIPRTC paths have functional coverage only; no portable contract invariant pins them yet |
| `runtime.services_and_entry_points` | contract_only | Covered by a contract invariant with no functional counterpart today |
| `memory.pools_and_virtual_memory` | capability_gated | Asserted only where the driver capability exists; unsupported outcomes are accepted, not failed |
| `interop.import_and_ipc` | environment_gated | Runs only where a graphics/producer environment is configured. Contract coverage (graphics-interop, external-resource, and IPC invariants in `contract-interop`) spans the AMD and NVIDIA backends per the inventory; only the AMD-only functional graphics portion (`unit-interop-graphics` Vulkan/OpenGL) is AMD-gated |
| `multi_gpu.peer_and_cooperative_launch` | distributed_classification_open | Cases live in unit directories but are integration/multi-GPU by execution context; classification is an open review item |

These statuses mirror the `gap_status` fields in [`test_coverage_model.yaml`](test_coverage_model.yaml); a gap here is a coverage shape to review, not a failing test. The narrative [Known gaps summary](#known-gaps-summary) below covers the current-state process gaps.

## Nightly validation

Nightly and multi-arch runs extend the per-PR build-and-test flow across a broader architecture set and longer-running suites. If `HIP_TEST_LEVEL` is unset, the workload parameters fall back to the default level (`level_2`). `HIP_TEST_LEVEL=level_N` only sets the workload parameters and runs every requested test; to restrict a run to one level's tests, add `ctest -L level_N` as well. An invalid `HIP_TEST_LEVEL` value fails the run rather than falling back to the default. On-demand runs such as multi-arch sweeps, WSL runtime checks, and performance or stress runs are invoked explicitly rather than on every PR. CI level cadence is not yet fully wired to `HIP_TEST_LEVEL`.

## Supported configurations

| Configuration | Validation level | Notes |
|---|---|---|
| AMD GPU, Linux | Primary | Main device-executed runtime validation path. |
| AMD GPU, Windows | Supported where CI/hardware is configured | Some tests are platform-gated when APIs or process semantics differ. |
| AMD GPU, WSL | Supported subset | Tests that depend on Linux process or graphics behavior may be gated. |
| Multi-GPU and peer paths | Capability-gated | Tests skip when hardware capability is not present. |
| Graphics and external interop | Environment-gated | Requires graphics/Vulkan/OpenGL-capable hosts and drivers. |
| SPIR-V build path | Specialized | Enabled with `-DENABLE_SPIRV=ON` where supported. |

This table describes where the HIP runtime tests are validated. It is not a product support matrix and does not imply that every individual case runs on every listed configuration.

## ASAN, TSAN, and sanitizer coverage

AddressSanitizer builds are selected with `ENABLE_SANITIZER`, which takes `ASAN` for host and device instrumentation or `HOST_ASAN` for host only. Under TheRock it is set from `THEROCK_SANITIZER`. Sanitizer builds can catch host-side memory errors in the test harness and runtime-facing code paths that execute on the host. Device-executed paths limit what host sanitizers observe, so sanitizer coverage of device-side behavior is inherently partial.

ThreadSanitizer and other sanitizer coverage are not part of the default per-PR gate for this component today. When sanitizer coverage is used, document the build option, architecture, tier, and limitations in the workflow that enables it rather than treating the sanitizer result as a complete substitute for device-executed testing.

## Choosing the right test type

| Scenario | Test type to add |
|---|---|
| Pinning a small, portable, device-only guarantee of a public API, such as invalid-input rejection, round-trip behavior, or accepted-or-unsupported outcome | Contract test in `catch/contract/` with an `@asserts:` annotation |
| Verifying in-depth functional behavior of an API or feature | Functional test under the relevant `catch/unit/` feature-area directory |
| Exercising interaction between components, such as interop, multi-GPU, loader plus RTC plus runtime, or graph behavior | Integration-style test in the relevant feature area, gated on required capability or environment |
| Measuring throughput or latency of a hot path | Performance scenario under `catch/performance/`, built only when `BUILD_PERF_TESTS=ON` |
| Validating behavior under sustained load | Stress test under `catch/stress/`, built only when `BUILD_STRESS_TESTS=ON` |

Use `isQuickLevel()` inside the test function to trim workload size at `level_0` rather than skipping code paths. Use deterministic seeds for randomized inputs, and clean up every resource the test allocates.

## Known gaps summary

| Gap | Regression risk | Impact | Mitigation today |
|---|---|---|---|
| Integration tests are still distributed across directories rather than a separate top-level tier | Medium | Cross-component coverage is harder to audit independently from single-API unit coverage | Existing interop, multi-GPU, multiprocess, RTC, loader, and graph tests continue to run in their current locations |
| CI level cadence is not fully wired to `HIP_TEST_LEVEL` | Medium | PR and nightly runs do not consistently select different workload levels | `HIP_TEST_LEVEL` sets the workload parameters and per-case generated `[level_N]` tags / `ctest -L` select tests; CI still needs to set them per cadence |
| Level-based test parameters are defined but not broadly utilized today | Medium | Levels can select or label tests, but many cases do not yet scale workload sizes, iteration counts, or generated parameter sets based on the active level | Prefer adding level-aware parameter use when touching long-running tests; use `isQuickLevel()` and generated level parameters where practical |
| No universal per-PR performance baseline gate | Medium | Throughput or latency regressions may be found after merge or by manual review | Performance tests exist and can run on demand or on a scheduled cadence |
| Device-side code coverage is not fully captured by host coverage tools | Medium | A high host coverage number can still miss device-only behavior | Contract, functional, integration, and stress tests execute behavior on real devices |
| Flaky and quarantined tracking should keep converging toward owner plus issue plus expiry | Medium | Suppressed failures can age into permanent blind spots | Treat flaky/quarantined cases as unstable signal until each has a tracking issue and removal plan |
| Sanitizer coverage of device-side paths is inherently partial and is not part of the default per-PR gate | Low | Memory or race issues can escape sanitizer lanes | Use sanitizers as an additional signal, not a replacement for device-executed tests |

### Restructuring and migration references

Restructuring is tracked as evidence, not as an assumed move. The inventory and coverage model record the current state; [`TEST_RESTRUCTURING.md`](TEST_RESTRUCTURING.md) records proposals. Concretely, the evidence shows:

- **Legacy and restructured coverage coexist.** The contract tier pins portable public-API invariants while the legacy `catch/unit/` functional groups remain active for deeper, sweep-sized behavior. Records such as `unit-memory`, `unit-graph`, and `unit-stream-event` in [`test_inventory.yaml`](test_inventory.yaml) carry `disposition: review`, not `retire`; both tiers run today.
- **Integration coverage is distributed.** There is no separate top-level integration build option. `integration-multiprocess`, `unit-cooperative-multigpu`, and `unit-interop-graphics` carry integration responsibility from inside the unit and multiproc directories rather than a dedicated tier.
- **Multi-GPU classification remains a specific open review item.** `contract-multi-device-launch` and `unit-cooperative-multigpu` are cooperative/multi-device by execution context but live in single-device directories; coverage class `multi_gpu.peer_and_cooperative_launch` in [`test_coverage_model.yaml`](test_coverage_model.yaml) marks this `distributed_classification_open`. This documentation change does not move any test source.
- **Retirement requires replacement evidence and owner review.** A suite is retired only when replacement records preserve the original oracle and meaningful prerequisites, provide equivalent or stronger coverage with CI evidence, and an owner approves. Until then the classification stays `review` and the legacy cases keep running; see the migration and retirement decision rules in [`TEST_RESTRUCTURING.md`](TEST_RESTRUCTURING.md).

## Why we test this way

HIP runtime correctness is validated primarily through device-executed tests because the most important behavior depends on the interaction among public HIP APIs, the runtime, the driver, code-object loading, memory management, streams, graphs, and real GPU hardware. CPU-only unit tests are valuable where logic can be isolated, but they cannot fully validate API contracts that require a HIP context, a device allocation, queue execution, peer topology, images, external handles, or backend-specific runtime behavior.

The current balance therefore emphasizes:

- contract tests for small public-API invariants and coverage drift;
- functional tests for deeper per-feature behavior;
- integration-style tests for interop, multi-GPU, multiprocess, HIPRTC, module, loader, and graph scenarios;
- periodic or on-demand performance and stress testing for costlier signals.

Past regressions and validation gaps have shown that source inspection is not enough for HIP runtime behavior. Tests should probe the installed runtime and hardware path they claim to validate, gate unsupported capabilities explicitly, and preserve failure signal rather than hiding crashes or sticky error leaks.

## Key quality concerns

| Concern | Why it matters | How it is validated today |
|---|---|---|
| Public API compatibility | Applications depend on stable return codes, argument validation, and observable API behavior across releases. | Contract tests, functional tests, and coverage drift checks for public HIP APIs. |
| Device-executed correctness | Many runtime defects only appear when commands execute on real GPU queues and memory. | Functional and integration tests that allocate memory, launch kernels, synchronize streams/graphs, and verify observable results. |
| Cross-platform and backend behavior | HIP runs across Linux, Windows, WSL, AMD backends, and portability paths; behavior and availability can differ. | Platform/backend guards, explicit capability skips, backend-portability build coverage, Windows/Linux CI, and documented backend differences in tests. |
| Multi-GPU, IPC, and interop behavior | Peer, IPC, graphics, and external-resource paths often depend on topology or external producers. | Capability-gated multi-GPU, IPC, multiprocess, graphics, and external-resource tests; unsupported paths are skipped or documented. |
| Performance-sensitive paths | Runtime changes can regress memcpy, memset, stream, event, graph, launch, and memory-pool performance. | Performance tests and scheduled/manual performance review; universal per-PR performance gating is a known gap. |
| Test signal integrity | Flaky, disabled, crashing, or poorly isolated tests can make CI results misleading. | Skip policy, sticky-error cleanup guidance, one process per case, YAML disabled entries for explicit cases, and known-gap tracking. |

## Release validation

Release validation extends the normal PR and nightly signals with broader hardware, OS, and configuration coverage. Before a release is considered validated for hip-tests, the relevant release branch should have:

- passing required PR and branch CI gates for the HIP runtime test set;
- successful nightly or multi-architecture validation across the supported AMD GPU architecture set configured for the release;
- Windows and WSL coverage where those lanes are part of the release criteria;
- review of known failing, flaky, disabled, or quarantined tests and their tracking issues;
- performance-regression review for hot paths where benchmark data is available;
- QA or release-owner sign-off according to the release process for the branch.

This document does not replace release checklists or QA sign-off. It records which hip-tests signals contribute to release confidence and where gaps remain.

## Dependencies and validation handoffs

hip-tests depends on the ROCm/HIP install it is built against, the selected HIP backend, the GPU driver and hardware topology, CMake/Catch2 test discovery, generated YAML tags, HIPRTC, module/library loading support, and optional OS or graphics facilities for interop paths.

Validation ownership changes hands at several boundaries:

- TheRock or rocm-systems build workflows prove that packages and test binaries can be built for the selected configuration.
- hip-tests owns runtime API, compiler-facing runtime test paths, HIPRTC, module/library loader, and harness behavior under `projects/hip-tests`.
- External producers and environment setup, such as graphics contexts, Vulkan/OpenGL resources, Windows facilities, or multi-node infrastructure, are owned by the corresponding platform or integration workflows.
- Performance baselines and regression triage are owned by the performance-tracking process that consumes `catch/performance/` results.

When a test depends on a capability outside hip-tests ownership, it should probe for that capability and skip or document the unsupported path rather than failing ambiguously.

## How this document will be used

`TESTING.md` is a living strategy artifact. It is used for:

- PR review discussions about what validation is expected;
- regression analysis when a defect escapes existing tests;
- release-readiness and known-gap review;
- onboarding engineers and contributors to the hip-tests validation model;
- planning CI, coverage, flaky-test, and automation improvements;
- future automation or AI-assisted validation feedback.

The value of this document depends on accuracy. If a workflow is manual, partial, or aspirational, it should be described that way rather than implied as an enforced gate.

## For new contributors

When adding or modifying HIP runtime behavior:

1. Identify whether the behavior can be validated without GPU hardware. If yes, add or update the closest unit-style test. If no, add a device-executed contract, functional, integration, or performance test as appropriate.
2. For public API changes, add or update a contract test for the small portable invariant and a functional test for the changed behavior.
3. For bug fixes, add a regression test that would fail without the fix whenever practical.
4. Use capability checks and `HIP_SKIP_TEST` for unsupported devices, platforms, or runtime paths.
5. Clear sticky HIP errors after intentional negative checks.
6. Run the focused label or executable for the area you changed, plus any relevant static config or coverage checks.
7. Update this document when the testing strategy, gate ownership, supported configuration, or known-gap status changes.

## Owners and review cadence

Keep this document updated in the same PR as any change that alters the testing strategy, such as a new tier, a new cadence, or a new required check. When a test layer or CI gate is added, add its row to the tables above and note its cadence in the relevant section.

Review this document when:

- a major architecture or test-framework change lands;
- a new test pattern, tier, or CI lane is introduced;
- a significant regression escapes existing validation;
- release validation assumptions change; or
- a supported configuration changes in a way that affects test coverage.

The lower-level references remain `CONTRIBUTING.md` for hip-tests contribution guidance, `catch/README.md` for the current Catch2 harness mechanics, and `catch/contract/AUTHORING.md` for the contract-tier how-to. Update this strategy document when those docs change behavior that affects test policy.
