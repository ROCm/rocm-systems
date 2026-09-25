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

## Development workflow

All tests are C++ built on Catch2 and discovered and run through CTest. Test executables are grouped by CMake `BUILD_*` options and by CTest labels generated from Catch2/YAML tags. Workload size is parameterized by level, selected at runtime with `HIP_TEST_LEVEL=level_N` or a Catch2 filter such as `[level_0]`.

| You changed | Run this before pushing | Needs GPU |
|---|---|---|
| Public HIP API behavior | Contract tests for the touched API plus the related functional test label | Yes |
| Functional behavior in a feature area | The matching `catch/unit/` feature-area CTest label or executable | Yes |
| YAML tags, levels, or config generation | Config validation and a focused CTest label using the generated tags | Usually yes |
| Build or test harness logic | Configure, build, and run a small representative label such as `contract` or the touched feature label | Usually yes |
| Performance-sensitive runtime path | Relevant functional tests plus the matching performance scenario when available | Yes |
| Documentation-only change | Markdown/link checks and review against the current test framework | No |

## Testing strategy and layers

The suite lives under `projects/hip-tests/catch/`. Each layer is gated by a CMake `BUILD_*` option and surfaced through CTest labels derived from Catch2/YAML tags via `catch_discover_tests(... ADD_TAGS_AS_LABELS ...)`.

| Layer | Location | Build status / cadence | Primary signal |
|---|---|---|---|
| Contract | `catch/contract/` | ON through `BUILD_CONTRACT_TESTS` | Small public-API invariants and coverage drift |
| Unit / functional | `catch/unit/` plus related focused suites | ON through `BUILD_UNIT_TESTS` | Per-API and per-feature behavior |
| Integration | Currently distributed across unit, interop, multi-GPU, multiprocess, RTC, loader, and graph areas | No separate top-level build option today; runs as a PR/nightly subset depending on platform and capability | Runtime interaction with the ROCm stack and real devices |
| Performance | `catch/performance/` | OFF by default through `BUILD_PERF_TESTS` | Hot-path throughput and latency measurements |
| Stress | `catch/stress/` | OFF by default through `BUILD_STRESS_TESTS` | Long-running and high-load behavior |

### Contract testing

Contract tests live in `catch/contract/` and are built ON by default through `BUILD_CONTRACT_TESTS`. Each case pins the small, portable, device-only semantic guarantee of a public HIP API: successful round-trips, accepted-or-unsupported capability probes, invalid-input rejection, and sticky-error behavior. Cases carry an `@asserts: API - invariant` annotation directly above the case. `catch/tools/gen_test_plan.py` compiles these annotations into `catch/TEST_PLAN.md`, an inventory of what each case asserts. Every declared public API is either covered by a contract test or listed with a rationale in `catch/contract/uncovered_apis.txt`. The repository-root coverage drift gate in `.github/workflows/hip-contract-coverage.yml` runs the static coverage checker and the generated test-plan staleness check so new public HIP APIs do not land without a contract test or an explicit allowlist entry.

### Unit and functional testing

Unit and functional tests live under `catch/unit/` feature-area directories and are built ON by default through `BUILD_UNIT_TESTS`. They validate functional correctness of individual HIP APIs and features across many feature areas, including memory, stream, graph, module, texture, cooperative groups, peer-to-peer, RTC, and virtual memory: arguments accepted or rejected, values returned, and observable side effects such as data movement and synchronization. Tests use the in-tree `HIP_CHECK` and `HIP_ASSERT` helpers, are discovered with `catch_discover_tests`, and run under CTest as one process per case. Workload size is parameterized by level and selected at runtime with `HIP_TEST_LEVEL=level_N` or a Catch2 filter such as `[level_0]`; `catch/config/configs/definitions.yaml` holds the generated per-level parameters and `catch/README.md` is the canonical reference for level semantics.

### Integration testing

Integration coverage is currently distributed across the unit, interop, multi-GPU, multiprocess, RTC, loader, and graph areas rather than a separate top-level tier. It exercises the runtime, compiler-facing paths, HIPRTC, and loader together against a full ROCm/HIP install on real hardware: cross-component interop validated against graphics and external producers, multi-GPU and peer paths that skip cleanly on single-GPU hosts, and an architecture matrix spanning supported CDNA and RDNA families on Linux and Windows. Some groups and cases are platform- or capability-gated, so the matrix describes where the suite runs rather than guaranteeing every case runs on every platform.

### Performance and benchmarking

Performance tests live in `catch/performance/` and are OFF by default through `BUILD_PERF_TESTS`. They measure throughput and latency of hot-path APIs and scenarios such as memcpy, memset, kernel launch, stream, event, and mempool paths on the same Catch2/CTest harness, reusing the per-level workload parameters in `definitions.yaml`. Because they are not built by default, they run on demand or on a dedicated cadence rather than gating every PR. Automated universal per-PR performance regression gating is not currently enforced by this component document; regression baselining is handled by the performance-tracking process that consumes these tests.

### Stress and system-style validation

Stress tests live in `catch/stress/` and are OFF by default through `BUILD_STRESS_TESTS`. They cover long-running and high-load soak behavior and are run on a periodic or release cadence rather than per PR. End-to-end, full-stack system-style scenarios are today expressed through the integration and stress areas rather than a distinct top-level system tier.

## Pre-submit and CI gates

### Validation gates and ownership

| Signal | Classification | Blocks merge? | Notes |
|---|---|---|---|
| TheRock build and HIP runtime test jobs | Trusted gate | Yes, when required by the PR policy | Builds ROCm packages and runs the configured HIP test set. |
| Contract API coverage drift check | Trusted gate | Yes | Requires new public APIs to have a contract test or an explicit allowlist rationale. |
| Generated test-plan staleness check | Trusted gate | Yes | Keeps `catch/TEST_PLAN.md` synchronized with `@asserts:` annotations. |
| Formatting and PR-description checks | Trusted gate | Yes, where configured | Maintains source and review hygiene. |
| Performance and stress sweeps | Informational or scheduled gate, depending on workflow | Not universally on every PR | Used for longer-running validation and regression investigation. |
| Flaky or quarantined cases | Unstable-flaky | No until fixed | Must have an owner, issue, and removal plan before becoming trusted signal. |

### PR test classification

When you change public HIP API behavior, run the contract tests for the touched API together with the related functional test label. When you change functional behavior in a feature area, run the matching `catch/unit/` feature-area CTest label or executable. Config, tag, or level changes should be validated with config validation plus a focused label using the generated tags. A code-changing PR is expected to include an accompanying test file, and a change to a public API needs at least one test covering the new or changed behavior, including at least one invalid-input path where the API defines one.

### Flaky, disabled, and known-bug policy

A flaky or quarantined case is treated as unstable signal: it does not block merge, but it must have an owner, a tracking issue, and a removal or fix plan before it can be promoted back to trusted signal. Use `HIP_SKIP_TEST` for a genuine capability gap such as an unsupported device or runtime path, too few GPUs, or no image support, so the case skips cleanly instead of failing. Never leave a test that crashes the process. Any test that intentionally triggers a HIP error must consume the sticky thread-local error with `(void)hipGetLastError()` before returning so it does not leak into later cases in the same process.

Disabling a test case should be rare and explicit. It is appropriate when a test cannot execute on a platform or architecture because a required capability, OS facility, graphics environment, or runtime mode is absent; when a known product or infrastructure defect has a tracking issue; or when keeping the test enabled would crash the process and hide later signal. Prefer runtime capability checks plus `HIP_SKIP_TEST` for environment-dependent behavior, and use YAML `disabled:` entries only when a case should be hidden for a specific platform, architecture, sanitizer mode, or tracked known issue. Do not disable a test only to make CI green. Every disabled or quarantined case should record why it is disabled, where it is tracked, and what condition allows it to be re-enabled.

## Coverage

Contract coverage is tracked statically: every declared public HIP API is either covered by a contract test or listed with a rationale in `catch/contract/uncovered_apis.txt`. The inventory of what each case asserts is generated into `catch/TEST_PLAN.md` by `catch/tools/gen_test_plan.py` from the `@asserts:` annotations, and the repository-root `.github/workflows/hip-contract-coverage.yml` enforces both the coverage drift check and the test-plan staleness check. Functional coverage is organized per feature area under `catch/unit/`, with the expectation that every public HIP API a change touches has at least one functional test exercising the changed behavior. Device-side code coverage is not fully captured by host coverage tools.

Code coverage and test coverage are different signals. Code coverage measures which host-side lines or branches executed. Test coverage measures which public API contracts, configurations, capabilities, and regression scenarios were intentionally validated. HIP runtime confidence depends heavily on test coverage across real hardware and runtime configurations, not only on line coverage.

## Nightly validation

Nightly and multi-arch runs extend the per-PR build-and-test flow across a broader architecture set and longer-running suites. Today, if `HIP_TEST_LEVEL` is unset, the listener infers the active level from the first generated `[level_N]` tag it sees, unless a scheduled or on-demand workflow explicitly overrides the level. The CI handling of levels is expected to change when the pending level-selection workflow update in ROCm/rocm-systems#8932 lands, so this section should be reviewed in the same PR or immediately after that merge. On-demand runs such as multi-arch sweeps, WSL runtime checks, and performance or stress runs are invoked explicitly rather than on every PR. CI level cadence is not yet fully wired to `HIP_TEST_LEVEL`.

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

AddressSanitizer builds are supported through `ENABLE_ADDRESS_SANITIZER` and through TheRock sanitizer settings such as `THEROCK_SANITIZER=ASAN` or `THEROCK_SANITIZER=HOST_ASAN`. Sanitizer builds can catch host-side memory errors in the test harness and runtime-facing code paths that execute on the host. Device-executed paths limit what host sanitizers observe, so sanitizer coverage of device-side behavior is inherently partial.

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
| CI level cadence is not fully wired to `HIP_TEST_LEVEL` | Medium | PR and nightly runs do not consistently select different workload levels through one global dial | Per-case generated `[level_N]` tags and on-demand overrides still provide level control; review after ROCm/rocm-systems#8932 lands |
| Level-based test parameters are defined but not broadly utilized today | Medium | Levels can select or label tests, but many cases do not yet scale workload sizes, iteration counts, or generated parameter sets based on the active level | Prefer adding level-aware parameter use when touching long-running tests; use `isQuickLevel()` and generated level parameters where practical |
| No universal per-PR performance baseline gate | Medium | Throughput or latency regressions may be found after merge or by manual review | Performance tests exist and can run on demand or on a scheduled cadence |
| Device-side code coverage is not fully captured by host coverage tools | Medium | A high host coverage number can still miss device-only behavior | Contract, functional, integration, and stress tests execute behavior on real devices |
| Flaky and quarantined tracking should keep converging toward owner plus issue plus expiry | Medium | Suppressed failures can age into permanent blind spots | Treat flaky/quarantined cases as unstable signal until each has a tracking issue and removal plan |
| Sanitizer coverage of device-side paths is inherently partial and is not part of the default per-PR gate | Low | Memory or race issues can escape sanitizer lanes | Use sanitizers as an additional signal, not a replacement for device-executed tests |

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
