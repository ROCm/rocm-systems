<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# hip-tests Testing Strategy

This document is the authoritative testing-strategy reference for the **hip-tests**
component (the test suite for the HIP runtime and its compiler/loader surface):
how it is validated, how test tiers are organized, and what engineers should
consider when writing tests. It exists so that:

- Engineers stop guessing what tests to write, and PR approvals move faster.
- Testing knowledge is documented and survives team changes.
- Test gaps are visible before they become release blockers.
- Fewer defects escape into integration and QA.
- There is a single place to audit the quality posture of the component.

It is a living design/strategy document. It has two parts:

- **[Part A — Current status](#part-a--current-status)** describes what is in the
  component **today** (as merged on `develop`).
- **[Part B — Future status](#part-b--future-status)** describes the direction:
  the semantic **contract** test tier and the progressive **test-tier
  restructure**. Items in Part B are in progress and land incrementally; this
  section is the plan of record, not a description of current `develop`.

Keep this document updated in the same PR as any change that alters the testing
strategy (a new tier, a new cadence, a new required check).

---

## Scope and what "the component" is

hip-tests validates the **public HIP API contract and behavior** exposed by the
HIP runtime (`libamdhip64`), the HIP compiler front end, HIPRTC, and the
module/library loader. It is built and run against a ROCm/HIP install and, on a
portability path, against the NVIDIA HIP-over-CUDA backend. It does **not** test
math libraries, communication libraries, or profiler internals — those components
own their own suites.

All tests are C++ built on **Catch2 v3.8.1** and discovered/run through **CTest**.
Test executables are grouped by CMake `BUILD_*` options and CTest **labels**.

---

# Part A — Current status

## A.1 Test categories (today)

The suite lives under `projects/hip-tests/catch/` and is organized into these
groups, each gated by a CMake option and built into its own set of executables:

| Group | Directory | CMake option (default) | Purpose |
|---|---|---|---|
| **Unit / functional** | `catch/unit/` | `BUILD_UNIT_TESTS` (ON) | The primary suite — per-API and per-feature functional tests, ~50 feature areas (memory, stream, graph, module, texture, cooperative groups, p2p, RTC, virtual memory, etc.). |
| **Performance** | `catch/performance/` | `BUILD_PERF_TESTS` (OFF) | API- and scenario-level performance/benchmark tests (memcpy, memset, kernel launch, streams, mempool, events). |
| **Stress** | `catch/stress/` | `BUILD_STRESS_TESTS` (OFF) | Long-running / high-load soak tests. |
| **Multi-process** | `catch/multiproc/` | (under `BUILD_UNIT_TESTS`, Linux/UNIX only) | Tests that fork/spawn multiple processes (IPC, multi-proc device sharing); not part of the Windows test set. |
| **ABM / TypeQualifiers** | `catch/ABM/`, `catch/TypeQualifiers/` | (under `BUILD_UNIT_TESTS`) | Focused compile/behavior checks. |

### A.1.1 Unit testing

- **What is tested:** functional correctness of individual HIP APIs and features —
  arguments accepted/rejected, values returned, observable side effects (data
  movement, synchronization, graph/stream semantics). Coverage is broad and
  organized by feature area under `catch/unit/<area>/`.
- **Tooling:** Catch2 v3.8.1 (`TEST_CASE`/`SECTION`), the in-tree `HIP_CHECK`,
  `HIP_ASSERT`, and helpers in `catch/include/`; kernels shared via the `KERNELS`
  and `Main_Object` object libraries. Tests are discovered with
  `catch_discover_tests` and run under CTest.
- **Level parameterization:** test workload size is parameterized by **level**
  (0–4). `projects/hip-tests/catch/README.md` is the canonical user-facing
  reference for level semantics; `catch/config/configs/definitions.yaml` holds the
  generated parameters. Levels are selected with string tags such as `level_0`
  (for example, `HIP_TEST_LEVEL=level_0` or the Catch2 filter `[level_0]`), not
  bare integers. `level_0` is quick/smoke, `level_2` is the standard/default
  suite level, and some other levels are still reserved/provisional. See
  [A.3](#a3-when-tests-run) for how CI uses levels today.
- **Coverage expectation (today):** every public HIP API a change touches should
  have at least one functional unit test exercising the changed behavior. The
  Systems PR bot enforces that a code-changing PR includes an accompanying test
  file (`test_*` / `*_test.*`).

### A.1.2 Integration testing

hip-tests **is** the integration layer for the HIP runtime: it is built against a
full ROCm/HIP install and exercises the runtime, compiler, HIPRTC, and loader
together on real hardware. Integration is covered along these axes today:

- **Backend integration:** the suite builds and runs against the AMD ROCm/HIP
  backend, and a portability subset builds against the NVIDIA HIP-over-CUDA
  backend (`HIP_PLATFORM=nvidia`) — see `catch/unit/` graphics/RTC/vulkan interop
  and the NVIDIA CI job.
- **Cross-component interop:** `gl_interop`, `vulkan_interop`, and external-memory
  / external-semaphore tests validate HIP against graphics and external producers
  (these require a graphics/Vulkan environment and are gated accordingly).
- **Multi-GPU / peer:** `p2p/`, cooperative multi-device, and multi-process tests
  exercise device-to-device and cross-process behavior; they skip cleanly on
  single-GPU hosts.
- **Architecture matrix:** built and run in CI across the supported GPU
  architecture set (CDNA and RDNA families) and both Linux and Windows. Some
  groups and cases are platform- or capability-gated (for example UNIX-only
  multi-process tests, graphics/Vulkan environment requirements, and selected
  Windows-specific paths), so the matrix describes where the suite runs rather
  than a guarantee that every individual case runs on every platform.

### A.1.3 Performance testing

- **What is tested:** throughput/latency of hot-path APIs and scenarios
  (`catch/performance/api/` and `catch/performance/scenarios/`) — memcpy, memset,
  kernel launch, stream, event, and mempool paths.
- **Tooling:** the same Catch2/CTest harness. The per-level workload parameters in
  `definitions.yaml` (memory sizes, iterations, warmups) are available to these
  tests, selected at runtime via `HIP_TEST_LEVEL=level_N` (for example,
  `HIP_TEST_LEVEL=level_0`) like the rest of the suite; CI does not auto-scale
  the level per cadence (see [A.3](#a3-when-tests-run)). As above, the current
  per-level values in `definitions.yaml` are still marked provisional.
- **Baselines & thresholds:** performance tests are **not built by default**
  (`BUILD_PERF_TESTS=OFF`) and are run on demand / on a dedicated cadence rather
  than gating every PR. Regression baselining is done by the perf-tracking process
  that consumes these tests; per-PR performance gating is **not** currently
  enforced by this component's CI. *(This is an area Part B tightens.)*

## A.2 How tests are built and run

```bash
# configure (from a ROCm/HIP install)
cmake -S projects/hip-tests/catch -B <build> \
      -DBUILD_UNIT_TESTS=ON            # perf/stress are OFF by default

# build the default target (unit tests are part of ALL)
cmake --build <build> -j

# run by a generated Catch2 tag/label (example: a feature-area label)
ctest --test-dir <build> -L memory --output-on-failure
```

Tests are labeled from their Catch2 tags (`ADD_TAGS_AS_LABELS`), so `ctest -L
<label>` selects a feature area or tier when that tag exists (for example,
`memory`, `stream`, or `contract`; labels are generated from YAML/Catch2 tags,
not from directory names alone). Each `HIP_TEST_CASE`/`TEST_CASE` becomes one
CTest entry (one process per case under `catch_discover_tests`). When running
from the generated `catch_tests/` directory, the build copies `DartConfiguration.tcl`
there so CTest settings such as the default timeout still apply.

> **Build note:** Catch2 test executables are created `EXCLUDE_FROM_ALL` and are
> pulled into the build only through their aggregate custom target. The unit
> aggregate (`build_tests`) is declared `ALL`, so unit tests build with the
> default target; a new test group must attach to an `ALL` aggregate (or be added
> to one) or CI — which builds the default target — will register the tests but
> never compile them, and CTest will report them as `<Target>_NOT_BUILT`.

## A.3 When tests run

| Cadence | Trigger | Level used today | What runs |
|---|---|---|---|
| **Per PR** | `pull_request` (TheRock CI, `therock-ci.yml`) | `HIP_TEST_LEVEL` unset; listener infers each case's generated `[level_N]` tag (`level_2` is the standard/default suite level) | Build the ROCm packages + hip-tests, then run the unit suite (sharded) across the supported Linux and Windows architecture matrix. Plus lightweight policy/format checks (`hip-formatting`, `hip-validate-pr-description`, NVIDIA build). |
| **Push to `develop` / release branches** | `push` (`therock-ci.yml`) | `HIP_TEST_LEVEL` unset; listener infers each case's generated `[level_N]` tag (`level_2` is the standard/default suite level) | Same build+test as per-PR, on the integration branch. |
| **Nightly / multi-arch** | scheduled multi-arch CI | `HIP_TEST_LEVEL` unset unless the workflow explicitly overrides it; otherwise per-case `[level_N]` tags are inferred | Broader architecture coverage; longer-running suites. |
| **On demand** | `workflow_dispatch` | caller/workflow-selected, otherwise `HIP_TEST_LEVEL` unset with per-case `[level_N]` inference | Multi-arch CI, WSL runtime checks, and perf/stress runs are invoked explicitly. |

A **level** parameterization (0–4) exists in `definitions.yaml` and is selected
at **runtime**. A caller can force a global level via `HIP_TEST_LEVEL` (using
values such as `HIP_TEST_LEVEL=level_0`, not `HIP_TEST_LEVEL=0`) or a `[level_N]`
Catch2 tag filter; otherwise the listener infers the level from each case's
generated `[level_N]` tag. Untagged tests fall back to the framework defaults,
and `level_2` is documented in `catch/README.md` as the standard/default suite
level. The YAML documents an intended cadence mapping (level 0 for PR/emulation,
level 1 for nightly), but **CI does not currently set a global level per cadence**
— wiring that cadence dial to CI is a Part B item (see
[B.3](#b3-planned-tightening-integration--performance--system)).

## A.4 How to write tests (today)

- **Location & naming:** put a test under `catch/unit/<feature-area>/`; name the
  file so it is recognized as a test (`test_*` or `*_test.*` per the PR-bot rule),
  and name each case `<Area>_<Behavior>` so intent is legible in CTest output.
- **Structure:** one `TEST_CASE` per behavior; use `SECTION`s for variants. Use
  `HIP_CHECK` for API calls that must succeed and assert the observable invariant
  with `REQUIRE`. Clean up every resource you allocate.
- **Level-aware workload trimming:** use `isQuickLevel()` (from
  `hip_test_common.hh`) to reduce buffer sizes, iteration counts, or generated
  parameter sets at `level_0` rather than skipping code paths entirely. Call
  `isQuickLevel()` inside the test function, not in global/static initializers,
  because the active level is detected at runtime. If the test uses randomized
  inputs, use deterministic/reproducible seeds and log or allow overriding the
  seed so failures can be replayed.
- **Skips vs failures:** use `HIP_SKIP_TEST` for a genuine capability gap
  (unsupported device/runtime path, too few GPUs, no image support) so the case
  skips instead of failing. Never leave a test that crashes the process — guard
  the unsafe call.
- **Sticky errors:** any test that intentionally triggers a HIP error (negative
  tests, invalid-argument paths, accepted-or-unsupported probes) must consume the
  sticky thread-local error with `(void)hipGetLastError()` before returning, unless
  the helper macro already does that. Otherwise the error can leak into later
  cases in the same process and create cascading, misleading failures.
- **Platform-specific behavior:** use explicit platform/capability guards for
  paths that only work on a subset of hosts (for example `#if defined(_WIN32)` or
  `#if !defined(_WIN32)`). Be especially careful with Windows/WSL differences:
  tests that depend on cwd-relative fixture files (`.code`, `.txt`, generated
  modules) should resolve paths through the existing test helpers rather than
  assuming the process starts in the source directory, and tests that can trigger
  WDDM/GPU-queue fatal errors may need a platform skip instead of relying on later
  recovery.
- **Device/process isolation:** clean up every resource you allocate. If a test can
  leave the device in a permanent error state (for example after OOM or queue
  failure paths), add cleanup that resets/reinitializes the device where safe
  (e.g. `hipDeviceReset()` in a guarded cleanup path) or split/skip the test so it
  cannot corrupt subsequent cases in the same executable.
- **Sufficient coverage:** a change to a public API needs at least one test
  covering the new/changed behavior, including at least one invalid-input path
  where the API defines one. Register the test's build target and (where used) its
  YAML config entry so it actually compiles and runs.
- **References:** `projects/hip-tests/CONTRIBUTING.md` is the repo-wide
  contribution guide; `projects/hip-tests/catch/README.md` documents the current
  Catch2 harness mechanics (YAML tags, levels, macros). This file is the testing
  strategy / tiering overview and should be updated when those lower-level docs
  change behavior that affects test policy.

---

# Part B — Future status

This section is the **plan of record** for where hip-tests testing is going. These
items land incrementally; they are not all present on `develop` yet.

> **Tracking:** the contract tier and tooling described in Part B were developed
> on the **[`users/mangupta/hip-contract-tests`](https://github.com/ROCm/rocm-systems/tree/users/mangupta/hip-contract-tests)**
> branch and are tracked by **[PR #8827](https://github.com/ROCm/rocm-systems/pull/8827)**.
> Branch-specific counts and validation evidence below are labeled as such and
> should be refreshed from the generated tooling when the PR lands or materially
> changes.

## B.1 Progressive test-tier restructure

The `catch/*` suite is being reorganized into a **progressive-complexity tier
ladder**, where the **tier is the top-level directory** under `catch/`:

```
contract → unit → integration → system → performance → stress
```

Each tier is a distinct, independently-buildable group with its own CMake
`BUILD_*` option and CTest label, ordered from cheapest/most-portable to
most-expensive:

| Tier | Question it answers | Cost / cadence |
|---|---|---|
| **contract** | Does each public API honor its small, portable, device-only semantic guarantee? | Cheapest; every PR |
| **unit** | Is each API/feature functionally correct in depth? | Every PR (planned level 0), nightly (planned level 1) |
| **integration** | Do components work together (interop, multi-GPU, loader + RTC + runtime)? | PR subset + nightly |
| **system** | Do end-to-end flows work on a full stack/host? | Nightly / periodic |
| **performance** | Are hot paths within throughput/latency baselines? | Periodic + regression gate |
| **stress** | Does the runtime hold up under sustained load? | Periodic / release |

**Why tiers:** they make the cost/coverage tradeoff explicit, let CI run the cheap
high-signal tiers on every PR and defer expensive tiers, and — critically — let us
**detect and remove redundant coverage across tiers** rather than testing the same
guarantee three times. The room for additional tiers is intentional.

### B.1.1 Cross-tier de-duplication

Every test case carries a machine-readable intent tag directly above it:

```cpp
// @asserts: <API> - <one-line portable invariant this case pins>
HIP_TEST_CASE(Contract_Domain_Behavior) { ... }
```

A generator (`catch/tools/gen_test_plan.py`) compiles these into
`catch/TEST_PLAN.md`, an inventory of *what each case asserts*, grouped by tier and
domain. This is the tool reviewers and authors use to spot the same API being
asserted in multiple tiers and decide where each guarantee belongs. The plan is
regenerated from source and staleness-checked in CI, so it cannot drift. The
generated inventory records whether each case has an `@asserts:` tag; stricter
CI linting of the invariant format is planned so the human-readable text does not
silently drift.

## B.2 The contract tier (new)

The first new tier is **contract** (`catch/contract/`): tests that pin the small,
**portable, device-only semantic guarantees** of the public HIP API — round-trips,
accepted-or-unsupported outcomes, and invalid-input rejection — as opposed to the
deep behavioral coverage the unit tier provides.

- **Scale (as of the `users/mangupta/hip-contract-tests` branch / PR #8827):**
  592 cases across 118 API domains, covering ~98% of the public APIs declared in
  `hip_runtime_api.h` (name-level). Re-check these generated numbers when PR
  #8827 lands and update this section alongside the branch-reference note above.
- **Portability:** the same sources build and run on the AMD backend and the
  NVIDIA HIP-over-CUDA backend; backend-divergent or CUDA-removed APIs are
  `#if HT_AMD`-gated (compile to empty on NVIDIA) and real behavioral differences
  are marked with a `BACKEND-DIFF:` comment.
- **Validation done (for the PR #8827 branch, before merge):** built and run
  across gfx1101 (RDNA3), MI100 (gfx908), MI200 (gfx90a), H100 (CUDA 13.1), and
  V100 (CUDA 12.9), including 2-GPU paths. Treat this as branch validation
  evidence, not a standing `develop` CI guarantee.

### B.2.1 Coverage drift gate (new)

A CI check (`.github/workflows/hip-contract-coverage.yml`, pure static analysis —
no GPU/build) fails a PR when a newly-added public HIP API ships **without** a
contract test and is not on an explicit, reasoned allowlist
(`catch/contract/uncovered_apis.txt`). This makes coverage regressions visible at
PR time instead of at release. The same workflow staleness-checks `TEST_PLAN.md`.

- **Coverage expectation (contract tier):** every public API declared in
  `hip_runtime_api.h` is either covered by a contract test or listed in the
  allowlist with a reason. The checker
  (`catch/contract/tools/check_contract_coverage.py`) reports the exact gap and is
  runnable locally.

### B.2.2 Authoring contract tests

On the PR #8827 branch, `catch/contract/AUTHORING.md` is the authoritative
how-to for the contract tier: the conventions (resource cleanup guard,
image-gating, `HT_AMD`/`BACKEND-DIFF` markers, sticky-error clearing, the
probe-first rule), a step-by-step add procedure, the `@asserts:` tag, and a
copyable skeleton (`TEMPLATE.cc.txt`). `projects/hip-tests/CONTRACT_COVERAGE.md`
tracks the coverage checker and the per-API gap rationale; refresh these links
and wording when the branch lands on `develop`.

## B.3 Planned tightening (integration / performance / system)

As the tiers are formalized, the following are planned and will be documented here
as they land:

- **Integration tier:** promote the interop/multi-GPU/loader-combination tests out
  of `unit/` into an explicit `integration/` tier with its own cadence, so
  cross-component coverage is auditable separately from single-API unit coverage.
- **Performance regression gate:** define per-scenario **baselines** and
  **regression thresholds** and wire a periodic (and optionally per-PR-on-hot-paths)
  gate, so performance regressions are caught automatically rather than by manual
  baselining.
- **System tier:** end-to-end, full-stack scenarios distinct from single-runtime
  integration.
- **Level cadence wiring:** connect the `HIP_TEST_LEVEL` dial to CI so PR/emulation
  runs use level 0 and nightly/regression runs use level 1 (and higher levels feed
  periodic sweeps), instead of the current runtime-only, unfiltered default.

---

## Maintenance

- Update this document in the same PR as any change to the testing strategy (a new
  tier, cadence change, or new required check).
- When you add a test tier or a CI gate, add its row to the tables above and note
  its cadence in [A.3](#a3-when-tests-run) / [B.1](#b1-progressive-test-tier-restructure).
- Keep Part A describing what is on `develop` and Part B describing the plan; move
  items from B to A as they merge.
