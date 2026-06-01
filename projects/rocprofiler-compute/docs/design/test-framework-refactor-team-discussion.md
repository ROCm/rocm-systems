## rocprofiler-compute — Test Framework Refactor (Team Discussion Draft, Detailed)

This document is a polished, detailed capture of the conversation that kicked off the
test-framework refactor effort. It is intended to be used as a discussion artifact with the
rocprofiler-compute team.

**Important guardrails (current agreement):**

- This stays at **design/proposal** level. No implementation steps have been executed yet.
- Any refactor that touches test infrastructure must respect the profiling infra constraints in
  `.ai/rules/profiling_infra.md` (determinism, fixture sizing, output schema stability, etc.).

---

## 1) Kickoff (what we’re trying to do)

We want to follow the spec-driven workflow to work on a new feature:

- **Goal**: refactor the test framework for `rocprofiler-compute`
- **Explicit objectives**:
  - Clearly define test definitions on each test level
  - Define test filters properly
  - Choose a refactoring strategy:
    - (1) gradually modify the current framework, or
    - (2) keep the current one alive and build the new one in parallel

**Reference inputs used in the discussion:**

- **AGSRCIT Test Framework Design (PDF)**: hierarchical test types, multi-dimensional filter engine
- **Yang Unit Test Framework (PDF)**: unit-test-first, pytest markers + CTest launcher, CTest as a thin registry
- **`test_refactor.md` (vedithal plan)**: unit/integration layout, migration phases, no test-logic changes
- **PR #6189**: refactor implementation branch; candidate cherry-pick core commit `07b0579`

---

## 2) Problem statement (why now)

The team’s top motivations (from your A1 answers) are:

- **Test code quality / maintainability / triage cost** is too high.
- **There is no clear boundary** between unit tests and GPU / CLI / integration tests.
- The suite is **hard to extend** for new GPU architecture support and new features.

What we want is a set of **authoritative definitions + a classification system** that allow us to:

- organize tests and suites consistently
- balance coverage vs cost
- reduce maintenance risk and triage overhead
- scale across new architectures/features

---

## 3) Deliverable intent & scope

**Outcome direction** (A2):

- Aim for a **framework-level refactor** with clear definitions and filters.
- Keep an **incremental migration path** (avoid large, risky big-bang).
- Start with a **design-first** deliverable (proposal/spec) before implementation.

**Schedule**: no deadline.  
**Reviewers**: rocprofiler-compute team; TheRock may review CI integration details.

**Scope boundary for the initial refactor phase (G1 “all out”):**

- No rewriting assertions / changing test expectations.
- No adding new tests or expanding coverage (coverage must be maintained).
- No fixing flaky tests as part of this refactor (unless required to keep CI green).
- No `src/` changes for testability.
- No custom test runner or heavy pytest plugin system in MVP.

---

## 4) Brainstorming stage Q&A (questions + your answers)

This section records the *brainstorming-stage* questions used to clarify requirements, plus
your answers. These details are kept explicitly because they drive the design trade-offs.

### 4.1 Goals & success criteria

- **A1 (primary problems to solve)**: `c)`, `d)`, `f)` — clear definition and classification to organize tests/suites, balance coverage vs cost, reduce maintenance, and improve extensibility.
- **A2 (target deliverable)**: combine `e)`, `c)`, `a)` — aim toward a full refactor direction, preserve the current system while bringing up the new one, and start from proposal/design first.
- **A3 (deadline)**: no deadline.
- **A4 (reviewers)**: rocprof-compute team members; TheRock review mainly for CI definition.

### 4.2 Test level definitions / taxonomy

- **B1 (model)**: start from **Option B (five-level thinking)**, but re-evaluate whether “component” and “integration” should be combined. Also requested open reference designs for profiling tools and Python/C++ mixed projects.
- **B2 (definitions)**: follow B1; additionally, performance tests should include **pressure/stress** testing.
- **B3 (current tests classification)**: unknown today; classification/discovery is part of the work.
- **B4 (scope for harnesses)**: CTest + PTest are the major target; for C++ prefer **GoogleTest** (not Catch2).
- **B5 (profile vs analyze / component meaning)**: prefer dropping “component” (fold into integration); “profile” and “analyze” are distinct modes; in the future support both “two-stage” and “one-stop” modes.

### 4.3 Filters & CI tiers

- **C1 (purpose of filters)**: filters are not only for CI — developers should run partial suites quickly with conditions; quick/standard/comprehensive/full should become “one line” presets (still thinking about exact mapping).
- **C2 (MVP filter dimensions)**: `type`, `arch`.
- **C3 (developer entrypoints)**: both `ctest` and direct `pytest` should be supported.
- **C4 (when to run)**: every PR (at least for the agreed set).
- **C5 (`test_categories.yaml`)**: keep for now; should be replaced by filters in the future.
- **C6 (GPU-less CI requirement)**: nice-to-have (not a hard requirement).

### 4.4 Refactoring strategy & PR shape

- **D1 (migration strategy)**: choose **(2) parallel bring-up**, and **merge (3) into (2)** (reuse PR #6189 partially). Also add **sample-first adoption**: for each category (Unit / Integration / Functional / Performance), land one sample test template (or a few real sample cases) in the new layout before bulk migration.
- **D2 (parallel coexistence length)**: keep both until CI is green on the new framework; then proceed to cutover path.
- **D3 (hard constraints)**: yes — no `src/` changes, no test-logic changes, preserve CTest args.
- **D4 (delivery shape)**: OpenSpec change first, then phased implementation PRs.
- **D5 (relationship to PR #6189)**: cherry-pick core commit(s) only.

### 4.5 Layout & fixtures

- **E1 (layout)**: combine folder structure (`tests/unit/`, `tests/integration/`) with markers.
- **E2 (conftest split)**: prefer per-subtree fixtures; asked about differences and leaned toward subtree conftest layout.
- **E3 (integration naming)**: `test_<feature>.py`.
- **E4 (borderline tests)**: fixed split rule (banner/module split), not re-deciding assertions.

### 4.6 Source of truth & documentation placement

- **F1 (authoritative source when conflicts exist)**: the new unified spec we write in this brainstorm (not any single prior doc).
- **F2 (artifact location)**: OpenSpec change directory under `openspec/changes/`; can link out to other docs if needed.
- **F3 (mapping between models)**: prefer “pick one model” (avoid large reconciliation tables in the spec).
- **F4 (TheRock constraints)**: keep artifact directories stable.

### 4.7 Scope boundaries & quality

- **G1 (non-goals)**: treat all “nice-to-have” work as out-of-scope for this phase (assertion changes, new tests, `src/` changes, etc.).
- **G2 (coverage gate)**: coverage must not regress.
- **G3 (experimental/arch-gated tests)**: out of scope for MVP.

### 4.8 Risks & preferences

- **H1 (biggest risks)**: CI breakage, taxonomy disagreement, over-engineering filters early.
- **H2 (bias if forced today)**:
  - Strategy: (2) parallel, partially based on #6189
  - Taxonomy: 2-level physical split
  - Filter MVP: both (markers + CI entrypoints)

---

## 5) Brainstorm conclusions (key decisions so far)

### 5.1 Taxonomy tension and the reconciliation

You initially favored the hierarchical AGSRCIT model (unit/component/integration/functional/performance),
but later indicated a practical preference for a **two-level physical split** (H2).

**Reconciliation** agreed in the discussion:

- **Physical layout (MVP)**: two roots only
  - `tests/unit/`
  - `tests/integration/`
- **Logical classification grows over time**:
  - MVP `type`: `unit`, `integration`
  - Later `type`: add `functional`, `performance` (including pressure/stress)
- **“Component”** is likely folded into integration for now (you preferred dropping it; B5).

This keeps navigation simple immediately while preserving a long-term taxonomy for classification/filtering.

### 5.2 Strategy (gradual vs parallel + sample-first)

You chose:

- **Strategy**: (2) **parallel** bring-up, plus selectively reuse PR #6189 (merge (3) into (2)).
- **Sample-first adoption**: before moving the full suite, add **one sample test template per category** (or a few real sample cases) in the new subsystem so contributors can copy the pattern:
  - **Unit** — e.g. `tests/unit/.../test_<module>.py` (no GPU, mocks only)
  - **Integration** — e.g. `tests/integration/test_<feature>.py` (CLI/GPU/workload fixture)
  - **Functional** — golden-output / end-user flow sample (later tier)
  - **Performance / pressure** — overhead or stress sample (later tier)
- **PR relationship**: cherry-pick core commit(s) only (`07b0579`), do not adopt the entire messy branch history.
- **Cutover rule**: after new framework is **CI-green**, keep legacy flat tests running **two more weeks in parallel**
  (soak period), then remove legacy.

### 5.3 Filter MVP

You selected:

- **MVP filter dimensions**: `type`, `arch`
- Filters are not only for CI; they must support dev partial runs as well.
- Keep `test_categories.yaml` for now, but long-term it should be replaced by filters.

### 5.4 CTest/PTest and C++ tests

You indicated:

- **CTest + PTest** are major targets for the overall test framework and CI execution.
- For C++ tests, prefer **GoogleTest** (not Catch2) for potential expansion.

---

## 6) Open reference patterns (what the design is inspired by)

The conversation referenced several open patterns to ground the design choices:

- **pytest + markers** as the classification/filter backbone (both PDFs trend that way).
- **CTest as a thin registry/launcher** (Yang PDF), which preserves TheRock/CI behavior.
- **Profiler-style pipeline testing** (capture → analyze → validate) commonly uses:
  - unit tests for pure logic/parsers,
  - integration tests for GPU/CLI glue paths and workload fixtures,
  - functional/perf as later layers (goldens/baselines).

### 6.1 TheRock reference design (target state)

For the **new test framework connected to TheRock**, the team wants to **eventually align** with the
**hipBLAS `test_categories.yaml` pattern** used across `rocm-libraries` (same tier model as rocBLAS).

**Reference design (canonical example):**

- [hipBLAS `test_categories.yaml`](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblas/clients/gtest/test_categories.yaml)
- Related: [hipBLAS gtest `CMakeLists.txt`](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblas/clients/gtest/CMakeLists.txt) (wires `apply_test_category_labels`)
- Shared spec: [rocm-libraries `shared/ctest/README.md`](https://github.com/ROCm/rocm-libraries/blob/develop/shared/ctest/README.md)

**What we adopt from that reference:**

| Element | hipBLAS reference | rocprofiler-compute direction |
|--------|-----------------|-------------------------------|
| Tier names | `quick`, `standard`, `comprehensive`, `full` | Keep (already in our YAML) |
| Per-tier `description` | Human-readable + Jenkins/TheRock intent | Align wording and timeouts |
| `test_patterns` / `exclude` | Glob patterns for test selection | Map to CTest test names and/or pytest markers |
| `labels` | CTest labels for `ctest -L` | Same — primary TheRock filter mechanism |
| `execution_settings` | `category_timeouts`, `environment`, `timeout_multiplier` | Adopt structure; tune for pytest/CI |
| GPU exclusions | Optional `exclude_gpu` block (see shared README) | Add when arch-gating is formalized |

**Important distinction (not a blocker):**

- hipBLAS uses **rocm-libraries** `apply_test_category_labels()` → one GoogleTest binary + gtest filters.
- rocprofiler-compute uses **rocm-systems** `apply_ctest_category_labels()` → labels on many existing `add_test(pytest …)` entries.

The **YAML shape and TheRock contract** should converge on the hipBLAS reference; the **CMake/parser layer** may remain the rocm-systems variant until/unless we unify parsers across monorepos.

**Phasing:**

1. **MVP / bridge:** Keep current `tests/test_categories.yaml` + `apply_ctest_category_labels()` (required for TheRock today).
2. **Sample-first scaffolding:** Add reference samples per category (Unit + Integration first; Functional + Performance when those tiers are enabled) — wired through markers, CTest, and YAML like production tests.
3. **Refactor:** Reorganize tests (`unit/` / `integration/`) and add marker dimensions (`type`, `arch`); migrate existing tests category-by-category using samples as templates.
4. **Target:** Restructure `test_categories.yaml` to match hipBLAS reference fields and semantics; retire duplicate selection in `therock_test_runner.py` (hardcoded `QUICK_TESTS` / `TEST_TYPE`).

---

## 7) Proposed design (Sections 1–7)

### Design — Section 1: Problem & goals

**Problem**

- Tests are hard to triage and organize.
- The suite lacks an authoritative boundary between “unit” and “GPU/CLI integration”.
- Filters are not defined as a coherent system usable by both CI and developers.
- Extending across architectures and new profiler features is costly.

**Goals**

- **Authoritative test definitions** and classification rules.
- **Filter MVP** that supports both CI and developer partial runs.
- **Physical organization** that makes “where is the test for this source file/feature?” obvious by path.
- Preserve infra constraints from `.ai/rules/profiling_infra.md`:
  - determinism (stable seeds/order),
  - fixture size discipline (`tests/workloads/` for larger data),
  - output schema stability when formats/counters change.

**Non-goals (for initial refactor phase)**

- No new assertions or coverage expansion.
- No `src/` changes for testability.
- Avoid heavy plugin systems; keep filter engine incremental.

### Design — Section 2: Test level definitions

**MVP “type” definitions**

- **Unit**:
  - in-process, no GPU requirement, no `rocprof-compute` subprocess
  - uses mocks/tiny fixtures; runs on dev laptop without ROCm
- **Integration**:
  - exercises real glue paths: CLI driving, GPU-required capture/analyze behavior,
    workloads as black-box fixtures

**Later (not MVP)**

- **Functional**: end-user flows and output correctness (golden baselines)
- **Performance**: overhead/throughput/regressions; includes pressure/stress tests

**Borderline rule**

When a file mixes types (common in the current flat layout), split by section banner/imported
source module without changing assertions or expectations.

### Design — Section 3: Layout & fixtures

**Target layout (MVP)**

- `tests/unit/` mirrors `src/`
- `tests/integration/` has one file per feature

**Fixture strategy**

Prefer per-subtree fixtures (clear boundaries, scales when we later add more types):

- `tests/conftest.py`: common/lightweight shared fixtures and option parsing
- `tests/unit/conftest.py`: unit-only fixtures (mocks, tiny data builders)
- `tests/integration/conftest.py`: integration-only fixtures (GPU/CLI/workloads)

### Design — Section 4: Filter system

**Key principle**

Filters must work for both CI and developer “partial run” workflows (not CI-only).

**MVP filter dimensions**

- `type`
- `arch`

**Bridge period**

- Keep `quick/standard/comprehensive/full` in `tests/test_categories.yaml` for now.
- **Target state (TheRock):** evolve this file toward the [hipBLAS reference design](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblas/clients/gtest/test_categories.yaml) (tiers, patterns, labels, `execution_settings`, optional GPU exclusions).
- Developer convenience: expose tier presets as one-line `ctest -L` / `pytest -m` equivalents (not a replacement for YAML — the YAML remains the TheRock-facing contract).

### Design — Section 5: CTest / PTest / GoogleTest

**Primary harness**

- CTest + PTest remain the major entry point for CI.

**C++ direction**

- Prefer GoogleTest for C++ tests (potential future expansion).

**Current repo reality (for awareness)**

- Current CTest tests are registered in `projects/rocprofiler-compute/CMakeLists.txt` with
  pytest marker selection and `--junitxml=tests/<name>.xml`.

### Design — Section 6: Migration strategy

**Preferred strategy**

- **Parallel bring-up**: keep current framework alive while the new layout/framework is built in parallel.
- **Sample-first adoption**: for each test category, land **sample template(s)** in the new subsystem before bulk moves — lowers adoption friction and documents conventions in code.
- Selectively incorporate PR `#6189` by cherry-picking the core refactor commit `07b0579`.

**Sample templates (by category)**

| Category | When | Purpose | Example location |
|----------|------|---------|------------------|
| **Unit** | Phase 1 (MVP) | Show mirror-`src/` layout, unit conftest, `@pytest.mark.type("unit")`, no GPU | `tests/unit/.../test_<module>.py` |
| **Integration** | Phase 1 (MVP) | Show feature file, integration conftest, CLI/GPU fixtures, `@pytest.mark.type("integration")` | `tests/integration/test_<feature>.py` |
| **Functional** | Phase 2+ | Golden workload / output validation pattern | `tests/integration/` or future `tests/functional/` |
| **Performance / pressure** | Phase 2+ | Baseline comparison or sustained-load pattern | future `tests/performance/` or marked integration |

Each sample should be **real and runnable** (not empty stubs): registered in CTest, labeled in `test_categories.yaml`, and selectable via `ctest -L` / `pytest -m`. Contributors copy the nearest sample when adding or migrating tests.

**Recommended migration order**

1. Scaffolding + **Unit** and **Integration** samples (parallel with legacy flat tests).
2. Cherry-pick / align layout from PR `#6189` where applicable.
3. Migrate existing tests in waves, using samples as the checklist (imports, markers, conftest, CTest name, YAML entry).
4. Add **Functional** and **Performance** samples when those `type` markers are enabled.
5. CI-green on new layout → **two-week parallel soak** → remove legacy flat tests.

**Hard constraints**

- No `src/` changes.
- No test-logic changes (moves/splits/import fixups only).
- Preserve existing CTest invocation behavior as much as feasible.
- Coverage must not regress.

**Cutover / soak policy**

- When the new layout/framework is **CI-green**, keep the legacy flat tests running in parallel for
  **two additional weeks** (soak period), then remove legacy.

### Design — Section 7: Risks & mitigations

- **Risk: CI breakage / TheRock integration issues**
  - Mitigation: parallel bring-up + two-week soak; keep artifact paths stable.
- **Risk: team disagreement on taxonomy**
  - Mitigation: authoritative definitions + borderline rules; keep MVP minimal (`type`, `arch`).
- **Risk: over-engineering filters too early**
  - Mitigation: MVP dimensions only; postpone other dimensions until after layout is stable.

---

## 8) CDash / dashboards considerations

rocprofiler-compute submits to CDash via `projects/rocprofiler-compute/tools/run-ci.py` (it
generates a CTest dashboard script, runs `ctest_test(...)`, and submits results).

**What to watch during the refactor**

- CTest test names (appear as entries in CDash)
- JUnit XML paths (current pattern uses `--junitxml=tests/<name>.xml`)
- Labels/tier tags used for grouping
- Artifact directories (team indicated this is important)

No `CTestConfig.cmake`/`CTestCustom.cmake` files were found in-repo; any CDash-visible
changes will be driven by how we modify CTest registration and JUnit output locations.

---

## 9) Next discussion prompts for the team

Use these prompts to drive alignment in the team meeting:

- Do we formally **drop “component”** and fold into integration for MVP?
- Confirm MVP is only **two filter dimensions**: `type`, `arch` (everything else later).
- Confirm the two-week **parallel soak** policy after new layout is CI-green.
- Decide what “artifact dirs stable” means concretely (JUnit XML paths, logs, coverage output).
- Agree whether PR `#6189` commit `07b0579` is acceptable as the starting scaffolding (cherry-pick only).
- Confirm **hipBLAS `test_categories.yaml`** as the long-term TheRock reference design (see §6.1) and what must match in the first convergence PR vs later.
- Agree on **sample-first** templates: which real tests become the Unit / Integration / Functional / Performance reference cases in Phase 1.

