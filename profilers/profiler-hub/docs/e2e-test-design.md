# End-to-End Test Suite Design: profiler-hub

**Status:** Design document - not yet implemented
**Last updated:** 2026-05-20
**Scope:** pytest-based functional, schema, cross-version, performance regression, and stress tests

---

## Table of Contents

1. [What "e2e" means for this library](#1-what-e2e-means-for-this-library)
2. [Test taxonomy](#2-test-taxonomy)
3. [Why pytest](#3-why-pytest)
4. [Repository layout](#4-repository-layout)
5. [Test invocation model](#5-test-invocation-model)
6. [Performance regression gating](#6-performance-regression-gating)
7. [Test data and fixtures](#7-test-data-and-fixtures)
8. [Failure surface](#8-failure-surface)
9. [Concrete first-cut test list](#9-concrete-first-cut-test-list)
10. [What this design intentionally does not cover](#10-what-this-design-intentionally-does-not-cover)

---

## 1. What "e2e" means for this library

### The spectrum

"End-to-end" for a library that writes files can mean many things. Ordered from narrowest to widest:

1. **Level 0 - unit test coverage:** Individual methods tested in isolation with mocked or in-memory storage. Already covered by `tests/unit/` via gtest (240 tests). Not the target of this document.

2. **Level 1 - binary smoke:** Build the benchmark binary, execute it against a real SQLite file on disk, assert the process exits 0. Tests the build but not the DB content.

3. **Level 2 - write-read round-trip (chosen rung):** Exercise `writer_t` through its public API to produce a `.db` file, then exercise `reader_t` to read it back and assert that every record inserted comes back with correct values, ordering, and counts. Uses `ctypes` or `cffi` to call the shared library directly from Python, so no C++ test binary is involved. The DB is a standard SQLite file that Python's `sqlite3` module can also query directly for schema assertions.

4. **Level 3 - bench-binary parse:** Execute the `profiler-hub_benchmarks` binary with `--benchmark_format=json`, capture stdout, and parse `items_per_second` to gate performance regressions. This is still Level 2 territory from the library's perspective - the binary is a driver that exercises the same `writer_t` path.

5. **Level 4 - real capture integration:** Run `rocprof-sys-run` against a HIP workload, get a `.db`, read it back. Requires GPU hardware, a working ROCm stack, and the full rocprofiler-systems build. That is downstream integration testing, not library testing. Not the target of this suite.

### Chosen rung: Level 2 + Level 3

The chosen rung is **Level 2 (write-read round-trip via shared library) + Level 3 (bench-binary perf gating)**. Justification:

- **The bug this would have caught** (PR #5478 push-ordering corruption) was a silent data corruption: the writer accepted data without error but stored rows in the wrong order. A unit test that only checks "insert succeeded" would not catch this. A pytest test that writes N records and reads them back in order through `reader_t` would catch it immediately.

- **Direct DB access from Python** is available for free via the standard library `sqlite3` module. This makes schema invariant assertions (column presence, CHECK constraint definitions, FK declarations) trivial to express without writing a C++ test helper.

- **The bench binary** already produces JSON output via `--benchmark_format=json`. Driving it from pytest is seven lines of `subprocess.run`. The alternative (a pure C++ perf test) cannot easily compare against a stored baseline or emit a structured failure report.

- **Level 4 is out of scope** because it requires hardware, a different build configuration, and imposes a multi-minute runtime per test. This suite is designed to run in under 5 minutes on a developer workstation (fast slice under 30 seconds), not as a GPU integration gate.

### The binding strategy

The suite calls `libprofiler-hub.so` from Python using `ctypes`. The public headers are C++, so the binding is generated from the C++ types via two strategies:

- For `storage_t`, `writer_t`, and `reader_t`: use `ctypes.cdll` to load the shared library, then drive the API through a thin C shim (`tests/e2e/shim/phub_shim.cpp`) that exposes plain `extern "C"` functions. The shim is compiled as part of the e2e test setup, not part of the library itself.
- For schema and DB structure assertions: bypass the C++ API entirely and use Python's `sqlite3` module to open the produced file directly.

This avoids generating Python bindings for complex C++ types (the `writer_types` structs are not ABI-stable in the usual sense and use `std::string_view`, making raw ctypes binding fragile). The shim is intentionally minimal: it only exposes what the e2e tests need, not the full API surface.

**Open question:** The shim must be compiled against the same build of `libprofiler-hub.so` that is being tested. The design assumes a `--build-dir` argument (see Section 5) that points to the CMake binary directory. Whether the e2e CI job builds the shim as a CMake custom target or as a `subprocess.run("cmake ...")` step inside `conftest.py` is an implementation choice left for the first PR.

---

## 2. Test taxonomy

The suite is divided into five buckets. Each bucket maps to one or more pytest markers and one or more CI gate stages.

### Bucket A: Functional correctness

**Goal:** Assert that data inserted via `writer_t` can be read back with correct values through `reader_t`.

**What it catches:** The PR #5478 push-ordering bug: 4-vtable-buffer rows were committed in insertion order of the buffer flush, not the order they were pushed. A reader-side assertion on `start_timestamp` ordering would have caught this.

**Shape of tests in this bucket:**
- Write N records to a table. Read them back through `reader_t`. Assert count, field values, and ordering.
- One test per writer entry point: `insert_region_data`, `insert_kernel_dispatch_data`, `insert_memory_copy_data`, `insert_memory_alloc_data`, `insert_pmc_event_data`.
- One test for argument attachment: write a region with args, read it back via `reader_t.get_arguments()`, assert positions and values.

**Markers:** `functional`, `fast`

### Bucket B: Schema invariants

**Goal:** Assert the schema that the library produces on disk matches the contract expected by downstream tooling. Does not require `reader_t` - opens the `.db` directly via `sqlite3`.

**What it catches:** An SDK upstream change to `rocpd_tables.sql` that silently drops a column, changes a CHECK constraint value, or removes a FK declaration.

**Shape of tests:**
- Assert that every expected table is present: `rocpd_metadata`, `rocpd_string`, `rocpd_info_node`, `rocpd_info_process`, `rocpd_info_thread`, `rocpd_info_agent`, `rocpd_info_queue`, `rocpd_info_stream`, `rocpd_info_pmc`, `rocpd_info_code_object`, `rocpd_info_kernel_symbol`, `rocpd_track`, `rocpd_event`, `rocpd_arg`, `rocpd_pmc_event`, `rocpd_region`, `rocpd_sample`, `rocpd_kernel_dispatch`, `rocpd_memory_copy`, `rocpd_memory_allocate`.
- Assert CHECK constraint definitions are present and not weakened. The schema currently has: `rocpd_info_agent.type IN ('CPU', 'GPU')`, `rocpd_info_pmc.target_arch IN ('CPU', 'GPU')`, `rocpd_info_pmc.value_type IN ('ABS', 'ACCUM', 'RELATIVE')`, `rocpd_info_code_object.storage_type IN ('FILE', 'MEMORY')`, `rocpd_memory_allocate.type IN ('ALLOC', 'FREE', 'REALLOC', 'RECLAIM')`, `rocpd_memory_allocate.level IN ('REAL', 'VIRTUAL', 'SCRATCH')`. Read these from `sqlite_master WHERE type='table'` and assert they match expected patterns.
- Assert FK declarations are present (even though `PRAGMA foreign_keys` is set to ON by the writer - the declarations should be in the schema for downstream readers that do enable FK enforcement).
- Assert `rocpd_metadata.schema_version = '3'` for schema_v3 databases.

**Markers:** `schema`, `fast`

### Bucket C: Cross-version compatibility

**Goal:** Assert that a database produced by the current library can be read by the current `reader_t`, and vice versa. Also assert the forward-compat contract when schema_v4 arrives.

**What it catches:** A writer that quietly upgrades schema without bumping the version tag, breaking downstream readers that depend on stable schema_v3 structure.

**Shape of tests:**
- The existing `tests/unit/rocpd.db` (a real capture from `bit_extract` on a specific host, checked in at `tests/unit/`) serves as the golden v3 reader fixture. The current `reader_t` must be able to open it and return expected values (this is already tested by `tests/unit/reader_test.cpp`). The e2e analog reads the same file via Python's `sqlite3`, cross-checks counts, and calls `reader_t` via the shim to assert counts match.
- A "writer-produces-readable" round-trip: write synthetic data at the current schema version, close the writer, open a new reader, assert the reader sees the same records. This is distinct from the unit tests because it uses a full `storage_t` lifetime (create, populate, flush, destroy) rather than an in-process in-memory backend.
- Schema version assertion: `SELECT value FROM rocpd_metadata WHERE tag = 'schema_version'` must return `'3'` for all databases produced by the current writer.

**Forward compat note (open question):** When schema_v4 lands, this bucket should gain a test that opens a v3 DB with a v4 reader and asserts graceful degradation (fields not present in v3 return None/nullopt rather than crashing). The exact API for this is not yet defined since v4 is not implemented.

**Markers:** `compat`, `fast`

### Bucket D: Performance regression

**Goal:** Assert that the writer hot path does not silently regress below a throughput floor derived from measured baseline numbers.

**What it catches:** Accidental reversion of any of the 8 patches from the recent perf sprint (fmt-lazy bind, prepared transactions, entity_utility cache, vtable buffer pattern from PR #5454, SQLITE_THREADSAFE=2).

**Shape of tests:**
- Drive the `profiler-hub_benchmarks` binary with `--benchmark_filter` and `--benchmark_format=json`. Parse `items_per_second` from stdout.
- Compare against stored baselines (see Section 6 for baseline management strategy).
- One perf gate per hot entry point: `kernel_dispatch`, `memory_copy`, `memory_alloc`, `region`, `pmc_event`. All at the 100K count level (not 1M, to keep runtime bounded).

**Markers:** `perf`, not `fast` (total runtime for the perf slice is roughly 3-5 minutes on a debug build)

### Bucket E: Stress and robustness

**Goal:** Assert correctness under concurrent or adversarial conditions that unit tests do not exercise.

**What it catches:**
- The `multiple_backends_independent` WAL sidecar cleanup bug: if two backends share a path prefix, teardown of one can delete WAL/SHM files belonging to the other. The e2e version exercises parallel Python subprocesses, each writing to a distinct path, and asserts that both files are readable after teardown.
- Large arg payloads: insert a region with 100 arguments, read it back, assert all 100 are recoverable with correct positions.
- Long-running captures: insert 500K kernel dispatch records in a single writer session, assert the final row count is exactly 500K.

**Markers:** `stress`, not `fast` (the 500K insertion test takes 10-30 seconds depending on build type)

---

## 3. Why pytest

This section gives concrete reasons, not generic praise.

**Reason 1: Subprocess + JSON parsing is the natural driver for the bench binary.**

The bench binary already emits `--benchmark_format=json`. The perf regression gate reads that JSON and compares it to a stored baseline. In Python this is:

```python
import subprocess, json
result = subprocess.run([bench_binary, "--benchmark_format=json", ...], capture_output=True)
data = json.loads(result.stdout)
```

The same logic in a shell script requires `jq`, which is not universally installed on CI runners. The same logic in C++ requires either linking a JSON library (already a dep) or parsing manually. pytest is the cleanest driver for "run a process, parse its output, fail with a useful message."

**Reason 2: The DB is a standard SQLite file. Python's `sqlite3` is in the stdlib.**

Schema assertions - "does this CHECK constraint exist?", "does this FK reference the right table?" - are SQL queries against `sqlite_master`. No C++ code is needed. In gtest, this would require either a raw SQLite C API call (verbose) or a custom query helper (already exists in `writer_test.cpp` but is not part of the library's public API and should not be). In Python: `conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", ("rocpd_info_agent_uuid",)).fetchone()`.

**Reason 3: Fixture lifecycle maps naturally to `tmp_path` and `yield`-fixtures.**

Each test needs an isolated DB directory. pytest's `tmp_path` fixture provides a unique temporary directory per test automatically. `yield`-based fixtures handle setup/teardown without the C++ RAII boilerplate. The parallel WAL cleanup test needs two writers running concurrently - this is trivial in pytest (`concurrent.futures.ThreadPoolExecutor`) and would require a second binary or manual `std::thread` juggling in gtest.

**Reason 4: Parametrize for the cross-version and schema tests.**

The schema test needs to assert the same set of invariants against two different DBs: a DB produced by the current writer, and the golden `rocpd.db` checked in at `tests/unit/`. `@pytest.mark.parametrize` lets the same assertion run against both without duplicating the test body.

**Reason 5: pytest markers implement the fast/perf/stress split without a separate test binary.**

ctest labels (used by the existing unit test suite via `PROPERTIES LABELS "unit"` in `tests/unit/CMakeLists.txt`) are per-binary. A single ctest invocation runs all tests in the binary. pytest markers allow `pytest -m fast` to select a subset of tests from a mixed file without a separate compilation target.

**Reason 6: Failure reporting with `pytest-rich` or `pytest-pretty` gives structured diff output.**

When a perf regression fires, the failure message can include: the benchmark name, the measured value, the baseline value, the percentage drop, and a recommendation. gtest's `FAIL()` message supports custom strings but requires manual formatting. pytest's parametrize + custom failure formatting produces a table that CI can annotate inline.

**Why not a pure shell script?**

Shell scripts are fine for "run binary, check exit code." They are not fine for "parse JSON baseline, compute percentage, compare against threshold with variance, format a diff." The maintenance cost of a non-trivial shell script is high and it is not composable.

**Why not extend the gtest suite?**

gtest excels at testing library internals. The e2e tests need to drive multiple processes, read files from different sources, and implement stateful baseline comparison. None of that maps cleanly to gtest fixtures. More importantly, the DB schema tests need to open files that the library produced and was then closed - a pattern that does not fit the existing `writer_test` fixture model, which requires access to internal `profiler_hub::data_storage::sqlite_backend` (a non-public header).

---

## 4. Repository layout

```
profilers/profiler-hub/
├── tests/
│   ├── unit/                           existing gtest suite (240 tests)
│   │   ├── CMakeLists.txt
│   │   ├── rocpd.db                    golden DB for reader_test.cpp
│   │   └── ...
│   ├── benchmarks/                     existing Google Benchmark suite
│   │   ├── CMakeLists.txt
│   │   └── ...
│   └── e2e/                            new pytest suite (this document)
│       ├── conftest.py                 session-scope fixtures: bin_dir, shim_lib
│       ├── pytest.ini                  or pyproject.toml [tool.pytest.ini_options]
│       ├── baselines/
│       │   └── perf_baselines.json     checked-in baseline numbers (see Section 6)
│       ├── golden/
│       │   └── rocpd_v3.db             copy of tests/unit/rocpd.db for reader compat tests
│       ├── shim/
│       │   ├── phub_shim.cpp           thin extern-C shim over the public C++ API
│       │   └── CMakeLists.txt          builds phub_shim.so, depends on profiler-hub target
│       ├── test_functional.py          Bucket A: write-read round-trip per table
│       ├── test_schema.py              Bucket B: schema invariants via sqlite3 module
│       ├── test_compat.py              Bucket C: cross-version reader round-trip
│       ├── test_perf.py                Bucket D: perf regression gates
│       └── test_stress.py              Bucket E: WAL cleanup, parallel writers, large payloads
└── docs/
    └── e2e-test-design.md              this document
```

### Coexistence with the ctest harness

The e2e tests do not touch `CMakeLists.txt` beyond adding `tests/e2e/shim/CMakeLists.txt` as a subdirectory under `tests/e2e/`. The shim library is a separate CMake target (`phub_shim`) that depends on `profiler-hub`. It is built as part of the normal `cmake --build` step so the shim is always available when tests run.

The existing CI workflow (`profiler-hub-ci.yml`) runs `ctest --output-on-failure` after the build. The e2e pytest suite runs as a separate step after ctest, using the binary dir produced by the same build:

```yaml
- name: Run e2e tests (fast + schema + compat)
  working-directory: profilers/profiler-hub
  run: |
    pip install pytest pytest-xdist
    pytest tests/e2e/ -m "fast or schema or compat" \
      --bin-dir build/bin --build-dir build \
      --junitxml=build/e2e-results.xml
```

The `--bin-dir` and `--build-dir` arguments are registered in `conftest.py` via `pytest_addoption`.

### Shared fixtures

Fixtures shared across test files live in `conftest.py`:

- `bin_dir(request)`: returns `Path` to the CMake binary output directory (`build/bin/`). Resolved from `--bin-dir` CLI argument, defaulting to `<repo_root>/profilers/profiler-hub/build/bin`.
- `build_dir(request)`: path to the CMake build tree root, used to find `phub_shim.so`.
- `bench_binary(bin_dir)`: path to `profiler-hub_benchmarks`. Asserts it exists. Session scope.
- `shim_lib(build_dir)`: path to `phub_shim.so`. Session scope.
- `writer_session(shim_lib, tmp_path)`: creates a `storage_t + writer_t` via the shim, yields handles, calls cleanup. Function scope (isolated per test).
- `golden_db()`: returns `Path` to `tests/e2e/golden/rocpd_v3.db`. Session scope.
- `baseline_numbers()`: loads `tests/e2e/baselines/perf_baselines.json`. Session scope.

### Where `rocpd.db` lives

The `tests/unit/rocpd.db` file (a real bit_extract capture, checked in as a golden reader fixture) is referenced by `reader_test.cpp` via a compile-time `ROCPD_DB_PATH` definition. For the e2e tests, a copy lives at `tests/e2e/golden/rocpd_v3.db` to make the e2e directory self-contained and avoid coupling to a path that is only available after a CMake configure. This is intentional duplication - the golden DB is small (the existing file in `tests/unit/` is a SQLite file from a brief HIP workload capture).

---

## 5. Test invocation model

### Local developer workflow

The inner loop for a developer who just landed a patch:

```bash
# After building:
cmake --build build --parallel $(nproc)

# Fast check (under 30 seconds, runs functional + schema + compat):
pytest tests/e2e/ -m "fast or schema or compat" \
  --bin-dir build/bin --build-dir build -v

# Perf regression check (2-5 minutes, runs only perf bucket):
pytest tests/e2e/ -m perf \
  --bin-dir build/bin --build-dir build -v

# Full e2e suite including stress (5-15 minutes):
pytest tests/e2e/ --bin-dir build/bin --build-dir build -v
```

The fast slice (`-m "fast or schema or compat"`) should complete in under 30 seconds on a development machine. This is the slice a developer runs 20x a day. The perf slice is run when landing a perf-sensitive patch. The stress slice is run before a PR merge.

### Binary location strategy

The test binary location is resolved in the following priority order:

1. `--bin-dir` CLI argument
2. `PROFILER_HUB_BIN_DIR` environment variable
3. Default: `<test_root>/../../build/bin` (relative to `tests/e2e/`)

The `bench_binary` fixture asserts that the binary exists before any perf test runs, and skips the test with `pytest.skip()` if not found (rather than failing). This allows the fast slice to run on machines where the bench binary is not present (e.g., a documentation-only CI job).

Similarly, the shim library is resolved from `--build-dir` and the tests that require it skip if not found.

### CI integration

The existing `profiler-hub-ci.yml` workflow triggers on PR and push to `develop` for paths under `profilers/profiler-hub/`. The e2e tests extend this workflow as additional steps.

**What runs on every PR (fast gate, < 2 min added):**
- `pytest -m "fast or schema or compat"`: all Bucket A, B, C tests.
- Bucket D and E are skipped on PR because (a) perf numbers are noisy on GitHub runners and (b) stress tests are too long.

**What runs on push to `develop` (merge gate, ~10 min added):**
- `pytest -m "fast or schema or compat or stress"`: adds Bucket E.

**What runs on a nightly (separate workflow `profiler-hub-nightly.yml`, not yet created):**
- `pytest -m perf`: Bucket D perf regression tests. Requires a dedicated runner with CPU frequency pinning. Nightly because perf numbers on shared GitHub runners (which have `cpu_scaling_enabled: true` as confirmed by the current bench output) are too noisy for a reliable gate.

**Open question:** The current CI runners are GitHub-hosted (`ubuntu-22.04`, `ubuntu-24.04`). The bench output shows `cpu_scaling_enabled: true`, which means absolute `items_per_second` numbers from CI are unreliable. Two options: (a) run the perf gate only on a self-hosted runner with frequency pinning, (b) use ratio-based thresholds (ratio between two benchmark variants run back-to-back on the same machine, which cancels out frequency noise). Section 6 discusses this further. This is a concrete open question that cannot be resolved from the repo alone - the CI runner configuration needs to be verified.

---

## 6. Performance regression gating

This is the hardest part of the design. The following decisions are opinionated and justified below.

### Baseline storage

Baselines live in `tests/e2e/baselines/perf_baselines.json`, checked into the repository. The file is organized by benchmark name and CPU family:

```json
{
  "schema_version": 1,
  "benchmarks": {
    "writer_fixture/kernel_dispatch/100000": {
      "x86_64_epyc": {
        "items_per_second_floor": 800000,
        "note": "floor set at 80% of 1.03M measured 2026-05-20 on RSN-SWSLAB-01-L Release build"
      },
      "generic": {
        "items_per_second_floor": 400000,
        "note": "conservative floor for unknown CPU - fails open if CPU family not recognized"
      }
    }
  }
}
```

**Why checked-in JSON, not a per-host file?**
Checked-in JSON means the baseline is part of the code review for every perf-changing PR. If a patch genuinely improves throughput, the developer updates the baseline in the same commit. If a patch regresses throughput, the CI catches it without any human intervention.

**Why per-CPU-family keys?**
The current dev machine is `RSN-SWSLAB-01-L` with 32 CPUs at 5756 MHz (EPYC class). GitHub runners are typically lower-clocked Xeon or EPYC with different cache geometries. Using a CPU-family key (detected from `/proc/cpuinfo` or `platform.processor()` in `conftest.py`) allows CI runners to use a different floor than the dev machine. Unrecognized CPUs fall through to `"generic"` which uses a conservative floor that should pass on any reasonable modern x86_64.

**Why `items_per_second_floor`, not a percentage threshold?**

A percentage threshold ("must not regress more than 20%") requires a previous measurement to compare against. That means storing previous measurements somewhere - either the CI artifact store or a separate file. The floor approach is simpler: the floor is the minimum acceptable throughput, set to roughly 70-80% of the current measured value on the reference machine. It does not require a "previous run" comparison.

The floor approach has one weakness: it does not catch a regression that stays above the floor but is still significantly slower than the previous run. This is acceptable because the floor is set conservatively enough that any serious regression (say, reverting the `prepared transactions` patch) would drop below it. Minor regressions (5-10%) are not worth the false-positive noise.

### Threshold policy

- Floor = 70% of the Release-build baseline on the reference machine. This provides a 30% buffer against noise and minor performance variation.
- The current Debug-build numbers (measured 2026-05-20 from `build-release/`, which reports `library_build_type: debug` despite the directory name) are:

  | Benchmark | Items/sec |
  |---|---|
  | kernel_dispatch/100K | 1.03M |
  | memory_copy/100K | 1.15M |
  | memory_alloc/100K | 0.44M |
  | region/100K | 1.51M |
  | pmc_event/100K | 1.61M |

  The Release build (with `-O2 -DNDEBUG`) will be significantly faster. The perf CI job must be run against a Release build. The test asserts `library_build_type == "release"` and skips with a warning (not a failure) if the binary is a debug build. This handles the "accidental debug run" case gracefully.

- The variance-aware threshold (2 sigma rule) is explicitly rejected for the first iteration. Variance estimation requires multiple runs, which multiplies the already-long bench runtime. The 30% buffer against the floor is a simpler proxy for "this is clearly wrong" without needing statistical machinery.

### Noise control

- **CPU frequency pinning:** Run bench with `taskset -c 0` to pin to a single core. This alone reduces run-to-run variance on non-scaling workloads from ~20% to ~5%. `conftest.py` checks whether `taskset` is available and applies it if so.
- **Governor check:** In `conftest.py`, before running perf tests, check `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`. If it reads `powersave` rather than `performance`, emit a `warnings.warn` but do not fail. (Failing would break developer workflows on laptops.)
- **Warmup:** The bench binary already handles warmup internally (Google Benchmark runs a warmup iteration before the measured iteration). No additional warmup is needed at the pytest level.
- **Repetitions:** Each benchmark is run once (`--benchmark_repetitions=1`) in the pytest-driven invocation. Google Benchmark with a single iteration is deterministic enough for floor-based gating. If variance becomes a problem, increase to 3 repetitions and use the median.
- **Aggregation:** Use `items_per_second` directly from the JSON output. For repetitions > 1, use the median across repetitions.

### Debug build handling

Google Benchmark prints `"library_build_type": "release"` or `"debug"` in the JSON context block. The test reads this field and:

- If `debug`: emit `pytest.warns` with the message "Benchmark binary is a debug build. Absolute numbers unreliable. Floor thresholds reduced by 50% for this run." Then multiply all floor values by 0.5 before comparison. This allows the fast-feedback inner loop to still catch gross regressions even on a debug build, while not producing false positives from the expected debug/release gap.
- If `release`: use the floor values as stored.

This preserves the "land patch + run pytest + see green/red" inner loop on a debug build while ensuring that the full perf gate (nightly, pre-merge) runs on release builds.

### First-time-run and new-CPU-host story

When the test runs on a host whose CPU family is not in `perf_baselines.json`:

1. Detect unknown CPU family.
2. Fall through to `"generic"` floor, which is set conservatively low.
3. Emit a warning: "Unrecognized CPU family. Using generic floor. Consider adding a CPU-family entry to baselines/perf_baselines.json."
4. Do not fail the test.

This is "fail open" for new hosts. The rationale: a developer on a new machine should not have their environment blocked by a missing baseline. The CI runner (which is known) will use a specific floor.

**Populating a new entry:** The pytest test has a `--update-baselines` flag. When passed, instead of comparing against the floor, it runs the benchmarks, computes 70% of the median `items_per_second`, and updates `perf_baselines.json` with the new entry. The developer then commits the updated baseline file. This is the correct workflow for adding a new machine.

---

## 7. Test data and fixtures

### Golden DBs for reader tests

`tests/e2e/golden/rocpd_v3.db` is a copy of `tests/unit/rocpd.db`. It is a real SQLite database produced by a `bit_extract` HIP workload on host `smci350-zts-gtu-c14-05` (as confirmed by the hostname value in `reader_test.cpp`). It contains:

- 1 node
- 1 process (pid 67979, command `./bit_extract`)
- 4 threads
- Known kernel dispatch and memory copy data

This file is checked in as a binary blob. It is small (inspect with `ls -lh tests/unit/rocpd.db`; HIP traces for short workloads are typically under 1 MB). It serves as the ground truth for reader compatibility tests.

**Refresh procedure:** If the schema changes (v3 to v4 transition), the golden DB must be regenerated. The procedure:
1. Build the new schema writer.
2. Run `./bit_extract` (or any HIP workload) with profiler-hub capture enabled.
3. Copy the produced `.db` to `tests/e2e/golden/rocpd_v3.db` (or `rocpd_v4.db` for a new schema version).
4. Update `test_compat.py` assertions to match the new file's expected counts.
5. Commit both the new DB and the updated test.

### Synthetic workloads

For Bucket A (functional correctness) tests, synthetic data is generated by Python factories modeled directly on the `make_*` functions in `tests/benchmarks/writer_bench.cpp`. The factories live in `tests/e2e/conftest.py` and produce the same field values, making it easy to cross-reference a failing test against the bench code.

The benchmark benchmark fixture (`writer_fixture::setup_schema`) establishes: node_id=1, pid=1000, tid=1001, one GPU agent ("GPU", type_index=0), one code object (id=1), one kernel symbol (id=1), PMC counters `SQ_WAVES`, `SQ_INSTS`, `TA_BUSY`, tracks `gpu_kernel`, `gpu_memcpy`, `cpu_sample`, `amd_smi`. The e2e functional tests use the same canonical values so that a hand-constructed `.db` can be compared against one produced by the bench binary.

### cpu-sampler as a canned trace source

`~/work/cpu-sampler` (`cpu-sampler` project noted in project memory as a standalone `perf_event_open` CPU sampler POC) is not a suitable source of canned HIP traces. It produces CPU sampling data only, not the GPU kernel dispatch / memory copy structure that profiler-hub writes. It is mentioned here to close the question: it does not help for e2e test data.

The simplest "realistic small HIP capture" that does not require hardware is the existing `rocpd.db` in `tests/unit/`. For the stress tests that need a large synthetic dataset, the bench binary itself is the driver (see Bucket D).

### Per-test isolated DB directories

Every test that creates a `.db` file uses `tmp_path` (pytest built-in, provides a unique `Path` per test invocation). The `writer_session` fixture creates `storage_t` in `tmp_path / "test.db"`. Tests never share a DB file. This prevents the WAL sidecar cleanup bug from causing test-to-test interference (which was the root cause of the `multiple_backends_independent` flake in the unit test suite).

---

## 8. Failure surface

When a test fails, the failure message must contain enough information for a developer to re-run and bisect without additional context.

### Functional test failure

```
FAILED tests/e2e/test_functional.py::test_kernel_dispatch_row_ordering

AssertionError: Row ordering mismatch in rocpd_kernel_dispatch after flush.
Expected start_timestamp ordering: [0, 1000, 2000, 3000, ..., 9999000] (ascending)
Actual start_timestamp ordering:   [0, 2000, 1000, 3000, ..., 9999000]
First mismatch at index 1: expected 1000, got 2000.

DB path: /tmp/pytest-of-user/pytest-42/test_kernel_dispatch_row_ordering0/test.db
Re-run: pytest tests/e2e/test_functional.py::test_kernel_dispatch_row_ordering \
  --bin-dir build/bin --build-dir build -s
Bisect tip: this ordering bug was introduced in PR #5478. Check vtable buffer push order.
```

The "DB path" line is critical - the developer can open the `.db` with `sqlite3` or `sqlitebrowser` to inspect the raw data.

### Perf regression failure

```
FAILED tests/e2e/test_perf.py::test_perf_kernel_dispatch_floor[100K]

Performance regression: kernel_dispatch/100K below floor.
  Measured:  780,432 items/sec
  Floor:     800,000 items/sec (x86_64_epyc, Release build)
  Gap:       -2.5% below floor
  Build:     release (correct)
  CPU:       x86_64_epyc
  Scaling:   performance governor (no noise warning)

Benchmark name: writer_fixture/kernel_dispatch/100000/iterations:1
Raw JSON: /tmp/pytest-of-user/pytest-42/bench_output_42.json

Re-run the benchmark manually:
  taskset -c 0 build/bin/profiler-hub_benchmarks \
    --benchmark_filter="kernel_dispatch/100000" \
    --benchmark_format=json --benchmark_repetitions=3

To update baseline:
  pytest tests/e2e/test_perf.py -m perf \
    --bin-dir build/bin --build-dir build --update-baselines
```

The raw JSON path allows the developer to inspect the full benchmark context (cache sizes, load average, num CPUs) to diagnose whether the regression is real or environmental.

### Schema invariant failure

```
FAILED tests/e2e/test_schema.py::test_check_constraints_present[rocpd_info_agent]

Schema CHECK constraint missing or changed for rocpd_info_agent.type.
Expected: CHECK ("type" IN ('CPU', 'GPU'))
Found in sqlite_master: CHECK ("type" IN ('CPU', 'GPU', 'NPU'))

This means the schema was extended without bumping schema_version. Either:
  1. Update the test to accept 'NPU' if this is intentional, or
  2. Bump schema_version in rocpd_tables.sql and update golden/rocpd_v3.db.

DB path: /tmp/pytest-of-user/pytest-42/test_check_constraints0/test.db
```

### WAL cleanup failure

```
FAILED tests/e2e/test_stress.py::test_wal_cleanup_independent_backends

WAL sidecar interference: database at path2 was corrupted or missing after path1 teardown.
  path1: /tmp/.../backend_0/db.db
  path2: /tmp/.../backend_1/db.db
  Expected path2 to be readable. Got: OperationalError: unable to open database file.

This reproduces the multiple_backends_independent flake. The WAL/SHM sidecars
for path2 were likely deleted during path1 cleanup.
See: tests/unit/database_test.cpp::multiple_backends_independent for the unit test analog.
```

---

## 9. Concrete first-cut test list

The following 14 named test functions form the bootstrap implementation. They cover at least one of each category.

### Bucket A: Functional correctness

**`test_functional.py::test_kernel_dispatch_smoke`**
Write 10 kernel dispatch records with sequential `start_timestamp` values (0, 1000, ..., 9000) and a fixed kernel symbol. Read back via `reader_t.get_events_for_track()`. Assert: count == 10, `start_timestamp` values are in ascending order, the kernel name matches the registered symbol.

**`test_functional.py::test_memory_copy_smoke`**
Write 5 memory copy records with varying `size` values. Read back via reader. Assert: count == 5, sizes match what was written.

**`test_functional.py::test_memory_alloc_smoke`**
Write 5 memory alloc records with `type="ALLOC"` and `level="REAL"`. Read back via reader. Assert: count == 5, type and level fields round-trip correctly.

**`test_functional.py::test_region_smoke`**
Write 5 region records with distinct names (`"hip_api_0"` through `"hip_api_4"`). Read back via reader. Assert: count == 5, names match.

**`test_functional.py::test_pmc_event_smoke`**
Write 5 PMC event records against `SQ_WAVES` with values `[100.0, 200.0, 300.0, 400.0, 500.0]`. Read back via reader. Assert: count == 5, values match.

**`test_functional.py::test_arg_round_trip`**
Write a region record with 3 args (positions 0, 1, 2; types `"int"`, `"char*"`, `"size_t"`). Call `reader_t.get_arguments()` on the resulting event. Assert: 3 args returned, positions and types match.

### Bucket B: Schema invariants

**`test_schema.py::test_expected_tables_present`**
Create a minimal writer session (write one node + one process + flush). Open the resulting `.db` with Python's `sqlite3`. Query `sqlite_master WHERE type = 'table'`. Assert that all 20 expected table name prefixes are present (with uuid suffix substituted via `LIKE 'rocpd_info_node%'`).

**`test_schema.py::test_check_constraints_present`**
Parametrized over 6 tables that have CHECK constraints (see Bucket B in Section 2). For each: query `sqlite_master`, extract the `sql` column, assert the expected CHECK clause is present as a substring. Any change to the constraint definition (e.g., adding `'NPU'` to the agent type enum) will fail this test and force an explicit decision.

**`test_schema.py::test_foreign_key_declarations`**
For `rocpd_kernel_dispatch` (the most FK-heavy table, with 8 FKs), assert that the `sqlite_master` SQL contains all 8 `REFERENCES` clauses. This does not verify FK enforcement (which is a runtime PRAGMA) but verifies the schema contract that downstream readers may rely on.

### Bucket C: Cross-version compatibility

**`test_compat.py::test_golden_db_readable`**
Open `tests/e2e/golden/rocpd_v3.db` via the `reader_t` shim. Assert: `get_all_nodes()` returns 1 node, `get_all_processes()` returns 1 process with pid=67979, `get_all_threads()` returns 4 threads. These values are hardcoded from `tests/unit/reader_test.cpp` and must not change for any schema-v3 compatible reader.

**`test_compat.py::test_writer_produces_readable_db`**
Full lifecycle: write a full synthetic session (node, process, threads, agents, code object, kernel symbol, 100 kernel dispatch records) into a temp `.db`. Close the writer. Open the same file with a new `reader_t` instance. Assert: `get_event_counts().kernel_dispatches == 100`.

### Bucket D: Performance regression

**`test_perf.py::test_perf_kernel_dispatch_floor`**
Run the bench binary with `--benchmark_filter="writer_fixture/kernel_dispatch/100000"` and `--benchmark_format=json`. Assert `items_per_second >= floor` from `perf_baselines.json`. Skips if build type is `debug` (emits warning, uses reduced floor) or if binary not found. Requires `perf` marker.

### Bucket E: Stress and robustness

**`test_stress.py::test_wal_cleanup_independent_backends`**
Create two writers in the same Python process, each with a distinct db path in `tmp_path`. Write 100 records to each. Close writer 1. Assert writer 2's DB is still readable. Close writer 2. Assert both DBs exist and have 100 rows. This reproduces the `multiple_backends_independent` cleanup bug in a pytest-native form.

**`test_stress.py::test_large_record_count`**
Write 500K kernel dispatch records in a single writer session. Assert the final row count (queried via `sqlite3`) is exactly 500,000. This catches any silent data loss (dropped batches, transaction rollbacks) that only appears at scale.

---

## 10. What this design intentionally does not cover

### Not covered: Unit test duplication

The 240 existing gtest tests cover individual method behavior with in-process assertions. The e2e suite does not replicate these. A test like `register_node_info_duplicate_is_ignored` belongs in the gtest suite (it tests deduplication logic) and would be redundant in the e2e suite.

The demarcation line: if the assertion requires access to `profiler_hub::data_storage` internal headers (which the e2e tests cannot import), it stays in gtest. If the assertion can be expressed against the public API or the SQLite file on disk, it can live in either place, and should be in the e2e suite if it also tests behavior observable from outside the process.

### Not covered: Individual insert query builder correctness

`tests/unit/insert_query_builders_test.cpp` and `tests/unit/table_insert_query_test.cpp` test the SQL query generation logic in isolation. The e2e tests do not duplicate this. They assume the query builders produce correct SQL; if they do not, the functional correctness tests will catch the result (wrong data in the DB) rather than the cause.

### Not covered: Benchmark binary correctness

The bench binary is a driver, not a library under test. The e2e tests use it as a black box: run it, parse the throughput, gate against the floor. They do not assert that the bench binary inserts the "correct" data - the data it inserts is synthetic and designed for throughput measurement, not semantic correctness.

### Not covered: Full rocprofiler-systems integration

A test that runs `rocprof-sys-run` against a real HIP workload and asserts the produced `.db` structure is a Level 4 integration test (see Section 1). This requires a GPU, a working ROCm stack, the full rocprofiler-systems build (not just profiler-hub), and runtime environment setup. It belongs in the rocprofiler-systems integration test suite, not in the profiler-hub standalone test suite.

### Not covered: Cross-host / cross-OS compatibility

The e2e tests run on the host where the library was built. They do not attempt to transfer a `.db` produced on Linux to a macOS host and read it back. SQLite is byte-order-neutral and the DB file format is portable, but the library itself (and the shim) are platform-specific binaries. Cross-OS compat testing belongs in the build matrix (which already covers `ubuntu-22.04` and `ubuntu-24.04`), not in the e2e test logic.

### Not covered: rocprofiler-sdk-rocpd schema path

When `USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=1`, the schema SQL is sourced from the SDK package rather than the vendored `source/data_storage/schema/` files. The e2e tests currently assume the vendored schema. Testing the SDK schema path requires the SDK installed on the CI runner, which is a separate CI job (`profiler-hub-install.yml` in the existing workflows). This is flagged as an open question for a future design revision.

---

## Appendix A: Open questions

1. **CI runner CPU scaling:** The current CI runners have `cpu_scaling_enabled: true`. The perf gate is unreliable on these runners. Is there a self-hosted runner available with `performance` governor? If not, should the perf gate be limited to the nightly workflow only?

2. **Shim build strategy:** Should `phub_shim.so` be built by CMake as a custom target (cleanest for CI) or by `subprocess.run` inside `conftest.py` (simpler for developers who do not have CMake on PATH)? The CMake approach requires updating `tests/e2e/shim/CMakeLists.txt` and `tests/CMakeLists.txt`. The subprocess approach is self-contained but fragile.

3. **SDK schema path:** When `USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=1`, the schema originates from the SDK. Should the schema invariant tests run against both code paths, or only the vendored path? The CI workflow builds with the vendored path only (no SDK installed on runners), so the SDK path is currently untested by any automated suite.

4. **v4 forward compat:** Schema v4 does not yet exist. When it does, the `test_compat.py` bucket should gain a test that opens a v3 DB with a v4 reader. The API contract for this (graceful degradation vs. exception) is not yet defined.

5. **Windows / macOS portability of the shim:** `phub_shim.so` uses Linux `dlopen` conventions. If the CI matrix ever gains Windows or macOS runners, the shim and the `conftest.py` binary discovery logic need platform-conditional paths.

6. **Python binding strategy review:** The shim approach (compile a thin `extern "C"` layer) works but adds a compilation dependency. An alternative is to use SWIG or pybind11 to generate a proper Python extension. This is heavier up front but more maintainable as the API grows. The shim approach is recommended for the first PR; revisit if the shim grows beyond ~200 lines.

---

## Appendix B: Dependency summary

| Dependency | Used for | Already in CI? |
|---|---|---|
| Python 3.8+ | pytest runtime | Yes (ubuntu-22.04+) |
| pytest | test runner | No - add `pip install pytest` step |
| pytest-xdist | parallel test execution for stress tests | No - add `pip install pytest-xdist` |
| sqlite3 (Python stdlib) | schema and DB content assertions | Yes (stdlib) |
| subprocess (stdlib) | driving bench binary | Yes (stdlib) |
| GCC/Clang | building phub_shim.so | Yes |
| libprofiler-hub.so | linked by phub_shim.so | Yes (built by CI) |
| profiler-hub_benchmarks | perf regression gating | Yes (built by CI with PROFILER_HUB_BUILD_BENCHMARKS=ON) |
| taskset | CPU pinning for perf tests | Available on Ubuntu, optional |

No new system packages are required beyond what the existing CI workflow installs (`cmake`, `gcc/clang`, `libsqlite3-dev`, `libbenchmark-dev`).
