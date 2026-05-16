# Python Test Suite Refactor Plan

## 1. Goal

Reorganize `projects/rocprofiler-compute/tests/` so that:

1. **Unit tests** live under `tests/unit/`, mirror the `src/` directory tree, and
   a test for `src/<path>/<module>.py` lives at `tests/unit/<path>/test_<module>.py`.
2. **Integration tests** live under `tests/integration/`, are organized one
   file per *feature* (not per mode), and named `test_<feature>.py`.
3. The mapping "which file tests what" is obvious from the path alone, so a
   contributor can find the test for a source change without grepping.

**Hard constraints:**

- **No `src/` changes.**
- **No test-logic changes.** Assertions, parametrize sets, fixture
  bodies, mock targets, and skip conditions are preserved byte-for-byte.
  Allowed edits: file moves, splits, renames, import-path fixups, and
  CMake/YAML re-wiring.
- Each CTest invocation runs the same pytest args, markers, `-n`
  parallelism, and coverage flags as before.
- The four-tier (`quick`/`standard`/`comprehensive`/`full`)
  categorization is unchanged; only the CTest *names* in
  `test_categories.yaml` move to match the new file names.

## 2. Definitions

**Unit test** — in-process, no GPU, no `rocprof-compute` subprocess, no
network. May import any module from `src/`. Allowed I/O: `tmp_path`,
`monkeypatch`, `unittest.mock`, in-memory fixtures, pre-baked tiny CSVs
or SQLite files built inside the test. Should run on any developer
laptop without ROCm installed.

**Integration test** — drives the user-visible CLI via the
`binary_handler_profile_rocprof_compute` /
`binary_handler_analyze_rocprof_compute` fixtures (in-process `main()` or
`subprocess` to `./rocprof-compute.bin`), or shells out to `rocminfo` /
`rocprof-compute -s`, or otherwise depends on a real GPU / pre-captured
workload as a black box. May still be heavily parametrized.

**Borderline rule.** When a single existing test file mixes both kinds
(e.g. `test_utils.py` is mostly unit tests but contains
`test_list_metrics` and `test_list_blocks` which drive
`binary_handler_analyze_rocprof_compute`), the integration tests move
to `tests/integration/`, the unit tests are split by source module
under `tests/unit/`. Splitting is by *section banner* / *imported
source module*, never by re-deciding what a test asserts.

## 3. Target layout

```
tests/
├── conftest.py                   # tier-agnostic fixtures (see §3b)
├── common.py                     # shared helpers (unchanged)
├── generate_*.py, *_utils.py     # generators stay at root
├── CMakeLists.txt                # C++ sample binaries
├── workloads/                    # captured profiling outputs
│
├── unit/
│   ├── __init__.py
│   ├── test_common.py            # tests/common.py helpers (no src/ counterpart)
│   ├── test_conftest.py          # tests/conftest.py (no src/ counterpart)
│   ├── test_argparser.py         # src/argparser.py (flat file in src/)
│   ├── rocprof_compute_analyze/
│   │   └── test_analysis_db.py
│   ├── rocprof_compute_profile/
│   │   ├── test_profiler_base.py
│   │   └── test_profiler_rocprofiler_sdk.py
│   ├── rocprof_compute_soc/
│   │   └── test_soc_base.py
│   ├── rocprof_compute_tui/
│   │   ├── test_analysis_tui.py                # src/rocprof_compute_tui/analysis_tui.py
│   │   └── widgets/
│   │       ├── test_instant_button.py          # src/rocprof_compute_tui/widgets/instant_button.py
│   │       └── menu_bar/
│   │           └── test_menu_bar.py            # src/rocprof_compute_tui/widgets/menu_bar/menu_bar.py
│   ├── roofline/
│   │   └── benchmark/
│   │       └── test_benchmark_base.py          # GPU lock tests
│   └── utils/
│       ├── test_amdsmi_interface.py
│       ├── test_file_io.py                     # pc_sampling loader part
│       ├── test_inject_roctx.py                # torch operator pattern matching
│       ├── test_mem_chart_gfx9.py                 # src/utils/mem_chart_gfx9.py
│       ├── test_mem_chart_gfx11.py                # src/utils/mem_chart_gfx11.py
│       │   # NOTE: src/utils/gui_components/memchart.py has NO existing test
│       │   # (test_mem_chart.py only imports mem_chart_gfx9/gfx11). Out of scope.
│       ├── test_native_tool_finder.py
│       ├── test_parser.py                      # search_pc_sampling_record etc.
│       ├── test_rocpd_data.py                  # torch trace queries / CSV
│       ├── test_roofline_calc.py
│       ├── test_schema.py                      # if non-trivial assertions exist
│       ├── test_specs.py                       # mi_gpu_spec loader, get_num_xcds
│       ├── test_utils_analysis.py              # large; see §5
│       ├── test_utils_common.py                # large; see §5
│       ├── test_utils_profile.py               # large; see §5
│       ├── test_utils_profile_csv.py
│       └── metrics/
│           ├── test_aggregation.py
│           ├── test_evaluation_pipeline.py
│           ├── test_expression.py
│           ├── test_metric_evaluator.py
│           └── test_noise_clamper.py
│
└── integration/
    ├── __init__.py
    ├── conftest.py               # integration-only fixtures, see §3b
    ├── test_analyze_cli.py       # filter_block / dispatch / normal_unit / etc.
    ├── test_analyze_workloads.py # per-arch analyze on captured workloads
    ├── test_autogen_config.py    # config-hash sanity (filesystem only, kept simple)
    ├── test_l1_cache_counters.py # was test_TCP_counters.py
    ├── test_list_metrics_cli.py  # --list-metrics / --list-blocks (moved from test_utils.py)
    ├── test_metric_validation.py
    ├── test_num_xcds_cli.py      # CLI portion of test_gpu_specs.py
    ├── test_pc_sampling.py
    ├── test_profile_dispatch.py
    ├── test_profile_iteration_multiplexing.py
    ├── test_profile_join.py
    ├── test_profile_kernel_execution.py
    ├── test_profile_live_attach_detach.py
    ├── test_profile_misc.py
    ├── test_profile_multi_rank.py
    ├── test_profile_path.py
    ├── test_profile_roofline.py
    ├── test_profile_section.py
    ├── test_profile_sets_func.py
    ├── test_profile_sort.py
    ├── test_roofline_analyze.py  # roofline analyze-only tests from test_analyze_commands.py
    ├── test_torch_trace_analysis.py    # rocpd-vs-csv torch trace parity
    └── test_torch_trace_coverage.py    # ROCTX marker coverage
```

`tests/workloads/` is referenced by absolute path
(`"tests/workloads/..."`) from both unit and integration tests; it stays
at `tests/workloads/` so existing path strings continue to work without
edits.

**`__init__.py` rule.** Mirror the source style: add `__init__.py` to a
test subdirectory **only** if the corresponding `src/` package has one.
`src/utils/` is a regular package (gets `__init__.py` in the mirror), but
`src/roofline/`, `src/rocprof_compute_soc/`, `src/rocprof_compute_profile/`,
and `src/rocprof_compute_analyze/` are namespace packages — adding
`__init__.py` in their test mirrors will shadow the src package and break
imports at collection time.

### 3a. Tests with no `src/` counterpart

Tier is determined by behavior, not subject. Tests for test
infrastructure (`tests/common.py`, `tests/conftest.py`) are in-process
with no GPU and no subprocess — so they're **unit** tests and live at
the `tests/unit/` root, alongside the src-mirroring subdirectories.

### 3b. `conftest.py` is split per tier; `common.py` stays shared

**`conftest.py` — layered, two files.** This follows pytest's standard
layered-conftest pattern: each test inherits every conftest on the path
from `tests/` down to its own directory.

- `tests/conftest.py` — tier-agnostic items. `ProfileModeImportGuard`
  stays here because it is used by both tiers (`test_import_guard.py`
  in unit, and the `binary_handler_profile_rocprof_compute` integration
  fixture invokes it internally). Plus any `sys.path` plumbing or
  cross-tier helpers.
- `tests/integration/conftest.py` — integration-scoped fixtures, helpers,
  and CLI options: `binary_handler_profile_rocprof_compute`,
  `binary_handler_analyze_rocprof_compute`, `require_torch_gpu`,
  `skip_monkeypatch_with_binary`, the private `inject_mpirun` helper,
  and the `pytest_addoption` definitions for `--call-binary`,
  `--rocprofiler-sdk-tool-path`, `--coverage-seed`, `--coverage-n`.
- `tests/unit/conftest.py` — not created; no unit-only fixtures exist
  today. Reserved for future unit-only fixtures if any are introduced.

Effect: a unit test that mistakenly requests an integration fixture
(`binary_handler_*`, `require_torch_gpu`) fails at collection
(`fixture not found`) instead of silently spinning up a subprocess.
Tier misclassification becomes a hard error.

**`common.py` — single shared module at `tests/`, unchanged.** Its
contents are deterministic utility functions (`get_output_dir`,
`setup_workload_dir`, `check_file_pattern`, `gpu_soc`, `check_csv_files`,
`check_resource_allocation`, `SUPPORTED_ARCHS`). Importing a pure
helper has no runtime side effects, so a tier split buys no isolation
and just creates re-hoisting churn. All tests continue to
`from common import ...` unchanged.

If `common.py` outgrows a single file later, split by **topic**
(`tests/helpers/paths.py`, `csv.py`, `gpu.py`), not tier. Out of scope
here.

## 4. File-level mapping (existing → target)

Every current `tests/test_*.py` file is accounted for. "Split" means the
file is divided across multiple destinations; "move" means the file
relocates whole (possibly renamed).

### 4.1 Pure-unit files — move (1:1)

| Current | Target | Source module under test |
|---|---|---|
| `test_analysis_db.py` | `unit/rocprof_compute_analyze/test_analysis_db.py` | `src/rocprof_compute_analyze/analysis_db.py` |
| `test_native_tool_finder.py` | `unit/utils/test_native_tool_finder.py` | `src/utils/native_tool_finder.py` |
| `test_roofline_calc_ai_analyze.py` | `unit/utils/test_roofline_calc.py` | `src/utils/roofline_calc.py` |
| `test_soc_base.py` | `unit/rocprof_compute_soc/test_soc_base.py` | `src/rocprof_compute_soc/soc_base.py` |
| `test_utils_profile_csv.py` | `unit/utils/test_utils_profile_csv.py` | `src/utils/utils_profile_csv.py` |
| `test_import_guard.py` | `unit/test_conftest.py` | `tests/conftest.py` (no `src/` counterpart, see §3a) |
| `test_profiler_base.py` | `unit/rocprof_compute_profile/test_profiler_base.py` | `src/rocprof_compute_profile/profiler_base.py` |

### 4.2 Pure-unit files — split per source module

| Current | Target(s) |
|---|---|
| `test_mem_chart.py` | `unit/utils/test_mem_chart_gfx9.py` + `unit/utils/test_mem_chart_gfx11.py` (`TestFormatBwHumanReadable`/`TestFormatSci`/`TestBar`/`TestMetricLine`/`TestGetSampleMetrics` → gfx9; `TestFormatValue`/`TestPlotMemChartGfx11`/`TestDefaultSampleMetrics` → gfx11; check final attribution against the actual imported symbol per class) |
| `test_metric_utils.py` | `unit/utils/metrics/test_aggregation.py` (`TestAggregation`), `test_expression.py` (`TestExpression`), `test_evaluation_pipeline.py` (`TestEvaluationPipeline`), `test_metric_evaluator.py` (`TestMetricEvaluator`) |
| `test_pc_sampling_analysis.py` | `unit/utils/test_parser.py` (`search_pc_sampling_record_*`, `load_per_kernel_*`, `load_pc_sampling_data_*`), `unit/utils/test_file_io.py` (`process_pc_sampling_*`, `build_agent_to_gpu_map_*`, `nullify_unevaluated_metrics_*`), `unit/utils/test_schema.py` (only if assertions about `utils.schema` are present, otherwise drop). ⚠ Do **not** merge into `tests/integration/test_pc_sampling.py` — that file uses `pytest.skip(..., allow_module_level=True)` to gate on GPU presence, which would suppress every unit assertion appended to it. |
| `test_torch_trace_analysis.py` | `unit/utils/test_rocpd_data.py` (`counters_query_uses_stack_id`, `marker_query_uses_stack_id`, `*_csv_has_correlation_id_from_stack_id`). The `test_torch_trace_output_same_for_rocpd_and_csv` test stays a unit test (in-memory CSV/SQLite, no CLI) — colocated in the same file. |
| `test_data_imputation.py` | `unit/utils/test_utils_analysis.py::imputation` section (imports `utils.utils_analysis` impute_*). Keep all 50+ tests; section header preserved as a comment block. CTest name and `test_categories.yaml` entry rename `test_data_imputation` → `test_utils_analysis` (the existing `add_test` now points at the new file). |
| `test_tui_components.py` | Split by `from rocprof_compute_tui...` import: `unit/rocprof_compute_tui/widgets/test_instant_button.py` (`TestInstantButton`), `unit/rocprof_compute_tui/widgets/menu_bar/test_menu_bar.py` (`TestDropdownMenu`, `TestMenuButton`), `unit/rocprof_compute_tui/test_analysis_tui.py` (`TestGetTopKernels`, `TestProcessPanelsToDataframes`, `TestCollapsiblesWidgetCreation` — all import from `src/rocprof_compute_tui/analysis_tui.py`). The `pytestmark = pytest.mark.tui` line is copied into each new file. |

### 4.3 Mixed files — split unit + integration

| Current | Unit destination | Integration destination |
|---|---|---|
| `test_gpu_specs.py` | `unit/utils/test_specs.py` — everything that monkeypatches `MachineSpecs` / `parse_*` / `get_*` / `load_yaml` (markers: `num_xcds_spec_class`, `misc` on local helpers) | `integration/test_num_xcds_cli.py` — `test_num_xcds_cli_output` only (uses `rocminfo` + `subprocess.run(['rocprof-compute', '-s'])`) |
| `test_utils.py` | Many — see §5 | `integration/test_list_metrics_cli.py` — the `test_list_metrics` and `test_list_blocks` functions under the `TESTS FOR MODELESS COMMAND LINE OPTIONS` banner (both use `binary_handler_analyze_rocprof_compute`) |
| `test_autogen_config.py` | If `test_config_hashes_match_files` is a pure filesystem check against repo state, classify as integration (depends on repo layout). Move to `integration/test_autogen_config.py`. | (single file, no split) |

### 4.4 Pure-integration files — move (1:1) or split by feature

| Current | Target(s) |
|---|---|
| `test_analyze_workloads.py` | `integration/test_analyze_workloads.py` (move) |
| `test_metric_validation.py` | `integration/test_metric_validation.py` (move) |
| `test_pc_sampling.py` | `integration/test_pc_sampling.py` (move) |
| `test_torch_trace_coverage.py` | `integration/test_torch_trace_coverage.py` (move) |
| `test_TCP_counters.py` | `integration/test_l1_cache_counters.py` (rename — file describes the feature, not the IP block, matching the CTest name `test_L1_cache_counters`) |
| `test_analyze_commands.py` | Split per feature group: `integration/test_roofline_analyze.py` gets every test whose name matches `test_*roofline*` or `test_roof_*`; `integration/test_analyze_cli.py` gets everything else (`test_valid_path`, `test_list_kernels`, `test_filter_block_*`, `test_filter_kernel_*`, `test_dispatch_*`, `test_gpu_ids`, `test_normal_unit_*`, etc.). All markers (`@pytest.mark.misc`, `serial`, `filter_block`, `list_metrics`, `normal_unit`, etc.) are copied verbatim. |
| `test_profile_general.py` | Split per marker into 12 files (see §6). Each new file contains exactly the tests tagged with the corresponding marker; helper classes (`MockProfiler`, `MockMachineSpecs`, `MockSoc`) and module-level constants are duplicated into each new file or moved to `integration/_profile_helpers.py` (no logic change). |

### 4.5 Helper / support files (stay at `tests/` root)

These are not test modules but are imported by tests:

- `common.py`, `conftest.py` — unchanged.
- `generate_test_analyze_workloads.py`, `generate_workloads.sh`, `4gpus.json` — generators; unchanged.
- `torch_trace_coverage_utils.py` — imported lazily by `test_torch_trace_coverage`; stays at root (or move alongside its test into `integration/` and update the single import).
- `tests/__init__.py` — unchanged (preserves `tests` as a package on `pythonpath`).
- `tests/workloads/` — unchanged.

## 5. Splitting `test_utils.py`

`test_utils.py` is the largest file and the worst offender for the "what
does this test?" question. Split strictly by the source module each
section already targets, identified by the `# ===== <banner> =====`
comment that opens each section.

| Section banner (verbatim) | Target file |
|---|---|
| `VERSION UTILITIES TESTS` | `unit/utils/test_utils_common.py` |
| `ROCPROF DETECTION TESTS` | `unit/utils/test_utils_common.py` for `test_detect_rocprof_*`; the `test_capture_subprocess_output_*` tests inside the same banner block go to `unit/utils/test_utils_profile.py` |
| `JSON DATA PARSING TESTS` (`get_agent_dict`) | `unit/utils/test_utils_analysis.py` |
| `Tests for get_gpuid_dict function` | `unit/utils/test_utils_analysis.py` |
| `test_check_resource_allocation_*` (no banner — sits between `get_gpuid_dict` and `FILE PATTERN MATCHING`) | `unit/test_common.py` (helper lives in `tests/common.py`; no `src/` counterpart, see §3a) |
| `FILE PATTERN MATCHING TESTS` (`check_file_pattern`) | `unit/test_common.py` (same — `tests/common.py`) |
| `PMC PERF PARSING UTILITIES TESTS` (`parse_pmc_perf`) | `unit/utils/test_utils_profile.py` |
| `RUN_PROF TESTS` | `unit/utils/test_utils_profile.py` |
| `ROCPROFV3 OUTPUT PROCESSING TESTS` | `unit/utils/test_utils_profile.py` |
| `KOKKOS TRACE PROCESSING TESTS` | `unit/utils/test_utils_profile.py` |
| `Normal Functionality:` (the `get_submodules` block) | `unit/utils/test_utils_profile.py` |
| `TESTS FOR EMPTY WORKLOAD` (`is_workload_empty`) | `unit/utils/test_utils_analysis.py` |
| `TESTS FOR merge_counters_spatial_multiplex FUNCTION` | `unit/utils/test_utils_analysis.py` |
| `Tests for convert_metric_id_to_panel_info function` | `unit/utils/test_utils_common.py` |
| `--- New test functions for add_counter_extra_config_input_yaml ---` | `unit/utils/test_utils_common.py` |
| `additional test detect_rocprof console error` | `unit/utils/test_utils_common.py` |
| `additional tests for v3_counter_csv_to_v2_csv function` | `unit/utils/test_utils_profile.py` |
| `Test PC_sampling function` | `unit/utils/test_utils_profile.py` |
| `test_format_scientific_notation_if_needed` (single function, no banner) | `unit/utils/test_utils_common.py` |
| **`TESTS FOR MODELESS COMMAND LINE OPTIONS`** (`--list-metrics`/`--list-blocks`) | **`integration/test_list_metrics_cli.py`** |
| `TESTS FOR AMDSMI INTERFACE` | `unit/utils/test_amdsmi_interface.py` |
| `TESTS FOR ITERATION MULTIPLEXING` (`impute_counters_iteration_multiplex`) | `unit/utils/test_utils_analysis.py` |
| `validate_roofline_csv TESTS` | `unit/utils/test_utils_common.py` |
| `TESTS FOR NOISE_CLAMP: Multi-Pass Profiling Variance Handling` | `unit/utils/metrics/test_noise_clamper.py` |
| `Experimental Feature Tests` (`argparser.ExperimentalAction`) | `unit/test_argparser.py` |
| `Test rocm library resolver` (`version_to_numeric`, `resolve_rocm_library_path`) | `unit/utils/test_utils_common.py` |
| `TESTS FOR Analysis DB mode: Analysis DB mode code path` (`calc_roofline_data`) | `unit/rocprof_compute_analyze/test_analysis_db.py` (append; merges with §4.1 file) |
| `GPU Benchmark Locking Tests` (`roofline.benchmark.benchmark_base`) | `unit/roofline/benchmark/test_benchmark_base.py` |
| `BUILD METRIC LIST TESTS` | follow the in-block `from utils... import ...` to determine the target. If it imports from `utils.utils_analysis`, route to `unit/utils/test_utils_analysis.py`; otherwise route per imported symbol. |
| `format_table_ascii TESTS` | `unit/utils/test_utils_common.py` |
| `Torch operator pattern matching` (and the sibling sub-banners `get_matched_torch_operators_for_display`, `parse_torch_operator_patterns`, `PatternMatcherEngine`, `Additional coverage`) | `unit/utils/test_inject_roctx.py` if in-block imports point at `utils.inject_roctx` / `utils.parser` / `utils.pattern_matching`; otherwise one file per imported source symbol. |
| `TESTS FOR reconfigure_stdio_utf8 FUNCTION` | `unit/utils/test_utils_common.py` |

**Mechanical rule for ambiguous sections.** Within each banner block,
look at the first `from <module> import ...` *inside the block* — most
sections import lazily inside each test function. That import
determines the target file. If no in-block import, fall back to the
top-of-file imports and pick the one matching the function name.
After splitting, sweep each new file for unqualified stdlib references
(`io.`, `builtins.`, `math.`, `mock.`) that were imported at the original
top-of-file and need to be re-added — section extraction does not carry
top-level imports.

## 6. Splitting `test_profile_general.py`

Each `@pytest.mark.<X>` marker becomes one feature file under
`integration/`. The marker decoration stays (so CTest keeps using `-m
<X>`). Helper definitions (`MockProfiler`, `MockMachineSpecs`,
`MockSoc`) move once to `integration/_profile_helpers.py` and are
imported from each new file.

| Marker | Target file |
|---|---|
| `path` | `integration/test_profile_path.py` |
| `roofline_1`, `roofline_2` | `integration/test_profile_roofline.py` (keep both markers; CTest runs them separately) |
| `misc` | `integration/test_profile_misc.py` |
| `kernel_execution` | `integration/test_profile_kernel_execution.py` |
| `dispatch` | `integration/test_profile_dispatch.py` |
| `join` | `integration/test_profile_join.py` |
| `sort` | `integration/test_profile_sort.py` |
| `section` | `integration/test_profile_section.py` |
| `sets_func` | `integration/test_profile_sets_func.py` |
| `iteration_multiplexing_1`, `iteration_multiplexing_2`, `iteration_multiplexing_stochastic` | `integration/test_profile_iteration_multiplexing.py` (one file, three markers preserved per-test) |
| `live_attach_detach` | `integration/test_profile_live_attach_detach.py` |
| `multi_rank` | `integration/test_profile_multi_rank.py` |
| `torch_trace` | merge into `integration/test_torch_trace_coverage.py` (only a handful) |

## 7. `conftest.py` strategy

See §3b for the file split. Implementation is mechanical: cut
integration-scoped definitions out of `tests/conftest.py`, paste into
`tests/integration/conftest.py`, no body edits. Pytest's layered
discovery handles inheritance.

## 8. `pyproject.toml` changes

```diff
 [tool.pytest.ini_options]
 addopts = [
     "--import-mode=importlib",
     "-rx",
 ]
 pythonpath = [
     ".",
     "src",
     "src/rocprof_compute_soc",
     "src/roofline",
     "src/rocprof_compute_analyze/utils",
-    "tests"
+    "tests",
+    "tests/unit",
+    "tests/integration"
 ]
```

The `import test_utils` statement in `test_torch_trace_coverage.py` is
rewritten to `import common` per the §14 risk row (the referenced
symbols `get_output_dir` and `clean_output_dir` actually live in
`tests/common.py`, not `test_utils.py`).

Markers: **no** `unit`/`integration` markers are added. Tier is
directory, not marker. The existing
`[tool.pytest.ini_options].markers` list is unchanged.

## 9. `CMakeLists.txt` changes

Every `add_test(...)` block in the top-level `CMakeLists.txt` gets
exactly two edits:

1. Updated path: `tests/test_foo.py` → `tests/unit/<sub>/test_foo.py`
   or `tests/integration/test_foo.py`.
2. Updated `--junitxml` path to match (kept under `tests/` so artifact
   collection paths in CI don't change): `--junitxml=tests/test_foo.xml`
   stays as is (the XML target is independent of the source path).
3. Add each new CTest name to any aggregate list near the bottom of
   `CMakeLists.txt` (e.g. `ALL_TEST_NAMES`) used to assemble install
   manifests or default tier membership.

Existing `add_test` blocks have their paths updated in place; the
splits in §4.2 and §5 add new `add_test` entries (counted below). For
`test_profile_general.py` splits, the existing 15 `add_test(...)` blocks
(`test_profile_path`, `test_profile_misc`, `test_profile_kernel_execution`,
`test_profile_dispatch`, `test_profile_join`, `test_profile_sort`,
`test_profile_section`, `test_profile_sets_func`,
`test_profile_live_attach_detach`, `test_profile_multi_rank`,
`test_profile_roofline_1`, `test_profile_roofline_2`,
`test_profile_iteration_multiplexing_1`, `_2`, `_stochastic`) already
pass `-m <marker> tests/test_profile_general.py` — change each to point
at the new per-marker file (`roofline_1`/`_2` share
`integration/test_profile_roofline.py`; `iteration_multiplexing_*`
share `integration/test_profile_iteration_multiplexing.py`):

```cmake
add_test(
    NAME test_profile_path
    COMMAND
        ${PYTHON_TEST_COMMAND} -m pytest ${STANDALONEBINARY_TEST_OPTION} -m path
        --junitxml=tests/test_profile_path.xml ${COV_OPTION}
-       tests/test_profile_general.py ${WORKING_DIR_OPTION}
+       tests/integration/test_profile_path.py ${WORKING_DIR_OPTION}
)
```

Keep the `-m <marker>` filter on each call even though it becomes
redundant once each marker has its own file — preserves selection
idioms (`pytest -m path`) and guards against future tests added without
the marker.

Splits per §4.2 and §5 get **one `add_test` per new file** (parallelism
via `ctest -j`, isolated junit XML). ~25 new entries replace the single
`test_utils` invocation.

Helper classes that were single-file in `test_utils.py` /
`test_profile_general.py` (`MockArgs`, `MockProfiler`,
`make_dummy_process`) get duplicated verbatim into each split file *or*
hoisted to a `_helpers.py` sibling — pick one and apply consistently per
split.

## 10. `tests/test_categories.yaml` changes

The four tiers stay. Each `test_patterns` entry is renamed to match
the new CTest names from §9 (one entry per new `add_test`).

A fifth tier `unit` is added, selecting every CTest whose name maps to
a file under `tests/unit/`. Entries are **CTest names**, not file
basenames — e.g. `test_num_xcds_spec_class` (the `add_test` name), **not**
`test_specs` (the file). Run `ctest -N` after the split and copy names
verbatim:

```yaml
  unit:
    description: "All unit tests - no GPU required (target: < 2 min)"
    test_patterns:
      - test_num_xcds_spec_class   # CTest name, not file basename
      - test_analysis_db
      - ...   # all unit/ tests
    labels:
      - "unit"
      - "developer-loop"
```

This lets a developer with no GPU run `ctest -L unit` for a fast signal.

## 11. CI workflow

`.github/workflows/rocprofiler-compute-continuous-integration.yml`
(at the **repo root** `.github/`, not `projects/rocprofiler-compute/.github/`)
does **not** need changes for the refactor itself. The single
`tools/run-ci.py` invocation passes `-E "${EXCLUDED_TESTS}"` (excludes
`test_profile_live_attach_detach`); the test name `test_profile_live_attach_detach`
is preserved verbatim, so the exclude still matches.

A fast-feedback PR job is added that runs `ctest -L unit` first and
short-circuits the GPU runner if it fails. Implemented in PR 5.

`.github/ci-matrix.yml`, `rocprofiler-compute-formatting.yml`,
`.pre-commit-config.yaml`: no changes.

## 12. Migration phases

Five PRs. Each must leave `ctest` fully green.

1. **PR 1 — scaffolding, conftest split, and all 1:1 moves.**
   *Prerequisite:* `git submodule update --init projects/rocprofiler-compute`
   so the vendored `pyyaml` submodule is populated; otherwise unit-test
   collection fails before any assertions run. Create
   `tests/unit/` and `tests/integration/` with empty `__init__.py`s.
   Add both dirs to `pyproject.toml` `pythonpath`.
   Split `tests/conftest.py` per §3b / §7 (cut–paste, no body edits).
   Move every file that relocates whole (no split): the seven pure-unit
   files in §4.1, the four integration files in §4.4 that don't split,
   plus the `test_TCP_counters.py` → `integration/test_l1_cache_counters.py`
   rename. Update each corresponding `add_test` path. No splits of test
   files, no merges, no helper extraction. *Diff: conftest 2-way split
   + ~12 file renames + ~12 CMake path edits + 3 pyproject lines.*

2. **PR 2 — small file splits.** All §4.2 splits (`test_mem_chart`,
   `test_metric_utils`, `test_pc_sampling_analysis`,
   `test_torch_trace_analysis`, `test_tui_components`), the §4.3
   borderline splits (`test_gpu_specs`, `test_autogen_config`), and the
   `test_data_imputation` → `test_utils_analysis.py` merge (§4.2 last
   row). One `add_test` per new file per §9. *Diff:
   ~25 new files, deletions of 8 old ones, matching `add_test` and
   `test_categories.yaml` edits.*

3. **PR 3 — `test_profile_general.py` + `test_analyze_commands.py`
   splits.** Per §6 and §4.4. Twelve new `test_profile_*.py` files +
   `_profile_helpers.py`, plus the two-way split of
   `test_analyze_commands.py`. Markers preserved verbatim; existing
   `add_test` blocks point at the new per-marker files. *Diff:
   ~16 new files, 2 deletions, ~14 CMake edits.*

4. **PR 4 — `test_utils.py` split.** Per §5. ~27 sub-files, the most
   error-prone PR — kept isolated so a regression here cannot mask one
   from PR 1–3. Review with `git diff --stat` per file and a
   `pytest --collect-only -q` diff that shows identical test IDs modulo
   path. Also routes the two integration tests under the
   `TESTS FOR MODELESS COMMAND LINE OPTIONS` banner into
   `integration/test_list_metrics_cli.py` and updates the single
   `import test_utils` in `test_torch_trace_coverage.py` (§14 risk).

5. **PR 5 — `unit` tier and fast-feedback CI.** Add the `unit` tier to
   `test_categories.yaml` (§10) and the fast-feedback CI job (§11).

## 13. Verification (every PR)

Before merge of each PR:

1. `pytest --collect-only -q tests/` before and after → diff the
   collected node IDs. They must differ **only** by path; the trailing
   `::test_name[params]` portion must be identical (and total count
   equal).
2. `ctest -N` before and after → number of registered tests is equal
   or larger (larger when an `add_test` is split per §9); no test name
   is silently dropped.
3. `ctest` on a GPU runner → all tests that passed before pass after.
   `git diff` of CDash output should show only timing variance.
4. `ruff check tests/` → clean (per `.pre-commit-config.yaml`; root
   `pyproject.toml` exempts non-`src/**` from ANN/UP/PTH/PLW1514, so no
   new type annotations are forced).
5. `grep -r "from test_utils import" tests/` and similar reverse-imports
   → zero matches, or all updated to the new module paths.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `test_torch_trace_coverage.py` does `import test_utils` and calls `test_utils.get_output_dir` / `test_utils.clean_output_dir`, but those functions live in `tests/common.py`. Latent `AttributeError` masked because the test is GPU-gated. Splitting `test_utils.py` turns this into a collection-time `ModuleNotFoundError`. | Apply the `import common` → `common.{get_output_dir,clean_output_dir}` rewrite as part of PR 4 (it is the import-fixup step, not a risk to mitigate); call out in the PR description. |
| Pytest marker `@pytest.mark.path` is used in both `test_profile_general.py` (`path` = output-directory variants) and `test_metric_validation.py` (`@pytest.mark.path` = something different). After the integration split, `pytest -m path` could pick up both files. | Existing CTest entries already invoke pytest with an explicit file path (`tests/test_profile_general.py`), so the file scoping already disambiguates. Preserve that pattern. Do not introduce a project-wide `pytest -m path` step. |
| `test_utils.py` sections lazily-import their source module inside the function body (e.g. `from utils.utils_common import parse_sets_yaml` inside the test body, rather than at module top). Splitting may produce files where the top-of-file imports do not reference anything. | Keep lazy imports as-is. Linters (`ruff F`) do not flag a lazy import. |
| Some currently-failing or `xfail` tests may be revalidated under their new collection path. | Run `ctest --rerun-failed` after each PR; if anything changes status, it must be reverted (the refactor must not flip pass/fail). |
| `--call-binary` mode (CMake `STANDALONEBINARY=ON`) wraps test invocations with `--call-binary`; the wrapped `pytest` discovers the new paths the same way. | No action needed; verified by `ctest` on standalone-binary CI. |

