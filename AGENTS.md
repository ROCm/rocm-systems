# AGENTS.md

## Cursor Cloud specific instructions

This repo (`ROCm/rocm-systems`) is a large super-repo of ROCm GPU systems libraries. **The Cloud VM has no AMD GPU and no ROCm toolchain**, so the C++/HIP projects under `projects/` cannot be built or run here. Development/testing in this environment is scoped to the **pure-Python components that work without a GPU**:

| Component | Path | Purpose |
|-----------|------|---------|
| `rocprofiler-compute` | `projects/rocprofiler-compute` | Python profiling CLI (`rocprof-compute`). `analyze` mode + most unit tests run CPU-only; `profile` mode needs an AMD GPU. |
| `rocm-bootstrap` | `python/rocm-bootstrap` | Installable Python package for AMD GPU detection / target naming. |
| `systems_pr_bot` | `tools/systems_pr_bot` | Standalone GitHub PR policy-check script (`requests` + `pyyaml`). |

### Environment / setup caveats
- Python deps are installed into the **system interpreter with `pip install --break-system-packages`** (Ubuntu 24.04 is externally-managed and there is no `python3-venv` by default). This matches the rocprofiler-compute CI. The update script handles this on startup.
- Console-script entry points (e.g. `pytest`, `rocprof-compute-detect`) land in `~/.local/bin`, which is **not on `PATH`**. Invoke via `python3 -m pytest`, `python3 -m ruff`, or by explicit path (`python3 projects/rocprofiler-compute/src/rocprof-compute`) instead of relying on the bare command name.
- `rocprofiler-compute` requires its `src/` git submodules (vendored PyYAML, etc.). The update script runs `git submodule update --init --recursive -- projects/rocprofiler-compute/src/`.

### Run / test / lint commands (non-GPU)
- **Run the app (hello world):** from `projects/rocprofiler-compute`, `python3 src/rocprof-compute analyze --path tests/workloads/vcopy/MI200 --block 2` analyzes a checked-in pre-collected profile and prints the performance report. `--version` also works.
- **rocprofiler-compute tests (CPU-only subset):** from `projects/rocprofiler-compute`, e.g. `python3 -m pytest tests/test_analyze_commands.py tests/test_argparser.py tests/test_mem_chart.py tests/test_soc_base.py tests/test_roofline_calc.py tests/test_metric_utils.py`. NOTE: `test_analyze_commands.py` spawns many subprocesses and takes ~6 min. Any `tests/test_profile_*.py` / PC-sampling live tests require an AMD GPU and will fail here — do not run them.
- **rocprofiler-compute lint:** from `projects/rocprofiler-compute`, `python3 -m ruff check .` and `python3 -m ruff format --check .`.
- **rocm-bootstrap tests:** from `python/rocm-bootstrap`, `python3 -m pytest` (GPU detection gracefully reports none).
- **systems_pr_bot tests:** from repo root, `python3 -m unittest tools/systems_pr_bot/test_policy_check_ut.py -v`.

### Not runnable here
Everything requiring ROCm/HIP compilers or AMD GPUs: all C++/HIP `projects/*` builds, `rocprof-compute profile` (live collection), full `ctest`/CI parity, and GPU-marked tests. Build those on an AMD GPU + ROCm host (see per-project READMEs).
