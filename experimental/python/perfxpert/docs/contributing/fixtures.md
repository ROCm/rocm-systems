# Contributing: new test fixture

## What you're adding

A SQLite database (`.db`) that represents a rocpd trace, paired with
documentation (`.md`) describing what it contains and how to regenerate it.
Fixtures are used across unit, integration, and benchmarking tests.

## File locations

- Database: `tests/fixtures/<name>.db` or `tests/fixtures/<category>/<name>.db`
- Documentation: `tests/fixtures/<name>.md` or `tests/fixtures/<category>/<name>.md`
- Generator (if synthetic): `tests/fixtures/_build_<name>.py` or `_generators/<name>.py`

## Size constraint

Each fixture `.db` must be < 1 MB. Larger traces should be:
- Downsampled (keep only hottest kernels)
- Synthetic (generated from parameters)
- Hosted externally with a download script

## Documentation template

```markdown
# Fixture: <name>

## What it contains

<Description of the GPU, kernel, workload, and metrics represented.>

## How to regenerate

<Step-by-step command or script to rebuild the .db from scratch.>

```bash
# SKIP-SAMPLE — template placeholder (<name> not a real path)
python3 tests/fixtures/_generators/<name>.py
```

## Baseline

Built on: <GPU model, ROCm version, hardware date>
MD5: <hash of the .db file>
```

## Schema requirements

All rocpd fixtures follow the rocpd UUID-based trace schema:
- Tables: `rocpd_data`, `metrics`, `kernel_events`
- ID columns: UUIDs (string)
- Metadata: creation timestamp, ROCm version, GPU ID

See `CLAUDE.md` §SQLite + rocpd for the full schema reference.

## Tests that use it

- List any existing tests that consume this fixture
- Example: `tests/test_tools/test_classifier.py::test_classify_compute_bound`

## Common pitfalls

- If real trace: sanitize kernel names (remove proprietary project names)
- If synthetic: keep DB size small (compress via VACUUM)
- Don't commit large binaries; use generators instead
- Update MD5 in docs if you regenerate

## Checklist before commit

- [ ] `.db` file < 1 MB
- [ ] `.md` file with regeneration steps
- [ ] Generator script in `_generators/` (if synthetic)
- [ ] Generator produces byte-identical output (deterministic)
- [ ] Fixture used by ≥ 1 test (no orphans)

## Related docs

- rocpd schema: https://github.com/ROCm/rocprofiler-sdk
- Real trace capture: `rocprofv3` CLI reference
