# Python

## Overview

This directory contains Python examples demonstrating rocprofiler-systems' Python profiling capabilities. The examples show different instrumentation approaches: the `@rocprofsys.profile()` decorator for automatic function tracing, `omni_user_region()` for manual named regions, `@noprofile` for excluding specific functions, and `rocprofsys.user.stop_trace()` for runtime tracing control. Workloads include recursive Fibonacci computation, nested loops, and optional NumPy array operations.

## Source Files

- `source.py` - Uses `@rocprofsys.profile()` decorator and `omni_user_region()` for manual region instrumentation with selective tracing control.
- `source-numpy.py` - Same as `source.py` with additional NumPy random array operations (falls back to standard `random` if NumPy is unavailable).
- `builtin.py` - Uses the built-in `@profile` decorator without rocprofiler-systems-specific integration.
- `external.py` - No profiling decorators; intended for external instrumentation via `rocprof-sys-python`.
- `noprofile.py` - Demonstrates `@noprofile` to exclude specific functions (fibonacci, inefficient loops) from profiling while tracing only the outer `run()` function.
- `fill.py` - Minimal loop-based test with `include_args = True` configuration for argument-level tracing.

## Prerequisites

- Python 3
- rocprofiler-systems Python module
- (Optional) NumPy for `source-numpy.py`

## Running

Python examples are run via `rocprof-sys-python` or `python -m rocprofsys`:

```bash
# Using rocprof-sys-python wrapper
rocprof-sys-python -- ./source.py -n 5 -v 20

# Using python module
python -m rocprofsys -- ./source.py -n 5 -v 20

# Scripts without rocprofiler-systems decorators
rocprof-sys-python -- ./external.py -n 3 -v 15

# NumPy variant
rocprof-sys-python -- ./source-numpy.py -n 3 -v 20
```

**Common arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `-n, --num-iterations` | Number of iterations | 3 |
| `-v, --value` | Starting Fibonacci value | 20 |
| `-s, --stop-profile` | Stop tracing after N iterations (0 = never) | 0 |

## Profiling with rocprofiler-systems

```bash
rocprof-sys-python -- ./source.py -n 5 -v 20
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace with Python call stacks |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |
| `ROCPROFSYS_USE_SAMPLING` | `ON` | Enable statistical sampling |
| `ROCPROFSYS_SAMPLING_FREQ` | `50` | Sampling frequency (interrupts/sec) |
