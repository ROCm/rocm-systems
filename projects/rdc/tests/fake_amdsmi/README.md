# Fake amd-smi for GPU-less RDC tests

`librdc_fake_amdsmi.so` implements every `amdsmi_*` function RDC calls and
answers them for simulated GPUs. Preloaded into `rdcd`, `rdci` or a process
that uses RDC in embedded mode, it lets
[`tests/ci/test_fake_gpu.py`](../ci/test_fake_gpu.py) run the daemon and the
CLI on CI hosts without a GPU and check the value RDC reports for each GPU.

It is a test-only library: CMake builds it with `BUILD_TESTS=ON` and installs it
with the `tests` component in `share/rdc/rdctst_tests/fake_gpu/`, next to the
test suite, never on a library search path.

## Readings

GPU `g` reports `base + step * g` for each reading in
[`fake_values.inc`](fake_values.inc), in amd-smi's units.
`tests/ci/fake_gpu_golden.py` reads the same table and converts each reading to
the unit RDC documents for the field, so tests do not depend on RDC's own
conversion code. For up to 8 GPUs no two readings of one GPU are equal and no
reading repeats across GPUs (`test_fake_readings_are_distinct` checks this), so
a value shown for the wrong GPU or the wrong field fails the comparison.

## Environment

| Variable | Effect |
| --- | --- |
| `RDC_FAKE_GPUS=N` | GPUs to simulate, 1 to 128 (default 4) |
| `RDC_FAKE_PARTITIONS=P` | compute partitions per GPU, 1 to 8 (default: not partitionable) |
| `RDC_FAKE_HIVES=H` | XGMI hives, 1 to 128; GPU `g` is in hive `g % H` (default 1) |
| `RDC_FAKE_FAIL=rule,...` | `rule` is `api[@gpu][=status]` (`gpu` is the processor index, which counts partitions when `RDC_FAKE_PARTITIONS` is set): the call returns `status` (a number or an `AMDSMI_STATUS_*` name, default `AMDSMI_STATUS_UNKNOWN_ERROR`; anything else aborts). `api` `*` matches every call that takes a GPU. Calls without a GPU argument, such as `amdsmi_init`, match only rules without `@gpu`. |
| `RDC_FAKE_POISON=1` | a failing call first fills its outputs with `0xA5` bytes, as a caller that ignores the status would read from a dirty stack |
| `RDC_FAKE_SENTINEL=api[@gpu],...` | the call succeeds but reports amd-smi's "not available" value (all bits set); honored by `amdsmi_get_power_info`, `amdsmi_get_gpu_activity`, `amdsmi_get_gpu_busy_percent`, `amdsmi_get_clock_info` and `amdsmi_get_gpu_metrics_info` |
| `RDC_FAKE_ARM=path` | `FAIL` and `SENTINEL` rules apply only once `path` exists, so a test can set RDC up first |
| `RDC_FAKE_EVENT_WAIT_MS=ms` | upper bound for the event-notification wait (default 100), so `rdcd` stops promptly |
| `RDC_FAKE_LOG=path` | append `api gpu` for every call (`-1` for calls without a GPU), plus ` !` when a rule failed the call |

## Running the tests locally

```shell
cmake -B build -DBUILD_TESTS=ON ... && cmake --build build && cmake --install build --component tests
RDC_TEST_PREFIX=<install prefix> LD_LIBRARY_PATH=<prefix>/lib:<prefix>/lib/rdc \
    python3 -m pytest tests/ci/test_fake_gpu.py
```

`rdcd` allows one instance per host, so stop any other `rdcd` first. Set
`RDC_TEST_TIME_FACTOR` to stretch the timeouts for sanitizer builds, and
`RDC_TEST_STRESS=1` to run the stress tests (long loops, marked `stress`).
With `RDC_CI_REQUIRE_STRESS=1` a run in which no stress test ran fails.
