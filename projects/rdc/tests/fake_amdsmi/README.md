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
| `RDC_FAKE_FAIL_ONCE=rule,...` | like `RDC_FAKE_FAIL`, but each rule fails only the first matching call |
| `RDC_FAKE_UNFILLED=api[@gpu],...` | the call returns `AMDSMI_STATUS_SUCCESS` without writing its outputs, as a library that reports success but leaves a structure unset |
| `RDC_FAKE_DELAY_MS=api[@gpu]=ms,...` | every matching call sleeps `ms` milliseconds (0 to 60000) first, whoever makes it |
| `RDC_FAKE_ARM=path` | `FAIL`, `FAIL_ONCE`, `UNFILLED`, `DELAY_MS`, `SENTINEL`, `CPER_COUNT`, `EVENTS`, `NUM_FREQ` and `COUNTER_RESET` apply once `path` exists, and from then on, so a test can set RDC up first |
| `RDC_FAKE_CPER_COUNT=n` | `amdsmi_get_gpu_cper_entries` reports `n` entries (0 to 1048576), even more than the caller's buffers hold; it writes header pointers only up to their capacity, all to one record outside the caller's data buffer, and reports 0 bytes of data |
| `RDC_FAKE_EVENTS=event@gpu,...[,count=n]` | `amdsmi_get_gpu_event_notification` delivers each of these events once, in order, as many per call as the caller has room for (`event` 0 to 15, of which RDC knows 1 to 13; `gpu` a processor index or `x` for a handle the fake never handed out); with `count`, the first call that delivers reports `n` events (0 to 65536) instead of the number it wrote. As from amd-smi, a call without room for an event gets `AMDSMI_STATUS_INVAL`. Process start and end events (12 and 13) carry amd-smi's message `PID: <1000 + gpu>  task: fake` (999 for `x`), the others `fake event <event>` |
| `RDC_FAKE_EVENT_MASKS=1` | the event calls keep an event file per processor, as amd-smi does: `amdsmi_init_gpu_event_notification` opens it with no events enabled (an open file stays as it is), `amdsmi_set_gpu_event_notification_mask` replaces its mask (`AMDSMI_STATUS_NOT_INIT` when it is not open) and `amdsmi_stop_gpu_event_notification` closes it (a stop that a rule fails changes nothing); the last `amdsmi_shut_down` closes every file. An `EVENTS` event of a processor is then delivered only once that processor's file is open and its mask has the event's bit, as the driver sends only enabled events; until then it waits, without holding back the others. Events of `x` do not wait |
| `RDC_FAKE_NUM_FREQ=n` | every clock frequency table has `n` levels (0 to 33) and no current level (`current` is `UINT32_MAX`, as amd-smi reports when the driver marks no level current) |
| `RDC_FAKE_PROCS=n` | `n` compute processes (0 to 1024), pid `1000 + i` on the one GPU `i % RDC_FAKE_GPUS` (a GPU index, not a partition). As with amd-smi, a list too small for every process gets `AMDSMI_STATUS_INSUFFICIENT_SIZE` and process IDs only, and `amdsmi_get_gpu_compute_process_gpus` with a NULL list reports the GPU count |
| `RDC_FAKE_PROCS_GROW=1` | one more process on every `amdsmi_get_gpu_compute_process_info` call, as when processes start between the count call and the fill call; `amdsmi_get_gpu_compute_process_gpus` knows the processes of the last list (the first `RDC_FAKE_PROCS` before any list call) |
| `RDC_FAKE_COUNTER_RESET=1` | the energy, activity, PCIe bandwidth, XGMI data, utilization and violation accumulators and the XGMI event counters restart from 0 when the `RDC_FAKE_ARM` file first exists, as after a GPU reset; needs `RDC_FAKE_ARM` |
| `RDC_FAKE_VERSION_BUILD_NULL=1` | `amdsmi_get_lib_version` returns a NULL build string |
| `RDC_FAKE_EVENT_WAIT_MS=ms` | upper bound for the event-notification wait (default 100), so `rdcd` stops promptly |
| `RDC_FAKE_LOG=path` | append `api gpu` for every call (`-1` for calls without a GPU), plus ` !` when a rule failed the call, except a call rejected for its processor handle (the source handle, in calls that take two) or because the library is not initialized; `amdsmi_init`, `amdsmi_get_lib_version` and `amdsmi_shut_down` are always logged |

`amdsmi_init` honors the `FAIL`, `FAIL_ONCE` and `DELAY_MS` rules without
`@gpu`, `amdsmi_get_lib_version` (valid before `amdsmi_init`) honors the
`FAIL`, `FAIL_ONCE`, `UNFILLED` and `DELAY_MS` rules without `@gpu`, and
`amdsmi_shut_down` honors none. Within a process, each `FAIL_ONCE` rule fails one
call, each `EVENTS` event is delivered once and the `PROCS_GROW` call count never
restarts: an `amdsmi_shut_down()` and a new `amdsmi_init()` do not start them
over.

A malformed rule (`FAIL`, `FAIL_ONCE`, `UNFILLED`, `DELAY_MS`, `SENTINEL`), a
rule whose `@gpu` is not a simulated processor, a malformed or out-of-range
value of `CPER_COUNT`, `EVENTS`, `EVENT_MASKS`, `NUM_FREQ`, `PROCS`, `PROCS_GROW`,
`COUNTER_RESET` or `VERSION_BUILD_NULL`, `COUNTER_RESET` without `ARM` and a
`LOG` file that cannot be opened abort the process, so a typo in a value cannot
turn into a test that passes without its fault. A rule for an API that nothing
calls still never fires. `GPUS`, `PARTITIONS`, `HIVES`, `POISON` and
`EVENT_WAIT_MS` are clamped to their ranges.

`CPER_COUNT` past the caller's capacity, `count` in `EVENTS` past what the call
wrote, `NUM_FREQ=0` and `VERSION_BUILD_NULL` break amd-smi's output contract on
purpose: amd-smi stops at the caller's capacity (for CPER entries with
`AMDSMI_STATUS_MORE_DATA`, or `AMDSMI_STATUS_OUT_OF_RESOURCES` when none fit,
and with header pointers into the caller's buffer), reports only the events it
wrote, fails the call instead of returning an empty frequency table, and never
returns a NULL build string. They check that RDC does not trust its library
blindly; a defect that only they reproduce is a hardening fix, not one users can
hit with amd-smi.

[`tests/ci/test_fake_gpu_fake.py`](../ci/test_fake_gpu_fake.py) checks the
knobs above from `RDC_FAKE_FAIL_ONCE` down to `RDC_FAKE_VERSION_BUILD_NULL`,
and the handle rules below. `RDC_FAKE_EVENT_WAIT_MS` and `RDC_FAKE_LOG` are
used by [`tests/ci/test_fake_gpu.py`](../ci/test_fake_gpu.py).

## Handles

Handles are numbers, never pointers: `(generation << 32) | index`, with bit 31
also set in socket handles. Like amd-smi's reference-counted initialization, the
`amdsmi_shut_down()` that releases the last `amdsmi_init()` reference ends the
generation. As from amd-smi, a processor handle from an earlier generation, or
any other value the fake did not hand out, gets `AMDSMI_STATUS_NOT_FOUND`, and
such a socket handle gets `AMDSMI_STATUS_INVAL`; NULL gets
`AMDSMI_STATUS_INVAL`. The generation starts at 1, so no handle is NULL.

## Running the tests locally

```shell
cmake -B build -DBUILD_TESTS=ON ... && cmake --build build && cmake --install build --component tests
RDC_TEST_PREFIX=<install prefix> LD_LIBRARY_PATH=<prefix>/lib:<prefix>/lib/rdc \
    python3 -m pytest tests/ci/test_fake_gpu.py tests/ci/test_fake_gpu_fake.py
```

`rdcd` allows one instance per host, so stop any other `rdcd` first. Set
`RDC_TEST_TIME_FACTOR` to stretch the timeouts for sanitizer builds, and
`RDC_TEST_STRESS=1` to run the stress tests (long loops, marked `stress`).
With `RDC_CI_REQUIRE_STRESS=1` a run in which no stress test ran fails.
