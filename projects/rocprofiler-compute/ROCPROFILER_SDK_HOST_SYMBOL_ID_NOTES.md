# rocprofiler-sdk host-function ID overflow (ROCM-28219)

Notes for a follow-up fix in `projects/rocprofiler-sdk`. Tracked as
[ROCM-28219](https://amd-hub.atlassian.net/browse/ROCM-28219); the
rocprofiler-compute side is [AIPROFCOMP-729](https://amd-hub.atlassian.net/browse/AIPROFCOMP-729).

## Symptom

Profiling a multiprocess PC sampling workload aborts during output generation:

```
terminate called after throwing an instance of 'std::out_of_range'
  what():  vector::_M_range_check: __n (which is 3) >= this->size() (which is 3)
```

rocprofv3's error signal handler catches the abort, then re-enters on the
chained signal and hangs. The child never exits, so a parent blocked in
`waitpid` waits forever. That is why the affected test is `skip` and not
`xfail`: the process hangs rather than failing.

## Root cause

`source/lib/output/metadata.cpp:511` sizes the result vector by the *count* of
host functions but indexes it by host-function *id*:

```cpp
_info.resize(_data_v.size() + 1, host_function_info{});   // size == count + 1
for(const auto& itr : _data_v)
    _info.at(itr.first) = itr.second;                     // index == host_function_id
```

That holds only while ids are dense. The id comes from a process-global atomic
at `source/lib/rocprofiler-sdk/code_object/code_object.cpp:1091`:

```cpp
host_data.host_function_id = ++get_host_function_id();
```

The increment sits inside the per-context loop, while the per-symbol
`beg_notified` dedup flag is only set after that loop (line 1112). With two
subscribed `ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT` contexts, every symbol
consumes two ids. A client records N host functions but holds ids up to ~2N, so
`resize(N + 1)` followed by `.at(id)` walks off the end.

rocprofiler-compute is the second client. It subscribes unconditionally at
`src/lib/rocprofiler_compute_tool/rocprofiler_compute_tool.cpp:127` and never
stores host symbols (`src/lib/rocprofiler_compute_tool/sdk_callbacks.cpp:328,337`
handle only `CODE_OBJECT_LOAD` and `DEVICE_KERNEL_SYMBOL_REGISTER`), so it
consumes ids without recording them.

## Why only some workloads trip it

`start_context` is deferred to `on_hsa_runtime_loaded`
(`rocprofiler_compute_tool.cpp:119`), so only symbols registered while both
contexts are live get double-counted.

| Workload | Code object load point | Result |
|---|---|---|
| `rocflop`, `vcopy` | Fat binary in the executable, loaded during HSA bring-up | Ids stay dense |
| `conjugate_gradient` | Two `dlopen`'d libraries, loaded on first launch | Double-counted |
| torch / Triton | JIT compiled at runtime | Double-counted |

Kernel count is not the discriminator: `rocflop` has four `__global__` kernels
and survives, `conjugate_gradient` has two and does not.

This table is inferred from reading the sources above, not measured. The
mechanism and the arithmetic are confirmed by the crash numbers (two kernels,
`resize(3)`, attempted index 3); the load-timing explanation for why `rocflop`
escapes is the best-supported hypothesis but is unverified.

## Suggested fix

In `get_host_symbols()`, size by `max(id) + 1` rather than `count + 1`, or
return an `id -> info` map and drop positional indexing. Separately, assign
`host_function_id` once per symbol rather than once per (symbol, context), which
is the real defect; the vector sizing is what turns it into a crash.

## Re-enabling the test

Once the SDK fix lands, drop the `@pytest.mark.skip` from
`test_multiprocess_pc_sampling_distinct_code_objects` in
`tests/integration/test_pc_sampling.py` and delete this file.
