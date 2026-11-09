# HIP-trace parity with rocprofv3

Runs an application under `rocprofv3 --hip-trace` while an LTTng session records the `rocm_hip`
probes (ROCm/rocm-systems#15122) in the same process, converts the probe trace to rocprofv3's
`hip_api_trace.csv` (`../correlator/join.py --rocprofv3-csv`) and compares the two
(`../correlator/compare_rocprofv3.py`).

    run_both.sh <out-dir> <app> [args...]

Environment (paths are set at the top of `run_both.sh`):
- `clr-trace-build:7.14-parity` image: the TheRock 7.14 image with LTTng and `python3-bt2`.
- TheRock nightly 10.3 (runtime, profiler, BLAS and their dev packages), unpacked, not installed.
- `libamdhip64`: the CLR branch with `HIP_ENABLE_ROCM_TRACE=ON` and `ROCM_KPACK_ENABLED=ON`, built
  against the nightly's dev packages (comgr 3.5, ROCr, rocm-kpack). It is bind-mounted over the
  nightly's library file, because the nightly binaries have `RPATH=$ORIGIN/../lib` and ignore
  `LD_LIBRARY_PATH`.
- A HIP built without kpack fails every kernel launch from TheRock's libraries with
  `hipErrorNotSupported` ("Kpack binary detected but ROCM_KPACK_ENABLED=OFF").

Results (gfx900, one run each):

| Workload | rocprofv3 calls | matched | LTTng lost |
|---|---|---|---|
| `vadd` | 27 | 27 | 0 |
| `mt 8 4 200` (threads, streams, graphs) | 40,469 | 40,469 | 0 |
| `rocblas-bench` GEMM, 200 iterations | 8,360 | 8,360 | 0 |
| `rocblas-test --yaml rocblas_smoke.yaml` (4,547 tests, 20 APIs) | 243,570 | 243,570 | 0 |

Test applications:
- `vadd.cpp`: one thread, one stream: allocation, async copies, three kernel launches.
- `mt.cpp [threads] [streams] [iters] [n] [graph]`: each thread with its own streams, events,
  pinned memory, memset, copies, kernel launches, and a captured graph launched every iteration.

Expected differences, which the comparison reports but does not fail on:
- Library teardown at exit (`__hipUnregisterFatBinary`, `hipModuleUnload`), after rocprofv3 has
  finalized.
- Calls other threads make while the first call is initializing HIP, before rocprofv3 has wrapped
  the runtime table: the probes record them, rocprofv3 does not.
- Calls rocprofv3 itself makes through the HIP table inside its wrappers (`hipStreamGetDevice` in
  `hipGraphLaunch`, `hipStreamIsCapturing` in `hipEventRecord`): only present when both run.
- Correlation ids: both number calls in start order, so the extra calls above shift them.

## Separate runs and overhead (`overhead.sh`)

    overhead.sh <out-dir> <reps> <app> [args...]

Runs the application alone in each mode, median wall time of `<reps>` runs:
`baseline` (nightly HIP, no probes), `probes-idle` (probes built in, no LTTng session),
`probes-rec` (LTTng session recording), `rocprofv3` (nightly HIP, `rocprofv3 --hip-trace`).

Call counts, `mt 8 4 200` run separately under each tool: identical per function (40,468 calls),
except that rocprofv3 records 1 of the 8 threads' first `hipSetDevice` (the probes record all 8:
the other threads enter while HIP initializes, before rocprofv3 wraps the table) and the
`__hipUnregisterFatBinary` at exit. The calls rocprofv3 makes itself (`hipStreamGetDevice`,
`hipStreamIsCapturing`) are absent when it is not in the process.

Wall time added over `baseline`, `mt 8 4 <iters>` (8 threads), median of 5:

| Mode | 10,476 calls | 160,476 calls | per call (slope) | fixed |
|---|---|---|---|---|
| probes-idle | +26 ms | +8 ms | ~0 (noise) | within noise |
| probes-rec | +161 ms | +140 ms | not measurable | ~150 ms |
| rocprofv3 | +241 ms | +1,060 ms | 5.5 us (wall) | ~180 ms |

The probes' fixed cost is liblttng-ust's registration wait at process start: with
`LTTNG_UST_REGISTER_TIMEOUT=0` the run takes the baseline time, but the first ~76 events
(`__hipRegisterFatBinary`, the first `hipSetDevice`) are lost before registration completes.

## Arguments (`compare_args.py`)

`run_both.sh` also writes rocprofv3's rocpd database and, with `join.py --calls-json`, every probe
call with its enter and exit fields. `compare_args.py <rocpd.db> <calls.json> <hip include dir>`
pairs the calls as before and compares each argument rocprofv3 recorded (`rocpd_arg`: position,
type, name, value as text) after normalizing the text: pointers and `(null)` to integers, enums to
their values from the HIP headers, `dim3` to `<name>_x/_y/_z`. rocprofv3 records arguments at entry
only, so the probes' exit fields (result, outputs) are not compared.

| Workload | argument values | equal | APIs |
|---|---|---|---|
| `vadd` | 105 | 104 | 12 |
| `mt 4 2 50` | 10,768 | 10,767 | 25 |
| `rocblas-test` smoke | 840,012 | 832,291 (all but `deviceFunction`) | 17 |

Differences:
- `__hipRegisterFunction(..., char* deviceFunction, ...)`: rocprofv3 prints the kernel name; the
  probe records the address, because the generator treats only `const char*` as a string. Fix in
  the CLR annotations (#15122): mark it as a string, uncapped, like `deviceName`.
- Names, values equal by position: `hipExtModuleLaunchKernel` `f`/`hStream` (the public
  declaration in `hip_ext.h`) are `func`/`stream` in rocprofv3; `hipGetDevicePropertiesR0600`
  `device` is `deviceId` in rocprofv3 and in `hip_runtime_api.h`. That declaration is inside
  `#if HIP_TARGET_API_VERSION >= 600`, which the generator's parse does not enter, so it took the
  dispatch-table typedef's name (#15087: define the version when parsing).
