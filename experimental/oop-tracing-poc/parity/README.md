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
