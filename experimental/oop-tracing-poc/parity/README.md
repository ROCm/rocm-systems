# HIP-trace parity with rocprofv3

Runs an application under `rocprofv3 --hip-trace` while an LTTng session records the `rocm_hip`
probes (ROCm/rocm-systems#15122) in the same process, converts the probe trace to rocprofv3's
`hip_api_trace.csv` (`../correlator/join.py --rocprofv3-csv`) and compares the two
(`../correlator/compare_rocprofv3.py`).

    run_both.sh <out-dir> <app> [args...]

Environment (paths are set at the top of `run_both.sh`):
- `clr-trace-build:7.14-parity` image: the TheRock 7.14 image with LTTng and `python3-bt2`.
- TheRock nightly 10.3 runtime (ROCr, comgr 3.5, rocprofiler-sdk, rocprofv3), unpacked, not
  installed; only `libamdhip64` is replaced by a CLR build with `HIP_ENABLE_ROCM_TRACE=ON`.

Test applications:
- `vadd.cpp`: one thread, one stream: allocation, async copies, three kernel launches.
- `mt.cpp [threads] [streams] [iters] [n] [graph]`: each thread with its own streams, events,
  pinned memory, memset, copies, kernel launches, and a captured graph launched every iteration.

Expected differences, which the comparison reports but does not fail on:
- `__hipUnregisterFatBinary` at exit, after rocprofv3 has finalized.
- Calls other threads make while the first call is initializing HIP, before rocprofv3 has wrapped
  the runtime table: the probes record them, rocprofv3 does not.
- Calls rocprofv3 itself makes through the HIP table inside its wrappers (`hipStreamGetDevice` in
  `hipGraphLaunch`, `hipStreamIsCapturing` in `hipEventRecord`): only present when both run.
- Correlation ids: both number calls in start order, so the extra calls above shift them.
