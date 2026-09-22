# Profile support and operations

Load this reference only when selecting hardware-specific settings,
troubleshooting a collection, or maintaining the skill.

## Support matrix

| Area | Supported scope | Validation source |
|---|---|---|
| OS | Linux distributions supported by the matching ROCm release | ROCm system requirements |
| GPUs | `gfx908`, `gfx90a`, `gfx940`, `gfx941`, `gfx942`, `gfx950`, `gfx1150`–`gfx1153`, `gfx1250` | Shipped profile/analysis configs |
| Applications | HIP/ROCm executables and MPI/multi-process applications | Profile integration tests |
| Annotations | ROCTX markers and ranges through `libroctx64.so` | Marker-trace tests |
| Collection APIs | rocprofiler-sdk through `rocprof-compute`; rocpd backend | Profile/unit tests |
| PyTorch | Experimental operator trace with PyTorch 2.13 or 2.14 | Torch-trace coverage tests |
| Triton | Experimental operator trace; version support follows installed release help | Triton integration tests |
| Other frameworks | Profile their emitted HIP kernels; no framework-level attribution is promised | Kernel counter path |

Confirm support on the installed version rather than assuming:

```bash
rocprof-compute --version
rocprof-compute profile --help
rocprof-compute --list-blocks <arch>
rocprof-compute --list-metrics <arch>
rocprofv3-avail info --pc-sampling
```

## Dependencies

- Use the rocprofiler-compute version shipped with the installed ROCm release.
  The ROCm 10.2 target is rocprofiler-compute 3.10.
- Profile mode supports Python 3.8 or newer. Analyze mode requires Python 3.9
  or newer.
- The native counter tool requires ROCm 7.x or newer. Older installations may
  use the legacy collector and expose fewer capabilities.
- The AMD GPU, KFD driver, rocprofiler-sdk runtime, and firmware must come from
  a compatible ROCm installation. Do not prescribe a firmware number: the
  collector validates device-specific minimums and reports an actionable
  error.
- The user must have access to the GPU device nodes and performance counters.
- ROCTX annotation requires headers and `libroctx64.so` from the matching ROCm
  installation.

## Detailed collection options

### Metric sets

There is no `--single-pass` flag. Use a topic-focused metric set:

```bash
rocprof-compute profile --list-sets
rocprof-compute profile --name <name> --set <set_name> -- <application>
```

`--set` cannot be combined with `--block`, `--roof-only`, or `--bench-only`.

### PC sampling

PC sampling is experimental. Stochastic sampling requires hardware support
(MI300 or newer); `host_trap` is the fallback.

```bash
rocprofv3-avail info --pc-sampling

rocprof-compute profile \
    --experimental \
    --name <name> \
    --pc-sampling \
    --pc-sampling-method stochastic \
    --pc-sampling-interval <interval-from-query> \
    -- <application>
```

Do not hardcode an interval for unknown hardware. Query valid configurations
with `rocprofv3-avail info --pc-sampling`. Stochastic intervals are cycles and
must be powers of two; host-trap intervals are microseconds.

### ROCTX

ROCTX is captured automatically when the application links `libroctx64.so`.

```cpp
#include <roctx.h>

roctxRangePush("gemm_compute_phase");
hipLaunchKernelGGL(my_gemm_kernel, grid, block, 0, stream, ...);
hipDeviceSynchronize();
roctxRangePop();
```

After profiling, use `analyze --list-stats` and select the 1-based dispatch
IDs enclosed by the range.

### Retaining raw rocpd

Profile mode always collects through rocpd. It normally converts the capture
into workload CSV files and removes the temporary database.

```bash
rocprof-compute profile \
    --name <name> \
    --retain-rocpd-output \
    -- <application>
```

`--retain-rocpd-output` is deprecated because a future release will retain
databases by default. Use it only when the raw SQLite database is explicitly
needed.

## Known issues and safeguards

- `--device` chooses a GPU only for standalone roofline microbenchmarks. Use
  `HIP_VISIBLE_DEVICES` to constrain an application profile.
- Do not use removed options such as profile `--path`, `--kernel-names`,
  `--single-pass`, `--list-devices`, `--timeout`, or
  `--format-rocprof-output`.
- Multi-pass profiling replays the application. Avoid it for nondeterministic
  or destructive workloads unless replay is safe; use a metric set as the
  named fallback.
- Tiny kernels can produce zero or unstable counter values. Increase workload
  iterations without changing the production code path.
- Profiling into a non-empty directory fails unless `--overwrite` is passed.
  Prefer a new directory; overwrite only when replacement is intentional.
- PC sampling, framework traces, and their option names may change while
  experimental. Re-check `--experimental --help` on each release.
- Do not continue after a firmware, permission, ROCm-version, or standalone
  application failure.

## Excluded workflows

This skill does not cover Windows, offline/air-gapped setup, GUIs/IDEs,
non-public NPI data, CUDA tools, or generating/replacing kernel source code.
