# Changelog for ROCprofiler

Full documentation for ROCprofiler is available at
[docs.amd.com](https://docs.amd.com/bundle/ROCm-Profiling-Tools-User-Guide-v5.3)

As of ROCm 5.5, the ROCm Profiler will not use terminologies like `rocmtools` or
`rocsight` to describe `rocrofiler` as was done in ROCm 5.4. To identify the
separation of the two versions of `rocprofiler`, the terms `rocprofilerV1` and
`rocprofilerV2` will be used. The `rocprofilerV2` API is currently considered a
beta release and subject to changes in future releases.

## ROCprofiler for rocm 5.4.4

In ROCm 5.4 the naming of the ROCm Profiler related files is:

  | ROCm 5.4        | rocprofilerv1                       | rocmtools                       |
  |-----------------|-------------------------------------|---------------------------------|
  | **Tool script** | `bin/rocprof`                       | `bin/rocsight`                  |
  | **API include** | `include/rocprofiler/rocprofiler.h` | `include/rocmtools/rocmtools.h` |
  | **API library** | `lib/librocprofiler64.so.1`         | `lib/librocmtools.so.1`         |

The ROCm Profiler Tool that uses `rocprofilerV1` can be invoked using the
following command:

```sh
$ rocprof …
```

To write a custom tool based on the `rocprofilerV1` API do the following:

```C
main.c:
#include <rocprofiler/rocprofiler.h> // Use the rocprofilerV1 API
int main() {
  // Use the rocprofilerV1 API
  return 0;
}
```

This can be built in the following manner:

```sh
$ gcc main.c -I/opt/rocm-5.4.4/include -L/opt/rocm-5.4.4/lib -lrocprofiler64
```

The resulting `a.out` will depend on
`/opt/rocm-5.4.4/lib/librocprofiler64.so.1`.

The ROCm Profiler that uses `rocprofilerV2` API can be invoked using the
following command:

```sh
$ rocsight …
```

To write a custom tool based on the `rocmtools` API do the following:

```C
main.c:
#include <rocmtools/rocmtools.h> // Use the rocmtools API
int main() {
  // Use the rocmtools API
  return 0;
}
```

This can be built in the following manner:

```sh
$ gcc main.c -I/opt/rocm-5.4.4/include -L/opt/rocm-5.4.4/lib -lrocmtools
```

The resulting `a.out` will depend on `/opt/rocm-5.4.4/lib/librocmtools.so.1`.

## ROCprofiler for rocm 5.5.0

In ROCm 5.5 the `rocprofilerv1` and `rocprofilerv2` include and library files
are merged into single files. The `rocmtools` available in ROCm 5.4 is also
available in ROCm 5.5 but is deprecated and will be removed in a future release.

  | ROCm 5.5        | rocprofilerv1                       | rocprofilerv2                       | rocmtools *(deprecated)*        |
  |-----------------|-------------------------------------|-------------------------------------|---------------------------------|
  | **Tool script** | `bin/rocprof`                       | `bin/rocprofv2`                     | `bin/rocsight`                  |
  | **API include** | `include/rocprofiler/rocprofiler.h` | `include/rocprofiler/rocprofiler.h` | `include/rocmtools/rocmtools.h` |
  | **API library** | `lib/librocprofiler64.so.1`         | `lib/librocprofiler64.so.1`         | `lib/librocmtools.so.1`         |


The ROCm Profiler Tool that uses `rocprofilerV1` can be invoked using the
following command:

```sh
$ rocprof …
```

To write a custom tool based on the `rocprofilerV1` API it is necessary to
define the macro `ROCPROFILER_V1`:

```C
main.c:
#define ROCPROFILER_V1
#include <rocprofiler/rocprofiler.h>
int main() {
  // Use the rocprofilerV1 API
  return 0;
}
```

This can be built in the following manner:

```sh
$ gcc main.c -I/opt/rocm-5.5.0/include -L/opt/rocm-5.5.0/lib -lrocprofiler64
```

The resulting `a.out` will depend on
`/opt/rocm-5.5.0/lib/librocprofiler64.so.1`.

The ROCm Profiler that uses `rocprofilerV2` API can be invoked using the
following command:

```sh
$ rocprofv2 …
```

To write a custom tool based on the `rocprofilerV2` API do the following:

```C
main.c:
#include <rocprofiler/rocprofiler.h>
int main() {
  // Use the rocprofilerV2 API
  return 0;
}
```

This can be built in the following manner:

```sh
$ gcc main.c -I/opt/rocm-5.5.0/include -L/opt/rocm-5.5.0/lib -lrocprofiler64
```

The resulting `a.out` will depend on
`/opt/rocm-5.5.0/lib/librocprofiler64.so.1`.

## ROCprofiler for rocm 5.6.0

In ROCm 5.6 the `rocprofilerv1` and `rocprofilerv2` include and library files of
ROCm 5.5 are split into separate files. The `rocmtools` files that were
deprecated in ROCm 5.5 have been removed.

  | ROCm 5.6        | rocprofilerv1                       | rocprofilerv2                          |
  |-----------------|-------------------------------------|----------------------------------------|
  | **Tool script** | `bin/rocprof`                       | `bin/rocprofv2`                        |
  | **API include** | `include/rocprofiler/rocprofiler.h` | `include/rocprofiler/v2/rocprofiler.h` |
  | **API library** | `lib/librocprofiler.so.1`           | `lib/librocprofiler.so.2`              |

The ROCm Profiler Tool that uses `rocprofilerV1` can be invoked using the
following command:

```sh
$ rocprof …
```

To write a custom tool based on the `rocprofilerV1` API do the following:

```C
main.c:
#include <rocprofiler/rocprofiler.h> // Use the rocprofilerV1 API
int main() {
  // Use the rocprofilerV1 API
  return 0;
}
```

This can be built in the following manner:

```sh
$ gcc main.c -I/opt/rocm-5.6.0/include -L/opt/rocm-5.6.0/lib -lrocprofiler64
```

The resulting `a.out` will depend on
`/opt/rocm-5.6.0/lib/librocprofiler64.so.1`.

The ROCm Profiler that uses `rocprofilerV2` API can be invoked using the
following command:

```sh
$ rocprofv2 …
```

To write a custom tool based on the `rocprofilerV2` API do the following:

```C
main.c:
#include <rocprofiler/v2/rocprofiler.h> // Use the rocprofilerV2 API
int main() {
  // Use the rocprofilerV2 API
  return 0;
}
```

This can be built in the following manner:

```sh
$ gcc main.c -I/opt/rocm-5.6.0/include -L/opt/rocm-5.6.0/lib -lrocprofiler64v2
```

The resulting `a.out` will depend on
`/opt/rocm-5.6.0/lib/librocprofiler64.so.2`.

### Optimized
- Improved Test Suite
### Added
- 'end_time' need to be disabled in roctx_trace.txt
- support for hsa_amd_memory_async_copy_on_engine API function trace

### Fixed
- rocprof in ROcm/5.4.0 gpu selector broken.
- rocprof in ROCm/5.4.1 fails to generate kernel info.
- rocprof clobbers LD_PRELOAD.

## ROCprofiler for rocm 5.7.0
### Navi support
Rocprofiler for ROCm 5.7 added support for counter collection (PMC) and advanced thread tracing (ATT) for Navi21 and Navi31 GPUs.
- On Navi3x, counter collection requires the GPU to be in a stable power state. See README.md for instructions. HIP RT in ATT not yet supported.

### Changed
- ATT analysis will not run by default. For ATT to have the same behaviour as 5.5, use --plugin att <as.s> --mode network
- Kernel Names are now removed from HIP API records, users of the API can get the kernel names from the corresponding HIP Dispatch OPS using the correlation ID, this change was done to optimize and to manage the data copied.

### Optimized
- ATT json filesizes
- Now profiler autocorrects user input errors for pmc and throws exception for wrong input with this message:"Bad input metric. usage --> pmc: [counter1] [counter2]"

### Added
- Every API trace in V2 reported synchronously will have two records, one for Enter phase and for Exit phase
- File Plugin now reports the HSA OPS operation kind as part of the output text
- MI300 counters support for rocprof v1 and v2.
- Limiting file name sizes for ATT plugin.
- Support for MI300 XCC modes for rocprof v2.
- MI300 individual XCC counters dumped per-xcc as separate records but with same record-id and kernel dispatch info
- Naming for MPI ranks. Filenames containing "%rank" are replaced by variables "MPI_RANK", "OMPI_COMM_WORLD_RANK" or "MV2_COMM_WORLD_RANK".
- MPI Rank will appear in perfetto track names.
- SE_MASK parameter in ATT, a binary mask specifying for which shader engines to run ATT.
  On GFX9, SEs are masked out completely. On Navi only part of the data is masked.
  The use of SE_MASK=0x1 is heavily encouraged to avoid packet lost events.
- "--mode file" option in ATT, which allows for parsed files to be stored. Run python3 httpserver.py from within ./UI/ to view files locally.
- "ROCPROFILER_MAX_ATT_PROFILES" environment variable can be set. Previously fixed at 16, now the default is 1.
- Increased ATT buffer size per collection to 1GB.
- File plugin is splitted to File & CLI plugins, CLI plugin is responsible for showing results on the terminal screen and will be automatically the choice if no -d option given in rocprof, File plugin on the other hand is responsible for writing the output results in files if -d option is given.
- Structure of the results is different for both CLI & File plugin; File plugin will make sure every type of result is in a separate file, starting by specifying the header; CLI plugin will have the records in the old way.
Example for file plugin output:
  ```
  Dispatch_ID,GPU_ID,Queue_ID,Queue_Index,PID,TID,GRD,WGR,LDS,SCR,Arch_VGPR,ACCUM_VGPR,SGPR,Wave_Size,SIG,OBJ,Kernel_Name,Start_Timestamp,End_Timestamp,Correlation_ID,GRBM_COUNT

  1,4,1,1,1584730,1584730,10,10,0,0,8,0,16,64,140464978048000,1,"helloworld(char*, char*) (.kd)",0,140469300947216,33,12637.000000
  ```
  ```
  Domain,Function,Kernel_Name,Start_Timestamp,End_Timestamp,Correlation_ID

  HIP_API_DOMAIN,hipGetDeviceProperties,,316678074094190,316678074098929,1
  HIP_API_DOMAIN,hipMalloc,,316678074105702,316678074130851,2
  HIP_API_DOMAIN,hipMalloc,,316678074131382,316678074136111,3
  ```
- Removing Record IDs from tracer records in CLI plugin.
- Added Flush Interval and Trace Period functionality, where --flush-interval <time_in_ms>, for flushing the buffers every given interval by the user, and --trace-period <delay>:<trace_time>, where delay is the time to wait before starting session, and trace_time is the time between every start and stop session. For more details please refer to the ROCProfV2 tool usage document.

### Fixed
- Samples are fixed to show the new usage of phases.
- Plugin option validates the plugin names.
- Fixing rocsys, for rocsys options, rocsys -h can be called
- "--output-file" option ignored when no output folder was specified.
- Perfetto crash when using ROCTX and/or no output file specified.
- Parsing of the getpc, setpc and swappc instructions with registers loaded from scratch space.
- Some browsers caching ATT data from older kernels.
- Navi2x GPUs required the first counter to be GRBM. This is fixed in 5.7.
- If ROCPROFILER_METRICS_PATH environment variable is not set, the counters xml path will be taken from the following path (../libexec/rocprofiler/counters/derived_counters.xml) which is relative to librocprofiler64.so.2.0.0
