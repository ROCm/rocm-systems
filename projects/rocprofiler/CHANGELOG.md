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
### Changed
- ATT analysis will not run by default. For ATT to have the same behaviour as 5.5, use --plugin att <as.s> --mode network
### Added
- 'end_time' need to be disabled in roctx_trace.txt
- support for hsa_amd_memory_async_copy_on_engine API function trace

### Fixed
- rocprof in ROcm/5.4.0 gpu selector broken.
- rocprof in ROCm/5.4.1 fails to generate kernel info.
- rocprof clobbers LD_PRELOAD.

## ROCprofiler for rocm 5.7.0
### Optimized
### Added
- Every API trace in V2 reported synchronously will have two records, one for Enter phase and for Exit phase
- File Plugin now reports the HSA OPS operation kind as part of the output text
- MI300 counters support for rocprof v1 and v2.
- Limiting file name sizes for ATT plugin.
- Support for MI300 XCC modes for rocprof v2.
- MI300 individual XCC counters dumped per-xcc as separate records but with same record-id and kernel dispatch info
### Fixed
- Samples are fixed to show the new usage of phases.
- Plugin option validates the plugin names.
- Fixing rocsys, for rocsys options, rocsys -h can be called
- "--output-file" option ignored when no output folder was specified.
