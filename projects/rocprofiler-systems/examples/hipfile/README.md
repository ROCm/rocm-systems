# hipFile Examples

This directory contains two small hipFile workloads. Both need the hipFile
runtime (`hipfile.h` and `libhipfile`). They are built as `hipfile-trace` and
`hipfile-io`.

## hipfile-trace — API tracing

A deliberately minimal host-side driver workout whose only purpose is to put
hipFile API calls into the `rocm_hipfile_api` category. It queries the version,
looks up error strings, gets and sets parameters, and opens and closes the
driver. It does not need a GPU file mapping or a storage device.

Source: `hipfile_trace.cpp`.

## hipfile-io — I/O telemetry

A short read/write loop against a GPU buffer and a regular file. The profiler
samples hipFile's in-process stats and reports per-GPU counters
(`GPU [<N>] Storage <metric> (S)`). The file is opened without `O_DIRECT` so
`open(2)` itself does not require an `O_DIRECT`-capable filesystem. hipFile
still tries to reopen the file through `/proc/self/fd` with `O_DIRECT` at
`hipFileHandleRegister`, and uses the GPU-direct fast path when that succeeds.
The POSIX fallback is used only when the reopen fails. The example is meant to
exercise telemetry, not to pin a particular I/O path.

Source: `hipfile-io.cpp`. Built whenever the hipFile runtime is found (same as
`hipfile-trace`). Profiling its I/O counters still requires hipFile telemetry
support in the profiler binaries (`ROCPROFSYS_BUILD_HIPFILE=ON`, or `AUTO` with a
new enough package).

Usage: `hipfile-io [FILE] [GPUID] [SECONDS]`

| Argument  | Default          | Description                                        |
| --------- | ---------------- | -------------------------------------------------- |
| `FILE`    | `hipfile-io.bin` | Scratch file in cwd; must not already exist        |
| `GPUID`   | `0`              | GPU ordinal to run on                              |
| `SECONDS` | `5`              | Duration of the I/O loop, at least 1               |

The workload writes and reads back a 1 MiB buffer per iteration, pausing 20 ms
between iterations so the profiler's periodic sampler observes several points.
It removes the scratch file before exiting, including when it fails during
setup, so repeated runs leave nothing behind.

Because the scratch file is deleted on exit, `FILE` is created with `O_EXCL` and
the run is refused if it already exists — a mistyped path cannot overwrite and
then delete something you meant to keep. If a previous run was killed before it
could clean up, remove the leftover file (or pass a different path) to run again.

Point `FILE` at the filesystem you want to measure. The default is deliberately
the current directory rather than `/tmp`, which on many systems is `tmpfs` and
therefore memory-backed — measuring it would say nothing about storage I/O.

## Prerequisites

- CMake 3.25+
- HIP runtime
- hipFile runtime library (`hipfile.h` and `libhipfile`)
- ROCProfiler-SDK 1.3.5 or later for `hipfile-trace` API tracing. Both
  examples build and run without it; the `hipfile_api` tracing domain is only
  available when profiling against 1.3.5 or newer.

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/hipfile -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target hipfile-trace
cmake --build <build_dir> --target hipfile-io
```

## Running

```bash
# Host-side hipFILE driver APIs (no arguments)
./hipfile-trace

# GPU-buffer read/write loop for telemetry, using the defaults
./hipfile-io

# ... or choose the file, GPU, and duration explicitly
./hipfile-io ./hipfile-io.bin 0 5
```

`hipfile-io` reports how much work it completed, which is a quick way to confirm
the workload actually ran before looking at a trace:

```text
hipfile-io: pid=12345 looping hipFile I/O for 5s
hipfile-io: completed after 208 write+read iterations
```

A read or write that fails or transfers fewer bytes than requested stops the loop,
reports the returned and expected byte counts on stderr, and exits non-zero:

```text
hipfile-io: pid=12345 looping hipFile I/O for 5s
hipFileWrite returned -5, expected 1048576
hipfile-io: failed after 12 write+read iterations
```

`hipfile-trace` prints the detected hipFILE version:

```text
hipFILE version: X.Y.Z
```

## Profiling with rocprofiler-systems

### API tracing (`hipfile-trace`)

Add `hipfile_api` to `ROCPROFSYS_ROCM_DOMAINS` (shorthand: `hipfile`):

```bash
ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,hipfile_api \
ROCPROFSYS_TRACE=true \
rocprof-sys-run -- ./hipfile-trace
```

The captured hipFILE API calls appear in the Perfetto trace and the RocPD
database under the `rocm_hipfile_api` category.

> **Note:** Requesting `hipfile_api` against a ROCProfiler-SDK older than 1.3.5
> results in an `unsupported ROCPROFSYS_ROCM_DOMAINS value: hipfile_api` error,
> the same as any other unavailable domain.

### I/O telemetry (`hipfile-io`)

Enable process sampling and hipFile telemetry. See
[hipFile GPU-direct storage I/O telemetry](../../docs/how-to/hipfile-telemetry.rst).

```bash
ROCPROFSYS_USE_HIPFILE=ON \
ROCPROFSYS_USE_PROCESS_SAMPLING=ON \
rocprof-sys-run -- ./hipfile-io ./hipfile-io.bin 0 5
```

Counter tracks appear as `GPU [<N>] Storage <metric> (S)` in Perfetto and RocPD.
