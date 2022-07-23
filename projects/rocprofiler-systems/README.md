# Omnitrace: Application Profiling, Tracing, and Analysis

[![Ubuntu 18.04 with GCC and MPICH](https://github.com/AMDResearch/omnitrace/actions/workflows/ubuntu-bionic.yml/badge.svg)](https://github.com/AMDResearch/omnitrace/actions/workflows/ubuntu-bionic.yml)
[![Ubuntu 20.04 with GCC, ROCm, and MPI](https://github.com/AMDResearch/omnitrace/actions/workflows/ubuntu-focal.yml/badge.svg)](https://github.com/AMDResearch/omnitrace/actions/workflows/ubuntu-focal.yml)
[![OpenSUSE 15.x with GCC](https://github.com/AMDResearch/omnitrace/actions/workflows/opensuse.yml/badge.svg)](https://github.com/AMDResearch/omnitrace/actions/workflows/opensuse.yml)

> ***[Omnitrace](https://github.com/AMDResearch/omnitrace) is an AMD open source research project and is not supported as part of the ROCm software stack.***

## Overview

AMD Research is seeking to improve observability and performance analysis for software running on AMD heterogeneous systems.
If you are familiar with [rocprof](https://rocmdocs.amd.com/en/latest/ROCm_Tools/ROCm-Tools.html) and/or [uProf](https://developer.amd.com/amd-uprof/),
you will find many of the capabilities of these tools available via Omnitrace in addition to many new capabilities.

Omnitrace is a comprehensive profiling and tracing tool for parallel applications written in C, C++, Fortran, HIP, OpenCL, and Python which execute on the CPU or CPU+GPU.
It is capable of gathering the performance information of functions through any combination of binary instrumentation, call-stack sampling, user-defined regions, and Python interpreter hooks.
Omnitrace supports interactive visualization of comprehensive traces in the web browser in addition to high-level summary profiles with mean/min/max/stddev statistics.
In addition to runtimes, omnitrace supports the collection of system-level metrics such as the CPU frequency, GPU temperature, and GPU utilization, process-level metrics
such as the memory usage, page-faults, and context-switches, and thread-level metrics such as memory usage, CPU time, and numerous hardware counters.

### Data Collection Modes

- Dynamic instrumentation
  - Runtime instrumentation
    - Instrument executable and shared libraries at runtime
  - Binary rewriting
    - Generate a new executable and/or library with instrumentation built-in
- Statistical sampling
  - Periodic software interrupts per-thread
- Process-level sampling
  - Background thread records process-, system- and device-level metrics while the application executes
- Critical trace generation

### Data Analysis

- High-level summary profiles with mean/min/max/stddev statistics
  - Low overhead, memory efficient
  - Ideal for running at scale
- Comprehensive traces
  - Every individual event/measurement
- Critical trace analysis (alpha)

### Parallelism API Support

- HIP
- HSA
- Pthreads
- MPI
- Kokkos-Tools (KokkosP)
- OpenMP-Tools (OMPT)

### GPU Metrics

- GPU hardware counters
- HIP API tracing
- HIP kernel tracing
- HSA API tracing
- HSA operation tracing
- System-level sampling (via rocm-smi)
  - Memory usage
  - Power usage
  - Temperature
  - Utilization

### CPU Metrics

- CPU hardware counters sampling and profiles
- CPU frequency sampling
- Various timing metrics
  - Wall time
  - CPU time (process and/or thread)
  - CPU utilization (process and/or thread)
  - User CPU time
  - Kernel CPU time
- Various memory metrics
  - High-water mark (sampling and profiles)
  - Memory page allocation
  - Virtual memory usage
- Network statistics
- I/O metrics
- ... many more

## Documentation

The full documentation for [omnitrace](https://github.com/AMDResearch/omnitrace) is available at [amdresearch.github.io/omnitrace](https://amdresearch.github.io/omnitrace/).

## Quick Start

### Omnitrace Settings

Generate an omnitrace configuration file using `omnitrace-avail -G omnitrace.cfg`. Optionally, use `omnitrace-avail -G omnitrace.cfg --all` for
a verbose configuration file with descriptions, categories, etc. Modify the configuration file as desired, e.g. enable
[perfetto](https://perfetto.dev/), [timemory](https://github.com/NERSC/timemory), sampling, and process-level sampling by default
and tweak some sampling default values:

```console
# ...
OMNITRACE_USE_PERFETTO         = true
OMNITRACE_USE_TIMEMORY         = true
OMNITRACE_USE_SAMPLING         = true
OMNITRACE_USE_PROCESS_SAMPLING = true
# ...
OMNITRACE_SAMPLING_FREQ        = 50
OMNITRACE_SAMPLING_CPUS        = all
OMNITRACE_SAMPLING_GPUS        = $env:HIP_VISIBLE_DEVICES
```

Once the configuration file is adjusted to your preferences, either export the path to this file via `OMNITRACE_CONFIG_FILE=/path/to/omnitrace.cfg`
or place this file in `${HOME}/.omnitrace.cfg` to ensure these values are always read as the default. If you wish to change any of these settings,
you can override them via environment variables or by specifying an alternative `OMNITRACE_CONFIG_FILE`.

### Omnitrace Executable

The `omnitrace` executable is used to instrument an existing binary.

```shell
omnitrace --help
omnitrace <omnitrace-options> -- <exe-or-library> <exe-options>
```

#### Binary Rewrite

Rewrite the text section of an executable or library with instrumentation:

```shell
omnitrace -o app.inst -- /path/to/app
```

In binary rewrite mode, if you also want instrumentation in the linked libraries, you must also rewrite those libraries.
Example of rewriting the functions starting with `"hip"` with instrumentation in the amdhip64 library:

```shell
mkdir -p ./lib
omnitrace -R '^hip' -o ./lib/libamdhip64.so.4 -- /opt/rocm/lib/libamdhip64.so.4
export LD_LIBRARY_PATH=${PWD}/lib:${LD_LIBRARY_PATH}
```

> ***Verify via `ldd` that your executable will load the instrumented library -- if you built your executable with***
> ***an RPATH to the original library's directory, then prefixing `LD_LIBRARY_PATH` will have no effect.***

Once you have rewritten your executable and/or libraries with instrumentation, you can just run the (instrumented) executable
or exectuable which loads the instrumented libraries normally, e.g.:

```shell
./app.inst
```

If you want to re-define certain settings to new default in a binary rewrite, use the `--env` option. This `omnitrace` option
will set the environment variable to the given value but will not override it. E.g. the default value of `OMNITRACE_PERFETTO_BUFFER_SIZE_KB`
is 1024000 KB (1 GiB):

```shell
# buffer size defaults to 1024000
omnitrace -o app.inst -- /path/to/app
./app.inst
```

Passing `--env OMNITRACE_PERFETTO_BUFFER_SIZE_KB=5120000` will change the default value in `app.inst` to 5120000 KiB (5 GiB):

```shell
# defaults to 5 GiB buffer size
omnitrace -o app.inst --env OMNITRACE_PERFETTO_BUFFER_SIZE_KB=5120000 -- /path/to/app
./app.inst
```

```shell
# override default 5 GiB buffer size to 200 MB
export OMNITRACE_PERFETTO_BUFFER_SIZE_KB=200000
./app.inst
```

#### Runtime Instrumentation

Runtime instrumentation will not only instrument the text section of the executable but also the text sections of the
linked libraries. Thus, it may be useful to exclude those libraries via the `-ME` (module exclude) regex option
or exclude specific functions with the `-E` regex option.

```shell
omnitrace -- /path/to/app
omnitrace -ME '^(libhsa-runtime64|libz\\.so)' -- /path/to/app
omnitrace -E 'rocr::atomic|rocr::core|rocr::HSA' --  /path/to/app
```

### Visualizing Perfetto Results

Visit [ui.perfetto.dev](https://ui.perfetto.dev) in your browser and open up the `.proto` file(s) created by omnitrace.

![omnitrace-perfetto](source/docs/images/omnitrace-perfetto.png)

![omnitrace-rocm](source/docs/images/omnitrace-rocm.png)

![omnitrace-rocm-flow](source/docs/images/omnitrace-rocm-flow.png)

![omnitrace-user-api](source/docs/images/omnitrace-user-api.png)

## Using Perfetto tracing with System Backend

Perfetto tracing with the system backend supports multiple processes writing to the same
output file. Thus, it is a useful technique if Omnitrace is built with partial MPI support
because all the perfetto output will be coalesced into a single file. The
installation docs for perfetto can be found [here](https://perfetto.dev/docs/contributing/build-instructions).
If you are building omnitrace from source, you can configure CMake with `OMNITRACE_INSTALL_PERFETTO_TOOLS=ON`
and the `perfetto` and `traced` applications will be installed as part of the build process. However,
it should be noted that to prevent this option from accidentally overwriting an existing perfetto install,
all the perfetto executables installed by omnitrace are prefixed with `omnitrace-perfetto-`, except for the `perfetto`
executable, which is just renamed `omnitrace-perfetto`.

Enable `traced` and `perfetto` in the background:

```shell
pkill traced
traced --background
perfetto --out ./omnitrace-perfetto.proto --txt -c ${OMNITRACE_ROOT}/share/omnitrace.cfg --background
```

> ***NOTE: if the perfetto tools were installed by omnitrace, replace `traced` with `omnitrace-perfetto-traced` and***
> ***`perfetto` with `omnitrace-perfetto`.***

Configure omnitrace to use the perfetto system backend:

```shell
export OMNITRACE_PERFETTO_BACKEND=system
```

And finally, execute your instrumented application. Either the binary rewritten application:

```shell
omnitrace -o ./myapp.inst -- ./myapp
./myapp.inst
```

Or with runtime instrumentation:

```shell
omnitrace -- ./myapp
```
