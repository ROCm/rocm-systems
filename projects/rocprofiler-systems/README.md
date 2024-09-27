# ROCm Systems Profiler: Application Profiling, Tracing, and Analysis

[![Ubuntu 20.04 with GCC, ROCm, and MPI](https://github.com/ROCm/rocprofiler-systems/actions/workflows/ubuntu-focal.yml/badge.svg)](https://github.com/ROCm/rocprofiler-systems/actions/workflows/ubuntu-focal.yml)
[![Ubuntu 22.04 (GCC, Python, ROCm)](https://github.com/ROCm/rocprofiler-systems/actions/workflows/ubuntu-jammy.yml/badge.svg)](https://github.com/ROCm/rocprofiler-systems/actions/workflows/ubuntu-jammy.yml)
[![OpenSUSE 15.x with GCC](https://github.com/ROCm/rocprofiler-systems/actions/workflows/opensuse.yml/badge.svg)](https://github.com/ROCm/rocprofiler-systems/actions/workflows/opensuse.yml)
[![RedHat Linux (GCC, Python, ROCm)](https://github.com/ROCm/rocprofiler-systems/actions/workflows/redhat.yml/badge.svg)](https://github.com/ROCm/rocprofiler-systems/actions/workflows/redhat.yml)
[![Installer Packaging (CPack)](https://github.com/ROCm/rocprofiler-systems/actions/workflows/cpack.yml/badge.svg)](https://github.com/ROCm/rocprofiler-systems/actions/workflows/cpack.yml)
[![Documentation](https://github.com/ROCm/rocprofiler-systems/actions/workflows/docs.yml/badge.svg)](https://github.com/ROCm/rocprofiler-systems/actions/workflows/docs.yml)

> [!NOTE]
> Perfetto validation is done with trace_processor v46.0, as there is a known issue with v47.0.
If you are experiencing problems viewing your trace in the latest version of [Perfetto](http://ui.perfetto.dev), then try using [Perfetto UI v46.0](https://ui.perfetto.dev/v46.0-35b3d9845/#!/).

## Overview

ROCm Systems Profiler (rocprof-sys), formerly Omnitrace, is a comprehensive profiling and tracing tool for parallel applications written in C, C++, Fortran, HIP, OpenCL, and Python which execute on the CPU or CPU+GPU.
It is capable of gathering the performance information of functions through any combination of binary instrumentation, call-stack sampling, user-defined regions, and Python interpreter hooks.
ROCm Systems Profiler supports interactive visualization of comprehensive traces in the web browser in addition to high-level summary profiles with mean/min/max/stddev statistics.
In addition to runtimes, ROCm Systems Profiler supports the collection of system-level metrics such as the CPU frequency, GPU temperature, and GPU utilization, process-level metrics
such as the memory usage, page-faults, and context-switches, and thread-level metrics such as memory usage, CPU time, and numerous hardware counters.

> [!NOTE]
> Full documentation is available at [ROCm Systems Profiler documentation](https://rocm.docs.amd.com/projects/omnitrace/en/latest/index.html) in an organized, easy-to-read, searchable format.
The documentation source files reside in the [`/docs`](/docs) folder of this repository. For information on contributing to the documentation, see
[Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html)

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
- Causal profiling
  - Quantifies the potential impact of optimizations in parallel codes

### Data Analysis

- High-level summary profiles with mean/min/max/stddev statistics
  - Low overhead, memory efficient
  - Ideal for running at scale
- Comprehensive traces
  - Every individual event/measurement
- Application speedup predictions resulting from potential optimizations in functions and lines of code (causal profiling)

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

## Quick Start

### Installation

- Visit [Releases](https://github.com/ROCm/rocprofiler-systems/releases) page
- Select appropriate installer (recommendation: `.sh` scripts do not require super-user priviledges unlike the DEB/RPM installers)
  - If targeting a ROCm application, find the installer script with the matching ROCm version
  - If you are unsure about your Linux distro, check `/etc/os-release` or use the `rocprof-sys-install.py` script

If the above recommendation is not desired, download the `rocprof-sys-install.py` and specify `--prefix <install-directory>` when
executing it. This script will attempt to auto-detect a compatible OS distribution and version.
If ROCm support is desired, specify `--rocm X.Y` where `X` is the ROCm major version and `Y`
is the ROCm minor version, e.g. `--rocm 5.4`.

```console
wget https://github.com/ROCm/rocprofiler-systems/releases/latest/download/rocprof-sys-install.py
python3 ./rocprof-sys-install.py --prefix /opt/rocprof-sys/rocm-5.4 --rocm 5.4
```

See the [ROCm Systems Profiler installation guide](https://rocm.docs.amd.com/projects/omnitrace/en/latest/install/install.html) for detailed information.

### Setup

> NOTE: Replace `/opt/rocprof-sys` below with installation prefix as necessary.

- Option 1: Source `setup-env.sh` script

```bash
source /opt/rocprof-sys/share/rocprof-sys/setup-env.sh
```

- Option 2: Load modulefile

```bash
module use /opt/rocprof-sys/share/modulefiles
module load rocprof-sys
```

- Option 3: Manual

```bash
export PATH=/opt/rocprof-sys/bin:${PATH}
export LD_LIBRARY_PATH=/opt/rocprof-sys/lib:${LD_LIBRARY_PATH}
```

### ROCm Systems Profiler Settings

Generate a rocprof-sys configuration file using `rocprof-sys-avail -G rocprof-sys.cfg`. Optionally, use `rocprof-sys-avail -G rocprof-sys.cfg --all` for
a verbose configuration file with descriptions, categories, etc. Modify the configuration file as desired, e.g. enable
[perfetto](https://perfetto.dev/), [timemory](https://github.com/NERSC/timemory), sampling, and process-level sampling by default
and tweak some sampling default values:

```console
# ...
OMNITRACE_TRACE                = true
OMNITRACE_PROFILE              = true
OMNITRACE_USE_SAMPLING         = true
OMNITRACE_USE_PROCESS_SAMPLING = true
# ...
OMNITRACE_SAMPLING_FREQ        = 50
OMNITRACE_SAMPLING_CPUS        = all
OMNITRACE_SAMPLING_GPUS        = $env:HIP_VISIBLE_DEVICES
```

Once the configuration file is adjusted to your preferences, either export the path to this file via `OMNITRACE_CONFIG_FILE=/path/to/rocprof-sys.cfg`
or place this file in `${HOME}/.rocprof-sys.cfg` to ensure these values are always read as the default. If you wish to change any of these settings,
you can override them via environment variables or by specifying an alternative `OMNITRACE_CONFIG_FILE`.

### Call-Stack Sampling

The `rocprof-sys-sample` executable is used to execute call-stack sampling on a target application without binary instrumentation.
Use a double-hypen (`--`) to separate the command-line arguments for `rocprof-sys-sample` from the target application and it's arguments.

```shell
rocprof-sys-sample --help
rocprof-sys-sample <rocprof-sys-options> -- <exe> <exe-options>
rocprof-sys-sample -f 1000 -- ls -la
```

### Binary Instrumentation

The `rocprof-sys-instrument` executable is used to instrument an existing binary. Call-stack sampling can be enabled alongside
the execution an instrumented binary, to help "fill in the gaps" between the instrumentation via setting the `OMNITRACE_USE_SAMPLING`
configuration variable to `ON`.
Similar to `rocprof-sys-sample`, use a double-hypen (`--`) to separate the command-line arguments for `rocprof-sys-instrument` from the target application and it's arguments.

```shell
rocprof-sys-instrument --help
rocprof-sys-instrument <rocprof-sys-options> -- <exe-or-library> <exe-options>
```

#### Binary Rewrite

Rewrite the text section of an executable or library with instrumentation:

```shell
rocprof-sys-instrument -o app.inst -- /path/to/app
```

In binary rewrite mode, if you also want instrumentation in the linked libraries, you must also rewrite those libraries.
Example of rewriting the functions starting with `"hip"` with instrumentation in the amdhip64 library:

```shell
mkdir -p ./lib
rocprof-sys-instrument -R '^hip' -o ./lib/libamdhip64.so.4 -- /opt/rocm/lib/libamdhip64.so.4
export LD_LIBRARY_PATH=${PWD}/lib:${LD_LIBRARY_PATH}
```

> ***Verify via `ldd` that your executable will load the instrumented library -- if you built your executable with***
> ***an RPATH to the original library's directory, then prefixing `LD_LIBRARY_PATH` will have no effect.***

Once you have rewritten your executable and/or libraries with instrumentation, you can just run the (instrumented) executable
or exectuable which loads the instrumented libraries normally, e.g.:

```shell
rocprof-sys-run -- ./app.inst
```

If you want to re-define certain settings to new default in a binary rewrite, use the `--env` option. This `rocprof-sys` option
will set the environment variable to the given value but will not override it. E.g. the default value of `OMNITRACE_PERFETTO_BUFFER_SIZE_KB`
is 1024000 KB (1 GiB):

```shell
# buffer size defaults to 1024000
rocprof-sys-instrument -o app.inst -- /path/to/app
rocprof-sys-run -- ./app.inst
```

Passing `--env OMNITRACE_PERFETTO_BUFFER_SIZE_KB=5120000` will change the default value in `app.inst` to 5120000 KiB (5 GiB):

```shell
# defaults to 5 GiB buffer size
rocprof-sys-instrument -o app.inst --env OMNITRACE_PERFETTO_BUFFER_SIZE_KB=5120000 -- /path/to/app
rocprof-sys-run -- ./app.inst
```

```shell
# override default 5 GiB buffer size to 200 MB via command-line
rocprof-sys-run --trace-buffer-size=200000 -- ./app.inst
# override default 5 GiB buffer size to 200 MB via environment
export OMNITRACE_PERFETTO_BUFFER_SIZE_KB=200000
rocprof-sys-run -- ./app.inst
```

#### Runtime Instrumentation

Runtime instrumentation will not only instrument the text section of the executable but also the text sections of the
linked libraries. Thus, it may be useful to exclude those libraries via the `-ME` (module exclude) regex option
or exclude specific functions with the `-E` regex option.

```shell
rocprof-sys-instrument -- /path/to/app
rocprof-sys-instrument -ME '^(libhsa-runtime64|libz\\.so)' -- /path/to/app
rocprof-sys-instrument -E 'rocr::atomic|rocr::core|rocr::HSA' --  /path/to/app
```

### Python Profiling and Tracing

Use the `rocprof-sys-python` script to profile/trace Python interpreter function calls.
Use a double-hypen (`--`) to separate the command-line arguments for `rocprof-sys-python` from the target script and it's arguments.

```shell
rocprof-sys-python --help
rocprof-sys-python <rocprof-sys-options> -- <python-script> <script-args>
rocprof-sys-python -- ./script.py
```

Please note, the first argument after the double-hyphen *must be a Python script*, e.g. `rocprof-sys-python -- ./script.py`.

If you need to specify a specific python interpreter version, use `rocprof-sys-python-X.Y` where `X.Y` is the Python
major and minor version:

```shell
rpcprof-sys-python-3.8 -- ./script.py
```

If you need to specify the full path to a Python interpreter, set the `PYTHON_EXECUTABLE` environment variable:

```shell
PYTHON_EXECUTABLE=/opt/conda/bin/python rocprof-sys-python -- ./script.py
```

If you want to restrict the data collection to specific function(s) and its callees, pass the `-b` / `--builtin` option after decorating the
function(s) with `@profile`. Use the `@noprofile` decorator for excluding/ignoring function(s) and its callees:

```python
def foo():
    pass

@noprofile
def bar():
    foo()

@profile
def spam():
    foo()
    bar()
```

Each time `spam` is called during profiling, the profiling results will include 1 entry for `spam` and 1 entry
for `foo` via the direct call within `spam`. There will be no entries for `bar` or the `foo` invocation within it.

### Trace Visualization

- Visit [ui.perfetto.dev](https://ui.perfetto.dev) in the web-browser
- Select "Open trace file" from panel on the left
- Locate the rocprof-sys perfetto output (extension: `.proto`)

![rocprof-sys-perfetto](docs/data/omnitrace-perfetto.png)

![rocprof-sys-rocm](docs/data/omnitrace-rocm.png)

![rocprof-sys-rocm-flow](docs/data/omnitrace-rocm-flow.png)

![rocprof-sys-user-api](docs/data/omnitrace-user-api.png)

## Using Perfetto tracing with System Backend

Perfetto tracing with the system backend supports multiple processes writing to the same
output file. Thus, it is a useful technique if rocprof-sys is built with partial MPI support
because all the perfetto output will be coalesced into a single file. The
installation docs for perfetto can be found [here](https://perfetto.dev/docs/contributing/build-instructions).
If you are building rocprof-sys from source, you can configure CMake with `OMNITRACE_INSTALL_PERFETTO_TOOLS=ON`
and the `perfetto` and `traced` applications will be installed as part of the build process. However,
it should be noted that to prevent this option from accidentally overwriting an existing perfetto install,
all the perfetto executables installed by omnitrace are prefixed with `rocprof-sys-perfetto-`, except for the `perfetto`
executable, which is just renamed `rocprof-sys-perfetto`.

Enable `traced` and `perfetto` in the background:

```shell
pkill traced
traced --background
perfetto --out ./rocprof-sys-perfetto.proto --txt -c ${OMNITRACE_ROOT}/share/perfetto.cfg --background
```

> ***NOTE: if the perfetto tools were installed by rocprof-sys, replace `traced` with `rocprof-sys-perfetto-traced` and***
> ***`perfetto` with `rocprof-sys-perfetto`.***

Configure rocprof-sys to use the perfetto system backend via the `--perfetto-backend` option of `rocprof-sys-run`:

```shell
# enable sampling on the uninstrumented binary
rocprof-sys-run --sample --trace --perfetto-backend=system -- ./myapp
# trace the instrument the binary
rocprof-sys-instrument -o ./myapp.inst -- ./myapp
rocprof-sys-run --trace --perfetto-backend=system -- ./myapp.inst
```

or via the `--env` option of `rocprof-sys-instrument` + runtime instrumentation:

```shell
rocprof-sys-instrument --env OMNITRACE_PERFETTO_BACKEND=system -- ./myapp
```
