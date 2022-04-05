# omnitrace: application tracing with static/dynamic binary instrumentation

[![Ubuntu 18.04 (GCC 7, 8, MPICH)](https://github.com/AMDResearch/omnitrace/actions/workflows/ubuntu-bionic.yml/badge.svg)](https://github.com/AMDResearch/omnitrace/actions/workflows/ubuntu-bionic.yml)
[![Ubuntu 20.04 (GCC 7, 8, 9, 10)](https://github.com/AMDResearch/omnitrace/actions/workflows/ubuntu-focal-external.yml/badge.svg)](https://github.com/AMDResearch/omnitrace/actions/workflows/ubuntu-focal-external.yml)
[![Ubuntu 20.04 (GCC 9, MPICH, OpenMPI)](https://github.com/AMDResearch/omnitrace/actions/workflows/ubuntu-focal.yml/badge.svg)](https://github.com/AMDResearch/omnitrace/actions/workflows/ubuntu-focal.yml)
[![Ubuntu 20.04 (GCC 9, MPICH, OpenMPI, ROCm 4.3, 4.5, 5.0)](https://github.com/AMDResearch/omnitrace/actions/workflows/ubuntu-focal-external-rocm.yml/badge.svg)](https://github.com/AMDResearch/omnitrace/actions/workflows/ubuntu-focal-external-rocm.yml)

Omnitrace is an AMD research project and should not be treated as an offical part of the ROCm software stack.
The documentation for omnitrace is available at [amdresearch.github.io/omnitrace](https://amdresearch.github.io/omnitrace/).

## Using Omnitrace Executable

```shell
omnitrace --help
omnitrace <omnitrace-options> -- <exe-or-library> <exe-options>
```

## Omnitrace Settings

`omnitrace-avail -Sd` will provide a list of all the possible omnitrace settings, their current value, and a description of the setting.

> NOTE: Some settings may only affect the timemory backend.

These settings can be set via environment variables or placed in a config file and specified via `OMNITRACE_CONFIG_FILE=/path/to/config/file`. The config file
can be a text, JSON, or XML file. Some of the most relevant settings are provided below:

| Environment Variable                       | Default Value            | Description                                                                                                      |
|--------------------------------------------|--------------------------|------------------------------------------------------------------------------------------------------------------|
| `OMNITRACE_USE_PERFETTO`                   | `false`                  | Enable perfetto backend                                                                                          |
| `OMNITRACE_USE_PID`                        | `true`                   | Enable tagging filenames with process identifier (either MPI rank or pid)                                        |
| `OMNITRACE_USE_ROCTRACER`                  | `true`                   | Enable ROCM tracing                                                                                              |
| `OMNITRACE_USE_SAMPLING`                   | `true`                   | Enable statistical sampling of call-stack                                                                        |
| `OMNITRACE_USE_TIMEMORY`                   | `false`                  | Enable timemory backend                                                                                          |
| `OMNITRACE_BACKEND`                        | `inprocess`              | Specify the perfetto backend to activate. Options are: 'inprocess', 'system', or 'all'                           |
| `OMNITRACE_BUFFER_SIZE_KB`                 | `1024000`                | Size of perfetto buffer (in KB)                                                                                  |
| `OMNITRACE_COUT_OUTPUT`                    | `false`                  | Write output to stdout                                                                                           |
| `OMNITRACE_CRITICAL_TRACE`                 | `false`                  | Enable generation of the critical trace                                                                          |
| `OMNITRACE_CRITICAL_TRACE_BUFFER_COUNT`    | `2000`                   | Number of critical trace records to store in thread-local memory before submitting to shared buffer              |
| `OMNITRACE_CRITICAL_TRACE_COUNT`           | `0`                      | Number of critical trace to export (0 == all)                                                                    |
| `OMNITRACE_CRITICAL_TRACE_DEBUG`           | `false`                  | Enable debugging for critical trace                                                                              |
| `OMNITRACE_CRITICAL_TRACE_NUM_THREADS`     | `8`                      | Number of threads to use when generating the critical trace                                                      |
| `OMNITRACE_CRITICAL_TRACE_PER_ROW`         | `0`                      | How many critical traces per row in perfetto (0 == all in one row)                                               |
| `OMNITRACE_CRITICAL_TRACE_SERIALIZE_NAMES` | `false`                  | Include names in serialization of critical trace (mainly for debugging)                                          |
| `OMNITRACE_DIFF_OUTPUT`                    | `false`                  | Generate a difference output vs. a pre-existing output (see also: TIMEMORY_INPUT_PATH and TIMEMORY_INPUT_PREFIX) |
| `OMNITRACE_FLAT_SAMPLING`                  | `false`                  | Ignore hierarchy in all statistical sampling entries                                                             |
| `OMNITRACE_INSTRUMENTATION_INTERVAL`       | `1`                      | Instrumentation only takes measurements once every N function calls (not statistical)                            |
| `OMNITRACE_JSON_OUTPUT`                    | `true`                   | Write json output files                                                                                          |
| `OMNITRACE_MEMORY_PRECISION`               | `-1`                     | Set the precision for components with 'is_memory_category' type-trait                                            |
| `OMNITRACE_MEMORY_SCIENTIFIC`              | `false`                  | Set the numerical reporting format for components with 'is_memory_category' type-trait                           |
| `OMNITRACE_MEMORY_UNITS`                   | `""`                     | Set the units for components with 'uses_memory_units' type-trait                                                 |
| `OMNITRACE_OUTPUT_FILE`                    | `""`                     | Perfetto filename                                                                                                |
| `OMNITRACE_OUTPUT_PATH`                    | `omnitrace-{EXE}-output` | Explicitly specify the output folder for results                                                                 |
| `OMNITRACE_OUTPUT_PREFIX`                  | `""`                     | Explicitly specify a prefix for all output files                                                                 |
| `OMNITRACE_PRECISION`                      | `-1`                     | Set the global output precision for components                                                                   |
| `OMNITRACE_ROCTRACER_FLAT_PROFILE`         | `false`                  | Ignore hierarchy in all kernels entries with timemory backend                                                    |
| `OMNITRACE_ROCTRACER_HSA_ACTIVITY`         | `false`                  | Enable HSA activity tracing support                                                                              |
| `OMNITRACE_ROCTRACER_HSA_API`              | `false`                  | Enable HSA API tracing support                                                                                   |
| `OMNITRACE_ROCTRACER_HSA_API_TYPES`        | `""`                     | HSA API type to collect                                                                                          |
| `OMNITRACE_ROCTRACER_TIMELINE_PROFILE`     | `false`                  | Create unique entries for every kernel with timemory backend                                                     |
| `OMNITRACE_SAMPLING_DELAY`                   | `1e-06`                  | Number of seconds to delay activating the statistical sampling                                                   |
| `OMNITRACE_SAMPLING_FREQ`                    | `10`                     | Number of software interrupts per second when OMNITTRACE_USE_SAMPLING=ON                                         |
| `OMNITRACE_SCIENTIFIC`                     | `false`                  | Set the global numerical reporting to scientific format                                                          |
| `OMNITRACE_SETTINGS_DESC`                  | `false`                  | Provide descriptions when printing settings                                                                      |
| `OMNITRACE_SHMEM_SIZE_HINT_KB`             | `40960`                  | Hint for shared-memory buffer size in perfetto (in KB)                                                           |
| `OMNITRACE_TEXT_OUTPUT`                    | `true`                   | Write text output files                                                                                          |
| `OMNITRACE_TIMELINE_SAMPLING`              | `false`                  | Create unique entries for every sample when statistical sampling is enabled                                      |
| `OMNITRACE_TIMEMORY_COMPONENTS`            | `wall_clock`             | List of components to collect via timemory (see omnitrace-avail)                                                  |
| `OMNITRACE_TIME_FORMAT`                    | `%F_%I.%M_%p`            | Customize the folder generation when TIMEMORY_TIME_OUTPUT is enabled (see also: strftime)                        |
| `OMNITRACE_TIME_OUTPUT`                    | `true`                   | Output data to subfolder w/ a timestamp (see also: TIMEMORY_TIME_FORMAT)                                         |
| `OMNITRACE_TIMING_PRECISION`               | `6`                      | Set the precision for components with 'is_timing_category' type-trait                                            |
| `OMNITRACE_TIMING_SCIENTIFIC`              | `false`                  | Set the numerical reporting format for components with 'is_timing_category' type-trait                           |
| `OMNITRACE_TIMING_UNITS`                   | `""`                     | Set the units for components with 'uses_timing_units' type-trait                                                 |
| `OMNITRACE_TREE_OUTPUT`                    | `true`                   | Write hierarchical json output files                                                                             |

### Example Omnitrace Instrumentation

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

> NOTE: Verify via `ldd` that your executable will load the instrumented library -- if you built your executable with
> an RPATH to the original library's directory, then prefixing `LD_LIBRARY_PATH` will have no effect.

Once you have rewritten your executable and/or libraries with instrumentation, you can just run the (instrumented) executable
or exectuable which loads the instrumented libraries normally, e.g.:

```shell
./app.inst
```

If you want to re-define certain settings to new default in a binary rewrite, use the `--env` option. This `omnitrace` option
will set the environment variable to the given value but will not override it. E.g. the default value of `OMNITRACE_BUFFER_SIZE_KB`
is 1024000 KB (1 GiB):

```shell
# buffer size defaults to 1024000
omnitrace -o app.inst -- /path/to/app
./app.inst
```

Passing `--env OMNITRACE_BUFFER_SIZE_KB=5120000` will change the default value in `app.inst` to 5120000 KiB (5 GiB):

```shell
# defaults to 5 GiB buffer size
omnitrace -o app.inst --env OMNITRACE_BUFFER_SIZE_KB=5120000 -- /path/to/app
./app.inst
```

```shell
# override default 5 GiB buffer size to 200 MB
export OMNITRACE_BUFFER_SIZE_KB=200000
./app.inst
```

#### Runtime Instrumentation

Runtime instrumentation will not only instrument the text section of the executable but also the text sections of the
linked libraries. Thus, it may be useful to exclude those libraries via the `-ME` (module exclude) regex option.

```shell
omnitrace -- /path/to/app
omnitrace -ME '^(libhsa-runtime64|libz\\.so)' -- /path/to/app
omnitrace -E 'rocr::atomic|rocr::core|rocr::HSA' --  /path/to/app
```

## Miscellaneous Features and Caveats

- You may need to increase the default perfetto buffer size (1 GiB) to capture all the information
  - E.g. `export OMNITRACE_BUFFER_SIZE_KB=10240000` increases the buffer size to 10 GiB
- The omnitrace library has various setting which can be configured via environment variables, you can
  configure these settings to custom defaults with the omnitrace command-line tool via the `--env` option
  - E.g. to default to a buffer size of 5 GB, use `--env OMNITRACE_BUFFER_SIZE_KB=5120000`
  - This is particularly useful in binary rewrite mode
- Perfetto tooling is enabled by default
- Timemory tooling is disabled by default
- Enabling/disabling one of the aformentioned tools but not specifying enabling/disable the other will assume the inverse of the other's enabled state, e.g.
  - `OMNITRACE_USE_PERFETTO=OFF` yields the same result `OMNITRACE_USE_TIMEMORY=ON`
  - `OMNITRACE_USE_PERFETTO=ON` yields the same result as `OMNITRACE_USE_TIMEMORY=OFF`
  - In order to enable _both_ timemory and perfetto, set both `OMNITRACE_USE_TIMEMORY=ON` and `OMNITRACE_USE_PERFETTO=ON`
  - Setting `OMNITRACE_USE_TIMEMORY=OFF` and `OMNITRACE_USE_PERFETTO=OFF` will disable all instrumentation but call-stack sampling (`OMNITRACE_USE_SAMPLING=ON`) is still available.
- Use `omnitrace-avail -S` to view the various settings for timemory
- Set `OMNITRACE_COMPONENTS="<comma-delimited-list-of-component-name>"` to control which components timemory collects
  - The list of components and their descriptions can be viewed via `omnitrace-avail -Cd`
  - The list of components and their string identifiers can be view via `omnitrace-avail -Cbs`
- You can filter any `omnitrace-avail` results via `-r <regex> -hl`

## Omnitrace Output

`omnitrace` will create an output directory named `omnitrace-<EXE_NAME>-output`, e.g. if your executable
is named `app.inst`, the output directory will be `omnitrace-app.inst-output`. Depending on whether
`OMNITRACE_TIME_OUTPUT=ON` (the default when perfetto is enabled), there will be a subdirectory with the date and time,
e.g. `2021-09-02_01.03_PM`. Within this directory, all perfetto files will be named `perfetto-trace.<PID>.proto` or
when `OMNITRACE_USE_MPI=ON`, `perfetto-trace.<RANK>.proto` (assuming omnitrace was built with MPI support).

You can explicitly control the output path and naming scheme of the files via the `OMNITRACE_OUTPUT_FILE` environment
variable. The special character sequences `%pid%` and `%rank%` will be replaced with the PID or MPI rank, respectively.

## Merging the traces from rocprof and omnitrace

This section requires installing [Julia](https://julialang.org/).

### Installing Julia

Julia is available via Linux package managers or may be available via a module. Debian-based distributions such as Ubuntu can run (as a super-user):

```shell
apt-get install julia
```

Once Julia is installed, install the necessary packages (this operation only needs to be performed once):

```shell
julia -e 'using Pkg; for name in ["JSON", "DataFrames", "Dates", "CSV", "Chain", "PrettyTables"]; Pkg.add(name); end'
```

> NOTE: Using `rocprof` externally for tracing is deprecated. The current version has built-in support for
> recording the GPU activity and HIP API calls. If you want to use an external rocprof, either
> configure CMake with `-DOMNITRACE_USE_ROCTRACER=OFF` or explicitly set `OMNITRACE_ROCTRACER_ENABLED=OFF` in the
> environment.

Use the `omnitrace-merge.jl` Julia script to merge rocprof and perfetto traces.

```shell
export OMNITRACE_ROCTRACER_ENABLED=OFF
rocprof --hip-trace --roctx-trace --stats ./app.inst
omnitrace-merge.jl results.json omnitrace-app.inst-output/2021-09-02_01.03_PM/*.proto
```

## Use Perfetto tracing with System Backend

In a separate window run:

```shell
pkill traced
traced --background
perfetto --out ./htrace.out --txt -c ${OMNITRACE_ROOT}/share/roctrace.cfg
```

then in the window running the application, configure the omnitrace instrumentation to use the system backend:

```shell
export OMNITRACE_BACKEND_SYSTEM=1
```

for the merge use the `htrace.out`:

```shell
omnitrace-merge.jl results.json htrace.out
```
