# omnitrace: application tracing with static/dynamic binary instrumentation

## Dependencies

- [DynInst](https://github.com/dyninst/dyninst) for dynamic or static instrumentation
- [Julia](https://julialang.org/) for merging perfetto traces

## Installing DynInst

The easiest way to install Dyninst is via spack

```shell
git clone https://github.com/spack/spack.git
source ./spack/share/spack/setup-env.sh
spack compiler find
spack external find
spack install dyninst
spack load -r dyninst
```

## Installing Julia

Julia is available via Linux package managers or may be available via a module. Debian-based distributions such as Ubuntu can run (as a super-user):

```shell
apt-get install julia
```

Once Julia is installed, install the necessary packages (this operation only needs to be performed once):

```shell
julia -e 'using Pkg; for name in ["JSON", "DataFrames", "Dates", "CSV", "Chain", "PrettyTables"]; Pkg.add(name); end'
```

## Installing omnitrace

```shell
OMNITRACE_ROOT=${HOME}/sw/omnitrace
git clone https://github.com/AARInternal/omnitrace-dyninst.git
cmake -B build-omnitrace -DOMNITRACE_USE_MPI=ON -DCMAKE_INSTALL_PREFIX=${OMNITRACE_ROOT} omnitrace-dyninst
cmake --build build-omnitrace --target all --parallel 8
cmake --build build-omnitrace --target install
export PATH=${OMNITRACE_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${OMNITRACE_ROOT}/lib64:${OMNITRACE_ROOT}/lib:${LD_LIBRARY_PATH}
```

## Using Omnitrace Executable

```shell
omnitrace --help
omnitrace <omnitrace-options> -- <exe-or-library> <exe-options>
```

## Omnitrace Library Environment Settings

| Environment Variable        | Default Value                 | Description                                                                      |
|-----------------------------|-------------------------------|----------------------------------------------------------------------------------|
| `OMNITRACE_DEBUG`           | `false`                       | Enable debugging statements                                                      |
| `OMNITRACE_USE_PERFETTO`    | `true`                        | Collect profiling data via perfetto                                              |
| `OMNITRACE_USE_TIMEMORY`    | `false`                       | Collection profiling data via timemory                                           |
| `OMNITRACE_SAMPLE_RATE`     | `1`                           | Invoke perfetto and/or timemory once every N function calls                      |
| `OMNITRACE_USE_MPI`         | `true`                        | Label perfetto output files via rank instead of PID                              |
| `OMNITRACE_OUTPUT_FILE`     | `perfetto-trace.%rank%.proto` | Output file for perfetto (may use `%pid`)                                        |
| `OMNITRACE_BACKEND`         | `"inprocess"`                 | Configure perfetto to use either "inprocess" data management, "system", or "all" |
| `OMNITRACE_COMPONENTS`      | `"wall_clock"`                | Timemory components to activate when enabled                                     |
| `OMNITRACE_SHMEM_SIZE_HINT` | `40960`                       | Hint for perfetto shared memory buffer                                           |
| `OMNITRACE_BUFFER_SIZE_KB`  | `1024000`                     | Maximum amount of memory perfetto will use to collect data in-process            |
| `TIMEMORY_TIME_OUTPUT`      | `true`                        | Create unique output subdirectory with date and launch time                      |

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
  - Setting `OMNITRACE_USE_TIMEMORY=OFF` and `OMNITRACE_USE_PERFETTO=OFF` will disable all instrumentation
- Use `timemory-avail -S` to view the various settings for timemory
- Set `OMNITRACE_COMPONENTS="<comma-delimited-list-of-component-name>"` to control which components timemory collects
  - The list of components and their descriptions can be viewed via `timemory-avail -Cd`
  - The list of components and their string identifiers can be view via `timemory-avail -Cbs`
- You can filter any `timemory-avail` results via `-r <regex> -hl`

## Omnitrace Output

`omnitrace` will create an output directory named `omnitrace-<EXE_NAME>-output`, e.g. if your executable
is named `app.inst`, the output directory will be `omnitrace-app.inst-output`. Depending on whether
`TIMEMORY_TIME_OUTPUT=ON` (the default when perfetto is enabled), there will be a subdirectory with the date and time,
e.g. `2021-09-02_01.03_PM`. Within this directory, all perfetto files will be named `perfetto-trace.<PID>.proto` or
when `OMNITRACE_USE_MPI=ON`, `perfetto-trace.<RANK>.proto` (assuming omnitrace was built with MPI support).

You can explicitly control the output path and naming scheme of the files via the `OMNITRACE_OUTPUT_FILE` environment
variable. The special character sequences `%pid%` and `%rank%` will be replaced with the PID or MPI rank, respectively.

## Merging the traces from rocprof and omnitrace

> NOTE: Using `rocprof` externally is deprecated. The current version has built-in support for
> recording the GPU activity and HIP API calls. If you want to use an external rocprof, either
> configure CMake with `-DOMNITRACE_USE_ROCTRACER=OFF` or explicitly set `TIMEMORY_ROCTRACER_ENABLED=OFF` in the
> environment.

Use the `omnitrace-merge.jl` Julia script to merge rocprof and perfetto traces.

```shell
export TIMEMORY_ROCTRACER_ENABLED=OFF
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
