# Call-Stack Sampling

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

> ***NOTE: Set `OMNITRACE_USE_SAMPLING=ON` to activate call-stack sampling when executing an instrumented binary***

Call-stack sampling can be activated with either a binary instrumented via the `omnitrace` executable or via the `omnitrace-sample` executable.
***Effectively***, all of the commands below are equivalent:

- Binary rewrite with only instrumentation necessary to start/stop sampling

```console
omnitrace-instrument -M sampling -o foo.inst -- foo
omnitrace-run -- ./foo.inst
```

- Runtime instrumentation with only instrumentation necessary to start/stop sampling

```console
omnitrace-instrument -M sampling -- foo
```

- No instrumentation required

```console
omnitrace-sample -- foo
```

All `omnitrace-instrument -M sampling` (referred to as "instrumented-sampling" henceforth) does is wrap the `main` of the executable with initialization
before `main` starts and finalization after `main` ends.
This can be easily accomplished without instrumentation via a `LD_PRELOAD` of a library with containing a dynamic symbol wrapper around `__libc_start_main`.
Thus, whenever binary instrumentation is unnecessary, using `omnitrace-sample` is recommended over `omnitrace-instrument -M sampling` for several reasons:

1. `omnitrace-sample` provides command-line options for controlling features of omnitrace instead of *requiring* configuration files or environment variables
2. Despite the fact that instrumented-sampling only requires inserting snippets around one function (`main`), Dyninst
   does not have a feature for specifying that parsing and processing all the other symbols in the binary is unnecessary,
   thus, in the best case scenario, instrumented-sampling has a slightly slower launch time when the target binary is relatively small
   but, in the worst case scenarios, requires a significant amount of time and memory to launch
3. `omnitrace-sample` is fully compatible with MPI, e.g. `mpirun -n 2 omnitrace-sample -- foo`, whereas `mpirun -n 2 omnitrace-instrument -M sampling -- foo`
   is incompatible with some MPI distributions (particularly OpenMPI) because of MPI restrictions against forking within an MPI rank
    - If you recall, when MPI and binary instrumentation is involved, two steps are involed: (1) do a binary rewrite of the executable
      and (2) use the instrumented executable in leiu of the original executable. `omnitrace-sample` is thus much easier to use with MPI.

## omnitrace-sample Executable

View the help menu of `omnitrace-sample` with the `-h` / `--help` option:

```console
$ omnitrace-sample --help
[omnitrace-sample] Usage: omnitrace-sample [ --help (count: 0, dtype: bool)
                                             --monochrome (max: 1, dtype: bool)
                                             --debug (max: 1, dtype: bool)
                                             --verbose (count: 1)
                                             --config (min: 0, dtype: filepath)
                                             --output (min: 1)
                                             --trace (max: 1, dtype: bool)
                                             --profile (max: 1, dtype: bool)
                                             --flat-profile (max: 1, dtype: bool)
                                             --host (max: 1, dtype: bool)
                                             --device (max: 1, dtype: bool)
                                             --trace-file (count: 1, dtype: filepath)
                                             --trace-buffer-size (count: 1, dtype: KB)
                                             --trace-fill-policy (count: 1)
                                             --profile-format (min: 1)
                                             --profile-diff (min: 1)
                                             --process-freq (count: 1)
                                             --process-wait (count: 1)
                                             --process-duration (count: 1)
                                             --cpus (count: unlimited, dtype: int or range)
                                             --gpus (count: unlimited, dtype: int or range)
                                             --freq (count: 1)
                                             --wait (count: 1)
                                             --duration (count: 1)
                                             --tids (min: 1)
                                             --cputime (min: 0)
                                             --realtime (min: 0)
                                             --include (count: unlimited)
                                             --exclude (count: unlimited)
                                             --cpu-events (count: unlimited)
                                             --gpu-events (count: unlimited)
                                             --inlines (max: 1, dtype: bool)
                                             --hsa-interrupt (count: 1, dtype: int)
                                           ]

Options:
    -h, -?, --help                 Shows this page

    [DEBUG OPTIONS]

    --monochrome                   Disable colorized output
    --debug                        Debug output
    -v, --verbose                  Verbose output

    [GENERAL OPTIONS]

    -c, --config                   Configuration file
    -o, --output                   Output path. Accepts 1-2 parameters corresponding to the output path and the output prefix
    -T, --trace                    Generate a detailed trace (perfetto output)
    -P, --profile                  Generate a call-stack-based profile (conflicts with --flat-profile)
    -F, --flat-profile             Generate a flat profile (conflicts with --profile)
    -H, --host                     Enable sampling host-based metrics for the process. E.g. CPU frequency, memory usage, etc.
    -D, --device                   Enable sampling device-based metrics for the process. E.g. GPU temperature, memory usage, etc.

    [TRACING OPTIONS]

    --trace-file                   Specify the trace output filename. Relative filepath will be with respect to output path and output prefix.
    --trace-buffer-size            Size limit for the trace output (in KB)
    --trace-fill-policy [ discard | ring_buffer ]

                                   Policy for new data when the buffer size limit is reached:
                                       - discard     : new data is ignored
                                       - ring_buffer : new data overwrites oldest data

    [PROFILE OPTIONS]

    --profile-format [ console | json | text ]
                                   Data formats for profiling results
    --profile-diff                 Generate a diff output b/t the profile collected and an existing profile from another run Accepts 1-2 parameters
                                   corresponding to the input path and the input prefix

    [HOST/DEVICE (PROCESS SAMPLING) OPTIONS]


    --process-freq                 Set the default host/device sampling frequency (number of interrupts per second)
    --process-wait                 Set the default wait time (i.e. delay) before taking first host/device sample (in seconds of realtime)
    --process-duration             Set the duration of the host/device sampling (in seconds of realtime)
    --cpus                         CPU IDs for frequency sampling. Supports integers and/or ranges
    --gpus                         GPU IDs for SMI queries. Supports integers and/or ranges

    [GENERAL SAMPLING OPTIONS]

    -f, --freq                     Set the default sampling frequency (number of interrupts per second)
    -w, --wait                     Set the default wait time (i.e. delay) before taking first sample (in seconds). This delay time is based on the clock
                                   of the sampler, i.e., a delay of 1 second for CPU-clock sampler may not equal 1 second of realtime
    -d, --duration                 Set the duration of the sampling (in seconds of realtime). I.e., it is possible (currently) to set a CPU-clock time
                                   delay that exceeds the real-time duration... resulting in zero samples being taken
    -t, --tids                     Specify the default thread IDs for sampling, where 0 (zero) is the main thread and each thread created by the target
                                   application is assigned an atomically incrementing value.

    [SAMPLING TIMER OPTIONS]

    --cputime                      Sample based on a CPU-clock timer (default). Accepts zero or more arguments:
                                       0. Enables sampling based on CPU-clock timer.
                                       1. Interrupts per second. E.g., 100 == sample every 10 milliseconds of CPU-time.
                                       2. Delay (in seconds of CPU-clock time). I.e., how long each thread should wait before taking first sample.
                                       3+ Thread IDs to target for sampling, starting at 0 (the main thread).
                                          May be specified as index or range, e.g., '0 2-4' will be interpreted as:
                                             sample the main thread (0), do not sample the first child thread but sample the 2nd, 3rd, and 4th child threads
    --realtime                     Sample based on a real-clock timer. Accepts zero or more arguments:
                                       0. Enables sampling based on real-clock timer.
                                       1. Interrupts per second. E.g., 100 == sample every 10 milliseconds of realtime.
                                       2. Delay (in seconds of real-clock time). I.e., how long each thread should wait before taking first sample.
                                       3+ Thread IDs to target for sampling, starting at 0 (the main thread).
                                          May be specified as index or range, e.g., '0 2-4' will be interpreted as:
                                             sample the main thread (0), do not sample the first child thread but sample the 2nd, 3rd, and 4th child threads
                                          When sampling with a real-clock timer, please note that enabling this will cause threads which are typically "idle"
                                          to consume more resources since, while idle, the real-clock time increases (and therefore triggers taking samples)
                                          whereas the CPU-clock time does not.

    [BACKEND OPTIONS]  (These options control region information captured w/o sampling or instrumentation)

    -I, --include [ all | kokkosp | mpip | mutex-locks | ompt | rcclp | rocm-smi | rocprofiler | roctracer | roctx | rw-locks | spin-locks ]
                                   Include data from these backends
    -E, --exclude [ all | kokkosp | mpip | mutex-locks | ompt | rcclp | rocm-smi | rocprofiler | roctracer | roctx | rw-locks | spin-locks ]
                                   Exclude data from these backends

    [HARDWARE COUNTER OPTIONS]

    -C, --cpu-events               Set the CPU hardware counter events to record (ref: `omnitrace-avail -H -c CPU`)
    -G, --gpu-events               Set the GPU hardware counter events to record (ref: `omnitrace-avail -H -c GPU`)

    [MISCELLANEOUS OPTIONS]

    -i, --inlines                  Include inline info in output when available
    --hsa-interrupt [ 0 | 1 ]      Set the value of the HSA_ENABLE_INTERRUPT environment variable.
                                     ROCm version 5.2 and older have a bug which will cause a deadlock if a sample is taken while waiting for the signal
                                     that a kernel completed -- which happens when sampling with a real-clock timer. We require this option to be set to
                                     when --realtime is specified to make users aware that, while this may fix the bug, it can have a negative impact on
                                     performance.
                                     Values:
                                       0     avoid triggering the bug, potentially at the cost of reduced performance
                                       1     do not modify how ROCm is notified about kernel completion
```

The general syntax for separating omnitrace command line arguments from the application arguments follows the
is consistent with the LLVM style of using a standalone double-hyphen (`--`). All arguments preceding the double-hyphen
are interpreted as belonging to omnitrace and all arguments following the double-hyphen are interpreted as the
application and it's arguments. The double-hyphen is only necessary when passing command line arguments to the target
which also use hyphens. E.g. `omnitrace-sample ls` works but, in order to run `ls -la`, use `omnitrace-sample -- ls -la`.

[Configuring OmniTrace Runtime](runtime.md) establish the precedence of environment variable values over values specified in the configuration files. This enables
the user to configure the omnitrace runtime to their preferred default behavior in a file such as `~/.omnitrace.cfg` and then easily override
those settings via something like `OMNITRACE_ENABLED=OFF omnitrace-sample -- foo`.
Similarly, the command line arguments passed to `omnitrace-sample` take precedence over environment variables.

All of the command-line options above correlate to one or more configuration settings, e.g. `--cpu-events` correlates to the `OMNITRACE_PAPI_EVENTS` configuration variable.
After the command-line arguments to `omnitrace-sample` have been processed but before the target application is executed, `omnitrace-sample` will emit a log
for which environment variables where set and/or modified:

The snippet below shows the environment updates when `omnitrace-sample` is invoked with no arguments

```console
$ omnitrace-sample -- ./parallel-overhead-locks 30 4 100

HSA_TOOLS_LIB=/opt/omnitrace/lib/libomnitrace-dl.so.1.7.1
HSA_TOOLS_REPORT_LOAD_FAILURE=1
LD_PRELOAD=/opt/omnitrace/lib/libomnitrace-dl.so.1.7.1
OMNITRACE_CRITICAL_TRACE=false
OMNITRACE_USE_PROCESS_SAMPLING=false
OMNITRACE_USE_SAMPLING=true
OMP_TOOL_LIBRARIES=/opt/omnitrace/lib/libomnitrace-dl.so.1.7.1
ROCP_TOOL_LIB=/opt/omnitrace/lib/libomnitrace.so.1.7.1

...
```

The snippet below shows the environment updates when `omnitrace-sample` enables profiling, tracing, host process-sampling, device process-sampling, and all the available backends:

```console
$ omnitrace-sample -PTDH -I all -- ./parallel-overhead-locks 30 4 100

HSA_TOOLS_LIB=/opt/omnitrace/lib/libomnitrace-dl.so.1.7.1
HSA_TOOLS_REPORT_LOAD_FAILURE=1
KOKKOS_PROFILE_LIBRARY=/opt/omnitrace/lib/libomnitrace.so.1.7.1
LD_PRELOAD=/opt/omnitrace/lib/libomnitrace-dl.so.1.7.1
OMNITRACE_CPU_FREQ_ENABLED=true
OMNITRACE_CRITICAL_TRACE=false
OMNITRACE_TRACE_THREAD_LOCKS=true
OMNITRACE_TRACE_THREAD_RW_LOCKS=true
OMNITRACE_TRACE_THREAD_SPIN_LOCKS=true
OMNITRACE_USE_KOKKOSP=true
OMNITRACE_USE_MPIP=true
OMNITRACE_USE_OMPT=true
OMNITRACE_TRACE=true
OMNITRACE_USE_PROCESS_SAMPLING=true
OMNITRACE_USE_RCCLP=true
OMNITRACE_USE_ROCM_SMI=true
OMNITRACE_USE_ROCPROFILER=true
OMNITRACE_USE_ROCTRACER=true
OMNITRACE_USE_ROCTX=true
OMNITRACE_USE_SAMPLING=true
OMNITRACE_PROFILE=true
OMP_TOOL_LIBRARIES=/opt/omnitrace/lib/libomnitrace-dl.so.1.7.1
ROCP_TOOL_LIB=/opt/omnitrace/lib/libomnitrace.so.1.7.1

...
```

The snippet below shows the environment updates when `omnitrace-sample` enables profiling, tracing, host process-sampling, device process-sampling,
sets the output path to `omnitrace-output`, the output prefix to `%tag%` and disables all the available backends:

```console
$ omnitrace-sample -PTDH -E all -o omnitrace-output %tag% -- ./parallel-overhead-locks 30 4 100

LD_PRELOAD=/opt/omnitrace/lib/libomnitrace-dl.so.1.7.1
OMNITRACE_CPU_FREQ_ENABLED=true
OMNITRACE_CRITICAL_TRACE=false
OMNITRACE_OUTPUT_PATH=omnitrace-output
OMNITRACE_OUTPUT_PREFIX=%tag%
OMNITRACE_TRACE_THREAD_LOCKS=false
OMNITRACE_TRACE_THREAD_RW_LOCKS=false
OMNITRACE_TRACE_THREAD_SPIN_LOCKS=false
OMNITRACE_USE_KOKKOSP=false
OMNITRACE_USE_MPIP=false
OMNITRACE_USE_OMPT=false
OMNITRACE_TRACE=true
OMNITRACE_USE_PROCESS_SAMPLING=true
OMNITRACE_USE_RCCLP=false
OMNITRACE_USE_ROCM_SMI=false
OMNITRACE_USE_ROCPROFILER=false
OMNITRACE_USE_ROCTRACER=false
OMNITRACE_USE_ROCTX=false
OMNITRACE_USE_SAMPLING=true
OMNITRACE_PROFILE=true

...
```

## omnitrace-sample Example

```console
$ omnitrace-sample -PTDH -E all -o omnitrace-output %tag% -c -- ./parallel-overhead-locks 30 4 100

LD_PRELOAD=/opt/omnitrace/lib/libomnitrace-dl.so.1.7.1
OMNITRACE_CONFIG_FILE=
OMNITRACE_CPU_FREQ_ENABLED=true
OMNITRACE_CRITICAL_TRACE=false
OMNITRACE_OUTPUT_PATH=omnitrace-output
OMNITRACE_OUTPUT_PREFIX=%tag%
OMNITRACE_TRACE_THREAD_LOCKS=false
OMNITRACE_TRACE_THREAD_RW_LOCKS=false
OMNITRACE_TRACE_THREAD_SPIN_LOCKS=false
OMNITRACE_USE_KOKKOSP=false
OMNITRACE_USE_MPIP=false
OMNITRACE_USE_OMPT=false
OMNITRACE_TRACE=true
OMNITRACE_USE_PROCESS_SAMPLING=true
OMNITRACE_USE_RCCLP=false
OMNITRACE_USE_ROCM_SMI=false
OMNITRACE_USE_ROCPROFILER=false
OMNITRACE_USE_ROCTRACER=false
OMNITRACE_USE_ROCTX=false
OMNITRACE_USE_SAMPLING=true
OMNITRACE_PROFILE=true

[omnitrace][omnitrace_init_tooling] Instrumentation mode: Sampling


      ______   .___  ___. .__   __.  __  .___________..______          ___       ______  _______
     /  __  \  |   \/   | |  \ |  | |  | |           ||   _  \        /   \     /      ||   ____|
    |  |  |  | |  \  /  | |   \|  | |  | `---|  |----`|  |_)  |      /  ^  \   |  ,----'|  |__
    |  |  |  | |  |\/|  | |  . `  | |  |     |  |     |      /      /  /_\  \  |  |     |   __|
    |  `--'  | |  |  |  | |  |\   | |  |     |  |     |  |\  \----./  _____  \ |  `----.|  |____
     \______/  |__|  |__| |__| \__| |__|     |__|     | _| `._____/__/     \__\ \______||_______|


[759.689]       perfetto.cc:55903 Configured tracing session 1, #sources:1, duration:0 ms, #buffers:1, total buffer size:1024000 KB, total sessions:1, uid:0 session name: ""

[parallel-overhead-locks] Threads: 4
[parallel-overhead-locks] Iterations: 100
[parallel-overhead-locks] fibonacci(30)...
[1] number of iterations: 100
[2] number of iterations: 100
[3] number of iterations: 100
[4] number of iterations: 100
[parallel-overhead-locks] fibonacci(30) x 4 = 394644873
[parallel-overhead-locks] number of mutex locks = 400
[omnitrace][107157][0][omnitrace_finalize]
[omnitrace][107157][0][omnitrace_finalize] finalizing...
[omnitrace][107157][0][omnitrace_finalize]
[omnitrace][107157][0][omnitrace_finalize] omnitrace/process/107157 : 0.610427 sec wall_clock,    2.248 MB peak_rss,    2.265 MB page_rss, 2.560000 sec cpu_clock,  419.4 % cpu_util [laps: 1]
[omnitrace][107157][0][omnitrace_finalize] omnitrace/process/107157/thread/0 : 0.608866 sec wall_clock, 0.000677 sec thread_cpu_clock,    0.1 % thread_cpu_util,    2.248 MB peak_rss [laps: 1]
[omnitrace][107157][0][omnitrace_finalize] omnitrace/process/107157/thread/1 : 0.608237 sec wall_clock, 0.603553 sec thread_cpu_clock,   99.2 % thread_cpu_util,    2.204 MB peak_rss [laps: 1]
[omnitrace][107157][0][omnitrace_finalize] omnitrace/process/107157/thread/2 : 0.601430 sec wall_clock, 0.598378 sec thread_cpu_clock,   99.5 % thread_cpu_util,    1.156 MB peak_rss [laps: 1]
[omnitrace][107157][0][omnitrace_finalize] omnitrace/process/107157/thread/3 : 0.570223 sec wall_clock, 0.568713 sec thread_cpu_clock,   99.7 % thread_cpu_util,    0.772 MB peak_rss [laps: 1]
[omnitrace][107157][0][omnitrace_finalize] omnitrace/process/107157/thread/4 : 0.557637 sec wall_clock, 0.557198 sec thread_cpu_clock,   99.9 % thread_cpu_util,    0.156 MB peak_rss [laps: 1]
[omnitrace][107157][0][omnitrace_finalize]
[omnitrace][107157][0][omnitrace_finalize] Finalizing perfetto...
[omnitrace][107157][perfetto]> Outputting '/home/user/data/omnitrace-output/2022-10-19_02.46/parallel-overhead-locksperfetto-trace-107157.proto' (842.90 KB / 0.84 MB / 0.00 GB)... Done
[omnitrace][107157][trip_count]> Outputting 'omnitrace-output/2022-10-19_02.46/parallel-overhead-lockstrip_count-107157.json'
[omnitrace][107157][trip_count]> Outputting 'omnitrace-output/2022-10-19_02.46/parallel-overhead-lockstrip_count-107157.txt'
[omnitrace][107157][sampling_percent]> Outputting 'omnitrace-output/2022-10-19_02.46/parallel-overhead-lockssampling_percent-107157.json'
[omnitrace][107157][sampling_percent]> Outputting 'omnitrace-output/2022-10-19_02.46/parallel-overhead-lockssampling_percent-107157.txt'
[omnitrace][107157][sampling_cpu_clock]> Outputting 'omnitrace-output/2022-10-19_02.46/parallel-overhead-lockssampling_cpu_clock-107157.json'
[omnitrace][107157][sampling_cpu_clock]> Outputting 'omnitrace-output/2022-10-19_02.46/parallel-overhead-lockssampling_cpu_clock-107157.txt'
[omnitrace][107157][sampling_wall_clock]> Outputting 'omnitrace-output/2022-10-19_02.46/parallel-overhead-lockssampling_wall_clock-107157.json'
[omnitrace][107157][sampling_wall_clock]> Outputting 'omnitrace-output/2022-10-19_02.46/parallel-overhead-lockssampling_wall_clock-107157.txt'
[omnitrace][107157][wall_clock]> Outputting 'omnitrace-output/2022-10-19_02.46/parallel-overhead-lockswall_clock-107157.json'
[omnitrace][107157][wall_clock]> Outputting 'omnitrace-output/2022-10-19_02.46/parallel-overhead-lockswall_clock-107157.txt'
[omnitrace][107157][metadata]> Outputting 'omnitrace-output/2022-10-19_02.46/parallel-overhead-locksmetadata-107157.json' and 'omnitrace-output/2022-10-19_02.46/parallel-overhead-locksfunctions-107157.json'
[omnitrace][107157][0][omnitrace_finalize] Finalized
[761.584]       perfetto.cc:57382 Tracing session 1 ended, total sessions:0
```
