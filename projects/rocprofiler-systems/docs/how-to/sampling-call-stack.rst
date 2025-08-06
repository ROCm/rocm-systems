.. meta::
   :description: ROCm Systems Profiler call stack sampling documentation and reference
   :keywords: rocprofiler-systems,rocprofsys, ROCm, profiler, sampling, call stack, tracking, visualization, tool, Instinct, accelerator, AMD

****************************************************
Sampling the call stack
****************************************************

`ROCm Systems Profiler <https://github.com/ROCm/rocprofiler-systems>`_ can use call-stack sampling
on a binary instrumented with either the ``rocprof-sys`` executable
or the ``rocprof-sys-sample`` executable.
For example, all of the following commands are effectively equivalent:

* Binary rewrite with only the instrumentation necessary to start and stop sampling

  .. code-block:: shell

     rocprof-sys-instrument -M sampling -o foo.inst -- foo
     rocprof-sys-run -- ./foo.inst

* Runtime instrumentation with only the instrumentation necessary to start and stop sampling

  .. code-block:: shell

     rocprof-sys-instrument -M sampling -- foo

* No instrumentation required

  .. code-block:: shell

     rocprof-sys-sample -- foo

.. note::

   Set ``ROCPROFSYS_USE_SAMPLING=ON`` to activate call-stack sampling when executing an instrumented binary.

All ``rocprof-sys-instrument -M sampling`` (subsequently referred to as "instrumented-sampling")
does is wrap the ``main`` of the executable with initialization
before ``main`` starts and finalization after ``main`` ends.
This can be accomplished without instrumentation through a ``LD_PRELOAD``
of a library containing a dynamic symbol wrapper around ``__libc_start_main``.

The use of ``rocprof-sys-sample`` is **recommended** over
``rocprof-sys-instrument -M sampling`` when binary instrumentation
is not necessary. This is for a number of reasons:

* ``rocprof-sys-sample`` provides command-line options for controlling the ROCm Systems Profiler feature set instead of
  requiring configuration files or environment variables
* Despite the fact that instrumented-sampling only requires inserting snippets
  around one function (``main``), Dyninst
  does not have a feature for specifying that parsing and processing all the
  other symbols in the binary is unnecessary.
  In the best-case scenario when the target binary is relatively small,
  instrumented-sampling has a slightly slower launch time,
  but in the worst case scenarios it requires a significant amount of time and memory to launch.
* ``rocprof-sys-sample`` is fully compatible with MPI. For example,
  the command ``mpirun -n 2 rocprof-sys-sample -- foo`` is valid,
  whereas ``mpirun -n 2 rocprof-sys-instrument -M sampling -- foo``
  is incompatible with some MPI distributions (particularly OpenMPI). This is because
  MPI prohibits forking within an MPI rank.

  * When MPI and binary instrumentation are both involved, two steps are required:
    performing a binary rewrite of the executable and then using the instrumented executable
    in lieu of the original executable. ``rocprof-sys-sample`` is therefore much easier to use with MPI.

The rocprof-sys-sample executable
========================================

View the help menu of ``rocprof-sys-sample`` with the ``-h`` / ``--help`` option:

.. code-block:: shell

   $ rocprof-sys-sample --help
   [rocprof-sys-sample] Usage: rocprof-sys-sample [ --help (count: 0, dtype: bool)
                                                --version (count: 0, dtype: bool)
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
                                                --wait (count: 1)
                                                --duration (count: 1)
                                                --trace-file (count: 1, dtype: filepath)
                                                --trace-buffer-size (count: 1, dtype: KB)
                                                --trace-fill-policy (count: 1)
                                                --trace-wait (count: 1)
                                                --trace-duration (count: 1)
                                                --trace-periods (min: 1)
                                                --trace-clock-id (count: 1)
                                                --profile-format (min: 1)
                                                --profile-diff (min: 1)
                                                --process-freq (count: 1)
                                                --process-wait (count: 1)
                                                --process-duration (count: 1)
                                                --cpus (count: unlimited, dtype: int or range)
                                                --gpus (count: unlimited, dtype: int or range)
                                                --freq (count: 1)
                                                --sampling-wait (count: 1)
                                                --sampling-duration (count: 1)
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
      -h, -?, --help                 Shows this page (count: 0, dtype: bool)
      --version                      Prints the version and exit (count: 0, dtype: bool)

      [DEBUG OPTIONS]

      --monochrome                   Disable colorized output (max: 1, dtype: bool)
      --debug                        Debug output (max: 1, dtype: bool)
      -v, --verbose                  Verbose output (count: 1)

      [GENERAL OPTIONS]  These are options which are ubiquitously applied

      -c, --config                   Configuration file (min: 0, dtype: filepath)
      -o, --output                   Output path. Accepts 1-2 parameters corresponding to the output path and the output prefix (min: 1)
      -T, --trace                    Generate a detailed trace (perfetto output) (max: 1, dtype: bool)
      -P, --profile                  Generate a call-stack-based profile (conflicts with --flat-profile) (max: 1, dtype: bool)
      -F, --flat-profile             Generate a flat profile (conflicts with --profile) (max: 1, dtype: bool)
      -H, --host                     Enable sampling host-based metrics for the process. E.g. CPU frequency, memory usage, etc. (max: 1, dtype: bool)
      -D, --device                   Enable sampling device-based metrics for the process. E.g. GPU temperature, memory usage, etc. (max: 1, dtype: bool)
      -w, --wait                     This option is a combination of '--trace-wait' and '--sampling-wait'. See the descriptions for those two options.
                                    (count: 1)
      -d, --duration                 This option is a combination of '--trace-duration' and '--sampling-duration'. See the descriptions for those two
                                    options. (count: 1)

      [TRACING OPTIONS]  Specific options controlling tracing (i.e. deterministic measurements of every event)

      --trace-file                   Specify the trace output filename. Relative filepath will be with respect to output path and output prefix. (count: 1,
                                    dtype: filepath)
      --trace-buffer-size            Size limit for the trace output (in KB) (count: 1, dtype: KB)
      --trace-fill-policy [ discard | ring_buffer ]

                                    Policy for new data when the buffer size limit is reached:
                                          - discard     : new data is ignored
                                          - ring_buffer : new data overwrites oldest data (count: 1)
      --trace-wait                   Set the wait time (in seconds) before collecting trace and/or profiling data(in seconds). By default, the duration is
                                    in seconds of realtime but that can changed via --trace-clock-id. (count: 1)
      --trace-duration               Set the duration of the trace and/or profile data collection (in seconds). By default, the duration is in seconds of
                                    realtime but that can changed via --trace-clock-id. (count: 1)
      --trace-periods                More powerful version of specifying trace delay and/or duration. Format is one or more groups of: <DELAY>:<DURATION>,
                                    <DELAY>:<DURATION>:<REPEAT>, and/or <DELAY>:<DURATION>:<REPEAT>:<CLOCK_ID>. (min: 1)
      --trace-clock-id [ 0 (realtime|CLOCK_REALTIME)
                        1 (monotonic|CLOCK_MONOTONIC)
                        2 (cputime|CLOCK_PROCESS_CPUTIME_ID)
                        4 (monotonic_raw|CLOCK_MONOTONIC_RAW)
                        5 (realtime_coarse|CLOCK_REALTIME_COARSE)
                        6 (monotonic_coarse|CLOCK_MONOTONIC_COARSE)
                        7 (boottime|CLOCK_BOOTTIME) ]
                                    Set the default clock ID for for trace delay/duration. Note: "cputime" is the *process* CPU time and might need to be
                                    scaled based on the number of threads, i.e. 4 seconds of CPU-time for an application with 4 fully active threads would
                                    equate to ~1 second of realtime. If this proves to be difficult to handle in practice, please file a feature request
                                    for rocprof-sys to auto-scale based on the number of threads. (count: 1)

      [PROFILE OPTIONS]  Specific options controlling profiling (i.e. deterministic measurements which are aggregated into a summary)

      --profile-format [ console | json | text ]
                                    Data formats for profiling results (min: 1)
      --profile-diff                 Generate a diff output b/t the profile collected and an existing profile from another run Accepts 1-2 parameters
                                    corresponding to the input path and the input prefix (min: 1)

      [HOST/DEVICE (PROCESS SAMPLING) OPTIONS]
                                    Process sampling is background measurements for resources available to the entire process. These samples are not tied
                                    to specific lines/regions of code

      --process-freq                 Set the default host/device sampling frequency (number of interrupts per second) (count: 1)
      --process-wait                 Set the default wait time (i.e. delay) before taking first host/device sample (in seconds of realtime) (count: 1)
      --process-duration             Set the duration of the host/device sampling (in seconds of realtime) (count: 1)
      --cpus                         CPU IDs for frequency sampling. Supports integers and/or ranges (count: unlimited, dtype: int or range)
      --gpus                         GPU IDs for SMI queries. Supports integers and/or ranges (count: unlimited, dtype: int or range)

      [GENERAL SAMPLING OPTIONS] General options for timer-based sampling per-thread

      -f, --freq                     Set the default sampling frequency (number of interrupts per second) (count: 1)
      --sampling-wait                Set the default wait time (i.e. delay) before taking first sample (in seconds). This delay time is based on the clock
                                    of the sampler, i.e., a delay of 1 second for CPU-clock sampler may not equal 1 second of realtime (count: 1)
      --sampling-duration            Set the duration of the sampling (in seconds of realtime). I.e., it is possible (currently) to set a CPU-clock time
                                    delay that exceeds the real-time duration... resulting in zero samples being taken (count: 1)
      -t, --tids                     Specify the default thread IDs for sampling, where 0 (zero) is the main thread and each thread created by the target
                                    application is assigned an atomically incrementing value. (min: 1)

      [SAMPLING TIMER OPTIONS] These options determine the heuristic for deciding when to take a sample

      --cputime                      Sample based on a CPU-clock timer (default). Accepts zero or more arguments:
                                          0. Enables sampling based on CPU-clock timer.
                                          1. Interrupts per second. E.g., 100 == sample every 10 milliseconds of CPU-time.
                                          2. Delay (in seconds of CPU-clock time). I.e., how long each thread should wait before taking first sample.
                                          3+ Thread IDs to target for sampling, starting at 0 (the main thread).
                                             May be specified as index or range, e.g., '0 2-4' will be interpreted as:
                                                sample the main thread (0), do not sample the first child thread but sample the 2nd, 3rd, and 4th child threads (min: 0)
      --realtime                     Sample based on a real-clock timer. Accepts zero or more arguments:
                                          0. Enables sampling based on real-clock timer.
                                          1. Interrupts per second. E.g., 100 == sample every 10 milliseconds of realtime.
                                          2. Delay (in seconds of real-clock time). I.e., how long each thread should wait before taking first sample.
                                          3+ Thread IDs to target for sampling, starting at 0 (the main thread).
                                             May be specified as index or range, e.g., '0 2-4' will be interpreted as:
                                                sample the main thread (0), do not sample the first child thread but sample the 2nd, 3rd, and 4th child threads
                                             When sampling with a real-clock timer, please note that enabling this will cause threads which are typically "idle"
                                             to consume more resources since, while idle, the real-clock time increases (and therefore triggers taking samples)
                                             whereas the CPU-clock time does not. (min: 0)

      [BACKEND OPTIONS]  These options control region information captured w/o sampling or instrumentation

      -I, --include [ all | kokkosp | mpip | mutex-locks | ompt | rcclp | amd-smi | rocprofiler-sdk | rw-locks | spin-locks ]
                                    Include data from these backends (count: unlimited)
      -E, --exclude [ all | kokkosp | mpip | mutex-locks | ompt | rcclp | amd-smi | rocprofiler-sdk | rw-locks | spin-locks ]
                                    Exclude data from these backends (count: unlimited)

      [HARDWARE COUNTER OPTIONS] See also: rocprof-sys-avail -H

      -C, --cpu-events               Set the CPU hardware counter events to record (ref: `rocprof-sys-avail -H -c CPU`) (count: unlimited)
      -G, --gpu-events               Set the GPU hardware counter events to record (ref: `rocprof-sys-avail -H -c GPU`) (count: unlimited)

      [MISCELLANEOUS OPTIONS]

      -i, --inlines                  Include inline info in output when available (max: 1, dtype: bool)
      --hsa-interrupt [ 0 | 1 ]      Set the value of the HSA_ENABLE_INTERRUPT environment variable.
                                       ROCm version 5.2 and older have a bug which will cause a deadlock if a sample is taken while waiting for the signal
                                       that a kernel completed -- which happens when sampling with a real-clock timer. We require this option to be set to
                                       when --realtime is specified to make users aware that, while this may fix the bug, it can have a negative impact on
                                       performance.
                                       Values:
                                          0     avoid triggering the bug, potentially at the cost of reduced performance
                                          1     do not modify how ROCm is notified about kernel completion (count: 1, dtype: int)

The general syntax for separating ROCm Systems Profiler command-line arguments from the
following application arguments
is consistent with the LLVM style of using a stand-alone double hyphen (``--``).
All arguments preceding the double hyphen
are interpreted as belonging to ROCm Systems Profiler and all arguments following it
are interpreted as the
application and its arguments. The double hyphen is only necessary when passing
command-line arguments to a target
which also uses hyphens. For example, you can run ``rocprof-sys-sample ls``, but
to run ``ls -la``, use ``rocprof-sys-sample -- ls -la``.

:doc:`Configuring the ROCm Systems Profiler runtime options <./configuring-runtime-options>`
establishes the precedence of environment variable values over values specified
in the configuration files. This enables
you to configure the ROCm Systems Profiler runtime to your preferred default behavior
in a file such as ``~/.rocprof-sys.cfg`` and then easily override
those settings in the command line, for example, ``ROCPROFSYS_ENABLED=OFF rocprof-sys-sample -- foo``.
Similarly, the command-line arguments passed to ``rocprof-sys-sample`` take precedence
over environment variables.

All of the command-line options above correlate to one or more configuration
settings, for example, ``--cpu-events`` correlates to the ``ROCPROFSYS_PAPI_EVENTS`` configuration variable.
``rocprof-sys-sample`` processes the arguments and outputs a summary of its configuration
before running the target application.

The following snippets show how ``rocprof-sys-sample`` runs with various environment updates.

*  This snippet shows the environment updates when ``rocprof-sys-sample`` is invoked with no arguments:

   .. code-block:: shell

      $ rocprof-sys-sample -- ./parallel-overhead-locks 30 4 100

      LD_PRELOAD=/opt/rocprofiler-systems/lib/librocprof-sys-dl.so.1.7.1
      ROCPROFSYS_USE_PROCESS_SAMPLING=false
      ROCPROFSYS_USE_SAMPLING=true
      OMP_TOOL_LIBRARIES=/opt/rocprofiler-systems/lib/librocprof-sys-dl.so.1.7.1
      ROCP_TOOL_LIB=/opt/rocprofiler-systems/lib/librocprof-sys.so.1.7.1

*  The next snippet shows the environment updates when ``rocprof-sys-sample`` enables
   profiling, tracing, device process-sampling, and does not enable host process-sampling:

   .. code-block:: shell

      $ rocprof-sys-sample -PTD -- ./parallel-overhead-locks 30 4 100

      LD_PRELOAD=/opt/rocprofiler-systems/lib/librocprof-sys-dl.so.1.7.1
      ROCPROFSYS_CPU_FREQ_ENABLED=false
      ROCPROFSYS_PROFILE=true
      ROCPROFSYS_TRACE=true
      ROCPROFSYS_USE_AMD_SMI=true
      ROCPROFSYS_USE_PROCESS_SAMPLING=true
      ROCPROFSYS_USE_SAMPLING=true
      OMP_TOOL_LIBRARIES=/opt/rocprofiler-systems/lib/librocprof-sys-dl.so.1.7.1
      ROCP_TOOL_LIB=/opt/rocprofiler-systems/lib/librocprof-sys.so.1.7.1

*  The next snippet shows the environment updates when ``rocprof-sys-sample`` enables
   profiling, tracing, device process-sampling, host process-sampling, and all the available backends:

   .. code-block:: shell

      $ rocprof-sys-sample -PTDH -I all -- ./parallel-overhead-locks 30 4 100

      KOKKOS_TOOLS_LIBS=/opt/rocprofiler-systems/lib/librocprof-sys.so.1.7.1
      LD_PRELOAD=/opt/rocprofiler-systems/lib/librocprof-sys-dl.so.1.7.1
      ROCPROFSYS_CPU_FREQ_ENABLED=true
      ROCPROFSYS_TRACE_THREAD_LOCKS=true
      ROCPROFSYS_TRACE_THREAD_RW_LOCKS=true
      ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS=true
      ROCPROFSYS_USE_KOKKOSP=true
      ROCPROFSYS_USE_MPIP=true
      ROCPROFSYS_USE_OMPT=true
      ROCPROFSYS_TRACE=true
      ROCPROFSYS_USE_PROCESS_SAMPLING=true
      ROCPROFSYS_USE_RCCLP=true
      ROCPROFSYS_USE_AMD_SMI=true
      ROCPROFSYS_USE_ROCM=true
      ROCPROFSYS_USE_SAMPLING=true
      ROCPROFSYS_PROFILE=true
      OMP_TOOL_LIBRARIES=/opt/rocprofiler-systems/lib/librocprof-sys-dl.so.1.7.1
      ROCP_TOOL_LIB=/opt/rocprofiler-systems/lib/librocprof-sys.so.1.7.1
      ...

*  The final snippet shows the environment updates when ``rocprof-sys-sample`` enables
   profiling, tracing, device process-sampling, and host process-sampling,
   sets the output path to ``rocprof-sys-output`` and the output prefix to ``%tag%``, and disables
   all the available backends:

   .. code-block:: shell

      $ rocprof-sys-sample -PTDH -E all -o rocprof-sys-output %tag% -- ./parallel-overhead-locks 30 4 100

      LD_PRELOAD=/opt/rocprofiler-systems/lib/librocprof-sys-dl.so.1.7.1
      ROCPROFSYS_CPU_FREQ_ENABLED=true
      ROCPROFSYS_OUTPUT_PATH=rocprof-sys-output
      ROCPROFSYS_OUTPUT_PREFIX=%tag%
      ROCPROFSYS_TRACE_THREAD_LOCKS=false
      ROCPROFSYS_TRACE_THREAD_RW_LOCKS=false
      ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS=false
      ROCPROFSYS_USE_KOKKOSP=false
      ROCPROFSYS_USE_MPIP=false
      ROCPROFSYS_USE_OMPT=false
      ROCPROFSYS_TRACE=true
      ROCPROFSYS_USE_PROCESS_SAMPLING=true
      ROCPROFSYS_USE_RCCLP=false
      ROCPROFSYS_USE_AMD_SMI=false
      ROCPROFSYS_USE_ROCM=false
      ROCPROFSYS_USE_SAMPLING=true
      ROCPROFSYS_PROFILE=true
      ...

A rocprof-sys-sample example
========================================

Here is the full output from the previous
``rocprof-sys-sample -PTDH -E all -o rocprof-sys-output %tag% -- ./parallel-overhead-locks 30 4 100`` command:

.. code-block:: shell-session

   $ rocprof-sys-sample -PTDH -E all -o rocprof-sys-output %tag% -c -- ./parallel-overhead-locks 30 4 100

   LD_PRELOAD=/opt/rocprofiler-systems/lib/librocprof-sys-dl.so.1.11.3
   ROCPROFSYS_CONFIG_FILE=
   ROCPROFSYS_CPU_FREQ_ENABLED=true
   ROCPROFSYS_OUTPUT_PATH=rocprof-sys-output
   ROCPROFSYS_OUTPUT_PREFIX=%tag%
   ROCPROFSYS_PROFILE=true
   ROCPROFSYS_TRACE=true
   ROCPROFSYS_TRACE_THREAD_LOCKS=false
   ROCPROFSYS_TRACE_THREAD_RW_LOCKS=false
   ROCPROFSYS_TRACE_THREAD_SPIN_LOCKS=false
   ROCPROFSYS_USE_KOKKOSP=false
   ROCPROFSYS_USE_MPIP=false
   ROCPROFSYS_USE_OMPT=false
   ROCPROFSYS_USE_PROCESS_SAMPLING=true
   ROCPROFSYS_USE_RCCLP=false
   ROCPROFSYS_USE_AMD_SMI=false
   ROCPROFSYS_USE_ROCM=false
   ROCPROFSYS_USE_SAMPLING=true
   [rocprof-sys][dl][1785877] rocprofsys_main
   [rocprof-sys][1785877][rocprofsys_init_tooling] Instrumentation mode: Sampling
                                                     __
       _ __    ___     ___   _ __    _ __    ___    / _|          ___   _   _   ___
      | '__|  / _ \   / __| | '_ \  | '__|  / _ \  | |_   _____  / __| | | | | / __|
      | |    | (_) | | (__  | |_) | | |    | (_) | |  _| |_____| \__ \ | |_| | \__ \
      |_|     \___/   \___| | .__/  |_|     \___/  |_|           |___/  \__, | |___/
                            |_|                                         |___/

      rocprof-sys v1.11.2 (rev: 2586b74db8bf335742600010b8d9f1ce8da9cf89, compiler: GNU v11.4.1, rocm: v6.1.x)
   [988.958]       perfetto.cc:58649 Configured tracing session 1, #sources:1, duration:0 ms, #buffers:1, total buffer size:1024000 KB, total sessions:1, uid:0 session name: ""
   [parallel-overhead-locks] Threads: 4
   [parallel-overhead-locks] Iterations: 100
   [parallel-overhead-locks] fibonacci(30)...
   [1] number of iterations: 100
   [2] number of iterations: 100
   [3] number of iterations: 100
   [4] number of iterations: 100
   [parallel-overhead-locks] fibonacci(30) x 4 = 409221992
   [parallel-overhead-locks] number of mutex locks = 400
   [rocprof-sys][1785877][0][rocprofsys_finalize] finalizing...
   [rocprof-sys][1785877][0][rocprofsys_finalize]
   [rocprof-sys][1785877][0][rocprofsys_finalize] rocprof-sys/process/1785877 : 0.294342 sec wall_clock,    4.776 MB peak_rss,    3.170 MB page_rss, 0.990000 sec cpu_clock,  336.3 % cpu_util [laps: 1]
   [rocprof-sys][1785877][0][rocprofsys_finalize] rocprof-sys/process/1785877/thread/0 : 0.291535 sec wall_clock, 0.002619 sec thread_cpu_clock,    0.9 % thread_cpu_util,    4.776 MB peak_rss [laps: 1]
   [rocprof-sys][1785877][0][rocprofsys_finalize] rocprof-sys/process/1785877/thread/1 : 0.271353 sec wall_clock, 0.222572 sec thread_cpu_clock,   82.0 % thread_cpu_util,    4.200 MB peak_rss [laps: 1]
   [rocprof-sys][1785877][0][rocprofsys_finalize] rocprof-sys/process/1785877/thread/2 : 0.238218 sec wall_clock, 0.206405 sec thread_cpu_clock,   86.6 % thread_cpu_util,    3.432 MB peak_rss [laps: 1]
   [rocprof-sys][1785877][0][rocprofsys_finalize] rocprof-sys/process/1785877/thread/3 : 0.209459 sec wall_clock, 0.193415 sec thread_cpu_clock,   92.3 % thread_cpu_util,    2.472 MB peak_rss [laps: 1]
   [rocprof-sys][1785877][0][rocprofsys_finalize] rocprof-sys/process/1785877/thread/4 : 0.212029 sec wall_clock, 0.211694 sec thread_cpu_clock,   99.8 % thread_cpu_util,    1.152 MB peak_rss [laps: 1]
   [rocprof-sys][1785877][0][rocprofsys_finalize]
   [rocprof-sys][1785877][0][rocprofsys_finalize] Finalizing perfetto...
   [rocprof-sys][1785877][perfetto]> Outputting '/home/user/code/rocprofiler-systems/build-release/rocprofiler-systems-output/2024-07-15_16.21/parallel-overhead-locksperfetto-trace-1785877.proto' (39.12 KB / 0.04 MB / 0.00 GB)... Done
   [rocprof-sys][1785877][wall_clock]> Outputting 'rocprof-sys-output/2024-07-15_16.21/parallel-overhead-lockswall_clock-1785877.json'
   [rocprof-sys][1785877][wall_clock]> Outputting 'rocprof-sys-output/2024-07-15_16.21/parallel-overhead-lockswall_clock-1785877.txt'
   [rocprof-sys][1785877][metadata]> Outputting 'rocprof-sys-output/2024-07-15_16.21/parallel-overhead-locksmetadata-1785877.json' and 'rocprof-sys-output/2024-07-15_16.21/parallel-overhead-locksfunctions-1785877.json'
   [rocprof-sys][1785877][0][rocprofsys_finalize] Finalized: 0.054582 sec wall_clock,    0.000 MB peak_rss,   -1.798 MB page_rss, 0.040000 sec cpu_clock,   73.3 % cpu_util
   [989.312]       perfetto.cc:60128 Tracing session 1 ended, total sessions:0

