.. meta::
   :description: ROCm Systems Profiler documentation and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

****************************************************
Using the ROCm Systems Profiler API
****************************************************

The following example shows how a program can use the ROCm Systems Profiler API
for run-time analysis.

ROCm Systems Profiler user API example program
==============================================

You can use the ROCm Systems Profiler API to define custom regions to profile and trace.
The following C++ program demonstrates this technique by calling several functions from the
ROCm Systems Profiler API, such as ``rocprofsys_user_push_region`` and
``rocprofsys_user_stop_thread_trace``.

.. note::

   By default, when ROCm Systems Profiler detects any ``rocprofsys_user_start_*`` or
   ``rocprofsys_user_stop_*`` function, instrumentation
   is disabled at start up, which means ``rocprofsys_user_stop_trace()`` is not
   required at the beginning of ``main``. This behavior
   can be manually controlled by using the ``ROCPROFSYS_INIT_ENABLED`` environment variable.
   User-defined regions are always
   recorded, regardless of whether ``rocprofsys_user_start_*`` or
   ``rocprofsys_user_stop_*`` has been called.

.. code-block:: shell

   #include <rocprofiler-systems/categories.h>
   #include <rocprofiler-systems/types.h>
   #include <rocprofiler-systems/user.h>

   #include <atomic>
   #include <cassert>
   #include <cerrno>
   #include <cstdio>
   #include <cstdlib>
   #include <cstring>
   #include <sstream>
   #include <thread>
   #include <vector>

   std::atomic<long> total{ 0 };

   long
   fib(long n) __attribute__((noinline));

   void
   run(size_t nitr, long) __attribute__((noinline));

   int
   custom_push_region(const char* name);

   namespace
   {
   rocprofsys_user_callbacks_t custom_callbacks   = ROCPROFSYS_USER_CALLBACKS_INIT;
   rocprofsys_user_callbacks_t original_callbacks = ROCPROFSYS_USER_CALLBACKS_INIT;
   }  // namespace

   int
   main(int argc, char** argv)
   {
      custom_callbacks.push_region = &custom_push_region;
      rocprofsys_user_configure(ROCPROFSYS_USER_UNION_CONFIG, custom_callbacks,
                              &original_callbacks);

      rocprofsys_user_push_region(argv[0]);
      rocprofsys_user_push_region("initialization");
      size_t nthread = std::min<size_t>(16, std::thread::hardware_concurrency());
      size_t nitr    = 50000;
      long   nfib    = 10;
      if(argc > 1) nfib = atol(argv[1]);
      if(argc > 2) nthread = atol(argv[2]);
      if(argc > 3) nitr = atol(argv[3]);
      rocprofsys_user_pop_region("initialization");

      printf("[%s] Threads: %zu\n[%s] Iterations: %zu\n[%s] fibonacci(%li)...\n", argv[0],
            nthread, argv[0], nitr, argv[0], nfib);

      rocprofsys_user_push_region("thread_creation");
      std::vector<std::thread> threads{};
      threads.reserve(nthread);
      // disable instrumentation for child threads
      rocprofsys_user_stop_thread_trace();
      for(size_t i = 0; i < nthread; ++i)
      {
         threads.emplace_back(&run, nitr, nfib);
      }
      // re-enable instrumentation
      rocprofsys_user_start_thread_trace();
      rocprofsys_user_pop_region("thread_creation");

      rocprofsys_user_push_region("thread_wait");
      for(auto& itr : threads)
         itr.join();
      rocprofsys_user_pop_region("thread_wait");

      run(nitr, nfib);

      printf("[%s] fibonacci(%li) x %lu = %li\n", argv[0], nfib, nthread, total.load());
      rocprofsys_user_pop_region(argv[0]);

      return 0;
   }

   long
   fib(long n)
   {
      return (n < 2) ? n : fib(n - 1) + fib(n - 2);
   }

   #define RUN_LABEL                                                                        \
      std::string{ std::string{ __FUNCTION__ } + "(" + std::to_string(n) + ") x " +        \
                  std::to_string(nitr) }                                                  \
         .c_str()

   void
   run(size_t nitr, long n)
   {
      rocprofsys_user_push_region(RUN_LABEL);
      long local = 0;
      for(size_t i = 0; i < nitr; ++i)
         local += fib(n);
      total += local;
      rocprofsys_user_pop_region(RUN_LABEL);
   }

   int
   custom_push_region(const char* name)
   {
      if(!original_callbacks.push_region || !original_callbacks.push_annotated_region)
         return ROCPROFSYS_USER_ERROR_NO_BINDING;

      printf("Pushing custom region :: %s\n", name);

      if(original_callbacks.push_annotated_region)
      {
         int32_t _err = errno;
         char*   _msg = nullptr;
         char    _buff[1024];
         if(_err != 0) _msg = strerror_r(_err, _buff, sizeof(_buff));

         rocprofsys_annotation_t _annotations[] = {
               { "errno", ROCPROFSYS_INT32, &_err }, { "strerror", ROCPROFSYS_STRING, _msg }
         };

         errno = 0;  // reset errno
         return (*original_callbacks.push_annotated_region)(
               name, _annotations, sizeof(_annotations) / sizeof(rocprofsys_annotation_t));
      }

      return (*original_callbacks.push_region)(name);
   }

Linking the ROCm Systems Profiler libraries to another program
==============================================================

To link the ``rocprofiler-systems-user-library`` to another program,
use the following CMake and ``g++`` directives.

CMake
-------------------------------------------------------

.. code-block:: cmake

   find_package(rocprofiler-systems REQUIRED COMPONENTS user)
   add_executable(foo foo.cpp)
   target_link_libraries(foo PRIVATE rocprofiler-systems::rocprofiler-systems-user-library)

g++ compilation
-------------------------------------------------------

Assuming ROCm Systems Profiler is installed in ``/opt/rocprofsys``, use the ``g++`` compiler
to build the application.

.. code-block:: shell

   g++ -g -I/opt/rocprofsys/include -L/opt/rocprofsys/lib foo.cpp -o foo -lrocprof-sys-user

Output from the API example program
========================================

First, instrument and run the program.

.. code-block:: shell-session

   $ rocprof-sys-instrument -l --min-instructions=8 -E custom_push_region -o user-api.inst -- ./user-api
   ...
   $ rocprof-sys-run --profile --trace -- ./user-api.inst 10 12 1000

   ROCPROFSYS: LD_PRELOAD=/home/rocm-dev/code/rocprofiler-systems/build/ubuntu/22.04/lib/librocprof-sys-dl.so.1.0.0
   ROCPROFSYS: OMP_TOOL_LIBRARIES=/home/rocm-dev/code/rocprofiler-systems/build/ubuntu/22.04/lib/librocprof-sys-dl.so.1.0.0
   ROCPROFSYS: ROCPROFSYS_PROFILE=true
   ROCPROFSYS: ROCPROFSYS_TRACE=true
   ROCPROFSYS: ROCPROFSYS_VERBOSE=0
   [rocprof-sys][dl][1827155] rocprofsys_main
   [rocprof-sys][1827155][rocprofsys_init_tooling] Instrumentation mode: Trace


       ____   ___   ____ __  __   ______   ______ _____ _____ __  __ ____    ____  ____   ___  _____ ___ _     _____ ____
      |  _ \ / _ \ / ___|  \/  | / ___\ \ / / ___|_   _| ____|  \/  / ___|  |  _ \|  _ \ / _ \|  ___|_ _| |   | ____|  _ \
      | |_) | | | | |   | |\/| | \___ \\ V /\___ \ | | |  _| | |\/| \___ \  | |_) | |_) | | | | |_   | || |   |  _| | |_) |
      |  _ <| |_| | |___| |  | |  ___) || |  ___) || | | |___| |  | |___) | |  __/|  _ <| |_| |  _|  | || |___| |___|  _ <
      |_| \_\\___/ \____|_|  |_| |____/ |_| |____/ |_| |_____|_|  |_|____/  |_|   |_| \_\\___/|_|   |___|_____|_____|_| \_\

      rocprof-sys v1.0.0 (rev: 3213dc652728f7ed01b62bf55f6af76c43bfcbdb, x86_64-linux-gnu, compiler: GNU v11.4.0, rocm: v6.3.x)
   [790.763]       perfetto.cc:47606 Configured tracing session 1, #sources:1, duration:0 ms, #buffers:1, total buffer size:1024000 KB, total sessions:1, uid:0 session name: ""
   [./user-api.inst] Threads: 12
   [./user-api.inst] Iterations: 1000
   [./user-api.inst] fibonacci(10)...
   [./user-api.inst] fibonacci(10) x 12 = 715000

   [rocprof-sys][1827155][0][rocprofsys_finalize] finalizing...
   [rocprof-sys][1827155][0][rocprofsys_finalize]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155 : 0.137404 sec wall_clock,    6.528 MB peak_rss,    6.685 MB page_rss, 0.540000 sec cpu_clock,  393.0 % cpu_util [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/0 : 0.135815 sec wall_clock, 0.035171 sec thread_cpu_clock,   25.9 % thread_cpu_util,    6.016 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/1 : 0.028336 sec wall_clock, 0.028336 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.640 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/2 : 0.030380 sec wall_clock, 0.030380 sec thread_cpu_clock,  100.0 % thread_cpu_util,    3.840 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/3 : 0.035233 sec wall_clock, 0.035227 sec thread_cpu_clock,  100.0 % thread_cpu_util,    3.840 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/4 : 0.035275 sec wall_clock, 0.035267 sec thread_cpu_clock,  100.0 % thread_cpu_util,    3.840 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/5 : 0.035452 sec wall_clock, 0.035452 sec thread_cpu_clock,  100.0 % thread_cpu_util,    3.840 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/6 : 0.036198 sec wall_clock, 0.036190 sec thread_cpu_clock,  100.0 % thread_cpu_util,    3.840 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/7 : 0.034709 sec wall_clock, 0.034702 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.640 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/8 : 0.036590 sec wall_clock, 0.033590 sec thread_cpu_clock,   91.8 % thread_cpu_util,    0.512 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/9 : 0.033108 sec wall_clock, 0.033098 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.384 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/10 : 0.032993 sec wall_clock, 0.032994 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.256 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/11 : 0.035687 sec wall_clock, 0.035368 sec thread_cpu_clock,   99.1 % thread_cpu_util,    0.128 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize] rocprofsys/process/1827155/thread/12 : 0.035359 sec wall_clock, 0.035354 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.128 MB peak_rss [laps: 1]
   [rocprof-sys][1827155][0][rocprofsys_finalize]
   [rocprof-sys][1827155][0][rocprofsys_finalize] Finalizing perfetto...
   [rocprofiler-systems][1827155][perfetto]> Outputting '/home/rocm-dev/opt/user-api-test/rocprofsys-user-api.inst-output/2025-01-15_17.57/perfetto-trace-1827155.proto' (17.20 KB / 0.02 MB / 0.00 GB)... Done
   [rocprofiler-systems][1827155][wall_clock]> Outputting 'rocprofsys-user-api.inst-output/2025-01-15_17.57/wall_clock-1827155.json'
   [rocprofiler-systems][1827155][wall_clock]> Outputting 'rocprofsys-user-api.inst-output/2025-01-15_17.57/wall_clock-1827155.txt'
   [rocprofiler-systems][1827155][metadata]> Outputting 'rocprofsys-user-api.inst-output/2025-01-15_17.57/metadata-1827155.json' and 'rocprofsys-user-api.inst-output/2025-01-15_17.57/functions-1827155.json'
   [rocprof-sys][1827155][0][rocprofsys_finalize] Finalized: 0.048039 sec wall_clock,    0.640 MB peak_rss,    0.655 MB page_rss, 0.020000 sec cpu_clock,   41.6 % cpu_util
   [790.953]       perfetto.cc:49204 Tracing session 1 ended, total sessions:0

Then review the output.

.. code-block:: shell

   $ cat rocprof-sys-example-output/wall_clock.txt
   |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   |                                                                                          REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                                          |
   |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   |                                                 LABEL                                                   | COUNT  | DEPTH  |   METRIC   | UNITS  |   SUM    |   MEAN   |   MIN    |   MAX    |   VAR    | STDDEV   | % SELF |
   |---------------------------------------------------------------------------------------------------------|--------|--------|------------|--------|----------|----------|----------|----------|----------|----------|--------|
   | |00>>> ./user-api.inst                                                                                  |      1 |      0 | wall_clock | sec    | 0.874293 | 0.874293 | 0.874293 | 0.874293 | 0.000000 | 0.000000 |    0.0 |
   | |00>>> |_initialization                                                                                 |      1 |      1 | wall_clock | sec    | 0.000015 | 0.000015 | 0.000015 | 0.000015 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> |_thread_creation                                                                                |      1 |      1 | wall_clock | sec    | 0.059934 | 0.059934 | 0.059934 | 0.059934 | 0.000000 | 0.000000 |    1.0 |
   | |00>>>   |_pthread_create                                                                               |     12 |      2 | wall_clock | sec    | 0.059338 | 0.004945 | 0.004455 | 0.005743 | 0.000000 | 0.000479 |    0.0 |
   | |01>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.027499 | 0.027499 | 0.027499 | 0.027499 | 0.000000 | 0.000000 |    0.1 |
   | |01>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.027463 | 0.027463 | 0.027463 | 0.027463 | 0.000000 | 0.000000 |  100.0 |
   | |02>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.027804 | 0.027804 | 0.027804 | 0.027804 | 0.000000 | 0.000000 |    0.2 |
   | |02>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.027752 | 0.027752 | 0.027752 | 0.027752 | 0.000000 | 0.000000 |  100.0 |
   | |03>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.027567 | 0.027567 | 0.027567 | 0.027567 | 0.000000 | 0.000000 |    0.1 |
   | |03>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.027529 | 0.027529 | 0.027529 | 0.027529 | 0.000000 | 0.000000 |  100.0 |
   | |05>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.027699 | 0.027699 | 0.027699 | 0.027699 | 0.000000 | 0.000000 |    0.2 |
   | |05>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.027651 | 0.027651 | 0.027651 | 0.027651 | 0.000000 | 0.000000 |  100.0 |
   | |04>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.033427 | 0.033427 | 0.033427 | 0.033427 | 0.000000 | 0.000000 |    0.2 |
   | |04>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.033376 | 0.033376 | 0.033376 | 0.033376 | 0.000000 | 0.000000 |  100.0 |
   | |06>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.032210 | 0.032210 | 0.032210 | 0.032210 | 0.000000 | 0.000000 |    0.1 |
   | |06>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.032168 | 0.032168 | 0.032168 | 0.032168 | 0.000000 | 0.000000 |  100.0 |
   | |07>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.030176 | 0.030176 | 0.030176 | 0.030176 | 0.000000 | 0.000000 |    0.2 |
   | |07>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.030122 | 0.030122 | 0.030122 | 0.030122 | 0.000000 | 0.000000 |  100.0 |
   | |08>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.027941 | 0.027941 | 0.027941 | 0.027941 | 0.000000 | 0.000000 |    0.1 |
   | |08>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.027899 | 0.027899 | 0.027899 | 0.027899 | 0.000000 | 0.000000 |  100.0 |
   | |09>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.034679 | 0.034679 | 0.034679 | 0.034679 | 0.000000 | 0.000000 |    0.1 |
   | |09>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.034636 | 0.034636 | 0.034636 | 0.034636 | 0.000000 | 0.000000 |  100.0 |
   | |11>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.028143 | 0.028143 | 0.028143 | 0.028143 | 0.000000 | 0.000000 |    0.1 |
   | |11>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.028103 | 0.028103 | 0.028103 | 0.028103 | 0.000000 | 0.000000 |  100.0 |
   | |10>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.033393 | 0.033393 | 0.033393 | 0.033393 | 0.000000 | 0.000000 |    0.1 |
   | |10>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.033354 | 0.033354 | 0.033354 | 0.033354 | 0.000000 | 0.000000 |  100.0 |
   | |12>>>     |_start_thread                                                                               |      1 |      3 | wall_clock | sec    | 0.027765 | 0.027765 | 0.027765 | 0.027765 | 0.000000 | 0.000000 |    0.2 |
   | |12>>>       |_run(10) x 1000                                                                           |      1 |      4 | wall_clock | sec    | 0.027710 | 0.027710 | 0.027710 | 0.027710 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> |_thread_wait                                                                                    |      1 |      1 | wall_clock | sec    | 0.027971 | 0.027971 | 0.027971 | 0.027971 | 0.000000 | 0.000000 |    1.3 |
   | |00>>>   |_std::vector<std::thread, std::allocator<std::thread> >::begin                                |      1 |      2 | wall_clock | sec    | 0.000003 | 0.000003 | 0.000003 | 0.000003 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>   |_std::vector<std::thread, std::allocator<std::thread> >::end                                  |      1 |      2 | wall_clock | sec    | 0.000002 | 0.000002 | 0.000002 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>   |___gnu_cxx::operator!=<std::thread*, std::vector<std::thread, std::allocator<std::thread> > > |     13 |      2 | wall_clock | sec    | 0.000024 | 0.000002 | 0.000001 | 0.000003 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>   |_pthread_join                                                                                 |     12 |      2 | wall_clock | sec    | 0.027583 | 0.002299 | 0.000003 | 0.011250 | 0.000011 | 0.003289 |  100.0 |
   | |00>>> |_run                                                                                            |      1 |      1 | wall_clock | sec    | 0.786236 | 0.786236 | 0.786236 | 0.786236 | 0.000000 | 0.000000 |    0.0 |
   | |00>>>   |_std::char_traits<char>::length                                                               |      1 |      2 | wall_clock | sec    | 0.000002 | 0.000002 | 0.000002 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>   |_std::distance<char const*>                                                                   |      1 |      2 | wall_clock | sec    | 0.000002 | 0.000002 | 0.000002 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>   |_std::operator+<char, std::char_traits<char>, std::allocator<char> >                          |      4 |      2 | wall_clock | sec    | 0.000006 | 0.000002 | 0.000001 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>   |_run(10) x 1000                                                                               |      1 |      2 | wall_clock | sec    | 0.786184 | 0.786184 | 0.786184 | 0.786184 | 0.000000 | 0.000000 |    0.0 |
   | |00>>>     |_run [{95,25}-{97,25}]                                                                      |      1 |      3 | wall_clock | sec    | 0.786141 | 0.786141 | 0.786141 | 0.786141 | 0.000000 | 0.000000 |    0.4 |
   | |00>>>       |_fib                                                                                      |   1000 |      4 | wall_clock | sec    | 0.782692 | 0.000783 | 0.000757 | 0.001397 | 0.000000 | 0.000026 |    1.0 |
   | |00>>>         |_fib                                                                                    |   2000 |      5 | wall_clock | sec    | 0.774875 | 0.000387 | 0.000282 | 0.000863 | 0.000000 | 0.000095 |    2.0 |
   | |00>>>           |_fib                                                                                  |   4000 |      6 | wall_clock | sec    | 0.759351 | 0.000190 | 0.000101 | 0.000570 | 0.000000 | 0.000068 |    4.0 |
   | |00>>>             |_fib                                                                                |   8000 |      7 | wall_clock | sec    | 0.728911 | 0.000091 | 0.000034 | 0.000350 | 0.000000 | 0.000042 |    8.5 |
   | |00>>>               |_fib                                                                              |  16000 |      8 | wall_clock | sec    | 0.666793 | 0.000042 | 0.000009 | 0.000206 | 0.000000 | 0.000025 |   18.5 |
   | |00>>>                 |_fib                                                                            |  32000 |      9 | wall_clock | sec    | 0.543524 | 0.000017 | 0.000001 | 0.000121 | 0.000000 | 0.000014 |   38.3 |
   | |00>>>                   |_fib                                                                          |  52000 |     10 | wall_clock | sec    | 0.335118 | 0.000006 | 0.000001 | 0.000070 | 0.000000 | 0.000008 |   61.0 |
   | |00>>>                     |_fib                                                                        |  44000 |     11 | wall_clock | sec    | 0.130629 | 0.000003 | 0.000001 | 0.000036 | 0.000000 | 0.000004 |   79.4 |
   | |00>>>                       |_fib                                                                      |  16000 |     12 | wall_clock | sec    | 0.026893 | 0.000002 | 0.000001 | 0.000026 | 0.000000 | 0.000002 |   91.5 |
   | |00>>>                         |_fib                                                                    |   2000 |     13 | wall_clock | sec    | 0.002279 | 0.000001 | 0.000001 | 0.000014 | 0.000000 | 0.000001 |  100.0 |
   | |00>>>     |_std::char_traits<char>::length                                                             |      1 |      3 | wall_clock | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>     |_std::distance<char const*>                                                                 |      1 |      3 | wall_clock | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |00>>>     |_std::operator+<char, std::char_traits<char>, std::allocator<char> >                        |      4 |      3 | wall_clock | sec    | 0.000005 | 0.000001 | 0.000001 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> |_std::operator&                                                                                 |      1 |      1 | wall_clock | sec    | 0.000003 | 0.000003 | 0.000003 | 0.000003 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> std::vector<std::thread, std::allocator<std::thread> >::~vector                                  |      1 |      0 | wall_clock | sec    | 0.000256 | 0.000256 | 0.000256 | 0.000256 | 0.000000 | 0.000000 |   20.9 |
   | |00>>> |_std::thread::~thread                                                                           |     12 |      1 | wall_clock | sec    | 0.000193 | 0.000016 | 0.000014 | 0.000025 | 0.000000 | 0.000004 |   31.9 |
   | |00>>>   |_std::thread::joinable                                                                        |     12 |      2 | wall_clock | sec    | 0.000131 | 0.000011 | 0.000010 | 0.000017 | 0.000000 | 0.000003 |   77.6 |
   | |00>>>     |_std::thread::id::id                                                                        |     12 |      3 | wall_clock | sec    | 0.000016 | 0.000001 | 0.000001 | 0.000004 | 0.000000 | 0.000001 |  100.0 |
   | |00>>>     |_std::operator==                                                                            |     12 |      3 | wall_clock | sec    | 0.000014 | 0.000001 | 0.000001 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> |_std::allocator_traits<std::allocator<std::thread> >::deallocate                                |      1 |      1 | wall_clock | sec    | 0.000008 | 0.000008 | 0.000008 | 0.000008 | 0.000000 | 0.000000 |   70.8 |
   | |00>>>   |___gnu_cxx::new_allocator<std::thread>::deallocate                                            |      1 |      2 | wall_clock | sec    | 0.000002 | 0.000002 | 0.000002 | 0.000002 | 0.000000 | 0.000000 |  100.0 |
   | |00>>> |_std::allocator<std::thread>::~allocator                                                        |      1 |      1 | wall_clock | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
