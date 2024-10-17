.. meta::
   :description: ROCm Systems Profiler documentation and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

****************************************************
Using the ROCm Systems Profiler API
****************************************************

The following example shows how a program can use the ROCm Systems Profiler API
for run-time analysis.

ROCm Systems Profiler user API example program
========================================

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
=======================================================

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

Assuming ROCm Systems Profiler is installed in ``/opt/rocprofiler-systems``, use the ``g++`` compiler
to build the application.

.. code-block:: shell

   g++ -I/opt/rocprofiler-systems foo.cpp -o foo -lrocprofiler-systems-user

Output from the API example program
========================================

First, instrument and run the program.

.. code-block:: shell

   $ rocprof-sys-instrument -l --min-instructions=8 -E custom_push_region -o -- ./user-api
   ...
   $ rocprof-sys-run --profile --use-pid off --time-output off -- ./user-api.inst 20 4 100
   Pushing custom region :: ./user-api.inst
   [rocprof-sys][rocprofsys_init_tooling] Instrumentation mode: Trace


                                                     __
       _ __    ___     ___   _ __    _ __    ___    / _|          ___   _   _   ___
      | '__|  / _ \   / __| | '_ \  | '__|  / _ \  | |_   _____  / __| | | | | / __|
      | |    | (_) | | (__  | |_) | | |    | (_) | |  _| |_____| \__ \ | |_| | \__ \
      |_|     \___/   \___| | .__/  |_|     \___/  |_|           |___/  \__, | |___/
                            |_|                                         |___/



   Pushing custom region :: initialization
   [./user-api.inst] Threads: 4
   [./user-api.inst] Iterations: 100
   [./user-api.inst] fibonacci(20)...
   Pushing custom region :: thread_creation
   Pushing custom region :: thread_wait
   Pushing custom region :: run(20) x 100
   Pushing custom region :: run(20) x 100
   Pushing custom region :: run(20) x 100
   Pushing custom region :: run(20) x 100
   Pushing custom region :: run(20) x 100
   [./user-api.inst] fibonacci(20) x 4 = 3382500
   [rocprof-sys][86267][0][rocprofsys_finalize] finalizing...


   [rocprof-sys][86267][0] rocprof-sys : 5.190895 sec wall_clock,    2.748 mb peak_rss, 6.330000 sec cpu_clock,  121.9 % cpu_util [laps: 1]
   [rocprof-sys][86267][0] user-api.inst/thread-0 : 5.078713 sec wall_clock, 4.722415 sec thread_cpu_clock,   93.0 % thread_cpu_util,    1.276 mb peak_rss [laps: 1]
   [rocprof-sys][86267][0] user-api.inst/thread-1 : 0.322248 sec wall_clock, 0.322191 sec thread_cpu_clock,  100.0 % thread_cpu_util,    1.000 mb peak_rss [laps: 1]
   [rocprof-sys][86267][0] user-api.inst/thread-2 : 0.323255 sec wall_clock, 0.323194 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.000 mb peak_rss [laps: 1]
   [rocprof-sys][86267][0] user-api.inst/thread-3 : 0.323569 sec wall_clock, 0.323484 sec thread_cpu_clock,  100.0 % thread_cpu_util,    1.092 mb peak_rss [laps: 1]
   [rocprof-sys][86267][0] user-api.inst/thread-4 : 0.324178 sec wall_clock, 0.324057 sec thread_cpu_clock,  100.0 % thread_cpu_util,    1.184 mb peak_rss [laps: 1]
   [rocprof-sys][86267][0] Post-processing 51 cpu frequency and memory usage entries...

   [rocprof-sys][wall_clock]|0> Outputting 'rocprof-sys-user-api.inst-output/wall_clock.json'...
   [rocprof-sys][wall_clock]|0> Outputting 'rocprof-sys-user-api.inst-output/wall_clock.tree.json'...
   [rocprof-sys][wall_clock]|0> Outputting 'rocprof-sys-user-api.inst-output/wall_clock.txt'...

   [rocprof-sys][manager::finalize][metadata]> Outputting 'rocprof-sys-user-api.inst-output/metadata.json' and 'rocprof-sys-user-api.inst-output/functions.json'...
   [rocprof-sys][86267][0][rocprofsys_finalize] Finalized

Then review the output.

.. code-block:: shell

   $ cat rocprof-sys-example-output/wall_clock.txt
   |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   |                                                                              REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                              |
   |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   |                                     LABEL                                       | COUNT  | DEPTH  |   METRIC   | UNITS  |   SUM    |   MEAN   |   MIN    |   MAX    |   VAR    | STDDEV   | % SELF |
   |---------------------------------------------------------------------------------|--------|--------|------------|--------|----------|----------|----------|----------|----------|----------|--------|
   | |0>>> ./user-api.inst                                                           |      1 |      0 | wall_clock | sec    | 5.078521 | 5.078521 | 5.078521 | 5.078521 | 0.000000 | 0.000000 |    0.0 |
   | |0>>> |_initialization                                                          |      1 |      1 | wall_clock | sec    | 0.000004 | 0.000004 | 0.000004 | 0.000004 | 0.000000 | 0.000000 |  100.0 |
   | |0>>> |_thread_creation                                                         |      1 |      1 | wall_clock | sec    | 0.000159 | 0.000159 | 0.000159 | 0.000159 | 0.000000 | 0.000000 |  100.0 |
   | |0>>> |_thread_wait                                                             |      1 |      1 | wall_clock | sec    | 0.355307 | 0.355307 | 0.355307 | 0.355307 | 0.000000 | 0.000000 |    0.0 |
   | |0>>>   |_std::vector<std::thread, std::allocator<std::thread> >::begin         |      1 |      2 | wall_clock | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |0>>>   |_std::vector<std::thread, std::allocator<std::thread> >::end           |      1 |      2 | wall_clock | sec    | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  100.0 |
   | |0>>>   |_pthread_join                                                          |      4 |      2 | wall_clock | sec    | 0.355257 | 0.088814 | 0.000001 | 0.333144 | 0.026559 | 0.162970 |  100.0 |
   | |2>>>     |_start_thread                                                        |      1 |      3 | wall_clock | sec    | 0.000032 | 0.000032 | 0.000032 | 0.000032 | 0.000000 | 0.000000 |  100.0 |
   | |1>>>     |_start_thread                                                        |      1 |      3 | wall_clock | sec    | 0.000036 | 0.000036 | 0.000036 | 0.000036 | 0.000000 | 0.000000 |  100.0 |
   | |3>>>     |_start_thread                                                        |      1 |      3 | wall_clock | sec    | 0.000034 | 0.000034 | 0.000034 | 0.000034 | 0.000000 | 0.000000 |  100.0 |
   | |4>>>     |_start_thread                                                        |      1 |      3 | wall_clock | sec    | 0.000039 | 0.000039 | 0.000039 | 0.000039 | 0.000000 | 0.000000 |  100.0 |
   | |0>>> |_run                                                                     |      1 |      1 | wall_clock | sec    | 4.722993 | 4.722993 | 4.722993 | 4.722993 | 0.000000 | 0.000000 |    0.0 |
   | |0>>>   |_std::char_traits<char>::length                                        |      1 |      2 | wall_clock | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |0>>>   |_std::distance<char const*>                                            |      1 |      2 | wall_clock | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |0>>>   |_std::operator+<char, std::char_traits<char>, std::allocator<char> >   |      2 |      2 | wall_clock | sec    | 0.000002 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |0>>>   |_run(20) x 100                                                         |      1 |      2 | wall_clock | sec    | 4.722951 | 4.722951 | 4.722951 | 4.722951 | 0.000000 | 0.000000 |    0.0 |
   | |0>>>     |_run [{94,25}-{96,25}]                                               |      1 |      3 | wall_clock | sec    | 4.722925 | 4.722925 | 4.722925 | 4.722925 | 0.000000 | 0.000000 |    0.0 |
   | |0>>>       |_fib                                                               |    100 |      4 | wall_clock | sec    | 4.722718 | 0.047227 | 0.046713 | 0.051987 | 0.000000 | 0.000625 |    0.0 |
   | |0>>>         |_fib                                                             |    200 |      5 | wall_clock | sec    | 4.722302 | 0.023612 | 0.017827 | 0.034091 | 0.000032 | 0.005627 |    0.0 |
   | |0>>>           |_fib                                                           |    400 |      6 | wall_clock | sec    | 4.721485 | 0.011804 | 0.006790 | 0.023003 | 0.000016 | 0.004024 |    0.0 |
   | |0>>>             |_fib                                                         |    800 |      7 | wall_clock | sec    | 4.719858 | 0.005900 | 0.002564 | 0.016078 | 0.000006 | 0.002498 |    0.1 |
   | |0>>>               |_fib                                                       |   1600 |      8 | wall_clock | sec    | 4.716572 | 0.002948 | 0.000977 | 0.011849 | 0.000002 | 0.001465 |    0.1 |
   | |0>>>                 |_fib                                                     |   3200 |      9 | wall_clock | sec    | 4.709918 | 0.001472 | 0.000371 | 0.008246 | 0.000001 | 0.000831 |    0.3 |
   | |0>>>                   |_fib                                                   |   6400 |     10 | wall_clock | sec    | 4.696775 | 0.000734 | 0.000140 | 0.005111 | 0.000000 | 0.000461 |    0.6 |
   | |0>>>                     |_fib                                                 |  12800 |     11 | wall_clock | sec    | 4.670093 | 0.000365 | 0.000050 | 0.003166 | 0.000000 | 0.000253 |    1.1 |
   | |0>>>                       |_fib                                               |  25600 |     12 | wall_clock | sec    | 4.617496 | 0.000180 | 0.000017 | 0.001959 | 0.000000 | 0.000137 |    2.3 |
   | |0>>>                         |_fib                                             |  51200 |     13 | wall_clock | sec    | 4.512671 | 0.000088 | 0.000004 | 0.001212 | 0.000000 | 0.000074 |    4.6 |
   | |0>>>                           |_fib                                           | 102400 |     14 | wall_clock | sec    | 4.304142 | 0.000042 | 0.000000 | 0.000752 | 0.000000 | 0.000039 |    9.6 |
   | |0>>>                             |_fib                                         | 202600 |     15 | wall_clock | sec    | 3.892580 | 0.000019 | 0.000000 | 0.000469 | 0.000000 | 0.000021 |   19.0 |
   | |0>>>                               |_fib                                       | 363200 |     16 | wall_clock | sec    | 3.151143 | 0.000009 | 0.000000 | 0.000293 | 0.000000 | 0.000011 |   33.2 |
   | |0>>>                                 |_fib                                     | 502000 |     17 | wall_clock | sec    | 2.105217 | 0.000004 | 0.000000 | 0.000183 | 0.000000 | 0.000006 |   49.1 |
   | |0>>>                                   |_fib                                   | 476000 |     18 | wall_clock | sec    | 1.071652 | 0.000002 | 0.000000 | 0.000114 | 0.000000 | 0.000004 |   63.6 |
   | |0>>>                                     |_fib                                 | 294200 |     19 | wall_clock | sec    | 0.390193 | 0.000001 | 0.000000 | 0.000071 | 0.000000 | 0.000003 |   75.3 |
   | |0>>>                                       |_fib                               | 115200 |     20 | wall_clock | sec    | 0.096190 | 0.000001 | 0.000000 | 0.000043 | 0.000000 | 0.000002 |   84.4 |
   | |0>>>                                         |_fib                             |  27400 |     21 | wall_clock | sec    | 0.015020 | 0.000001 | 0.000000 | 0.000025 | 0.000000 | 0.000001 |   91.1 |
   | |0>>>                                           |_fib                           |   3600 |     22 | wall_clock | sec    | 0.001336 | 0.000000 | 0.000000 | 0.000013 | 0.000000 | 0.000001 |   96.3 |
   | |0>>>                                             |_fib                         |    200 |     23 | wall_clock | sec    | 0.000050 | 0.000000 | 0.000000 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |0>>>     |_std::char_traits<char>::length                                      |      1 |      3 | wall_clock | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |0>>>     |_std::distance<char const*>                                          |      1 |      3 | wall_clock | sec    | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  100.0 |
   | |0>>>     |_std::operator+<char, std::char_traits<char>, std::allocator<char> > |      2 |      3 | wall_clock | sec    | 0.000001 | 0.000001 | 0.000000 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
   | |0>>> |_std::operator&                                                          |      1 |      1 | wall_clock | sec    | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  100.0 |
   | |0>>> std::vector<std::thread, std::allocator<std::thread> >::~vector           |      1 |      0 | wall_clock | sec    | 0.000045 | 0.000045 | 0.000045 | 0.000045 | 0.000000 | 0.000000 |   32.7 |
   | |0>>> |_std::thread::~thread                                                    |      4 |      1 | wall_clock | sec    | 0.000030 | 0.000007 | 0.000007 | 0.000009 | 0.000000 | 0.000001 |   31.2 |
   | |0>>>   |_std::thread::joinable                                                 |      4 |      2 | wall_clock | sec    | 0.000021 | 0.000005 | 0.000005 | 0.000006 | 0.000000 | 0.000001 |   89.4 |
   | |0>>>     |_std::thread::id::id                                                 |      4 |      3 | wall_clock | sec    | 0.000001 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  100.0 |
   | |0>>>     |_std::operator==                                                     |      4 |      3 | wall_clock | sec    | 0.000001 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  100.0 |
   | |0>>> |_std::allocator_traits<std::allocator<std::thread> >::deallocate         |      1 |      1 | wall_clock | sec    | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  100.0 |
   | |0>>> |_std::allocator<std::thread>::~allocator                                 |      1 |      1 | wall_clock | sec    | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  100.0 |
   |----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
