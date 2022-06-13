# User API

```eval_rst
.. doxygenfile:: omnitrace/user.h
```

By default, when omnitrace detects any `omnitrace_user_start_*` or `omnitrace_user_stop_*` function, instrumentation
is disabled at start-up -- thus, `omnitrace_user_stop_trace()` is not required at the beginning of main. This is
can be manually controlled via the `OMNITRACE_INIT_ENABLED` environment variable. User-defined regions are always
recorded, regardless of whether whether `omnitrace_user_start_*` or `omnitrace_user_stop_*` has been called.

## Example

### User API Implementation

```cpp
#include <omnitrace/user.h>

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
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
int (*omnitrace_push_region_f)(const char*) = nullptr;
}

int
main(int argc, char** argv)
{
    // get the internal callback to start a user-defined region
    omnitrace_user_get_callbacks(OMNITRACE_USER_REGION, (void**) &omnitrace_push_region_f,
                                 nullptr);
    // assign the custom callback to start a user-defined region
    if(omnitrace_push_region_f)
        omnitrace_user_configure(OMNITRACE_USER_REGION, (void*) &custom_push_region,
                                 nullptr);

    omnitrace_user_push_region(argv[0]);
    omnitrace_user_push_region("initialization");
    size_t nthread = std::min<size_t>(16, std::thread::hardware_concurrency());
    size_t nitr    = 50000;
    long   nfib    = 10;
    if(argc > 1) nfib = atol(argv[1]);
    if(argc > 2) nthread = atol(argv[2]);
    if(argc > 3) nitr = atol(argv[3]);
    omnitrace_user_pop_region("initialization");

    printf("[%s] Threads: %zu\n[%s] Iterations: %zu\n[%s] fibonacci(%li)...\n", argv[0],
           nthread, argv[0], nitr, argv[0], nfib);

    omnitrace_user_push_region("thread_creation");
    std::vector<std::thread> threads{};
    threads.reserve(nthread);
    // disable instrumentation for child threads
    omnitrace_user_stop_thread_trace();
    for(size_t i = 0; i < nthread; ++i)
    {
        size_t _nitr = ((i % 2) == 1) ? (nitr - (0.1 * nitr)) : (nitr + (0.1 * nitr));
        long   _nfib = ((i % 2) == 1) ? (nfib - (0.1 * nfib)) : (nfib + (0.1 * nfib));
        threads.emplace_back(&run, _nitr, _nfib);
    }
    // re-enable instrumentation
    omnitrace_user_start_thread_trace();
    omnitrace_user_pop_region("thread_creation");

    omnitrace_user_push_region("thread_wait");
    for(auto& itr : threads)
        itr.join();
    omnitrace_user_pop_region("thread_wait");

    run(nitr, nfib);

    printf("[%s] fibonacci(%li) x %lu = %li\n", argv[0], nfib, nthread, total.load());
    omnitrace_user_pop_region(argv[0]);

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
    omnitrace_user_push_region(RUN_LABEL);
    long local = 0;
    for(size_t i = 0; i < nitr; ++i)
        local += fib(n);
    total += local;
    omnitrace_user_pop_region(RUN_LABEL);
}

int
custom_push_region(const char* name)
{
    printf("Pushing custom region :: %s\n", name);
    return (*omnitrace_push_region_f)(name);
}
```

### User API Output

```console
$ omnitrace -l --min-instructions=8 -E custom_push_region -o -- ./user-api
...
$ export OMNITRACE_USE_TIMEMORY=ON
$ export OMNITRACE_USE_PID=OFF
$ export OMNITRACE_TIME_OUTPUT=OFF
$ export OMNITRACE_OUTPUT_PATH=omnitrace-example-output
$ ./user-api.inst 20 4 100
Pushing custom region :: ./user-api.inst
[omnitrace][omnitrace_init_tooling] Instrumentation mode: Trace


      ______   .___  ___. .__   __.  __  .___________..______          ___       ______  _______
     /  __  \  |   \/   | |  \ |  | |  | |           ||   _  \        /   \     /      ||   ____|
    |  |  |  | |  \  /  | |   \|  | |  | `---|  |----`|  |_)  |      /  ^  \   |  ,----'|  |__
    |  |  |  | |  |\/|  | |  . `  | |  |     |  |     |      /      /  /_\  \  |  |     |   __|
    |  `--'  | |  |  |  | |  |\   | |  |     |  |     |  |\  \----./  _____  \ |  `----.|  |____
     \______/  |__|  |__| |__| \__| |__|     |__|     | _| `._____/__/     \__\ \______||_______|



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
[omnitrace][86267][0][omnitrace_finalize] finalizing...


[omnitrace][86267][0] omnitrace : 5.190895 sec wall_clock,    2.748 mb peak_rss, 6.330000 sec cpu_clock,  121.9 % cpu_util [laps: 1]
[omnitrace][86267][0] user-api.inst/thread-0 : 5.078713 sec wall_clock, 4.722415 sec thread_cpu_clock,   93.0 % thread_cpu_util,    1.276 mb peak_rss [laps: 1]
[omnitrace][86267][0] user-api.inst/thread-1 : 0.322248 sec wall_clock, 0.322191 sec thread_cpu_clock,  100.0 % thread_cpu_util,    1.000 mb peak_rss [laps: 1]
[omnitrace][86267][0] user-api.inst/thread-2 : 0.323255 sec wall_clock, 0.323194 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.000 mb peak_rss [laps: 1]
[omnitrace][86267][0] user-api.inst/thread-3 : 0.323569 sec wall_clock, 0.323484 sec thread_cpu_clock,  100.0 % thread_cpu_util,    1.092 mb peak_rss [laps: 1]
[omnitrace][86267][0] user-api.inst/thread-4 : 0.324178 sec wall_clock, 0.324057 sec thread_cpu_clock,  100.0 % thread_cpu_util,    1.184 mb peak_rss [laps: 1]
[omnitrace][86267][0] Post-processing 51 cpu frequency and memory usage entries...

[omnitrace][wall_clock]|0> Outputting 'omnitrace-user-api.inst-output/wall_clock.json'...
[omnitrace][wall_clock]|0> Outputting 'omnitrace-user-api.inst-output/wall_clock.tree.json'...
[omnitrace][wall_clock]|0> Outputting 'omnitrace-user-api.inst-output/wall_clock.txt'...

[omnitrace][manager::finalize][metadata]> Outputting 'omnitrace-user-api.inst-output/metadata.json' and 'omnitrace-user-api.inst-output/functions.json'...
[omnitrace][86267][0][omnitrace_finalize] Finalized
$ cat omnitrace-example-output/wall_clock.txt
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
```
