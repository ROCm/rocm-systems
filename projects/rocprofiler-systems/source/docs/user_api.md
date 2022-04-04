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
$ omnitrace -l --min-address-range=0 --min-address-range-loop=0 --min-instructions=8 -E custom_push_region -o -- ./user-api
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
Pushing custom region :: run(20) x 100
Pushing custom region :: thread_wait
Pushing custom region :: run(20) x 100
Pushing custom region :: run(20) x 100
Pushing custom region :: run(20) x 100
Pushing custom region :: run(20) x 100
[./user-api.inst] fibonacci(20) x 4 = 3382500


[omnitrace][2637959][0] omnitrace : 2.716905 sec wall_clock,    1.216 mb peak_rss, 3.680000 sec cpu_clock,  135.4 % cpu_util [laps: 1]
[omnitrace][2637959][0] user-api.inst/thread-0 : 2.715708 sec wall_clock, 2.354223 sec thread_cpu_clock,   86.7 % thread_cpu_util,    1.216 mb peak_rss [laps: 1]
[omnitrace][2637959][0] user-api.inst/thread-1 : 0.329802 sec wall_clock, 0.329739 sec thread_cpu_clock,  100.0 % thread_cpu_util,    0.000 mb peak_rss [laps: 1]
[omnitrace][2637959][0] user-api.inst/thread-2 : 0.355981 sec wall_clock, 0.335795 sec thread_cpu_clock,   94.3 % thread_cpu_util,    0.528 mb peak_rss [laps: 1]
[omnitrace][2637959][0] user-api.inst/thread-3 : 0.341329 sec wall_clock, 0.331214 sec thread_cpu_clock,   97.0 % thread_cpu_util,    0.456 mb peak_rss [laps: 1]
[omnitrace][2637959][0] user-api.inst/thread-4 : 0.360631 sec wall_clock, 0.330374 sec thread_cpu_clock,   91.6 % thread_cpu_util,    0.600 mb peak_rss [laps: 1]
[wall_clock]|0> Outputting 'omnitrace-example-output/wall_clock.json'...
[wall_clock]|0> Outputting 'omnitrace-example-output/wall_clock.tree.json'...
[wall_clock]|0> Outputting 'omnitrace-example-output/wall_clock.txt'...


[metadata::manager::finalize]> Outputting 'omnitrace-example-output/metadata.json' and 'omnitrace-example-output/functions.json'...
$ cat omnitrace-example-output/wall_clock.txt
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                          REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                LABEL                                  | COUNT   | DEPTH  |   METRIC   | UNITS  |   SUM    |   MEAN   |   MIN    |   MAX    |   VAR    | STDDEV   | % SELF |
|-----------------------------------------------------------------------|---------|--------|------------|--------|----------|----------|----------|----------|----------|----------|--------|
| |0>>> ./user-api.inst                                                 |       1 |      0 | wall_clock | sec    | 2.715611 | 2.715611 | 2.715611 | 2.715611 | 0.000000 | 0.000000 |    0.0 |
| |0>>> |_initialization                                                |       1 |      1 | wall_clock | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
| |0>>> |_thread_creation                                               |       1 |      1 | wall_clock | sec    | 0.000170 | 0.000170 | 0.000170 | 0.000170 | 0.000000 | 0.000000 |  100.0 |
| |0>>> |_thread_wait                                                   |       1 |      1 | wall_clock | sec    | 0.360751 | 0.360751 | 0.360751 | 0.360751 | 0.000000 | 0.000000 |    0.0 |
| |1>>>   |_run(20) x 100                                               |       1 |      2 | wall_clock | sec    | 0.329472 | 0.329472 | 0.329472 | 0.329472 | 0.000000 | 0.000000 |  100.0 |
| |3>>>   |_run(20) x 100                                               |       1 |      2 | wall_clock | sec    | 0.331028 | 0.331028 | 0.331028 | 0.331028 | 0.000000 | 0.000000 |  100.0 |
| |2>>>   |_run(20) x 100                                               |       1 |      2 | wall_clock | sec    | 0.335554 | 0.335554 | 0.335554 | 0.335554 | 0.000000 | 0.000000 |  100.0 |
| |4>>>   |_run(20) x 100                                               |       1 |      2 | wall_clock | sec    | 0.330220 | 0.330220 | 0.330220 | 0.330220 | 0.000000 | 0.000000 |  100.0 |
| |0>>> |_run                                                           |       1 |      1 | wall_clock | sec    | 2.354618 | 2.354618 | 2.354618 | 2.354618 | 0.000000 | 0.000000 |    0.0 |
| |0>>>   |_run(20) x 100                                               |       1 |      2 | wall_clock | sec    | 2.354600 | 2.354600 | 2.354600 | 2.354600 | 0.000000 | 0.000000 |   48.3 |
| |0>>>     |_fib                                                       | 1094600 |      3 | wall_clock | sec    | 1.217671 | 0.000001 | 0.000000 | 0.000055 | 0.000000 | 0.000002 |   41.3 |
| |0>>>       |_fib                                                     |  418100 |      4 | wall_clock | sec    | 0.714197 | 0.000002 | 0.000000 | 0.000050 | 0.000000 | 0.000002 |   38.1 |
| |0>>>         |_fib                                                   |  258400 |      5 | wall_clock | sec    | 0.441874 | 0.000002 | 0.000000 | 0.000047 | 0.000000 | 0.000002 |   37.9 |
| |0>>>           |_fib                                                 |  159700 |      6 | wall_clock | sec    | 0.274224 | 0.000002 | 0.000000 | 0.000044 | 0.000000 | 0.000002 |   37.9 |
| |0>>>             |_fib                                               |   98700 |      7 | wall_clock | sec    | 0.170399 | 0.000002 | 0.000000 | 0.000042 | 0.000000 | 0.000002 |   37.7 |
| |0>>>               |_fib                                             |   61000 |      8 | wall_clock | sec    | 0.106093 | 0.000002 | 0.000000 | 0.000039 | 0.000000 | 0.000002 |   37.5 |
| |0>>>                 |_fib                                           |   37700 |      9 | wall_clock | sec    | 0.066316 | 0.000002 | 0.000000 | 0.000036 | 0.000000 | 0.000002 |   40.2 |
| |0>>>                   |_fib                                         |   23300 |     10 | wall_clock | sec    | 0.039640 | 0.000002 | 0.000000 | 0.000033 | 0.000000 | 0.000002 |   38.2 |
| |0>>>                     |_fib                                       |   14400 |     11 | wall_clock | sec    | 0.024504 | 0.000002 | 0.000000 | 0.000030 | 0.000000 | 0.000002 |   37.9 |
| |0>>>                       |_fib                                     |    8900 |     12 | wall_clock | sec    | 0.015219 | 0.000002 | 0.000000 | 0.000027 | 0.000000 | 0.000002 |   38.1 |
| |0>>>                         |_fib                                   |    5500 |     13 | wall_clock | sec    | 0.009417 | 0.000002 | 0.000000 | 0.000024 | 0.000000 | 0.000002 |   38.3 |
| |0>>>                           |_fib                                 |    3400 |     14 | wall_clock | sec    | 0.005806 | 0.000002 | 0.000000 | 0.000021 | 0.000000 | 0.000002 |   38.4 |
| |0>>>                             |_fib                               |    2100 |     15 | wall_clock | sec    | 0.003576 | 0.000002 | 0.000000 | 0.000019 | 0.000000 | 0.000002 |   38.4 |
| |0>>>                               |_fib                             |    1300 |     16 | wall_clock | sec    | 0.002201 | 0.000002 | 0.000000 | 0.000016 | 0.000000 | 0.000002 |   40.3 |
| |0>>>                                 |_fib                           |     800 |     17 | wall_clock | sec    | 0.001315 | 0.000002 | 0.000000 | 0.000014 | 0.000000 | 0.000002 |   42.1 |
| |0>>>                                   |_fib                         |     500 |     18 | wall_clock | sec    | 0.000762 | 0.000002 | 0.000000 | 0.000010 | 0.000000 | 0.000001 |   42.1 |
| |0>>>                                     |_fib                       |     300 |     19 | wall_clock | sec    | 0.000441 | 0.000001 | 0.000000 | 0.000008 | 0.000000 | 0.000001 |   47.8 |
| |0>>>                                       |_fib                     |     200 |     20 | wall_clock | sec    | 0.000230 | 0.000001 | 0.000000 | 0.000006 | 0.000000 | 0.000001 |   49.0 |
| |0>>>                                         |_fib                   |     100 |     21 | wall_clock | sec    | 0.000117 | 0.000001 | 0.000001 | 0.000003 | 0.000000 | 0.000000 |   84.5 |
| |0>>>                                           |_fib                 |     100 |     22 | wall_clock | sec    | 0.000018 | 0.000000 | 0.000000 | 0.000001 | 0.000000 | 0.000000 |  100.0 |
| |0>>> std::vector<std::thread, std::allocator<std::thread> >::~vector |       1 |      0 | wall_clock | sec    | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  100.0 |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
```
