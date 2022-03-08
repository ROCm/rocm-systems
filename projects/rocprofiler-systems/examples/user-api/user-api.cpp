
#include <omnitrace/user.h>

#include <atomic>
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
    omnitrace_user_stop_thread_trace();
    omnitrace_user_push_region(RUN_LABEL);
    long local = 0;
    for(size_t i = 0; i < nitr; ++i)
        local += fib(n);
    total += local;
    omnitrace_user_pop_region(RUN_LABEL);
    omnitrace_user_start_thread_trace();
}

int
main(int argc, char** argv)
{
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
    for(size_t i = 0; i < nthread; ++i)
    {
        size_t _nitr = ((i % 2) == 1) ? (nitr - (0.1 * nitr)) : (nitr + (0.1 * nitr));
        threads.emplace_back(&run, _nitr, nfib);
    }
    omnitrace_user_pop_region("thread_creation");

    run(nitr - 0.25 * nitr, nfib - 0.1 * nfib);

    omnitrace_user_push_region("thread_wait");
    for(auto& itr : threads)
        itr.join();
    omnitrace_user_pop_region("thread_wait");

    printf("[%s] fibonacci(%li) x %lu = %li\n", argv[0], nfib, nthread, total.load());
    omnitrace_user_pop_region(argv[0]);

    return 0;
}
