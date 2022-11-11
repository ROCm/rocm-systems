
#include "omnitrace/categories.h"
#include "omnitrace/types.h"
#include <omnitrace/user.h>

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
omnitrace_user_callbacks_t custom_callbacks   = OMNITRACE_USER_CALLBACKS_INIT;
omnitrace_user_callbacks_t original_callbacks = OMNITRACE_USER_CALLBACKS_INIT;
}  // namespace

int
main(int argc, char** argv)
{
    custom_callbacks.push_region = &custom_push_region;
    omnitrace_user_configure(OMNITRACE_USER_UNION_CONFIG, custom_callbacks,
                             &original_callbacks);

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
        threads.emplace_back(&run, nitr, nfib);
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
    if(!original_callbacks.push_region || !original_callbacks.push_annotated_region)
        return OMNITRACE_USER_ERROR_NO_BINDING;

    printf("Pushing custom region :: %s\n", name);

    if(original_callbacks.push_annotated_region)
    {
        int32_t _err = errno;
        char*   _msg = nullptr;
        char    _buff[1024];
        if(_err != 0) _msg = strerror_r(_err, _buff, sizeof(_buff));

        omnitrace_annotation_t _annotations[] = {
            { "errno", OMNITRACE_INT32, &_err }, { "strerror", OMNITRACE_STRING, _msg }
        };

        errno = 0;  // reset errno
        return (*original_callbacks.push_annotated_region)(
            name, _annotations, sizeof(_annotations) / sizeof(omnitrace_annotation_t));
    }

    return (*original_callbacks.push_region)(name);
}
