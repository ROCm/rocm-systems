
#include <dlfcn.h>
#include <pthread.h>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "common/fwd.hpp"

void
run(const std::string& name, pthread_barrier_t* _barrier)
{
    if(_barrier) pthread_barrier_wait(_barrier);

    if(hsa_init_fn)
    {
        if(_barrier) pthread_barrier_wait(_barrier);
        hsa_init_fn();
    }

    if(hip_init_fn)
    {
        if(_barrier) pthread_barrier_wait(_barrier);
        hip_init_fn();
    }

    if(roctxRangePush_fn)
    {
        if(_barrier) pthread_barrier_wait(_barrier);
        roctxRangePush_fn(name.c_str());
    }

    if(roctxRangePop_fn)
    {
        if(_barrier) pthread_barrier_wait(_barrier);
        roctxRangePop_fn(name.c_str());
    }
}

auto
run_threads(unsigned long n)
{
    resolve_symbols<ROCP_REG_TEST_HIP>();

    auto threads = std::vector<std::thread>{};
    auto names   = std::vector<std::string>{};

    for(unsigned long i = 0; i < n; ++i)
        names.emplace_back(std::string{ "thread-" } + std::to_string(i));

    auto _barrier = pthread_barrier_t{};
    pthread_barrier_init(&_barrier, nullptr, n);

    for(unsigned long i = 0; i < n; ++i)
        threads.emplace_back(run, names.at(i), &_barrier);

    for(auto& itr : threads)
        itr.join();

    pthread_barrier_destroy(&_barrier);

    return n;
}

auto run_n = run_threads(4);

int
main()
{
    return (run_n == 4) ? EXIT_SUCCESS : EXIT_FAILURE;
}
