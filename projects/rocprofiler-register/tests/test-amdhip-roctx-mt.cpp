
#include <dlfcn.h>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

#include "common/fwd.hpp"

void
run(const std::string& name, pthread_barrier_t* _barrier)
{
    if(_barrier) pthread_barrier_wait(_barrier);

    if(hip_init_fn)
    {
        if(_barrier) pthread_barrier_wait(_barrier);
        hip_init_fn();
    }

    if(hsa_init_fn)
    {
        if(_barrier) pthread_barrier_wait(_barrier);
        hsa_init_fn();
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

void
run_threads(unsigned long n)
{
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
}

int
main(int argc, char** argv)
{
    unsigned long n = 4;
    if(argc > 1) n = std::stoul(argv[1]);

    resolve_symbols<ROCP_REG_TEST_HIP | ROCP_REG_TEST_ROCTX>();

    run_threads(n);

    return 0;
}
