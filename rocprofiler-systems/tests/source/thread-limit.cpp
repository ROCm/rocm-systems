
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <random>
#include <ratio>
#include <string>
#include <thread>
#include <vector>

long
fib(long n)
{
    return (n < 2) ? n : fib(n - 1) + fib(n - 2);
}

#if !defined(MAX_THREADS)
#    define MAX_THREADS 4000
#endif

auto total_duration = std::chrono::duration<long, std::nano>{};

int
main(int argc, char** argv)
{
    std::string _name = argv[0];
    auto        _pos  = _name.find_last_of('/');
    if(_pos != std::string::npos) _name = _name.substr(_pos + 1);

    size_t nthread     = 2 * MAX_THREADS;
    size_t concurrency = std::thread::hardware_concurrency();
    long   nfib        = 35;

    if(argc > 1) nfib = atol(argv[1]);
    if(argc > 2) concurrency = atol(argv[2]);
    if(argc > 3) nthread = atol(argv[3]);

    printf("\n[%s] Threads: %zu\n[%s] concurrency: %zu\n[%s] fibonacci(%li)\n",
           _name.c_str(), nthread, _name.c_str(), concurrency, _name.c_str(), nfib);

    auto threads = std::vector<std::thread>{};
    auto _sync   = [_name, &threads]() {
        std::this_thread::yield();
        for(auto& itr : threads)
            itr.join();
        threads.clear();
    };

    threads.reserve(concurrency);
    for(size_t i = 0; i < nthread; ++i)
    {
        if(i > MAX_THREADS - 8)
        {
            printf("[%s] launching thread %zu (max: %d)...\n", _name.c_str(), i,
                   MAX_THREADS);
            fflush(stdout);
        }

        threads.emplace_back(
            [](auto n) {
                auto t0 = std::chrono::steady_clock::now();
                n       = fib(n);
                (void) n;
                auto        diff   = (std::chrono::steady_clock::now() - t0);
                static auto _mutex = std::mutex{};
                _mutex.lock();
                total_duration += diff;
                _mutex.unlock();
            },
            nfib);

        if(i % concurrency == (concurrency - 1)) _sync();
    }

    _sync();

    printf("[%s] ... completed with an average of %.3f msec per thread\n", _name.c_str(),
           std::chrono::duration_cast<std::chrono::milliseconds>(total_duration).count() *
               (1.0 / nthread));

    return 0;
}
