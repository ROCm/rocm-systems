#include "causal.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using mutex_t     = std::timed_mutex;
using auto_lock_t = std::unique_lock<mutex_t>;
using clock_type  = std::chrono::high_resolution_clock;
using nanosec     = std::chrono::nanoseconds;

namespace
{
std::chrono::duration<double, std::milli> t_ms;
std::chrono::duration<double, std::milli> slow_ms;
std::chrono::duration<double, std::milli> fast_ms;

template <typename... Args>
inline void
consume_variables(Args&&...)
{}
}  // namespace

template <bool>
bool
rng_func_impl(int64_t n, uint64_t rseed);

template <bool>
bool
cpu_func_impl(int64_t n, int nloop);

void
rng_slow_func(int64_t n, uint64_t rseed) __attribute__((noinline));

void
rng_fast_func(int64_t n, uint64_t rseed) __attribute__((noinline));

void
cpu_slow_func(int64_t n, int nloop) __attribute__((noinline));

void
cpu_fast_func(int64_t n, int nloop) __attribute__((noinline));

#if USE_CPU > 0
#    define CPU_SLOW_FUNC(...) cpu_slow_func(__VA_ARGS__)
#    define CPU_FAST_FUNC(...) cpu_fast_func(__VA_ARGS__)
#else
#    define CPU_SLOW_FUNC(...) consume_variables(__VA_ARGS__)
#    define CPU_FAST_FUNC(...) consume_variables(__VA_ARGS__)
#endif

#if USE_RNG > 0
#    define RNG_SLOW_FUNC(...) rng_slow_func(__VA_ARGS__)
#    define RNG_FAST_FUNC(...) rng_fast_func(__VA_ARGS__)
#else
#    define RNG_SLOW_FUNC(...) consume_variables(__VA_ARGS__)
#    define RNG_FAST_FUNC(...) consume_variables(__VA_ARGS__)
#endif

int
main(int argc, char** argv)
{
    uint64_t rseed    = std::random_device{}();
    int      nitr     = 200;
    double   frac     = 70;
    int64_t  slow_val = 100000000L;

    if(argc > 1) frac = std::stod(argv[1]);
    if(argc > 2) nitr = std::stoi(argv[2]);
    if(argc > 3) rseed = std::stoul(argv[3]);
    if(argc > 4) slow_val = std::stol(argv[4]);

    int64_t fast_val = (frac / 100.0) * slow_val;
    double  rfrac    = (fast_val / static_cast<double>(slow_val));
    if(argc > 5) fast_val = std::stol(argv[5]);

    printf("\nIterations: %i, fraction: %6.2f, random seed: %lu :: slow = %zu, "
           "fast = %zu, expected ratio = %6.2f\n",
           nitr, frac, rseed, slow_val, fast_val, rfrac * 100.0);

    auto _t = clock_type::now();
    for(int i = 0; i < nitr; ++i)
    {
        if(i == 0 || i + 1 == nitr || i % (nitr / 5) == 0)
            printf("executing iteration: %i\n", i);
        //
        auto&& _slow_func = [](auto _nsec, auto _seed, auto _nloop) {
            auto _t = clock_type::now();
            CPU_SLOW_FUNC(_nsec, _nloop);
            RNG_SLOW_FUNC(_nsec / 5, _seed);
            slow_ms += (clock_type::now() - _t);
        };
        //
        auto&& _fast_func = [](auto _nsec, auto _seed, auto _nloop) {
            auto _t = clock_type::now();
            CPU_FAST_FUNC(_nsec, _nloop);
            RNG_FAST_FUNC(_nsec / 5, _seed);
            fast_ms += (clock_type::now() - _t);
        };
        //
        CAUSAL_BEGIN("main_iteration");
        //
        auto _threads = std::vector<std::thread>{};
        _threads.emplace_back(std::move(_slow_func), slow_val, rseed, 10000);
        _threads.emplace_back(std::move(_fast_func), fast_val, rseed, 10000);
        for(auto& itr : _threads)
            itr.join();
        CAUSAL_END("main_iteration");
        CAUSAL_PROGRESS;
    }
    t_ms += clock_type::now() - _t;
    auto rms = (fast_ms.count() / slow_ms.count());
    printf("slow_func() took %10.3f ms\n", slow_ms.count());
    printf("fast_func() took %10.3f ms\n", fast_ms.count());
    printf("total is %18.3f ms\n", t_ms.count());
    printf("ratio is %18.3f %s\n", 100.0 * rms, "%");
    printf("rdiff is %18.3f %s\n", 100.0 * (rms - rfrac), "%");
}
//
//
//
void
rng_slow_func(int64_t n, uint64_t rseed)
{
    // clang-format off
    while(rng_func_impl<false>(n, rseed) != false) {}
    // clang-format on
}
//
//
//
void
rng_fast_func(int64_t n, uint64_t rseed)
{
    // clang-format off
    while(rng_func_impl<true>(n, rseed) != true) {}
    // clang-format on
}
//
//
//
void
cpu_slow_func(int64_t n, int nloop)
{
    // clang-format off
    while(cpu_func_impl<false>(n, nloop) != false) {}
    // clang-format on
}
//
//
//
void
cpu_fast_func(int64_t n, int nloop)
{
    // clang-format off
    while(cpu_func_impl<true>(n, nloop) != true) {}
    // clang-format on
}
