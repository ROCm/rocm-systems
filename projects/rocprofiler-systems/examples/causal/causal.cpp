#include "causal.hpp"
#include "impl.hpp"

namespace
{
std::chrono::duration<double, std::milli> t_ms;
std::chrono::duration<double, std::milli> slow_ms;
std::chrono::duration<double, std::milli> fast_ms;
}  // namespace

int
main(int argc, char** argv)
{
    uint64_t rseed    = std::random_device{}();
    size_t   nitr     = 50;
    double   frac     = 70;
    int64_t  slow_val = 200000000L;
    size_t   nsync    = 1;

    if(argc > 1) frac = std::stod(argv[1]);
    if(argc > 2) nitr = std::stoull(argv[2]);
    if(argc > 3) rseed = std::stoul(argv[3]);
    if(argc > 4) slow_val = std::stol(argv[4]);
    if(argc > 5) nsync = std::stoull(argv[5]);

    nsync            = std::min<size_t>(std::max<size_t>(nsync, 1), nitr);
    int64_t fast_val = (frac / 100.0) * slow_val;
    double  rfrac    = (fast_val / static_cast<double>(slow_val));
    if(argc > 5) fast_val = std::stol(argv[5]);

    printf("\nFraction: %6.2f, iterations: %zu, random seed: %lu :: slow = %zu, "
           "fast = %zu, expected ratio = %6.2f, sync every %lu iterations\n",
           frac, nitr, rseed, slow_val, fast_val, rfrac * 100.0, nsync);

    auto _wait_barrier = pthread_barrier_t{};
    pthread_barrier_init(&_wait_barrier, nullptr, 3);
    auto _thread_func = [nitr, nsync, &_wait_barrier](const auto& _func, auto* _timer,
                                                      auto _nsec, auto _nseed,
                                                      auto _nloop) {
        pthread_barrier_wait(&_wait_barrier);
        for(size_t i = 0; i < nitr; ++i)
        {
            auto _t = clock_type::now();
            _func(_nsec, _nseed, _nloop);
            (*_timer) += (clock_type::now() - _t);
            CAUSAL_PROGRESS_NAMED("iteration");
            if(i % nsync == (nsync - 1)) pthread_barrier_wait(&_wait_barrier);
        }
    };

    auto _t       = clock_type::now();
    auto _threads = std::vector<std::thread>{};
    _threads.emplace_back(_thread_func, SLOW_FUNC, &slow_ms, slow_val, rseed, 10000);
    _threads.emplace_back(_thread_func, FAST_FUNC, &fast_ms, fast_val, rseed, 10000);
    pthread_barrier_wait(&_wait_barrier);
    for(size_t i = 0; i < nitr; ++i)
    {
        if(i == 0 || i + 1 == nitr || i % (nitr / 5) == 0)
            (printf("executing iteration: %zu\n", i), fflush(stdout));
        if(i % nsync == (nsync - 1)) pthread_barrier_wait(&_wait_barrier);
    }
    for(auto& itr : _threads)
        itr.join();

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
    while(rng_impl_func<false>(n, rseed) != false) {}
    // clang-format on
}
//
//
//
void
rng_fast_func(int64_t n, uint64_t rseed)
{
    // clang-format off
    while(rng_impl_func<true>(n, rseed) != true) {}
    // clang-format on
}
//
//
//
void
cpu_slow_func(int64_t n, int nloop)
{
    // clang-format off
    while(cpu_impl_func<false>(n, nloop) != false) {}
    // clang-format on
}
//
//
//
void
cpu_fast_func(int64_t n, int nloop)
{
    // clang-format off
    while(cpu_impl_func<true>(n, nloop) != true) {}
    // clang-format on
}
