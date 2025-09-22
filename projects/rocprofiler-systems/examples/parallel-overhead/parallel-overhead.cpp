
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if !defined(USE_LOCKS)
#    define USE_LOCKS 0
#endif

#if USE_LOCKS > 0
#    include <mutex>
using auto_lock_t     = std::unique_lock<std::mutex>;
long       total      = 0;
long       lock_count = 0;
std::mutex mtx{};
#else
std::atomic<long> total{ 0 };
long              lock_count = 0;
#endif

long
fib(long n) __attribute__((noinline));

void
run(size_t nitr, long) __attribute__((noinline));

long
fib(long n)
{
    return (n < 2) ? n : fib(n - 1) + fib(n - 2);
}

void
run(size_t nitr, long n)
{
    static std::atomic<int> _tids{ 0 };
    auto                    _tid = ++_tids;

    std::default_random_engine          eng(std::random_device{}() * (100 + _tid));
    std::uniform_int_distribution<long> distr{ n - 2, n + 2 };

    auto _get_n = [&]() { return distr(eng); };

    printf("[%i] number of iterations: %zu\n", _tid, nitr);

#if USE_LOCKS > 0
    for(size_t i = 0; i < nitr; ++i)
    {
        auto        _v = fib(_get_n());
        auto_lock_t _lk{ mtx };
        total += _v;
        ++lock_count;
    }
#else
    long local = 0;
    for(size_t i = 0; i < nitr; ++i)
    {
        local += fib(_get_n());
    }
    total += local;
#endif
}

int
main(int argc, char** argv)
{
    std::string _name = argv[0];
    auto        _pos  = _name.find_last_of('/');
    if(_pos != std::string::npos) _name = _name.substr(_pos + 1);

    size_t nthread = std::min<size_t>(16, std::thread::hardware_concurrency());
    size_t nitr    = 50000;
    long   nfib    = 10;

    if(argc > 1) nfib = atol(argv[1]);
    if(argc > 2) nthread = atol(argv[2]);
    if(argc > 3) nitr = atol(argv[3]);

    printf("\n[%s] Threads: %zu\n[%s] Iterations: %zu\n[%s] fibonacci(%li)...\n",
           _name.c_str(), nthread, _name.c_str(), nitr, _name.c_str(), nfib);

    bool run_on_main_thread = (USE_LOCKS == 0);
    auto nwait              = nthread + ((run_on_main_thread) ? 1 : 0);

    pthread_barrier_t _barrier;
    pthread_barrier_init(&_barrier, nullptr, nwait);

    auto _run = [&_barrier](size_t nitr, long n) {
        pthread_barrier_wait(&_barrier);
        run(nitr, n);
    };

    std::vector<std::thread> threads{};
    for(size_t i = 0; i < nthread; ++i)
    {
        threads.emplace_back(_run, nitr, nfib);
    }

    if(run_on_main_thread)
    {
        _run(nitr, nfib);
    }

    for(auto& itr : threads)
        itr.join();

    pthread_barrier_destroy(&_barrier);

    printf("[%s] fibonacci(%li) x %lu = %li\n", _name.c_str(), nfib, nthread,
           static_cast<long>(total));
    printf("[%s] number of mutex locks = %li\n", _name.c_str(), lock_count);

    return 0;
}
