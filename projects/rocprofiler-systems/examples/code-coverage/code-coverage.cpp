
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#define NOINLINE __attribute__((noinline))

std::atomic<long> total{ 0 };

long
fib(long n) NOINLINE;

void
run_real(size_t nitr, long) NOINLINE;

void
run_fake(size_t nitr, long) NOINLINE;

int
main(int argc, char** argv)
{
    using exec_t = void (*)(size_t, long);

    std::string _name = argv[0];
    auto        _pos  = _name.find_last_of('/');
    if(_pos != std::string::npos) _name = _name.substr(_pos + 1);

    size_t nthread = std::min<size_t>(16, std::thread::hardware_concurrency());
    size_t nitr    = 5000;
    long   nfib    = 10;

    if(argc > 1) nfib = atol(argv[1]);
    if(argc > 2) nthread = atol(argv[2]);
    if(argc > 3) nitr = atol(argv[3]);

    exec_t _exec = &run_real;

    // ensure that compiler cannot optimize run_fake away
    if(std::getenv("CODE_COVERAGE_USE_FAKE") != nullptr) _exec = &run_fake;

    printf("[%s] Threads: %zu\n[%s] Iterations: %zu\n[%s] fibonacci(%li)...\n",
           _name.c_str(), nthread, _name.c_str(), nitr, _name.c_str(), nfib);

    std::vector<std::thread> threads{};
    for(size_t i = 0; i < nthread; ++i)
    {
        size_t _nitr = ((i % 2) == 1) ? (nitr - (0.1 * nitr)) : (nitr + (0.1 * nitr));
        _nitr        = std::max<size_t>(_nitr, 1);
        threads.emplace_back(_exec, _nitr, nfib);
    }

    auto _nitr = std::max<size_t>(nitr - 0.25 * nitr, 1);
    (*_exec)(_nitr, nfib - 0.1 * nfib);
    for(auto& itr : threads)
        itr.join();

    printf("[%s] fibonacci(%li) x %lu = %li\n", _name.c_str(), nfib, nthread,
           total.load());

    return 0;
}

long
fib(long n)
{
    return (n < 2) ? n : fib(n - 1) + fib(n - 2);
}

void
run_real(size_t nitr, long n)
{
    long local = 0;
    for(size_t i = 0; i < nitr; ++i)
        local += fib(n);
    total += local;
}

void
run_fake(size_t nitr, long n)
{
    long local = 0;
    for(size_t i = 0; i < nitr; ++i)
        local += fib(n);
    total += local;
}
