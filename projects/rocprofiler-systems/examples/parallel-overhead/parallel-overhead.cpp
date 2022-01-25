
#include <atomic>
#include <cstdio>
#include <cstdlib>
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

void
run(size_t nitr, long n)
{
    long local = 0;
    for(size_t i = 0; i < nitr; ++i)
        local += fib(n);
    total += local;
}

int
main(int argc, char** argv)
{
    size_t nthread = std::min<size_t>(16, std::thread::hardware_concurrency());
    size_t nitr    = 50000;
    long   nfib    = 10;
    if(argc > 1) nfib = atol(argv[1]);
    if(argc > 2) nthread = atol(argv[2]);
    if(argc > 3) nitr = atol(argv[3]);

    printf("[%s] Threads: %zu\n[%s] Iterations: %zu\n[%s] fibonacci(%li)...\n", argv[0],
           nthread, argv[0], nitr, argv[0], nfib);

    std::vector<std::thread> threads{};
    for(size_t i = 0; i < nthread; ++i)
    {
        size_t _nitr = ((i % 2) == 1) ? (nitr - (0.1 * nitr)) : (nitr + (0.1 * nitr));
        threads.emplace_back(&run, _nitr, nfib);
    }

    run(nitr - 0.25 * nitr, nfib - 0.1 * nfib);
    for(auto& itr : threads)
        itr.join();

    printf("[%s] fibonacci(%li) x %lu = %li\n", argv[0], nfib, nthread, total.load());

    return 0;
}
