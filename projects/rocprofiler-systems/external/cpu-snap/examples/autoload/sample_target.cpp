// sample_target.cpp
//
// Single-threaded target for the LD_PRELOAD autoload sampler: main() spins
// through a nested call stack.
//
//   $ cmake --build build --target callstack_autoload sample_target
//   $ LD_PRELOAD=$PWD/build/libcallstack_autoload.so
//         CALLSTACK_TRACE=/tmp/autoload.pftrace
//         ./build/sample_target

#include <chrono>
#include <cstdio>

#include <pthread.h>

#include "workload.hpp"

int
main()
{
    constexpr auto run_for = std::chrono::seconds{ 3 };

    ::pthread_setname_np(::pthread_self(), "main");

    const auto deadline = std::chrono::steady_clock::now() + run_for;
    double     acc      = 0.0;
    while(std::chrono::steady_clock::now() < deadline) {
        acc += cs_examples::run_workload(acc, 200'000);
}
    if(acc == 12345.6789) std::printf("unreachable\n");

    return 0;
}
