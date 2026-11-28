// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

// main() ends with a call to exit() instead of a ret. That call does not return and is
// not a tail call, so the profiler must keep treating the end of main() as an exit
// point. The output goes to stderr, which is unbuffered, so its order relative to the
// profiler's own messages is preserved.
void __attribute__((noinline))
run_workload()
{
    constexpr auto k_iterations = 5;
    constexpr auto k_sleep_usec = 100000;

    fprintf(stderr, "[exit-in-main] workload begin\n");
    for(int i = 0; i < k_iterations; ++i)
    {
        usleep(k_sleep_usec);
    }
    fprintf(stderr, "[exit-in-main] workload end\n");
}

int
main()
{
    run_workload();
    exit(0);
}
