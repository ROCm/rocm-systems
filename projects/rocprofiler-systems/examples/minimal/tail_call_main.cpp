// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <unistd.h>

// main() ends with a tail call: compiled with sibling-call optimization it jumps to
// run_workload() instead of calling it and then executing a ret. The profiler must not
// treat that jump as the end of main(). The output goes to stderr, which is unbuffered,
// so its order relative to the profiler's own messages is preserved.
int __attribute__((noinline))
run_workload(int argc)
{
    constexpr auto k_iterations = 5;
    constexpr auto k_sleep_usec = 100000;

    fprintf(stderr, "[tail-call-main] workload begin\n");
    for(int i = 0; i < k_iterations; ++i)
    {
        usleep(k_sleep_usec);
    }
    fprintf(stderr, "[tail-call-main] workload end\n");

    return argc > 1 ? 1 : 0;
}

int
main(int argc, char** /*argv*/)
{
    return run_workload(argc);
}
