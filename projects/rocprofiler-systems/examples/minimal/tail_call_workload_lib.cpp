// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <unistd.h>

// The workload lives in a shared library, so the tail call at the end of main() jumps
// through the PLT into another object instead of to a function in the executable.
extern "C" int __attribute__((visibility("default"), noinline))
tail_call_shared_workload(int argc)
{
    constexpr auto k_iterations = 5;
    constexpr auto k_sleep_usec = 100000;

    fprintf(stderr, "[tail-call-shared] workload begin\n");
    for(int i = 0; i < k_iterations; ++i)
    {
        usleep(k_sleep_usec);
    }
    fprintf(stderr, "[tail-call-shared] workload end\n");
    return argc > 1 ? 1 : 0;
}
