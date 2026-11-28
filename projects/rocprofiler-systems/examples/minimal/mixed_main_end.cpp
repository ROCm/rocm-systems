// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

// main() has several exit points of different kinds:
//  - with an argument, it ends with a tail call, which the compiler turns into a jmp
//  - without an argument, it prints a message and ends with a ret
//  - with two arguments, it ends with a call to exit()
// The profiler must skip the tail call and keep the other exit points. Whichever way
// main() ends at run time, the session has to be finalized after the workload.
// The output goes to stderr, which is unbuffered, so its order relative to the
// profiler's own messages is preserved.

// Exit status of the workload. It is a global the compiler cannot fold, so that every
// path through main() still ends with a real ret, jmp or call.
int g_status = 0;

int __attribute__((noinline))
run_workload()
{
    constexpr auto k_iterations = 5;
    constexpr auto k_sleep_usec = 100000;

    fprintf(stderr, "[mixed-main-end] workload begin\n");
    for(int i = 0; i < k_iterations; ++i)
    {
        usleep(k_sleep_usec);
    }
    fprintf(stderr, "[mixed-main-end] workload end\n");
    return g_status;
}

int
main(int argc, char** /*argv*/)
{
    if(argc == 2) return run_workload();
    if(argc > 2)
    {
        run_workload();
        exit(0);
    }
    fprintf(stderr, "[mixed-main-end] returning from main\n");
    return run_workload() != 0 ? 1 : 0;
}
