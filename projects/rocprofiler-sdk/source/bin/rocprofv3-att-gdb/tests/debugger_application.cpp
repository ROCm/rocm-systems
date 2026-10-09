// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include <csignal>
#include <cstdio>
#include <cstring>

extern "C" __attribute__((noinline)) void
capture_begin()
{
    asm volatile("" ::: "memory");
}
extern "C" __attribute__((noinline)) void
capture_end()
{
    asm volatile("" ::: "memory");
}
extern "C" __attribute__((noinline)) void
user_stop()
{
    asm volatile("" ::: "memory");
}
extern "C" __attribute__((noinline)) void
between_captures()
{
    asm volatile("" ::: "memory");
}

int
main(int argc, char** argv)
{
    std::setbuf(stdout, nullptr);
    for(int i = 0; i < 2; ++i)
    {
        capture_begin();
        if(argc > 1 && !std::strcmp(argv[1], "signal"))
            std::raise(SIGUSR1);
        else
            user_stop();
        std::puts("PASSED_USER_STOP");
        capture_end();
        between_captures();
    }
    std::puts("APPLICATION_DONE");
}
