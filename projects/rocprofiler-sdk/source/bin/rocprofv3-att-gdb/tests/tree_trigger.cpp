// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include <string>
#include "tree_ipc.hpp"

extern "C" __attribute__((noinline)) void
tree_begin()
{
    asm volatile("" ::: "memory");
}
extern "C" __attribute__((noinline)) void
tree_end()
{
    asm volatile("" ::: "memory");
}
extern "C" __attribute__((noinline)) void
tree_between()
{
    asm volatile("" ::: "memory");
}

int
main(int argc, char** argv)
{
    require(argc == 3);
    int command[2], notification[2];
    require(pipe(command) == 0 && pipe(notification) == 0);
    auto  input = std::to_string(command[0]), output = std::to_string(notification[1]);
    char* args[] = {argv[1], input.data(), output.data(), nullptr};
    auto  child  = launch(args, std::strstr(argv[2], "spawn"));
    close(command[0]);
    close(notification[1]);
    require(get(notification[0]) == 'R');
    std::printf("TRIGGER_PID=%d WORKER_PID=%d\n", getpid(), child);
    std::fflush(stdout);
    int iterations = std::strstr(argv[2], "repeat") ? 2 : 1;
    for(int i = 0; i < iterations; ++i)
    {
        tree_begin();
        put(command[1], 'W');
        require(get(notification[0]) == 'D');
        if(std::strstr(argv[2], "timeout")) usleep(150000);
        tree_end();
        tree_between();
    }
    put(command[1], 'Q');
    return wait_child(child);
}
