// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#include "tree_ipc.hpp"
int
main(int argc, char** argv)
{
    require(argc == 4);
    auto child = launch(argv + 1, std::strstr(argv[3], "spawn"));
    std::printf("ROOT_PID=%d CHILD_PID=%d\n", getpid(), child);
    std::fflush(stdout);
    if(std::strstr(argv[3], "orphan")) return 0;
    return wait_child(child);
}
