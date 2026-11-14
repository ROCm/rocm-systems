// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cpu_worker.hpp"

int
main(int argc, char** argv)
{
    return ai_hpc::run<ai_hpc::cpu_worker>(argc, argv, "ai_hpc_cpu", { 256, 250, 1 });
}
