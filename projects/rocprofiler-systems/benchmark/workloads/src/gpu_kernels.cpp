// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_workload.hpp"

int
main(int argc, char** argv)
{
    return gpu_workload::run(argc, argv,
                             { "gpu_kernels",
                               "size = float elements per kernel launch, iters = number "
                               "of kernel launches (many short kernels).",
                               { 1 << 16, 65000, 1 },
                               64 });
}
