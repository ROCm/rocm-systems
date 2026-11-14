// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_workload.hpp"

int
main(int argc, char** argv)
{
    return gpu_workload::run(argc, argv,
                             { "gpu_long_kernel",
                               "size = float elements per kernel launch, iters = number "
                               "of kernel launches (few long kernels).",
                               { 1 << 20, 40, 1 },
                               300000 });
}
