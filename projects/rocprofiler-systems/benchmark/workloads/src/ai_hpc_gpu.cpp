// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_worker.hpp"

int
main(int argc, char** argv)
{
    return ai_hpc::run<ai_hpc::gpu_worker>(
        argc, argv, "ai_hpc_gpu", { 512, 250, 1, 0, 0, 4 },
        "GPU only. size = matrix dimension N, iters = layers per "
        "stream, depth = nested calls above each layer, threads = "
        "worker threads, streams = streams per thread.");
}
