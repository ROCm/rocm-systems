// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cpu_worker.hpp"
#include "gpu_worker.hpp"
#include <cstdint>

namespace
{
// One worker thread drives both devices: the GPU layers are launched asynchronously, the
// CPU layer runs while the GPU computes, and the thread then waits for the GPU at the
// leaf.
class hybrid_worker
{
public:
    hybrid_worker(const workload::args& a, std::uint64_t seed)
    : cpu_(a, seed)
    , gpu_(a.gpu_size, a.streams, seed)
    {}

    void layer()
    {
        gpu_.launch();
        cpu_.layer();
        gpu_.wait();
    }

    std::uint64_t checksum() const
    {
        return workload::mix(cpu_.checksum(), gpu_.checksum());
    }

private:
    ai_hpc::cpu_worker cpu_;
    ai_hpc::gpu_worker gpu_;
};
}  // namespace

int
main(int argc, char** argv)
{
    return ai_hpc::run<hybrid_worker>(
        argc, argv, "ai_hpc_hybrid", { 256, 250, 1, 0, 0, 4, 1024 },
        "CPU and GPU compute at the same time in every thread. size = CPU matrix "
        "dimension N, "
        "gpu-size = GPU matrix dimension, iters = layers per thread, depth = nested "
        "calls above "
        "each layer, threads = worker threads, streams = GPU streams per thread.");
}
