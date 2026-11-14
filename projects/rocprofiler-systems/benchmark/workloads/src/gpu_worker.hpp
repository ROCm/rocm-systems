// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "ai_hpc.hpp"
#include "gpu_workload.hpp"  // HIP helpers: HIP_CHECK, device_buffer
#include <cstdint>

namespace ai_hpc
{
using gpu_workload::check;  // HIP_CHECK expands to an unqualified check()

constexpr int tile = 16;

// Shared-memory tiled GEMM with the tanh fused in: out = tanh(in x w), all n x n.
__global__ void
mlp_layer_kernel(const float* in, const float* w, float* out, std::uint64_t n)
{
    __shared__ float    a_tile[tile][tile];
    __shared__ float    w_tile[tile][tile];
    const std::uint64_t row = static_cast<std::uint64_t>(blockIdx.y) * tile + threadIdx.y;
    const std::uint64_t col = static_cast<std::uint64_t>(blockIdx.x) * tile + threadIdx.x;
    float               acc = 0.0f;
    for(std::uint64_t t = 0; t < n; t += tile)
    {
        a_tile[threadIdx.y][threadIdx.x] =
            (row < n && t + threadIdx.x < n) ? in[row * n + t + threadIdx.x] : 0.0f;
        w_tile[threadIdx.y][threadIdx.x] =
            (col < n && t + threadIdx.y < n) ? w[(t + threadIdx.y) * n + col] : 0.0f;
        __syncthreads();
        for(int k = 0; k < tile; ++k)
            acc += a_tile[threadIdx.y][k] * w_tile[k][threadIdx.x];
        __syncthreads();
    }
    if(row < n && col < n) out[row * n + col] = tanhf(acc);
}

// One model (weights shared) served on `streams` independent streams, each with its own
// activations, so the kernels of one layer run concurrently on the device.
class gpu_worker
{
public:
    gpu_worker(std::uint64_t n, std::uint64_t streams, std::uint64_t seed)
    : n_(n)
    , weights_(n * n * sizeof(float))
    {
        std::vector<float> act(n * n), weights(n * n);
        init_matrices(act, weights, n, seed);
        HIP_CHECK(hipMemcpy(weights_.ptr, weights.data(), weights.size() * sizeof(float),
                            hipMemcpyHostToDevice));
        for(std::uint64_t s = 0; s < streams; ++s)
        {
            init_matrices(act, weights, n,
                          seed + (s << 32));  // distinct activations per stream
            lanes_.push_back(std::make_unique<lane>(n * n * sizeof(float)));
            HIP_CHECK(hipMemcpy(lanes_.back()->act.ptr, act.data(),
                                act.size() * sizeof(float), hipMemcpyHostToDevice));
        }
        layer();  // warm-up: loads the kernel module outside the measured region
    }
    gpu_worker(const workload::args& a, std::uint64_t seed)
    : gpu_worker(a.size, a.streams, seed)
    {}

    void launch()
    {
        const unsigned blocks = static_cast<unsigned>((n_ + tile - 1) / tile);
        for(auto& l : lanes_)
        {
            hipLaunchKernelGGL(mlp_layer_kernel, dim3(blocks, blocks), dim3(tile, tile),
                               0, l->stream, l->act.ptr, weights_.ptr, l->next.ptr, n_);
            std::swap(l->act.ptr, l->next.ptr);
        }
    }

    void wait()
    {
        for(auto& l : lanes_)
            HIP_CHECK(hipStreamSynchronize(l->stream));
    }

    // Synchronizes at the leaf of the deep call chain, like a framework waiting on a
    // result.
    void layer()
    {
        launch();
        wait();
    }

    std::uint64_t checksum() const
    {
        std::uint64_t      sum = 0;
        std::vector<float> host(n_ * n_);
        for(auto& l : lanes_)
        {
            HIP_CHECK(hipMemcpy(host.data(), l->act.ptr, host.size() * sizeof(float),
                                hipMemcpyDeviceToHost));
            sum = workload::mix(sum, gpu_workload::checksum_of(host));
        }
        return sum;
    }

private:
    struct lane
    {
        gpu_workload::device_buffer act, next;
        hipStream_t                 stream{};

        explicit lane(size_t bytes)
        : act(bytes)
        , next(bytes)
        {
            HIP_CHECK(hipStreamCreate(&stream));
        }
        ~lane() { (void) hipStreamDestroy(stream); }
        lane(const lane&)            = delete;
        lane& operator=(const lane&) = delete;
    };

    std::uint64_t                      n_;
    gpu_workload::device_buffer        weights_;
    std::vector<std::unique_ptr<lane>> lanes_;
};
}  // namespace ai_hpc
