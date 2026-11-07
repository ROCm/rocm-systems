// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "ai_hpc.hpp"
#include <cstdint>

#include <algorithm>
#include <cstring>

namespace ai_hpc
{
class cpu_worker
{
public:
    cpu_worker(const workload::args& a, std::uint64_t seed)
    : n_(a.size)
    , act_(n_ * n_)
    , next_(n_ * n_)
    , weights_(n_ * n_)
    {
        init_matrices(act_, weights_, n_, seed);
    }

    void layer()
    {
        // i-k-j order: the inner loop streams rows of W and next_, so it vectorizes.
        std::fill(next_.begin(), next_.end(), 0.0f);
        for(std::uint64_t i = 0; i < n_; ++i)
            for(std::uint64_t k = 0; k < n_; ++k)
            {
                const float a = act_[i * n_ + k];
                for(std::uint64_t j = 0; j < n_; ++j)
                    next_[i * n_ + j] += a * weights_[k * n_ + j];
            }
        for(auto& x : next_)
            x = std::tanh(x);
        act_.swap(next_);
    }

    std::uint64_t checksum() const
    {
        std::uint64_t sum = 0;
        for(float x : act_)
        {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &x, sizeof(bits));
            sum = workload::mix(sum, bits);
        }
        return sum;
    }

private:
    std::uint64_t      n_;
    std::vector<float> act_, next_, weights_;
};
}  // namespace ai_hpc
