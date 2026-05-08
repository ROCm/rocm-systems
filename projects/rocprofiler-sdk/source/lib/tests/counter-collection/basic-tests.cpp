// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "counters-tool.hpp"
#include "kernels.hip"

#include <hip/hip_runtime.h>

#include <fmt/core.h>
#include <gtest/gtest.h>

#include <thread>
#include <vector>

#define HIP_CALL(call)                                                                             \
    {                                                                                              \
        hipError_t err = call;                                                                     \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            FAIL() << #call << " failed: " << hipGetErrorString(err);                              \
        }                                                                                          \
    }

using namespace std::chrono_literals;

namespace
{
const tool::counter_results_t&
get_values(const tool::collection_results_t& results,
           const std::string&                kernel_substr,
           const std::string&                counter_name,
           size_t                            agent_idx    = 0,
           size_t                            dispatch_idx = 0)
{
    size_t ai = 0;
    for(const auto& [agent_id, agent] : results)
    {
        bool agent_has_kernel = false;
        for(const auto& [kernel_id, dispatches] : agent)
        {
            if(tool::kernel_name(kernel_id).find(kernel_substr) == std::string::npos) continue;
            agent_has_kernel = true;

            if(ai != agent_idx) break;

            size_t di = 0;
            for(const auto& [dispatch_id, counters] : dispatches)
            {
                auto it = counters.find(counter_name);
                if(it == counters.end()) continue;
                if(di++ == dispatch_idx) return it->second;
            }
        }
        if(agent_has_kernel) ++ai;
    }

    throw std::runtime_error(
        fmt::format("get_values: no match for kernel='{}' counter='{}' agent={} dispatch={}",
                    kernel_substr,
                    counter_name,
                    agent_idx,
                    dispatch_idx));
}

size_t
count_dispatches(const tool::collection_results_t& results,
                 const std::string&                kernel_substr,
                 const std::string&                counter_name)
{
    size_t count = 0;
    for(const auto& [agent_id, agent] : results)
        for(const auto& [kernel_id, dispatches] : agent)
        {
            if(tool::kernel_name(kernel_id).find(kernel_substr) == std::string::npos) continue;
            for(const auto& [dispatch_id, counters] : dispatches)
                if(counters.count(counter_name) > 0) ++count;
        }
    return count;
}
}  // namespace

class aql_counter_test : public ::testing::Test
{
protected:
    void SetUp() override { HIP_CALL(hipSetDevice(0)); }
};

TEST_F(aql_counter_test, copy)
{
    constexpr int data_size = 64 * 1024;
    constexpr int nblocks   = 256;
    constexpr int nthreads  = (data_size + nblocks - 1) / nblocks;

    float *d_a = nullptr, *d_b = nullptr;
    HIP_CALL(hipMalloc(&d_a, data_size * sizeof(float)));
    HIP_CALL(hipMalloc(&d_b, data_size * sizeof(float)));

    std::vector<float> h_b(data_size, 1.0f);
    HIP_CALL(hipMemcpy(d_b, h_b.data(), data_size * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipDeviceSynchronize());

    auto copy_res1 = tool::collect({"SQ_WAVES"}, [&] {
        kernels::copy<<<dim3(nblocks), dim3(nthreads)>>>(d_a, d_b, data_size);
        kernels::copy<<<dim3(nblocks), dim3(nthreads)>>>(d_a, d_b, data_size);
    });

    auto copy_res2 = tool::collect({"SQ_WAVES"}, [&] {
        kernels::copy<<<dim3(nblocks), dim3(nthreads)>>>(d_a, d_b, data_size);
        kernels::iops2<<<dim3(nblocks), dim3(nthreads)>>>();
    });

    {
        auto res1 = copy_res1.get();
        ASSERT_EQ(count_dispatches(res1, "copy", "SQ_WAVES"), 2u)
            << "res1 should have 2 copy dispatches";

        const auto& d0 = get_values(res1, "copy", "SQ_WAVES", 0, 0);
        const auto& d1 = get_values(res1, "copy", "SQ_WAVES", 0, 1);
        EXPECT_GE(d0.sum(), 0.0) << "dispatch 0 SQ_WAVES should be >= 0";
        EXPECT_GE(d1.sum(), 0.0) << "dispatch 1 SQ_WAVES should be >= 0";
        fmt::println(stderr, "[res1] copy dispatch 0: sum={} count={}", d0.sum(), d0.size());
        fmt::println(stderr, "[res1] copy dispatch 1: sum={} count={}", d1.sum(), d1.size());
    }

    {
        auto res2 = copy_res2.get();
        ASSERT_EQ(count_dispatches(res2, "copy", "SQ_WAVES"), 1u)
            << "res2 should have 1 copy dispatch";
        ASSERT_EQ(count_dispatches(res2, "iops2", "SQ_WAVES"), 1u)
            << "res2 should have 1 iops2 dispatch";

        const auto& sq_copy = get_values(res2, "copy", "SQ_WAVES");
        EXPECT_GE(sq_copy.sum(), 0.0) << "copy SQ_WAVES should be >= 0";
        fmt::println(
            stderr, "[res2] copy SQ_WAVES: sum={} count={}", sq_copy.sum(), sq_copy.size());

        const auto& sq_iops = get_values(res2, "iops2", "SQ_WAVES");
        EXPECT_GT(sq_iops.sum(), 0.0) << "iops2 SQ_WAVES should be > 0";
        fmt::println(
            stderr, "[res2] iops2 SQ_WAVES: sum={} count={}", sq_iops.sum(), sq_iops.size());
    }

    HIP_CALL(hipDeviceSynchronize());

    HIP_CALL(hipFree(d_a));
    HIP_CALL(hipFree(d_b));
}
