/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ROCM-33800: a grouped send+recv to self on a 1-rank comm must return from ncclGroupEnd and
// copy the data with 256 P2P channels per peer. That is the default on single-node gfx1250;
// the env of the second case forces it on any arch.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <chrono>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{
namespace
{
using EnvMap = std::unordered_map<std::string, std::string>;

// A regression here spins while growing host memory by gigabytes per second; keep the timeout short.
constexpr int    kTimeoutSeconds = 60;
constexpr size_t kCount          = 16;

void groupedSendRecvToSelf()
{
    int dev = 0;
    ASSERT_EQ(hipSetDevice(dev), hipSuccess);
    ncclComm_t comm;
    ASSERT_EQ(ncclCommInitAll(&comm, 1, &dev), ncclSuccess);
    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    float* send = nullptr;
    float* recv = nullptr;
    ASSERT_EQ(hipMalloc(&send, kCount * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&recv, kCount * sizeof(float)), hipSuccess);
    std::vector<float> in(kCount), out(kCount, 0.0f);
    std::iota(in.begin(), in.end(), 1.0f);
    ASSERT_EQ(hipMemcpy(send, in.data(), kCount * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(recv, 0, kCount * sizeof(float)), hipSuccess);

    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    ASSERT_EQ(ncclSend(send, kCount, ncclFloat32, 0, comm, stream), ncclSuccess);
    ASSERT_EQ(ncclRecv(recv, kCount, ncclFloat32, 0, comm, stream), ncclSuccess);
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    ASSERT_EQ(hipMemcpy(out.data(), recv, kCount * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(in, out);
    EXPECT_EQ(hipFree(send), hipSuccess);
    EXPECT_EQ(hipFree(recv), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

ProcessIsolatedTestRunner::TestConfig isolated(const std::string& name, const EnvMap& env)
{
    EnvMap full = {{"NCCL_DEBUG", "WARN"}};
    full.insert(env.begin(), env.end());
    return ProcessIsolatedTestRunner::TestConfig(name, groupedSendRecvToSelf)
        .withNumGpus(1)
        .withEnvironment(full)
        .withTimeout(std::chrono::seconds(kTimeoutSeconds));
}
} // namespace

TEST(SendRecv, SingleRankGroupedSendRecvToSelf)
{
    RUN_ISOLATED_TESTS(isolated("DefaultEnv", {}),
                       isolated("Forced256ChannelsPerPeer",
                                {{"RCCL_SATURATE_P2P_NCHANNELS", "1"},
                                 {"NCCL_MAX_P2P_NCHANNELS", "256"},
                                 {"NCCL_MIN_P2P_NCHANNELS", "256"}}));
}
} // namespace RcclUnitTesting
