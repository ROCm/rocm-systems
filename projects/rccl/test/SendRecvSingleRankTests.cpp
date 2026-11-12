/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ROCM-33800: 1-rank grouped send/recv to self at 256 P2P channels per peer must complete.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <chrono>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "common/TestChecks.hpp"

namespace RcclUnitTesting
{
namespace
{
using EnvMap = std::unordered_map<std::string, std::string>;

// A regression here spins while growing host memory by gigabytes per second, so the timeout stays
// short; each healthy case takes about 2 s on MI350X.
constexpr int    kTimeoutSeconds = 30;
constexpr size_t kCount          = 16;

void groupedSendRecvToSelf()
{
    int dev = 0;
    HIP_CHECK(hipSetDevice(dev));
    ncclComm_t comm;
    ASSERT_EQ(ncclCommInitAll(&comm, 1, &dev), ncclSuccess);
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    float* send = nullptr;
    float* recv = nullptr;
    HIP_CHECK(hipMalloc(&send, kCount * sizeof(float)));
    HIP_CHECK(hipMalloc(&recv, kCount * sizeof(float)));
    std::vector<float> in(kCount), out(kCount, 0.0f);
    std::iota(in.begin(), in.end(), 1.0f);
    HIP_CHECK(hipMemcpy(send, in.data(), kCount * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(recv, 0, kCount * sizeof(float)));

    ASSERT_EQ(ncclGroupStart(), ncclSuccess);
    ASSERT_EQ(ncclSend(send, kCount, ncclFloat32, 0, comm, stream), ncclSuccess);
    ASSERT_EQ(ncclRecv(recv, kCount, ncclFloat32, 0, comm, stream), ncclSuccess);
    ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipMemcpy(out.data(), recv, kCount * sizeof(float), hipMemcpyDeviceToHost));
    EXPECT_EQ(in, out);
    HIP_EXPECT(hipFree(send));
    HIP_EXPECT(hipFree(recv));
    HIP_EXPECT(hipStreamDestroy(stream));
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
