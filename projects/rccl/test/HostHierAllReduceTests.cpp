/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/ProcessIsolatedTestRunner.hpp"

#include "comm.h"
#include "hier_host.h"
#include "nccl.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace RcclUnitTesting
{

namespace
{
constexpr size_t kMiB = size_t{1} << 20;

// A comm whose ranks sit on the hosts listed by index: hostOf[r] is rank r's host.
struct HostHierMockComm
{
    // ncclComm is too large to keep several on the stack.
    std::unique_ptr<ncclComm> storage = std::make_unique<ncclComm>();
    ncclComm&                 comm    = *storage;
    std::vector<ncclPeerInfo> peers;
    char                      archBuf[16]{};

    HostHierMockComm(const std::vector<int>& hostOf, int rank, const char* arch = "gfx1250")
    {
        std::memset(&comm, 0, sizeof(comm));
        std::strncpy(archBuf, arch, sizeof(archBuf) - 1);
        comm.archName = archBuf;
        comm.nRanks   = static_cast<int>(hostOf.size());
        comm.rank     = rank;
        setHosts(hostOf);
        comm.hostHierState = rcclHostHierUnprobed;
    }

    HostHierMockComm(const HostHierMockComm&)            = delete;
    HostHierMockComm& operator=(const HostHierMockComm&) = delete;

    ~HostHierMockComm() { rcclHostHierTeardown(&comm); }

    void setHosts(const std::vector<int>& hostOf)
    {
        peers.assign(hostOf.size(), ncclPeerInfo{});
        for(size_t r = 0; r < hostOf.size(); r++)
            peers[r].hostHash = 0x5eed0000u + static_cast<uint64_t>(hostOf[r]);
        comm.peerInfo      = peers.data();
        comm.peerInfoValid = true;
    }

    ncclComm* get() { return &comm; }
};

// nHosts hosts of ranksPerHost consecutive ranks each.
std::vector<int> blockHosts(int nHosts, int ranksPerHost)
{
    std::vector<int> hostOf;
    for(int h = 0; h < nHosts; h++)
        hostOf.insert(hostOf.end(), ranksPerHost, h);
    return hostOf;
}

size_t floats(size_t bytes) { return bytes / sizeof(float); }
} // namespace

TEST(HostHierLayout, ContiguousHosts)
{
    HostHierMockComm mock(blockHosts(2, 8), /*rank=*/9);
    rcclHostLayout   layout{};
    ASSERT_TRUE(rcclHostLayoutGet(mock.get(), &layout));
    EXPECT_EQ(2, layout.nHosts);
    EXPECT_EQ(8, layout.ranksPerHost);
    EXPECT_EQ(1, layout.hostIndex);
    EXPECT_EQ(1, layout.localRank);
}

TEST(HostHierLayout, InterleavedHosts)
{
    HostHierMockComm mock({0, 1, 0, 1, 0, 1, 0, 1}, /*rank=*/5);
    rcclHostLayout   layout{};
    ASSERT_TRUE(rcclHostLayoutGet(mock.get(), &layout));
    EXPECT_EQ(2, layout.nHosts);
    EXPECT_EQ(4, layout.ranksPerHost);
    EXPECT_EQ(1, layout.hostIndex);
    EXPECT_EQ(2, layout.localRank);
}

// Host indices follow each host's first rank, not the hash value.
TEST(HostHierLayout, HostsNumberedByFirstRank)
{
    HostHierMockComm mock({7, 7, 3, 3}, /*rank=*/2);
    rcclHostLayout   layout{};
    ASSERT_TRUE(rcclHostLayoutGet(mock.get(), &layout));
    EXPECT_EQ(1, layout.hostIndex);
    EXPECT_EQ(0, layout.localRank);
}

TEST(HostHierLayout, SingleHostIsALayout)
{
    HostHierMockComm mock(blockHosts(1, 8), /*rank=*/3);
    rcclHostLayout   layout{};
    ASSERT_TRUE(rcclHostLayoutGet(mock.get(), &layout));
    EXPECT_EQ(1, layout.nHosts);
    EXPECT_EQ(8, layout.ranksPerHost);
    EXPECT_EQ(3, layout.localRank);
}

TEST(HostHierLayout, RejectsUnusableLayouts)
{
    rcclHostLayout layout{};
    {
        HostHierMockComm mock({0, 0, 0, 1, 1, 1, 1, 1}, 0);
        EXPECT_FALSE(rcclHostLayoutGet(mock.get(), &layout)) << "non-uniform";
    }
    {
        HostHierMockComm mock({0, 1, 2, 3}, 0);
        EXPECT_FALSE(rcclHostLayoutGet(mock.get(), &layout)) << "one rank per host";
    }
    {
        HostHierMockComm mock({0}, 0);
        EXPECT_FALSE(rcclHostLayoutGet(mock.get(), &layout)) << "single rank";
    }
    {
        HostHierMockComm mock(blockHosts(2, 8), 0);
        mock.comm.peerInfoValid = false;
        EXPECT_FALSE(rcclHostLayoutGet(mock.get(), &layout)) << "peer info not exchanged";
    }
    EXPECT_FALSE(rcclHostLayoutGet(nullptr, &layout));
}

TEST(HostHierLayout, CachedOnFirstProbe)
{
    HostHierMockComm      mock(blockHosts(2, 4), 6);
    const rcclHostLayout* layout = rcclHostHierGetLayout(mock.get());
    ASSERT_NE(nullptr, layout);
    EXPECT_EQ(rcclHostHierLayout, mock.comm.hostHierState);
    EXPECT_EQ(4, layout->ranksPerHost);

    mock.setHosts({0, 0, 0, 1, 1, 1, 1, 1});
    EXPECT_EQ(layout, rcclHostHierGetLayout(mock.get()));
}

TEST(HostHierLayout, NoLayoutIsCachedToo)
{
    HostHierMockComm mock({0, 0, 0, 1, 1, 1, 1, 1}, 0);
    EXPECT_EQ(nullptr, rcclHostHierGetLayout(mock.get()));
    EXPECT_EQ(rcclHostHierOff, mock.comm.hostHierState);
    EXPECT_EQ(nullptr, mock.comm.hostHier);

    mock.setHosts(blockHosts(2, 4));
    EXPECT_EQ(nullptr, rcclHostHierGetLayout(mock.get()));
}

// Child comms are marked off so they never split again, and a failed split is
// never retried.
TEST(HostHierLayout, OffAndFailedCommsReportNoLayout)
{
    for(int state : {static_cast<int>(rcclHostHierOff), static_cast<int>(rcclHostHierFailed)})
    {
        HostHierMockComm mock(blockHosts(2, 8), 0);
        mock.comm.hostHierState = state;
        EXPECT_EQ(nullptr, rcclHostHierGetLayout(mock.get())) << "state " << state;
        EXPECT_EQ(state, mock.comm.hostHierState);
    }
}

TEST(HostHierLayout, EnsureWithoutLayoutIsInvalidUsage)
{
    HostHierMockComm mock({0, 0, 0, 1, 1, 1, 1, 1}, 0);
    EXPECT_EQ(ncclInvalidUsage, rcclHostHierEnsure(mock.get()));
    EXPECT_EQ(rcclHostHierOff, mock.comm.hostHierState);
}

TEST(HostHierAllReduce, NeedsChildComms)
{
    HostHierMockComm mock(blockHosts(2, 8), 0);
    ASSERT_NE(nullptr, rcclHostHierGetLayout(mock.get()));
    EXPECT_EQ(ncclInternalError,
              rcclHostHierAllReduce(nullptr, nullptr, 1024, ncclFloat32, ncclSum, mock.get(), nullptr));
}

TEST(HostHierAllReduce, AutoWindowEdges)
{
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(2, kMiB - 1));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(2, kMiB));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(2, 2 * kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(2, 2 * kMiB + 1));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(2, 4 * kMiB));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(2, 8 * kMiB));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(2, 16 * kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(2, 64 * kMiB));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(2, 128 * kMiB));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(2, 512 * kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(2, 1024 * kMiB));

    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(4, kMiB - 1));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(4, kMiB));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(4, 1024 * kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(4, 1024 * kMiB + 1));

    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(8, kMiB - 1));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(8, kMiB));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(8, 2048 * kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(8, 2048 * kMiB + 1));

    for(int nHosts : {0, 1, 3, 5, 6, 7, 16})
        EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(nHosts, 16 * kMiB)) << nHosts << " hosts";
}

// RCCL_HOST_HIER_ALLREDUCE is cached on first read, so each mode runs in its own process.
TEST(HostHierAllReduce, EligibilityPerMode_Isolated)
{
    auto autoMode = ProcessIsolatedTestRunner::TestConfig(
        "HostHierAllReduce_AutoMode_Isolated",
        []()
        {
            auto eligible = [](const std::vector<int>& hostOf, size_t count, ncclDataType_t type = ncclFloat32,
                               ncclRedOp_t op = ncclSum, const char* arch = "gfx1250") {
                HostHierMockComm mock(hostOf, 0, arch);
                return rcclHostHierAllReduceEligible(mock.get(), count, type, op);
            };
            EXPECT_TRUE(eligible(blockHosts(2, 8), floats(kMiB)));
            EXPECT_FALSE(eligible(blockHosts(2, 8), floats(kMiB) - 8)) << "below the window";
            EXPECT_FALSE(eligible(blockHosts(2, 8), floats(4 * kMiB))) << "2-host gap";
            EXPECT_TRUE(eligible(blockHosts(4, 8), floats(1024 * kMiB)));
            EXPECT_FALSE(eligible(blockHosts(4, 8), floats(2048 * kMiB)));
            EXPECT_TRUE(eligible(blockHosts(8, 8), floats(2048 * kMiB)));
            EXPECT_FALSE(eligible(blockHosts(1, 8), floats(16 * kMiB))) << "single host";
            EXPECT_FALSE(eligible(blockHosts(2, 4), floats(16 * kMiB))) << "four ranks per host";
            EXPECT_FALSE(eligible(blockHosts(2, 8), kMiB / 2, ncclBfloat16)) << "bf16";
            EXPECT_FALSE(eligible(blockHosts(2, 8), floats(kMiB), ncclFloat32, ncclMax)) << "max";
            EXPECT_FALSE(eligible(blockHosts(2, 8), floats(kMiB), ncclFloat32, ncclSum, "gfx950")) << "gfx950";
        })
        .withTimeout(std::chrono::seconds(30))
        .withNumGpus(0);
    autoMode.clearVariable("RCCL_HOST_HIER_ALLREDUCE");

    RUN_ISOLATED_TESTS(
        autoMode,
        ProcessIsolatedTestRunner::TestConfig(
            "HostHierAllReduce_DisabledMode_Isolated",
            []()
            {
                HostHierMockComm mock(blockHosts(2, 8), 0);
                EXPECT_FALSE(rcclHostHierAllReduceEligible(mock.get(), floats(kMiB), ncclFloat32, ncclSum));
                EXPECT_EQ(rcclHostHierUnprobed, mock.comm.hostHierState) << "disabled mode must not probe";
            })
            .withEnvironment({{"RCCL_HOST_HIER_ALLREDUCE", "0"}})
            .withTimeout(std::chrono::seconds(30))
            .withNumGpus(0),
        ProcessIsolatedTestRunner::TestConfig(
            "HostHierAllReduce_ForcedMode_Isolated",
            []()
            {
                auto eligible = [](const std::vector<int>& hostOf, size_t count, ncclDataType_t type,
                                   ncclRedOp_t op, const char* arch = "gfx1250") {
                    HostHierMockComm mock(hostOf, 0, arch);
                    return rcclHostHierAllReduceEligible(mock.get(), count, type, op);
                };
                EXPECT_TRUE(eligible(blockHosts(2, 4), 1000, ncclBfloat16, ncclMax, "gfx950"));
                EXPECT_TRUE(eligible(blockHosts(1, 8), 8, ncclInt32, ncclMin));
                EXPECT_TRUE(eligible(blockHosts(2, 8), 16, ncclInt64, ncclProd));
                EXPECT_TRUE(eligible(blockHosts(2, 8), 16, ncclFloat32, ncclAvg));
                EXPECT_TRUE(eligible(blockHosts(2, 8), 16, ncclFloat8e4m3, ncclAvg));
                EXPECT_FALSE(eligible(blockHosts(2, 8), 16, ncclInt32, ncclAvg)) << "integer avg";
                EXPECT_FALSE(eligible(blockHosts(2, 8), 16, ncclFloat32, static_cast<ncclRedOp_t>(ncclNumOps)))
                    << "PreMulSum op";
                EXPECT_FALSE(eligible(blockHosts(2, 8), 12, ncclFloat32, ncclSum)) << "count not divisible";
                EXPECT_FALSE(eligible(blockHosts(2, 8), 0, ncclFloat32, ncclSum)) << "empty";
                EXPECT_FALSE(eligible({0, 0, 0, 1, 1, 1, 1, 1}, 16, ncclFloat32, ncclSum)) << "non-uniform";
            })
            .withEnvironment({{"RCCL_HOST_HIER_ALLREDUCE", "1"}})
            .withTimeout(std::chrono::seconds(30))
            .withNumGpus(0));
}

} // namespace RcclUnitTesting
