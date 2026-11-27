/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/ProcessIsolatedTestRunner.hpp"

#include "comm.h"
#include "hier_host.h"
#include "nccl.h"
#include "rccl_arch_thresholds.h"

#include "gtest/gtest.h"

#include <algorithm>
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
    char                      fakeDdaHandler = 0;

    HostHierMockComm(const std::vector<int>& hostOf, int rank, const char* arch = "gfx1250", bool mnnvl = true)
    {
        std::memset(&comm, 0, sizeof(comm));
        std::strncpy(archBuf, arch, sizeof(archBuf) - 1);
        comm.archName = archBuf;
        comm.MNNVL    = mnnvl ? 1 : 0;
        comm.nRanks   = static_cast<int>(hostOf.size());
        // One clique holding every rank, as on a single MNNVL fabric.
        comm.clique.size = mnnvl ? comm.nRanks : 0;
        // A parent comm on that fabric sets up DDA; eligibility only checks that it did.
        comm.ddaFabricMemHandler = mnnvl ? reinterpret_cast<ncclFabricMemHandler*>(&fakeDdaHandler) : nullptr;
        comm.rank     = rank;
        setHosts(hostOf);
        comm.hostHierState = rcclHostHierUnprobed;
        // The table of the partition mode that puts this many ranks on each host.
        const bool spx = std::count(hostOf.begin(), hostOf.end(), hostOf.empty() ? 0 : hostOf[0]) == 4;
        comm.archThresholds
            = rcclGetArchThresholds(archBuf, spx ? RCCL_TUNING_PARTITION_SPX : RCCL_TUNING_PARTITION_DPX);
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

// The fused kernel takes its intra-host and inter-host peers from hostRanks.
TEST(HostHierLayout, HostRanksOfInterleavedHosts)
{
    HostHierMockComm mock({0, 1, 0, 1, 0, 1, 0, 1}, /*rank=*/5);
    ASSERT_NE(nullptr, rcclHostHierGetLayout(mock.get()));
    ASSERT_NE(nullptr, mock.comm.hostHier);
    const int* hostRanks = mock.comm.hostHier->hostRanks;
    ASSERT_NE(nullptr, hostRanks);
    EXPECT_EQ(std::vector<int>({0, 2, 4, 6, 1, 3, 5, 7}), std::vector<int>(hostRanks, hostRanks + 8));
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

TEST(HostHierLayout, ChildrenListsTheChildCommsThatExist)
{
    HostHierMockComm mock(blockHosts(2, 8), 0);
    ncclComm_t       children[2] = {};
    EXPECT_EQ(0, rcclHostHierChildren(mock.get(), children)) << "before the layout probe";
    ASSERT_NE(nullptr, rcclHostHierGetLayout(mock.get()));
    EXPECT_EQ(0, rcclHostHierChildren(mock.get(), children)) << "before the split";

    // Stand-ins for the split comms; cleared before teardown, which would release them.
    auto intra = std::make_unique<ncclComm>();
    auto inter = std::make_unique<ncclComm>();
    mock.comm.hostHier->intra = intra.get();
    EXPECT_EQ(1, rcclHostHierChildren(mock.get(), children)) << "one host: no inter-host comm";
    EXPECT_EQ(intra.get(), children[0]);
    mock.comm.hostHier->inter = inter.get();
    ASSERT_EQ(2, rcclHostHierChildren(mock.get(), children));
    EXPECT_EQ(intra.get(), children[0]);
    EXPECT_EQ(inter.get(), children[1]);
    mock.comm.hostHier->intra = nullptr;
    mock.comm.hostHier->inter = nullptr;
}

TEST(HostHierAllReduce, AutoWindowEdges)
{
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(2, 8, kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(2, 8, 2 * kMiB - 1));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(2, 8, 2 * kMiB));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(2, 8, 16 * kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(2, 8, 16 * kMiB + 1));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(2, 8, 32 * kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(2, 8, 64 * kMiB - 1));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(2, 8, 64 * kMiB));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(2, 8, 128 * kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(2, 8, 128 * kMiB + 1));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(2, 8, 512 * kMiB));

    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(4, 8, kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(4, 8, 2 * kMiB - 1));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(4, 8, 2 * kMiB));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(4, 8, 256 * kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(4, 8, 256 * kMiB + 1));

    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(8, 8, kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(8, 8, 2 * kMiB - 1));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(8, 8, 2 * kMiB));
    EXPECT_TRUE(rcclHostHierAllReduceAutoWindow(8, 8, 512 * kMiB));
    EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(8, 8, 512 * kMiB + 1));

    for(int nHosts : {1, 2, 4, 8, 16})
        for(size_t bytes : {2 * kMiB, 32 * kMiB, 128 * kMiB, 512 * kMiB})
            EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(nHosts, 4, bytes)) << bytes << " B on " << nHosts
                                                                            << " hosts of four ranks";

    for(int nHosts : {0, 1, 3, 5, 6, 7, 16})
        EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(nHosts, 8, 16 * kMiB)) << nHosts << " hosts";
    for(int ranksPerHost : {2, 16})
        EXPECT_FALSE(rcclHostHierAllReduceAutoWindow(4, ranksPerHost, 16 * kMiB)) << ranksPerHost << " ranks per host";
}

TEST(HostHierAllReduce, FusedWindowEdges)
{
    for(int nHosts : {1, 2})
    {
        EXPECT_FALSE(rcclHostHierAllReduceFusedWindow(nHosts, 8, 4 * kMiB - 1)) << nHosts << " hosts";
        EXPECT_TRUE(rcclHostHierAllReduceFusedWindow(nHosts, 8, 4 * kMiB)) << nHosts << " hosts";
        EXPECT_TRUE(rcclHostHierAllReduceFusedWindow(nHosts, 8, 16 * kMiB)) << nHosts << " hosts";
        EXPECT_FALSE(rcclHostHierAllReduceFusedWindow(nHosts, 8, 16 * kMiB + 1)) << nHosts << " hosts";
        EXPECT_FALSE(rcclHostHierAllReduceFusedWindow(nHosts, 8, 128 * kMiB)) << nHosts << " hosts";
        EXPECT_FALSE(rcclHostHierAllReduceFusedWindow(nHosts, 4, 16 * kMiB)) << nHosts << " hosts, 4 ranks each";
    }
    for(int nHosts : {4, 8})
        EXPECT_FALSE(rcclHostHierAllReduceFusedWindow(nHosts, 8, 16 * kMiB)) << nHosts << " hosts";
}

TEST(HostHierAllReduce, AutoTypes)
{
    const ncclDataType_t all[] = {ncclInt8,     ncclUint8,   ncclInt32,   ncclUint32,
                                  ncclInt64,    ncclUint64,  ncclFloat16, ncclFloat32,
                                  ncclFloat64,  ncclBfloat16, ncclFloat8e4m3, ncclFloat8e5m2};
    for(ncclDataType_t t : all)
    {
        const bool fp8      = t == ncclFloat8e4m3 || t == ncclFloat8e5m2;
        const bool floatSum = fp8 || t == ncclFloat32 || t == ncclFloat16 || t == ncclBfloat16;
        EXPECT_EQ(floatSum, rcclHostHierAllReduceAutoType(t, ncclSum)) << t;
        for(ncclRedOp_t op : {ncclProd, ncclMax, ncclMin, ncclAvg})
            EXPECT_FALSE(rcclHostHierAllReduceAutoType(t, op)) << t << " op " << op;
    }
}

// RCCL_HOST_HIER_ALLREDUCE is cached on first read, so each mode runs in its own process.
TEST(HostHierAllReduce, EligibilityPerMode_Isolated)
{
    auto autoMode = ProcessIsolatedTestRunner::TestConfig(
        "HostHierAllReduce_AutoMode_Isolated",
        []()
        {
            auto eligible = [](const std::vector<int>& hostOf, size_t count, ncclDataType_t type = ncclFloat32,
                               ncclRedOp_t op = ncclSum, const char* arch = "gfx1250", bool mnnvl = true,
                               bool registered = false) {
                HostHierMockComm mock(hostOf, 0, arch, mnnvl);
                return rcclHostHierAllReduceEligible(mock.get(), count, type, op, registered);
            };
            EXPECT_TRUE(eligible(blockHosts(2, 8), floats(2 * kMiB)));
            EXPECT_FALSE(eligible(blockHosts(2, 8), floats(2 * kMiB) - 8)) << "below the window";
            EXPECT_TRUE(eligible(blockHosts(2, 8), floats(128 * kMiB)));
            EXPECT_FALSE(eligible(blockHosts(2, 8), floats(256 * kMiB))) << "two hosts, above the window";
            EXPECT_FALSE(eligible(blockHosts(2, 8), floats(1024 * kMiB))) << "above the window";
            EXPECT_FALSE(eligible(blockHosts(4, 8), floats(kMiB))) << "four hosts, below the window";
            EXPECT_TRUE(eligible(blockHosts(4, 8), floats(2 * kMiB)));
            EXPECT_TRUE(eligible(blockHosts(4, 8), floats(256 * kMiB)));
            EXPECT_FALSE(eligible(blockHosts(4, 8), floats(512 * kMiB))) << "four hosts, above the window";
            EXPECT_TRUE(eligible(blockHosts(8, 8), floats(512 * kMiB)));
            EXPECT_FALSE(eligible(blockHosts(8, 8), floats(1024 * kMiB))) << "eight hosts, above the window";
            EXPECT_FALSE(eligible(blockHosts(1, 8), floats(16 * kMiB))) << "single host";
            EXPECT_FALSE(eligible(blockHosts(2, 4), floats(16 * kMiB))) << "two hosts of four ranks";
            for(int nHosts : {4, 8})
                for(size_t bytes : {2 * kMiB, 128 * kMiB, 512 * kMiB})
                    EXPECT_FALSE(eligible(blockHosts(nHosts, 4), floats(bytes)))
                        << bytes << " B on " << nHosts << " hosts of four ranks (SPX)";
            EXPECT_FALSE(eligible(blockHosts(4, 8), floats(16 * kMiB), ncclFloat32, ncclMax)) << "fp32 max, DPX";
            EXPECT_TRUE(eligible(blockHosts(2, 8), kMiB, ncclBfloat16)) << "bf16, 2 MiB";
            EXPECT_TRUE(eligible(blockHosts(2, 8), 64 * kMiB, ncclFloat16)) << "fp16, 128 MiB";
            EXPECT_FALSE(eligible(blockHosts(2, 8), kMiB - 8, ncclBfloat16)) << "bf16 below the window";
            EXPECT_TRUE(eligible(blockHosts(4, 8), 8 * kMiB, ncclBfloat16)) << "bf16 on four hosts";
            EXPECT_TRUE(eligible(blockHosts(8, 8), 8 * kMiB, ncclFloat8e4m3)) << "fp8 sum on eight hosts";
            EXPECT_FALSE(eligible(blockHosts(2, 8), 8 * kMiB, ncclInt32)) << "int32 sum, DPX";
            EXPECT_FALSE(eligible(blockHosts(2, 8), kMiB, ncclFloat64)) << "fp64";
            EXPECT_FALSE(eligible(blockHosts(2, 8), floats(2 * kMiB), ncclFloat32, ncclMax)) << "max";
            EXPECT_FALSE(eligible(blockHosts(2, 8), floats(2 * kMiB), ncclFloat32, ncclSum, "gfx950")) << "gfx950";
            EXPECT_FALSE(eligible(blockHosts(2, 8), floats(2 * kMiB), ncclFloat32, ncclSum, "gfx1250", false))
                << "scale-out gfx1250 without MNNVL";
            EXPECT_FALSE(eligible(blockHosts(2, 8), floats(2 * kMiB), ncclFloat32, ncclSum, "gfx1250", true, true))
                << "registered window";
            {
                HostHierMockComm mock(blockHosts(2, 8), 0);
                mock.comm.clique.size = 8;
                EXPECT_FALSE(rcclHostHierAllReduceEligible(mock.get(), floats(2 * kMiB), ncclFloat32, ncclSum, false))
                    << "two MNNVL cliques";
            }
            // The mock comm has no DDA scratch, so the fused kernel never runs on it.
            for(size_t bytes : {4 * kMiB, 8 * kMiB - 64})
            {
                EXPECT_FALSE(eligible(blockHosts(2, 8), floats(bytes))) << bytes << " B on two hosts, no fused kernel";
                EXPECT_FALSE(eligible(blockHosts(2, 8), bytes / 2, ncclBfloat16)) << bytes << " B bf16, no fused kernel";
                EXPECT_TRUE(eligible(blockHosts(2, 8), bytes, ncclFloat8e4m3)) << bytes << " B fp8 on two hosts";
                EXPECT_TRUE(eligible(blockHosts(4, 8), floats(bytes))) << bytes << " B on four hosts";
            }
            for(size_t bytes : {32 * kMiB, 64 * kMiB - 64})
            {
                EXPECT_FALSE(eligible(blockHosts(2, 8), floats(bytes))) << bytes << " B on two hosts, outside the window";
                EXPECT_FALSE(eligible(blockHosts(2, 8), bytes, ncclFloat8e4m3)) << bytes << " B fp8, outside the window";
                EXPECT_TRUE(eligible(blockHosts(4, 8), floats(bytes))) << bytes << " B on four hosts";
            }
            for(size_t bytes : {2 * kMiB, 8 * kMiB, 16 * kMiB, 64 * kMiB, 128 * kMiB - 64, 128 * kMiB})
                EXPECT_TRUE(eligible(blockHosts(2, 8), floats(bytes))) << bytes << " B on two hosts, three collectives";
            {
                HostHierMockComm mock(blockHosts(2, 8), 0);
                mock.comm.ddaFabricMemHandler = nullptr;
                EXPECT_FALSE(rcclHostHierAllReduceEligible(mock.get(), floats(2 * kMiB), ncclFloat32, ncclSum, false))
                    << "no DDA fabric, as on a comm made by ncclCommSplit";
            }
            {
                // A rank whose GPU reports CPX or no mode, or RCCL_TUNING_PARTITION_MODE=SPX, gives every rank the
                // SPX table.
                HostHierMockComm mock(blockHosts(2, 8), 0);
                mock.comm.archThresholds = rcclGetArchThresholds("gfx1250", RCCL_TUNING_PARTITION_SPX);
                EXPECT_FALSE(rcclHostHierAllReduceEligible(mock.get(), floats(2 * kMiB), ncclFloat32, ncclSum, false))
                    << "eight ranks per host with the SPX table";
            }
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
                EXPECT_FALSE(rcclHostHierAllReduceEligible(mock.get(), floats(2 * kMiB), ncclFloat32, ncclSum, false));
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
                                   ncclRedOp_t op, const char* arch = "gfx1250", bool mnnvl = true,
                                   bool registered = false) {
                    HostHierMockComm mock(hostOf, 0, arch, mnnvl);
                    return rcclHostHierAllReduceEligible(mock.get(), count, type, op, registered);
                };
                EXPECT_TRUE(eligible(blockHosts(2, 4), 1000, ncclBfloat16, ncclMax, "gfx950"));
                EXPECT_TRUE(eligible(blockHosts(2, 8), 16, ncclFloat32, ncclSum, "gfx1250", false)) << "no MNNVL";
                EXPECT_TRUE(eligible(blockHosts(2, 8), 16, ncclFloat32, ncclSum, "gfx1250", true, true))
                    << "registered window";
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
