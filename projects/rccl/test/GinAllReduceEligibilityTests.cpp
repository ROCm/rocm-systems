/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/GinAllReduceTestHelpers.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

#include "algorithms/gin/gin_all_reduce.h"
#include "nccl_device/net_device.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace RcclUnitTesting
{

#if defined(ENABLE_ROCSHMEM_GIN)

class GinAllReduceEligibilityTest : public ::testing::Test
{
protected:
    GinAllReduceMockComm mockComm_;
    void*                sendbuff_{reinterpret_cast<void*>(0x1000)};
    void*                recvbuff_{reinterpret_cast<void*>(0x2000)};
    // 4 MiB of float32: below the default 256 MiB GIN floor, above zero.
    static constexpr size_t kSmallCount = (4ull * 1024 * 1024) / 4;
};

TEST_F(GinAllReduceEligibilityTest, NullComm)
{
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(nullptr, sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
    EXPECT_FALSE(ncclAllReduceGinSdmaYieldToDda(nullptr, sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
}

TEST_F(GinAllReduceEligibilityTest, NullSendbuff)
{
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), nullptr, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
}

TEST_F(GinAllReduceEligibilityTest, NullRecvbuff)
{
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, nullptr, kSmallCount, ncclFloat32, ncclSum));
}

TEST_F(GinAllReduceEligibilityTest, ZeroCount)
{
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32, ncclSum));
}

TEST_F(GinAllReduceEligibilityTest, WrongArchRejected)
{
    mockComm_.setArch("gfx942");
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
    EXPECT_FALSE(ncclAllReduceGinSdmaYieldToDda(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
}

TEST_F(GinAllReduceEligibilityTest, WrongOpRejected)
{
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclProd));
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclMax));
}

TEST_F(GinAllReduceEligibilityTest, UnsupportedDatatypeRejected)
{
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclInt32, ncclSum));
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat64, ncclSum));
}

TEST_F(GinAllReduceEligibilityTest, NoSymmetricSupportRejected)
{
    mockComm_.comm.symmetricSupport = false;
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
    EXPECT_FALSE(ncclAllReduceGinSdmaYieldToDda(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
}

// Stand-in that passes every gate after the gfx1250 opt-in, so deleting that
// check flips the disabled case from false to true. symkState.initialized and
// devrState.bigSize skip ncclSymkInitOnce / ncclDevrInitOnce. The sorted-window
// record matches ncclDevrWindowSorted in src/dev_runtime.cc (private to that file).
struct Gfx1250EligibleStandIn
{
    ncclComm            comm{};
    ncclSharedResources shared{};
    ncclDevrWindow      win{};
    struct SortedWin
    {
        uintptr_t       userAddr;
        size_t          size;
        ncclDevrWindow* win;
    } sorted{};
    char arch_[16]{};

    static constexpr uintptr_t kBase = 0x1000;
    static constexpr size_t    kSpan = 0x10000;

    Gfx1250EligibleStandIn()
    {
        static_assert(offsetof(SortedWin, userAddr) == 0, "window sort key must lead");
        static_assert(offsetof(SortedWin, size) == sizeof(uintptr_t), "window sort size offset");
        static_assert(offsetof(SortedWin, win) == sizeof(uintptr_t) + sizeof(size_t), "window sort pointer offset");

        std::memcpy(arch_, "gfx1250", 7);
        comm.archName              = arch_;
        comm.nNodes                = 1;
        comm.nRanks                = 8;
        comm.rank                  = 0;
        comm.symmetricSupport      = true;
        comm.isAllDirectNvlink     = true;
        comm.globalGinSupport      = NCCL_GIN_CONNECTION_FULL;
        comm.symkState.initialized = true;
        comm.devrState.bigSize     = 1;
        comm.devrState.lsaSize     = comm.nRanks;
        comm.devrState.lsaSelf     = 0;
        comm.sharedRes             = &shared;

        shared.ginState.finalized           = false;
        shared.ginState.numActiveBackends   = 1;
        shared.ginState.backends[0].ginType = static_cast<ncclGinType_t>(NCCL_NET_DEVICE_GIN_ANVIL_SDMA);

        win.winFlags    = NCCL_WIN_COLL_SYMMETRIC;
        win.size        = kSpan;
        sorted.userAddr = kBase;
        sorted.size     = kSpan;
        sorted.win      = &win;
        comm.devrState.winSorted =
            reinterpret_cast<ncclDevrWindowSorted*>(&sorted);
        comm.devrState.winSortedCount = 1;
    }

    ncclComm* get() { return &comm; }
    void*     send() const { return reinterpret_cast<void*>(kBase); }
    void*     recv() const { return reinterpret_cast<void*>(kBase + (kSpan / 2)); }
};

// NCCL_GIN_ALLREDUCE_GFX1250_ENABLE is cached on first read, so each setting
// runs in its own process. Default off rejects gfx1250 even in the GinScatter
// band. On: [64 MiB, 4 GiB] is eligible, below 64 MiB yields to DDA, above
// 4 GiB is left to the other selectors.
TEST_F(GinAllReduceEligibilityTest, Gfx1250OptIn)
{
    auto countForBytes = [](size_t bytes) { return bytes / sizeof(float); };
    const size_t scatterCount = countForBytes(kGinAllReduceGinScatterMinBytes);
    const size_t belowCount   = countForBytes(kGinAllReduceLsaOneShotMaxBytes);
    const size_t aboveCount   = countForBytes(kGinAllReduceGinScatterMaxBytes + 256);

    auto checkDisabled = [&]() {
        Gfx1250EligibleStandIn standIn;
        EXPECT_FALSE(ncclAllReduceGinSdmaEligible(standIn.get(), standIn.send(), standIn.recv(), scatterCount,
                                                  ncclFloat32, ncclSum));
        EXPECT_FALSE(ncclAllReduceGinSdmaYieldToDda(standIn.get(), standIn.send(), standIn.recv(), scatterCount,
                                                    ncclFloat32, ncclSum));
        EXPECT_FALSE(ncclAllReduceGinSdmaEligible(standIn.get(), standIn.send(), standIn.recv(), belowCount,
                                                  ncclFloat32, ncclSum));
        EXPECT_FALSE(ncclAllReduceGinSdmaYieldToDda(standIn.get(), standIn.send(), standIn.recv(), belowCount,
                                                    ncclFloat32, ncclSum));
    };
    auto checkEnabled = [&]() {
        Gfx1250EligibleStandIn standIn;
        EXPECT_TRUE(ncclAllReduceGinSdmaEligible(standIn.get(), standIn.send(), standIn.recv(), scatterCount,
                                                 ncclFloat32, ncclSum));
        EXPECT_FALSE(ncclAllReduceGinSdmaYieldToDda(standIn.get(), standIn.send(), standIn.recv(), scatterCount,
                                                    ncclFloat32, ncclSum));
        EXPECT_FALSE(ncclAllReduceGinSdmaEligible(standIn.get(), standIn.send(), standIn.recv(), belowCount,
                                                  ncclFloat32, ncclSum));
        EXPECT_TRUE(ncclAllReduceGinSdmaYieldToDda(standIn.get(), standIn.send(), standIn.recv(), belowCount,
                                                   ncclFloat32, ncclSum));
        EXPECT_FALSE(ncclAllReduceGinSdmaEligible(standIn.get(), standIn.send(), standIn.recv(), aboveCount,
                                                  ncclFloat32, ncclSum));
        EXPECT_FALSE(ncclAllReduceGinSdmaYieldToDda(standIn.get(), standIn.send(), standIn.recv(), aboveCount,
                                                    ncclFloat32, ncclSum));
    };

    auto cleared = [](ProcessIsolatedTestRunner::TestConfig cfg) {
        cfg.clearVariable("NCCL_GIN_ALLREDUCE_ENABLE")
            .clearVariable("NCCL_GIN_ALLREDUCE_SCRATCH_BYTES")
            .clearVariable("RCCL_GIN_ALLREDUCE_FORCE_ENABLE");
        return cfg;
    };

    RUN_ISOLATED_TESTS(
        cleared(ProcessIsolatedTestRunner::TestConfig("Gfx1250DisabledByDefault", checkDisabled))
            .clearVariable("NCCL_GIN_ALLREDUCE_GFX1250_ENABLE"),
        cleared(ProcessIsolatedTestRunner::TestConfig("Gfx1250EnabledBands", checkEnabled))
            .setVariable("NCCL_GIN_ALLREDUCE_GFX1250_ENABLE", "1"));
}

#endif // ENABLE_ROCSHMEM_GIN

#if !defined(ENABLE_ROCSHMEM_GIN)
TEST(GinAllReduceEligibilityTest, SkippedWithoutGinBuild)
{
    GTEST_SKIP() << "GIN AllReduce eligibility tests require ENABLE_ROCSHMEM_GIN";
}
#endif

} // namespace RcclUnitTesting
