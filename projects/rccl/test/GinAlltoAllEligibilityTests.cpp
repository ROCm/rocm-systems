/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#include "algorithms/gin/gin_alltoall.h"
#include "comm.h"
#include "dev_runtime.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{

#if defined(ENABLE_ROCSHMEM_GIN)

namespace
{

constexpr uintptr_t kSendAddr = 0x1000;
constexpr uintptr_t kRecvAddr = 0x2000;

// Stand-in ncclComm that passes every ncclAllToAllGinSdmaEligible() gate, so each
// test below spoils one field and the verdict flips only because of it. bigSize
// must stay non-zero or ncclTeamLsa() runs real device-runtime init here and
// crashes. Purpose-built like DdaIpcMockComm: common/MockComm.hpp builds topology
// and arch state that no eligibility gate reads.
struct GinAlltoAllMockComm
{
    ncclComm             comm{};
    ncclSharedResources  sharedRes{};
    char                 bootstrapPlaceholder{0};
    ncclDevrWindow       window{};
    ncclDevrWindowSorted sorted{};

    GinAlltoAllMockComm()
    {
        std::memset(&comm, 0, sizeof(comm));
        comm.bootstrap        = &bootstrapPlaceholder;
        comm.nNodes           = 1;
        comm.nRanks           = 8;
        comm.rank             = 0;
        comm.symmetricSupport = true;
        comm.globalGinSupport = NCCL_GIN_CONNECTION_FULL;

        comm.devrState.bigSize = 1;
        comm.devrState.lsaSize = comm.nRanks;
        comm.devrState.lsaSelf = comm.rank;

        // One window spanning both buffers is enough for ncclDevrFindWindow.
        window.userPtr                = reinterpret_cast<void*>(kSendAddr);
        window.size                   = 2 * (kRecvAddr - kSendAddr);
        sorted                        = {kSendAddr, window.size, &window};
        comm.devrState.winSorted      = &sorted;
        comm.devrState.winSortedCount = 1;

        setBackends({NCCL_GIN_TYPE_ANVIL_SDMA});
        comm.sharedRes = &sharedRes;
    }

    void setBackends(std::initializer_list<ncclGinType_t> types)
    {
        int n = 0;
        for (ncclGinType_t type : types) sharedRes.ginState.backends[n++].ginType = type;
        sharedRes.ginState.numActiveBackends = n;
    }

    ncclComm* get() { return &comm; }
};

bool envEquals(const char* name, const char* value)
{
    const char* env = std::getenv(name);
    return env != nullptr && std::strcmp(env, value) == 0;
}

// The test_runner config reruns this binary with these params set to reject every size.
bool paramRejectsAll()
{
    return envEquals("NCCL_GIN_A2A_ENABLE", "0") || std::getenv("NCCL_GIN_A2A_MIN_BYTES") != nullptr;
}

} // namespace

// Baseline is eligible, so each case below fails if the gate it targets is deleted.
// The 16 B alignment gate stays with the MPI tests in transport/GinDeviceMPITests.cpp,
// since the size used here sits on the SDMA tier that skips it.
class GinAlltoAllEligibilityTest : public ::testing::Test
{
protected:
    GinAlltoAllMockComm mockComm_;
    void*               sendbuff_{reinterpret_cast<void*>(kSendAddr)};
    void*               recvbuff_{reinterpret_cast<void*>(kRecvAddr)};
    // 8 MiB per peer of float32, i.e. the NCCL_GIN_A2A_SDMA_MIN_BYTES default, so
    // neither the minimum-size nor the 16 B alignment gate can be what rejects.
    static constexpr size_t kBytesPerPeer = 8ull * 1024 * 1024;
    static constexpr size_t kCount        = kBytesPerPeer / sizeof(float);

    bool eligible(size_t count)
    {
        return ncclAllToAllGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, count, ncclFloat32);
    }
};

TEST_F(GinAlltoAllEligibilityTest, BaselineEligible)
{
    if (paramRejectsAll()) GTEST_SKIP() << "a pinned param rejects every size";
    EXPECT_TRUE(eligible(kCount));
}

TEST_F(GinAlltoAllEligibilityTest, NullComm)
{
    EXPECT_FALSE(ncclAllToAllGinSdmaEligible(nullptr, sendbuff_, recvbuff_, kCount, ncclFloat32));
}

TEST_F(GinAlltoAllEligibilityTest, MissingBootstrap)
{
    mockComm_.comm.bootstrap = nullptr;
    EXPECT_FALSE(eligible(kCount));
}

TEST_F(GinAlltoAllEligibilityTest, ZeroCount)
{
    EXPECT_FALSE(eligible(0));
}

// The path is scaleup-only: anything spanning nodes has to fall back.
TEST_F(GinAlltoAllEligibilityTest, MultiNode)
{
    mockComm_.comm.nNodes = 2;
    EXPECT_FALSE(eligible(kCount));
}

// markSdmaDirty packs (peer, channel) into a 64-bit mask, which caps the path at
// kGinA2AMaxRanks (16) in gin_alltoall_sdma.cu.
TEST_F(GinAlltoAllEligibilityTest, TooManyRanks)
{
    mockComm_.comm.nRanks            = 32;
    mockComm_.comm.devrState.lsaSize = mockComm_.comm.nRanks;
    EXPECT_FALSE(eligible(kCount));
}

TEST_F(GinAlltoAllEligibilityTest, NoSymmetricSupport)
{
    mockComm_.comm.symmetricSupport = false;
    EXPECT_FALSE(eligible(kCount));
}

// Partial GIN connectivity is not enough; every peer has to be reachable.
TEST_F(GinAlltoAllEligibilityTest, PartialGinConnectivity)
{
    mockComm_.comm.globalGinSupport = NCCL_GIN_CONNECTION_NONE;
    EXPECT_FALSE(eligible(kCount));

    mockComm_.comm.globalGinSupport = NCCL_GIN_CONNECTION_RAIL;
    EXPECT_FALSE(eligible(kCount));
}

// A comm whose LSA team covers only part of the world cannot reach every peer
// with stores, so the alltoall has to fall back even on one node.
TEST_F(GinAlltoAllEligibilityTest, LsaTeamSmallerThanComm)
{
    mockComm_.comm.devrState.lsaSize = mockComm_.comm.nRanks / 2;
    EXPECT_FALSE(eligible(kCount));
}

// The kernel only runs on SDMA, so a comm with no SDMA backend must not be claimed.
TEST_F(GinAlltoAllEligibilityTest, SdmaNotActive)
{
    mockComm_.setBackends({NCCL_GIN_TYPE_ROCSHMEM_GDA});
    EXPECT_FALSE(eligible(kCount));

    mockComm_.setBackends({NCCL_GIN_TYPE_PROXY});
    EXPECT_FALSE(eligible(kCount));

    mockComm_.setBackends({NCCL_GIN_TYPE_ROCSHMEM_GDA, NCCL_GIN_TYPE_PROXY});
    EXPECT_FALSE(eligible(kCount));
}

// The DevComm requests SDMA by type, so SDMA does not have to be the default.
TEST_F(GinAlltoAllEligibilityTest, SdmaActiveButNotDefault)
{
    if (paramRejectsAll()) GTEST_SKIP() << "a pinned param rejects every size";
    mockComm_.setBackends({NCCL_GIN_TYPE_PROXY, NCCL_GIN_TYPE_ANVIL_SDMA});
    EXPECT_TRUE(eligible(kCount));

    mockComm_.setBackends({NCCL_GIN_TYPE_ROCSHMEM_GDA, NCCL_GIN_TYPE_PROXY, NCCL_GIN_TYPE_ANVIL_SDMA});
    EXPECT_TRUE(eligible(kCount));
}

// A finalized backend list is kept only for plugin teardown and must not be selected.
TEST_F(GinAlltoAllEligibilityTest, GinFinalized)
{
    mockComm_.sharedRes.ginState.finalized = true;
    EXPECT_FALSE(eligible(kCount));
}

// The kernel puts straight into symmetric windows, so unregistered buffers are rejected.
TEST_F(GinAlltoAllEligibilityTest, BuffersNotRegistered)
{
    mockComm_.comm.devrState.winSortedCount = 0;
    EXPECT_FALSE(eligible(kCount));
}

// NCCL_GIN_A2A_ENABLE is an NCCL_PARAM, so it is read once per process and cannot
// be flipped from inside the binary. The test_runner config gives this filter a
// run of its own with the kill switch thrown; see
// tools/scripts/test_runner/configs/gin_rocshmem_backends_mellanox.json.
TEST_F(GinAlltoAllEligibilityTest, A2ADisabledByParam)
{
    if (!envEquals("NCCL_GIN_A2A_ENABLE", "0")) {
        GTEST_SKIP() << "needs NCCL_GIN_A2A_ENABLE=0 in the environment";
    }
    EXPECT_FALSE(eligible(kCount));
}

// Same deal: the config pins NCCL_GIN_A2A_MIN_BYTES above the size this test asks
// for, so the floor is what rejects.
TEST_F(GinAlltoAllEligibilityTest, BelowMinBytesParam)
{
    const char* env = std::getenv("NCCL_GIN_A2A_MIN_BYTES");
    if (env == nullptr) {
        GTEST_SKIP() << "needs NCCL_GIN_A2A_MIN_BYTES set above " << kBytesPerPeer;
    }
    ASSERT_GT(std::strtoull(env, nullptr, 0), kBytesPerPeer)
        << "config must pin the floor above the per-peer size under test";
    EXPECT_FALSE(eligible(kCount));
}

#endif // ENABLE_ROCSHMEM_GIN

#if !defined(ENABLE_ROCSHMEM_GIN)
TEST(GinAlltoAllEligibilityTest, SkippedWithoutGinBuild)
{
    GTEST_SKIP() << "GIN AllToAll eligibility tests require ENABLE_ROCSHMEM_GIN";
}
#endif

} // namespace RcclUnitTesting
