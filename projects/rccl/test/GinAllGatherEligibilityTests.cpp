/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "algorithms/gin/gin_all_gather.h"
#include "comm.h"
#include "dev_runtime_internal.h"
#include "group.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

#if defined(ENABLE_ROCSHMEM_GIN)
namespace {

// Host-only stand-in. bigSize stays non-zero so ncclTeamLsa() does not run device-runtime init.
// Window cases install ncclDevrWindowSorted directly; they do not allocate GPU memory.
struct GinWindow {
  ncclDevrMemory mem{};
  ncclDevrWindow win{};
  ncclDevrWindowSorted sorted{};

  void set(void* ptr, size_t size, int segments, bool sysmem, int flags, size_t lsaMin) {
    std::memset(&mem, 0, sizeof(mem));
    std::memset(&win, 0, sizeof(win));
    mem.maxGlobalNumSegments = segments;
    mem.globalHasSysmemSegment = sysmem;
    mem.lsaMinSize = lsaMin;
    mem.bigOffset = 0;
    win.memory = &mem;
    win.userPtr = ptr;
    win.size = size;
    win.winFlags = flags;
    win.bigOffset = 0;
    sorted.userAddr = reinterpret_cast<uintptr_t>(ptr);
    sorted.size = size;
    sorted.win = &win;
  }
};

struct GinAllGatherMockComm {
  ncclComm comm{};
  ncclSharedResources sharedRes{};
  char bootstrapPlaceholder{0};
  char archName_[16]{};

  GinAllGatherMockComm() {
    std::memset(&comm, 0, sizeof(comm));
    std::memcpy(archName_, "gfx950", 6);
    comm.archName = archName_;
    comm.bootstrap = &bootstrapPlaceholder;
    comm.nNodes = 1;
    comm.nRanks = 8;
    comm.rank = 0;
    comm.symmetricSupport = true;
    comm.globalGinSupport = NCCL_GIN_CONNECTION_FULL;
    comm.devrState.bigSize = 1;
    comm.devrState.lsaSize = comm.nRanks;
    comm.devrState.lsaSelf = comm.rank;
    sharedRes.ginState.numActiveBackends = 1;
    sharedRes.ginState.backends[0].ginType = (ncclGinType_t)NCCL_NET_DEVICE_GIN_ANVIL_SDMA;
    comm.sharedRes = &sharedRes;
  }
};

class GinAllGatherEligibilityTest : public ::testing::Test {
 protected:
  GinAllGatherMockComm mock_;
  GinWindow sendWin_{};
  GinWindow recvWin_{};
  ncclDevrWindowSorted sorted_[2]{};
  // Far enough apart that an 8 MiB send window cannot cover the receive pointer.
  void* send_{reinterpret_cast<void*>(0x100000000ull)};
  void* recv_{reinterpret_cast<void*>(0x200000000ull)};
  static constexpr size_t kCount = (8ull * 1024 * 1024) / sizeof(float);

  bool eligible(size_t count = kCount) {
    return ncclAllGatherGinSdmaEligible(&mock_.comm, send_, recv_, count, ncclFloat32);
  }

  bool eligiblePtrs(const void* send, void* recv, size_t count) {
    return ncclAllGatherGinSdmaEligible(&mock_.comm, send, recv, count, ncclFloat32);
  }

  void install(int sendSegments, bool sendSysmem, int sendFlags, size_t sendSize, size_t sendLsaMin, int recvSegments,
               bool recvSysmem, int recvFlags, size_t recvSize, size_t recvLsaMin) {
    sendWin_.set(send_, sendSize, sendSegments, sendSysmem, sendFlags, sendLsaMin);
    recvWin_.set(recv_, recvSize, recvSegments, recvSysmem, recvFlags, recvLsaMin);
    if (sendWin_.sorted.userAddr < recvWin_.sorted.userAddr) {
      sorted_[0] = sendWin_.sorted;
      sorted_[1] = recvWin_.sorted;
    } else {
      sorted_[0] = recvWin_.sorted;
      sorted_[1] = sendWin_.sorted;
    }
    mock_.comm.devrState.winSorted = sorted_;
    mock_.comm.devrState.winSortedCount = 2;
  }

  void installSingleSegment() {
    const size_t sendBytes = kCount * sizeof(float);
    const size_t recvBytes = sendBytes * static_cast<size_t>(mock_.comm.nRanks);
    install(1, false, NCCL_WIN_COLL_SYMMETRIC, sendBytes, sendBytes, 1, false, NCCL_WIN_COLL_SYMMETRIC, recvBytes,
            recvBytes);
  }
};
} // namespace

TEST_F(GinAllGatherEligibilityTest, RejectsNullOrZeroInputs) {
  installSingleSegment();
  EXPECT_TRUE(eligible());
  EXPECT_FALSE(ncclAllGatherGinSdmaEligible(nullptr, send_, recv_, kCount, ncclFloat32));
  EXPECT_FALSE(eligible(0));
}

TEST_F(GinAllGatherEligibilityTest, RejectsMissingBootstrapAndMultiNode) {
  installSingleSegment();
  EXPECT_TRUE(eligible());
  mock_.comm.bootstrap = nullptr;
  EXPECT_FALSE(eligible());
  mock_.comm.bootstrap = &mock_.bootstrapPlaceholder;
  mock_.comm.nNodes = 2;
  EXPECT_FALSE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, RejectsUnsupportedTopologyAndBackend) {
  mock_.comm.nRanks = 32;
  mock_.comm.devrState.lsaSize = 32;
  installSingleSegment();
  EXPECT_FALSE(eligible());
  mock_.comm.nRanks = 8;
  mock_.comm.devrState.lsaSize = 8;
  installSingleSegment();
  EXPECT_TRUE(eligible());
  mock_.sharedRes.ginState.backends[0].ginType = (ncclGinType_t)NCCL_NET_DEVICE_GIN_PROXY;
  EXPECT_FALSE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, RejectsNonSymmetricAndPartialLsa) {
  installSingleSegment();
  EXPECT_TRUE(eligible());
  mock_.comm.symmetricSupport = false;
  EXPECT_FALSE(eligible());
  mock_.comm.symmetricSupport = true;
  mock_.comm.devrState.lsaSize = 4;
  EXPECT_FALSE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, RequiresRegisteredBuffers) {
  ASSERT_EQ(mock_.comm.devrState.winSortedCount, 0);
  EXPECT_FALSE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, AcceptsSingleSegmentSymmetricWindows) {
  installSingleSegment();
  EXPECT_TRUE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, RejectsMultiSegmentRegistration) {
  installSingleSegment();
  sendWin_.mem.maxGlobalNumSegments = 2;
  EXPECT_FALSE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, RejectsHostBackedSegment) {
  installSingleSegment();
  recvWin_.mem.globalHasSysmemSegment = true;
  EXPECT_FALSE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, RejectsNonSymmetricWindows) {
  installSingleSegment();
  sendWin_.win.winFlags = NCCL_WIN_DEFAULT;
  EXPECT_FALSE(eligible());
  sendWin_.win.winFlags = NCCL_WIN_GIN_ONLY;
  EXPECT_FALSE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, RejectsReceivePastPeerSafeSize) {
  installSingleSegment();
  EXPECT_TRUE(eligible());
  const size_t recvBytes = kCount * sizeof(float) * static_cast<size_t>(mock_.comm.nRanks);
  recvWin_.mem.lsaMinSize = recvBytes - 1;
  EXPECT_FALSE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, AcceptsUnalignedSendThatFitsLocally) {
  // 32-byte send at offset 4. Peer-safe size is smaller than the local window, which
  // used to reject this rank while another rank with offset 0 stayed eligible.
  constexpr size_t kBytes = 32;
  constexpr size_t kCount32 = kBytes / sizeof(float);
  install(1, false, NCCL_WIN_COLL_SYMMETRIC, 4096, /*sendLsaMin=*/8, 1, false, NCCL_WIN_COLL_SYMMETRIC,
          kBytes * 8, kBytes * 8);
  void* send = static_cast<char*>(send_) + 4;
  EXPECT_TRUE(eligiblePtrs(send, recv_, kCount32));
}

TEST_F(GinAllGatherEligibilityTest, AcceptsGfx950AndGfx1250) {
  installSingleSegment();
  EXPECT_TRUE(eligible());
  std::memset(mock_.archName_, 0, sizeof(mock_.archName_));
  std::memcpy(mock_.archName_, "gfx1250", 7);
  const char* gfx1250 = std::getenv("NCCL_GIN_ALLGATHER_GFX1250_ENABLE");
  if (gfx1250 != nullptr && std::strcmp(gfx1250, "1") == 0) {
    EXPECT_TRUE(eligible());
  } else {
    EXPECT_FALSE(eligible());
  }
}

TEST_F(GinAllGatherEligibilityTest, RejectsWrongArch) {
  installSingleSegment();
  std::memcpy(mock_.archName_, "gfx942", 6);
  EXPECT_FALSE(eligible());
}

TEST_F(GinAllGatherEligibilityTest, RejectsGroupedAndImplicitLaunch) {
  installSingleSegment();
  const int savedDepth = ncclGroupDepth;
  ncclGroupDepth = 1;
  EXPECT_FALSE(eligible());
  ncclGroupDepth = savedDepth;
  mock_.comm.config.launchOrderImplicit = 1;
  EXPECT_FALSE(eligible());
  mock_.comm.config.launchOrderImplicit = 0;
}

TEST(GinAllGatherCtaCount, ReportsLsaAndSdmaTiers) {
  if (std::getenv("NCCL_GIN_ALLGATHER_SDMA_MIN_BYTES") != nullptr) {
    GTEST_SKIP() << "CTA tiers assume the default NCCL_GIN_ALLGATHER_SDMA_MIN_BYTES";
  }
  // Under 16 KiB the LSA band uses 2 chunks: 8 ranks * 2 = 16 CTAs. At 8 MiB, one SDMA CTA.
  EXPECT_EQ(16, ncclAllGatherGinSdmaCtaCount(8, 1024));
  EXPECT_EQ(1, ncclAllGatherGinSdmaCtaCount(8, 8ull * 1024 * 1024));
  // 16 ranks at 1 MiB ask for 8 chunks, which the 64-CTA LSA barrier pool clamps to 4.
  EXPECT_EQ(64, ncclAllGatherGinSdmaCtaCount(16, 1ull << 20));
}
#else
TEST(GinAllGatherEligibilityTest, SkippedWithoutGinBuild) {
  GTEST_SKIP() << "GIN AllGather eligibility tests require ENABLE_ROCSHMEM_GIN";
}
#endif

} // namespace RcclUnitTesting
