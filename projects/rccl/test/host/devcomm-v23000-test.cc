/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/devcomm/devcomm_v23000.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>

#include <gtest/gtest.h>

#include "comm.h"
#include "dev_runtime.h"

#include DEVCOMM_V23000_CC_PATH

namespace {

class DevcommV23000Microtest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  std::unique_ptr<ncclSharedResources> shared_ = std::make_unique<ncclSharedResources>();
  alignas(uint64_t) std::array<char, 128> storage_{};

  void SetUp() override { comm_->sharedRes = shared_.get(); }

  template <typename DeviceComm>
  void Populate(DeviceComm& device) {
    device.magic = 0x12345678;
    device.version = NCCL_VERSION(2, 30, 3);
    device.rank = 13;
    device.nRanks = 24;
    device.nRanks_rcp32 = 0x12345;
    device.lsaRank = 5;
    device.lsaSize = 8;
    device.lsaSize_rcp32 = 0x54321;
    device.windowTable = reinterpret_cast<ncclDevCommWindowTable_t>(storage_.data());
    device.resourceWindow = reinterpret_cast<ncclWindow_t>(storage_.data() + 8);
    device.resourceWindow_inlined.lsaFlatBase = storage_.data() + 16;
    device.resourceWindow_inlined.stride4G = 0x10203040;
    device.resourceWindow_inlined.mcOffset4K = 0x50607080;
    device.lsaMultimem.mcBasePtr = storage_.data() + 24;
    device.lsaBarrier.bufHandle = 31;
    device.lsaBarrier.nBarriers = 32;
    device.railGinBarrier.signal0 = 41;
    device.railGinBarrier.unused = 42;
    device.ginConnectionCount = NCCL_GIN_MAX_CONNECTIONS;
    device.ginSignalCount = 53;
    device.ginCounterCount = 67;
    device.ginSignalShadows = reinterpret_cast<uint64_t*>(storage_.data() + 64);
    device.ginContextCount = 71;
    device.ginStrongLegacySignals = true;
    device.abortFlag = reinterpret_cast<uint32_t*>(storage_.data() + 80);
    device.hybridLsaBarrier.bufHandle = 81;
    device.hybridLsaBarrier.nBarriers = 82;
    device.hybridRailGinBarrier.signal0 = 91;
    device.hybridRailGinBarrier.unused = 92;
    device.worldGinBarrier.signal0 = 101;
    device.worldGinBarrier.unused = 102;
    for (int connection = 0; connection < NCCL_GIN_MAX_CONNECTIONS; connection++) {
      device.ginNetDeviceTypes[connection] = connection + 3;
      device.ginHandles[connection] = storage_.data() + 32 + connection;
    }
  }

  template <typename Actual, typename Expected>
  void ExpectCommon(const Actual& actual, const Expected& expected) {
    EXPECT_EQ(expected.magic, actual.magic);
    EXPECT_EQ(expected.version, actual.version);
    EXPECT_EQ(expected.rank, actual.rank);
    EXPECT_EQ(expected.nRanks, actual.nRanks);
    EXPECT_EQ(expected.nRanks_rcp32, actual.nRanks_rcp32);
    EXPECT_EQ(expected.lsaRank, actual.lsaRank);
    EXPECT_EQ(expected.lsaSize, actual.lsaSize);
    EXPECT_EQ(expected.lsaSize_rcp32, actual.lsaSize_rcp32);
    EXPECT_EQ(expected.windowTable, actual.windowTable);
    EXPECT_EQ(expected.resourceWindow, actual.resourceWindow);
    EXPECT_EQ(expected.resourceWindow_inlined.lsaFlatBase, actual.resourceWindow_inlined.lsaFlatBase);
    EXPECT_EQ(expected.resourceWindow_inlined.stride4G, actual.resourceWindow_inlined.stride4G);
    EXPECT_EQ(expected.resourceWindow_inlined.mcOffset4K, actual.resourceWindow_inlined.mcOffset4K);
    EXPECT_EQ(expected.lsaMultimem.mcBasePtr, actual.lsaMultimem.mcBasePtr);
    EXPECT_EQ(expected.lsaBarrier.bufHandle, actual.lsaBarrier.bufHandle);
    EXPECT_EQ(expected.lsaBarrier.nBarriers, actual.lsaBarrier.nBarriers);
    EXPECT_EQ(expected.railGinBarrier.signal0, actual.railGinBarrier.signal0);
    EXPECT_EQ(expected.railGinBarrier.unused, actual.railGinBarrier.unused);
    EXPECT_EQ(expected.ginConnectionCount, actual.ginConnectionCount);
    EXPECT_EQ(expected.ginSignalCount, actual.ginSignalCount);
    EXPECT_EQ(expected.ginCounterCount, actual.ginCounterCount);
    EXPECT_EQ(expected.ginSignalShadows, actual.ginSignalShadows);
    EXPECT_EQ(expected.ginContextCount, actual.ginContextCount);
    EXPECT_EQ(expected.ginStrongLegacySignals, actual.ginStrongLegacySignals);
    EXPECT_EQ(expected.abortFlag, actual.abortFlag);
    EXPECT_EQ(expected.hybridLsaBarrier.bufHandle, actual.hybridLsaBarrier.bufHandle);
    EXPECT_EQ(expected.hybridLsaBarrier.nBarriers, actual.hybridLsaBarrier.nBarriers);
    EXPECT_EQ(expected.hybridRailGinBarrier.signal0, actual.hybridRailGinBarrier.signal0);
    EXPECT_EQ(expected.hybridRailGinBarrier.unused, actual.hybridRailGinBarrier.unused);
    EXPECT_EQ(expected.worldGinBarrier.signal0, actual.worldGinBarrier.signal0);
    EXPECT_EQ(expected.worldGinBarrier.unused, actual.worldGinBarrier.unused);
    for (int connection = 0; connection < NCCL_GIN_MAX_CONNECTIONS; connection++) {
      SCOPED_TRACE(connection);
      EXPECT_EQ(expected.ginNetDeviceTypes[connection], actual.ginNetDeviceTypes[connection]);
      EXPECT_EQ(expected.ginHandles[connection], actual.ginHandles[connection]);
    }
  }
};

TEST_F(DevcommV23000Microtest, Compat_ExportsVersionRangeAndCallbacks) {
  EXPECT_EQ(NCCL_VERSION(2, 30, 0), ncclDevCommCompat_v23000.minVersion);
  EXPECT_EQ(NCCL_VERSION(2, 30, 7), ncclDevCommCompat_v23000.maxVersion);
  EXPECT_EQ(nullptr, ncclDevCommCompat_v23000.commPropertiesFilter);
  EXPECT_EQ(ncclDevCommRequirementsFilter_v23000, ncclDevCommCompat_v23000.devCommRequirementsFilter);
  EXPECT_EQ(ncclDevCommCopyNewToOld_v23000, ncclDevCommCompat_v23000.devCommCopyNewToOld);
  EXPECT_EQ(ncclDevCommCopyOldToNew_v23000, ncclDevCommCompat_v23000.devCommCopyOldToNew);
}

TEST_F(DevcommV23000Microtest, RequirementsFilter_UnfinalizedUsesFirstBackend) {
  ncclDevCommRequirements_t requirements{};
  requirements.ginContextCount = 19;
  shared_->ginState.backends[0].ginType = NCCL_GIN_TYPE_PROXY;
  shared_->ginState.backends[1].ginType = NCCL_GIN_TYPE_NONE;

  ASSERT_EQ(ncclSuccess, ncclDevCommCompat_v23000.devCommRequirementsFilter(comm_.get(), &requirements));

  EXPECT_EQ(NCCL_GIN_TYPE_PROXY, requirements.ginType);
  EXPECT_EQ(19, requirements.ginContextCount);
}

TEST_F(DevcommV23000Microtest, RequirementsFilter_FinalizedDisablesGin) {
  ncclDevCommRequirements_t requirements{};
  requirements.ginType = NCCL_GIN_TYPE_PROXY;
  shared_->ginState.backends[0].ginType = NCCL_GIN_TYPE_PROXY;
  shared_->ginState.finalized = true;

  ASSERT_EQ(ncclSuccess, ncclDevCommRequirementsFilter_v23000(comm_.get(), &requirements));

  EXPECT_EQ(NCCL_GIN_TYPE_NONE, requirements.ginType);
}

TEST_F(DevcommV23000Microtest, CopyNewToOld_CopiesEveryFieldAndClearsReservedStorage) {
  ncclDevComm source{};
  Populate(source);
  source.hybridDenseGinBarrier = {111, 112};
  source.ginConnectionStride = 8;
  source.ginContextStride = 1;
  struct Guarded {
    std::array<unsigned char, 16> before;
    ncclDevComm_v23000 device;
    std::array<unsigned char, 16> after;
  } destination;
  std::memset(&destination, 0x5a, sizeof(destination));

  ASSERT_EQ(ncclSuccess, ncclDevCommCompat_v23000.devCommCopyNewToOld(comm_.get(), &destination.device, &source));

  ExpectCommon(destination.device, source);
  EXPECT_EQ(111u, destination.device.hybridWorldGinBarrier.signal0);
  EXPECT_EQ(112u, destination.device.hybridWorldGinBarrier.unused);
  EXPECT_TRUE(destination.device.ginConnectionsRailed);
  EXPECT_FALSE(destination.device.ginContextsRailed);
  for (char byte : destination.device.resourceWindow_inlined.reserved1) EXPECT_EQ(0, byte);
  for (char byte : destination.device.resourceWindow_inlined.reserved2) EXPECT_EQ(0, byte);
  for (char byte : destination.device.resourceWindow_inlined.reserved3) EXPECT_EQ(0, byte);
  for (auto byte : destination.before) EXPECT_EQ(0x5a, byte);
  for (auto byte : destination.after) EXPECT_EQ(0x5a, byte);
}

TEST_F(DevcommV23000Microtest, CopyOldToNew_CopiesEveryFieldAndPreservesNewOnlyFields) {
  ncclDevComm_v23000 source{};
  Populate(source);
  source.hybridWorldGinBarrier = {113, 114};
  source.ginConnectionsRailed = false;
  source.ginContextsRailed = true;
  ncclDevComm destination{};
  std::memset(&destination, 0x5a, sizeof(destination));
  destination.backendIndex = 3;
  constexpr size_t tailOffset = offsetof(ncclDevComm, ginConnectionStride_rcp32);
  std::array<unsigned char, sizeof(ncclDevComm) - tailOffset> savedTail{};
  std::memcpy(savedTail.data(), reinterpret_cast<const unsigned char*>(&destination) + tailOffset, savedTail.size());

  ASSERT_EQ(ncclSuccess, ncclDevCommCompat_v23000.devCommCopyOldToNew(comm_.get(), &destination, &source));

  ExpectCommon(destination, source);
  EXPECT_EQ(113u, destination.hybridDenseGinBarrier.signal0);
  EXPECT_EQ(114u, destination.hybridDenseGinBarrier.unused);
  EXPECT_EQ(1, destination.ginConnectionStride);
  EXPECT_EQ(8, destination.ginContextStride);
  EXPECT_EQ(3, destination.backendIndex);
  EXPECT_EQ(0, std::memcmp(savedTail.data(), reinterpret_cast<const unsigned char*>(&destination) + tailOffset,
                           savedTail.size()));
}

TEST_F(DevcommV23000Microtest, CopyNewToOld_StridesIndependentlyControlRailedFlags) {
  for (int connectionStride : {1, 2, 8}) {
    for (int contextStride : {1, 2, 8}) {
      SCOPED_TRACE(::testing::Message() << connectionStride << ',' << contextStride);
      ncclDevComm source{};
      source.ginConnectionStride = connectionStride;
      source.ginContextStride = contextStride;
      ncclDevComm_v23000 destination{};
      ASSERT_EQ(ncclSuccess, ncclDevCommCopyNewToOld_v23000(comm_.get(), &destination, &source));
      EXPECT_EQ(connectionStride != 1, destination.ginConnectionsRailed);
      EXPECT_EQ(contextStride != 1, destination.ginContextsRailed);
      EXPECT_FALSE(destination.ginStrongLegacySignals);
    }
  }
}

TEST_F(DevcommV23000Microtest, CopyOldToNew_RailedFlagsIndependentlySelectLsaSize) {
  for (bool connectionsRailed : {false, true}) {
    for (bool contextsRailed : {false, true}) {
      SCOPED_TRACE(::testing::Message() << connectionsRailed << ',' << contextsRailed);
      ncclDevComm_v23000 source{};
      source.lsaSize = 5;
      source.ginConnectionsRailed = connectionsRailed;
      source.ginContextsRailed = contextsRailed;
      ncclDevComm destination{};
      ASSERT_EQ(ncclSuccess, ncclDevCommCopyOldToNew_v23000(comm_.get(), &destination, &source));
      EXPECT_EQ(connectionsRailed ? 5 : 1, destination.ginConnectionStride);
      EXPECT_EQ(contextsRailed ? 5 : 1, destination.ginContextStride);
      EXPECT_FALSE(destination.ginStrongLegacySignals);
    }
  }
}

}  // namespace
