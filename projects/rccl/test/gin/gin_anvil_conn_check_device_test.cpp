/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gin/gin_anvil_conn_check.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

namespace RcclUnitTesting {
namespace {

class GinAnvilConnCheckDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count == 0) {
      GTEST_SKIP() << "GPU required for conn-check kernel test";
    }
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
  }

  void TearDown() override {
    for (void* ptr : allocations_) (void)hipFree(ptr);
    if (stream_) (void)hipStreamDestroy(stream_);
  }

  template <typename T>
  hipError_t Allocate(T** ptr, size_t count) {
    hipError_t result = hipMalloc(ptr, sizeof(T) * count);
    if (result == hipSuccess) allocations_.push_back(*ptr);
    return result;
  }

  hipError_t CreateStream() { return hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking); }

  hipStream_t stream_{nullptr};
  std::vector<void*> allocations_;
};

TEST_F(GinAnvilConnCheckDeviceTest, WritesSelfRankSlotOnEveryPeer) {
  constexpr int kRanks = 2;
  constexpr unsigned long long kStamp = 0xC0FFEE01ULL;
  uint64_t* peer0Signals = nullptr;
  uint64_t* peer1Signals = nullptr;
  uintptr_t* remoteAddrs = nullptr;

  ASSERT_EQ(CreateStream(), hipSuccess);
  ASSERT_EQ(Allocate(&peer0Signals, kRanks), hipSuccess);
  ASSERT_EQ(Allocate(&peer1Signals, kRanks), hipSuccess);
  ASSERT_EQ(Allocate(&remoteAddrs, kRanks), hipSuccess);
  ASSERT_EQ(hipMemsetAsync(peer0Signals, 0, sizeof(uint64_t) * kRanks, stream_), hipSuccess);
  ASSERT_EQ(hipMemsetAsync(peer1Signals, 0, sizeof(uint64_t) * kRanks, stream_), hipSuccess);

  const uintptr_t hostAddrs[kRanks] = {
      reinterpret_cast<uintptr_t>(peer0Signals), reinterpret_cast<uintptr_t>(peer1Signals)};
  ASSERT_EQ(hipMemcpyAsync(remoteAddrs, hostAddrs, sizeof(hostAddrs), hipMemcpyHostToDevice, stream_),
            hipSuccess);

  ASSERT_EQ(ginAnvilConnWrite(remoteAddrs, kRanks, 0, kStamp, stream_), 0);
  uint64_t peer0Host[kRanks] = {};
  uint64_t peer1Host[kRanks] = {};
  ASSERT_EQ(hipMemcpyAsync(peer0Host, peer0Signals, sizeof(peer0Host), hipMemcpyDeviceToHost, stream_),
            hipSuccess);
  ASSERT_EQ(hipMemcpyAsync(peer1Host, peer1Signals, sizeof(peer1Host), hipMemcpyDeviceToHost, stream_),
            hipSuccess);
  ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
  EXPECT_EQ(peer0Host[0], kStamp);
  EXPECT_EQ(peer0Host[1], 0);
  EXPECT_EQ(peer1Host[0], kStamp);
  EXPECT_EQ(peer1Host[1], 0);
}

TEST_F(GinAnvilConnCheckDeviceTest, ReportsAnUnwrittenSourceSlot) {
  constexpr int kRanks = 2;
  constexpr unsigned long long kStamp = 0xC0FFEE02ULL;
  uint64_t* signals = nullptr;
  int* missing = nullptr;

  ASSERT_EQ(CreateStream(), hipSuccess);
  ASSERT_EQ(Allocate(&signals, kRanks), hipSuccess);
  ASSERT_EQ(Allocate(&missing, kRanks), hipSuccess);
  const uint64_t hostSignals[kRanks] = {kStamp, 0};
  ASSERT_EQ(hipMemcpyAsync(signals, hostSignals, sizeof(hostSignals), hipMemcpyHostToDevice, stream_),
            hipSuccess);

  ASSERT_EQ(ginAnvilConnCheck(signals, kRanks, kStamp, missing, stream_), 0);
  int hostMissing[kRanks] = {-1, -1};
  ASSERT_EQ(hipMemcpyAsync(hostMissing, missing, sizeof(hostMissing), hipMemcpyDeviceToHost, stream_),
            hipSuccess);
  ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
  EXPECT_EQ(hostMissing[0], 0);
  EXPECT_EQ(hostMissing[1], 1);
}

TEST(GinAnvilConnCheckDeviceValidationTest, RejectsTooManyRanksBeforeLaunch) {
  EXPECT_EQ(ginAnvilConnWrite(nullptr, 1025, 0, 1, nullptr), -1);
  EXPECT_EQ(ginAnvilConnCheck(nullptr, 1025, 1, nullptr, nullptr), -1);
}

}  // namespace
}  // namespace RcclUnitTesting
