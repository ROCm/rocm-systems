/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define NCCL_HOSTLIB_ONLY
#include "nccl_device.h"
#undef NCCL_HOSTLIB_ONLY
#include "reshard_internal.h"
#include "reshard_call_setup.h"
#include "reshard_split.h"

namespace {

class DevCommCacheTest : public ::testing::Test {
protected:
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
  int savedCapacity = 0;
  std::string savedEnv;
  bool hadEnv = false;

  void SetUp() override {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    packStagingSynchronize();
    reshardSplitSynchronizeCommBLanes();
    cacheFinalize();
    reshardSplitCommFinalize();
    reshardClearResourceQuarantine();
    savedCapacity = gReshardDevCommCacheCapacity;
    const char* value = getenv("NCCL_RESHARD_DEVCOMM_CACHE_CAPACITY");
    if (value != nullptr) { hadEnv = true; savedEnv = value; }
    gReshardDevCommCacheCapacity = 2;
    int devices[] = {0};
    ASSERT_EQ(ncclSuccess, ncclCommInitAll(&comm, 1, devices));
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  void TearDown() override {
    reshardSplitSynchronizeCommBLanes();
    cacheFinalize();
    reshardSplitCommFinalize();
    reshardClearResourceQuarantine();
    gReshardDevCommCacheCapacity = savedCapacity;
    if (hadEnv) setenv("NCCL_RESHARD_DEVCOMM_CACHE_CAPACITY", savedEnv.c_str(), 1);
    else unsetenv("NCCL_RESHARD_DEVCOMM_CACHE_CAPACITY");
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (comm != nullptr) ncclCommDestroy(comm);
  }

  ncclResult_t acquire(int contexts) {
    ncclDevComm dev{};
    ReshardDevCommUse use;
    ncclResult_t result = reshardGetOrCreateDevComm(comm, 0, 0, 0, RESHARD_DEVCOMM_BARRIER_NONE,
                                                    contexts, stream, &dev, &use);
    return result == ncclSuccess ? reshardRecordDevCommUse(&use, stream) : result;
  }
};

TEST_F(DevCommCacheTest, CapacityFailurePreservesExistingEntriesAndHitsStillWork) {
  ASSERT_EQ(ncclSuccess, acquire(1));
  ASSERT_EQ(ncclSuccess, acquire(2));
  const ReshardDevCommCacheKey first{comm, 0, 0, 0, 1, NCCL_GIN_CONNECTION_FULL, RESHARD_DEVCOMM_BARRIER_NONE};
  const ReshardDevCommCacheKey second{comm, 0, 0, 0, 2, NCCL_GIN_CONNECTION_FULL, RESHARD_DEVCOMM_BARRIER_NONE};
  ncclDevComm* firstEntry = findCachedDevComm(first);
  ncclDevComm* secondEntry = findCachedDevComm(second);
  ASSERT_NE(nullptr, firstEntry);
  ASSERT_NE(nullptr, secondEntry);
  EXPECT_EQ(ncclInvalidUsage, acquire(3));
  const std::string error = ncclM2nGetLastError();
  EXPECT_NE(std::string::npos, error.find("runtime eviction is disabled"));
  EXPECT_NE(std::string::npos, error.find("NCCL_RESHARD_DEVCOMM_CACHE_CAPACITY"));
  EXPECT_EQ(firstEntry, findCachedDevComm(first));
  EXPECT_EQ(secondEntry, findCachedDevComm(second));
  EXPECT_EQ(ncclSuccess, acquire(1));
  EXPECT_EQ(ncclSuccess, acquire(2));
}

TEST_F(DevCommCacheTest, PendingReservationsConsumeCapacityAndReleaseOnFailure) {
  ReshardDevCommCacheReservation first;
  ASSERT_EQ(ncclSuccess, reserveDevCommCacheEntry(&first));
  int released = -1;
  {
    ReshardDevCommCacheReservation pending;
    ASSERT_EQ(ncclSuccess, reserveDevCommCacheEntry(&pending));
    released = pending.index;
    EXPECT_NE(first.index, pending.index);
    ReshardDevCommCacheReservation rejected;
    EXPECT_EQ(ncclInvalidUsage, reserveDevCommCacheEntry(&rejected));
    EXPECT_EQ(-1, rejected.index);
  }
  ReshardDevCommCacheReservation retry;
  ASSERT_EQ(ncclSuccess, reserveDevCommCacheEntry(&retry));
  EXPECT_EQ(released, retry.index);
}

TEST_F(DevCommCacheTest, EnvironmentCanIncreaseCapacityWithoutChangingSplitLimits) {
  ASSERT_EQ(0, setenv("NCCL_RESHARD_DEVCOMM_CACHE_CAPACITY", "128", 1));
  applyReshardEnv();
  ASSERT_EQ(128, gReshardDevCommCacheCapacity);
  EXPECT_EQ(64, MAX_DEVCOMM_CACHE_ENTRIES);
  std::vector<std::unique_ptr<ReshardDevCommCacheReservation>> entries;
  for (int i = 0; i < 128; i++) {
    auto entry = std::make_unique<ReshardDevCommCacheReservation>();
    ASSERT_EQ(ncclSuccess, reserveDevCommCacheEntry(entry.get()));
    entries.push_back(std::move(entry));
  }
  ReshardDevCommCacheReservation rejected;
  EXPECT_EQ(ncclInvalidUsage, reserveDevCommCacheEntry(&rejected));
}

TEST_F(DevCommCacheTest, AllocatedCapacityDoesNotChangeWithinEpoch) {
  gReshardDevCommCacheCapacity = 1;
  ReshardDevCommCacheReservation first;
  ASSERT_EQ(ncclSuccess, reserveDevCommCacheEntry(&first));
  gReshardDevCommCacheCapacity = 128;
  ReshardDevCommCacheReservation rejected;
  EXPECT_EQ(ncclInvalidUsage, reserveDevCommCacheEntry(&rejected));
}

TEST_F(DevCommCacheTest, InvalidCapacityEnvironmentDoesNotReplaceCurrentValue) {
  for (const char* value : {"0", "-1", "garbage", "2147483648"}) {
    ASSERT_EQ(0, setenv("NCCL_RESHARD_DEVCOMM_CACHE_CAPACITY", value, 1));
    applyReshardEnv();
    EXPECT_EQ(2, gReshardDevCommCacheCapacity);
  }
}

struct ReleaseGate {
  cudaStream_t stream = nullptr;
  std::atomic<bool> open{false};
  ~ReleaseGate() {
    open.store(true, std::memory_order_release);
    cudaStreamSynchronize(stream);
  }
};

void CUDART_CB waitForGate(void* data) {
  auto* gate = static_cast<ReleaseGate*>(data);
  while (!gate->open.load(std::memory_order_acquire)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

TEST_F(DevCommCacheTest, TeardownDrainWaitsForAllExistingLaneEvents) {
  cudaEvent_t blocked = nullptr, complete = nullptr;
  ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&blocked, cudaEventDisableTiming));
  ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&complete, cudaEventDisableTiming));
  reshardSplitSetLaneCompletionForTest(0, blocked);
  reshardSplitSetLaneCompletionForTest(1, complete);
  ReleaseGate gate;
  gate.stream = stream;
  ASSERT_EQ(cudaSuccess, cudaLaunchHostFunc(stream, waitForGate, &gate));
  ASSERT_EQ(cudaSuccess, cudaEventRecord(blocked, stream));
  ASSERT_EQ(cudaSuccess, cudaEventRecord(complete, nullptr));
  std::atomic<bool> started{false}, drained{false};
  std::thread waiter([&] {
    cudaSetDevice(0);
    started.store(true, std::memory_order_release);
    reshardSplitSynchronizeCommBLanes();
    drained.store(true, std::memory_order_release);
  });
  while (!started.load(std::memory_order_acquire)) std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(drained.load(std::memory_order_acquire));
  gate.open.store(true, std::memory_order_release);
  waiter.join();
  EXPECT_TRUE(drained.load(std::memory_order_acquire));
  EXPECT_FALSE(reshardResourcesNeedQuarantine());
}

TEST_F(DevCommCacheTest, MultiBucketFallbackReturnsBeforeSplitSetup) {
  const ReshardCopyAlgorithm copy = gReshardCopyAlgorithm;
  const int buckets = gReshardStagingBucketCount;
  gReshardCopyAlgorithm = RESHARD_COPY_ALGO_PACK;
  gReshardStagingBucketCount = 2;
  int dims[] = {1, 1};
  ncclMesh_t src = NCCL_M2N_MESH_INITIALIZER, dst = NCCL_M2N_MESH_INITIALIZER;
  src.ndims = dst.ndims = 2;
  src.dims = dst.dims = dims;
  dst.startRank = 1;
  ReshardSplitComms split{};
  EXPECT_EQ(ncclSuccess, reshardGetOrCreateSplitComms(comm, &src, &dst, 1, 1, false, 1, stream, &split));
  EXPECT_FALSE(split.active);
  EXPECT_EQ(nullptr, split.commA);
  EXPECT_EQ(nullptr, split.commB);
  gReshardCopyAlgorithm = copy;
  gReshardStagingBucketCount = buckets;
}

} // namespace
