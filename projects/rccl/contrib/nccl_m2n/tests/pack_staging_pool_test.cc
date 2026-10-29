/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/* Resource, configuration, and ordering coverage for the bounded
 * PACK staging pool in src/pack_staging.cc. */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

#include <cuda_runtime.h>
#include <nccl.h>

#include "reshard_internal.h"
#include "test_helpers.h"

namespace {

#define ASSERT_CUDA(call) ASSERT_EQ(cudaSuccess, (call))

constexpr size_t kSlotBytes = 4096;
constexpr size_t kUnsatisfiableBytes = static_cast<size_t>(1) << 44;

void CUDART_CB waitForSlotRelease(void* data) {
  auto* release = static_cast<std::atomic<bool>*>(data);
  while (!release->load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

class PackStagingPoolTest : public ::testing::Test {
protected:
  ncclComm_t comm = nullptr;
  cudaStream_t stream = nullptr;
  int savedBucketCount = 0;
  bool savedImplicitDefault = true;
  bool hadSavedProfile = false;
  std::string savedProfile;
  ReshardStagingBucketCfg savedBuckets[kMaxStagingBuckets] = {};
  ReshardCopyAlgorithm savedCopyAlgorithm = RESHARD_COPY_ALGO_PACK;
  int savedCommBConcurrency = 0;

  void SetUp() override {
    ASSERT_CUDA(cudaSetDevice(0));
    int devices[1] = {0};
    ASSERT_NCCL(ncclCommInitAll(&comm, 1, devices), comm);
    ASSERT_CUDA(cudaStreamCreate(&stream));

    packStagingFinalize();
    reshardClearResourceQuarantine();
    savedBucketCount = gReshardStagingBucketCount;
    savedImplicitDefault = gReshardStagingBucketsImplicitDefault;
    savedCopyAlgorithm = gReshardCopyAlgorithm;
    savedCommBConcurrency = gReshardCommBConcurrency;
    const char* profile = getenv("NCCL_RESHARD_PACK_BUFFSIZES");
    if (profile != nullptr) {
      hadSavedProfile = true;
      savedProfile = profile;
    }
    unsetenv("NCCL_RESHARD_PACK_BUFFSIZES");
    for (int i = 0; i < kMaxStagingBuckets; i++) {
      savedBuckets[i] = gReshardStagingBuckets[i];
      gReshardStagingBuckets[i] = {};
    }
    gReshardStagingBuckets[0] = {kSlotBytes, 1};
    gReshardStagingBucketCount = 1;
    gReshardStagingBucketsImplicitDefault = false;
  }

  void TearDown() override {
    packStagingFinalize();
    reshardClearResourceQuarantine();
    gReshardStagingBucketCount = savedBucketCount;
    gReshardStagingBucketsImplicitDefault = savedImplicitDefault;
    gReshardCopyAlgorithm = savedCopyAlgorithm;
    gReshardCommBConcurrency = savedCommBConcurrency;
    for (int i = 0; i < kMaxStagingBuckets; i++) {
      gReshardStagingBuckets[i] = savedBuckets[i];
    }
    if (hadSavedProfile) setenv("NCCL_RESHARD_PACK_BUFFSIZES", savedProfile.c_str(), 1);
    else unsetenv("NCCL_RESHARD_PACK_BUFFSIZES");
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (comm != nullptr) {
      ASSERT_NCCL(ncclCommDestroy(comm));
    }
  }

  void enableSingleSlotBucket() {
    gReshardStagingBuckets[0] = {kSlotBytes, 1};
    gReshardStagingBucketCount = 1;
    gReshardStagingBucketsImplicitDefault = false;
  }
};

TEST_F(PackStagingPoolTest, ParsesConcisePackBufferProfile) {
  ASSERT_EQ(0, setenv("NCCL_RESHARD_PACK_BUFFSIZES", "8k:2,1m", 1));
  applyReshardEnv();
  ASSERT_EQ(2, gReshardStagingBucketCount);
  EXPECT_EQ(8U * 1024U, gReshardStagingBuckets[0].size);
  EXPECT_EQ(2, gReshardStagingBuckets[0].numSlots);
  EXPECT_EQ(1024U * 1024U, gReshardStagingBuckets[1].size);
  EXPECT_EQ(1, gReshardStagingBuckets[1].numSlots);
  EXPECT_FALSE(gReshardStagingBucketsImplicitDefault);
}

TEST_F(PackStagingPoolTest, RejectsProfileBelowMinimumUsableSize) {
  ASSERT_EQ(0, setenv("NCCL_RESHARD_PACK_BUFFSIZES", "1k", 1));
  applyReshardEnv();
  ASSERT_EQ(1, gReshardStagingBucketCount);
  EXPECT_EQ(kDefaultStagingBucketBytes, gReshardStagingBuckets[0].size);
  EXPECT_EQ(kDefaultStagingBucketSlots, gReshardStagingBuckets[0].numSlots);
  EXPECT_TRUE(gReshardStagingBucketsImplicitDefault);
}

TEST_F(PackStagingPoolTest, AllocatesOnlySelectedSlotsLazily) {
  gReshardStagingBuckets[0] = {kSlotBytes, 4};
  gReshardStagingBucketCount = 1;
  gReshardStagingBucketsImplicitDefault = false;

  EXPECT_EQ(0, packStagingAllocationCountForTest());
  ASSERT_NCCL(ensurePackStagingBuffer(comm, kSlotBytes, stream));
  EXPECT_EQ(1, packStagingAllocationCountForTest());
  ASSERT_NCCL(packStagingRecordEvent(comm, stream));

  ASSERT_NCCL(ensurePackStagingBuffer(comm, kSlotBytes, stream));
  EXPECT_EQ(1, packStagingAllocationCountForTest());
  ASSERT_NCCL(packStagingRecordEvent(comm, stream));
}

TEST_F(PackStagingPoolTest, SplitResourceLaneCountUsesPhysicalPackSlotsAndPipeConcurrency) {
  gReshardStagingBuckets[0] = {kSlotBytes, 2};
  gReshardStagingBuckets[1] = {2 * kSlotBytes, 3};
  gReshardStagingBucketCount = 2;

  gReshardCopyAlgorithm = RESHARD_COPY_ALGO_PACK;
  EXPECT_EQ(5, reshardGetSplitSlotCount());

  gReshardCopyAlgorithm = RESHARD_COPY_ALGO_PIPE;
  gReshardCommBConcurrency = 7;
  EXPECT_EQ(7, reshardGetSplitSlotCount());
}

TEST_F(PackStagingPoolTest, MultiBucketPackDisablesSplitEvenWithAdaptiveOverride) {
  const bool adaptive = gReshardAdaptiveCallConfigValid;
  const ReshardAdaptiveCallConfig savedAdaptive = gReshardAdaptiveCallConfig;
  gReshardAdaptiveCallConfigValid = true;
  gReshardAdaptiveCallConfig.splitComm = true;
  gReshardCopyAlgorithm = RESHARD_COPY_ALGO_PACK;
  gReshardStagingBucketCount = 1;
  EXPECT_TRUE(reshardGetSplitCommEnabled());
  gReshardStagingBucketCount = 2;
  EXPECT_FALSE(reshardGetSplitCommEnabled());
  gReshardCopyAlgorithm = RESHARD_COPY_ALGO_PIPE;
  EXPECT_TRUE(reshardGetSplitCommEnabled());
  gReshardAdaptiveCallConfigValid = adaptive;
  gReshardAdaptiveCallConfig = savedAdaptive;
}

TEST_F(PackStagingPoolTest, AllocationFailureDoesNotCommitMapping) {
  gReshardStagingBuckets[0] = {kSlotBytes, 1};
  gReshardStagingBuckets[1] = {kUnsatisfiableBytes, 1};
  gReshardStagingBucketCount = 2;
  gReshardStagingBucketsImplicitDefault = false;

  EXPECT_NE(ncclSuccess, ensurePackStagingBuffer(comm, kUnsatisfiableBytes, stream));
  EXPECT_EQ(nullptr, getPackStagingBuffer(comm));
  EXPECT_EQ(0U, getPackStagingCapacity(comm));
  EXPECT_EQ(0, packStagingAllocationCountForTest());

  ASSERT_NCCL(ensurePackStagingBuffer(comm, kSlotBytes, stream));
  EXPECT_NE(nullptr, getPackStagingBuffer(comm));
  EXPECT_EQ(kSlotBytes, getPackStagingCapacity(comm));
  EXPECT_EQ(1, packStagingAllocationCountForTest());
  ASSERT_NCCL(packStagingRecordEvent(comm, stream));
}

TEST_F(PackStagingPoolTest, ImplicitDefaultOversizeReportsSizingGuidance) {
  constexpr size_t kLargerRequest = 1U << 20;
  gReshardStagingBuckets[0] = {kSlotBytes, kDefaultStagingBucketSlots};
  gReshardStagingBucketCount = 1;
  gReshardStagingBucketsImplicitDefault = true;

  m2nClearLastError();
  EXPECT_EQ(ncclInvalidArgument, ensurePackStagingBuffer(comm, kLargerRequest, stream));
  const std::string detail = ncclM2nGetLastError();
  EXPECT_NE(std::string::npos, detail.find("implicit default"));
  EXPECT_NE(std::string::npos, detail.find("NCCL_RESHARD_PACK_BUFFSIZES="));
  EXPECT_NE(std::string::npos, detail.find(std::to_string(kLargerRequest)));
  EXPECT_EQ(0, packStagingAllocationCountForTest());
}

TEST_F(PackStagingPoolTest, AssignsStableRoundRobinLanesAcrossCommunicators) {
  gReshardStagingBuckets[0] = {kSlotBytes, 2};
  gReshardStagingBucketCount = 1;

  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ncclComm_t comm3 = nullptr;
  ncclComm_t comm4 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices), comm2);
  ASSERT_NCCL(ncclCommInitAll(&comm3, 1, devices), comm3);
  ASSERT_NCCL(ncclCommInitAll(&comm4, 1, devices), comm4);

  ASSERT_NCCL(ensurePackStagingBuffer(comm, kSlotBytes, stream));
  void* slot0 = getPackStagingBuffer(comm);
  EXPECT_EQ(0, getStagingPhysicalSlotIndex(comm));
  ASSERT_NCCL(packStagingRecordEvent(comm, stream));
  ASSERT_NCCL(ensurePackStagingBuffer(comm2, kSlotBytes, stream));
  void* slot1 = getPackStagingBuffer(comm2);
  EXPECT_EQ(1, getStagingPhysicalSlotIndex(comm2));
  ASSERT_NCCL(packStagingRecordEvent(comm2, stream));
  ASSERT_NE(slot0, slot1);

  ASSERT_NCCL(ensurePackStagingBuffer(comm3, kSlotBytes, stream));
  EXPECT_EQ(slot0, getPackStagingBuffer(comm3));
  EXPECT_EQ(0, getStagingPhysicalSlotIndex(comm3));
  ASSERT_NCCL(packStagingRecordEvent(comm3, stream));
  ASSERT_NCCL(ensurePackStagingBuffer(comm4, kSlotBytes, stream));
  EXPECT_EQ(slot1, getPackStagingBuffer(comm4));
  EXPECT_EQ(1, getStagingPhysicalSlotIndex(comm4));
  ASSERT_NCCL(packStagingRecordEvent(comm4, stream));

  ASSERT_NCCL(ensurePackStagingBuffer(comm3, kSlotBytes, stream));
  EXPECT_EQ(slot0, getPackStagingBuffer(comm3));
  EXPECT_EQ(0, getStagingPhysicalSlotIndex(comm3));
  ASSERT_NCCL(packStagingRecordEvent(comm3, stream));

  packStagingFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
  ASSERT_NCCL(ncclCommDestroy(comm3));
  ASSERT_NCCL(ncclCommDestroy(comm4));
}

TEST_F(PackStagingPoolTest, PhysicalSlotIdentityIsUniqueAcrossBuckets) {
  gReshardStagingBuckets[0] = {kSlotBytes, 2};
  gReshardStagingBuckets[1] = {2 * kSlotBytes, 2};
  gReshardStagingBucketCount = 2;

  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ncclComm_t comm3 = nullptr;
  ncclComm_t comm4 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices));
  ASSERT_NCCL(ncclCommInitAll(&comm3, 1, devices));
  ASSERT_NCCL(ncclCommInitAll(&comm4, 1, devices));

  ASSERT_NCCL(ensurePackStagingBuffer(comm, kSlotBytes, stream));
  EXPECT_EQ(0, getStagingPhysicalSlotIndex(comm));
  ASSERT_NCCL(packStagingRecordEvent(comm, stream));
  ASSERT_NCCL(ensurePackStagingBuffer(comm2, 2 * kSlotBytes, stream));
  EXPECT_EQ(2, getStagingPhysicalSlotIndex(comm2));
  ASSERT_NCCL(packStagingRecordEvent(comm2, stream));
  ASSERT_NCCL(ensurePackStagingBuffer(comm3, kSlotBytes, stream));
  EXPECT_EQ(1, getStagingPhysicalSlotIndex(comm3));
  ASSERT_NCCL(packStagingRecordEvent(comm3, stream));
  ASSERT_NCCL(ensurePackStagingBuffer(comm4, 2 * kSlotBytes, stream));
  EXPECT_EQ(3, getStagingPhysicalSlotIndex(comm4));
  ASSERT_NCCL(packStagingRecordEvent(comm4, stream));

  packStagingFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
  ASSERT_NCCL(ncclCommDestroy(comm3));
  ASSERT_NCCL(ncclCommDestroy(comm4));
}

TEST_F(PackStagingPoolTest, RejectsOverlappingUsersOfOnePhysicalLane) {
  enableSingleSlotBucket();
  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices), comm2);

  std::atomic<bool> slotReady{false};
  std::atomic<bool> releaseOwner{false};
  cudaError_t ownerCudaResult = cudaSuccess;
  ncclResult_t ownerAcquireResult = ncclSuccess;
  ncclResult_t ownerRecordResult = ncclSuccess;
  std::thread owner([&] {
    ownerCudaResult = cudaSetDevice(0);
    if (ownerCudaResult != cudaSuccess) return;
    M2nApiLock apiLock;
    ownerAcquireResult = ensurePackStagingBuffer(comm, kSlotBytes, stream);
    slotReady.store(true, std::memory_order_release);
    {
      M2nApiUnlock apiUnlock;
      while (!releaseOwner.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    }
    if (ownerAcquireResult == ncclSuccess) {
      ownerRecordResult = packStagingRecordEvent(comm, stream);
    }
  });

  while (!slotReady.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  ncclResult_t overlappingResult = ncclSuccess;
  {
    M2nApiLock apiLock;
    overlappingResult = ensurePackStagingBuffer(comm2, kSlotBytes, stream);
  }
  releaseOwner.store(true, std::memory_order_release);
  owner.join();

  EXPECT_EQ(cudaSuccess, ownerCudaResult);
  EXPECT_EQ(ncclSuccess, ownerAcquireResult);
  EXPECT_EQ(ncclInvalidUsage, overlappingResult);
  EXPECT_EQ(ncclSuccess, ownerRecordResult);
  ASSERT_NCCL(ensurePackStagingBuffer(comm2, kSlotBytes, stream));
  ASSERT_NCCL(packStagingRecordEvent(comm2, stream));

  packStagingFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
}

TEST_F(PackStagingPoolTest, WaitsForPriorCommunicatorOnAnotherStream) {
  enableSingleSlotBucket();
  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices), comm2);

  ASSERT_NCCL(ensurePackStagingBuffer(comm, kSlotBytes, nullptr));
  cudaStream_t nonblocking = nullptr;
  cudaEvent_t marker = nullptr;
  ASSERT_CUDA(cudaStreamCreateWithFlags(&nonblocking, cudaStreamNonBlocking));
  ASSERT_CUDA(cudaEventCreateWithFlags(&marker, cudaEventDisableTiming));

  std::atomic<bool> release{false};
  ASSERT_CUDA(cudaLaunchHostFunc(nullptr, waitForSlotRelease, &release));
  ASSERT_NCCL(packStagingRecordEvent(comm, nullptr));
  ASSERT_NCCL(ensurePackStagingBuffer(comm2, kSlotBytes, nonblocking));
  ASSERT_CUDA(cudaEventRecord(marker, nonblocking));
  ASSERT_NCCL(packStagingRecordEvent(comm2, nonblocking));

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_EQ(cudaErrorNotReady, cudaEventQuery(marker));
  release.store(true, std::memory_order_release);
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(nonblocking));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(nullptr));

  EXPECT_EQ(cudaSuccess, cudaEventDestroy(marker));
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(nonblocking));
  packStagingFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
}

TEST_F(PackStagingPoolTest, KeepsHostRmaWarmupStatePerCommunicator) {
  enableSingleSlotBucket();
  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices), comm2);

  ASSERT_NCCL(ensurePackStagingBuffer(comm, kSlotBytes, stream));
  void* sharedSlot = getPackStagingBuffer(comm);
  ASSERT_NCCL(packStagingRecordEvent(comm, stream));
  ASSERT_NCCL(ensurePackStagingBuffer(comm2, kSlotBytes, stream));
  EXPECT_EQ(sharedSlot, getPackStagingBuffer(comm2));
  ASSERT_NCCL(packStagingRecordEvent(comm2, stream));

  ASSERT_NCCL(setPackRmaWarmed(comm, true));
  ASSERT_NCCL(setPackRmaWarmed(comm2, false));
  bool warmed = false;
  ASSERT_NCCL(getPackRmaWarmed(comm, &warmed));
  EXPECT_TRUE(warmed);
  ASSERT_NCCL(getPackRmaWarmed(comm2, &warmed));
  EXPECT_FALSE(warmed);

  packStagingFinalize();
  ASSERT_NCCL(ncclCommDestroy(comm2));
}

TEST_F(PackStagingPoolTest, NewMappingsSkipPoisonedLanes) {
  gReshardStagingBuckets[0] = {kSlotBytes, 2};
  gReshardStagingBucketCount = 1;
  int devices[1] = {0};
  ncclComm_t comm2 = nullptr;
  ncclComm_t comm3 = nullptr;
  ASSERT_NCCL(ncclCommInitAll(&comm2, 1, devices), comm2);
  ASSERT_NCCL(ncclCommInitAll(&comm3, 1, devices), comm3);

  ASSERT_NCCL(ensurePackStagingBuffer(comm, kSlotBytes, stream));
  reshardFailNextCompletionEventRecordForTest(/*bFailStreamSynchronize=*/true);
  EXPECT_EQ(ncclSystemError, packStagingRecordEvent(comm, stream));
  EXPECT_EQ(ncclUnhandledCudaError, ensurePackStagingBuffer(comm, kSlotBytes, stream));

  ASSERT_NCCL(ensurePackStagingBuffer(comm2, kSlotBytes, stream));
  void* healthySlot = getPackStagingBuffer(comm2);
  ASSERT_NCCL(packStagingRecordEvent(comm2, stream));
  ASSERT_NCCL(ensurePackStagingBuffer(comm3, kSlotBytes, stream));
  EXPECT_EQ(healthySlot, getPackStagingBuffer(comm3));
  ASSERT_NCCL(packStagingRecordEvent(comm3, stream));

  packStagingFinalize();
  reshardClearResourceQuarantine();
  ASSERT_NCCL(ncclCommDestroy(comm2));
  ASSERT_NCCL(ncclCommDestroy(comm3));
}

} // namespace
