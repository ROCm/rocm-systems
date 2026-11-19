/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>

#include <cstdint>
#include <vector>

HIP_TEST_CASE(Unit_hipExtMallocWithFlags_Positive_Basic) {
  void* ptr = nullptr;

  SECTION("hipDeviceMallocDefault") {
    const auto alloc_size =
        GENERATE_COPY(10, kPageSize / 2, kPageSize, kPageSize * 3 / 2, kPageSize * 2);
    HIP_CHECK(hipExtMallocWithFlags(&ptr, alloc_size, hipDeviceMallocDefault));
    CHECK(ptr != nullptr);
    CHECK(reinterpret_cast<intptr_t>(ptr) % 256 == 0);
    HIP_CHECK(hipFree(ptr));
  }

  SECTION("hipDeviceMallocFinegrained") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeFineGrainSupport)) {
      WARN("Skipping section: " << HipTest::SkipReason::kFineGrainHwUnsupported);
      return;
    }
    const auto alloc_size =
        GENERATE_COPY(10, kPageSize / 2, kPageSize, kPageSize * 3 / 2, kPageSize * 2);
    HIP_CHECK(hipExtMallocWithFlags(&ptr, alloc_size, hipDeviceMallocFinegrained));
    CHECK(ptr != nullptr);
    CHECK(reinterpret_cast<intptr_t>(ptr) % 256 == 0);
    HIP_CHECK(hipFree(ptr));
  }

  SECTION("hipMallocSignalMemory") {
    HIP_CHECK(hipExtMallocWithFlags(&ptr, 8, hipMallocSignalMemory));
    CHECK(ptr != nullptr);
    HIP_CHECK(hipFree(ptr));
  }
}

HIP_TEST_CASE(Unit_hipExtMallocWithFlags_Positive_Zero_Size) {
  void* ptr = reinterpret_cast<void*>(0x1);
  const auto flag = GENERATE(hipDeviceMallocDefault, hipDeviceMallocFinegrained);
  HIP_CHECK(hipExtMallocWithFlags(&ptr, 0, flag));
  REQUIRE(ptr == nullptr);
}

HIP_TEST_CASE(Unit_hipExtMallocWithFlags_Positive_Alignment) {
  void *ptr1 = nullptr, *ptr2 = nullptr;
  const auto flag = GENERATE(hipDeviceMallocDefault, hipDeviceMallocFinegrained);
  if (flag == hipDeviceMallocFinegrained &&
      !DeviceAttributesSupport(0, hipDeviceAttributeFineGrainSupport)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFineGrainHwUnsupported);
  }
  HIP_CHECK(hipExtMallocWithFlags(&ptr1, 1, flag));
  HIP_CHECK(hipExtMallocWithFlags(&ptr2, 10, flag));
  CHECK(reinterpret_cast<intptr_t>(ptr1) % 256 == 0);
  CHECK(reinterpret_cast<intptr_t>(ptr2) % 256 == 0);
  HIP_CHECK(hipFree(ptr1));
  HIP_CHECK(hipFree(ptr2));
}

HIP_TEST_CASE(Unit_hipExtMallocWithFlags_Negative_Parameters) {
  SECTION("Invalid flags") {
    void* ptr = nullptr;
    HIP_CHECK_ERROR(
        hipExtMallocWithFlags(&ptr, 4096, hipDeviceMallocDefault | hipMallocSignalMemory),
        hipErrorInvalidValue);
  }

  SECTION("hipDeviceMallocDefault") {
    SECTION("ptr == nullptr") {
      HIP_CHECK_ERROR(hipExtMallocWithFlags(nullptr, 4096, hipDeviceMallocDefault),
                      hipErrorInvalidValue);
    }

    SECTION("size == max size_t") {
      void* ptr;
      HIP_CHECK_ERROR(
          hipExtMallocWithFlags(&ptr, std::numeric_limits<size_t>::max(), hipDeviceMallocDefault),
          hipErrorOutOfMemory);
    }
  }

  SECTION("hipDeviceMallocFinegrained") {
    SECTION("ptr == nullptr") {
      HIP_CHECK_ERROR(hipExtMallocWithFlags(nullptr, 4096, hipDeviceMallocFinegrained),
                      hipErrorInvalidValue);
    }

    SECTION("size == max size_t") {
      void* ptr;
      HIP_CHECK_ERROR(hipExtMallocWithFlags(&ptr, std::numeric_limits<size_t>::max(),
                                            hipDeviceMallocFinegrained),
                      hipErrorOutOfMemory);
    }
  }

  SECTION("hipMallocSignalMemory") {
    SECTION("ptr == nullptr") {
      HIP_CHECK_ERROR(hipExtMallocWithFlags(nullptr, 4096, hipMallocSignalMemory),
                      hipErrorInvalidValue);
    }

    SECTION("size == 0") {
      void* ptr;
      HIP_CHECK_ERROR(hipExtMallocWithFlags(&ptr, 0, hipMallocSignalMemory), hipErrorInvalidValue);
    }

    SECTION("size != 8") {
      void* ptr;
      HIP_CHECK_ERROR(hipExtMallocWithFlags(&ptr, 16, hipMallocSignalMemory), hipErrorInvalidValue);
    }
  }
}

namespace {
// Fill values for the two lives of the same VRAM. The previous life's value never matches the
// value the peer writes, so stale data shows up as a mismatch.
__device__ inline unsigned int PreviousLifeValue(unsigned int round, unsigned int buffer,
                                                 size_t word) {
  return 0xdead0000u ^ (round << 24) ^ (buffer << 12) ^ static_cast<unsigned int>(word);
}

__device__ inline unsigned int PeerValue(unsigned int round, unsigned int buffer, size_t word) {
  const unsigned int value = 0x5a000000u ^ (round << 26) ^ (buffer << 16) ^
                             (static_cast<unsigned int>(word) * 2654435761u);
  // Never zero, so the zeroed new allocation does not pass the check.
  return value | 1u;
}

// Writes one buffer with ordinary 16-byte stores from a grid-stride loop.
__global__ void FillPreviousLife(uint4* buffer, size_t words, unsigned int round,
                                 unsigned int index) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < words / 4;
       i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    buffer[i] = make_uint4(
        PreviousLifeValue(round, index, 4 * i), PreviousLifeValue(round, index, 4 * i + 1),
        PreviousLifeValue(round, index, 4 * i + 2), PreviousLifeValue(round, index, 4 * i + 3));
  }
}

// One block per buffer.
__global__ void PeerWrite(unsigned int* const* buffers, size_t words, unsigned int round) {
  uint4* const buffer = reinterpret_cast<uint4*>(buffers[blockIdx.x]);
  for (size_t i = threadIdx.x; i < words / 4; i += blockDim.x) {
    buffer[i] = make_uint4(
        PeerValue(round, blockIdx.x, 4 * i), PeerValue(round, blockIdx.x, 4 * i + 1),
        PeerValue(round, blockIdx.x, 4 * i + 2), PeerValue(round, blockIdx.x, 4 * i + 3));
  }
}

__global__ void CountMismatches(unsigned int* const* buffers, size_t words, unsigned int round,
                                unsigned long long* mismatches) {
  const uint4* const buffer = reinterpret_cast<const uint4*>(buffers[blockIdx.x]);
  unsigned long long count = 0;
  for (size_t i = threadIdx.x; i < words / 4; i += blockDim.x) {
    const uint4 v = buffer[i];
    count += (v.x != PeerValue(round, blockIdx.x, 4 * i)) +
             (v.y != PeerValue(round, blockIdx.x, 4 * i + 1)) +
             (v.z != PeerValue(round, blockIdx.x, 4 * i + 2)) +
             (v.w != PeerValue(round, blockIdx.x, 4 * i + 3));
  }
  if (count != 0) atomicAdd(&mismatches[blockIdx.x], count);
}
}  // namespace

/**
 * Test Description
 * ------------------------
 * - Reuses device memory across allocation types: in each round the owner GPU writes ordinary
 *   hipMalloc buffers and frees them, then allocates the same amount with
 *   hipExtMallocWithFlags(hipDeviceMallocUncached) and zeroes it. The peer GPU writes the new
 *   buffers through peer access, and the owner GPU must then read the peer's data, not zeros and
 *   not the freed buffers' data.
 * - Guards against an MTYPE mapping of uncached memory under which the owner GPU read back the
 *   freed allocation's data after the peer's first write (seen on gfx950). The mismatch is a race,
 *   so the test uses many buffers and rounds; on a heavily loaded node it may not trigger every
 *   run.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipExtMallocWithFlags.cc
 * Test requirements
 * ------------------------
 * - Multi-device
 * - Peer access supported
 */
HIP_TEST_CASE(Unit_hipExtMallocWithFlags_Uncached_ReusedMemory_PeerWriteVisible) {
  if (HipTest::getDeviceCount() < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  constexpr int kOwner = 0;
  constexpr int kPeer = 1;
  int can_access_peer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, kPeer, kOwner));
  if (can_access_peer == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }

  constexpr unsigned int kRounds = 10;
  constexpr unsigned int kBuffers = 128;
  constexpr size_t kBufferBytes = 6 * 1024 * 1024;
  constexpr size_t kWords = kBufferBytes / sizeof(unsigned int);
  constexpr unsigned int kThreads = 256;
  constexpr unsigned int kFillBlocks = 64;

  int original_device = 0;
  HIP_CHECK(hipGetDevice(&original_device));
  HIP_CHECK(hipSetDevice(kOwner));
  size_t free_bytes = 0, total_bytes = 0;
  HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
  if (free_bytes < 2 * kBuffers * kBufferBytes) {
    HIP_CHECK(hipSetDevice(original_device));
    HIP_SKIP_TEST(HipTest::SkipReason::kNotEnoughFreeGpuMemory);
  }

  HIP_CHECK(hipSetDevice(kPeer));
  const hipError_t enable_status = hipDeviceEnablePeerAccess(kOwner, 0);
  if (enable_status != hipErrorPeerAccessAlreadyEnabled) {
    HIP_CHECK(enable_status);
  }
  static_cast<void>(hipGetLastError());

  // Pointer tables and counters live outside the rounds so that only the test buffers recycle VRAM.
  unsigned int** peer_table = nullptr;
  HIP_CHECK(hipMalloc(&peer_table, kBuffers * sizeof(unsigned int*)));
  HIP_CHECK(hipSetDevice(kOwner));
  unsigned int** owner_table = nullptr;
  unsigned long long* mismatches = nullptr;
  HIP_CHECK(hipMalloc(&owner_table, kBuffers * sizeof(unsigned int*)));
  HIP_CHECK(hipMalloc(&mismatches, kBuffers * sizeof(unsigned long long)));

  std::vector<unsigned int*> buffers(kBuffers);
  std::vector<unsigned long long> counts(kBuffers);
  unsigned int bad_rounds = 0;
  unsigned int first_bad_round = 0;
  unsigned long long bad_words = 0;
  for (unsigned int round = 1; round <= kRounds; ++round) {
    // Previous life: cacheable buffers written with ordinary stores, then freed.
    HIP_CHECK(hipSetDevice(kOwner));
    for (auto& buffer : buffers) {
      HIP_CHECK(hipMalloc(&buffer, kBufferBytes));
    }
    // One launch per buffer. A single launch that writes all buffers does not leave the VRAM in
    // the state that exposed the problem.
    for (unsigned int i = 0; i < kBuffers; ++i) {
      FillPreviousLife<<<kFillBlocks, kThreads>>>(reinterpret_cast<uint4*>(buffers[i]), kWords,
                                                  round, i);
    }
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    for (auto buffer : buffers) {
      HIP_CHECK(hipFree(buffer));
    }

    // New life: uncached buffers on the owner, zeroed.
    for (auto& buffer : buffers) {
      HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&buffer), kBufferBytes,
                                      hipDeviceMallocUncached));
      HIP_CHECK(hipMemset(buffer, 0, kBufferBytes));
    }
    HIP_CHECK(hipMemcpy(owner_table, buffers.data(), kBuffers * sizeof(unsigned int*),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(mismatches, 0, kBuffers * sizeof(unsigned long long)));
    HIP_CHECK(hipDeviceSynchronize());

    // The peer writes every buffer once, then the owner checks it.
    HIP_CHECK(hipSetDevice(kPeer));
    HIP_CHECK(hipMemcpy(peer_table, buffers.data(), kBuffers * sizeof(unsigned int*),
                        hipMemcpyHostToDevice));
    PeerWrite<<<kBuffers, kThreads>>>(peer_table, kWords, round);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipSetDevice(kOwner));
    CountMismatches<<<kBuffers, kThreads>>>(owner_table, kWords, round, mismatches);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(counts.data(), mismatches, kBuffers * sizeof(unsigned long long),
                        hipMemcpyDeviceToHost));
    for (auto buffer : buffers) {
      HIP_CHECK(hipFree(buffer));
    }

    unsigned long long round_words = 0;
    for (auto count : counts) {
      round_words += count;
    }
    if (round_words != 0) {
      if (bad_rounds++ == 0) first_bad_round = round;
      bad_words += round_words;
    }
  }

  HIP_CHECK(hipFree(owner_table));
  HIP_CHECK(hipFree(mismatches));
  HIP_CHECK(hipSetDevice(kPeer));
  HIP_CHECK(hipFree(peer_table));
  if (enable_status == hipSuccess) {
    HIP_CHECK(hipDeviceDisablePeerAccess(kOwner));
  }
  HIP_CHECK(hipSetDevice(original_device));

  INFO("rounds with mismatches: " << bad_rounds << "/" << kRounds << ", first bad round: "
                                  << first_bad_round << ", mismatched words: " << bad_words);
  REQUIRE(bad_words == 0);
}
