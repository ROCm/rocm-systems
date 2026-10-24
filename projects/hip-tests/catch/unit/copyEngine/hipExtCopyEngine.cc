/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip/hip_ext_copy_engine.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

/**
 * @addtogroup hipExtCopyEngineGetDeviceHandles hipExtCopyEngineGetDeviceHandles
 * @{
 * @ingroup PeerToPeerTest
 * `hipError_t hipExtCopyEngineGetDeviceHandles(int device, int srcDevice, int dstDevice,
 *                                             int maxHandles, hipExtCopyEngineDeviceHandle* handles,
 *                                             int* numHandles);`
 * - Returns the handles, one per engine, a kernel on @p device submits SDMA
 *   copies, writes and signals from @p srcDevice to @p dstDevice memory through,
 *   and waits on.
 */

namespace {

constexpr uint64_t kMaxSpins = 1ull << 34;

constexpr char kCopyEngineUnavailable[] =
    "kernel-side copies are off (set DEBUG_CLR_ENABLE_ABCE=1 and GPU_MAX_SDMA_DEVICE_QUEUES=16) "
    "or unsupported on this GPU";

bool GetCopyEngineHandle(int device, int src_device, int dst_device,
                         hipExtCopyEngineDeviceHandle* handle) {
  int num_handles = 0;
  const hipError_t status =
      hipExtCopyEngineGetDeviceHandles(device, src_device, dst_device, 1, handle, &num_handles);
  if (status == hipErrorNotSupported) return false;
  HIP_CHECK(status);
  return true;
}

// How a kernel waits on a completion: relaxed spins, then one system-scope
// acquire so the engine's writes are visible. hipErrorNotReady after max_spins.
__device__ hipError_t WaitFor(const hipExtCopyEngineCondition& condition, uint64_t max_spins) {
  const uint64_t* word = reinterpret_cast<const uint64_t*>(condition.signalAddress);
  for (uint64_t spin = 0; spin <= max_spins; ++spin) {
    const uint64_t value = __hip_atomic_load(word, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    if (condition.compare == hipExtCopyEngineCompareLessEqual ? value <= condition.value
                                                              : value >= condition.value) {
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
      return hipSuccess;
    }
    __builtin_amdgcn_s_sleep(1);
  }
  return hipErrorNotReady;
}

__device__ hipExtCopyEngineSignal SetOnCompletion(uint64_t* signal, uint64_t value = 1) {
  return hipExtCopyEngineSignal{reinterpret_cast<uint64_t>(signal), value,
                                hipExtCopyEngineSignalSet, 0};
}

__device__ hipExtCopyEngineCondition Reached(uint64_t* signal, uint64_t value = 1) {
  return hipExtCopyEngineCondition{reinterpret_cast<uint64_t>(signal), value,
                                   hipExtCopyEngineCompareGreaterEqual};
}

__device__ hipExtCopyEngineCondition CountedDownTo(uint64_t* signal, uint64_t value = 0) {
  return hipExtCopyEngineCondition{reinterpret_cast<uint64_t>(signal), value,
                                   hipExtCopyEngineCompareLessEqual};
}

// The block produces the source, one thread submits and waits, and the whole
// block then checks the destination. The barrier plus the system fence make the
// block's stores visible to the copy engine.
__global__ void ProduceCopyConsume(hipExtCopyEngineDeviceHandle handle, uint32_t* dst,
                                   uint32_t* src, uint32_t num_words, uint64_t* signal,
                                   hipError_t* status, uint32_t* num_mismatches) {
  for (uint32_t word = threadIdx.x; word < num_words; word += blockDim.x) src[word] = ~word;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    const hipExtCopyEngineSignal completion = SetOnCompletion(signal);
    const hipExtCopyEngineCondition done = Reached(signal);
    hipError_t result = hipExtCopyEngineSubmitLinearOneLane(
        &handle, dst, src, uint64_t{num_words} * sizeof(uint32_t), &completion);
    if (result == hipSuccess) result = WaitFor(done, kMaxSpins);
    *status = result;
  }
  __syncthreads();
  for (uint32_t word = threadIdx.x; word < num_words; word += blockDim.x) {
    if (dst[word] != ~word) atomicAdd(num_mismatches, 1u);
  }
}

// Every lane submits its own chunk on its own signal, then waits for it.
__global__ void LaneCopies(hipExtCopyEngineDeviceHandle handle, char* dst, const char* src,
                           uint64_t chunk_bytes, uint64_t* signals, hipError_t* statuses) {
  const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  const hipExtCopyEngineSignal completion = SetOnCompletion(&signals[lane]);
  const hipExtCopyEngineCondition done = Reached(&signals[lane]);
  hipError_t result =
      hipExtCopyEngineSubmitLinear(&handle, dst + lane * chunk_bytes, src + lane * chunk_bytes,
                                   chunk_bytes, &completion);
  if (result == hipSuccess) result = WaitFor(done, kMaxSpins);
  statuses[lane] = result;
}

// Every lane copies a chunk and adds (or subtracts) one on a shared counter,
// then waits for all lanes.
__global__ void LaneCopiesCountedOnce(hipExtCopyEngineDeviceHandle handle, char* dst,
                                      const char* src, uint64_t chunk_bytes, uint64_t* counter,
                                      bool count_down, hipError_t* statuses) {
  const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  const hipExtCopyEngineSignal completion{
      reinterpret_cast<uint64_t>(counter), 1,
      count_down ? hipExtCopyEngineSignalSub : hipExtCopyEngineSignalAdd, 0};
  const hipExtCopyEngineCondition all_done =
      count_down ? CountedDownTo(counter) : Reached(counter, blockDim.x * gridDim.x);
  hipError_t result =
      hipExtCopyEngineSubmitLinear(&handle, dst + lane * chunk_bytes, src + lane * chunk_bytes,
                                   chunk_bytes, &completion);
  if (result == hipSuccess) result = WaitFor(all_done, kMaxSpins);
  statuses[lane] = result;
}

// One copy that sets the signal's word to |value|.
__global__ void CopySettingValue(hipExtCopyEngineDeviceHandle handle, void* dst, const void* src,
                                 uint64_t size, uint64_t* signal, uint64_t value,
                                 hipError_t* status) {
  const hipExtCopyEngineSignal completion = SetOnCompletion(signal, value);
  const hipExtCopyEngineCondition done = Reached(signal, value);
  hipError_t result = hipExtCopyEngineSubmitLinearOneLane(&handle, dst, src, size, &completion);
  if (result == hipSuccess) result = WaitFor(done, kMaxSpins);
  *status = result;
}

// A copy with no completion, then one on the same handle whose completion
// covers both.
__global__ void TwoCopiesOneCompletion(hipExtCopyEngineDeviceHandle handle, void* first_dst,
                                       const void* first_src, void* second_dst,
                                       const void* second_src, uint64_t size, uint64_t* signal,
                                       hipError_t* status) {
  hipError_t result = hipExtCopyEngineSubmitLinear(&handle, first_dst, first_src, size);
  const hipExtCopyEngineSignal completion = SetOnCompletion(signal);
  const hipExtCopyEngineCondition done = Reached(signal);
  if (result == hipSuccess)
    result = hipExtCopyEngineSubmitLinear(&handle, second_dst, second_src, size, &completion);
  if (result == hipSuccess) result = WaitFor(done, kMaxSpins);
  *status = result;
}

// Inline 1-, 2-, 4- and 8-byte writes into |bytes|; the last one's completion
// covers all of them. Other sizes are refused.
__global__ void InlineWrites(hipExtCopyEngineDeviceHandle handle, uint8_t* bytes,
                             uint64_t* signal, hipError_t* status) {
  hipError_t result = hipExtCopyEngineSubmitWriteOneLane(&handle, bytes + 0, 0xc0ffee11u, 4);
  if (result == hipSuccess)
    result = hipExtCopyEngineSubmitWriteOneLane(&handle, bytes + 17, 0x5a, 1);
  if (result == hipSuccess)
    result = hipExtCopyEngineSubmitWriteOneLane(&handle, bytes + 22, 0xbeef, 2);
  for (const uint32_t refused_size : {3u, 16u}) {
    if (result == hipSuccess &&
        hipExtCopyEngineSubmitWriteOneLane(&handle, bytes + 16, 0x5a, refused_size) !=
            hipErrorInvalidValue)
      result = hipErrorUnknown;
  }
  const hipExtCopyEngineSignal completion = SetOnCompletion(signal);
  const hipExtCopyEngineCondition done = Reached(signal);
  if (result == hipSuccess)
    result = hipExtCopyEngineSubmitWriteOneLane(&handle, bytes + 8, 0x0123456789abcdefull, 8,
                                                &completion);
  if (result == hipSuccess) result = WaitFor(done, kMaxSpins);
  *status = result;
}

// Two copies submitted back to back, the second reading what the first writes:
// a slow one from host memory into a staging buffer, then a fast device-local
// one out of it, each on its own route's handle (their engines may differ).
// Only the dependency keeps the second from reading the staging buffer early;
// the thread waits once, on the last copy.
__global__ void DependentCopies(hipExtCopyEngineDeviceHandle stage_handle,
                                hipExtCopyEngineDeviceHandle deliver_handle, void* dst,
                                void* staging, const void* host_src, uint64_t size,
                                uint64_t* signals, hipError_t* status) {
  const hipExtCopyEngineSignal staged = SetOnCompletion(&signals[0]);
  const hipExtCopyEngineCondition staged_done = Reached(&signals[0]);
  const hipExtCopyEngineSignal delivered = SetOnCompletion(&signals[1]);
  const hipExtCopyEngineCondition delivered_done = Reached(&signals[1]);
  hipError_t result = hipExtCopyEngineSubmitLinear(&stage_handle, staging, host_src, size, &staged);
  if (result == hipSuccess)
    result = hipExtCopyEngineSubmitLinear(&deliver_handle, dst, staging, size, &delivered,
                                          &staged_done);
  if (result == hipSuccess) result = WaitFor(delivered_done, kMaxSpins);
  *status = result;
}

// DependentCopies with countdown signals (1 -> 0).
__global__ void CountdownDependentCopies(hipExtCopyEngineDeviceHandle stage_handle,
                                         hipExtCopyEngineDeviceHandle deliver_handle, void* dst,
                                         void* staging, const void* host_src, uint64_t size,
                                         uint64_t* signals, hipError_t* status) {
  const hipExtCopyEngineSignal staged{reinterpret_cast<uint64_t>(&signals[0]), 1,
                                      hipExtCopyEngineSignalSub, 0};
  const hipExtCopyEngineCondition staged_done = CountedDownTo(&signals[0]);
  const hipExtCopyEngineSignal delivered{reinterpret_cast<uint64_t>(&signals[1]), 1,
                                         hipExtCopyEngineSignalSub, 0};
  const hipExtCopyEngineCondition delivered_done = CountedDownTo(&signals[1]);
  hipError_t result =
      hipExtCopyEngineSubmitLinearOneLane(&stage_handle, staging, host_src, size, &staged);
  if (result == hipSuccess)
    result = hipExtCopyEngineSubmitLinearOneLane(&deliver_handle, dst, staging, size, &delivered,
                                                 &staged_done);
  if (result == hipSuccess) result = WaitFor(delivered_done, kMaxSpins);
  *status = result;
}

// One lane submits |num_copies| copies of |chunk_bytes|, enough to wrap the
// device ring several times; the last one's set signal covers them all.
__global__ void OneLaneManyCopies(hipExtCopyEngineDeviceHandle handle, char* dst, const char* src,
                                  uint64_t chunk_bytes, uint32_t num_copies, uint64_t* signal,
                                  hipError_t* status) {
  hipError_t result = hipSuccess;
  for (uint32_t copy_idx = 0; copy_idx + 1 < num_copies && result == hipSuccess; ++copy_idx)
    result = hipExtCopyEngineSubmitLinearOneLane(&handle, dst + copy_idx * chunk_bytes,
                                                 src + copy_idx * chunk_bytes, chunk_bytes);
  const hipExtCopyEngineSignal completion = SetOnCompletion(signal);
  const hipExtCopyEngineCondition done = Reached(signal);
  const uint64_t last = num_copies - 1;
  if (result == hipSuccess)
    result = hipExtCopyEngineSubmitLinearOneLane(&handle, dst + last * chunk_bytes,
                                                 src + last * chunk_bytes, chunk_bytes,
                                                 &completion);
  if (result == hipSuccess) result = WaitFor(done, kMaxSpins);
  *status = result;
}

// Every lane of the block calls the one-lane submit: it must refuse rather than
// let the lanes spin on each other.
__global__ void OneLaneCalledByTwoLanes(hipExtCopyEngineDeviceHandle handle, void* dst,
                                        const void* src, hipError_t* statuses) {
  statuses[threadIdx.x] = hipExtCopyEngineSubmitLinearOneLane(&handle, dst, src, 64);
}

// Each block's lanes copy their chunks on the block's handle, picked by
// blockIdx, and wait for them.
__global__ void SpreadCopies(const hipExtCopyEngineDeviceHandle* handles, int num_handles,
                             char* dst, const char* src, uint64_t chunk_bytes, uint64_t* signals,
                             hipError_t* statuses) {
  const hipExtCopyEngineDeviceHandle handle = handles[blockIdx.x % num_handles];
  const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  const hipExtCopyEngineSignal completion = SetOnCompletion(&signals[lane]);
  const hipExtCopyEngineCondition done = Reached(&signals[lane]);
  hipError_t result =
      hipExtCopyEngineSubmitLinear(&handle, dst + lane * chunk_bytes, src + lane * chunk_bytes,
                                   chunk_bytes, &completion);
  if (result == hipSuccess) result = WaitFor(done, kMaxSpins);
  statuses[lane] = result;
}

// Every lane of each wave submits on handles[lane % num_handles]; every
// |invalid_every|th lane passes a null destination and must fail alone.
__global__ void MixedLaneCopies(const hipExtCopyEngineDeviceHandle* handles, int num_handles,
                                char* dst, const char* src, uint64_t chunk_bytes,
                                uint32_t invalid_every, uint64_t* signals, hipError_t* statuses) {
  const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
  const hipExtCopyEngineDeviceHandle handle = handles[threadIdx.x % num_handles];
  const bool invalid = lane % invalid_every == invalid_every - 1;
  const hipExtCopyEngineSignal completion = SetOnCompletion(&signals[lane]);
  const hipExtCopyEngineCondition done = Reached(&signals[lane]);
  hipError_t result = hipExtCopyEngineSubmitLinear(
      &handle, invalid ? nullptr : dst + lane * chunk_bytes, src + lane * chunk_bytes, chunk_bytes,
      &completion);
  if (result == hipSuccess) result = WaitFor(done, kMaxSpins);
  statuses[lane] = result;
}

__global__ void ReadEngineIds(const hipExtCopyEngineDeviceHandle* handles, int num_handles,
                              uint32_t* engine_ids) {
  if (threadIdx.x < num_handles)
    engine_ids[threadIdx.x] = hip::detail::CopyEngineHandle(&handles[threadIdx.x]).engine->engine_id;
}

// |num_copies| copies with no signal, then |drain| lands once all of them have:
// a signal-only submit, or an add signal on one more copy (ordered by default).
__global__ void CopiesThenOneSignal(hipExtCopyEngineDeviceHandle handle, char* dst, const char* src,
                                   uint64_t chunk_bytes, uint32_t num_copies, bool signal_only,
                                   uint64_t* signal, hipError_t* status) {
  hipError_t result = hipSuccess;
  const uint32_t num_unsignaled = signal_only ? num_copies : num_copies - 1;
  for (uint32_t copy_idx = 0; copy_idx < num_unsignaled && result == hipSuccess; ++copy_idx)
    result = hipExtCopyEngineSubmitLinearOneLane(&handle, dst + copy_idx * chunk_bytes,
                                                 src + copy_idx * chunk_bytes, chunk_bytes);
  const hipExtCopyEngineSignal drain{reinterpret_cast<uint64_t>(signal), 1,
                                    hipExtCopyEngineSignalAdd, 0};
  const hipExtCopyEngineSignal set_drain = SetOnCompletion(signal);
  const hipExtCopyEngineCondition done = Reached(signal);
  if (result == hipSuccess) {
    const uint64_t last = num_copies - 1;
    result = signal_only
                 ? hipExtCopyEngineSubmitSignalOneLane(&handle, &set_drain)
                 : hipExtCopyEngineSubmitLinearOneLane(&handle, dst + last * chunk_bytes,
                                                       src + last * chunk_bytes, chunk_bytes,
                                                       &drain);
  }
  if (result == hipSuccess) result = WaitFor(done, kMaxSpins);
  *status = result;
}

// A non-blocking submit on an unmet dependency queues nothing; once the word is
// set, the same submit goes through.
__global__ void NonblockingOnUnmetDependency(hipExtCopyEngineDeviceHandle handle, void* dst,
                                             const void* src, uint64_t size, uint64_t* gate,
                                             uint64_t* signal, hipError_t* statuses) {
  const hipExtCopyEngineCondition open = Reached(gate);
  const hipExtCopyEngineSignal completion = SetOnCompletion(signal);
  const hipExtCopyEngineCondition done = Reached(signal);
  statuses[0] = hipExtCopyEngineSubmitLinearOneLane(&handle, dst, src, size, &completion, &open,
                                                    HIP_EXT_COPY_ENGINE_SUBMIT_NONBLOCKING);
  __hip_atomic_store(gate, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  hipError_t result = hipExtCopyEngineSubmitLinearOneLane(&handle, dst, src, size, &completion,
                                                          &open,
                                                          HIP_EXT_COPY_ENGINE_SUBMIT_NONBLOCKING);
  if (result == hipSuccess) result = WaitFor(done, kMaxSpins);
  statuses[1] = result;
}

struct ProduceCopyConsumeResult {
  hipError_t status;
  uint32_t num_mismatches;
};

ProduceCopyConsumeResult RunProduceCopyConsume(const hipExtCopyEngineDeviceHandle& handle,
                                               uint32_t* dst, uint32_t* src, uint32_t num_words) {
  uint64_t* signal = nullptr;
  ProduceCopyConsumeResult* result = nullptr;
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&signal), sizeof(uint64_t),
                                  hipMallocSignalMemory));
  HIP_CHECK(hipMemset(signal, 0, sizeof(uint64_t)));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result), sizeof(*result)));
  *result = {hipErrorUnknown, 0};
  ProduceCopyConsume<<<1, 256>>>(handle, dst, src, num_words, signal, &result->status,
                                 &result->num_mismatches);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  const ProduceCopyConsumeResult copy = *result;
  HIP_CHECK(hipHostFree(result));
  HIP_CHECK(hipFree(signal));
  return copy;
}

}  // namespace

/**
 * Test Description
 * ------------------------
 * - Validates argument handling: null outputs or a zero capacity are rejected
 *   with hipErrorInvalidValue, and an out-of-range submitting, source or
 *   destination device with hipErrorInvalidDevice.
 * Test source
 * ------------------------
 *    - unit/copyEngine/hipExtCopyEngine.cc
 * Test requirements
 * ------------------------
 *    - Platform specific (AMD)
 */
HIP_TEST_CASE(Unit_hipExtCopyEngineGetDeviceHandles_Negative) {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  hipExtCopyEngineDeviceHandle handle{};
  int num_handles = 0;
  HIP_CHECK_ERROR(hipExtCopyEngineGetDeviceHandles(0, 0, 0, 1, nullptr, &num_handles),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipExtCopyEngineGetDeviceHandles(0, 0, 0, 1, &handle, nullptr),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipExtCopyEngineGetDeviceHandles(0, 0, 0, 0, &handle, &num_handles),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipExtCopyEngineGetDeviceHandles(-1, 0, 0, 1, &handle, &num_handles),
                  hipErrorInvalidDevice);
  HIP_CHECK_ERROR(hipExtCopyEngineGetDeviceHandles(hipCpuDeviceId, 0, 0, 1, &handle, &num_handles),
                  hipErrorInvalidDevice);
  HIP_CHECK_ERROR(hipExtCopyEngineGetDeviceHandles(num_devices, 0, 0, 1, &handle, &num_handles),
                  hipErrorInvalidDevice);
  HIP_CHECK_ERROR(hipExtCopyEngineGetDeviceHandles(0, num_devices, 0, 1, &handle, &num_handles),
                  hipErrorInvalidDevice);
  HIP_CHECK_ERROR(hipExtCopyEngineGetDeviceHandles(0, 0, -2, 1, &handle, &num_handles),
                  hipErrorInvalidDevice);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidDevice);
}

/**
 * Test Description
 * ------------------------
 * - Produce, release, copy, wait and check, to device, pinned host and peer,
 *   each through its route's handle.
 * - Concurrent lanes in one block, and across XCCs (128 blocks on one ring).
 * - Add/subtract completions on a shared counter from every lane.
 * - Set completions across the 32-bit boundary.
 * - Two copies on different engines are submitted back to back, the second
 *   reading what the first writes, ordered only by passing the first's
 *   completion as the second's dependency (count-up and countdown signals).
 * - A copy without a completion, then one on the same handle whose completion
 *   covers both.
 * - Inline 1-, 2-, 4- and 8-byte writes to device and host memory, leaving the
 *   bytes around them untouched; other sizes are refused.
 * - One lane submits enough copies to wrap the device ring several times; the
 *   one-lane submit refuses two lanes of a wave.
 * - Blocks spread over several handles for one route, each on its own engine;
 *   lanes of one wave on different handles, some with invalid requests.
 * - A signal-only submit, and a default (ordered) add signal, land only after
 *   every earlier copy on the engine.
 * - A non-blocking submit on an unmet dependency returns hipErrorNotReady.
 * Test source
 * ------------------------
 *    - unit/copyEngine/hipExtCopyEngine.cc
 * Test requirements
 * ------------------------
 *    - Platform specific (AMD)
 *    - DEBUG_CLR_ENABLE_ABCE=1 and GPU_MAX_SDMA_DEVICE_QUEUES > 0
 */
HIP_TEST_CASE(Unit_hipExtCopyEngine_KernelCopies) {
  constexpr int kLocalDevice = 0;
  constexpr uint32_t kNumWords = 1u << 20;
  constexpr size_t kBytes = kNumWords * sizeof(uint32_t);
  HIP_CHECK(hipSetDevice(kLocalDevice));
  hipExtCopyEngineDeviceHandle handle{};
  if (!GetCopyEngineHandle(kLocalDevice, kLocalDevice, kLocalDevice, &handle))
    HIP_SKIP_TEST(kCopyEngineUnavailable);

  uint32_t* src = nullptr;
  HIP_CHECK(hipMalloc(&src, kBytes));

  SECTION("Same device") {
    uint32_t* dst = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    const ProduceCopyConsumeResult result = RunProduceCopyConsume(handle, dst, src, kNumWords);
    HIP_CHECK(result.status);
    REQUIRE(result.num_mismatches == 0);
    HIP_CHECK(hipFree(dst));
  }

  SECTION("Pinned host memory") {
    uint32_t* dst = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&dst), kBytes, hipHostMallocCoherent));
    hipExtCopyEngineDeviceHandle to_host{};
    REQUIRE(GetCopyEngineHandle(kLocalDevice, kLocalDevice, hipCpuDeviceId, &to_host));
    const ProduceCopyConsumeResult result = RunProduceCopyConsume(to_host, dst, src, kNumWords);
    HIP_CHECK(result.status);
    REQUIRE(result.num_mismatches == 0);
    for (uint32_t word = 0; word < kNumWords; ++word) REQUIRE(dst[word] == ~word);
    HIP_CHECK(hipHostFree(dst));
  }

  SECTION("Peer device") {
    int num_devices = 0;
    HIP_CHECK(hipGetDeviceCount(&num_devices));
    int can_access_peer = 0;
    if (num_devices >= 2) HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, kLocalDevice, 1));
    if (!can_access_peer) HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
    constexpr int kPeerDevice = 1;
    HIP_CHECK(hipSetDevice(kPeerDevice));
    uint32_t* dst = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    HIP_CHECK(hipSetDevice(kLocalDevice));
    const hipError_t peer_status = hipDeviceEnablePeerAccess(kPeerDevice, 0);
    if (peer_status != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(peer_status);
    hipExtCopyEngineDeviceHandle to_peer{};
    REQUIRE(GetCopyEngineHandle(kLocalDevice, kLocalDevice, kPeerDevice, &to_peer));
    const ProduceCopyConsumeResult result = RunProduceCopyConsume(to_peer, dst, src, kNumWords);
    HIP_CHECK(result.status);
    REQUIRE(result.num_mismatches == 0);
    HIP_CHECK(hipFree(dst));
  }

  SECTION("Concurrent lanes") {
    constexpr uint32_t kNumLanes = 256;
    constexpr uint64_t kChunkBytes = kBytes / kNumLanes;
    std::vector<uint32_t> pattern(kNumWords);
    for (uint32_t word = 0; word < kNumWords; ++word) pattern[word] = word * 2654435761u;
    HIP_CHECK(hipMemcpy(src, pattern.data(), kBytes, hipMemcpyHostToDevice));
    char* dst = nullptr;
    uint64_t* signals = nullptr;
    hipError_t* statuses = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&signals), kNumLanes * sizeof(uint64_t),
                            hipHostMallocCoherent));
    std::fill(signals, signals + kNumLanes, uint64_t{0});
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&statuses), kNumLanes * sizeof(hipError_t)));
    LaneCopies<<<1, kNumLanes>>>(handle, dst, reinterpret_cast<const char*>(src), kChunkBytes,
                                 signals, statuses);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    for (uint32_t lane = 0; lane < kNumLanes; ++lane) HIP_CHECK(statuses[lane]);
    std::vector<uint32_t> copied(kNumWords);
    HIP_CHECK(hipMemcpy(copied.data(), dst, kBytes, hipMemcpyDeviceToHost));
    REQUIRE(copied == pattern);
    HIP_CHECK(hipHostFree(statuses));
    HIP_CHECK(hipHostFree(signals));
    HIP_CHECK(hipFree(dst));
  }

  SECTION("Concurrent producers across XCCs") {
    // Blocks spread over XCCs, all submitting to one ring.
    constexpr uint32_t kNumBlocks = 128;
    constexpr uint32_t kLanesPerBlock = 8;
    constexpr uint32_t kNumLanes = kNumBlocks * kLanesPerBlock;
    constexpr uint64_t kChunkBytes = kBytes / kNumLanes;
    std::vector<uint32_t> pattern(kNumWords);
    for (uint32_t word = 0; word < kNumWords; ++word) pattern[word] = word * 2246822519u + 3u;
    HIP_CHECK(hipMemcpy(src, pattern.data(), kBytes, hipMemcpyHostToDevice));
    char* dst = nullptr;
    uint64_t* signals = nullptr;
    hipError_t* statuses = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&signals), kNumLanes * sizeof(uint64_t),
                            hipHostMallocCoherent));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&statuses), kNumLanes * sizeof(hipError_t)));
    for (int round = 0; round < 8; ++round) {
      INFO("round " << round);
      HIP_CHECK(hipMemset(dst, 0, kBytes));
      std::fill(signals, signals + kNumLanes, uint64_t{0});
      LaneCopies<<<kNumBlocks, kLanesPerBlock>>>(handle, dst, reinterpret_cast<const char*>(src),
                                                 kChunkBytes, signals, statuses);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      for (uint32_t lane = 0; lane < kNumLanes; ++lane) HIP_CHECK(statuses[lane]);
      std::vector<uint32_t> copied(kNumWords);
      HIP_CHECK(hipMemcpy(copied.data(), dst, kBytes, hipMemcpyDeviceToHost));
      REQUIRE(copied == pattern);
    }
    HIP_CHECK(hipHostFree(statuses));
    HIP_CHECK(hipHostFree(signals));
    HIP_CHECK(hipFree(dst));
  }

  SECTION("One lane wraps the device ring") {
    constexpr uint64_t kChunkBytes = 32;
    constexpr uint32_t kNumCopies = kBytes / kChunkBytes;  // ~3.6 MB of frames: wraps a 1 MiB ring
    std::vector<uint32_t> pattern(kNumWords);
    for (uint32_t word = 0; word < kNumWords; ++word) pattern[word] = word * 2654435769u + 11u;
    HIP_CHECK(hipMemcpy(src, pattern.data(), kBytes, hipMemcpyHostToDevice));
    char* dst = nullptr;
    uint64_t* signal = nullptr;
    hipError_t* status = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    HIP_CHECK(hipMemset(dst, 0, kBytes));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&signal), sizeof(uint64_t),
                            hipHostMallocCoherent));
    *signal = 0;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&status), sizeof(hipError_t)));
    OneLaneManyCopies<<<1, 1>>>(handle, dst, reinterpret_cast<const char*>(src), kChunkBytes,
                                kNumCopies, signal, status);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(*status);
    std::vector<uint32_t> copied(kNumWords);
    HIP_CHECK(hipMemcpy(copied.data(), dst, kBytes, hipMemcpyDeviceToHost));
    REQUIRE(copied == pattern);
    HIP_CHECK(hipHostFree(status));
    HIP_CHECK(hipHostFree(signal));
    HIP_CHECK(hipFree(dst));
  }

  SECTION("The one-lane submit refuses two lanes of a wave") {
    void* dst = nullptr;
    hipError_t* statuses = nullptr;
    HIP_CHECK(hipMalloc(&dst, 64));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&statuses), 2 * sizeof(hipError_t)));
    OneLaneCalledByTwoLanes<<<1, 2>>>(handle, dst, src, statuses);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(statuses[0] == hipErrorInvalidValue);
    REQUIRE(statuses[1] == hipErrorInvalidValue);
    HIP_CHECK(hipHostFree(statuses));
    HIP_CHECK(hipFree(dst));
  }

  SECTION("Blocks spread over several handles") {
    constexpr int kMaxHandles = 4;
    hipExtCopyEngineDeviceHandle* handles = nullptr;
    uint32_t* engine_ids = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&handles),
                            kMaxHandles * sizeof(hipExtCopyEngineDeviceHandle)));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&engine_ids), kMaxHandles * sizeof(uint32_t)));
    int num_handles = 0;
    HIP_CHECK(hipExtCopyEngineGetDeviceHandles(kLocalDevice, kLocalDevice, kLocalDevice,
                                               kMaxHandles, handles, &num_handles));
    REQUIRE(num_handles >= 1);
    REQUIRE(num_handles <= kMaxHandles);
    ReadEngineIds<<<1, kMaxHandles>>>(handles, num_handles, engine_ids);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> distinct(engine_ids, engine_ids + num_handles);
    std::sort(distinct.begin(), distinct.end());
    REQUIRE(std::unique(distinct.begin(), distinct.end()) == distinct.end());

    constexpr uint32_t kNumBlocks = 64;
    constexpr uint32_t kLanesPerBlock = 4;
    constexpr uint32_t kNumLanes = kNumBlocks * kLanesPerBlock;
    constexpr uint64_t kChunkBytes = kBytes / kNumLanes;
    std::vector<uint32_t> pattern(kNumWords);
    for (uint32_t word = 0; word < kNumWords; ++word) pattern[word] = word * 2246822519u + 5u;
    HIP_CHECK(hipMemcpy(src, pattern.data(), kBytes, hipMemcpyHostToDevice));
    char* dst = nullptr;
    uint64_t* signals = nullptr;
    hipError_t* statuses = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&signals), kNumLanes * sizeof(uint64_t),
                            hipHostMallocCoherent));
    std::fill(signals, signals + kNumLanes, uint64_t{0});
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&statuses), kNumLanes * sizeof(hipError_t)));
    SpreadCopies<<<kNumBlocks, kLanesPerBlock>>>(handles, num_handles, dst,
                                                 reinterpret_cast<const char*>(src), kChunkBytes,
                                                 signals, statuses);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    for (uint32_t lane = 0; lane < kNumLanes; ++lane) HIP_CHECK(statuses[lane]);
    std::vector<uint32_t> copied(kNumWords);
    HIP_CHECK(hipMemcpy(copied.data(), dst, kBytes, hipMemcpyDeviceToHost));
    REQUIRE(copied == pattern);
    HIP_CHECK(hipHostFree(statuses));
    HIP_CHECK(hipHostFree(signals));
    HIP_CHECK(hipFree(dst));
    HIP_CHECK(hipHostFree(engine_ids));
    HIP_CHECK(hipHostFree(handles));
  }

  SECTION("Lanes of a wave on different handles, some invalid") {
    constexpr int kMaxHandles = 4;
    constexpr uint32_t kInvalidEvery = 7;
    hipExtCopyEngineDeviceHandle* handles = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&handles),
                            kMaxHandles * sizeof(hipExtCopyEngineDeviceHandle)));
    int num_handles = 0;
    HIP_CHECK(hipExtCopyEngineGetDeviceHandles(kLocalDevice, kLocalDevice, kLocalDevice,
                                               kMaxHandles, handles, &num_handles));
    constexpr uint32_t kNumBlocks = 16;
    constexpr uint32_t kLanesPerBlock = 64;
    constexpr uint32_t kNumLanes = kNumBlocks * kLanesPerBlock;
    constexpr uint64_t kChunkBytes = kBytes / kNumLanes;
    std::vector<uint32_t> pattern(kNumWords);
    for (uint32_t word = 0; word < kNumWords; ++word) pattern[word] = word * 2654435761u + 9u;
    HIP_CHECK(hipMemcpy(src, pattern.data(), kBytes, hipMemcpyHostToDevice));
    char* dst = nullptr;
    uint64_t* signals = nullptr;
    hipError_t* statuses = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    HIP_CHECK(hipMemset(dst, 0, kBytes));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&signals), kNumLanes * sizeof(uint64_t),
                            hipHostMallocCoherent));
    std::fill(signals, signals + kNumLanes, uint64_t{0});
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&statuses), kNumLanes * sizeof(hipError_t)));
    MixedLaneCopies<<<kNumBlocks, kLanesPerBlock>>>(handles, num_handles, dst,
                                                    reinterpret_cast<const char*>(src),
                                                    kChunkBytes, kInvalidEvery, signals, statuses);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint32_t> copied(kNumWords);
    HIP_CHECK(hipMemcpy(copied.data(), dst, kBytes, hipMemcpyDeviceToHost));
    constexpr uint32_t kWordsPerChunk = kChunkBytes / sizeof(uint32_t);
    for (uint32_t lane = 0; lane < kNumLanes; ++lane) {
      INFO("lane " << lane);
      const bool invalid = lane % kInvalidEvery == kInvalidEvery - 1;
      REQUIRE(statuses[lane] == (invalid ? hipErrorInvalidValue : hipSuccess));
      const uint32_t first_word = lane * kWordsPerChunk;
      for (uint32_t word = first_word; word < first_word + kWordsPerChunk; ++word)
        REQUIRE(copied[word] == (invalid ? 0u : pattern[word]));
    }
    HIP_CHECK(hipHostFree(statuses));
    HIP_CHECK(hipHostFree(signals));
    HIP_CHECK(hipFree(dst));
    HIP_CHECK(hipHostFree(handles));
  }

  SECTION("One signal covers every earlier copy") {
    constexpr uint32_t kNumCopies = 256;
    constexpr uint64_t kChunkBytes = kBytes / kNumCopies;
    std::vector<uint32_t> pattern(kNumWords);
    for (uint32_t word = 0; word < kNumWords; ++word) pattern[word] = word * 3266489917u + 1u;
    HIP_CHECK(hipMemcpy(src, pattern.data(), kBytes, hipMemcpyHostToDevice));
    char* dst = nullptr;
    uint64_t* signal = nullptr;
    hipError_t* status = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&signal), sizeof(uint64_t),
                            hipHostMallocCoherent));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&status), sizeof(hipError_t)));
    for (const bool signal_only : {true, false}) {
      INFO((signal_only ? "signal-only submit" : "add signal on the last copy"));
      HIP_CHECK(hipMemset(dst, 0, kBytes));
      *signal = 0;
      CopiesThenOneSignal<<<1, 1>>>(handle, dst, reinterpret_cast<const char*>(src), kChunkBytes,
                                   kNumCopies, signal_only, signal, status);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      if (!signal_only && *status == hipErrorNotSupported) continue;
      HIP_CHECK(*status);
      std::vector<uint32_t> copied(kNumWords);
      HIP_CHECK(hipMemcpy(copied.data(), dst, kBytes, hipMemcpyDeviceToHost));
      REQUIRE(copied == pattern);
    }
    HIP_CHECK(hipHostFree(status));
    HIP_CHECK(hipHostFree(signal));
    HIP_CHECK(hipFree(dst));
  }

  SECTION("A non-blocking submit on an unmet dependency is not ready") {
    uint64_t* gate = nullptr;
    uint64_t* signal = nullptr;
    hipError_t* statuses = nullptr;
    uint32_t* dst = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&gate), sizeof(uint64_t),
                            hipHostMallocCoherent));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&signal), sizeof(uint64_t),
                            hipHostMallocCoherent));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&statuses), 2 * sizeof(hipError_t)));
    *gate = 0;
    *signal = 0;
    NonblockingOnUnmetDependency<<<1, 1>>>(handle, dst, src, kBytes, gate, signal, statuses);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    REQUIRE(statuses[0] == hipErrorNotReady);
    HIP_CHECK(statuses[1]);
    REQUIRE(*signal == 1);
    HIP_CHECK(hipHostFree(statuses));
    HIP_CHECK(hipHostFree(signal));
    HIP_CHECK(hipHostFree(gate));
    HIP_CHECK(hipFree(dst));
  }

  SECTION("Counted completion from every lane") {
    constexpr uint32_t kNumLanes = 256;
    constexpr uint64_t kChunkBytes = kBytes / kNumLanes;
    std::vector<uint32_t> pattern(kNumWords);
    for (uint32_t word = 0; word < kNumWords; ++word) pattern[word] = word * 40503u + 7u;
    HIP_CHECK(hipMemcpy(src, pattern.data(), kBytes, hipMemcpyHostToDevice));
    char* dst = nullptr;
    uint64_t* counter = nullptr;
    hipError_t* statuses = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&counter), sizeof(uint64_t),
                            hipHostMallocCoherent));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&statuses), kNumLanes * sizeof(hipError_t)));
    for (const bool count_down : {false, true}) {
      INFO((count_down ? "subtract down to zero" : "add up to the lane count"));
      HIP_CHECK(hipMemset(dst, 0, kBytes));
      *counter = count_down ? kNumLanes : 0;
      LaneCopiesCountedOnce<<<1, kNumLanes>>>(handle, dst, reinterpret_cast<const char*>(src),
                                              kChunkBytes, counter, count_down, statuses);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      if (statuses[0] == hipErrorNotSupported) {
        INFO("this device's copy engines cannot add to or subtract from a signal");
        continue;
      }
      for (uint32_t lane = 0; lane < kNumLanes; ++lane) HIP_CHECK(statuses[lane]);
      REQUIRE(*counter == (count_down ? 0 : kNumLanes));
      std::vector<uint32_t> copied(kNumWords);
      HIP_CHECK(hipMemcpy(copied.data(), dst, kBytes, hipMemcpyDeviceToHost));
      REQUIRE(copied == pattern);
    }
    HIP_CHECK(hipHostFree(statuses));
    HIP_CHECK(hipHostFree(counter));
    HIP_CHECK(hipFree(dst));
  }

  SECTION("Set completion values across the 32-bit boundary") {
    uint64_t* signal = nullptr;
    hipError_t* status = nullptr;
    uint32_t* dst = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&signal), sizeof(uint64_t),
                            hipHostMallocCoherent));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&status), sizeof(hipError_t)));
    for (const uint64_t value :
         {uint64_t{1}, uint64_t{0xffffffff}, uint64_t{1} << 32, ~uint64_t{1}}) {
      INFO("value 0x" << std::hex << value);
      *signal = 0;
      CopySettingValue<<<1, 1>>>(handle, dst, src, kBytes, signal, value, status);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(*status);
      REQUIRE(*signal == value);
    }
    HIP_CHECK(hipHostFree(status));
    HIP_CHECK(hipHostFree(signal));
    HIP_CHECK(hipFree(dst));
  }

  SECTION("Dependent copies") {
    hipExtCopyEngineDeviceHandle from_host{};
    REQUIRE(GetCopyEngineHandle(kLocalDevice, hipCpuDeviceId, kLocalDevice, &from_host));
    uint32_t* host_src = nullptr;
    uint32_t* staging = nullptr;
    uint32_t* dst = nullptr;
    uint64_t* signals = nullptr;
    hipError_t* status = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&host_src), kBytes, hipHostMallocCoherent));
    for (uint32_t word = 0; word < kNumWords; ++word) host_src[word] = word ^ 0x5a5a5a5au;
    HIP_CHECK(hipMalloc(&staging, kBytes));
    HIP_CHECK(hipMemset(staging, 0, kBytes));
    HIP_CHECK(hipMalloc(&dst, kBytes));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&signals), 2 * sizeof(uint64_t),
                            hipHostMallocCoherent));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&status), sizeof(hipError_t)));
    for (const bool count_down : {false, true}) {
      INFO((count_down ? "countdown signals" : "count-up signals"));
      HIP_CHECK(hipMemset(staging, 0, kBytes));
      HIP_CHECK(hipMemset(dst, 0, kBytes));
      signals[0] = signals[1] = count_down ? 1 : 0;
      if (count_down) {
        CountdownDependentCopies<<<1, 1>>>(from_host, handle, dst, staging, host_src, kBytes,
                                           signals, status);
      } else {
        DependentCopies<<<1, 1>>>(from_host, handle, dst, staging, host_src, kBytes, signals,
                                  status);
      }
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      if (count_down && *status == hipErrorNotSupported) continue;
      HIP_CHECK(*status);
      std::vector<uint32_t> copied(kNumWords);
      HIP_CHECK(hipMemcpy(copied.data(), dst, kBytes, hipMemcpyDeviceToHost));
      REQUIRE(copied == std::vector<uint32_t>(host_src, host_src + kNumWords));
    }
    HIP_CHECK(hipHostFree(status));
    HIP_CHECK(hipHostFree(signals));
    HIP_CHECK(hipFree(dst));
    HIP_CHECK(hipFree(staging));
    HIP_CHECK(hipHostFree(host_src));
  }

  SECTION("A copy without a completion, covered by the next one's") {
    std::vector<uint32_t> pattern(kNumWords);
    for (uint32_t word = 0; word < kNumWords; ++word) pattern[word] = word ^ 0xa5a5a5a5u;
    HIP_CHECK(hipMemcpy(src, pattern.data(), kBytes, hipMemcpyHostToDevice));
    uint32_t* first_dst = nullptr;
    uint32_t* second_dst = nullptr;
    uint64_t* signal = nullptr;
    hipError_t* status = nullptr;
    HIP_CHECK(hipMalloc(&first_dst, kBytes));
    HIP_CHECK(hipMalloc(&second_dst, kBytes));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&signal), sizeof(uint64_t),
                            hipHostMallocCoherent));
    *signal = 0;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&status), sizeof(hipError_t)));
    TwoCopiesOneCompletion<<<1, 1>>>(handle, first_dst, src, second_dst, src, kBytes, signal,
                                     status);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(*status);
    std::vector<uint32_t> copied(kNumWords);
    HIP_CHECK(hipMemcpy(copied.data(), first_dst, kBytes, hipMemcpyDeviceToHost));
    REQUIRE(copied == pattern);
    HIP_CHECK(hipMemcpy(copied.data(), second_dst, kBytes, hipMemcpyDeviceToHost));
    REQUIRE(copied == pattern);
    HIP_CHECK(hipHostFree(status));
    HIP_CHECK(hipHostFree(signal));
    HIP_CHECK(hipFree(second_dst));
    HIP_CHECK(hipFree(first_dst));
  }

  SECTION("Inline writes") {
    uint64_t* signal = nullptr;
    hipError_t* status = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&signal), sizeof(uint64_t),
                            hipHostMallocCoherent));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&status), sizeof(hipError_t)));
    for (const bool to_host : {false, true}) {
      INFO((to_host ? "pinned host memory" : "device memory"));
      constexpr size_t kNumBytes = 32;
      uint8_t* bytes = nullptr;
      if (to_host) {
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&bytes), kNumBytes,
                                hipHostMallocCoherent));
      } else {
        HIP_CHECK(hipMalloc(&bytes, kNumBytes));
      }
      HIP_CHECK(hipMemset(bytes, 0xaa, kNumBytes));
      hipExtCopyEngineDeviceHandle writes{};
      REQUIRE(GetCopyEngineHandle(kLocalDevice, kLocalDevice,
                                  to_host ? hipCpuDeviceId : kLocalDevice, &writes));
      *signal = 0;
      InlineWrites<<<1, 1>>>(writes, bytes, signal, status);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(*status);
      std::vector<uint8_t> expected(kNumBytes, 0xaa);
      const uint32_t word = 0xc0ffee11u;
      const uint64_t pair = 0x0123456789abcdefull;
      const uint16_t half = 0xbeef;
      std::memcpy(&expected[0], &word, sizeof(word));
      std::memcpy(&expected[8], &pair, sizeof(pair));
      expected[17] = 0x5a;
      std::memcpy(&expected[22], &half, sizeof(half));
      std::vector<uint8_t> landed(kNumBytes);
      HIP_CHECK(hipMemcpy(landed.data(), bytes, kNumBytes, hipMemcpyDefault));
      REQUIRE(landed == expected);
      if (to_host) {
        HIP_CHECK(hipHostFree(bytes));
      } else {
        HIP_CHECK(hipFree(bytes));
      }
    }
    HIP_CHECK(hipHostFree(status));
    HIP_CHECK(hipHostFree(signal));
  }

  HIP_CHECK(hipFree(src));
}

#ifdef __linux__
namespace {

constexpr char kKfdNodes[] = "/sys/class/kfd/kfd/topology/nodes/";

// The KFD node of HIP device |device|, matched by PCI location; -1 if none or
// more than one node sits there (a partitioned GPU).
int KfdNodeOf(int device) {
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  const uint64_t bus_and_slot = (uint64_t(props.pciBusID) << 5) | uint64_t(props.pciDeviceID);
  int found = -1;
  for (int node = 0;; ++node) {
    std::ifstream properties(kKfdNodes + std::to_string(node) + "/properties");
    if (!properties) return found;
    std::string key;
    uint64_t value = 0;
    uint64_t location_id = ~uint64_t{0};
    uint64_t domain = ~uint64_t{0};
    while (properties >> key >> value) {
      if (key == "location_id") location_id = value;
      if (key == "domain") domain = value;
    }
    if (domain != uint64_t(props.pciDomainID) || (location_id >> 3) != bus_and_slot) continue;
    if (found != -1) return -1;
    found = node;
  }
}

// The engines KFD recommends for copies from |src_node| to |dst_node| over
// their direct link, or zero when it names none.
uint64_t KfdRecommendedEngines(int src_node, int dst_node) {
  const std::string links = kKfdNodes + std::to_string(src_node) + "/io_links/";
  for (int link = 0;; ++link) {
    std::ifstream properties(links + std::to_string(link) + "/properties");
    if (!properties) return 0;
    std::string key;
    uint64_t value = 0;
    uint64_t node_to = ~uint64_t{0};
    uint64_t recommended = 0;
    while (properties >> key >> value) {
      if (key == "node_to") node_to = value;
      if (key == "recommended_sdma_engine_id_mask") recommended = value;
    }
    if (node_to == static_cast<uint64_t>(dst_node)) return recommended;
  }
}

__global__ void ReadEngineId(hipExtCopyEngineDeviceHandle handle, uint32_t* engine_id) {
  *engine_id = hip::detail::CopyEngineHandle(&handle).engine->engine_id;
}

// |sequence| rises per copy so the reused word never reads as already done.
__global__ void CopyAndWait(hipExtCopyEngineDeviceHandle handle, void* dst, const void* src,
                            uint64_t size, uint64_t* signal, uint64_t sequence,
                            hipError_t* status) {
  const hipExtCopyEngineSignal completion = SetOnCompletion(signal, sequence);
  const hipExtCopyEngineCondition done = Reached(signal, sequence);
  hipError_t result = hipExtCopyEngineSubmitLinear(&handle, dst, src, size, &completion);
  if (result == hipSuccess) result = WaitFor(done, kMaxSpins);
  *status = result;
}

}  // namespace

/**
 * Test Description
 * ------------------------
 * - For every pair of peer devices, the local device's push and pull handles
 *   for the peer must be on the engine KFD recommends for the local device's
 *   link to it (recommended_sdma_engine_id_mask): the copy runs on the local
 *   device's engines in both directions, so both use the local side of the
 *   link. A copy through each handle must land.
 * Test source
 * ------------------------
 *    - unit/copyEngine/hipExtCopyEngine.cc
 * Test requirements
 * ------------------------
 *    - Platform specific (AMD, Linux)
 *    - Multi-device with peer access and KFD recommended SDMA engines
 *    - DEBUG_CLR_ENABLE_ABCE=1 and GPU_MAX_SDMA_DEVICE_QUEUES > 0
 */
HIP_TEST_CASE(Unit_hipExtCopyEngine_PeerCopiesUseRecommendedEngine) {
  constexpr size_t kBytes = 1u << 20;
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  if (num_devices < 2) HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);

  std::vector<int> kfd_nodes(num_devices);
  std::vector<char*> buffers(num_devices);
  for (int device = 0; device < num_devices; ++device) {
    kfd_nodes[device] = KfdNodeOf(device);
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipMalloc(&buffers[device], kBytes));
  }

  uint32_t* engine_id = nullptr;
  hipError_t* status = nullptr;
  uint64_t* signal = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&engine_id), sizeof(uint32_t)));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&status), sizeof(hipError_t)));
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&signal), sizeof(uint64_t),
                                  hipMallocSignalMemory));
  HIP_CHECK(hipMemset(signal, 0, sizeof(uint64_t)));
  uint64_t sequence = 0;

  int num_checked_links = 0;
  for (int local = 0; local < num_devices; ++local) {
    if (kfd_nodes[local] < 0) continue;
    HIP_CHECK(hipSetDevice(local));
    // The engine |handle| is on, after checking a copy through it lands.
    auto engine_of = [&](const hipExtCopyEngineDeviceHandle& handle, void* dst, const void* src) {
      CopyAndWait<<<1, 1>>>(handle, dst, src, kBytes, signal, ++sequence, status);
      ReadEngineId<<<1, 1>>>(handle, engine_id);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(*status);
      return *engine_id;
    };

    for (int peer = 0; peer < num_devices; ++peer) {
      if (peer == local || kfd_nodes[peer] < 0) continue;
      int can_access_peer = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, local, peer));
      if (!can_access_peer) continue;
      const uint64_t recommended = KfdRecommendedEngines(kfd_nodes[local], kfd_nodes[peer]);
      if (recommended == 0) continue;
      const hipError_t peer_status = hipDeviceEnablePeerAccess(peer, 0);
      if (peer_status != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(peer_status);
      hipExtCopyEngineDeviceHandle push{};
      hipExtCopyEngineDeviceHandle pull{};
      if (!GetCopyEngineHandle(local, local, peer, &push) ||
          !GetCopyEngineHandle(local, peer, local, &pull))
        HIP_SKIP_TEST(kCopyEngineUnavailable);

      INFO("device " << local << " (KFD node " << kfd_nodes[local] << ") <-> device " << peer
                     << " (KFD node " << kfd_nodes[peer] << "), recommended engine mask 0x"
                     << std::hex << recommended);
      const uint32_t push_engine = engine_of(push, buffers[peer], buffers[local]);
      const uint32_t pull_engine = engine_of(pull, buffers[local], buffers[peer]);
      CHECK(((recommended >> push_engine) & 1u) == 1u);
      CHECK(((recommended >> pull_engine) & 1u) == 1u);
      ++num_checked_links;
    }
  }

  HIP_CHECK(hipFree(signal));
  HIP_CHECK(hipHostFree(status));
  HIP_CHECK(hipHostFree(engine_id));
  for (int device = 0; device < num_devices; ++device) {
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipFree(buffers[device]));
  }
  if (num_checked_links == 0) HIP_SKIP_TEST("no peer link has a KFD-recommended SDMA engine");
}
#endif  // __linux__

/**
 * End doxygen group hipExtCopyEngineGetDeviceHandles.
 * @}
 */
