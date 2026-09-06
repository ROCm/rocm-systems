/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <hip/hip_ext_copy_engine.h>

#include <fstream>
#include <string>
#include <vector>

/**
 * @addtogroup hipExtCopyEngineGetDeviceHandle hipExtCopyEngineGetDeviceHandle
 * @{
 * @ingroup PeerToPeerTest
 * `hipError_t hipExtCopyEngineGetDeviceHandle(int device, hipExtCopyEngineDeviceHandle* handle);`
 * - Returns the handle a kernel submits SDMA copies through with
 *   hipExtCopyEngineSubmitLinear, and waits on with hipExtCopyEngineTokenWait.
 */

namespace {

constexpr uint64_t kMaxSpins = 1ull << 34;

constexpr char kCopyEngineUnavailable[] =
    "kernel-side copies are off (set DEBUG_CLR_ENABLE_ABCE=1 and GPU_MAX_SDMA_DEVICE_QUEUES=16) "
    "or unsupported on this GPU";

bool GetCopyEngineHandle(int device, hipExtCopyEngineDeviceHandle* handle) {
  const hipError_t status = hipExtCopyEngineGetDeviceHandle(device, handle);
  if (status == hipErrorNotSupported) return false;
  HIP_CHECK(status);
  return true;
}

// A completion token on |signal|. The submit arms the word; the copy engine
// lands it on the reference when the copy retires.
__device__ hipExtCopyEngineToken CompletionOn(uint64_t* signal) {
  return hipExtCopyEngineToken{reinterpret_cast<uint64_t>(signal), 0};
}

// The block produces the source, one thread submits and waits, and the whole
// block then checks the destination. The barrier puts every thread's stores
// under the submitting thread's release; the wait acquires the engine's writes.
// |dst_device| is a HIP ordinal or HIP_EXT_COPY_ENGINE_AGENT_HOST; naming the
// ends only steers which engine serves the copy.
__global__ void ProduceCopyConsume(hipExtCopyEngineDeviceHandle handle, uint32_t* dst,
                                   uint32_t* src, uint32_t num_words, int dst_device,
                                   uint64_t* signal, hipError_t* status,
                                   uint32_t* num_mismatches) {
  for (uint32_t word = threadIdx.x; word < num_words; word += blockDim.x) src[word] = ~word;
  __syncthreads();
  if (threadIdx.x == 0) {
    const hipExtCopyEngineToken completion = CompletionOn(signal);
    hipError_t result = hipExtCopyEngineSubmitLinear(
        &handle, dst, src, uint64_t{num_words} * sizeof(uint32_t), &completion, nullptr,
        hipExtCopyEngineLocalAgent(&handle), hipExtCopyEngineAgentFor(&handle, dst_device));
    if (result == hipSuccess) result = hipExtCopyEngineTokenWait(&completion, kMaxSpins);
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
  const hipExtCopyEngineToken completion = CompletionOn(&signals[lane]);
  hipError_t result =
      hipExtCopyEngineSubmitLinear(&handle, dst + lane * chunk_bytes, src + lane * chunk_bytes,
                                   chunk_bytes, &completion);
  if (result == hipSuccess) result = hipExtCopyEngineTokenWait(&completion, kMaxSpins);
  statuses[lane] = result;
}

// One copy whose completion lands on |reference|, so a test can pick values
// whose arming (reference + 1) changes the high word.
__global__ void CopyToReference(hipExtCopyEngineDeviceHandle handle, void* dst, const void* src,
                                uint64_t size, uint64_t* signal, uint64_t reference,
                                hipError_t* status) {
  const hipExtCopyEngineToken completion{reinterpret_cast<uint64_t>(signal), reference};
  hipError_t result = hipExtCopyEngineSubmitLinear(&handle, dst, src, size, &completion);
  if (result == hipSuccess) result = hipExtCopyEngineTokenWait(&completion, kMaxSpins);
  *status = result;
}

// Two copies submitted back to back, the second reading what the first writes:
// a slow one from host memory into a staging buffer, then a fast device-local
// one out of it, on different engines. Only the dependency keeps the second
// from reading the staging buffer early; the thread waits once, on the last
// copy.
__global__ void DependentCopies(hipExtCopyEngineDeviceHandle handle, void* dst, void* staging,
                                const void* host_src, uint64_t size, uint64_t* signals,
                                hipError_t* status) {
  const int8_t local = hipExtCopyEngineLocalAgent(&handle);
  const hipExtCopyEngineToken staged = CompletionOn(&signals[0]);
  const hipExtCopyEngineToken delivered = CompletionOn(&signals[1]);
  hipError_t result = hipExtCopyEngineSubmitLinear(&handle, staging, host_src, size, &staged,
                                                   nullptr, HIP_EXT_COPY_ENGINE_AGENT_HOST, local);
  if (result == hipSuccess) {
    result = hipExtCopyEngineSubmitLinear(&handle, dst, staging, size, &delivered, &staged, local,
                                          local);
  }
  if (result == hipSuccess) result = hipExtCopyEngineTokenWait(&delivered, kMaxSpins);
  *status = result;
}

struct ProduceCopyConsumeResult {
  hipError_t status;
  uint32_t num_mismatches;
};

ProduceCopyConsumeResult RunProduceCopyConsume(const hipExtCopyEngineDeviceHandle& handle,
                                               uint32_t* dst, uint32_t* src, uint32_t num_words,
                                               int dst_device) {
  uint64_t* signal = nullptr;
  ProduceCopyConsumeResult* result = nullptr;
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&signal), sizeof(uint64_t),
                                  hipMallocSignalMemory));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&result), sizeof(*result)));
  *result = {hipErrorUnknown, 0};
  ProduceCopyConsume<<<1, 256>>>(handle, dst, src, num_words, dst_device, signal,
                                 &result->status, &result->num_mismatches);
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
 * - Validates argument handling: a null handle is rejected with
 *   hipErrorInvalidValue and an out-of-range ordinal with hipErrorInvalidDevice.
 * Test source
 * ------------------------
 *    - unit/copyEngine/hipExtCopyEngine.cc
 * Test requirements
 * ------------------------
 *    - Platform specific (AMD)
 */
HIP_TEST_CASE(Unit_hipExtCopyEngineGetDeviceHandle_Negative) {
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  hipExtCopyEngineDeviceHandle handle{};
  HIP_CHECK_ERROR(hipExtCopyEngineGetDeviceHandle(0, nullptr), hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipExtCopyEngineGetDeviceHandle(-1, &handle), hipErrorInvalidDevice);
  HIP_CHECK_ERROR(hipExtCopyEngineGetDeviceHandle(num_devices, &handle), hipErrorInvalidDevice);
  HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidDevice);
}

/**
 * Test Description
 * ------------------------
 * - A kernel produces data, copies it with hipExtCopyEngineSubmitLinear, waits on
 *   the completion token and checks the destination, for a destination on the
 *   same device, in pinned host memory and on a peer device.
 * - 256 lanes each submit and wait on their own copy at once.
 * - Completion values whose arming crosses the 32-bit boundary still retire on
 *   the exact value.
 * - Two copies on different engines are submitted back to back, the second
 *   reading what the first writes, ordered only by passing the first's
 *   completion as the second's dependency.
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
  if (!GetCopyEngineHandle(kLocalDevice, &handle)) HIP_SKIP_TEST(kCopyEngineUnavailable);

  uint32_t* src = nullptr;
  HIP_CHECK(hipMalloc(&src, kBytes));

  SECTION("Same device") {
    uint32_t* dst = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    const ProduceCopyConsumeResult result =
        RunProduceCopyConsume(handle, dst, src, kNumWords, kLocalDevice);
    HIP_CHECK(result.status);
    REQUIRE(result.num_mismatches == 0);
    HIP_CHECK(hipFree(dst));
  }

  SECTION("Pinned host memory") {
    uint32_t* dst = nullptr;
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&dst), kBytes, hipHostMallocCoherent));
    const ProduceCopyConsumeResult result =
        RunProduceCopyConsume(handle, dst, src, kNumWords, HIP_EXT_COPY_ENGINE_AGENT_HOST);
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
    const ProduceCopyConsumeResult result =
        RunProduceCopyConsume(handle, dst, src, kNumWords, kPeerDevice);
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

  SECTION("Completion values across the 32-bit boundary") {
    uint64_t* signal = nullptr;
    hipError_t* status = nullptr;
    uint32_t* dst = nullptr;
    HIP_CHECK(hipMalloc(&dst, kBytes));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&signal), sizeof(uint64_t),
                            hipHostMallocCoherent));
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&status), sizeof(hipError_t)));
    for (const uint64_t reference :
         {uint64_t{0}, uint64_t{0xffffffff}, uint64_t{1} << 32, ~uint64_t{1}}) {
      INFO("reference 0x" << std::hex << reference);
      CopyToReference<<<1, 1>>>(handle, dst, src, kBytes, signal, reference, status);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(*status);
      REQUIRE(*signal == reference);
    }
    HIP_CHECK(hipHostFree(status));
    HIP_CHECK(hipHostFree(signal));
    HIP_CHECK(hipFree(dst));
  }

  SECTION("Dependent copies") {
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
    DependentCopies<<<1, 1>>>(handle, dst, staging, host_src, kBytes, signals, status);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(*status);
    std::vector<uint32_t> copied(kNumWords);
    HIP_CHECK(hipMemcpy(copied.data(), dst, kBytes, hipMemcpyDeviceToHost));
    REQUIRE(copied == std::vector<uint32_t>(host_src, host_src + kNumWords));
    HIP_CHECK(hipHostFree(status));
    HIP_CHECK(hipHostFree(signals));
    HIP_CHECK(hipFree(dst));
    HIP_CHECK(hipFree(staging));
    HIP_CHECK(hipHostFree(host_src));
  }

  HIP_CHECK(hipFree(src));
}

#ifdef __linux__
namespace {

constexpr uint32_t kMaxEngines = ABCE_DEVICE_MAX_ENGINES;

// The engines KFD recommends for copies from |src_node| to |dst_node| over
// their direct link, or zero when it names none.
uint64_t KfdRecommendedEngines(int src_node, int dst_node) {
  const std::string links = "/sys/class/kfd/kfd/topology/nodes/" + std::to_string(src_node) +
                            "/io_links/";
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

// Reads the runtime's published device view: the engines with a kernel ring,
// and each ring's write index, which advances once per copy submitted to it.
__global__ void ReadRings(hipExtCopyEngineDeviceHandle handle, uint64_t* usable_engines,
                          uint64_t* write_indices) {
  const abce_device_engine_set_t* engine_set = hip::detail::CopyEngineHandle(&handle).engine_set;
  const uint32_t engine = threadIdx.x;
  const bool usable = (engine_set->device_usable_engines >> engine) & 1u;
  if (engine == 0) *usable_engines = engine_set->device_usable_engines;
  write_indices[engine] = usable ? __hip_atomic_load(engine_set->engines[engine].write_ptr,
                                                     __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)
                                 : 0;
}

__global__ void AgentIds(hipExtCopyEngineDeviceHandle handle, int num_devices, int8_t* agents) {
  const int device = threadIdx.x;
  if (device < num_devices) agents[device] = hipExtCopyEngineAgentFor(&handle, device);
}

__global__ void CopyAndWait(hipExtCopyEngineDeviceHandle handle, void* dst, const void* src,
                            uint64_t size, int8_t src_agent, int8_t dst_agent, uint64_t* signal,
                            hipError_t* status) {
  const hipExtCopyEngineToken completion = CompletionOn(signal);
  hipError_t result = hipExtCopyEngineSubmitLinear(&handle, dst, src, size, &completion, nullptr,
                                                   src_agent, dst_agent);
  if (result == hipSuccess) result = hipExtCopyEngineTokenWait(&completion, kMaxSpins);
  *status = result;
}

}  // namespace

/**
 * Test Description
 * ------------------------
 * - For every pair of peer devices, a kernel on the local device pushes a copy to
 *   the peer and pulls one back, naming both ends with hipExtCopyEngineAgentFor.
 *   Each copy must run on the engine KFD recommends for the local device's link
 *   to that peer (recommended_sdma_engine_id_mask): the copy runs on the local
 *   device's engines in both directions, so both use the local side of the link.
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

  std::vector<hipExtCopyEngineDeviceHandle> handles(num_devices);
  std::vector<char*> buffers(num_devices);
  for (int device = 0; device < num_devices; ++device) {
    HIP_CHECK(hipSetDevice(device));
    if (!GetCopyEngineHandle(device, &handles[device])) HIP_SKIP_TEST(kCopyEngineUnavailable);
    HIP_CHECK(hipMalloc(&buffers[device], kBytes));
  }

  int8_t* agents = nullptr;
  uint64_t* usable_engines = nullptr;
  uint64_t* write_indices = nullptr;
  hipError_t* status = nullptr;
  uint64_t* signal = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&agents), num_devices));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&usable_engines), sizeof(uint64_t)));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&write_indices),
                          2 * kMaxEngines * sizeof(uint64_t)));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&status), sizeof(hipError_t)));
  HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&signal), sizeof(uint64_t),
                                  hipMallocSignalMemory));

  int num_checked_links = 0;
  for (int local = 0; local < num_devices; ++local) {
    HIP_CHECK(hipSetDevice(local));
    AgentIds<<<1, 64>>>(handles[local], num_devices, agents);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // The one engine whose write index moved across a single copy.
    auto engine_used = [&](void* dst, const void* src, int8_t src_agent, int8_t dst_agent) {
      uint64_t* indices_before = write_indices;
      uint64_t* indices_after = write_indices + kMaxEngines;
      ReadRings<<<1, kMaxEngines>>>(handles[local], usable_engines, indices_before);
      CopyAndWait<<<1, 1>>>(handles[local], dst, src, kBytes, src_agent, dst_agent, signal,
                            status);
      ReadRings<<<1, kMaxEngines>>>(handles[local], usable_engines, indices_after);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(*status);
      int advanced_engine = -1;
      for (uint32_t engine = 0; engine < kMaxEngines; ++engine) {
        if (indices_after[engine] == indices_before[engine]) continue;
        REQUIRE(advanced_engine == -1);
        advanced_engine = static_cast<int>(engine);
      }
      REQUIRE(advanced_engine >= 0);
      return advanced_engine;
    };

    for (int peer = 0; peer < num_devices; ++peer) {
      if (peer == local) continue;
      int can_access_peer = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, local, peer));
      if (!can_access_peer) continue;
      const hipError_t peer_status = hipDeviceEnablePeerAccess(peer, 0);
      if (peer_status != hipErrorPeerAccessAlreadyEnabled) HIP_CHECK(peer_status);

      const uint64_t recommended = KfdRecommendedEngines(agents[local], agents[peer]);
      // Without a recommendation, or with the recommended engine lacking a
      // kernel ring (its queues may all be in use), any legal engine is right.
      if (recommended == 0) continue;
      ReadRings<<<1, kMaxEngines>>>(handles[local], usable_engines, write_indices);
      HIP_CHECK(hipDeviceSynchronize());
      if ((recommended & *usable_engines) == 0) continue;

      INFO("device " << local << " (KFD node " << int{agents[local]} << ") <-> device " << peer
                     << " (KFD node " << int{agents[peer]} << "), recommended engine mask 0x"
                     << std::hex << recommended);
      const int push_engine =
          engine_used(buffers[peer], buffers[local], agents[local], agents[peer]);
      const int pull_engine =
          engine_used(buffers[local], buffers[peer], agents[peer], agents[local]);
      CHECK(((recommended >> push_engine) & 1u) == 1u);
      CHECK(((recommended >> pull_engine) & 1u) == 1u);
      ++num_checked_links;
    }
  }

  HIP_CHECK(hipFree(signal));
  HIP_CHECK(hipHostFree(status));
  HIP_CHECK(hipHostFree(write_indices));
  HIP_CHECK(hipHostFree(usable_engines));
  HIP_CHECK(hipHostFree(agents));
  for (int device = 0; device < num_devices; ++device) {
    HIP_CHECK(hipSetDevice(device));
    HIP_CHECK(hipFree(buffers[device]));
  }
  if (num_checked_links == 0) HIP_SKIP_TEST("no peer link has a KFD-recommended SDMA engine");
}
#endif  // __linux__

/**
 * End doxygen group hipExtCopyEngineGetDeviceHandle.
 * @}
 */
