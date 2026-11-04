/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipMemImportFromShareableHandle hipMemImportFromShareableHandle
 * @{
 * @ingroup PerformanceTestVMM
 * `hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void* osHandle,
 *                                  hipMemAllocationHandleType shHandleType)` -
 * Imports an allocation from a requested shareable handle type.
 */

#include <hip_test_common.hh>
#include <hip_test_params.hh>

#include <algorithm>
#include <chrono>
#include <vector>

#if HT_AMD && defined(__linux__)
#include <unistd.h>

namespace {
constexpr int kMaxPeers = 7;                    // device 0 receives from devices 1..7
constexpr size_t kMinChunkBytes = size_t(16) << 20;  // per peer, above ROC_P2P_SDMA_SIZE
constexpr int kAttempts = 3;  // each attempt times both cases, results are reported per attempt

// Median time, in microseconds, of one hipMemcpyBatchAsync that copies from every source buffer
// into its own slice of dst, over `iterations` timed runs after `warmups` untimed ones.
double MedianBatchCopyTimeUs(const std::vector<void*>& srcs, void* dst, size_t chunk,
                             hipStream_t stream, int warmups, int iterations) {
  const size_t count = srcs.size();
  std::vector<void*> dsts(count);
  std::vector<void*> src_ptrs(srcs);
  std::vector<size_t> sizes(count, chunk);
  for (size_t i = 0; i < count; ++i) {
    dsts[i] = static_cast<char*>(dst) + i * chunk;
  }
  hipMemcpyAttributes attrs = {};
  attrs.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attrs.flags = hipMemcpyFlagPreferOverlapWithCompute;
  size_t attrs_idx = 0;
  size_t fail_idx = 0;
  std::vector<double> times;
  for (int it = -warmups; it < iterations; ++it) {
    const auto start = std::chrono::steady_clock::now();
    HIP_CHECK(hipMemcpyBatchAsync(dsts.data(), src_ptrs.data(), sizes.data(), count, &attrs,
                                  &attrs_idx, 1, &fail_idx, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    if (it >= 0) {
      times.push_back(
          std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
              .count());
    }
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

// Owns everything the test allocates on devices 0..N and releases it on every exit path, including
// a failing assertion halfway through the setup. Errors during release are ignored on purpose.
struct PeerBuffers {
  struct Peer {
    int device = 0;
    hipMemGenericAllocationHandle_t allocation = {};
    hipMemGenericAllocationHandle_t imported = {};
    bool has_allocation = false;
    bool has_imported = false;
    void* va = nullptr;
    bool mapped = false;
    void* device_buffer = nullptr;  // hipMalloc reference buffer on the peer
    bool peer_access_enabled = false;
  };
  std::vector<Peer> peers;
  size_t chunk = 0;
  void* dst = nullptr;
  hipStream_t stream = nullptr;

  PeerBuffers() = default;
  PeerBuffers(const PeerBuffers&) = delete;
  PeerBuffers& operator=(const PeerBuffers&) = delete;

  ~PeerBuffers() {
    (void)hipSetDevice(0);
    if (stream != nullptr) (void)hipStreamDestroy(stream);
    if (dst != nullptr) (void)hipFree(dst);
    for (auto& peer : peers) {
      if (peer.peer_access_enabled) (void)hipDeviceDisablePeerAccess(peer.device);
      if (peer.device_buffer != nullptr) (void)hipFree(peer.device_buffer);
      if (peer.mapped) (void)hipMemUnmap(peer.va, chunk);
      if (peer.has_imported) (void)hipMemRelease(peer.imported);
      if (peer.has_allocation) (void)hipMemRelease(peer.allocation);
      if (peer.va != nullptr) (void)hipMemAddressFree(peer.va, chunk);
    }
  }
};
}  // namespace

/**
 * Test Description
 * ------------------------
 *    - ROCM-33394. Device 0 receives from up to 7 peers with a single hipMemcpyBatchAsync, once
 *      from VMM allocations created on the peers and imported on device 0 through a POSIX fd (the
 *      pattern RCCL's copy engine collectives use with NCCL_CUMEM_ENABLE=1), once from hipMalloc
 *      buffers with peer access enabled. The imported case is expected to be as fast as the peer
 *      access case: when the runtime takes the imported peer buffers for local memory, the copies
 *      lose their per-peer transfer engine and the imported case is much slower (about 1.75x on
 *      8 x MI300X, versus about 1.00x when the owner is resolved correctly).
 *    - The device name, architecture, both medians and the ratio are printed for each attempt,
 *      with the median ratio at the end. Like the other performance tests it does not fail on the
 *      timing: the results are tracked outside of the test, and a limit depends on the product and
 *      the topology. The number of warm-up and timed iterations comes from the current test level
 *      (definitions.yaml).
 *    - Meant for weekly or manual performance runs (BUILD_PERF_TESTS=ON), not for the functional
 *      CI. The functional coverage of the same import path is
 *      Unit_hipMemImportFromShareableHandle_MulProc_CloseFdBeforeMapAndBatchCopy.
 * Test source
 * ------------------------
 *    - performance/scenarios/vmm/hipPerfVMMImportedPeersBatchCopy.cc
 * Test requirements
 * ------------------------
 *    - Host specific (LINUX)
 *    - At least 3 devices (device 0 and 2 to 7 peers), with peer access from device 0 to them
 *    - HIP_VERSION >= 7.1
 */
HIP_TEST_CASE(Performance_hipMemImportFromShareableHandle_ImportedPeersBatchCopy) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 3) {
    HIP_SKIP_TEST(HipTest::SkipReason::kRequiredDeviceCountNotMet);
  }
  const int num_peers = std::min(device_count - 1, kMaxPeers);
  for (int dev = 0; dev <= num_peers; ++dev) {
    int vmm_supported = 0;
    HIP_CHECK(hipDeviceGetAttribute(
        &vmm_supported, hipDeviceAttributeVirtualMemoryManagementSupported, dev));
    if (vmm_supported == 0) {
      HIP_SKIP_TEST(HipTest::SkipReason::kVmmUnsupported);
    }
    if (dev != 0) {
      int can_access_peer = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 0, dev));
      if (can_access_peer == 0) {
        HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
      }
    }
  }

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  prop.location.type = hipMemLocationTypeDevice;
  size_t granularity = 0;
  for (int dev = 0; dev <= num_peers; ++dev) {
    prop.location.id = dev;
    size_t device_granularity = 0;
    HIP_CHECK(hipMemGetAllocationGranularity(&device_granularity, &prop,
                                             hipMemAllocationGranularityMinimum));
    granularity = std::max(granularity, device_granularity);
  }
  REQUIRE(granularity > 0);
  // Every device is given the same chunk size, a multiple of the largest granularity.
  const size_t chunk = (kMinChunkBytes + granularity - 1) / granularity * granularity;

  hipDeviceProp_t device_props;
  HIP_CHECK(hipGetDeviceProperties(&device_props, 0));

  PeerBuffers buffers;
  buffers.chunk = chunk;
  buffers.peers.resize(num_peers);
  std::vector<void*> imported_ptrs;
  std::vector<void*> peer_ptrs;
  for (int i = 0; i < num_peers; ++i) {
    auto& peer = buffers.peers[i];
    peer.device = i + 1;
    prop.location.id = peer.device;
    HIP_CHECK(hipSetDevice(peer.device));
    HIP_CHECK(hipMemCreate(&peer.allocation, chunk, &prop, 0));
    peer.has_allocation = true;
    int shareable_handle = -1;
    HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, peer.allocation,
                                            hipMemHandleTypePosixFileDescriptor, 0));
    HIP_CHECK(hipSetDevice(0));
    const hipError_t import_status = hipMemImportFromShareableHandle(
        &peer.imported, reinterpret_cast<void*>(static_cast<uintptr_t>(shareable_handle)),
        hipMemHandleTypePosixFileDescriptor);
    // The fd is not needed once the handle is imported, whatever the result.
    REQUIRE(close(shareable_handle) == 0);
    HIP_CHECK(import_status);
    peer.has_imported = true;
    HIP_CHECK(hipMemAddressReserve(&peer.va, chunk, 0, nullptr, 0));
    HIP_CHECK(hipMemMap(peer.va, chunk, 0, peer.imported, 0));
    peer.mapped = true;
    hipMemAccessDesc access_desc = {};
    access_desc.location.type = hipMemLocationTypeDevice;
    access_desc.location.id = 0;
    access_desc.flags = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK(hipMemSetAccess(peer.va, chunk, &access_desc, 1));
    imported_ptrs.push_back(peer.va);

    HIP_CHECK(hipSetDevice(peer.device));
    HIP_CHECK(hipMalloc(&peer.device_buffer, chunk));
    HIP_CHECK(hipSetDevice(0));
    const hipError_t peer_status = hipDeviceEnablePeerAccess(peer.device, 0);
    REQUIRE((peer_status == hipSuccess || peer_status == hipErrorPeerAccessAlreadyEnabled));
    // Only undo what this test enabled.
    peer.peer_access_enabled = (peer_status == hipSuccess);
    peer_ptrs.push_back(peer.device_buffer);
  }

  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipMalloc(&buffers.dst, chunk * num_peers));
  HIP_CHECK(hipStreamCreate(&buffers.stream));

  const int warmups = TestParameterStore::instance().getWarmupsForCurrentLevel();
  const int iterations = TestParameterStore::instance().getIterationsForCurrentLevel();
  std::vector<double> ratios;
  for (int attempt = 0; attempt < kAttempts; ++attempt) {
    const double imported_us =
        MedianBatchCopyTimeUs(imported_ptrs, buffers.dst, chunk, buffers.stream, warmups, iterations);
    const double reference_us =
        MedianBatchCopyTimeUs(peer_ptrs, buffers.dst, chunk, buffers.stream, warmups, iterations);
    ratios.push_back(imported_us / reference_us);
    CONSOLE_PRINT("%s (%s): %d peers x %zu MiB, attempt %d: imported VMM %.1f us, hipMalloc + peer "
                  "access %.1f us, ratio %.2f\n",
                  device_props.name, device_props.gcnArchName, num_peers, chunk >> 20,
                  attempt, imported_us, reference_us, ratios.back());
  }
  std::sort(ratios.begin(), ratios.end());
  CONSOLE_PRINT("median ratio imported / peer access: %.2f (%d warm-up, %d timed iterations)\n",
                ratios[ratios.size() / 2], warmups, iterations);
}
#endif  // HT_AMD && __linux__

/**
 * End doxygen group hipMemImportFromShareableHandle.
 * @}
 */
