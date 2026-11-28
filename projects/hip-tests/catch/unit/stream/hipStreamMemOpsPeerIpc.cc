/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <resource_guards.hh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#ifdef __linux__
#include <unistd.h>
#endif

/**
 * @addtogroup hipStreamBatchMemOp hipStreamBatchMemOp
 * @{
 * @ingroup StreamTest
 * Stream memory operations targeting memory that was exported and imported through the virtual
 * memory management APIs and shared between devices.
 *
 * Existing stream-memory-op coverage only ever targets memory the calling device allocated
 * itself. Communication libraries do the opposite: they build a barrier out of
 * `hipStreamWaitValue32` / `hipStreamWriteValue32` / `hipStreamBatchMemOp` whose operands live in
 * an imported window that every participating device has been granted access to. These tests pin
 * that usage.
 */

#ifdef __linux__

namespace {
constexpr uint32_t kBarrierToken = 0x5eed0001u;
constexpr int kSentinelValue = 0x1234;
constexpr int kMaxBarrierDevices = 4;

/* How long a gated stream is given to run before its sentinel is sampled. Long enough that a
   stream-memory-op that never held would have let its kernel through by now. */
constexpr std::chrono::milliseconds kGateSettleDelay{200};
/* Upper bound on how long a released barrier may take to drain before it counts as stalled. */
constexpr std::chrono::seconds kStreamWaitTimeout{30};
/* How often WaitForStreams re-queries the streams it is waiting on. */
constexpr std::chrono::milliseconds kStreamPollInterval{1};

typedef int ShareableHandle;

void CheckVmmSupported(int device_ordinal) {
  int value = 0;
  HIP_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeVirtualMemoryManagementSupported,
                                  device_ordinal));
  if (value == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kVmmUnsupported);
  }
}

void CheckStreamWaitValueSupported(int device_ordinal) {
  int value = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&value, hipDeviceAttributeCanUseStreamWaitValue, device_ordinal));
  if (value == 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kStreamWaitValueUnsupported);
  }
}

size_t RoundUpToGranularity(size_t size, int owner_device) {
  hipMemAllocationProp prop{};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = owner_device;
  size_t granularity = 0;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum));
  REQUIRE(granularity > 0);
  return ((size + granularity - 1) / granularity) * granularity;
}

/**
 * Creates an allocation on `owner_device`, exports it as a POSIX fd, imports it back, maps it,
 * and grants read-write access to every device in `access_devices` with a single
 * `hipMemSetAccess` call over the whole range -- the multi-device access-descriptor array shape a
 * communication library uses to publish a window to all of its ranks. The window is unmapped and
 * both handles released when the guard goes out of scope, so a failed assertion cannot leak it.
 */
class ImportedWindowGuard {
 public:
  ImportedWindowGuard(int owner_device, const std::vector<int>& access_devices,
                      size_t requested_size)
      : size_{RoundUpToGranularity(requested_size, owner_device)} {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = owner_device;

    HIP_CHECK(hipSetDevice(owner_device));
    HIP_CHECK(hipMemCreate(&allocation_handle_, size_, &prop, 0));

    ShareableHandle shareable_handle = -1;
    HIP_CHECK(hipMemExportToShareableHandle(&shareable_handle, allocation_handle_,
                                            hipMemHandleTypePosixFileDescriptor, 0));
    HIP_CHECK(hipMemImportFromShareableHandle(
        &imported_handle_, reinterpret_cast<void*>(static_cast<uintptr_t>(shareable_handle)),
        hipMemHandleTypePosixFileDescriptor));
    REQUIRE(close(shareable_handle) == 0);

    HIP_CHECK(hipMemAddressReserve(&ptr_, size_, 0, nullptr, 0));
    HIP_CHECK(hipMemMap(ptr_, size_, 0, imported_handle_, 0));

    std::vector<hipMemAccessDesc> access_descs;
    access_descs.reserve(access_devices.size());
    for (int device : access_devices) {
      hipMemAccessDesc desc{};
      desc.location.type = hipMemLocationTypeDevice;
      desc.location.id = device;
      desc.flags = hipMemAccessFlagsProtReadWrite;
      access_descs.push_back(desc);
    }
    HIP_CHECK(hipMemSetAccess(ptr_, size_, access_descs.data(), access_descs.size()));
  }

  ImportedWindowGuard(const ImportedWindowGuard&) = delete;
  ImportedWindowGuard(ImportedWindowGuard&&) = delete;
  ImportedWindowGuard& operator=(const ImportedWindowGuard&) = delete;
  ImportedWindowGuard& operator=(ImportedWindowGuard&&) = delete;

  ~ImportedWindowGuard() {
    // No Catch macros, don't want to possibly throw in the destructor
    if (ptr_ != nullptr) {
      static_cast<void>(hipMemUnmap(ptr_, size_));
      static_cast<void>(hipMemAddressFree(ptr_, size_));
      static_cast<void>(hipMemRelease(imported_handle_));
      static_cast<void>(hipMemRelease(allocation_handle_));
    }
  }

  template <typename T> T* ptr() const { return reinterpret_cast<T*>(ptr_); }

 private:
  const size_t size_;
  hipMemGenericAllocationHandle_t allocation_handle_{};
  hipMemGenericAllocationHandle_t imported_handle_{};
  void* ptr_ = nullptr;
};

/* Returns the first `count` devices that can all reach each other, or an empty vector. */
std::vector<int> GetMutuallyPeerAccessibleDevices(int count) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < count) {
    return {};
  }

  std::vector<int> devices;
  for (int device = 0; device < count; ++device) {
    devices.push_back(device);
  }

  for (int lhs : devices) {
    for (int rhs : devices) {
      if (lhs == rhs) {
        continue;
      }
      int can_access = 0;
      HIP_CHECK(hipDeviceCanAccessPeer(&can_access, lhs, rhs));
      if (can_access == 0) {
        return {};
      }
    }
  }
  return devices;
}

/* Waits until every stream has completed, or the deadline expires. */
bool WaitForStreams(const std::vector<hipStream_t>& streams, std::chrono::seconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  do {
    bool all_done = true;
    for (hipStream_t stream : streams) {
      if (hipStreamQuery(stream) != hipSuccess) {
        (void)hipGetLastError();
        all_done = false;
      }
    }
    if (all_done) {
      return true;
    }
    std::this_thread::sleep_for(kStreamPollInterval);
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}
}  // namespace

static __global__ void write_sentinel_kernel(int* out, int value) { *out = value; }

/**
 * Test Description
 * ------------------------
 *  - Creates a window on device 0, imports it, and grants both device 0 and device 1 access with
 *    one `hipMemSetAccess` call.
 *  - Device 1 enqueues `hipStreamWaitValue32` on a flag inside that window, followed by a kernel.
 *  - Verifies the kernel is still gated while the flag is unset, then has device 0 release it
 *    with `hipStreamWriteValue32` and verifies device 1 makes progress and the kernel ran.
 *  - This is the cross-device signalling path a communication library's copy-engine collectives
 *    depend on; it is not covered by the single-device stream-memory-op tests.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamMemOpsPeerIpc.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - Virtual memory management supported
 *  - Stream wait-value support
 *  - Multi-device, with peer access between device 0 and device 1
 *  - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipStreamMemOps_PeerImport_WaitValueGatedByPeerWrite) {
  const std::vector<int> devices = GetMutuallyPeerAccessibleDevices(2);
  if (devices.empty()) {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    const char* const reason = (device_count < 2) ? HipTest::SkipReason::kFewerThanTwoGpus
                                                  : HipTest::SkipReason::kPeerAccessUnavailable;
    HIP_SKIP_TEST(reason);
  }
  CheckVmmSupported(devices[0]);
  CheckVmmSupported(devices[1]);
  CheckStreamWaitValueSupported(devices[1]);

  // Declared before the streams and the sentinel so that it is torn down after them.
  const ImportedWindowGuard window(devices[0], devices, sizeof(uint32_t));
  uint32_t* flag = window.ptr<uint32_t>();

  HIP_CHECK(hipSetDevice(devices[0]));
  const StreamGuard writer(Streams::withFlags, hipStreamNonBlocking);
  const hipStream_t writer_stream = writer.stream();
  HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(flag), 0, 1));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipSetDevice(devices[1]));
  const StreamGuard waiter(Streams::withFlags, hipStreamNonBlocking);
  const hipStream_t waiter_stream = waiter.stream();
  LinearAllocGuard<int> sentinel_alloc(LinearAllocs::hipMalloc, sizeof(int));
  int* const sentinel = sentinel_alloc.ptr();
  HIP_CHECK(hipMemset(sentinel, 0, sizeof(int)));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipStreamWaitValue32(waiter_stream, flag, kBarrierToken, hipStreamWaitValueGte,
                                 0xFFFFFFFF));
  write_sentinel_kernel<<<1, 1, 0, waiter_stream>>>(sentinel, kSentinelValue);
  HIP_CHECK(hipGetLastError());

  // Give the waiter a chance to run if the gate is broken, then confirm it has not.
  //
  // The gate must be observed through the sentinel the kernel writes, not through
  // hipStreamQuery: a stream carrying a stream-memory-op keeps reporting hipErrorNotReady for a
  // while after its work has actually completed, so a query result here would pass whether or not
  // the wait held. `waiter_stream` is non-blocking, so this host copy on the null stream is not
  // ordered behind the gated kernel.
  std::this_thread::sleep_for(kGateSettleDelay);
  int gated_sentinel = -1;
  HIP_CHECK(hipMemcpy(&gated_sentinel, sentinel, sizeof(gated_sentinel), hipMemcpyDeviceToHost));

  // Release before asserting so a failed gate check cannot leave the stream waiting forever.
  HIP_CHECK(hipSetDevice(devices[0]));
  HIP_CHECK(hipStreamWriteValue32(writer_stream, flag, kBarrierToken, 0));
  HIP_CHECK(hipStreamSynchronize(writer_stream));

  const bool waiter_progressed = WaitForStreams({waiter_stream}, kStreamWaitTimeout);

  REQUIRE(gated_sentinel == 0);
  REQUIRE(waiter_progressed);

  int observed = 0;
  HIP_CHECK(hipMemcpy(&observed, sentinel, sizeof(observed), hipMemcpyDeviceToHost));
  REQUIRE(observed == kSentinelValue);

  HIP_CHECK(hipSetDevice(0));
}

/**
 * Test Description
 * ------------------------
 *  - Builds an all-device barrier out of `hipStreamBatchMemOp` over a single imported window, the
 *    shape a communication library uses for its copy-engine collectives: each device issues one
 *    batch writing its own slot, then a second batch waiting on every device's slot.
 *  - Holds rank 0 back and verifies every other rank is genuinely blocked: each rank's
 *    post-barrier kernel must not have written its sentinel yet.
 *  - Releases rank 0 and verifies every stream completes, every post-barrier kernel ran, and
 *    every slot holds the barrier token.
 *  - A stalled barrier is released from a rescue stream before asserting, so a failure reports as
 *    a failure rather than a hang.
 * Test source
 * ------------------------
 *  - unit/stream/hipStreamMemOpsPeerIpc.cc
 * Test requirements
 * ------------------------
 *  - Host specific (LINUX)
 *  - Virtual memory management supported
 *  - Stream wait-value support
 *  - At least two mutually peer-accessible devices
 *  - HIP_VERSION >= 6.4
 */
HIP_TEST_CASE(Unit_hipStreamMemOps_PeerImport_BatchMemOpBarrierAcrossDevices) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  std::vector<int> devices;
  for (int count = std::min(device_count, kMaxBarrierDevices); count >= 2; --count) {
    devices = GetMutuallyPeerAccessibleDevices(count);
    if (!devices.empty()) {
      break;
    }
  }
  if (devices.empty()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
  }

  const int rank_count = static_cast<int>(devices.size());
  for (int device : devices) {
    CheckVmmSupported(device);
  }
  for (int device : devices) {
    CheckStreamWaitValueSupported(device);
  }

  // Declared before the streams and the sentinels so that it is torn down after them.
  const ImportedWindowGuard window(devices[0], devices, rank_count * sizeof(uint32_t));
  uint32_t* slots = window.ptr<uint32_t>();

  HIP_CHECK(hipSetDevice(devices[0]));
  HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(slots), 0, rank_count));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<StreamGuard> stream_guards;
  std::vector<LinearAllocGuard<int>> sentinels;
  stream_guards.reserve(rank_count);
  sentinels.reserve(rank_count);
  for (int rank = 0; rank < rank_count; ++rank) {
    HIP_CHECK(hipSetDevice(devices[rank]));
    stream_guards.emplace_back(Streams::withFlags, hipStreamNonBlocking);
    sentinels.emplace_back(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK(hipMemset(sentinels[rank].ptr(), 0, sizeof(int)));
    HIP_CHECK(hipDeviceSynchronize());
  }

  // Raw handles, for the barrier issue path and for WaitForStreams.
  std::vector<hipStream_t> streams;
  streams.reserve(rank_count);
  for (const StreamGuard& guard : stream_guards) {
    streams.push_back(guard.stream());
  }

  // Each rank publishes its own slot, then waits on every rank's slot -- two batches per rank,
  // matching the order a communication library issues them in -- and finally writes a sentinel
  // that can only be set once the barrier has released.
  const auto issueRank = [&](int rank) {
    HIP_CHECK(hipSetDevice(devices[rank]));

    hipStreamBatchMemOpParams write_op{};
    write_op.operation = hipStreamMemOpWriteValue32;
    write_op.writeValue.address = reinterpret_cast<hipDeviceptr_t>(slots + rank);
    write_op.writeValue.value = kBarrierToken;
    write_op.writeValue.flags = 0;
    write_op.writeValue.alias = 0;
    HIP_CHECK(hipStreamBatchMemOp(streams[rank], 1, &write_op, 0));

    std::vector<hipStreamBatchMemOpParams> wait_ops(rank_count);
    for (int peer = 0; peer < rank_count; ++peer) {
      wait_ops[peer].operation = hipStreamMemOpWaitValue32;
      wait_ops[peer].waitValue.address = reinterpret_cast<hipDeviceptr_t>(slots + peer);
      wait_ops[peer].waitValue.value = kBarrierToken;
      wait_ops[peer].waitValue.flags = hipStreamWaitValueGte;
    }
    HIP_CHECK(hipStreamBatchMemOp(streams[rank], rank_count, wait_ops.data(), 0));

    write_sentinel_kernel<<<1, 1, 0, streams[rank]>>>(sentinels[rank].ptr(), kSentinelValue);
    HIP_CHECK(hipGetLastError());
  };

  // Hold rank 0 back: every other rank must block, because each waits on rank 0's slot too.
  for (int rank = 1; rank < rank_count; ++rank) {
    issueRank(rank);
  }

  std::this_thread::sleep_for(kGateSettleDelay);
  std::vector<int> gated_sentinels(rank_count, -1);
  for (int rank = 1; rank < rank_count; ++rank) {
    HIP_CHECK(hipSetDevice(devices[rank]));
    HIP_CHECK(hipMemcpy(&gated_sentinels[rank], sentinels[rank].ptr(), sizeof(int),
                        hipMemcpyDeviceToHost));
  }

  issueRank(0);

  const bool barrier_completed = WaitForStreams(streams, kStreamWaitTimeout);

  if (!barrier_completed) {
    // Release the stalled barrier from an independent stream so teardown cannot hang.
    HIP_CHECK(hipSetDevice(devices[0]));
    const StreamGuard rescue(Streams::withFlags, hipStreamNonBlocking);
    HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(slots), kBarrierToken, rank_count,
                                rescue.stream()));
    HIP_CHECK(hipStreamSynchronize(rescue.stream()));
    (void)WaitForStreams(streams, kStreamWaitTimeout);
  }

  std::vector<int> released_sentinels(rank_count, -1);
  for (int rank = 0; rank < rank_count; ++rank) {
    HIP_CHECK(hipSetDevice(devices[rank]));
    HIP_CHECK(hipStreamSynchronize(streams[rank]));
    HIP_CHECK(hipMemcpy(&released_sentinels[rank], sentinels[rank].ptr(), sizeof(int),
                        hipMemcpyDeviceToHost));
  }

  // While rank 0 was held back, no other rank may have passed the barrier.
  for (int rank = 1; rank < rank_count; ++rank) {
    INFO("rank " << rank << " ran its post-barrier kernel before rank 0 joined");
    REQUIRE(gated_sentinels[rank] == 0);
  }
  REQUIRE(barrier_completed);
  REQUIRE(std::all_of(released_sentinels.begin(), released_sentinels.end(),
                      [](int value) { return value == kSentinelValue; }));

  HIP_CHECK(hipSetDevice(devices[0]));
  std::vector<uint32_t> observed(rank_count, 0);
  HIP_CHECK(hipMemcpy(observed.data(), slots, rank_count * sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  REQUIRE(std::all_of(observed.begin(), observed.end(),
                      [](uint32_t value) { return value == kBarrierToken; }));

  HIP_CHECK(hipSetDevice(0));
}

#endif  // __linux__

/**
 * End doxygen group StreamTest.
 * @}
 */
