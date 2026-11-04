/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Tests for ddaLL128StoreWire / ddaLL128PollWire, the wire used by every DDA LL128 receive, and for
// ddaLL128RsWaitBatch, the batched poll of the LL128 reduce-scatter.
// On gfx1250 the poll keeps the payload from the load that saw the flags, so these check that
// a poll never returns payload older than its flags.
//
//   CooperativeLoadOnGfx1250 - gfx1250 builds the cooperative load
//   EveryWireLineCarriesItsFlag - word 15 of every 128B wire line is the flag
//   RoundTrip* - pack, store, poll and unpack return the source and write nothing past it
//   WriterOn*NeverYieldsOldPayload - a writer on another CU or a peer GPU fills the slice while
//                the receiver polls; no returned word may be older than its flags
//   AbortEndsPoll - a raised abort flag ends the poll
//   WaitBatch* - the batched poll returns every valid entry's payload under the current flag, waits for a
//                late entry, ignores an invalid entry that never gets the flag, and ends on abort
//
// The host raises the abort flag after kWatchdog, so a broken poll fails instead of hanging.

#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "../common/DeviceBufferHelpers.hpp"
#include "../common/ResourceGuards.hpp"
#include "DeviceTestBase.hpp"
#include "algorithms/dda/device/CollCommon_ll128.h"
#include "algorithms/dda/reduce_scatter/reduce_scatter_dda_fabric_ll128.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DdaKernelParams;
using dda::common::ddaLL128IsFlagLane;
using dda::common::ddaLL128Slices;
using dda::common::kDdaLL128DataBytesPerSlice;
using dda::common::kDdaLL128FlagElem;
using dda::common::kDdaLL128LineElems;
using dda::common::kDdaLL128Pairs;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireWordsPerSlice;
using dda::common::kDdaLL128WordsPerThread;

constexpr int kRounds = 20000;
// A passing launch takes milliseconds; a broken one still ends inside the CI timeout.
constexpr auto kWatchdog = std::chrono::seconds(5);
constexpr int8_t kCanary = 0x5a;

// Payload word u of `lane` in round `round`; differs every round and is never 0.
__host__ __device__ inline uint64_t payloadWord(uint32_t round, int lane, int u) {
  uint64_t x = (uint64_t)(round + 1) * 0x9E3779B97F4A7C15ull ^
               ((uint64_t)lane << 8 | (uint64_t)u) * 0xBF58476D1CE4E5B9ull;
  return x ^ (x >> 31);
}

// The wire starts zeroed, so round 0's flag must not be 0.
__host__ __device__ inline uint64_t roundFlag(uint32_t round) { return (uint64_t)round + 1; }

// This lane's first 16B chunk of the slice, as the production callers address it.
__device__ inline uint64_t* laneSlot(uint64_t* wire, int lane) { return wire + 2 * lane; }

struct LiveResult {
  unsigned long long mismatches;   // returned words that differ from the round their flags announced
  unsigned long long roundsRead;   // rounds the receiver finished
  unsigned long long mixedSlices;  // rounds whose poll started with some lines new and some old
};

__global__ void cooperativeLoadEnabled(int* out) { *out = RCCL_HAVE_COOPERATIVE_ATOMIC_BUILTINS; }

// One 32-thread block per slice: pack, store, poll back, unpack.
__global__ void roundTripSlices(const int8_t* src, int8_t* dst, uint64_t* wire, int bytes, uint64_t flag,
                                DdaKernelParams params, int* aborted) {
  const int lane = threadIdx.x;
  const bool flagLane = ddaLL128IsFlagLane(lane);
  const int dataByte = blockIdx.x * kDdaLL128DataBytesPerSlice;
  const int elt = min(bytes - dataByte, kDdaLL128DataBytesPerSlice);
  uint64_t* slot = laneSlot(wire + (size_t)blockIdx.x * kDdaLL128WireWordsPerSlice, lane);
  uint64_t regs[kDdaLL128WordsPerThread];
  dda::common::ddaLL128LoadRegs<int8_t>(regs, src + dataByte, elt, lane, flagLane);
  dda::common::ddaLL128StoreWire(slot, regs, flag, flagLane);
  uint64_t vr[kDdaLL128WordsPerThread];
  if (!dda::common::ddaLL128PollWire(slot, vr, flag, lane, params)) {
    atomicAdd(aborted, 1);
    return;
  }
  dda::common::ddaLL128StoreRegs<int8_t>(dst + dataByte, vr, elt, lane, flagLane);
}

__device__ bool waitAtLeast(const uint32_t* word, uint32_t want, DdaKernelParams params) {
  int spins = 0;
  while (__scoped_atomic_load_n(const_cast<uint32_t*>(word), __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM) < want) {
    if (params.abortSpinTick(spins)) return false;
  }
  return true;
}

// Writer: after the receiver acks round r-1, store round r like ddaLL128StoreWire, four lines at a
// time with pauses. The group order rotates so every group is sometimes written last.
__device__ void writeRounds(uint64_t* slot, const uint32_t* ack, int lane, int rounds, DdaKernelParams params) {
  const bool flagLane = ddaLL128IsFlagLane(lane);
  for (uint32_t r = 0; r < (uint32_t)rounds; ++r) {
    if (!waitAtLeast(ack, r, params)) return;
    const int first = r % kDdaLL128Pairs;
    const bool reverse = ((r / kDdaLL128Pairs) & 1) != 0;
    const int pause = r % 3;
    for (int k = 0; k < kDdaLL128Pairs; ++k) {
      const int g = reverse ? (first - k + kDdaLL128Pairs) % kDdaLL128Pairs : (first + k) % kDdaLL128Pairs;
      const int u = 2 * g;
      dda::common::ddaLL128StoreAtomic16B(slot + u * kDdaLL128Warp, payloadWord(r, lane, u),
                                          flagLane ? roundFlag(r) : payloadWord(r, lane, u + 1));
      for (int p = 0; p < pause; ++p) __builtin_amdgcn_s_sleep(8);
    }
  }
}

// Waits until the writer has started round r. Returns 1 if the slice was half written, 0 if
// complete, -1 on abort.
__device__ int waitForFirstLine(const uint64_t* slot, uint32_t r, int lane, DdaKernelParams params) {
  int spins = 0;
  while (true) {
    bool anyNew = false, allNew = true;
    for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
      uint64_t lo, hi;
      dda::common::ddaLL128LoadAtomic16B(slot + u * kDdaLL128Warp, lo, hi);
      if (ddaLL128IsFlagLane(lane)) {
        anyNew |= hi == roundFlag(r);
        allNew &= hi == roundFlag(r);
      }
    }
    if (__any(anyNew)) return __all(allNew) ? 0 : 1;
    if (params.abortSpinTick(spins)) return -1;
  }
}

// Receiver: poll round r, check every word, then ack.
__device__ void readRounds(const uint64_t* slot, uint32_t* ack, int lane, int rounds, DdaKernelParams params,
                           LiveResult* out) {
  const bool flagLane = ddaLL128IsFlagLane(lane);
  unsigned long long mismatches = 0, mixed = 0;
  uint32_t r = 0;
  for (; r < (uint32_t)rounds; ++r) {
    const int halfWritten = waitForFirstLine(slot, r, lane, params);
    if (halfWritten < 0) break;
    mixed += halfWritten;
    uint64_t vr[kDdaLL128WordsPerThread];
    if (!dda::common::ddaLL128PollWire(slot, vr, roundFlag(r), lane, params)) break;
    for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
      mismatches += vr[u] != payloadWord(r, lane, u);
      mismatches += vr[u + 1] != (flagLane ? roundFlag(r) : payloadWord(r, lane, u + 1));
    }
    __syncthreads();  // every lane has its words before the writer may overwrite them
    if (lane == 0) __scoped_atomic_store_n(ack, r + 1, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  }
  if (mismatches != 0) atomicAdd(&out->mismatches, mismatches);
  if (lane == 0) {
    out->roundsRead = r;
    out->mixedSlices = mixed;
  }
}

// Block 0 receives, block 1 writes.
__global__ void liveSameGpu(uint64_t* wire, uint32_t* ack, int rounds, DdaKernelParams params, LiveResult* out) {
  const int lane = threadIdx.x;
  if (blockIdx.x == 0) {
    readRounds(laneSlot(wire, lane), ack, lane, rounds, params, out);
  } else {
    writeRounds(laneSlot(wire, lane), ack, lane, rounds, params);
  }
}

__global__ void liveReader(uint64_t* wire, uint32_t* ack, int rounds, DdaKernelParams params, LiveResult* out) {
  const int lane = threadIdx.x;
  readRounds(laneSlot(wire, lane), ack, lane, rounds, params, out);
}

__global__ void liveWriter(uint64_t* wire, const uint32_t* ack, int rounds, DdaKernelParams params) {
  const int lane = threadIdx.x;
  writeRounds(laneSlot(wire, lane), ack, lane, rounds, params);
}

__global__ void pollUnwrittenSlice(uint64_t* wire, DdaKernelParams params, int* returned) {
  const int lane = threadIdx.x;
  uint64_t vr[kDdaLL128WordsPerThread];
  const bool ok = dda::common::ddaLL128PollWire(laneSlot(wire, lane), vr, roundFlag(0), lane, params);
  if (lane == 0) *returned = ok ? 1 : 0;
}

struct BatchResult {
  int returned;                   // 1 if the wait reported every valid flag current, 0 on abort
  unsigned long long mismatches;  // valid entries' words that differ from the payload stored under the flag
};

// The current flag of the batch tests, and the one a stale entry carries.
constexpr uint64_t kBatchFlag = 0x2b0000002bull;
constexpr uint64_t kStaleFlag = kBatchFlag - 1;
// A stale entry k carries tag k + kStaleTag, so a wait that returns it early shows up as mismatches.
constexpr uint32_t kStaleTag = 100;

// Slice k of a batch laid out back to back.
__device__ inline uint64_t* batchSlice(uint64_t* wire, int k) {
  return wire + (size_t)k * kDdaLL128WireWordsPerSlice;
}

// Stores this lane's words of a slice tagged `tag` under `flag`, as ddaLL128StoreWire lays them out.
__device__ void storeTagged(uint64_t* slice, int lane, uint32_t tag, uint64_t flag) {
  const bool flagLane = ddaLL128IsFlagLane(lane);
  for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
    dda::common::ddaLL128StoreAtomic16B(laneSlot(slice, lane) + u * kDdaLL128Warp, payloadWord(tag, lane, u),
                                        flagLane ? flag : payloadWord(tag, lane, u + 1));
  }
}

// Entries in currentMask get tag k under kBatchFlag, entries in staleMask tag k + kStaleTag under kStaleFlag.
template <int kBatch>
__global__ void fillBatch(uint64_t* wire, int currentMask, int staleMask) {
  const int lane = threadIdx.x;
  for (int k = 0; k < kBatch; ++k) {
    if ((currentMask >> k) & 1) {
      storeTagged(batchSlice(wire, k), lane, k, kBatchFlag);
    } else if ((staleMask >> k) & 1) {
      storeTagged(batchSlice(wire, k), lane, k + kStaleTag, kStaleFlag);
    }
  }
}

// Block 0 waits on the batch, reading every entry as the runtime-rank kernel does, and checks each valid entry holds
// tag k under kBatchFlag. Block 1, if launched, stores entry `late` once block 0 has started waiting.
template <int kBatch>
__global__ void waitBatchLive(uint64_t* wire, int validMask, int late, uint32_t* started, DdaKernelParams params,
                              BatchResult* out) {
  const int lane = threadIdx.x;
  if (blockIdx.x == 1) {
    if (waitAtLeast(started, 1, params)) {
      for (int p = 0; p < 64; ++p) {
        __builtin_amdgcn_s_sleep(127);
      }
      storeTagged(batchSlice(wire, late), lane, late, kBatchFlag);
    }
    return;
  }
  const bool flagLane = ddaLL128IsFlagLane(lane);
  const uint64_t* lines[kBatch];
  bool valid[kBatch];
  bool read[kBatch];
  for (int k = 0; k < kBatch; ++k) {
    lines[k] = batchSlice(wire, k);
    valid[k] = ((validMask >> k) & 1) != 0;
    read[k] = true;
  }
  if (lane == 0) {
    __scoped_atomic_store_n(started, 1u, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
  }
  uint64_t vr[kBatch][kDdaLL128WordsPerThread];
  const bool ok = dda::common::ddaLL128RsWaitBatch(lines, valid, read, 2 * lane, kBatchFlag, flagLane, params, vr);
  unsigned long long mismatches = 0;
  for (int k = 0; k < kBatch; ++k) {
    if (!ok || !valid[k]) {
      continue;
    }
    for (int u = 0; u < kDdaLL128WordsPerThread; u += 2) {
      mismatches += vr[k][u] != payloadWord(k, lane, u);
      mismatches += vr[k][u + 1] != (flagLane ? kBatchFlag : payloadWord(k, lane, u + 1));
    }
  }
  if (mismatches != 0) {
    atomicAdd(&out->mismatches, mismatches);
  }
  if (lane == 0) {
    out->returned = ok ? 1 : 0;
  }
}

// Zeroed scratch from the allocator the DDA fabric scratch uses; peer GPUs can access it.
RCCLTestGuards::HipMemBufferAutoGuard allocScratch(size_t bytes) {
  void* p = nullptr;
  EXPECT_EQ(RCCLTestHelpers::allocateDeviceBuffer(&p, bytes), ncclSuccess);
  if (p != nullptr) {
    HIP_EXPECT(hipMemset(p, 0, bytes));
    HIP_EXPECT(hipDeviceSynchronize());
  }
  return RCCLTestGuards::makeHipMemBufferAutoGuard(p);
}

RCCLTestGuards::HipStreamAutoGuard makeStream() {
  hipStream_t stream = nullptr;
  HIP_EXPECT(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  return RCCLTestGuards::makeStreamAutoGuard(stream);
}

// Host-mapped abort flag, standing in for comm->abortFlagDev.
class AbortFlag {
public:
  AbortFlag() {
    HIP_EXPECT(hipHostMalloc(reinterpret_cast<void**>(&host_), sizeof(uint32_t),
                             hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable));
    if (host_ != nullptr) *host_ = 0;
  }
  ~AbortFlag() {
    if (host_ != nullptr) HIP_EXPECT(hipHostFree(host_));
  }
  AbortFlag(const AbortFlag&) = delete;
  AbortFlag& operator=(const AbortFlag&) = delete;

  DdaKernelParams params() const {
    void* dev = nullptr;
    HIP_EXPECT(hipHostGetDevicePointer(&dev, host_, 0));
    return DdaKernelParams{static_cast<const uint32_t*>(dev)};
  }
  void raise() {
    if (host_ != nullptr) __atomic_store_n(host_, 1u, __ATOMIC_SEQ_CST);
  }

private:
  uint32_t* host_ = nullptr;
};

using DeviceStreams = std::vector<std::pair<int, hipStream_t>>;

// Waits for the streams and raises the abort flag after kWatchdog; returns true if it had to.
// Never gives up on a running kernel: freeing its memory or killing it can wedge the GPU.
bool waitOrAbort(const DeviceStreams& streams, AbortFlag& abort) {
  const auto start = std::chrono::steady_clock::now();
  bool fired = false, reported = false;
  for (const auto& [dev, stream] : streams) {
    HIP_EXPECT(hipSetDevice(dev));
    while (hipStreamQuery(stream) == hipErrorNotReady) {
      const auto waited = std::chrono::steady_clock::now() - start;
      if (!fired && waited > kWatchdog) {
        abort.raise();
        fired = true;
      }
      if (!reported && waited > 2 * kWatchdog) {
        ADD_FAILURE() << "kernel still running " << (2 * kWatchdog).count() << " s in, with the abort flag raised";
        reported = true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    HIP_EXPECT(hipStreamSynchronize(stream));
  }
  return fired;
}

// Declare after the buffers: on any exit it stops the kernels before the buffers are freed.
auto drainOnExit(AbortFlag& abort, DeviceStreams streams) {
  return RCCLTestGuards::makeScopeGuard([&abort, streams = std::move(streams)]() {
    abort.raise();
    waitOrAbort(streams, abort);
  });
}

bool havePeerAccess(int a, int b) {
  int ab = 0, ba = 0;
  return hipDeviceCanAccessPeer(&ab, a, b) == hipSuccess && ab != 0 &&
         hipDeviceCanAccessPeer(&ba, b, a) == hipSuccess && ba != 0;
}

void expectLiveResult(const LiveResult& r, bool timedOut) {
  EXPECT_FALSE(timedOut) << "receiver stuck after " << r.roundsRead << " rounds";
  EXPECT_EQ(r.roundsRead, (unsigned long long)kRounds);
  EXPECT_EQ(r.mismatches, 0ull) << "the poll returned words older than the flags they came with";
  EXPECT_GT(r.mixedSlices, 0ull) << "no poll started on a half-written slice, so the test proved nothing";
}

} // namespace

class DdaLL128PollWire : public DeviceTestBase {
protected:
  void SetUp() override {
    DeviceTestBase::SetUp();
    // DDA LL128 runs only on gfx1250 (wave32).
    if (!isGfx1250()) {
      GTEST_SKIP() << "DDA LL128 runs on gfx1250 only";
    }
  }

  // Sends `bytes` through the wire and back and returns the wire. A canary after dst's last byte
  // catches over-writes.
  std::vector<uint64_t> RoundTrip(int bytes, uint64_t flag) {
    const int slices = (int)ddaLL128Slices((size_t)bytes);
    const size_t dstBytes = (size_t)slices * kDdaLL128DataBytesPerSlice;
    std::vector<int8_t> hSrc(bytes);
    for (int i = 0; i < bytes; ++i) hSrc[i] = static_cast<int8_t>((i * 131 + 7) ^ (i >> 8));
    DeviceBuffer<int8_t> src(bytes);
    DeviceBuffer<int8_t> dst(dstBytes);
    DeviceBuffer<int> aborted(1);
    src.copyFrom(hSrc);
    dst.copyFrom(std::vector<int8_t>(dstBytes, kCanary));
    aborted.zero();
    const size_t wireWords = (size_t)slices * kDdaLL128WireWordsPerSlice;
    auto wire = allocScratch(wireWords * sizeof(uint64_t));
    AbortFlag abort;
    auto stream = makeStream();

    hipLaunchKernelGGL(roundTripSlices, dim3(slices), dim3(kDdaLL128Warp), 0, stream.get(), src.ptr, dst.ptr,
                       static_cast<uint64_t*>(wire.get()), bytes, flag, abort.params(), aborted.ptr);
    HIP_EXPECT(hipGetLastError());
    EXPECT_FALSE(waitOrAbort({{0, stream.get()}}, abort)) << "round trip of " << bytes << " bytes stuck";

    EXPECT_EQ(aborted.download(), 0);
    const std::vector<int8_t> hDst = dst.copyTo();
    int firstBad = -1;
    for (int i = 0; i < bytes && firstBad < 0; ++i) {
      if (hDst[i] != hSrc[i]) firstBad = i;
    }
    EXPECT_EQ(firstBad, -1) << "unpacked slice differs from the source first at byte " << firstBad << " of " << bytes;
    size_t overwritten = 0;
    for (size_t i = bytes; i < dstBytes; ++i) overwritten += hDst[i] != kCanary;
    EXPECT_EQ(overwritten, 0u) << "unpack wrote past the " << bytes << " bytes it was given";

    auto [err, hWire] = RCCLTestHelpers::downloadBuffer<uint64_t>(wire.get(), wireWords);
    HIP_EXPECT(err);
    return hWire;
  }

  // Fills a batch per currentMask and staleMask (others stay zeroed, never flagged), then waits on it with validMask.
  // With late >= 0 a second block stores that entry under the current flag once the wait has begun.
  template <int kBatch>
  BatchResult WaitBatch(int currentMask, int staleMask, int validMask, int late, bool raiseAbort, bool& timedOut) {
    auto wire = allocScratch((size_t)kBatch * kDdaLL128WireWordsPerSlice * sizeof(uint64_t));
    auto started = allocScratch(sizeof(uint32_t));
    DeviceBuffer<BatchResult> result(1);
    result.upload(BatchResult{-1, 0});
    AbortFlag abort;
    if (raiseAbort) {
      abort.raise();
    }
    auto stream = makeStream();
    auto drain = drainOnExit(abort, {{0, stream.get()}});

    uint64_t* w = static_cast<uint64_t*>(wire.get());
    hipLaunchKernelGGL((fillBatch<kBatch>), dim3(1), dim3(kDdaLL128Warp), 0, stream.get(), w, currentMask, staleMask);
    HIP_EXPECT(hipGetLastError());
    hipLaunchKernelGGL((waitBatchLive<kBatch>), dim3(late >= 0 ? 2 : 1), dim3(kDdaLL128Warp), 0, stream.get(), w,
                       validMask, late, static_cast<uint32_t*>(started.get()), abort.params(), result.ptr);
    HIP_EXPECT(hipGetLastError());
    timedOut = waitOrAbort({{0, stream.get()}}, abort);
    return result.download();
  }

  template <int kBatch>
  void ExpectWaitBatchCases() {
    constexpr int kAll = (1 << kBatch) - 1;
    constexpr int kLast = kBatch - 1;
    bool timedOut = false;
    {
      SCOPED_TRACE("all flags current");
      const BatchResult r = WaitBatch<kBatch>(kAll, 0, kAll, -1, false, timedOut);
      EXPECT_FALSE(timedOut);
      EXPECT_EQ(r.returned, 1);
      EXPECT_EQ(r.mismatches, 0ull);
    }
    {
      SCOPED_TRACE("entry 1 lands late");
      const BatchResult r = WaitBatch<kBatch>(kAll & ~2, 2, kAll, 1, false, timedOut);
      EXPECT_FALSE(timedOut);
      EXPECT_EQ(r.returned, 1);
      EXPECT_EQ(r.mismatches, 0ull) << "the wait returned before the late entry's flag was current";
    }
    {
      SCOPED_TRACE("invalid last entry never flagged");
      const BatchResult r = WaitBatch<kBatch>(kAll & ~(1 << kLast), 0, kAll & ~(1 << kLast), -1, false, timedOut);
      EXPECT_FALSE(timedOut) << "an invalid entry blocked the wait";
      EXPECT_EQ(r.returned, 1);
      EXPECT_EQ(r.mismatches, 0ull);
    }
    {
      SCOPED_TRACE("abort while entry 0 is stale");
      const BatchResult r = WaitBatch<kBatch>(kAll & ~1, 1, kAll, -1, true, timedOut);
      EXPECT_FALSE(timedOut);
      EXPECT_EQ(r.returned, 0) << "an aborted wait must report it did not see the flags";
    }
  }
};

TEST_F(DdaLL128PollWire, CooperativeLoadOnGfx1250) {
  DeviceBuffer<int> out(1);
  out.upload(-1);
  hipLaunchKernelGGL(cooperativeLoadEnabled, dim3(1), dim3(1), 0, nullptr, out.ptr);
  syncAndCheck();
  EXPECT_EQ(out.download(), 1) << "gfx1250 should build the cooperative 8x16B wire load";
}

TEST_F(DdaLL128PollWire, EveryWireLineCarriesItsFlag) {
  constexpr uint64_t kFlag = 0x5a5a00000000002bull;
  const std::vector<uint64_t> wire = RoundTrip(3 * kDdaLL128DataBytesPerSlice, kFlag);
  int badLines = 0;
  for (size_t line = 0; line < wire.size() / kDdaLL128LineElems; ++line) {
    badLines += wire[line * kDdaLL128LineElems + kDdaLL128FlagElem] != kFlag;
  }
  EXPECT_EQ(badLines, 0) << "a line whose flag is not in its own word 15 cannot be read with one load";
}

TEST_F(DdaLL128PollWire, RoundTripFullSlices) {
  RoundTrip(kDdaLL128DataBytesPerSlice, 3);
  RoundTrip(8 * kDdaLL128DataBytesPerSlice, 4);
}

TEST_F(DdaLL128PollWire, RoundTripPartialSlices) {
  for (int bytes : {16, 112, 128, 1904, kDdaLL128DataBytesPerSlice + 16, 5 * kDdaLL128DataBytesPerSlice + 480}) {
    SCOPED_TRACE(bytes);
    RoundTrip(bytes, 5);
  }
}

TEST_F(DdaLL128PollWire, WriterOnAnotherCuNeverYieldsOldPayload) {
  auto wire = allocScratch(kDdaLL128WireWordsPerSlice * sizeof(uint64_t));
  auto ack = allocScratch(sizeof(uint32_t));
  DeviceBuffer<LiveResult> result(1);
  result.upload(LiveResult{0, 0, 0});
  AbortFlag abort;
  auto stream = makeStream();
  auto drain = drainOnExit(abort, {{0, stream.get()}});

  hipLaunchKernelGGL(liveSameGpu, dim3(2), dim3(kDdaLL128Warp), 0, stream.get(), static_cast<uint64_t*>(wire.get()),
                     static_cast<uint32_t*>(ack.get()), kRounds, abort.params(), result.ptr);
  HIP_CHECK(hipGetLastError());
  const bool timedOut = waitOrAbort({{0, stream.get()}}, abort);
  expectLiveResult(result.download(), timedOut);
}

// The writer stores into the receiver GPU's memory, as DDA pushes do. Not process-isolated: the
// runner kills on timeout, which can wedge the GPU; the abort flag bounds this test instead.
TEST_F(DdaLL128PollWire, WriterOnPeerGpuNeverYieldsOldPayload) {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n < 2 || !isGfx1250(1) || !havePeerAccess(0, 1)) {
    GTEST_SKIP() << "needs two gfx1250 GPUs with peer access";
  }
  constexpr int kReader = 0, kWriter = 1;
  for (auto [self, peer] : {std::pair<int, int>{kReader, kWriter}, std::pair<int, int>{kWriter, kReader}}) {
    HIP_CHECK(hipSetDevice(self));
    const hipError_t e = hipDeviceEnablePeerAccess(peer, 0);
    ASSERT_TRUE(e == hipSuccess || e == hipErrorPeerAccessAlreadyEnabled) << hipGetErrorString(e);
  }
  (void)hipGetLastError();  // clear a sticky hipErrorPeerAccessAlreadyEnabled

  HIP_CHECK(hipSetDevice(kReader));
  auto wire = allocScratch(kDdaLL128WireWordsPerSlice * sizeof(uint64_t));
  DeviceBuffer<LiveResult> result(1);
  result.upload(LiveResult{0, 0, 0});
  auto readerStream = makeStream();
  HIP_CHECK(hipSetDevice(kWriter));
  auto ack = allocScratch(sizeof(uint32_t));
  auto writerStream = makeStream();
  AbortFlag abort;
  const DeviceStreams streams = {{kReader, readerStream.get()}, {kWriter, writerStream.get()}};
  auto drain = drainOnExit(abort, streams);

  HIP_CHECK(hipSetDevice(kReader));
  hipLaunchKernelGGL(liveReader, dim3(1), dim3(kDdaLL128Warp), 0, readerStream.get(),
                     static_cast<uint64_t*>(wire.get()), static_cast<uint32_t*>(ack.get()), kRounds, abort.params(),
                     result.ptr);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipSetDevice(kWriter));
  hipLaunchKernelGGL(liveWriter, dim3(1), dim3(kDdaLL128Warp), 0, writerStream.get(),
                     static_cast<uint64_t*>(wire.get()), static_cast<uint32_t*>(ack.get()), kRounds, abort.params());
  HIP_CHECK(hipGetLastError());
  const bool timedOut = waitOrAbort(streams, abort);

  HIP_CHECK(hipSetDevice(kReader));
  expectLiveResult(result.download(), timedOut);
}

TEST_F(DdaLL128PollWire, AbortEndsPoll) {
  auto wire = allocScratch(kDdaLL128WireWordsPerSlice * sizeof(uint64_t));
  DeviceBuffer<int> returned(1);
  returned.upload(-1);
  AbortFlag abort;
  abort.raise();
  auto stream = makeStream();
  auto drain = drainOnExit(abort, {{0, stream.get()}});

  hipLaunchKernelGGL(pollUnwrittenSlice, dim3(1), dim3(kDdaLL128Warp), 0, stream.get(),
                     static_cast<uint64_t*>(wire.get()), abort.params(), returned.ptr);
  HIP_CHECK(hipGetLastError());
  EXPECT_FALSE(waitOrAbort({{0, stream.get()}}, abort));
  EXPECT_EQ(returned.download(), 0) << "an aborted poll must report it did not see the flags";
}

TEST_F(DdaLL128PollWire, WaitBatchOfTwo) {
  ExpectWaitBatchCases<2>();
}

TEST_F(DdaLL128PollWire, WaitBatchOfThree) {
  ExpectWaitBatchCases<3>();
}

} // namespace RcclUnitTesting
