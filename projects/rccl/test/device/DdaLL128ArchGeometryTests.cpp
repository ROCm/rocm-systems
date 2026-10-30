/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// DDA LL128 wire helpers at the per-arch geometry: 128B lines / wave32 on gfx1250, 64B lines / wave64 on gfx9.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "DeviceTestBase.hpp"
#include "algorithms/dda/device/CollCommon_ll128.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DdaKernelParams;
using dda::common::kDdaLL128WordsPerThread;

// Device-side geometry; these constants differ between the gfx1250 and gfx9 device passes.
constexpr int kL = dda::common::kDdaLL128ArchLineElems;
constexpr int kW = dda::common::kDdaLL128ArchWarp;
constexpr int kSliceWords = dda::common::kDdaLL128WireWordsPerSliceW<kW>;
constexpr int kSliceBytes = dda::common::kDdaLL128DataBytesPerSliceL<kL, kW>;

constexpr uint64_t kFlag = 0xF1A6F1A6F1A6F1A6ull;
constexpr uint64_t kCanaryWord = 0xC0FFEE00C0FFEE00ull;
constexpr int8_t kCanary = 0x5a;
// Canary words on each side of the wire slices.
constexpr size_t kGuardWords = 64;
constexpr auto kWatchdog = std::chrono::seconds(5);

// Host view of the geometry a GPU should run, from its wave size.
struct Geometry {
  int line;        // 8B words per line
  int warp;        // lanes per slice
  int sliceWords;  // wire words per slice
  int sliceBytes;  // payload bytes per slice
};

Geometry expectedGeometry(int warpSize) {
  const int line = warpSize == 32 ? 16 : 8;
  const int sliceWords = warpSize * kDdaLL128WordsPerThread;
  return {line, warpSize, sliceWords, (sliceWords - sliceWords / line) * 8};
}

__global__ void reportGeometry(int* out) {
  out[0] = kL;
  out[1] = kW;
  out[2] = kSliceWords;
  out[3] = kSliceBytes;
}

// One block (one warp) per slice. Stores the slice, then optionally polls, unpacks and clears it.
// kBatched uses the batched load and poll, which need cooperative loads (gfx1250).
template <bool kAligned, bool kBatched>
__global__ void sliceKernel(const int8_t* src, int8_t* dst, uint64_t* wire, int bytes, bool poll, bool clearEven,
                            DdaKernelParams params, int* aborted) {
  const int lane = threadIdx.x;
  const bool flagLane = dda::common::ddaLL128IsFlagLane<kL>(lane);
  const int dataByte = blockIdx.x * kSliceBytes;
  const int elt = min(bytes - dataByte, kSliceBytes);
  uint64_t* slot = wire + (size_t)blockIdx.x * kSliceWords + 2 * lane;
  uint64_t regs[kDdaLL128WordsPerThread] = {};
  if constexpr (kBatched) {
    dda::common::ddaLL128LoadRegsBatched<int8_t, kL, kAligned, kW>(regs, src + dataByte, elt, lane, flagLane);
  } else {
    dda::common::ddaLL128LoadRegs<int8_t, kL, kAligned, kW>(regs, src + dataByte, elt, lane, flagLane);
  }
  dda::common::ddaLL128StoreWire<kW>(slot, regs, kFlag, flagLane);
  if (!poll) return;
  uint64_t vr[kDdaLL128WordsPerThread];
  const bool landed = kBatched ? dda::common::ddaLL128PollWireBatched<kL, kW>(slot, vr, kFlag, lane, params)
                               : dda::common::ddaLL128PollWire<kL, kW>(slot, vr, kFlag, lane, params);
  if (!landed) {
    atomicAdd(aborted, 1);
    return;
  }
  dda::common::ddaLL128StoreRegs<int8_t, kL, kAligned, kW>(dst + dataByte, vr, elt, lane, flagLane);
  if (clearEven && blockIdx.x % 2 == 0) dda::common::ddaLL128ClearWire<kW>(slot);
}

class AbortFlag {
public:
  AbortFlag() {
    HIP_EXPECT(hipHostMalloc(reinterpret_cast<void**>(&host_), sizeof(uint32_t),
                             hipHostMallocMapped | hipHostMallocCoherent));
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

// Raises the abort flag after kWatchdog so a broken poll fails instead of hanging; true if it fired.
bool waitOrAbort(hipStream_t stream, AbortFlag& abort) {
  const auto start = std::chrono::steady_clock::now();
  bool fired = false;
  while (hipStreamQuery(stream) == hipErrorNotReady) {
    if (!fired && std::chrono::steady_clock::now() - start > kWatchdog) {
      abort.raise();
      fired = true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  HIP_EXPECT(hipStreamSynchronize(stream));
  return fired;
}

// Never zero, so an unwritten payload word stands out.
int8_t sourceByte(int i) {
  const int8_t b = static_cast<int8_t>((i * 131 + 7) ^ (i >> 8));
  return b == 0 ? 1 : b;
}

struct SliceRun {
  std::vector<int8_t> src;   // payload
  std::vector<int8_t> dst;   // dst allocation from its base
  std::vector<uint64_t> wire;  // guards, slices, guards
  int aborted = 0;
  bool timedOut = false;
};

} // namespace

class DdaLL128ArchGeometry : public DeviceTestBase {
protected:
  Geometry geo_{};

  void SetUp() override {
    DeviceTestBase::SetUp();
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
    const std::string arch = prop.gcnArchName;
    if (!isGfx1250() && arch.rfind("gfx9", 0) != 0) {
      GTEST_SKIP() << "DDA LL128 geometry is defined for gfx9 and gfx1250 only, not " << arch;
    }
    geo_ = expectedGeometry(prop.warpSize);
    ASSERT_EQ(geo_.warp, isGfx1250() ? 32 : 64) << arch << " reports warpSize " << prop.warpSize;
  }

  // Runs one warp per slice over `bytes` with src/dst at the given byte offsets.
  SliceRun run(int bytes, size_t srcOff, size_t dstOff, bool poll, bool clearEven, bool batched = false) {
    SliceRun r;
    const int slices = (bytes + geo_.sliceBytes - 1) / geo_.sliceBytes;
    const size_t dstBytes = dstOff + (size_t)slices * geo_.sliceBytes;
    const size_t wireWords = 2 * kGuardWords + (size_t)slices * geo_.sliceWords;
    r.src.resize(bytes);
    for (int i = 0; i < bytes; ++i) r.src[i] = sourceByte(i);
    std::vector<int8_t> hSrc(srcOff + bytes, kCanary);
    std::copy(r.src.begin(), r.src.end(), hSrc.begin() + srcOff);

    DeviceBuffer<int8_t> src(hSrc.size());
    DeviceBuffer<int8_t> dst(dstBytes);
    DeviceBuffer<uint64_t> wire(wireWords);
    DeviceBuffer<int> aborted(1);
    src.copyFrom(hSrc);
    dst.copyFrom(std::vector<int8_t>(dstBytes, kCanary));
    wire.copyFrom(std::vector<uint64_t>(wireWords, kCanaryWord));
    aborted.zero();
    AbortFlag abort;
    hipStream_t stream = nullptr;
    HIP_EXPECT(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // Same rule as the launchers: kAligned only when both user buffers are 16B aligned.
    const bool aligned = srcOff % 16 == 0 && dstOff % 16 == 0;
    auto* kernel = aligned ? (batched ? sliceKernel<true, true> : sliceKernel<true, false>)
                           : (batched ? sliceKernel<false, true> : sliceKernel<false, false>);
    hipLaunchKernelGGL(kernel, dim3(slices), dim3(geo_.warp), 0, stream, src.ptr + srcOff, dst.ptr + dstOff,
                       wire.ptr + kGuardWords, bytes, poll, clearEven, abort.params(), aborted.ptr);
    HIP_EXPECT(hipGetLastError());
    r.timedOut = waitOrAbort(stream, abort);
    HIP_EXPECT(hipStreamDestroy(stream));
    r.aborted = aborted.download();
    r.dst = dst.copyTo();
    r.wire = wire.copyTo();
    return r;
  }

  bool isFlagWord(size_t w) const {
    const int lane = (int)(w % (size_t)(2 * geo_.warp)) / 2;
    return w % 2 == 1 && lane % (geo_.line / 2) == geo_.line / 2 - 1;
  }

  void expectRoundTrip(int bytes, size_t srcOff, size_t dstOff, bool batched = false) {
    SCOPED_TRACE("bytes=" + std::to_string(bytes) + " srcOff=" + std::to_string(srcOff) +
                 " dstOff=" + std::to_string(dstOff) + " batched=" + std::to_string(batched));
    const SliceRun r = run(bytes, srcOff, dstOff, /*poll=*/true, /*clearEven=*/false, batched);
    ASSERT_FALSE(r.timedOut) << "poll did not see the flags it was sent";
    EXPECT_EQ(r.aborted, 0);
    int firstBad = -1;
    for (int i = 0; i < bytes && firstBad < 0; ++i) {
      if (r.dst[dstOff + i] != r.src[i]) firstBad = i;
    }
    EXPECT_EQ(firstBad, -1) << "unpacked data differs from the source first at byte " << firstBad;
    size_t touched = 0;
    for (size_t i = 0; i < dstOff; ++i) touched += r.dst[i] != kCanary;
    for (size_t i = dstOff + bytes; i < r.dst.size(); ++i) touched += r.dst[i] != kCanary;
    EXPECT_EQ(touched, 0u) << "unpack wrote outside the " << bytes << " bytes it was given";
  }
};

TEST_F(DdaLL128ArchGeometry, DeviceGeometryMatchesArch) {
  DeviceBuffer<int> out(4);
  hipLaunchKernelGGL(reportGeometry, dim3(1), dim3(1), 0, nullptr, out.ptr);
  syncAndCheck();
  const std::vector<int> g = out.copyTo();
  EXPECT_EQ(g[0], geo_.line);
  EXPECT_EQ(g[1], geo_.warp);
  EXPECT_EQ(g[2], geo_.sliceWords);
  EXPECT_EQ(g[3], geo_.sliceBytes);
  EXPECT_EQ(geo_.sliceBytes, geo_.warp == 32 ? 1920 : 3584);
}

TEST_F(DdaLL128ArchGeometry, FlagWordsAndPayloadBytes) {
  constexpr int kSlices = 3;
  const SliceRun r = run(kSlices * geo_.sliceBytes, 0, 0, /*poll=*/false, /*clearEven=*/false);
  for (int s = 0; s < kSlices; ++s) {
    SCOPED_TRACE("slice " + std::to_string(s));
    std::vector<int8_t> payload;
    int flags = 0, misplacedFlags = 0, missingFlags = 0;
    for (int w = 0; w < geo_.sliceWords; ++w) {
      const uint64_t v = r.wire[kGuardWords + (size_t)s * geo_.sliceWords + w];
      if (isFlagWord(w)) {
        flags++;
        missingFlags += v != kFlag;
      } else {
        misplacedFlags += v == kFlag;
        const int8_t* b = reinterpret_cast<const int8_t*>(&v);
        payload.insert(payload.end(), b, b + 8);
      }
    }
    EXPECT_EQ(flags, geo_.sliceWords / geo_.line) << "one flag word per line";
    EXPECT_EQ(missingFlags, 0) << "a flag-lane word does not hold the flag";
    EXPECT_EQ(misplacedFlags, 0) << "the flag landed in a payload word";
    ASSERT_EQ((int)payload.size(), geo_.sliceBytes);
    // The slice's payload words hold exactly its source bytes, in the packed order.
    std::vector<int8_t> want(r.src.begin() + (size_t)s * geo_.sliceBytes,
                             r.src.begin() + (size_t)(s + 1) * geo_.sliceBytes);
    std::sort(payload.begin(), payload.end());
    std::sort(want.begin(), want.end());
    EXPECT_EQ(payload, want) << "a slice does not carry exactly " << geo_.sliceBytes << " payload bytes";
  }
  size_t guards = 0;
  for (size_t i = 0; i < kGuardWords; ++i) {
    guards += r.wire[i] != kCanaryWord;
    guards += r.wire[r.wire.size() - 1 - i] != kCanaryWord;
  }
  EXPECT_EQ(guards, 0u) << "StoreWire wrote outside its slices";
}

TEST_F(DdaLL128ArchGeometry, RoundTripAligned) {
  const int s = geo_.sliceBytes;
  for (int bytes : {16, 112, s - 16, s, s + 16, 3 * s + 480}) expectRoundTrip(bytes, 0, 0);
}

TEST_F(DdaLL128ArchGeometry, RoundTripMisaligned) {
  const int s = geo_.sliceBytes;
  for (auto [srcOff, dstOff] : {std::pair<size_t, size_t>{1, 1}, {4, 4}, {1, 4}, {0, 4}, {4, 0}}) {
    for (int bytes : {16, s - 16, s + 16, 3 * s + 480}) expectRoundTrip(bytes, srcOff, dstOff);
  }
}

// The batched load and poll (Broadcast LL128) need cooperative loads, so gfx1250 only.
TEST_F(DdaLL128ArchGeometry, RoundTripBatched) {
  if (geo_.warp != 32) GTEST_SKIP() << "batched LL128 load/poll are gfx1250 only";
  const int s = geo_.sliceBytes;
  for (auto [srcOff, dstOff] : {std::pair<size_t, size_t>{0, 0}, {1, 1}, {4, 4}, {1, 4}, {0, 4}, {4, 0}}) {
    for (int bytes : {16, 112, s - 16, s, s + 16, 3 * s + 480}) expectRoundTrip(bytes, srcOff, dstOff, true);
  }
}

TEST_F(DdaLL128ArchGeometry, ClearWireZeroesWhatPollRead) {
  constexpr int kSlices = 4;
  const SliceRun r = run(kSlices * geo_.sliceBytes - 16, 0, 0, /*poll=*/true, /*clearEven=*/true);
  ASSERT_FALSE(r.timedOut);
  EXPECT_EQ(r.aborted, 0);
  for (int s = 0; s < kSlices; ++s) {
    int nonZero = 0, flagsLeft = 0;
    for (int w = 0; w < geo_.sliceWords; ++w) {
      const uint64_t v = r.wire[kGuardWords + (size_t)s * geo_.sliceWords + w];
      nonZero += v != 0;
      flagsLeft += isFlagWord(w) && v == kFlag;
    }
    if (s % 2 == 0) {
      EXPECT_EQ(nonZero, 0) << "cleared slice " << s << " still has " << nonZero << " non-zero words";
    } else {
      EXPECT_EQ(flagsLeft, geo_.sliceWords / geo_.line) << "uncleared slice " << s << " lost flags";
    }
  }
  size_t guards = 0;
  for (size_t i = 0; i < kGuardWords; ++i) {
    guards += r.wire[i] != kCanaryWord;
    guards += r.wire[r.wire.size() - 1 - i] != kCanaryWord;
  }
  EXPECT_EQ(guards, 0u) << "ClearWire wrote outside the slice it polled";
}

} // namespace RcclUnitTesting
