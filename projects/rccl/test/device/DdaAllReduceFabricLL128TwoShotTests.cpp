/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// DDA LL128 AllReduce two-shots (cooperative, LDS-staged writer, peer-split) on one GPU; helper kernels play the peers.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
#include <vector>

#include "DeviceTestBase.hpp"
#include "StreamWaitHelpers.hpp"
#include "algorithms/dda/all_reduce/all_reduce_dda_ll128_twoshot.h"
#include "algorithms/dda/fabric/fabric_gpu_barrier.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

using dda::common::DdaArSplitB;
using dda::common::DdaKernelParams;
using dda::common::kDdaArSplitBDma;
using dda::common::kDdaArSplitBPairwise;
using dda::common::kDdaArSplitBWide;
using dda::common::kDdaArSplitBWideChunked;
using dda::common::kDdaLL128DataBytesPerSlice;
using dda::common::kDdaLL128FlagBytesNarrow;
using dda::common::kDdaLL128FlagBytesWide;
using dda::common::kDdaLL128NarrowDataBytesPerSlice;
using dda::common::kDdaLL128Pairs;
using dda::common::kDdaLL128Warp;
using dda::common::kDdaLL128WireAsync;
using dda::common::kDdaLL128WireCoop;
using dda::common::kDdaLL128WireWordsPerSlice;
using dda::common::kDdaLL128WordsPerThread;

// Two full slices and a partial third.
constexpr size_t kSmallShardBytes = 2 * kDdaLL128DataBytesPerSlice + 112;
// 40 slices and a partial one: on 1 x 512 most warps fill phase 2 twice, so a 16-rank fill waits on 15 pushes.
constexpr size_t kMediumShardBytes = 40 * kDdaLL128DataBytesPerSlice + 112;
// The launcher's staged cutoff plus a partial last slice: every warp of an 8 x 512 grid refills its windows ~9 times.
constexpr size_t kLargeShardBytes = 2 * 1024 * 1024 + 112;
constexpr uint32_t kFlag = 2;  // epoch cells start at kFlag - 1, so the kernels use flag 2 and bank 0
constexpr uint8_t kUntouched = 0xA5;
// A launch still running past this hangs: the test raises the abort and fails.
constexpr std::chrono::seconds kDrainTimeout{60};

// kSplitNarrow: the split with the narrow 4 B flag.
enum class Kind { kCoop, kStaged, kSplit, kSplitNarrow };

int flagBytes(Kind kind) {
  return kind == Kind::kSplitNarrow ? kDdaLL128FlagBytesNarrow : kDdaLL128FlagBytesWide;
}

struct Case {
  int nRanks;
  int self;
  int ct;  // NRANKS_CT of the cooperative and staged instances: 4, 8 or 0
  Kind kind;
  unsigned grid;
  unsigned block;
  size_t shardBytes;
  uint32_t flag = kFlag;  // the epoch the launch takes: kFlag + 1 uses bank 1
  int epochLen = 0;       // epoch cells the launch owns; 0 is one per block
  bool inPlace = false;   // sendbuff == recvbuff
  bool handoff = false;   // the split's B -> C hand-off through LDS
  DdaArSplitB b = kDdaArSplitBPairwise;  // B's form: wide forms with the hand-off; DMA B with the 8 B flag
};

int epochLen(const Case& c) {
  return c.epochLen != 0 ? c.epochLen : (int)c.grid;
}

size_t slices(size_t shardBytes) {
  return dda::common::ddaLL128Slices(shardBytes);
}

// Shard slices of a launch of `kind`: the narrow flag's are 1984 B.
size_t slicesOf(Kind kind, size_t shardBytes) {
  return kind == Kind::kSplitNarrow ? dda::common::ddaLL128SlicesOf<kDdaLL128FlagBytesNarrow>(shardBytes) :
                                     slices(shardBytes);
}

// A bank of exactly two staging areas of nRanks slots, each slot one shard's slices.
size_t bankBytes(int nRanks, size_t shardBytes) {
  return 2 * (size_t)nRanks * slices(shardBytes) * kDdaLL128WireWordsPerSlice * sizeof(uint64_t);
}

size_t slotWords(int nRanks, size_t shardBytes) {
  return dda::common::ddaLL128ArTwoShotSlotWords(bankBytes(nRanks, shardBytes), nRanks);
}

uint64_t wireFlag(uint32_t flag) {
  return ((uint64_t)flag << 32) | flag;
}

// Writes slice s of rows[peer] into area + peer * slotWords as the LL128 line that peer sends; one warp.
template <int kFlagBytes = kDdaLL128FlagBytesWide>
__device__ void PublishLine(uint64_t* area, const int8_t* rows, int peer, size_t s, size_t slotWordsArg,
                            size_t shardBytes, uint64_t flag) {
  const int lane = threadIdx.x;
  const bool flagLane = dda::common::ddaLL128IsFlagLane(lane);
  const size_t sliceByte = s * (size_t)dda::common::ddaLL128DataBytesPerSliceOf<kFlagBytes>();
  uint64_t regs[kDdaLL128WordsPerThread] = {};
  dda::common::ddaLL128LoadRegs<int8_t, kFlagBytes>(regs, rows + (size_t)peer * shardBytes + sliceByte,
                                                    dda::common::ddaLL128SliceBytesOf<kFlagBytes>(shardBytes, s), lane,
                                                    flagLane);
  dda::common::ddaLL128StoreWire<kFlagBytes>(
    area + (size_t)peer * slotWordsArg + s * kDdaLL128WireWordsPerSlice + 2 * lane, regs, flag, flagLane);
}

// One warp per (peers[blockIdx.x], slice blockIdx.y), once `ticks` of wall clock passed (0: at once).
template <int kFlagBytes = kDdaLL128FlagBytesWide>
__global__ void PublishLines(uint64_t* area, const int8_t* rows, const int* peers, size_t slotWordsArg,
                             size_t shardBytes, uint64_t flag, long long ticks) {
  const long long start = wall_clock64();
  while (wall_clock64() - start < ticks) {
  }
  PublishLine<kFlagBytes>(area, rows, peers[blockIdx.x], blockIdx.y, slotWordsArg, shardBytes, flag);
}

// In place, peer peers[blockIdx.x] publishes its reduced slices in order, each once my copy of it reached the peer.
template <int kFlagBytes = kDdaLL128FlagBytesWide>
__global__ void PublishLinesAfterMyCopy(uint64_t* area, const int8_t* rows, const int* peers, size_t slotWordsArg,
                                        size_t shardBytes, uint64_t flag, uint64_t* const* myCopyAt,
                                        DdaKernelParams params) {
  for (size_t s = 0; s < dda::common::ddaLL128SlicesOf<kFlagBytes>(shardBytes); ++s) {
    const uint64_t* at[1] = {myCopyAt[blockIdx.x] + s * kDdaLL128WireWordsPerSlice + 2 * threadIdx.x};
    uint64_t seen[1][kDdaLL128WordsPerThread];
    if (!dda::common::ddaLL128PollWireN<1, kFlagBytes>(at, 1u, seen, flag, threadIdx.x, params)) {
      return;
    }
    PublishLine<kFlagBytes>(area, rows, peers[blockIdx.x], s, slotWordsArg, shardBytes, flag);
  }
}

// Garbles (data and flag) each line of wire slice blockIdx.x at `slot` whose bit in `keep` is clear: it reads stale.
__global__ void TearLines(uint32_t* slot, uint32_t keep) {
  constexpr int kWords = (int)(dda::common::kDdaLL128WireBytesPerSlice / sizeof(uint32_t));
  for (int w = threadIdx.x; w < kWords; w += blockDim.x) {
    if (((keep >> (w * sizeof(uint32_t) / dda::common::kDdaLL128LineBytes)) & 1u) == 0) {
      slot[(size_t)blockIdx.x * kWords + w] = 0x5A5A5A5Au;
    }
  }
}

__global__ void ReportStaged(int* out) {
  *out = dda::common::ddaLL128Staged<kDdaLL128WireAsync>() ? 1 : 0;
}

// Most blocks of one launch resident at once: each block holds its slot for `ticks` of wall clock.
__global__ void CountResident(int* live, int* most, long long ticks) {
  if (threadIdx.x == 0) {
    atomicMax(most, atomicAdd(live, 1) + 1);
    const long long start = wall_clock64();
    while (wall_clock64() - start < ticks) {
    }
    atomicSub(live, 1);
  }
  __syncthreads();
}

// The split kernel of T with kFlagBytes for c: my own slot, or the hand-off with c.b's B; DMA B either way. Uneven =
// size_t picks the uneven-shard instance.
template <typename T, int kFlagBytes, typename... Uneven>
auto splitKernel(const Case& c) {
  namespace dc = dda::common;
  if constexpr (kFlagBytes == kDdaLL128FlagBytesWide) {
    if (c.b == kDdaArSplitBDma) {
      return c.handoff ? &dc::ddaAllReduceTwoShotLL128Split<T, kFlagBytes, true, kDdaArSplitBDma, Uneven...>
                       : &dc::ddaAllReduceTwoShotLL128Split<T, kFlagBytes, false, kDdaArSplitBDma, Uneven...>;
    }
  }
  if (!c.handoff) {
    return &dc::ddaAllReduceTwoShotLL128Split<T, kFlagBytes, false, kDdaArSplitBPairwise, Uneven...>;
  }
  switch (c.b) {
    case kDdaArSplitBWide:
      return &dc::ddaAllReduceTwoShotLL128Split<T, kFlagBytes, true, kDdaArSplitBWide, Uneven...>;
    case kDdaArSplitBWideChunked:
      return &dc::ddaAllReduceTwoShotLL128Split<T, kFlagBytes, true, kDdaArSplitBWideChunked, Uneven...>;
    default:
      return &dc::ddaAllReduceTwoShotLL128Split<T, kFlagBytes, true, kDdaArSplitBPairwise, Uneven...>;
  }
}

// Dynamic LDS of c's split launch: DMA B's group is the one the kernel derives from a block's most slices, as the
// launcher sizes it (0 where none fits the budget).
size_t splitLds(const Case& c) {
  if (c.b != kDdaArSplitBDma) {
    return dda::common::ddaLL128ArSplitSharedBytes(c.block, c.nRanks, c.handoff, c.b);
  }
  const size_t group = dda::common::ddaArSplitDmaGroupSlices(c.block, c.nRanks, c.handoff,
                                                             (slicesOf(c.kind, c.shardBytes) + c.grid - 1) / c.grid);
  return group == 0 ? 0 : dda::common::ddaLL128ArSplitDmaSharedBytes(c.block, c.nRanks, c.handoff, group);
}

// The split with kFlagBytes, handing B -> C off through LDS (its LDS limit raised first) where c.handoff. On uneven
// shards c.shardBytes is the stride and totalBytes the message's bytes.
template <typename T, int kFlagBytes, typename... Uneven>
void launchSplit(const Case& c, T** scratch, T* recv, const T* send, uint32_t* epoch, const DdaKernelParams& params,
                 hipStream_t stream, Uneven... totalBytes) {
  const size_t lds = splitLds(c);
  ASSERT_NE(lds, 0u) << "no DMA B group fits the LDS budget (the only form with no LDS size)";
  auto* kernel = splitKernel<T, kFlagBytes, Uneven...>(c);
  if (c.handoff || c.b == kDdaArSplitBDma) {
    HIP_CHECK(
      hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, (int)lds));
  }
  hipLaunchKernelGGL(kernel, dim3(c.grid), dim3(c.block), lds, stream, scratch, recv, send, c.shardBytes, c.self,
                     c.nRanks, epoch, epochLen(c), slicesOf(c.kind, c.shardBytes), bankBytes(c.nRanks, c.shardBytes),
                     params, totalBytes...);
}

// Whether c's blocks own at most the hand-off area's slices.
bool handoffFits(const Case& c) {
  return (slicesOf(c.kind, c.shardBytes) + c.grid - 1) / c.grid <= dda::common::kDdaArSplitHandoffSlices;
}

// A split case handing off in B form b: n ranks, rank self, blocks x threads, the shard, the epoch, in place, the
// epoch cells it owns (0: one a block).
Case wideCase(int n, int self, unsigned grid, unsigned block, size_t shard, DdaArSplitB b, Kind kind = Kind::kSplit,
              uint32_t flag = kFlag, bool inPlace = false, int epochCells = 0) {
  Case c{n, self, 0, kind, grid, block, shard, flag, epochCells};
  c.inPlace = inPlace;
  c.handoff = true;
  c.b = b;
  return c;
}

template <typename T, int N>
void launch(const Case& c, T** scratch, T* recv, const T* send, uint32_t* epoch, uint32_t* abortFlag,
            hipStream_t stream) {
  const DdaKernelParams params{abortFlag};
  const dim3 grid(c.grid);
  const dim3 block(c.block);
  const size_t bank = bankBytes(c.nRanks, c.shardBytes);
  const size_t n = slices(c.shardBytes);
  switch (c.kind) {
    case Kind::kCoop:
      hipLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128<T, N, kDdaLL128WireCoop>), grid, block, 0, stream,
                         scratch, recv, send, c.shardBytes, c.self, c.nRanks, epoch, epochLen(c), n, bank, params);
      break;
    case Kind::kStaged:
      hipLaunchKernelGGL((dda::common::ddaAllReduceTwoShotLL128<T, N, kDdaLL128WireAsync>), grid, block,
                         dda::common::ddaLL128StageBytes(c.block), stream, scratch, recv, send, c.shardBytes, c.self,
                         c.nRanks, epoch, epochLen(c), n, bank, params);
      break;
    case Kind::kSplit:
      launchSplit<T, kDdaLL128FlagBytesWide>(c, scratch, recv, send, epoch, params, stream);
      break;
    case Kind::kSplitNarrow:
      launchSplit<T, kDdaLL128FlagBytesNarrow>(c, scratch, recv, send, epoch, params, stream);
      break;
  }
}

// Inexact float sums pin the fold order: my shard first, then the peers from me + 1 on. `seed` varies the values.
std::vector<std::vector<uint8_t>> FloatMessages(int nRanks, size_t shardBytes, int seed = 0) {
  std::vector<std::vector<uint8_t>> send((size_t)nRanks, std::vector<uint8_t>((size_t)nRanks * shardBytes));
  for (int r = 0; r < nRanks; ++r) {
    float* f = reinterpret_cast<float*>(send[(size_t)r].data());
    for (size_t i = 0; i < (size_t)nRanks * shardBytes / sizeof(float); ++i) {
      f[i] = 1.0f / (float)(3 + r + seed) + (float)((r * 131 + (int)i * 17) % 1000) * 1.0e-3f;
    }
  }
  return send;
}

// Shard j folded by its owner: rank j's copy, then ranks j + 1, j + 2, ...
std::vector<std::vector<uint8_t>> FloatReduced(const std::vector<std::vector<uint8_t>>& send, int nRanks,
                                               size_t shardBytes) {
  std::vector<std::vector<uint8_t>> reduced((size_t)nRanks, std::vector<uint8_t>(shardBytes));
  for (int j = 0; j < nRanks; ++j) {
    float* acc = reinterpret_cast<float*>(reduced[(size_t)j].data());
    for (size_t i = 0; i < shardBytes / sizeof(float); ++i) {
      const size_t at = (size_t)j * shardBytes / sizeof(float) + i;
      acc[i] = reinterpret_cast<const float*>(send[(size_t)j].data())[at];
      for (int r = 1; r < nRanks; ++r) {
        acc[i] = acc[i] + reinterpret_cast<const float*>(send[(size_t)((j + r) % nRanks)].data())[at];
      }
    }
  }
  return reduced;
}

// bf16 a + b as the device folds it: the fp32 sum rounded to nearest even (DdaVecElementAddTests' Bf16AddRef).
uint16_t Bf16Add(uint16_t a, uint16_t b) {
  float x;
  float y;
  const uint32_t ua = (uint32_t)a << 16;
  const uint32_t ub = (uint32_t)b << 16;
  std::memcpy(&x, &ua, sizeof(x));
  std::memcpy(&y, &ub, sizeof(y));
  const float sum = x + y;
  uint32_t u;
  std::memcpy(&u, &sum, sizeof(u));
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// FloatMessages' values as bf16, truncated: every value finite, the 8-rank sums inexact.
std::vector<std::vector<uint8_t>> Bf16Messages(int nRanks, size_t shardBytes) {
  const std::vector<std::vector<uint8_t>> f = FloatMessages(nRanks, 2 * shardBytes);
  std::vector<std::vector<uint8_t>> send((size_t)nRanks, std::vector<uint8_t>((size_t)nRanks * shardBytes));
  for (int r = 0; r < nRanks; ++r) {
    for (size_t i = 0; i < send[(size_t)r].size() / 2; ++i) {
      std::memcpy(&send[(size_t)r][2 * i], &f[(size_t)r][4 * i + 2], 2);
    }
  }
  return send;
}

// FloatReduced in bf16.
std::vector<std::vector<uint8_t>> Bf16Reduced(const std::vector<std::vector<uint8_t>>& send, int nRanks,
                                              size_t shardBytes) {
  std::vector<std::vector<uint8_t>> reduced((size_t)nRanks, std::vector<uint8_t>(shardBytes));
  for (int j = 0; j < nRanks; ++j) {
    for (size_t i = 0; i < shardBytes / 2; ++i) {
      const size_t at = (size_t)j * shardBytes + 2 * i;
      uint16_t acc;
      std::memcpy(&acc, &send[(size_t)j][at], sizeof(acc));
      for (int r = 1; r < nRanks; ++r) {
        uint16_t v;
        std::memcpy(&v, &send[(size_t)((j + r) % nRanks)][at], sizeof(v));
        acc = Bf16Add(acc, v);
      }
      std::memcpy(&reduced[(size_t)j][2 * i], &acc, sizeof(acc));
    }
  }
  return reduced;
}

// fp8 e4m3 positive normals up to 1.0 (0x08..0x38): an 8-rank sum stays finite, so the bytes compare exactly.
std::vector<std::vector<uint8_t>> Fp8Rows(int rows, size_t bytes, int seed) {
  std::vector<std::vector<uint8_t>> out((size_t)rows, std::vector<uint8_t>(bytes));
  for (int r = 0; r < rows; ++r) {
    for (size_t i = 0; i < bytes; ++i) {
      out[(size_t)r][i] = (uint8_t)(0x08 + (r * 37 + (int)i * 11 + seed) % 0x31);
    }
  }
  return out;
}

// The reduced shards end to end: the whole reduced message.
std::vector<uint8_t> Concat(const std::vector<std::vector<uint8_t>>& shards) {
  std::vector<uint8_t> out;
  for (const std::vector<uint8_t>& shard : shards) {
    out.insert(out.end(), shard.begin(), shard.end());
  }
  return out;
}

// What one launch leaves behind: recvbuff and every peer's scratch (self's own scratch is the kernel's private state).
struct Snapshot {
  std::vector<uint8_t> recv;
  std::vector<std::vector<uint8_t>> peerScratch;
};

class DdaAllReduceFabricLL128TwoShot : public DeviceTestBase {
 protected:
  void SetUp() override {
    DeviceTestBase::SetUp();
    if (!isGfx1250()) {
      GTEST_SKIP() << "the LL128 AllReduce tiers run on gfx1250";
    }
    if (!stagedBuilt()) {
      GTEST_SKIP() << "this device pass has no async LDS stores";
    }
    // Host mapped, so a hung launch can be told to leave.
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&abortHost_), sizeof(uint32_t),
                            hipHostMallocMapped | hipHostMallocCoherent));
    *abortHost_ = 0;
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&abortDev_), abortHost_, 0));
    HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&side_, hipStreamNonBlocking));
  }

  void TearDown() override {
    for (hipStream_t st : {stream_, side_}) {
      if (st != nullptr) {
        HIP_EXPECT(hipStreamDestroy(st));
      }
    }
    if (abortHost_ != nullptr) {
      HIP_EXPECT(hipHostFree(abortHost_));
    }
    DeviceTestBase::TearDown();
  }

  // False on a device pass without async LDS stores, where the staged instance is the cooperative one.
  bool stagedBuilt() {
    DeviceBuffer<int> staged(1);
    hipLaunchKernelGGL(ReportStaged, dim3(1), dim3(1), 0, 0, staged.ptr);
    HIP_EXPECT(hipDeviceSynchronize());
    return staged.download() != 0;
  }

  // send_[r] is rank r's message (nRanks shards); reduced_[j] is the reduced shard j a peer publishes.
  void setUp(int nRanks, int self, size_t shardBytes, const std::vector<std::vector<uint8_t>>& send,
             const std::vector<std::vector<uint8_t>>& reduced) {
    nRanks_ = nRanks;
    self_ = self;
    shardBytes_ = shardBytes;
    scratch_.clear();
    std::vector<uint8_t*> ptrs;
    for (int r = 0; r < nRanks; ++r) {
      scratch_.emplace_back(2 * bankBytes(nRanks, shardBytes));
      ptrs.push_back(scratch_.back().ptr);
    }
    scratchPtrs_ = std::make_unique<DeviceBuffer<uint8_t*>>(ptrs.size());
    scratchPtrs_->copyFrom(ptrs);
    sendBuf_ = std::make_unique<DeviceBuffer<uint8_t>>((size_t)nRanks * shardBytes);
    recvBuf_ = std::make_unique<DeviceBuffer<uint8_t>>((size_t)nRanks * shardBytes);
    setMessages(send, reduced);
  }

  // New messages on the same scratch: send_, reduced_ and my sendbuff.
  void setMessages(const std::vector<std::vector<uint8_t>>& send, const std::vector<std::vector<uint8_t>>& reduced) {
    send_ = send;
    reduced_ = reduced;
    sendBuf_->copyFrom(send_[(size_t)self_]);
  }

  // setUp for c on FloatMessages and their host fold.
  void setUpFloat(const Case& c) {
    const auto send = FloatMessages(c.nRanks, c.shardBytes);
    setUp(c.nRanks, c.self, c.shardBytes, send, FloatReduced(send, c.nRanks, c.shardBytes));
  }

  // Zeroed scratch, untouched recvbuff (my message in place), fresh epoch cells, then the peers' lines in c's bank.
  // skipCopy sends no first-area copy and skipReduced no reduced shard; in place, run() publishes the reduced shards.
  void reset(const Case& c, uint32_t abortFlag, int skipCopy = -1, int skipReduced = -1) {
    for (DeviceBuffer<uint8_t>& b : scratch_) {
      b.zero();
    }
    if (c.inPlace) {
      recvBuf_->copyFrom(send_[(size_t)self_]);
    } else {
      HIP_CHECK(hipMemset(recvBuf_->ptr, kUntouched, recvBuf_->count));
    }
    epoch_.copyFrom(std::vector<uint32_t>(kEpochCells, c.flag - 1));
    *abortHost_ = abortFlag;
    kind_ = c.kind;
    bankOffset_ = (c.flag & 1u) * bankBytes(nRanks_, shardBytes_);

    const std::vector<int8_t> copies = copyRows();
    const std::vector<int8_t> reduced = reducedRows();
    publish(area(self_, 0), copies, skipCopy, c.flag);
    if (c.inPlace) {
      pendingReduced_ = std::make_unique<DeviceBuffer<int8_t>>(reduced.size());
      pendingReduced_->copyFrom(reduced);
    } else {
      publish(area(self_, nRanks_), reduced, skipReduced, c.flag);
    }
  }

  // Row p: peer p's copy of my shard, which it sends to my first area.
  std::vector<int8_t> copyRows() {
    std::vector<int8_t> copies((size_t)nRanks_ * shardBytes_);
    for (int p = 0; p < nRanks_; ++p) {
      std::memcpy(&copies[(size_t)p * shardBytes_], &send_[(size_t)p][(size_t)self_ * shardBytes_], shardBytes_);
    }
    return copies;
  }

  // Row p: peer p's reduced shard, which it sends to my second area.
  std::vector<int8_t> reducedRows() {
    std::vector<int8_t> reduced((size_t)nRanks_ * shardBytes_);
    for (int p = 0; p < nRanks_; ++p) {
      std::memcpy(&reduced[(size_t)p * shardBytes_], reduced_[(size_t)p].data(), shardBytes_);
    }
    return reduced;
  }

  // Rank r's staging area that starts `slot` slots into c's bank.
  uint64_t* area(int r, int slot) {
    return reinterpret_cast<uint64_t*>(scratch_[(size_t)r].ptr + bankOffset_) +
           (size_t)slot * slotWords(nRanks_, shardBytes_);
  }

  std::vector<int> peersBut(int skip) {
    std::vector<int> peers;
    for (int p = 0; p < nRanks_; ++p) {
      if (p != self_ && p != skip) {
        peers.push_back(p);
      }
    }
    return peers;
  }

  // The peers but skip (or only peer `only`) publish rows at `at` in kind_'s line format; on `stream` after `ticks` of
  // wall clock, else now.
  void publish(uint64_t* at, const std::vector<int8_t>& rows, int skip, uint32_t flag, hipStream_t stream = nullptr,
               long long ticks = 0, int only = -1) {
    const std::vector<int> peers = only >= 0 ? std::vector<int>{only} : peersBut(skip);
    auto dRows = std::make_shared<DeviceBuffer<int8_t>>(rows.size());
    dRows->copyFrom(rows);
    auto dPeers = std::make_shared<DeviceBuffer<int>>(peers.size());
    dPeers->copyFrom(peers);
    const dim3 grid((unsigned)peers.size(), (unsigned)slicesOf(kind_, shardBytes_));
    auto* kernel = flagBytes(kind_) == kDdaLL128FlagBytesNarrow ? &PublishLines<kDdaLL128FlagBytesNarrow> :
                                                                  &PublishLines<kDdaLL128FlagBytesWide>;
    hipLaunchKernelGGL(kernel, grid, dim3(kDdaLL128Warp), 0, stream, at, dRows->ptr, dPeers->ptr,
                       slotWords(nRanks_, shardBytes_), shardBytes_, wireFlag(flag), ticks);
    HIP_CHECK(hipGetLastError());
    if (stream == nullptr) {
      HIP_CHECK(hipDeviceSynchronize());
    } else {
      latePublishBuffers_.push_back(dRows);
      latePublishBuffers_.push_back(dPeers);
    }
  }

  bool drain(const std::vector<hipStream_t>& streams) {
    return drainOrAbort(streams, kDrainTimeout, abortHost_);
  }

  template <typename T>
  void run(const Case& c) {
    T** scratch = reinterpret_cast<T**>(scratchPtrs_->ptr);
    T* recv = reinterpret_cast<T*>(recvBuf_->ptr);
    const T* send = c.inPlace ? recv : reinterpret_cast<const T*>(sendBuf_->ptr);
    std::unique_ptr<DeviceBuffer<int>> dPeers;
    std::unique_ptr<DeviceBuffer<uint64_t*>> dCopyAt;
    std::vector<hipStream_t> streams = {stream_};
    if (c.inPlace) {
      // The peers, one block each on their own stream, publish each reduced slice once my copy of it arrived.
      const std::vector<int> peers = peersBut(-1);
      std::vector<uint64_t*> copyAt;
      for (int p : peers) {
        copyAt.push_back(area(p, self_));
      }
      dPeers = std::make_unique<DeviceBuffer<int>>(peers.size());
      dPeers->copyFrom(peers);
      dCopyAt = std::make_unique<DeviceBuffer<uint64_t*>>(copyAt.size());
      dCopyAt->copyFrom(copyAt);
      auto* publisher = &PublishLinesAfterMyCopy<kDdaLL128FlagBytesWide>;
      if (flagBytes(c.kind) == kDdaLL128FlagBytesNarrow) {
        publisher = &PublishLinesAfterMyCopy<kDdaLL128FlagBytesNarrow>;
      }
      hipLaunchKernelGGL(publisher, dim3((unsigned)peers.size()), dim3(kDdaLL128Warp), 0, side_, area(self_, nRanks_),
                         pendingReduced_->ptr, dPeers->ptr, slotWords(nRanks_, shardBytes_), shardBytes_,
                         wireFlag(c.flag), dCopyAt->ptr, DdaKernelParams{abortDev_});
      HIP_CHECK(hipGetLastError());
      streams.push_back(side_);
    }
    switch (c.ct) {
      case 4:
        launch<T, 4>(c, scratch, recv, send, epoch_.ptr, abortDev_, stream_);
        break;
      case 8:
        launch<T, 8>(c, scratch, recv, send, epoch_.ptr, abortDev_, stream_);
        break;
      default:
        launch<T, 0>(c, scratch, recv, send, epoch_.ptr, abortDev_, stream_);
        break;
    }
    HIP_CHECK(hipGetLastError());
    EXPECT_TRUE(drain(streams)) << "the launch did not finish within " << kDrainTimeout.count() << " s";
  }

  Snapshot snapshot() {
    Snapshot snap{recvBuf_->copyTo(), {}};
    for (int p = 0; p < nRanks_; ++p) {
      snap.peerScratch.push_back(p == self_ ? std::vector<uint8_t>() : scratch_[(size_t)p].copyTo());
    }
    return snap;
  }

  // The second area slot[self] of every rank in the last reset's bank, where only a fully reduced shard of mine lands.
  bool reducedPublished() {
    const size_t words = slotWords(nRanks_, shardBytes_);
    const size_t at = bankOffset_ + (size_t)(nRanks_ + self_) * words * sizeof(uint64_t);
    for (const DeviceBuffer<uint8_t>& b : scratch_) {
      const std::vector<uint8_t> bytes = b.copyTo();
      for (size_t i = at; i < at + words * sizeof(uint64_t); ++i) {
        if (bytes[i] != 0) {
          return true;
        }
      }
    }
    return false;
  }

  // Runs the cooperative two-shot out of place, then c (and the staged writer for a 512-thread split): same results.
  template <typename T>
  Snapshot expectMatchesCoop(const Case& c) {
    SCOPED_TRACE(testing::Message() << c.nRanks << "r kind " << (int)c.kind << " ct " << c.ct << " shard "
                                    << c.shardBytes << " " << c.grid << "x" << c.block << " flag " << c.flag
                                    << " epochLen " << epochLen(c) << " inPlace " << c.inPlace << " handoff "
                                    << c.handoff << " b " << c.b);
    Case ref = c;
    ref.kind = Kind::kCoop;
    ref.inPlace = false;
    reset(ref, 0);
    run<T>(ref);
    const Snapshot want = snapshot();
    std::vector<Case> runs = {c};
    if (c.kind == Kind::kSplit && c.block == 512) {
      ref.kind = Kind::kStaged;
      runs.push_back(ref);
    }
    for (const Case& r : runs) {
      reset(r, 0);
      run<T>(r);
      const Snapshot got = snapshot();
      EXPECT_TRUE(got.recv == want.recv)
          << "kind " << (int)r.kind << ": recvbuff differs from the cooperative writer's";
      for (int p = 0; p < c.nRanks; ++p) {
        EXPECT_TRUE(got.peerScratch[(size_t)p] == want.peerScratch[(size_t)p])
            << "kind " << (int)r.kind << ": peer " << p << " lines differ";
      }
      expectEpochAdvanced(r);
    }
    return want;
  }

  // Every cell the launch owns moved to its flag; the cells past them stayed.
  void expectEpochAdvanced(const Case& c) {
    const std::vector<uint32_t> epoch = epoch_.copyTo();
    for (int e = 0; e < (int)kEpochCells; ++e) {
      EXPECT_EQ(epoch[(size_t)e], e < epochLen(c) ? c.flag : c.flag - 1) << "cell " << e;
    }
  }

  // My shard of recvbuff: untouched, or the host fold.
  void expectMyShard(bool reduced) {
    const std::vector<uint8_t> recv = recvBuf_->copyTo();
    for (size_t i = 0; i < shardBytes_; ++i) {
      ASSERT_EQ(recv[(size_t)self_ * shardBytes_ + i], reduced ? reduced_[(size_t)self_][i] : kUntouched)
          << "byte " << i;
    }
  }

  // The shard in rank r's narrow slot `slot` of the bank; *flagged is false if a line lacks the flag.
  std::vector<uint8_t> decodeNarrow(int r, int slot, uint32_t flag, bool* flagged) {
    const std::vector<uint8_t> bytes = scratch_[(size_t)r].copyTo();
    const uint64_t* words =
      reinterpret_cast<const uint64_t*>(bytes.data() + bankOffset_) + (size_t)slot * slotWords(nRanks_, shardBytes_);
    std::vector<uint8_t> out(shardBytes_, kUntouched);
    *flagged = true;
    for (size_t s = 0; s < slicesOf(Kind::kSplitNarrow, shardBytes_); ++s) {
      const uint64_t* wire = words + s * kDdaLL128WireWordsPerSlice;
      for (int lane = 0; lane < kDdaLL128Warp; ++lane) {
        const bool flagLane = dda::common::ddaLL128IsFlagLane(lane);
        uint64_t regs[kDdaLL128WordsPerThread];
        for (int g = 0; g < kDdaLL128Pairs; ++g) {
          regs[2 * g] = wire[(size_t)g * 2 * kDdaLL128Warp + 2 * lane];
          regs[2 * g + 1] = wire[(size_t)g * 2 * kDdaLL128Warp + 2 * lane + 1];
          if (flagLane && !dda::common::ddaLL128NarrowFlagSeen(regs[2 * g + 1], wireFlag(flag))) {
            *flagged = false;
          }
        }
        dda::common::ddaLL128NarrowUnpack(regs, flagLane);
        for (int g = 0; g < kDdaLL128Pairs; ++g) {
          const size_t at =
            s * kDdaLL128NarrowDataBytesPerSlice + 16 * (size_t)dda::common::ddaLL128NarrowChunkIx(g, lane);
          if ((!flagLane || g < dda::common::kDdaLL128NarrowFullPairs) && at < shardBytes_) {
            std::memcpy(&out[at], &regs[2 * g], 16);
          }
        }
      }
    }
    return out;
  }

  // The narrow split c matches the cooperative writer; each peer holds my copy and reduced shard as narrow lines.
  template <typename T>
  void expectNarrowMatchesCoop(const Case& c, const char* type) {
    SCOPED_TRACE(testing::Message() << type << " " << c.nRanks << "r narrow shard " << c.shardBytes << " " << c.grid
                                    << "x" << c.block << " flag " << c.flag << " epochLen " << epochLen(c)
                                    << " inPlace " << c.inPlace << " handoff " << c.handoff << " b " << c.b);
    Case ref = c;
    ref.kind = Kind::kCoop;
    ref.inPlace = false;
    reset(ref, 0);
    run<T>(ref);
    const std::vector<uint8_t> want = recvBuf_->copyTo();
    reset(c, 0);
    run<T>(c);
    EXPECT_TRUE(recvBuf_->copyTo() == want) << "recvbuff differs from the cooperative writer's";
    const std::vector<uint8_t> mine(want.begin() + (ptrdiff_t)(self_ * shardBytes_),
                                    want.begin() + (ptrdiff_t)((self_ + 1) * shardBytes_));
    for (int p = 0; p < nRanks_; ++p) {
      if (p == self_) {
        continue;
      }
      bool flagged = false;
      const std::vector<uint8_t> copy(send_[(size_t)self_].begin() + (ptrdiff_t)(p * shardBytes_),
                                      send_[(size_t)self_].begin() + (ptrdiff_t)((p + 1) * shardBytes_));
      EXPECT_TRUE(decodeNarrow(p, self_, c.flag, &flagged) == copy) << "peer " << p << ": my copy of its shard";
      EXPECT_TRUE(flagged) << "peer " << p << ": a copy line lacks the flag";
      EXPECT_TRUE(decodeNarrow(p, nRanks_ + self_, c.flag, &flagged) == mine) << "peer " << p << ": my reduced shard";
      EXPECT_TRUE(flagged) << "peer " << p << ": a reduced line lacks the flag";
    }
    expectEpochAdvanced(c);
  }

  // The 8 B or the narrow split c against the cooperative writer.
  template <typename T>
  void expectSplitMatchesCoop(const Case& c, const char* type) {
    if (c.kind == Kind::kSplitNarrow) {
      expectNarrowMatchesCoop<T>(c, type);
    } else {
      expectMatchesCoop<T>(c);
    }
  }

  static constexpr size_t kEpochCells = 64;
  Kind kind_ = Kind::kCoop;
  // Rows and peer lists of publish() calls still in flight on a stream.
  std::vector<std::shared_ptr<void>> latePublishBuffers_;
  int nRanks_ = 0;
  int self_ = 0;
  size_t shardBytes_ = 0;
  size_t bankOffset_ = 0;
  std::vector<std::vector<uint8_t>> send_;
  std::vector<std::vector<uint8_t>> reduced_;
  std::vector<DeviceBuffer<uint8_t>> scratch_;
  std::unique_ptr<DeviceBuffer<uint8_t*>> scratchPtrs_;
  std::unique_ptr<DeviceBuffer<uint8_t>> sendBuf_;
  std::unique_ptr<DeviceBuffer<uint8_t>> recvBuf_;
  std::unique_ptr<DeviceBuffer<int8_t>> pendingReduced_;
  DeviceBuffer<uint32_t> epoch_{kEpochCells};
  uint32_t* abortHost_ = nullptr;
  uint32_t* abortDev_ = nullptr;
  hipStream_t stream_ = nullptr;
  hipStream_t side_ = nullptr;
};

// The staged writer on 4, 8 (unrolled), 6 and 16 (runtime) ranks, below and above the launcher's 2 MiB staged cutoff:
// every wait rung (0 and 1, 3, 7, 15 pushes), one or many window refills per warp, and a partial last slice.
const Case kStagedCases[] = {
  {4, 1, 4, Kind::kStaged, 2, 512, kSmallShardBytes},    {6, 5, 0, Kind::kStaged, 2, 512, kSmallShardBytes},
  {8, 5, 8, Kind::kStaged, 2, 512, kSmallShardBytes},    {16, 9, 0, Kind::kStaged, 1, 512, kMediumShardBytes},
  {4, 1, 4, Kind::kStaged, 8, 512, kLargeShardBytes},    {6, 5, 0, Kind::kStaged, 8, 512, kLargeShardBytes},
  {8, 5, 8, Kind::kStaged, 8, 512, kLargeShardBytes},
};

// The split on 3 and 5 ranks (pairs only) and 4 to 32 (an odd last peer), 1 x 64 (look-ahead past the end) to 16 x 512.
const Case kSplitCases[] = {
  {3, 1, 0, Kind::kSplit, 2, 512, kSmallShardBytes},
  {5, 2, 0, Kind::kSplit, 1, 64, kSmallShardBytes},
  {4, 1, 4, Kind::kSplit, 1, 64, kSmallShardBytes},
  {6, 5, 0, Kind::kSplit, 2, 512, kSmallShardBytes},
  {8, 5, 8, Kind::kSplit, 2, 512, kSmallShardBytes},
  {32, 17, 0, Kind::kSplit, 4, 512, kSmallShardBytes},  // 3 slices on 4 blocks: one block owns no slice
  {5, 3, 0, Kind::kSplit, 4, 512, kMediumShardBytes},
  {16, 9, 0, Kind::kSplit, 16, 512, kMediumShardBytes},
  {4, 1, 4, Kind::kSplit, 8, 512, kLargeShardBytes},
  {8, 5, 8, Kind::kSplit, 8, 512, kLargeShardBytes},
  {3, 2, 0, Kind::kSplit, 8, 512, kLargeShardBytes},
  {8, 5, 8, Kind::kSplit, 8, 512, kMediumShardBytes, kFlag + 1, 24},  // bank 1, more epoch cells than blocks
  {5, 2, 0, Kind::kSplit, 2, 512, kMediumShardBytes, kFlag + 1, 7},
  {8, 5, 8, Kind::kSplit, 2, 512, kMediumShardBytes, kFlag, 0, true},  // in place
  {5, 2, 0, Kind::kSplit, 4, 512, kLargeShardBytes, kFlag + 1, 9, true},
  {4, 1, 4, Kind::kSplit, 4, 480, kMediumShardBytes},  // 15 and 12 warps a block, as the 4-rank grids at 2 and 4 MiB
  {4, 2, 4, Kind::kSplit, 16, 384, kMediumShardBytes, kFlag + 1},
};

// The cooperative two-shot gives the host fold, and the staged writer and the split leave its recvbuff and peer lines.
TEST_F(DdaAllReduceFabricLL128TwoShot, MatchesTheCooperativeWriterAndTheHostFold) {
  auto check = [&](const Case& c) {
    setUpFloat(c);
    const Snapshot want = expectMatchesCoop<float>(c);
    ASSERT_TRUE(want.recv == Concat(reduced_)) << "the cooperative two-shot left the host fold";
  };
  for (const Case& c : kStagedCases) {
    check(c);
  }
  for (const Case& c : kSplitCases) {
    check(c);
  }
}

// fp8 at 8 ranks has the most register pressure; a 512-thread split row also runs the staged writer as its reference.
TEST_F(DdaAllReduceFabricLL128TwoShot, MatchesTheCooperativeWriterOnFp8) {
  const Case cases[] = {{5, 2, 0, Kind::kSplit, 1, 64, kSmallShardBytes},
                        {8, 5, 8, Kind::kSplit, 2, 512, kSmallShardBytes},
                        {8, 5, 8, Kind::kSplit, 8, 512, kLargeShardBytes}};
  for (const Case& c : cases) {
    setUp(c.nRanks, c.self, c.shardBytes, Fp8Rows(c.nRanks, (size_t)c.nRanks * c.shardBytes, 0),
          Fp8Rows(c.nRanks, c.shardBytes, 5));
    expectMatchesCoop<dda::common::fp8e4m3>(c);
  }
}

// A peer that never sends its copy aborts the reduce: my shard is never stored or published, and the epoch moves on.
TEST_F(DdaAllReduceFabricLL128TwoShot, AbortInTheReduceLeavesNoPartialSum) {
  const Case eight{8, 5, 8, Kind::kSplit, 2, 512, kSmallShardBytes};
  const struct {
    Case c;
    // r of the peer self + r that never sends: first or second of a pair, a later pair, the odd last; wide and chunked:
    // the last copy, a middle one, the short third chunk and the first copy of the second chunk (a partial sum held).
    int skip;
  } cells[] = {{{8, 5, 8, Kind::kStaged, 2, 512, kSmallShardBytes}, 2},
               {{8, 5, 8, Kind::kStaged, 8, 512, kLargeShardBytes}, 2},
               {eight, 1},
               {eight, 2},
               {eight, 6},
               {eight, 7},
               {{5, 2, 0, Kind::kSplit, 1, 64, kSmallShardBytes}, 3},
               {{8, 5, 8, Kind::kSplit, 8, 512, kLargeShardBytes, kFlag + 1, 12}, 7},
               {{8, 5, 8, Kind::kSplitNarrow, 2, 512, kSmallShardBytes}, 1},
               {{8, 5, 8, Kind::kSplitNarrow, 2, 512, kSmallShardBytes}, 6},
               {{5, 2, 0, Kind::kSplitNarrow, 1, 64, kSmallShardBytes}, 3},
               {{8, 5, 8, Kind::kSplitNarrow, 8, 512, kLargeShardBytes, kFlag + 1, 12}, 7},
               {{8, 5, 8, Kind::kSplit, 2, 512, kSmallShardBytes, kFlag, 0, false, /*handoff=*/true}, 2},
               {{8, 5, 8, Kind::kSplitNarrow, 2, 512, kSmallShardBytes, kFlag, 0, false, /*handoff=*/true}, 7},
               {{5, 2, 0, Kind::kSplitNarrow, 2, 64, kMediumShardBytes, kFlag + 1, 0, false, /*handoff=*/true}, 3},
               {wideCase(8, 5, 2, 512, kSmallShardBytes, kDdaArSplitBWide, Kind::kSplitNarrow), 7},
               {wideCase(9, 4, 2, 512, kMediumShardBytes, kDdaArSplitBWide, Kind::kSplit, kFlag + 1), 3},
               {wideCase(18, 3, 2, 512, kMediumShardBytes, kDdaArSplitBWideChunked, Kind::kSplitNarrow), 17},
               {wideCase(20, 7, 1, 64, kSmallShardBytes, kDdaArSplitBWideChunked), 9},
               {wideCase(8, 5, 2, 512, kSmallShardBytes, kDdaArSplitBDma), 1},
               {wideCase(16, 9, 2, 512, kMediumShardBytes, kDdaArSplitBDma), 15},
               {{16, 9, 0, Kind::kSplit, 1, 512, kMediumShardBytes, kFlag + 1, 0, false, /*handoff=*/false,
                 kDdaArSplitBDma},
                7},
               {{5, 2, 0, Kind::kSplit, 1, 64, kSmallShardBytes, kFlag, 0, false, /*handoff=*/false, kDdaArSplitBDma},
                3}};
  for (const auto& cell : cells) {
    const Case& c = cell.c;
    SCOPED_TRACE(testing::Message() << c.nRanks << "r kind " << (int)c.kind << " shard " << c.shardBytes
                                    << " missing self + " << cell.skip);
    setUpFloat(c);
    reset(c, 1, (c.self + cell.skip) % c.nRanks);
    run<float>(c);
    expectMyShard(/*reduced=*/false);
    EXPECT_FALSE(reducedPublished());
    expectEpochAdvanced(c);
  }
}

// A peer whose copies stop after the first round aborts wide and chunked B in the second (one block, 21 slices,
// rounds of 6): the first round's slices are stored, the later ones untouched, and nothing is pushed.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitWideBAbortInALaterRoundKeepsTheFirstRound) {
  constexpr size_t kShard = 20 * kDdaLL128DataBytesPerSlice + 112;
  for (const Case& c : {wideCase(9, 4, 1, 512, kShard, kDdaArSplitBWide),
                        wideCase(18, 3, 1, 512, kShard, kDdaArSplitBWideChunked)}) {
    SCOPED_TRACE(testing::Message() << c.nRanks << "r b " << c.b);
    const size_t round = c.b == kDdaArSplitBWide ? dda::common::ddaArSplitWideRoundSlices(c.nRanks)
                                                 : dda::common::ddaArSplitWideChunkRoundSlices(c.block / kDdaLL128Warp);
    ASSERT_EQ(round, 6u);
    setUpFloat(c);
    const int late = (c.self + 2) % c.nRanks;
    reset(c, 1, late);
    // The late peer's copies of the first round's slices only.
    DeviceBuffer<int8_t> rows(copyRows().size());
    rows.copyFrom(copyRows());
    DeviceBuffer<int> peer(1);
    peer.copyFrom(std::vector<int>{late});
    hipLaunchKernelGGL(PublishLines<kDdaLL128FlagBytesWide>, dim3(1, (unsigned)round), dim3(kDdaLL128Warp), 0, nullptr,
                       area(self_, 0), rows.ptr, peer.ptr, slotWords(nRanks_, shardBytes_), shardBytes_,
                       wireFlag(c.flag), 0ll);
    HIP_CHECK(hipDeviceSynchronize());
    run<float>(c);
    const std::vector<uint8_t> recv = recvBuf_->copyTo();
    for (size_t i = 0; i < kShard; ++i) {
      const bool first = i < round * kDdaLL128DataBytesPerSlice;
      ASSERT_EQ(recv[(size_t)self_ * kShard + i], first ? reduced_[(size_t)self_][i] : kUntouched) << "byte " << i;
    }
    EXPECT_FALSE(reducedPublished());
    expectEpochAdvanced(c);
  }
}

// A peer that never sends its reduced shard aborts the split's gather: my shard is still stored, the peer's untouched.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitAbortInTheGatherKeepsMyReducedShard) {
  for (const Case& c : {Case{8, 5, 8, Kind::kSplit, 2, 512, kSmallShardBytes},
                        Case{5, 2, 0, Kind::kSplit, 4, 512, kMediumShardBytes, kFlag + 1},
                        Case{8, 5, 8, Kind::kSplitNarrow, 2, 512, kSmallShardBytes},
                        Case{5, 2, 0, Kind::kSplitNarrow, 4, 512, kMediumShardBytes, kFlag + 1},
                        Case{8, 5, 8, Kind::kSplitNarrow, 2, 512, kSmallShardBytes, kFlag, 0, false, /*handoff=*/true},
                        Case{5, 2, 0, Kind::kSplit, 4, 512, kMediumShardBytes, kFlag + 1, 0, false,
                             /*handoff=*/true},
                        wideCase(8, 5, 2, 512, kSmallShardBytes, kDdaArSplitBWide, Kind::kSplitNarrow),
                        wideCase(18, 3, 2, 512, kMediumShardBytes, kDdaArSplitBWideChunked, Kind::kSplit,
                                 kFlag + 1),
                        wideCase(8, 5, 2, 512, kSmallShardBytes, kDdaArSplitBDma),
                        Case{5, 2, 0, Kind::kSplit, 4, 512, kMediumShardBytes, kFlag + 1, 0, false, /*handoff=*/false,
                             kDdaArSplitBDma}}) {
    // Peer self + 1's shard starts where my partial last slice ends: nothing of mine may spill into it.
    for (const int r : {3, 1}) {
      SCOPED_TRACE(testing::Message() << c.nRanks << "r kind " << (int)c.kind << " shard " << c.shardBytes
                                      << " missing self + " << r);
      setUpFloat(c);
      const int missing = (c.self + r) % c.nRanks;
      reset(c, 1, -1, missing);
      run<float>(c);
      expectMyShard(/*reduced=*/true);
      const std::vector<uint8_t> recv = recvBuf_->copyTo();
      for (size_t i = 0; i < c.shardBytes; ++i) {
        ASSERT_EQ(recv[(size_t)missing * c.shardBytes + i], kUntouched) << "byte " << i;
      }
      expectEpochAdvanced(c);
    }
  }
}

// Every rank's split on two CUs each, far fewer blocks than the grid: waits on one block index let the grid drain.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitDrainsWhenTheGridIsNotCoResident) {
  constexpr int kRanks = 4;
  constexpr int kCusPerRank = 2;
  const Case c{kRanks, 0, 4, Kind::kSplit, 16, 512, kLargeShardBytes};
  int cus = 0;
  int clockKhz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
  HIP_CHECK(hipDeviceGetAttribute(&clockKhz, hipDeviceAttributeWallClockRate, 0));
  if (cus < kRanks * kCusPerRank) {
    GTEST_SKIP() << "needs " << kRanks * kCusPerRank << " CUs";
  }

  // setUp's scratch and peer table serve every rank; each rank's send, recv and epoch buffers are its own.
  setUpFloat(c);
  const std::vector<uint8_t> expected = Concat(reduced_);
  for (DeviceBuffer<uint8_t>& b : scratch_) {
    b.zero();
  }
  std::vector<DeviceBuffer<uint8_t>> sendBufs;
  std::vector<DeviceBuffer<uint8_t>> recvBufs;
  std::vector<DeviceBuffer<uint32_t>> epochs;
  for (int r = 0; r < kRanks; ++r) {
    sendBufs.emplace_back(send_[(size_t)r].size());
    sendBufs.back().copyFrom(send_[(size_t)r]);
    recvBufs.emplace_back(send_[(size_t)r].size());
    epochs.emplace_back(kEpochCells);
    epochs.back().copyFrom(std::vector<uint32_t>(kEpochCells, c.flag - 1));
  }

  std::vector<hipStream_t> streams((size_t)kRanks, nullptr);
  for (int r = 0; r < kRanks; ++r) {
    std::vector<uint32_t> mask(((size_t)cus + 31) / 32, 0u);
    for (int cu = r * kCusPerRank; cu < (r + 1) * kCusPerRank; ++cu) {
      mask[(size_t)cu / 32] |= 1u << (cu % 32);
    }
    HIP_CHECK(hipExtStreamCreateWithCUMask(&streams[(size_t)r], (uint32_t)mask.size(), mask.data()));
  }

  // With fewer blocks resident than the grid the run proves the drain; with all resident it still checks the harness.
  DeviceBuffer<int> counts(2);
  counts.zero();
  hipLaunchKernelGGL(CountResident, dim3(c.grid), dim3(c.block),
                     dda::common::ddaLL128ArSplitSharedBytes(c.block, kRanks), streams[0], counts.ptr, counts.ptr + 1,
                     (long long)clockKhz);
  ASSERT_EQ(waitStreamBounded(streams[0], kDrainTimeout), hipSuccess) << "the residency probe did not finish";
  const int mostResident = counts.copyTo()[1];

  // The 8 B split, the narrow one, then the narrow one handing off on 64 blocks (17 slices each) with pairwise, wide
  // and chunked B, and the 8 B one with DMA B (18 slices each), each a new epoch.
  struct Pass {
    Kind kind;
    bool handoff;
    unsigned grid;
    uint32_t cell;  // every epoch cell before the launch
    DdaArSplitB b = kDdaArSplitBPairwise;
  };
  const Pass passes[] = {{Kind::kSplit, false, c.grid, c.flag - 1}, {Kind::kSplitNarrow, false, c.grid, c.flag},
                         {Kind::kSplitNarrow, true, 64, c.flag + 1},
                         {Kind::kSplitNarrow, true, 64, c.flag + 2, kDdaArSplitBWide},
                         {Kind::kSplitNarrow, true, 64, c.flag + 3, kDdaArSplitBWideChunked},
                         {Kind::kSplit, true, 64, c.flag + 4, kDdaArSplitBDma}};
  for (const Pass& pass : passes) {
    SCOPED_TRACE(testing::Message() << "kind " << (int)pass.kind << " handoff " << pass.handoff << " b " << pass.b);
    // Host and null-stream work first: between two launches it would queue rank r + 1 behind rank r, which waits on it.
    *abortHost_ = 0;
    for (int r = 0; r < kRanks; ++r) {
      epochs[(size_t)r].copyFrom(std::vector<uint32_t>(kEpochCells, pass.cell));
      HIP_CHECK(hipMemset(recvBufs[(size_t)r].ptr, kUntouched, recvBufs[(size_t)r].count));
    }
    float** table = reinterpret_cast<float**>(scratchPtrs_->ptr);
    for (int r = 0; r < kRanks; ++r) {
      Case rank = c;
      rank.self = r;
      rank.kind = pass.kind;
      rank.handoff = pass.handoff;
      rank.grid = pass.grid;
      rank.b = pass.b;
      launch<float, 4>(rank, table, reinterpret_cast<float*>(recvBufs[(size_t)r].ptr),
                       reinterpret_cast<const float*>(sendBufs[(size_t)r].ptr), epochs[(size_t)r].ptr, abortDev_,
                       streams[(size_t)r]);
      HIP_CHECK(hipGetLastError());
    }
    const bool drained = drain(streams);
    EXPECT_TRUE(drained) << "the split did not drain with at most " << mostResident << " of " << pass.grid
                         << " blocks resident per rank";
    if (!drained) {
      break;  // the abort left this pass's lines and epochs half done
    }
    for (int r = 0; r < kRanks; ++r) {
      EXPECT_TRUE(recvBufs[(size_t)r].copyTo() == expected) << "rank " << r << " recvbuff is not the host fold";
    }
  }

  for (hipStream_t st : streams) {
    HIP_EXPECT(hipStreamDestroy(st));
  }
  if (mostResident >= (int)c.grid) {
    GTEST_SKIP() << "drained and folded, but the CU masks left all " << c.grid << " blocks resident";
  }
}

// ---- The split with the narrow 4 B flag ----

// Shards at the narrow slice's edges: one chunk, one short of a slice, one slice, one chunk past it, a partial third.
const size_t kNarrowEdgeShards[] = {16, kDdaLL128NarrowDataBytesPerSlice - 16, kDdaLL128NarrowDataBytesPerSlice,
                                   kDdaLL128NarrowDataBytesPerSlice + 16, 2 * kDdaLL128NarrowDataBytesPerSlice + 112};

// Every split case and the 1984 B slice edges in float: the host fold, and narrow lines carrying my shards.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitNarrowMatchesTheHostFold) {
  std::vector<Case> cases;
  for (const Case& c : kSplitCases) {
    cases.push_back(c);
    cases.back().kind = Kind::kSplitNarrow;
  }
  for (const size_t shard : kNarrowEdgeShards) {
    cases.push_back({8, 5, 8, Kind::kSplitNarrow, 2, 512, shard});
    cases.push_back({5, 2, 0, Kind::kSplitNarrow, 1, 64, shard, kFlag, 0, true});
  }
  for (const Case& c : cases) {
    setUpFloat(c);
    expectNarrowMatchesCoop<float>(c, "float");
    EXPECT_TRUE(recvBuf_->copyTo() == Concat(reduced_)) << "not the host fold";
  }
}

// bf16 against its host fold, half and both fp8 against the cooperative writer, out of place and in place.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitNarrowMatchesOnSmallerTypes) {
  const Case cases[] = {{5, 2, 0, Kind::kSplitNarrow, 1, 64, kSmallShardBytes},
                        {8, 5, 8, Kind::kSplitNarrow, 2, 512, kMediumShardBytes, kFlag + 1},
                        {8, 5, 8, Kind::kSplitNarrow, 8, 512, kLargeShardBytes},
                        {16, 9, 0, Kind::kSplitNarrow, 16, 512, kMediumShardBytes, kFlag, 0, true}};
  for (const Case& c : cases) {
    const auto send = Bf16Messages(c.nRanks, c.shardBytes);
    setUp(c.nRanks, c.self, c.shardBytes, send, Bf16Reduced(send, c.nRanks, c.shardBytes));
    expectNarrowMatchesCoop<bf16>(c, "bf16");
    EXPECT_TRUE(recvBuf_->copyTo() == Concat(reduced_)) << "bf16: not the host fold";
    expectNarrowMatchesCoop<half>(c, "half");  // the same bits are finite halves
    setUp(c.nRanks, c.self, c.shardBytes, Fp8Rows(c.nRanks, (size_t)c.nRanks * c.shardBytes, 0),
          Fp8Rows(c.nRanks, c.shardBytes, 5));
    expectNarrowMatchesCoop<dda::common::fp8e4m3>(c, "fp8e4m3");
    expectNarrowMatchesCoop<dda::common::fp8e5m2>(c, "fp8e5m2");
  }
}

// 12 launches through the 2^32 wrap, peers 1 ms late, on a scratch first holding launch 0's plain epoch in every dword.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitNarrowFlagWrapsAcrossLaunches) {
  constexpr int kLaunches = 12;
  constexpr uint32_t kFirstCell = 0xFFFFFFF8u;
  Case c{8, 5, 8, Kind::kSplitNarrow, 2, 512, kMediumShardBytes};
  int clockKhz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clockKhz, hipDeviceAttributeWallClockRate, 0));
  const long long ticks = clockKhz;  // 1 ms
  setUpFloat(c);
  kind_ = c.kind;
  const std::vector<uint32_t> poison(2 * bankBytes(c.nRanks, c.shardBytes) / sizeof(uint32_t), kFirstCell + 1);
  for (DeviceBuffer<uint8_t>& b : scratch_) {
    HIP_CHECK(hipMemcpy(b.ptr, poison.data(), b.count, hipMemcpyHostToDevice));
  }
  epoch_.copyFrom(std::vector<uint32_t>(kEpochCells, kFirstCell));
  uint32_t cell = kFirstCell;
  for (int i = 0; i < kLaunches; ++i) {
    *abortHost_ = 0;
    c.flag = cell + 1 == 0 ? 2 : cell + 1;
    SCOPED_TRACE(testing::Message() << "launch " << i << " epoch " << c.flag);
    const auto send = FloatMessages(c.nRanks, c.shardBytes, i);
    setMessages(send, FloatReduced(send, c.nRanks, c.shardBytes));
    HIP_CHECK(hipMemset(recvBuf_->ptr, kUntouched, recvBuf_->count));
    bankOffset_ = (c.flag & 1u) * bankBytes(nRanks_, shardBytes_);
    publish(area(self_, 0), copyRows(), -1, c.flag, side_, ticks);
    publish(area(self_, nRanks_), reducedRows(), -1, c.flag, side_, ticks);
    run<float>(c);
    EXPECT_TRUE(drain({side_})) << "the late peers did not finish";
    latePublishBuffers_.clear();
    if (*abortHost_ != 0) {
      break;  // a drain timed out: later launches would start aborted
    }
    EXPECT_TRUE(recvBuf_->copyTo() == Concat(reduced_)) << "not the host fold";
    const std::vector<uint32_t> epoch = epoch_.copyTo();
    for (int e = 0; e < (int)kEpochCells; ++e) {
      EXPECT_EQ(epoch[(size_t)e], e < epochLen(c) ? c.flag : kFirstCell) << "cell " << e;
    }
    cell = c.flag;
  }
  EXPECT_EQ(cell, 2u + kLaunches - 8) << "the run crossed the wrap";
}

// ---- The split's B -> C hand-off through LDS ----

// Every split case whose blocks fit the hand-off area, with each flag, 18 slices a block and the area's 32 in place.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitHandoffMatchesTheHostFold) {
  // The runtime accepts the limit the launcher raises once per device: a full block and the most ranks.
  HIP_CHECK(hipFuncSetAttribute(
    reinterpret_cast<const void*>(&dda::common::ddaAllReduceTwoShotLL128Split<float, kDdaLL128FlagBytesNarrow, true>),
    hipFuncAttributeMaxDynamicSharedMemorySize,
    (int)dda::common::ddaLL128ArSplitSharedBytes(dda::common::kDdaLL128ArThreads, dda::common::kDdaMaxNranks, true)));
  std::vector<Case> cases(std::begin(kSplitCases), std::end(kSplitCases));
  cases.push_back({8, 5, 8, Kind::kSplit, 64, 512, kLargeShardBytes});
  cases.push_back({4, 1, 4, Kind::kSplit, 1, 512, 32 * kDdaLL128DataBytesPerSlice, kFlag, 0, true});
  for (Case c : cases) {
    c.handoff = true;
    for (const Kind kind : {Kind::kSplit, Kind::kSplitNarrow}) {
      c.kind = kind;
      if (!handoffFits(c)) {
        continue;
      }
      setUpFloat(c);
      expectSplitMatchesCoop<float>(c, "float");
      EXPECT_TRUE(recvBuf_->copyTo() == Concat(reduced_)) << "not the host fold";
    }
  }
}

// bf16 against its host fold and fp8 against the cooperative writer, handed off with each flag, in place too.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitHandoffMatchesOnBf16AndFp8) {
  for (Case c : {Case{8, 5, 8, Kind::kSplit, 2, 512, kMediumShardBytes, kFlag + 1},
                 Case{16, 9, 0, Kind::kSplit, 16, 512, kMediumShardBytes, kFlag, 0, true}}) {
    c.handoff = true;
    for (const Kind kind : {Kind::kSplit, Kind::kSplitNarrow}) {
      c.kind = kind;
      const auto send = Bf16Messages(c.nRanks, c.shardBytes);
      setUp(c.nRanks, c.self, c.shardBytes, send, Bf16Reduced(send, c.nRanks, c.shardBytes));
      expectSplitMatchesCoop<bf16>(c, "bf16");
      EXPECT_TRUE(recvBuf_->copyTo() == Concat(reduced_)) << "bf16: not the host fold";
      setUp(c.nRanks, c.self, c.shardBytes, Fp8Rows(c.nRanks, (size_t)c.nRanks * c.shardBytes, 0),
            Fp8Rows(c.nRanks, c.shardBytes, 5));
      expectSplitMatchesCoop<dda::common::fp8e4m3>(c, "fp8e4m3");
    }
  }
}

// ---- Wide B ----

// Wide B in 1 to 7 rounds (n - 1 copies of ddaArSplitWideRoundSlices(n) slices each; 17 ranks, past the rule's 8
// copies, fill the stage), 2-warp blocks, a block owning no slice; chunked B in chunks of 8 copies and a short last
// one, rounds of 2 or 6 slices; in place and bank 1.
std::vector<Case> wideCases() {
  return {wideCase(9, 4, 2, 512, kMediumShardBytes, kDdaArSplitBWide),
          wideCase(8, 5, 4, 512, kMediumShardBytes, kDdaArSplitBWide),
          wideCase(5, 2, 1, 64, kSmallShardBytes, kDdaArSplitBWide),
          wideCase(3, 1, 2, 512, kSmallShardBytes, kDdaArSplitBWide),
          wideCase(4, 1, 4, 512, kSmallShardBytes, kDdaArSplitBWide),
          wideCase(17, 16, 2, 512, kMediumShardBytes, kDdaArSplitBWide),
          wideCase(8, 5, 64, 512, kLargeShardBytes, kDdaArSplitBWide, Kind::kSplit, kFlag + 1),
          wideCase(8, 5, 2, 512, kMediumShardBytes, kDdaArSplitBWide, Kind::kSplit, kFlag, /*inPlace=*/true),
          wideCase(18, 3, 2, 512, kMediumShardBytes, kDdaArSplitBWideChunked),
          wideCase(32, 17, 4, 512, kSmallShardBytes, kDdaArSplitBWideChunked),
          wideCase(20, 7, 1, 64, kSmallShardBytes, kDdaArSplitBWideChunked),
          wideCase(25, 24, 8, 512, kMediumShardBytes, kDdaArSplitBWideChunked, Kind::kSplit, kFlag + 1, false, 11),
          wideCase(5, 2, 2, 512, kMediumShardBytes, kDdaArSplitBWideChunked),
          wideCase(18, 0, 2, 512, kMediumShardBytes, kDdaArSplitBWideChunked, Kind::kSplit, kFlag, /*inPlace=*/true)};
}

// Wide and chunked B with each flag against the cooperative writer and the host fold.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitWideBMatchesTheHostFold) {
  // The runtime accepts the limit the launcher raises for each form: a full block and the most ranks.
  for (const DdaArSplitB b : {kDdaArSplitBWide, kDdaArSplitBWideChunked}) {
    const size_t lds =
      dda::common::ddaLL128ArSplitSharedBytes(dda::common::kDdaLL128ArThreads, dda::common::kDdaMaxNranks, true, b);
    const Case c = wideCase(8, 0, 1, 512, kSmallShardBytes, b, Kind::kSplitNarrow);
    HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(splitKernel<float, kDdaLL128FlagBytesNarrow>(c)),
                                  hipFuncAttributeMaxDynamicSharedMemorySize, (int)lds));
  }
  for (Case c : wideCases()) {
    for (const Kind kind : {Kind::kSplit, Kind::kSplitNarrow}) {
      c.kind = kind;
      ASSERT_TRUE(handoffFits(c));
      setUpFloat(c);
      expectSplitMatchesCoop<float>(c, "float");
      EXPECT_TRUE(recvBuf_->copyTo() == Concat(reduced_)) << "not the host fold";
    }
  }
}

// bf16 against its host fold and fp8 against the cooperative writer, wide and chunked, with each flag, in place too.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitWideBMatchesOnBf16AndFp8) {
  for (Case c : {wideCase(9, 4, 2, 512, kMediumShardBytes, kDdaArSplitBWide),
                 wideCase(8, 5, 2, 512, kMediumShardBytes, kDdaArSplitBWide, Kind::kSplit, kFlag, /*inPlace=*/true),
                 wideCase(18, 3, 2, 512, kMediumShardBytes, kDdaArSplitBWideChunked),
                 wideCase(18, 0, 2, 512, kMediumShardBytes, kDdaArSplitBWideChunked, Kind::kSplit, kFlag,
                          /*inPlace=*/true)}) {
    for (const Kind kind : {Kind::kSplit, Kind::kSplitNarrow}) {
      c.kind = kind;
      const auto send = Bf16Messages(c.nRanks, c.shardBytes);
      setUp(c.nRanks, c.self, c.shardBytes, send, Bf16Reduced(send, c.nRanks, c.shardBytes));
      expectSplitMatchesCoop<bf16>(c, "bf16");
      EXPECT_TRUE(recvBuf_->copyTo() == Concat(reduced_)) << "bf16: not the host fold";
      setUp(c.nRanks, c.self, c.shardBytes, Fp8Rows(c.nRanks, (size_t)c.nRanks * c.shardBytes, 0),
            Fp8Rows(c.nRanks, c.shardBytes, 5));
      expectSplitMatchesCoop<dda::common::fp8e4m3>(c, "fp8e4m3");
    }
  }
}

// ---- DMA B ----

// Every split case with the 8 B flag, own slot and (where it fits) the hand-off, plus many groups: 16 ranks of 41
// slices on one block (7 a group, own slot) and on two (5 a group handed off), 32 ranks (3 a group), 33 ranks on
// 256-thread blocks in bank 1, in place, and shards of 16 B to 1936 B (one or two slices, a block owning none).
std::vector<Case> dmaCases() {
  std::vector<Case> base(std::begin(kSplitCases), std::end(kSplitCases));
  base.push_back({16, 9, 0, Kind::kSplit, 1, 512, kMediumShardBytes});
  base.push_back({16, 9, 0, Kind::kSplit, 2, 512, kMediumShardBytes});
  base.push_back({32, 17, 0, Kind::kSplit, 1, 512, kMediumShardBytes});
  base.push_back({33, 5, 0, Kind::kSplit, 2, 256, kMediumShardBytes, kFlag + 1});
  base.push_back({16, 3, 0, Kind::kSplit, 1, 512, kMediumShardBytes, kFlag, 0, true});
  for (const int shard : {16, 128, kDdaLL128DataBytesPerSlice - 16, kDdaLL128DataBytesPerSlice,
                          kDdaLL128DataBytesPerSlice + 16}) {
    base.push_back({8, 5, 8, Kind::kSplit, 2, 512, (size_t)shard});  // 1 or 2 slices, a full or partial last one
  }
  std::vector<Case> out;
  for (Case c : base) {
    c.b = kDdaArSplitBDma;
    for (const bool handoff : {false, true}) {
      c.handoff = handoff;
      if (!handoff || handoffFits(c)) {
        out.push_back(c);
      }
    }
  }
  return out;
}

// DMA B leaves the cooperative two-shot's recvbuff and every peer's lines: the fold order and the add are pairwise B's.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitDmaMatchesTheHostFold) {
  // The runtime accepts the launcher's limit for DMA B: its LDS budget.
  HIP_CHECK(hipFuncSetAttribute(
    reinterpret_cast<const void*>(
      &dda::common::ddaAllReduceTwoShotLL128Split<float, kDdaLL128FlagBytesWide, true, kDdaArSplitBDma>),
    hipFuncAttributeMaxDynamicSharedMemorySize, (int)dda::common::kDdaArSplitDmaLdsBudget));
  int handoffRows = 0;
  for (const Case& c : dmaCases()) {
    handoffRows += c.handoff ? 1 : 0;
    setUpFloat(c);
    const Snapshot want = expectMatchesCoop<float>(c);
    ASSERT_TRUE(want.recv == Concat(reduced_)) << "the cooperative two-shot left the host fold";
  }
  EXPECT_GT(handoffRows, 10) << "the hand-off rows ran";
}

// bf16 against its host fold and fp8 against the cooperative writer, own slot and hand-off, in place too.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitDmaMatchesOnBf16AndFp8) {
  for (Case c : {Case{5, 2, 0, Kind::kSplit, 1, 64, kSmallShardBytes},
                 Case{8, 5, 8, Kind::kSplit, 8, 512, kLargeShardBytes},
                 Case{16, 9, 0, Kind::kSplit, 1, 512, kMediumShardBytes},
                 Case{8, 5, 8, Kind::kSplit, 2, 512, kMediumShardBytes, kFlag, 0, true}}) {
    c.b = kDdaArSplitBDma;
    for (const bool handoff : {false, true}) {
      c.handoff = handoff;
      if (handoff && !handoffFits(c)) {
        continue;
      }
      const auto send = Bf16Messages(c.nRanks, c.shardBytes);
      setUp(c.nRanks, c.self, c.shardBytes, send, Bf16Reduced(send, c.nRanks, c.shardBytes));
      expectMatchesCoop<bf16>(c);
      EXPECT_TRUE(recvBuf_->copyTo() == Concat(reduced_)) << "bf16: not the host fold";
      setUp(c.nRanks, c.self, c.shardBytes, Fp8Rows(c.nRanks, (size_t)c.nRanks * c.shardBytes, 0),
            Fp8Rows(c.nRanks, c.shardBytes, 5));
      expectMatchesCoop<dda::common::fp8e4m3>(c);
    }
  }
}

// One peer's copy lands 1 ms after the launch, so the first load likely sees it stale (a wall-clock race, not a sync)
// and reloads until fresh: missing (zeroed), or partly published with only its first 512 B, only each 512 B's first
// line (the lines lane 7 checks), or all but those fresh. The result is the same.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitDmaReloadsStaleCopies) {
  int clockKhz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&clockKhz, hipDeviceAttributeWallClockRate, 0));
  for (Case c : {Case{8, 5, 8, Kind::kSplit, 2, 512, kSmallShardBytes},
                 Case{16, 9, 0, Kind::kSplit, 1, 512, kMediumShardBytes},
                 Case{5, 2, 0, Kind::kSplit, 4, 512, kLargeShardBytes, kFlag + 1}}) {
    c.b = kDdaArSplitBDma;
    for (const bool handoff : {false, true}) {
      c.handoff = handoff;
      if (handoff && !handoffFits(c)) {
        continue;
      }
      for (const int late : {1, c.nRanks - 1}) {
        for (const uint32_t keep : {0x0000u, 0x000Fu, 0x1111u, 0xFFF0u, 0xEEEEu}) {
          SCOPED_TRACE(testing::Message() << c.nRanks << "r shard " << c.shardBytes << " handoff " << handoff
                                          << " late copy from self + " << late << " fresh lines 0x" << std::hex
                                          << keep);
          setUpFloat(c);
          const int peer = (c.self + late) % c.nRanks;
          reset(c, 0, keep == 0 ? peer : -1);
          if (keep != 0) {
            uint64_t* const copy = area(self_, 0) + (size_t)peer * slotWords(nRanks_, shardBytes_);
            hipLaunchKernelGGL(TearLines, dim3((unsigned)slicesOf(c.kind, c.shardBytes)), dim3(256), 0, nullptr,
                               reinterpret_cast<uint32_t*>(copy), keep);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
          }
          // Only that copy, whole, on the side stream once 1 ms passed: the launch sees it late.
          publish(area(self_, 0), copyRows(), -1, c.flag, side_, clockKhz, peer);
          run<float>(c);
          EXPECT_TRUE(drain({side_})) << "the late peer did not finish";
          latePublishBuffers_.clear();
          EXPECT_TRUE(recvBuf_->copyTo() == Concat(reduced_)) << "recvbuff is not the host fold";
          expectEpochAdvanced(c);
        }
      }
    }
  }
}

// ---- Uneven shards ----

// A message of `total` bytes (16 B chunks that do not split into n equal ones) over n ranks, every rank on this GPU at
// once on one shared scratch table: stride ceil(total / n) up to 16 B, the last shards shorter or empty.
struct UnevenCase {
  int nRanks;
  size_t total;
  unsigned grid;
  unsigned block;
  Kind kind;  // kSplit (8 B flag) or kSplitNarrow
  bool handoff;
  DdaArSplitB b;
  bool inPlace;
};

size_t unevenStride(const UnevenCase& c) {
  return ((c.total + (size_t)c.nRanks - 1) / (size_t)c.nRanks + 15) / 16 * 16;
}

// Element i belongs to shard i / stride, folded by its owner: the owner's value, then ranks owner + 1, owner + 2, ...
std::vector<uint8_t> UnevenFold(const std::vector<std::vector<uint8_t>>& send, int nRanks, size_t total,
                                size_t stride) {
  std::vector<uint8_t> out(total);
  float* acc = reinterpret_cast<float*>(out.data());
  for (size_t i = 0; i < total / sizeof(float); ++i) {
    const int owner = (int)(i * sizeof(float) / stride);
    acc[i] = reinterpret_cast<const float*>(send[(size_t)owner].data())[i];
    for (int r = 1; r < nRanks; ++r) {
      acc[i] = acc[i] + reinterpret_cast<const float*>(send[(size_t)((owner + r) % nRanks)].data())[i];
    }
  }
  return out;
}

// Rank r's split launch for c on uneven shards.
Case unevenRank(const UnevenCase& c, int r) {
  Case rank{c.nRanks, r, 0, c.kind, c.grid, c.block, unevenStride(c)};
  rank.inPlace = c.inPlace;
  rank.handoff = c.handoff;
  rank.b = c.b;
  return rank;
}

// Whether every rank's grid of c can be resident at once: a rank's blocks wait on the other ranks' blocks, so a GPU
// that cannot hold them all hangs the run. Unknown counts as resident.
bool unevenResident(const UnevenCase& c) {
  const Case rank = unevenRank(c, 0);
  const void* kernel =
    c.kind == Kind::kSplitNarrow
      ? reinterpret_cast<const void*>(splitKernel<float, kDdaLL128FlagBytesNarrow, size_t>(rank))
      : reinterpret_cast<const void*>(splitKernel<float, kDdaLL128FlagBytesWide, size_t>(rank));
  const size_t lds = splitLds(rank);
  int perCu = 0;
  int cus = 0;
  if (hipFuncSetAttribute(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, (int)lds) != hipSuccess ||
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&perCu, kernel, (int)c.block, lds) != hipSuccess ||
      hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0) != hipSuccess) {
    return true;
  }
  return (size_t)perCu * (size_t)cus >= (size_t)c.nRanks * c.grid;
}

// Bytes past each rank's message that no launch may touch.
constexpr size_t kUnevenGuardBytes = 256;

// Uneven rows: 3, 5, 6, 9, 12 and 18 ranks; the last shard shorter, empty (12 ranks of one 16 B chunk each, 18 ranks),
// or a partial slice; each flag, own slot and hand-off, in place; wide, chunked and DMA B on uneven shards.
const UnevenCase kUnevenCases[] = {
  {3, 750016, 4, 512, Kind::kSplit, false, kDdaArSplitBPairwise, false},          // last shard 249984 B
  {5, 1000016, 4, 512, Kind::kSplitNarrow, true, kDdaArSplitBPairwise, false},    // stride 200016
  {6, 3 * 1920 * 6 + 160, 2, 512, Kind::kSplitNarrow, true, kDdaArSplitBPairwise, false},  // stride 5792, 3 slices
  {12, 500000, 4, 512, Kind::kSplit, true, kDdaArSplitBPairwise, false},          // stride 41680
  {12, 12 * 4096 + 48, 2, 256, Kind::kSplitNarrow, false, kDdaArSplitBPairwise, true},  // last shard 3968 B
  {12, 11 * 16, 1, 128, Kind::kSplit, false, kDdaArSplitBPairwise, false},        // one chunk a shard, the 12th empty
  {6, 1125008, 4, 512, Kind::kSplit, false, kDdaArSplitBPairwise, true},          // stride 187504
  {9, 750016, 4, 512, Kind::kSplitNarrow, true, kDdaArSplitBWide, false},         // stride 83344
  {9, 64 * 16, 1, 512, Kind::kSplitNarrow, true, kDdaArSplitBWide, true},        // eight 128 B shards, the 9th empty
  {18, 1000000, 4, 512, Kind::kSplitNarrow, true, kDdaArSplitBWideChunked, false},
  {18, 17 * 160, 1, 512, Kind::kSplit, true, kDdaArSplitBWideChunked, false},     // the 18th shard empty
  {9, 750016, 4, 512, Kind::kSplit, true, kDdaArSplitBDma, false},
  {5, 1000016, 8, 512, Kind::kSplit, false, kDdaArSplitBDma, true},
  {12, 500000, 8, 512, Kind::kSplit, true, kDdaArSplitBDma, false},
  // The last shard one slice short of the stride's: a dead slice on that rank only.
  {9, 743168, 4, 512, Kind::kSplit, true, kDdaArSplitBDma, false},                // stride 82576: 44 slices, last 43
  {9, 1227008, 9, 512, Kind::kSplit, true, kDdaArSplitBDma, false},               // 72 slices, last 71; 8 a block
  {9, 1024, 1, 512, Kind::kSplit, true, kDdaArSplitBDma, true},                   // eight 128 B shards, the 9th empty
  {5, 992064, 4, 512, Kind::kSplitNarrow, true, kDdaArSplitBPairwise, false},     // 101 narrow slices, last 100
  {6, 35792, 2, 512, Kind::kSplitNarrow, true, kDdaArSplitBPairwise, false},      // stride 5968: 4 slices, last 3
};

// Every rank's recvbuff is the host fold of the whole message, and no rank writes past it.
TEST_F(DdaAllReduceFabricLL128TwoShot, SplitUnevenShardsMatchTheHostFold) {
  int notRun = 0;
  for (const UnevenCase& c : kUnevenCases) {
    const size_t stride = unevenStride(c);
    SCOPED_TRACE(testing::Message() << c.nRanks << "r total " << c.total << " stride " << stride << " " << c.grid
                                    << "x" << c.block << " flag " << flagBytes(c.kind) << " handoff " << c.handoff
                                    << " b " << c.b << " inPlace " << c.inPlace);
    ASSERT_NE(c.total % (16 * (size_t)c.nRanks), 0u) << "an uneven message";
    ASSERT_TRUE(!c.handoff || handoffFits(unevenRank(c, 0)));
    if (!unevenResident(c)) {
      // Not a pass: the row did not run; logged so a gate report shows which rows this GPU could not hold.
      std::printf("[ NOT RUN  ] uneven %dr total %zu %ux%u: the ranks' grids cannot all be resident\n", c.nRanks,
                  c.total, c.grid, c.block);
      ++notRun;
      continue;
    }
    std::vector<std::vector<uint8_t>> send((size_t)c.nRanks, std::vector<uint8_t>(c.total + kUnevenGuardBytes,
                                                                                  kUntouched));
    for (int r = 0; r < c.nRanks; ++r) {
      float* f = reinterpret_cast<float*>(send[(size_t)r].data());
      for (size_t i = 0; i < c.total / sizeof(float); ++i) {
        f[i] = 1.0f / (float)(3 + r) + (float)((r * 131 + (int)i * 17) % 1000) * 1.0e-3f;
      }
    }
    std::vector<uint8_t> expected = UnevenFold(send, c.nRanks, c.total, stride);
    expected.resize(c.total + kUnevenGuardBytes, kUntouched);
    const size_t bank = bankBytes(c.nRanks, stride);
    std::vector<DeviceBuffer<uint8_t>> scratch;
    std::vector<uint8_t*> table;
    std::vector<DeviceBuffer<uint8_t>> sendBufs;
    std::vector<DeviceBuffer<uint8_t>> recvBufs;
    std::vector<DeviceBuffer<uint32_t>> epochs;
    std::vector<hipStream_t> streams((size_t)c.nRanks, nullptr);
    for (int r = 0; r < c.nRanks; ++r) {
      scratch.emplace_back(bank);
      scratch.back().zero();
      table.push_back(scratch.back().ptr);
      sendBufs.emplace_back(send[(size_t)r].size());
      sendBufs.back().copyFrom(send[(size_t)r]);
      recvBufs.emplace_back(send[(size_t)r].size());
      HIP_CHECK(hipMemset(recvBufs.back().ptr, kUntouched, recvBufs.back().count));
      epochs.emplace_back(kEpochCells);
      epochs.back().copyFrom(std::vector<uint32_t>(kEpochCells, kFlag - 1));
      HIP_CHECK(hipStreamCreateWithFlags(&streams[(size_t)r], hipStreamNonBlocking));
    }
    DeviceBuffer<uint8_t*> dTable(table.size());
    dTable.copyFrom(table);
    // Host and null-stream work above, before any launch: between two launches it would hold rank r + 1 behind rank r.
    *abortHost_ = 0;
    float** tab = reinterpret_cast<float**>(dTable.ptr);
    for (int r = 0; r < c.nRanks; ++r) {
      const Case rank = unevenRank(c, r);
      float* recv = reinterpret_cast<float*>((c.inPlace ? sendBufs : recvBufs)[(size_t)r].ptr);
      const float* snd = reinterpret_cast<const float*>(sendBufs[(size_t)r].ptr);
      if (c.kind == Kind::kSplitNarrow) {
        launchSplit<float, kDdaLL128FlagBytesNarrow>(rank, tab, recv, snd, epochs[(size_t)r].ptr,
                                                     DdaKernelParams{abortDev_}, streams[(size_t)r], c.total);
      } else {
        launchSplit<float, kDdaLL128FlagBytesWide>(rank, tab, recv, snd, epochs[(size_t)r].ptr,
                                                   DdaKernelParams{abortDev_}, streams[(size_t)r], c.total);
      }
      HIP_CHECK(hipGetLastError());
    }
    const bool drained = drain(streams);
    EXPECT_TRUE(drained) << "the uneven split did not finish within " << kDrainTimeout.count() << " s";
    for (int r = 0; drained && r < c.nRanks; ++r) {
      EXPECT_TRUE((c.inPlace ? sendBufs : recvBufs)[(size_t)r].copyTo() == expected)
        << "rank " << r << ": not the host fold, or bytes past the message touched";
    }
    for (hipStream_t st : streams) {
      HIP_EXPECT(hipStreamDestroy(st));
    }
  }
  if (notRun == (int)std::size(kUnevenCases)) {
    GTEST_SKIP() << "no uneven row fits this GPU's resident blocks";
  }
}

}  // namespace

}  // namespace RcclUnitTesting
