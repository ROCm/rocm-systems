/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the agreement that creates the DDA AllToAll push / Direct buffers on the
// first eligible AllToAll (src/algorithms/dda/alltoall/dda_alltoall_lazy.h). N ranks run as N threads
// over an in-memory allgather; the per-rank ops are fakes that can fail any step. The properties
// under test: every rank ends with the same parts and push size whatever fails where, every rank
// makes the same number of bootstrap calls (no rank is ever left waiting), and every local resource
// of a part that is not agreed is released. Also the HIP-free push host helpers of dda_alltoall.h
// (DdaA2aPushHostMicrotest): topology sibling, the round loop's poison on failure, and the error-word
// mapping.

#include <gtest/gtest.h>

#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "algorithms/dda/alltoall/dda_alltoall.h"
#include "algorithms/dda/alltoall/dda_alltoall_lazy.h"

namespace {

using dda_a2a_lazy::kDirect;
using dda_a2a_lazy::kImplicit;
using dda_a2a_lazy::kPush;

constexpr int64_t kFlagBytes = 2ll << 20;  // dda::common::kDdaA2aPushFlagBytes (HIP-only header)
constexpr int64_t kGiB = 1ll << 30;

// Generation-counted allgather over threads: every rank deposits its slot, the last arrival
// publishes, everyone copies the full array out. A failing rank still takes part (its failure is
// reported to it only), like a broken socket on one side.
class FakeBootstrap {
 public:
  explicit FakeBootstrap(int n) : n_(n) {}
  ncclResult_t allGather(int rank, void* buf, size_t bytes) {
    std::unique_lock<std::mutex> lk(mu_);
    if (data_.size() != n_ * bytes) data_.assign(n_ * bytes, 0);
    std::memcpy(data_.data() + rank * bytes, static_cast<char*>(buf) + rank * bytes, bytes);
    const uint64_t gen = gen_;
    if (++arrived_ == n_) {
      published_ = data_;
      data_.clear();
      arrived_ = 0;
      ++gen_;
      cv_.notify_all();
    } else {
      cv_.wait(lk, [&] { return gen_ != gen; });
    }
    std::memcpy(buf, published_.data(), n_ * bytes);
    ++calls_[rank];
    return ncclSuccess;
  }
  int calls(int rank) {
    std::lock_guard<std::mutex> lk(mu_);
    return calls_[rank];
  }

 private:
  std::mutex mu_;
  std::condition_variable cv_;
  int n_;
  int arrived_ = 0;
  uint64_t gen_ = 0;
  std::vector<char> data_, published_;
  int calls_[64] = {};
};

struct Blob {
  uint64_t pushDesc;
  uint64_t directDesc;
  uint64_t implicitDesc;
};

struct RankPlan {
  uint32_t setupFails = 0;   // parts whose local create/export fails
  uint32_t importFails = 0;  // parts whose peer import fails
  int64_t pushBytes = 1ll << 30;
  uint32_t want = kPush | kDirect;
  int64_t freeBytes = -1;    // >= 0: device memory model; setup() sizes push as the HIP ops do
};

class FakeOps {
 public:
  using Blob = ::Blob;
  FakeOps(FakeBootstrap* bs, int rank, int n, RankPlan plan) : bs_(bs), rank_(rank), n_(n), plan_(plan) {}
  int rank() const { return rank_; }
  int nRanks() const { return n_; }
  ncclResult_t allGather(void* buf, size_t bytes) { return bs_->allGather(rank_, buf, bytes); }
  void setup(uint32_t want, int64_t pushBytes, dda_a2a_lazy::Rec<Blob>* mine) {
    mine->pushBytes = pushBytes;
    mine->blob.pushDesc = 0x1000 + rank_;
    mine->blob.directDesc = 0x2000 + rank_;
    mine->blob.implicitDesc = 0x3000 + rank_;
    freeLeft_ = plan_.freeBytes;
    for (uint32_t part : {kPush, kDirect, kImplicit}) {
      if (!(want & part)) continue;
      if (plan_.setupFails & part) continue;
      if (part == kPush && plan_.freeBytes >= 0) {
        // As DdaA2aLazyOps::setup: try pushSizes() largest first; an allocation succeeds whenever
        // it physically fits (the h13-13 case: 1 GiB fits, the later p2p connections do not).
        bool got = false;
        for (int64_t b : dda_a2a_lazy::pushSizes(pushBytes, n_, kFlagBytes, plan_.freeBytes)) {
          if (kFlagBytes + b > plan_.freeBytes) continue;
          mine->pushBytes = b;
          freeLeft_ = plan_.freeBytes - kFlagBytes - b;
          got = true;
          break;
        }
        if (!got) continue;
      }
      held_ |= part;
      mine->ok |= part;
    }
  }
  uint32_t import(const dda_a2a_lazy::Rec<Blob>* all, uint32_t parts, int64_t) {
    for (int i = 0; i < n_; ++i) {
      if ((parts & kPush) && all[i].blob.pushDesc != uint64_t(0x1000 + i)) descOk_ = false;
      if ((parts & kDirect) && all[i].blob.directDesc != uint64_t(0x2000 + i)) descOk_ = false;
      if ((parts & kImplicit) && all[i].blob.implicitDesc != uint64_t(0x3000 + i)) descOk_ = false;
    }
    imported_ |= parts & ~plan_.importFails;
    return parts & ~plan_.importFails;
  }
  void release(uint32_t parts) {
    if ((parts & held_ & kPush) && plan_.freeBytes >= 0) freeLeft_ = plan_.freeBytes;
    released_ |= parts;
    held_ &= ~parts;
    imported_ &= ~parts;
  }
  uint32_t held_ = 0, imported_ = 0, released_ = 0;
  int64_t freeLeft_ = -1;
  bool descOk_ = true;

 private:
  FakeBootstrap* bs_;
  int rank_, n_;
  RankPlan plan_;
};

struct Outcome {
  ncclResult_t r;
  dda_a2a_lazy::Result res;
  uint32_t held, released;
  bool descOk;
  int calls;
  int64_t freeLeft;
};

std::vector<Outcome> RunRanks(const std::vector<RankPlan>& plans) {
  const int n = (int)plans.size();
  FakeBootstrap bs(n);
  std::vector<Outcome> out(n);
  std::vector<std::thread> th;
  for (int r = 0; r < n; ++r) {
    th.emplace_back([&, r] {
      FakeOps ops(&bs, r, n, plans[r]);
      Outcome& o = out[r];
      o.r = dda_a2a_lazy::agree(ops, plans[r].want, plans[r].pushBytes, &o.res);
      o.held = ops.held_;
      o.released = ops.released_;
      o.descOk = ops.descOk_;
      o.freeLeft = ops.freeLeft_;
    });
  }
  for (auto& t : th) t.join();
  for (int r = 0; r < n; ++r) out[r].calls = bs.calls(r);
  return out;
}

void ExpectUniform(const std::vector<Outcome>& out, uint32_t parts, int64_t pushBytes, int calls) {
  for (size_t r = 0; r < out.size(); ++r) {
    EXPECT_EQ(ncclSuccess, out[r].r) << r;
    EXPECT_EQ(parts, out[r].res.parts) << r;
    EXPECT_EQ(pushBytes, out[r].res.pushBytes) << r;
    EXPECT_EQ(calls, out[r].calls) << r;
    EXPECT_EQ(parts, out[r].held) << "rank " << r << " keeps exactly the agreed parts";
    EXPECT_TRUE(out[r].descOk) << r;
  }
}

}  // namespace

TEST(DdaA2aLazyMicrotest, AllRanksOk_BothPartsAgreed) {
  ExpectUniform(RunRanks(std::vector<RankPlan>(8)), kPush | kDirect, 1ll << 30, 2);
}

// Allocation failure on one rank only (e.g. OOM at the first large AllToAll): that part is off on
// every rank, the other part still comes up, nobody waits.
TEST(DdaA2aLazyMicrotest, SetupFailureOnOneRankDisablesThatPartEverywhere) {
  std::vector<RankPlan> plans(8);
  plans[3].setupFails = kPush;
  auto out = RunRanks(plans);
  ExpectUniform(out, kDirect, 0, 2);
  for (int r = 0; r < 8; ++r) {
    if (r != 3) EXPECT_TRUE(out[r].released & kPush) << "rank " << r << " frees its unused push buffer";
  }
}

TEST(DdaA2aLazyMicrotest, SetupFailureOfEverythingOnOneRankIsOneRound) {
  std::vector<RankPlan> plans(4);
  plans[1].setupFails = kPush | kDirect;
  ExpectUniform(RunRanks(plans), 0, 0, 1);
}

// Peer import failure on one rank: decided in the second round, still uniform.
TEST(DdaA2aLazyMicrotest, ImportFailureOnOneRankDisablesThatPartEverywhere) {
  std::vector<RankPlan> plans(16);
  plans[11].importFails = kDirect;
  auto out = RunRanks(plans);
  ExpectUniform(out, kPush, 1ll << 30, 2);
  for (int r = 0; r < 16; ++r) EXPECT_TRUE(out[r].released & kDirect) << r;
}

// Per-rank RCCL_DDA_A2A_PUSH_BUFFER_SIZE: the agreed size is the minimum (each rank allocated at
// least that), the same on every rank.
TEST(DdaA2aLazyMicrotest, MismatchedPushSizesAgreeOnMinimum) {
  std::vector<RankPlan> plans(8);
  plans[2].pushBytes = 256ll << 20;
  plans[5].pushBytes = 512ll << 20;
  ExpectUniform(RunRanks(plans), kPush | kDirect, 256ll << 20, 2);
}

// A part disabled by env on one rank (RCCL_DDA_A2A_DIRECT=0 there) joins as a failure.
TEST(DdaA2aLazyMicrotest, PartNotWantedOnOneRankIsOffEverywhere) {
  std::vector<RankPlan> plans(8);
  plans[0].want = kPush;
  ExpectUniform(RunRanks(plans), kPush, 1ll << 30, 2);
  plans[0].want = 0;
  ExpectUniform(RunRanks(plans), 0, 0, 1);
}

// A zero push size (RCCL_DDA_A2A_PUSH_BUFFER_SIZE=0 on one rank) disables push only.
TEST(DdaA2aLazyMicrotest, ZeroPushSizeDisablesPush) {
  std::vector<RankPlan> plans(2);
  plans[1].pushBytes = 0;
  ExpectUniform(RunRanks(plans), kDirect, 0, 2);
}

// Failures on different ranks in both rounds at once.
TEST(DdaA2aLazyMicrotest, MixedFailuresStayUniform) {
  std::vector<RankPlan> plans(16);
  plans[3].setupFails = kDirect;
  plans[9].importFails = kPush;
  ExpectUniform(RunRanks(plans), 0, 0, 2);
}

// Implicit staging buffer (RCCL_DDA_A2A_IMPLICIT=1, opt-in): agreed only when every rank wants and
// creates it, and never without the Direct flag buffer it serves.
TEST(DdaA2aLazyMicrotest, ImplicitAgreedWhenEveryRankHasIt) {
  std::vector<RankPlan> plans(8);
  for (auto& p : plans) p.want = kPush | kDirect | kImplicit;
  ExpectUniform(RunRanks(plans), kPush | kDirect | kImplicit, 1ll << 30, 2);
}

TEST(DdaA2aLazyMicrotest, ImplicitNotWantedOnOneRankIsOffEverywhere) {
  std::vector<RankPlan> plans(8);
  for (auto& p : plans) p.want = kPush | kDirect | kImplicit;
  plans[5].want = kPush | kDirect;  // RCCL_DDA_A2A_IMPLICIT unset on one rank
  auto out = RunRanks(plans);
  ExpectUniform(out, kPush | kDirect, 1ll << 30, 2);
  for (int r = 0; r < 8; ++r) {
    if (r != 5) EXPECT_TRUE(out[r].released & kImplicit) << "rank " << r << " frees its unused staging buffer";
  }
}

TEST(DdaA2aLazyMicrotest, ImplicitDroppedWithDirect) {
  std::vector<RankPlan> plans(4);
  for (auto& p : plans) p.want = kPush | kDirect | kImplicit;
  plans[2].setupFails = kDirect;  // the staging buffer exists everywhere, the flag buffer does not
  auto out = RunRanks(plans);
  ExpectUniform(out, kPush, 1ll << 30, 2);
  for (int r = 0; r < 4; ++r) EXPECT_TRUE(out[r].released & kImplicit) << r;
  plans[2].setupFails = 0;
  plans[1].importFails = kDirect;  // decided in the second round
  out = RunRanks(plans);
  ExpectUniform(out, kPush, 1ll << 30, 2);
  for (int r = 0; r < 4; ++r) EXPECT_TRUE(out[r].released & kImplicit) << r;
}

TEST(DdaA2aLazyMicrotest, ImplicitFailureLeavesDirectAndPush) {
  std::vector<RankPlan> plans(16);
  for (auto& p : plans) p.want = kPush | kDirect | kImplicit;
  plans[7].setupFails = kImplicit;
  ExpectUniform(RunRanks(plans), kPush | kDirect, 1ll << 30, 2);
  plans[7].setupFails = 0;
  plans[12].importFails = kImplicit;
  ExpectUniform(RunRanks(plans), kPush | kDirect, 1ll << 30, 2);
}

// Push buffer sizes against free device memory: the configured size and its halves down to 64 MiB,
// each only if kPushMinFreeBytes (2 GiB) stays free for p2p after the allocation.
TEST(DdaA2aLazyMicrotest, PushSizesKeepP2pHeadroom) {
  using dda_a2a_lazy::pushSizes;
  const int64_t M = 1ll << 20;
  EXPECT_EQ((std::vector<int64_t>{kGiB, 512 * M, 256 * M, 128 * M, 64 * M}), pushSizes(kGiB, 8, kFlagBytes, 100 * kGiB));
  EXPECT_EQ((std::vector<int64_t>{kGiB, 512 * M, 256 * M, 128 * M, 64 * M}),
            pushSizes(kGiB, 8, kFlagBytes, kFlagBytes + kGiB + 2 * kGiB));
  EXPECT_EQ((std::vector<int64_t>{512 * M, 256 * M, 128 * M, 64 * M}),
            pushSizes(kGiB, 8, kFlagBytes, kFlagBytes + kGiB + 2 * kGiB - 1));
  EXPECT_EQ((std::vector<int64_t>{256 * M, 128 * M, 64 * M}), pushSizes(kGiB, 8, kFlagBytes, 2 * kGiB + 300 * M));
  // h13-13 s05_agr_fill_r5: 1 GiB is allocatable, but nothing leaves 2 GiB.
  EXPECT_TRUE(pushSizes(kGiB, 8, kFlagBytes, kGiB + 512 * M).empty());
  EXPECT_TRUE(pushSizes(kGiB, 8, kFlagBytes, 512 * M).empty());
  EXPECT_TRUE(pushSizes(kGiB, 8, kFlagBytes, 0).empty());
  // a configured size below 64 MiB is still tried, alone
  EXPECT_EQ((std::vector<int64_t>{32 * M}), pushSizes(32 * M, 8, kFlagBytes, 100 * kGiB));
  EXPECT_TRUE(pushSizes(0, 8, kFlagBytes, 100 * kGiB).empty());
}

// HW hang (h13-13, s05_agr_fill_r5_P5): rank 5 had ~1.5 GiB free at the first large out-of-place
// AllToAll. It could allocate the 1 GiB push buffer, so push was agreed on every rank; the in-place
// AllToAll that followed took p2p, whose connection buffers then failed to allocate on rank 5 only,
// and the other ranks waited for rank 5 in bootstrap until the job was killed. With the headroom
// rank 5 offers no buffer: push is off on every rank (the 2 MiB Direct flag buffer still comes up),
// and rank 5 keeps its memory for p2p.
TEST(DdaA2aLazyMicrotest, LowHeadroomOnOneRankDisablesPushEverywhere) {
  std::vector<RankPlan> plans(8);
  for (auto& p : plans) p.freeBytes = 100 * kGiB;
  plans[5].freeBytes = kGiB + (512ll << 20);
  auto out = RunRanks(plans);
  ExpectUniform(out, kDirect, 0, 2);
  for (int r = 0; r < 8; ++r) {
    EXPECT_EQ(plans[r].freeBytes, out[r].freeLeft) << "rank " << r << " keeps its free memory for p2p";
  }
}

// Short (but sufficient) headroom on one rank shrinks the buffer on every rank to what that rank
// can offer with 2 GiB left; every rank keeps at least the headroom.
TEST(DdaA2aLazyMicrotest, LowHeadroomOnOneRankShrinksPushUniformly) {
  std::vector<RankPlan> plans(16);
  for (auto& p : plans) p.freeBytes = 100 * kGiB;
  plans[9].freeBytes = 3 * kGiB;
  auto out = RunRanks(plans);
  ExpectUniform(out, kPush | kDirect, 512ll << 20, 2);
  for (int r = 0; r < 16; ++r) EXPECT_GE(out[r].freeLeft, dda_a2a_lazy::kPushMinFreeBytes) << r;
}

// ---------------------------------------------------------------------------
// Push host helpers (dda_alltoall.h)
// ---------------------------------------------------------------------------
namespace {

struct Topo {
  std::vector<uint64_t> host;
  std::vector<int64_t> bus;
  int sib(int r) const {
    return rcclDdaA2aTopoSibling(
        (int)host.size(), r, [this](int i) { return host[i]; }, [this](int i) { return bus[i]; });
  }
};

}  // namespace

// DPX tray, by-slot mapping (busIds as logged on the DPX tray: 0x104000 / 0x104001 per package):
// rank^1 pairs.
TEST(DdaA2aPushHostMicrotest, TopoSibling_DpxBySlot) {
  Topo t{{7, 7, 7, 7, 7, 7, 7, 7}, {0x104000, 0x104001, 0x204000, 0x204001, 0x304000, 0x304001, 0x404000, 0x404001}};
  for (int r = 0; r < 8; ++r) EXPECT_EQ(r ^ 1, t.sib(r)) << r;
}

// The same tray with packages interleaved over the ranks (e.g. --map-by that puts the two partitions
// of a package on ranks r and r+4): the sibling follows the topology, not rank^1.
TEST(DdaA2aPushHostMicrotest, TopoSibling_DpxInterleaved) {
  Topo t{{7, 7, 7, 7, 7, 7, 7, 7}, {0x104000, 0x204000, 0x304000, 0x404000, 0x104001, 0x204001, 0x304001, 0x404001}};
  for (int r = 0; r < 8; ++r) EXPECT_EQ((r + 4) % 8, t.sib(r)) << r;
}

// SPX tray (one logical GPU per package; busIds as logged on the SPX tray): no siblings.
TEST(DdaA2aPushHostMicrotest, TopoSibling_SpxNone) {
  Topo t{{7, 7, 7, 7}, {0x5000, 0x26000, 0x46000, 0x65000}};
  for (int r = 0; r < 4; ++r) EXPECT_EQ(-1, t.sib(r)) << r;
}

// Two trays with identical PCI layouts: the same busId on another host is not a sibling.
TEST(DdaA2aPushHostMicrotest, TopoSibling_TwoTraysSameBusIds) {
  Topo t{{1, 1, 2, 2}, {0x104000, 0x104001, 0x104000, 0x104001}};
  EXPECT_EQ(1, t.sib(0));
  EXPECT_EQ(0, t.sib(1));
  EXPECT_EQ(3, t.sib(2));
  EXPECT_EQ(2, t.sib(3));
  Topo spx2{{1, 2}, {0x5000, 0x5000}};
  EXPECT_EQ(-1, spx2.sib(0));
  EXPECT_EQ(-1, spx2.sib(1));
}

// Only one partition of a package in the comm, two ranks on one device, or three ranks in a group:
// no sibling, and the relation stays symmetric.
TEST(DdaA2aPushHostMicrotest, TopoSibling_AmbiguousOrUnpairedIsNone) {
  Topo half{{7, 7, 7}, {0x104000, 0x104001, 0x204000}};
  EXPECT_EQ(1, half.sib(0));
  EXPECT_EQ(0, half.sib(1));
  EXPECT_EQ(-1, half.sib(2));
  Topo sameDev{{7, 7, 7}, {0x104000, 0x104001, 0x104001}};
  for (int r = 0; r < 3; ++r) EXPECT_EQ(-1, sameDev.sib(r)) << r;
  Topo three{{7, 7, 7}, {0x104000, 0x104001, 0x104002}};
  for (int r = 0; r < 3; ++r) EXPECT_EQ(-1, three.sib(r)) << r;
  // symmetric on a random-ish layout
  Topo mix{{1, 1, 1, 2, 2, 1, 2}, {0x10, 0x11, 0x20, 0x10, 0x11, 0x30, 0x20}};
  for (int r = 0; r < 7; ++r) {
    const int s = mix.sib(r);
    if (s >= 0) EXPECT_EQ(r, mix.sib(s)) << r;
  }
}

// F8: a push call whose launch fails (any round) poisons the comm; a clean call does not.
TEST(DdaA2aPushHostMicrotest, PushRunPoisonsOnAnyFailure) {
  struct Run {
    std::vector<size_t> offs;
    int poisoned = 0;
  };
  auto run = [](size_t perPeer, size_t c, int failAt, Run* out) {
    return rcclDdaA2aPushRun(
        perPeer, c,
        [&](size_t off, size_t, bool) {
          out->offs.push_back(off);
          return (int)out->offs.size() - 1 == failAt ? ncclUnhandledCudaError : ncclSuccess;
        },
        [&] { ++out->poisoned; });
  };
  Run ok;
  EXPECT_EQ(ncclSuccess, run(4096 * 3, 4096, -1, &ok));
  EXPECT_EQ((std::vector<size_t>{0, 4096, 8192}), ok.offs);
  EXPECT_EQ(0, ok.poisoned);
  for (int failAt : {0, 1, 2}) {
    Run r;
    EXPECT_EQ(ncclUnhandledCudaError, run(4096 * 3, 4096, failAt, &r)) << failAt;
    EXPECT_EQ((size_t)failAt + 1, r.offs.size()) << "no round after the failed one";
    EXPECT_EQ(1, r.poisoned) << failAt;
  }
  Run none;  // no usable round size: nothing launched, still poisoned (peers may have launched)
  EXPECT_EQ(ncclInvalidUsage, run(4096, 0, -1, &none));
  EXPECT_TRUE(none.offs.empty());
  EXPECT_EQ(1, none.poisoned);
  // last flag only on the final round
  std::vector<bool> last;
  (void)rcclDdaA2aPushRun(
      10000, 4096,
      [&](size_t, size_t len, bool l) {
        last.push_back(l);
        EXPECT_LE(len, 4096u);
        return ncclSuccess;
      },
      [] {});
  EXPECT_EQ((std::vector<bool>{false, false, true}), last);
}

// The host mirror keeps the first failure; a launch poison maps to ncclUnhandledCudaError for the
// refusal of later AllToAlls and for ncclCommGetAsyncError.
TEST(DdaA2aPushHostMicrotest, PoisonHostWordFirstFailureWins) {
  uint32_t w = 0;
  rcclDdaA2aPoisonHostWord(&w, kDdaA2aErrLaunch);
  EXPECT_EQ(kDdaA2aErrLaunch, w);
  rcclDdaA2aPoisonHostWord(&w, 3);
  EXPECT_EQ(kDdaA2aErrLaunch, w);
  uint32_t tripped = 3;
  rcclDdaA2aPoisonHostWord(&tripped, kDdaA2aErrLaunch);
  EXPECT_EQ(3u, tripped);
  EXPECT_EQ(ncclUnhandledCudaError, rcclDdaA2aErrorResult(kDdaA2aErrLaunch));
  EXPECT_EQ(ncclUnhandledCudaError, rcclDdaA2aAsyncResult(kDdaA2aErrLaunch));
}
