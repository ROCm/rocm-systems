/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "fakes/dev_runtime_fakes.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/nccl_stubs.h"
#include "fakes/param_redirect.h"

#include CE_COLL_CC_PATH

namespace {

constexpr size_t kMiB = 1024 * 1024;
constexpr uintptr_t kSendBase = 0x100000000000;
constexpr uintptr_t kRecvBase = 0x200000000000;
constexpr uintptr_t kPeerWindowBase = 0x300000000000;
constexpr uintptr_t kPeerWindowStride = 0x10000000000;
constexpr unsigned long long kCaptureGraphId = 7;
constexpr int kBatchAsyncDriverVersion = ROCM_VER_7_12_0;

enum class LaunchPath { kSingleStream, kMultiStream, kBatchAsync, kLegacyNullStream, kGraphCapture };

std::string LaunchPathName(const ::testing::TestParamInfo<LaunchPath>& info) {
  switch (info.param) {
    case LaunchPath::kSingleStream:
      return "SingleStream";
    case LaunchPath::kMultiStream:
      return "MultiStream";
    case LaunchPath::kBatchAsync:
      return "BatchAsync";
    case LaunchPath::kLegacyNullStream:
      return "LegacyNullStream";
    case LaunchPath::kGraphCapture:
      return "GraphCapture";
  }
  return "Unknown";
}

using SizeMatrix = std::vector<std::vector<size_t>>;

// Row r sends totalBytes over peerCounts[r] consecutive destinations from itself; the first takes any remainder.
SizeMatrix EqualTotalsMatrix(const std::vector<int>& peerCounts, size_t totalBytes) {
  const int nRanks = static_cast<int>(peerCounts.size());
  SizeMatrix send(nRanks, std::vector<size_t>(nRanks, 0));
  for (int src = 0; src < nRanks; ++src) {
    for (int j = 0; j < peerCounts[src]; ++j) {
      send[src][(src + j) % nRanks] = totalBytes / peerCounts[src] + (j == 0 ? totalBytes % peerCounts[src] : 0);
    }
  }
  return send;
}

std::vector<int> CyclingPeerCounts(int nRanks, int minPeers, int span) {
  std::vector<int> counts(nRanks);
  for (int r = 0; r < nRanks; ++r) {
    counts[r] = minPeers + r % span;
  }
  return counts;
}

std::vector<size_t> GatherSizes(const SizeMatrix& send) {
  const int nRanks = static_cast<int>(send.size());
  std::vector<size_t> gathered(4 * static_cast<size_t>(nRanks) * nRanks);
  for (int rank = 0; rank < nRanks; ++rank) {
    std::vector<size_t> sendDispls(nRanks);
    std::vector<size_t> recvSizes(nRanks);
    std::vector<size_t> recvDispls(nRanks);
    size_t sendOff = 0;
    size_t recvOff = 0;
    for (int peer = 0; peer < nRanks; ++peer) {
      sendDispls[peer] = sendOff;
      sendOff += send[rank][peer];
      recvSizes[peer] = send[peer][rank];
      recvDispls[peer] = recvOff;
      recvOff += recvSizes[peer];
    }
    ncclAlltoAllvPackLocalSizes(ncclAlltoAllvRankMetaBlock(gathered.data(), rank, nRanks), nRanks,
                                send[rank].data(), sendDispls.data(), recvSizes.data(), recvDispls.data());
  }
  return gathered;
}

struct RankRun {
  uint32_t barriers = 0;
  size_t copies = 0;
  size_t bytes = 0;
  size_t batchCopies = 0;
  size_t copyStreamCopies = 0;
  // Barriers already issued when the copy to each destination slot was enqueued; 0 for a slot with nothing to send.
  std::vector<uint32_t> slotEpochs;
};

constexpr uintptr_t kCopyStreamBase = 0x6000;

class CeAlltoAllvSyncMicrotest : public ::testing::TestWithParam<LaunchPath> {
 protected:
  void SetUp() override {
    ResetAllFakes();
    g_hipStreamBatchMemOp = [](hipStream_t, unsigned int, hipStreamBatchMemOpParams*, unsigned int) {
      return hipSuccess;
    };
    g_hipMemcpyAsync = [this](void* dst, const void*, size_t bytes, hipMemcpyKind, hipStream_t stream) {
      RecordCopy(dst, bytes);
      const uintptr_t s = reinterpret_cast<uintptr_t>(stream);
      if (s >= kCopyStreamBase && s < kCopyStreamBase + RCCL_CE_NUM_COPY_STREAMS) {
        ++current_.copyStreamCopies;
      }
      return hipSuccess;
    };
#ifdef CE_BATCH_ASYNC_SUPPORTED
    g_hipMemcpyBatchAsync = [this](void** dsts, void**, size_t* sizes, size_t count, hipMemcpyAttributes*,
                                   size_t*, size_t, size_t*, hipStream_t) {
      for (size_t i = 0; i < count; ++i) {
        RecordCopy(dsts[i], sizes[i]);
      }
      current_.batchCopies += count;
      return hipSuccess;
    };
#endif
    g_hipEventRecord = [](hipEvent_t, hipStream_t) { return hipSuccess; };
    g_hipStreamWaitEvent = [](hipStream_t, hipEvent_t, unsigned int) { return hipSuccess; };
    g_devrGetLsaRankPtr = [](ncclComm* comm, ncclDevrWindow*, size_t offset, int lsaRank, void** outPtr) {
      if (lsaRank < 0 || lsaRank >= comm->devrState.lsaSize) {
        return ncclInvalidArgument;
      }
      *outPtr = reinterpret_cast<void*>(kPeerWindowBase + lsaRank * kPeerWindowStride + offset);
      return ncclSuccess;
    };
    const bool batchAsync = GetParam() == LaunchPath::kBatchAsync;
    g_loadParam = [batchAsync](const char* env, int64_t deftVal) -> int64_t {
      if (std::strcmp(env, "RCCL_CE_BATCH_ASYNC_ENABLE") == 0) {
        return batchAsync ? 1 : 0;
      }
      return deftVal;
    };
    if (batchAsync) {
      ncclCudaDriverVersionCache = kBatchAsyncDriverVersion;
    }
  }

  void TearDown() override { ResetAllFakes(); }

  static void ResetAllFakes() {
    ResetHipFakes();
    ResetDevRuntimeFakes();
    ResetNcclFakes();
    ResetNcclStubs();
    ResetEnvFakes();
  }

  void RecordCopy(void* dst, size_t bytes) {
    ++current_.copies;
    current_.bytes += bytes;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(dst);
    const int nRanks = activeComm_->nRanks;
    const int dstRank =
        addr >= kPeerWindowBase ? static_cast<int>((addr - kPeerWindowBase) / kPeerWindowStride) : activeComm_->rank;
    current_.slotEpochs[(dstRank - activeComm_->rank + nRanks) % nRanks] = activeComm_->ceColl.ceSeqNum;
  }

  // Runs ncclCeAlltoAllv once per simulated rank; every rank sees the same gathered size matrix, as in production.
  std::vector<RankRun> Run(const SizeMatrix& send, uint32_t freq) {
    const int nRanks = static_cast<int>(send.size());
    std::vector<size_t> gathered = GatherSizes(send);
    std::vector<int> lsaRankList(nRanks);
    for (int r = 0; r < nRanks; ++r) {
      lsaRankList[r] = r;
    }
    std::vector<uint32_t> syncFlags(2 * nRanks, 0);
    ncclDevrWindow syncWin = {};
    syncWin.userPtr = syncFlags.data();
    ncclDevrWindow recvWin = {};
    recvWin.userPtr = reinterpret_cast<void*>(kRecvBase);

    const LaunchPath path = GetParam();
    hipStream_t stream = path == LaunchPath::kLegacyNullStream ? nullptr : reinterpret_cast<hipStream_t>(0x5717);
    std::vector<RankRun> runs;
    for (int rank = 0; rank < nRanks; ++rank) {
      auto comm = std::make_unique<ncclComm>();
      comm->rank = rank;
      comm->nRanks = nRanks;
      comm->devrState.lsaSize = nRanks;
      comm->devrState.lsaSelf = rank;
      comm->devrState.lsaRankList = lsaRankList.data();
      comm->symkState.hasLsaMultimem = false;
      comm->planner.capturingGraph.graphId = path == LaunchPath::kGraphCapture ? kCaptureGraphId : ULLONG_MAX;
      comm->ceColl.intraBatchSyncFreq = freq;
      comm->ceColl.intraBatchSyncMsgThreshold = CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD;
      comm->ceColl.baseUCSymReadyPtr = reinterpret_cast<uint8_t*>(syncFlags.data());
      comm->ceColl.baseUCSymComplPtr = reinterpret_cast<uint8_t*>(syncFlags.data() + nRanks);
      comm->ceColl.ceSyncWin = &syncWin;
      if (path == LaunchPath::kMultiStream) {
        comm->ceColl.nCopyStreams = RCCL_CE_NUM_COPY_STREAMS;
        for (int s = 0; s < RCCL_CE_NUM_COPY_STREAMS; ++s) {
          comm->ceColl.copyStreams[s] = reinterpret_cast<hipStream_t>(kCopyStreamBase + s);
          comm->ceColl.copyEvents[s] = reinterpret_cast<hipEvent_t>(0x7000 + s);
        }
      }

      ncclCeCollArgs args = {};
      args.func = ncclFuncAlltoAllv;
      args.sendBuff = reinterpret_cast<uint8_t*>(kSendBase);
      args.recvBuff = reinterpret_cast<uint8_t*>(kRecvBase);
      args.recvWin = &recvWin;
      args.sizes = gathered.data();

      current_ = RankRun{};
      current_.slotEpochs.assign(nRanks, 0);
      activeComm_ = comm.get();
      EXPECT_EQ(ncclCeAlltoAllv(comm.get(), &args, stream), ncclSuccess) << "rank " << rank;
      activeComm_ = nullptr;
      current_.barriers = comm->ceColl.ceSeqNum;
      runs.push_back(current_);
    }
    return runs;
  }

  static std::vector<uint32_t> Barriers(const std::vector<RankRun>& runs) {
    std::vector<uint32_t> barriers;
    for (const RankRun& run : runs) {
      barriers.push_back(run.barriers);
    }
    return barriers;
  }

  // Each rank copies exactly its nonzero row, slot s after the ready barrier and s / roundSlots round barriers.
  static void ExpectRowsCopied(const SizeMatrix& send, const std::vector<RankRun>& runs, uint32_t roundSlots) {
    const int nRanks = static_cast<int>(send.size());
    for (int rank = 0; rank < nRanks; ++rank) {
      size_t copies = 0;
      size_t bytes = 0;
      std::vector<uint32_t> epochs(nRanks, 0);
      for (int slot = 0; slot < nRanks; ++slot) {
        const size_t chunk = send[rank][(rank + slot) % nRanks];
        if (chunk != 0) {
          ++copies;
          bytes += chunk;
          epochs[slot] = 1 + slot / roundSlots;
        }
      }
      EXPECT_EQ(runs[rank].copies, copies) << "rank " << rank;
      EXPECT_EQ(runs[rank].bytes, bytes) << "rank " << rank;
      EXPECT_EQ(runs[rank].slotEpochs, epochs) << "rank " << rank;
    }
  }

  // Guards the parameter itself: each arm must really reach the copy API it is named for.
  void ExpectLaunchPathTaken(const std::vector<RankRun>& runs) {
    for (size_t rank = 0; rank < runs.size(); ++rank) {
      const size_t expectedBatch = GetParam() == LaunchPath::kBatchAsync ? runs[rank].copies : 0;
      EXPECT_EQ(runs[rank].batchCopies, expectedBatch) << "rank " << rank;
      if (GetParam() != LaunchPath::kMultiStream) {
        EXPECT_EQ(runs[rank].copyStreamCopies, 0u) << "rank " << rank;
      } else if (runs[rank].barriers == 2 && runs[rank].copies > 1) {
        EXPECT_EQ(runs[rank].copyStreamCopies, runs[rank].copies) << "rank " << rank;
      }
    }
  }

  // Ready and complete barriers around the transfers, plus one per further round of freq destination slots.
  static uint32_t RoundSyncBarriers(int nRanks, uint32_t freq) { return 2 + (nRanks - 1) / freq; }

  // Graph capture rounds up to an even count, so back-to-back replays keep alternating the ready/complete flags.
  void ExpectRoundSyncBarriers(const std::vector<RankRun>& runs, int nRanks, uint32_t freq) {
    uint32_t expected = RoundSyncBarriers(nRanks, freq);
    if (GetParam() == LaunchPath::kGraphCapture && expected % 2 != 0) {
      ++expected;
    }
    EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(nRanks, expected));
  }

  RankRun current_;
  ncclComm* activeComm_ = nullptr;
};

TEST_P(CeAlltoAllvSyncMicrotest, Sparse16Ranks_EqualTotalsAt512MiB_EveryRankIssuesSameBarrierCount) {
  const int kNRanks = 16;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  const SizeMatrix send = EqualTotalsMatrix(CyclingPeerCounts(kNRanks, 6, 5), CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD);

  const std::vector<RankRun> runs = Run(send, kFreq);

  ExpectRoundSyncBarriers(runs, kNRanks, kFreq);
  ExpectRowsCopied(send, runs, kFreq);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, Sparse32Ranks_EveryRankOverFreqButUnevenRounds_EveryRankIssuesSameBarrierCount) {
  const int kNRanks = 32;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  const SizeMatrix send = EqualTotalsMatrix(CyclingPeerCounts(kNRanks, 12, 7), 12252240ull * 64);

  const std::vector<RankRun> runs = Run(send, kFreq);

  ExpectRoundSyncBarriers(runs, kNRanks, kFreq);
  ExpectRowsCopied(send, runs, kFreq);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, Sparse4RanksFreq2_EqualTotalsOver512MiB_EveryRankIssuesSameBarrierCount) {
  const int kNRanks = 4;
  const uint32_t kFreq = 2;
  const SizeMatrix send = EqualTotalsMatrix({2, 3, 4, 2}, 12 * 48 * kMiB);

  const std::vector<RankRun> runs = Run(send, kFreq);

  ExpectRoundSyncBarriers(runs, kNRanks, kFreq);
  ExpectRowsCopied(send, runs, kFreq);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, OneHeavyRank_OthersUnderThreshold_EveryRankJoinsTheRoundBarriers) {
  const int kNRanks = 16;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  for (int heavyRank : {0, kNRanks - 1}) {
    SCOPED_TRACE("heavy rank " + std::to_string(heavyRank));
    SizeMatrix send = EqualTotalsMatrix(std::vector<int>(kNRanks, 4), 4 * kMiB);
    for (int j = 0; j < 10; ++j) {
      send[heavyRank][(heavyRank + j) % kNRanks] = 64 * kMiB;
    }

    const std::vector<RankRun> runs = Run(send, kFreq);

    ExpectRoundSyncBarriers(runs, kNRanks, kFreq);
    ExpectRowsCopied(send, runs, kFreq);
    ExpectLaunchPathTaken(runs);
  }
}

TEST_P(CeAlltoAllvSyncMicrotest, Sparse16Ranks_TotalsUnder512MiB_OnlyReadyAndCompleteBarriers) {
  const int kNRanks = 16;
  const SizeMatrix send = EqualTotalsMatrix(CyclingPeerCounts(kNRanks, 6, 5), 2520 * 100 * 1024);

  const std::vector<RankRun> runs = Run(send, CE_COLL_INTRA_BATCH_SYNC_FREQ);

  EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(kNRanks, 2));
  ExpectRowsCopied(send, runs, kNRanks);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, ManyPeersEachUnder512MiB_NoRankCrossesBothGates_OnlyReadyAndCompleteBarriers) {
  const int kNRanks = 16;
  const SizeMatrix send = EqualTotalsMatrix(std::vector<int>(kNRanks, kNRanks), kNRanks * 31 * kMiB);

  const std::vector<RankRun> runs = Run(send, CE_COLL_INTRA_BATCH_SYNC_FREQ);

  EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(kNRanks, 2));
  ExpectRowsCopied(send, runs, kNRanks);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, Dense16Ranks_Over512MiB_EveryRankIssuesSameBarrierCount) {
  const int kNRanks = 16;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  const SizeMatrix send = EqualTotalsMatrix(std::vector<int>(kNRanks, kNRanks), kNRanks * 40 * kMiB);

  const std::vector<RankRun> runs = Run(send, kFreq);

  EXPECT_GT(runs[0].barriers, 2u);
  EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(kNRanks, runs[0].barriers));
  ExpectRowsCopied(send, runs, kFreq);
  ExpectLaunchPathTaken(runs);
}

TEST_P(CeAlltoAllvSyncMicrotest, SixteenRanks_EachSendsToExactlyFreqPeers_OnlyReadyAndCompleteBarriers) {
  const int kNRanks = 16;
  const uint32_t kFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  const SizeMatrix send = EqualTotalsMatrix(std::vector<int>(kNRanks, kFreq), 1024 * kMiB);

  const std::vector<RankRun> runs = Run(send, kFreq);

  EXPECT_EQ(Barriers(runs), std::vector<uint32_t>(kNRanks, 2));
  ExpectRowsCopied(send, runs, kNRanks);
  ExpectLaunchPathTaken(runs);
}

INSTANTIATE_TEST_SUITE_P(LaunchPaths, CeAlltoAllvSyncMicrotest,
                         ::testing::Values(LaunchPath::kSingleStream, LaunchPath::kMultiStream,
#ifdef CE_BATCH_ASYNC_SUPPORTED
                                           LaunchPath::kBatchAsync,
#endif
                                           LaunchPath::kLegacyNullStream, LaunchPath::kGraphCapture),
                         LaunchPathName);

}  // namespace
