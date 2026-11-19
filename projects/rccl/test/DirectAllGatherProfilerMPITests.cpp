/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Direct AllGather runs as one Send and one Recv task per peer, so it reports
// P2p events and no Coll event. A profiler can tie those tasks back to the
// AllGather only because taskAppend wraps them in the call's CollApi event. Each
// call must report exactly one AllGather CollApi event and no Coll event. Every
// Send/Recv P2pApi event it posts must start and stop inside that CollApi event,
// and every one of them must parent exactly one P2p task event, which needs the
// in-place send/recv-to-self pair to be skipped up front rather than dropped
// later by the scheduler.
//
// Direct AllGather is gfx942/gfx950 only and needs a rank count that is a
// multiple of 8; the suite skips on every rank when RCCL picks another backend.
// Launch with DDA and hierarchical AllGather off, since both preempt Direct.

#ifdef MPI_TESTS_ENABLED

#include <mpi.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ProfilerRecorderHelpers.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "comm.h"
#include "plugin/nccl_profiler.h"
#include "rccl_common.h"

namespace {

using RCCLTestHelpers::initializeBufferWithPattern;
using RCCLTestHelpers::verifyBufferData;

constexpr size_t kCounts[] = {7, 1024};
constexpr size_t kNumCounts = sizeof(kCounts) / sizeof(kCounts[0]);
constexpr size_t kNumPlacements = 2;  // out of place and in place

int32_t inputValue(int rank, size_t i) {
  return static_cast<int32_t>(1 + rank * 1000 + i % 997);
}

}  // namespace

class DirectAllGatherProfilerMPITest : public MPITestBase {
 protected:
  void SetUp() override {
    MPITestBase::SetUp();
    const std::string mask = std::to_string(ncclProfileGroup | ncclProfileColl | ncclProfileP2p |
                                            ncclProfileCollApi | ncclProfileP2pApi);
    maskGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("RCCL_TEST_KCH_EVENT_MASK", mask.c_str());
    recorder_.Load();
  }

  void TearDown() override {
    MPITestBase::TearDown();
    recorder_.Unload();
    maskGuard_.reset();
  }

  void createComm() {
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    comm_ = getActiveCommunicator();
    stream_ = getActiveStream();
    const std::string notLoaded = recorder_.NotLoadedOnAnyRank();
    ASSERT_TRUE(notLoaded.empty()) << notLoaded;
  }

  // Empty when the records since `from` show one Direct AllGather bracketing all of its tasks.
  std::string checkBracket(size_t from, size_t count, bool inPlace) {
    const std::vector<RcclKchRecord> recs = recorder_.Snapshot();
    const int nranks = MPIEnvironment::world_size;
    std::ostringstream why;
    int64_t collApi = -1;
    int nCollApi = 0;
    int nColl = 0;
    for (size_t i = from; i < recs.size(); ++i) {
      if (recs[i].type == ncclProfileColl) ++nColl;
      if (recs[i].type == ncclProfileCollApi) {
        ++nCollApi;
        if (strcmp(recs[i].func, "AllGather") == 0) collApi = static_cast<int64_t>(i);
      }
    }
    if (nColl != 0) why << " " << nColl << " Coll events, expected none";
    if (nCollApi != 1 || collApi < 0) {
      why << " " << nCollApi << " CollApi events, expected one AllGather";
      return why.str();
    }
    const RcclKchRecord& bracket = recs[collApi];
    if (bracket.count != count) why << " CollApi count=" << bracket.count << ", expected " << count;
    if (bracket.stopOrder == 0) why << " CollApi never stopped";

    std::set<int64_t> apis;
    for (size_t i = from; i < recs.size(); ++i) {
      const RcclKchRecord& r = recs[i];
      if (r.type != ncclProfileP2pApi) continue;
      const bool inside = r.order > bracket.order && r.stopOrder != 0 && r.stopOrder < bracket.stopOrder;
      if (!inside) {
        why << " P2pApi " << r.func << " outside the AllGather CollApi";
      } else if ((strcmp(r.func, "Send") != 0 && strcmp(r.func, "Recv") != 0) || r.count != count) {
        why << " P2pApi func=" << r.func << " count=" << r.count;
      } else {
        apis.insert(static_cast<int64_t>(i));
      }
    }
    // Out of place posts a Send and a Recv per peer. In place, the self pair is
    // skipped when P2P batching is off.
    const size_t full = 2 * static_cast<size_t>(nranks);
    if (apis.size() != full && !(inPlace && apis.size() == full - 2)) {
      why << " " << apis.size() << " P2pApi events, expected " << full << (inPlace ? " or " : "")
          << (inPlace ? std::to_string(full - 2) : "");
    }

    std::multiset<int64_t> parents;
    for (size_t i = from; i < recs.size(); ++i) {
      if (recs[i].type == ncclProfileP2p) parents.insert(recs[i].parentIndex);
    }
    for (int64_t api : apis) {
      if (parents.count(api) != 1) {
        why << " P2pApi " << recs[api].func << " parents " << parents.count(api) << " P2p events, expected 1";
      }
    }
    if (parents.size() != apis.size()) {
      why << " " << parents.size() << " P2p events for " << apis.size() << " P2pApi events";
    }
    return why.str();
  }

  // Returns false only when a step fails on some rank, so the caller can stop.
  bool allGather(size_t count, bool inPlace) {
    const int nranks = MPIEnvironment::world_size;
    const int rank = MPIEnvironment::world_rank;
    std::ostringstream what;
    what << "AllGather int32 count=" << count << (inPlace ? " in-place" : " out-of-place");

    RCCLTestGuards::DeviceBufferAutoGuard recvBuf;
    RCCLTestGuards::DeviceBufferAutoGuard sendBuf;
    const size_t blockBytes = count * sizeof(int32_t);
    bool failed = hipMalloc(recvBuf.ptr(), nranks * blockBytes) != hipSuccess;
    void* recv = recvBuf.get();
    void* send = nullptr;
    if (!failed && inPlace) {
      send = static_cast<char*>(recv) + rank * blockBytes;
    } else if (!failed) {
      failed = hipMalloc(sendBuf.ptr(), blockBytes) != hipSuccess;
      send = sendBuf.get();
    }
    if (!failed) {
      failed = hipMemset(recv, 0, nranks * blockBytes) != hipSuccess ||
               initializeBufferWithPattern<int32_t>(send, count, [&](size_t i) { return inputValue(rank, i); }) !=
                   hipSuccess;
    }
    int algo = -1, protocol = -1, maxChannels = -1;
    if (!failed) {
      failed = rcclGetCollImplInfo(comm_, ncclFuncAllGather, count, ncclInt32, ncclSum, send, recv,
                                   /*graphCapturing=*/0, &algo, &protocol, &maxChannels) != ncclSuccess;
    }
    if (MPIHelpers::anyRankTrue(failed)) {
      ADD_FAILURE() << what.str() << ": setup failed on at least one rank";
      return false;
    }
    notDirect_ = mpiCoordinatedSkipReason(algo != RCCL_DIRECT_ALLGATHER,
                                          "RCCL did not select Direct AllGather (needs gfx942/gfx950, a rank count "
                                          "that is a multiple of 8, and DDA and hierarchical AllGather off)");
    if (!notDirect_.empty()) return false;

    const size_t from = recorder_.Snapshot().size();
    const bool ok = ncclAllGather(send, recv, count, ncclInt32, comm_, stream_) == ncclSuccess &&
                    hipStreamSynchronize(stream_) == hipSuccess;
    if (MPIHelpers::anyRankTrue(!ok)) {
      ADD_FAILURE() << what.str() << ": AllGather failed on at least one rank";
      return false;
    }
    const std::string bracket = checkBracket(from, count, inPlace);
    if (!bracket.empty()) {
      ADD_FAILURE() << what.str() << ":" << bracket;
      ++localFailures_;
    }
    size_t idx = 0;
    int32_t exp = 0, act = 0;
    auto expected = [&](size_t j) { return inputValue(static_cast<int>(j / count), j % count); };
    if (!verifyBufferData<int32_t>(recv, nranks * count, expected, 0, 0.0, &idx, &exp, &act)) {
      ADD_FAILURE() << what.str() << ": element " << idx << " expected " << exp << " got " << act;
      ++localFailures_;
    }
    ++calls_;
    return true;
  }

  ProfilerRecorder recorder_;
  std::unique_ptr<MPIHelpers::MpiEnvGuard> maskGuard_;
  ncclComm_t comm_ = nullptr;
  hipStream_t stream_ = nullptr;
  std::string notDirect_;
  int localFailures_ = 0;
  size_t calls_ = 0;
};

// Every call is correct and reports its Send/Recv tasks inside one AllGather CollApi event.
TEST_F(DirectAllGatherProfilerMPITest, TasksNestInsideCollApi) {
  RECORDER_REQUIRE(recorder_);
  ASSERT_NO_FATAL_FAILURE(createComm());
  for (size_t count : kCounts) {
    for (bool inPlace : {false, true}) {
      if (!allGather(count, inPlace)) {
        if (!notDirect_.empty()) GTEST_SKIP() << notDirect_;
        return;  // allGather already recorded the failure
      }
    }
  }
  EXPECT_EQ(kNumCounts * kNumPlacements, calls_);
  EXPECT_FALSE(MPIHelpers::anyRankTrue(localFailures_ != 0))
      << localFailures_ << " check(s) failed on this rank; other ranks may report theirs";
}

#endif  // MPI_TESTS_ENABLED
