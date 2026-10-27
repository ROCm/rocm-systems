/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// AllGather under forced PAT. Each call must be correct and must report
// algo "PAT" and proto "SIMPLE" through the profiler plugin.
//
// Launch with NCCL_PAT_ENABLE=1 NCCL_ALGO=PAT NCCL_PROTO=Simple and the paths
// that intercept PAT turned off. PAT needs one rank per node on more than two
// nodes; RCCL_TEST_PAT_HOST_PER_RANK=1 gives every rank its own host id so
// ranks sharing a node also qualify.

#ifdef MPI_TESTS_ENABLED

#include <mpi.h>
#include <strings.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "DeviceBufferHelpers.hpp"
#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ProfilerRecorderHelpers.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"
#include "comm.h"
#include "plugin/nccl_profiler.h"

namespace {

using RCCLTestHelpers::getNcclDataType;
using RCCLTestHelpers::getTypeName;
using RCCLTestHelpers::initializeBufferWithPattern;
using RCCLTestHelpers::verifyBufferData;

constexpr int kMinPatNodes = 3;
constexpr int kSentinelByte = 0x7b;   // fills the output before every call
constexpr size_t kNumTypes = 4;       // int32, float, half, bf16
constexpr size_t kNumPlacements = 2;  // out of place and in place

// Counts per rank: a single element, odd tails, and a 4 MiB block plus a tail.
template <typename T>
std::vector<size_t> patCounts() {
  return {1, 7, 1023, 65537, (4u << 20) / sizeof(T) + 3};
}
const size_t kNumCounts = patCounts<float>().size();

struct PatFunc {
  ncclFunc_t func;
  const char* name;
};
constexpr PatFunc kPatFuncs[] = {{ncclFuncReduceScatter, "ReduceScatter"},
                                 {ncclFuncAllGather, "AllGather"}};

// The suite checks forced PAT, so it needs NCCL_ALGO=PAT at launch.
bool patForcedAtLaunch() {
  const char* algo = getenv("NCCL_ALGO");
  return algo != nullptr && strcasecmp(algo, "PAT") == 0;
}

// Small integers keep every op exact in half and bf16 up to dozens of ranks.
double inputValue(int rank, size_t index, ncclRedOp_t op) {
  if (op == ncclProd) {
    return ((rank + index) % 3 == 0) ? 2.0 : 1.0;
  }
  return static_cast<double>((rank * 7 + index * 3) % 11) - 5.0;
}

template <typename T>
T fromDouble(double v) {
  if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(v);
  } else {
    return static_cast<T>(static_cast<float>(v));
  }
}

}  // namespace

class PatForcedMPITest : public MPITestBase {
 protected:
  void SetUp() override {
    MPITestBase::SetUp();
    // getHostHash reads NCCL_HOSTID once per process, so this id holds for later tests too.
    const char* perRank = getenv("RCCL_TEST_PAT_HOST_PER_RANK");
    if (perRank != nullptr && atoi(perRank) == 1) {
      const std::string hostId =
          "pat-" + std::to_string(MPIEnvironment::world_rank);
      hostGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>("NCCL_HOSTID",
                                                             hostId.c_str());
    }
    const std::string mask =
        std::to_string(ncclProfileGroup | ncclProfileColl | ncclProfileP2p);
    maskGuard_ = std::make_unique<MPIHelpers::MpiEnvGuard>(
        "RCCL_TEST_KCH_EVENT_MASK", mask.c_str());
    recorder_.Load();
    notForcedReason_ = mpiCoordinatedSkipReason(
        !patForcedAtLaunch(),
        "NCCL_ALGO=PAT is not set at launch; this suite checks forced PAT");
  }

  void TearDown() override {
    MPITestBase::TearDown();
    recorder_.Unload();
    maskGuard_.reset();
    hostGuard_.reset();
  }

  // Empty when every rank can run PAT for ReduceScatter and AllGather.
  std::string patIneligibleReason(const ncclComm* comm) {
    std::ostringstream why;
    if (comm->nNodes != comm->nRanks) {
      why << "PAT needs one rank per node, got nNodes=" << comm->nNodes
          << " nRanks=" << comm->nRanks
          << "; on one host set RCCL_TEST_PAT_HOST_PER_RANK=1";
    } else if (comm->nNodes < kMinPatNodes) {
      why << "PAT tuning needs more than 2 nodes, got nNodes=" << comm->nNodes;
    } else {
      for (const PatFunc& f : kPatFuncs) {
        if (comm->bandwidths[f.func][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] <= 0) {
          why << "tuning or eligibility disables PAT for " << f.name
              << " on this arch/nNodes; launch with the forced-PAT env and a "
                 "host-driven network";
          break;
        }
      }
    }
    return why.str();
  }

  // Under forced PAT a missing PAT path is a bug, so preconditions fail rather than skip.
  void createPatComm() {
    ASSERT_MPI_EQ(ncclSuccess, createTestCommunicator());
    comm_ = getActiveCommunicator();
    stream_ = getActiveStream();
    const std::string notLoaded = recorder_.NotLoadedOnAnyRank();
    ASSERT_TRUE(notLoaded.empty()) << notLoaded;
    const char* qps = getenv("RCCL_PAT_SHARED_QPS");
    const bool wantShared = qps == nullptr || atoi(qps) != 0;
    ASSERT_FALSE(MPIHelpers::anyRankTrue(comm_->patSharedQps != wantShared))
        << "comm->patSharedQps does not match RCCL_PAT_SHARED_QPS="
        << (qps ? qps : "unset (default 1)") << " on at least one rank";
    std::string local = patIneligibleReason(comm_);
    std::string why = mpiCoordinatedSkipReason(!local.empty(), local.c_str());
    ASSERT_TRUE(why.empty()) << why;
    TEST_INFO(
        "PAT/SIMPLE bandwidth: ReduceScatter %.2f AllGather %.2f, nNodes %d",
        comm_->bandwidths[ncclFuncReduceScatter][NCCL_ALGO_PAT]
                         [NCCL_PROTO_SIMPLE],
        comm_->bandwidths[ncclFuncAllGather][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE],
        comm_->nNodes);
  }

  // Empty when the records since `from` hold one Coll of `func` with the expected algo.
  std::string checkAlgo(size_t from, const char* func, bool expectPat) {
    std::vector<RcclKchRecord> recs = recorder_.Snapshot();
    int colls = 0;
    std::ostringstream why;
    for (size_t i = from; i < recs.size(); ++i) {
      const RcclKchRecord& r = recs[i];
      if (r.type != ncclProfileColl) {
        continue;
      }
      ++colls;
      bool isPat = strcmp(r.algo, "PAT") == 0;
      if (strcmp(r.func, func) != 0) {
        why << " func=" << r.func;
      } else if (expectPat && (!isPat || strcmp(r.proto, "SIMPLE") != 0)) {
        why << " algo=" << r.algo << " proto=" << r.proto;
      } else if (!expectPat && isPat) {
        why << " algo=PAT where PAT must not run";
      }
    }
    if (colls != 1) {
      why << " " << colls << " Coll events, expected 1";
    }
    return why.str();
  }

  // A failed local launch skips the wait, but its peers stay in the collective until the timeout.
  template <typename T, typename Launch, typename Expected>
  bool runAndCheck(const std::string& what, const char* func, bool expectPat,
                   Launch launch, const void* out, size_t outCount,
                   Expected expected, double tolerance) {
    size_t from = recorder_.Snapshot().size();
    bool ok =
        launch() == ncclSuccess && hipStreamSynchronize(stream_) == hipSuccess;
    if (MPIHelpers::anyRankTrue(!ok)) {
      ADD_FAILURE() << what << ": collective failed on at least one rank";
      return false;
    }
    std::string algo = checkAlgo(from, func, expectPat);
    if (!algo.empty()) {
      ADD_FAILURE() << what << ":" << algo;
      ++localFailures_;
    }
    size_t idx = 0;
    T exp{};
    T act{};
    if (!verifyBufferData<T>(out, outCount, expected, 0, tolerance, &idx, &exp,
                             &act)) {
      ADD_FAILURE() << what << ": element " << idx << " expected "
                    << static_cast<double>(static_cast<float>(exp)) << " got "
                    << static_cast<double>(static_cast<float>(act));
      ++localFailures_;
    }
    ++calls_;
    return true;
  }

  // `large` holds nranks blocks; `small` holds this rank's block, aliased into `large` in place.
  struct CallBuffers {
    RCCLTestGuards::DeviceBufferAutoGuard large;
    RCCLTestGuards::DeviceBufferAutoGuard small;
    void* send = nullptr;
    void* recv = nullptr;
  };

  // The sentinel goes in before the input, so an unwritten output element outside the input fails.
  template <typename T, typename Pattern>
  bool setUpBuffers(const std::string& what, size_t count, bool inPlace,
                    bool sendIsLarge, Pattern pattern, CallBuffers* bufs) {
    const int nranks = MPIEnvironment::world_size;
    const size_t blockBytes = count * sizeof(T);
    void* small = nullptr;
    bool failed =
        hipMalloc(bufs->large.ptr(), nranks * blockBytes) != hipSuccess;
    if (!failed && inPlace) {
      small = static_cast<char*>(bufs->large.get()) +
              MPIEnvironment::world_rank * blockBytes;
    } else if (!failed) {
      failed = hipMalloc(bufs->small.ptr(), blockBytes) != hipSuccess;
      small = bufs->small.get();
    }
    bufs->send = sendIsLarge ? bufs->large.get() : small;
    bufs->recv = sendIsLarge ? small : bufs->large.get();
    const size_t recvBytes = sendIsLarge ? blockBytes : nranks * blockBytes;
    const size_t sendCount = sendIsLarge ? nranks * count : count;
    if (!failed) {
      failed = hipMemsetAsync(bufs->recv, kSentinelByte, recvBytes, stream_) !=
                   hipSuccess ||
               hipStreamSynchronize(stream_) != hipSuccess;
    }
    if (!failed) {
      failed = initializeBufferWithPattern<T>(bufs->send, sendCount, pattern) !=
               hipSuccess;
    }
    if (MPIHelpers::anyRankTrue(failed)) {
      ADD_FAILURE() << what << ": buffer setup failed on at least one rank";
      return false;
    }
    return true;
  }

  template <typename T>
  bool allGather(size_t count, bool inPlace) {
    const int rank = MPIEnvironment::world_rank;
    std::ostringstream what;
    what << "AllGather " << getTypeName<T>() << " count=" << count
         << (inPlace ? " in-place" : " out-of-place");
    CallBuffers bufs;
    auto input = [&](size_t i) {
      return fromDouble<T>(inputValue(rank, i, ncclSum));
    };
    if (!setUpBuffers<T>(what.str(), count, inPlace, /*sendIsLarge=*/false,
                         input, &bufs)) {
      return false;
    }
    auto launch = [&]() {
      return ncclAllGather(bufs.send, bufs.recv, count, getNcclDataType<T>(),
                           comm_, stream_);
    };
    auto expected = [&](size_t j) {
      return fromDouble<T>(inputValue(j / count, j % count, ncclSum));
    };
    return runAndCheck<T>(what.str(), "AllGather", /*expectPat=*/true, launch,
                          bufs.recv, MPIEnvironment::world_size * count,
                          expected, 0.0);
  }

  template <typename T>
  bool allGatherMatrix() {
    for (size_t count : patCounts<T>()) {
      for (bool inPlace : {false, true}) {
        if (!allGather<T>(count, inPlace)) {
          return false;
        }
      }
    }
    return true;
  }

  // Every rank fails if any rank saw a wrong algo or wrong data.
  void expectNoFailuresOnAnyRank(size_t expectedCalls) {
    EXPECT_EQ(expectedCalls, calls_) << "not every collective ran";
    EXPECT_FALSE(MPIHelpers::anyRankTrue(localFailures_ != 0))
        << localFailures_ << " PAT check(s) failed on this rank; other ranks "
        << "may report theirs";
  }

  ProfilerRecorder recorder_;
  std::unique_ptr<MPIHelpers::MpiEnvGuard> hostGuard_;
  std::unique_ptr<MPIHelpers::MpiEnvGuard> maskGuard_;
  ncclComm_t comm_ = nullptr;
  hipStream_t stream_ = nullptr;
  std::string notForcedReason_;
  int localFailures_ = 0;
  size_t calls_ = 0;
};

// Opens every test: the recorder must be usable and the launch must force PAT.
#define PAT_FORCED_REQUIRE()            \
  do {                                  \
    RECORDER_REQUIRE(recorder_);        \
    if (!notForcedReason_.empty()) {    \
      GTEST_SKIP() << notForcedReason_; \
    }                                   \
  } while (0)

// Every count, type and placement takes PAT and gathers every rank's block.
TEST_F(PatForcedMPITest, AllGatherTakesPat) {
  PAT_FORCED_REQUIRE();
  ASSERT_NO_FATAL_FAILURE(createPatComm());
  bool ran = allGatherMatrix<int32_t>() && allGatherMatrix<float>() &&
             allGatherMatrix<__half>() && allGatherMatrix<hip_bfloat16>();
  ASSERT_TRUE(ran);
  expectNoFailuresOnAnyRank(kNumTypes * kNumCounts * kNumPlacements);
}

#endif  // MPI_TESTS_ENABLED
