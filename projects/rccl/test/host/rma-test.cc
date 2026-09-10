/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/rma/rma.cc (AICOMRCCL-2307).
 *
 * rma.cc is pure dispatch -- which launcher runs, on which stream, in which
 * order, and how queued tasks split between the CE and proxy paths -- so none
 * of it needs a GPU. The unit is #include-d via RMA_CC_PATH to reach its
 * file-static helpers; its launchers are seams in fakes/rma_fakes.cc.
 *
 * Suites appear in the same order as the functions they cover in rma.cc.
 *************************************************************************/

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/rma_fakes.h"

#include "nccl.h"
#include "comm.h"
#include "rma/rma.h"

#include RMA_CC_PATH

namespace {

// Sentinel handles. Nothing dereferences them; being distinct is the point,
// since every launch assertion is about which stream a launcher was handed.
hipStream_t const kMainStream = reinterpret_cast<hipStream_t>(0x5111ull);
hipStream_t const kCeStream = reinterpret_cast<hipStream_t>(0xCE11ull);
hipEvent_t const kCeEvent = reinterpret_cast<hipEvent_t>(0xEEEEull);

// Records every launcher call in order with the stream it got. Order is
// load-bearing: per-launcher counters could not tell a correct interleaving on
// the mixed path from a scrambled one.
struct LaunchLog {
  enum Which { kProxyPut, kCePut, kProxyWait, kCeWait, kEventRecord, kStreamWait };

  struct Entry {
    Which which;
    hipStream_t stream;   // for kEventRecord/kStreamWait: the stream argument
    ncclComm* comm;       // launchers only
    ncclKernelPlan* plan; // launchers only
    unsigned flags;       // kStreamWait only
  };

  std::vector<Entry> entries;

  void RecordLaunch(Which which, ncclComm* comm, ncclKernelPlan* plan, hipStream_t stream) {
    entries.push_back({which, stream, comm, plan, 0});
  }
  void RecordFence(Which which, hipStream_t stream, unsigned flags) {
    entries.push_back({which, stream, nullptr, nullptr, flags});
  }

  std::vector<Which> Sequence() const {
    std::vector<Which> out;
    out.reserve(entries.size());
    for (const auto& e : entries) out.push_back(e.which);
    return out;
  }

  // Stream handed to the first (and normally only) call of `which`. Returns a
  // distinct sentinel when `which` never ran: nullptr is the null stream, a
  // legitimate value, so it cannot double as "not found".
  static hipStream_t NotCalled() { return reinterpret_cast<hipStream_t>(~std::uintptr_t(0)); }

  hipStream_t StreamOf(Which which) const {
    for (const auto& e : entries) {
      if (e.which == which) return e.stream;
    }
    return NotCalled();
  }

  int CountOf(Which which) const {
    int n = 0;
    for (const auto& e : entries) {
      if (e.which == which) ++n;
    }
    return n;
  }
};

// Minimal ncclComm/ncclKernelPlan for rma.cc. ncclComm is ~3.8 MB, so it is
// heap-allocated, never a stack fixture member.
class RmaTestBase : public ::testing::Test {
protected:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclKernelPlan> plan_;
  LaunchLog log_;

  void SetUp() override {
    // Reset on entry as well as exit: a test that dies mid-body never reaches
    // its TearDown, and these seams are process-wide.
    ResetRmaFakes();
    ResetHipFakes();

    comm_ = std::make_unique<ncclComm>();  // value-initialises the whole object
    comm_->rank = 0;

    comm_->rmaState.rmaCeState.ceStream = kCeStream;
    comm_->rmaState.rmaCeState.ceEvent = kCeEvent;

    plan_ = std::make_unique<ncclKernelPlan>();  // zeroed, so its queues are empty
  }

  void TearDown() override {
    ResetRmaFakes();
    ResetHipFakes();
  }
};

// Success hooks on every launcher and HIP ordering seam, recording into `log`.
// Bundled because each launch test needs the whole set live.
struct AllHooks {
  ScopedHook<ncclResult_t(ncclComm*, ncclKernelPlan*, hipStream_t)> proxyPut;
  ScopedHook<ncclResult_t(ncclComm*, ncclKernelPlan*, hipStream_t)> cePut;
  ScopedHook<ncclResult_t(ncclComm*, ncclKernelPlan*, hipStream_t)> proxyWait;
  ScopedHook<ncclResult_t(ncclComm*, ncclKernelPlan*, hipStream_t)> ceWait;
  ScopedHook<hipError_t(hipEvent_t, hipStream_t)> eventRecord;
  ScopedHook<hipError_t(hipStream_t, hipEvent_t, unsigned int)> streamWait;

  explicit AllHooks(LaunchLog& log)
    : proxyPut(g_rmaProxyPutLaunch,
               [&log](ncclComm* c, ncclKernelPlan* p, hipStream_t s) {
                 log.RecordLaunch(LaunchLog::kProxyPut, c, p, s);
                 return ncclSuccess;
               }),
      cePut(g_rmaCePutLaunch,
            [&log](ncclComm* c, ncclKernelPlan* p, hipStream_t s) {
              log.RecordLaunch(LaunchLog::kCePut, c, p, s);
              return ncclSuccess;
            }),
      proxyWait(g_rmaProxyWaitLaunch,
                [&log](ncclComm* c, ncclKernelPlan* p, hipStream_t s) {
                  log.RecordLaunch(LaunchLog::kProxyWait, c, p, s);
                  return ncclSuccess;
                }),
      ceWait(g_rmaCeWaitLaunch,
             [&log](ncclComm* c, ncclKernelPlan* p, hipStream_t s) {
               log.RecordLaunch(LaunchLog::kCeWait, c, p, s);
               return ncclSuccess;
             }),
      eventRecord(g_hipEventRecord,
                  [&log](hipEvent_t e, hipStream_t s) {
                    EXPECT_EQ(e, kCeEvent) << "fence recorded on an unexpected event";
                    log.RecordFence(LaunchLog::kEventRecord, s, 0);
                    return hipSuccess;
                  }),
      streamWait(g_hipStreamWaitEvent,
                 [&log](hipStream_t s, hipEvent_t e, unsigned int f) {
                   EXPECT_EQ(e, kCeEvent) << "fence waited on an unexpected event";
                   log.RecordFence(LaunchLog::kStreamWait, s, f);
                   return hipSuccess;
                 }) {}
};

// --- ncclRmaWaitSignal (rma.cc:24) -----------------------------------------
class RmaWaitSignalTest : public RmaTestBase {
protected:
  ncclRmaArgs args_{};

  void SetUp() override {
    RmaTestBase::SetUp();
    plan_->rmaArgs = &args_;
    args_.func = ncclFuncWaitSignal;
  }

  void SetCounts(int proxy, int ce) {
    args_.nRmaTasksProxy = proxy;
    args_.nRmaTasksCe = ce;
  }
};

// Both counters positive: proxy on the caller's stream, CE on ceStream, fenced either side.
TEST_F(RmaWaitSignalTest, BothProxyAndCe_FencesAndLaunchesOnSeparateStreams) {
  SetCounts(/*proxy=*/1, /*ce=*/1);
  AllHooks hooks(log_);

  ASSERT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclSuccess);

  ASSERT_EQ(log_.Sequence(),
            (std::vector<LaunchLog::Which>{LaunchLog::kEventRecord, LaunchLog::kStreamWait,
                                           LaunchLog::kProxyWait, LaunchLog::kCeWait,
                                           LaunchLog::kEventRecord, LaunchLog::kStreamWait}));
  EXPECT_EQ(log_.StreamOf(LaunchLog::kProxyWait), kMainStream);
  EXPECT_EQ(log_.StreamOf(LaunchLog::kCeWait), kCeStream);
  // First fence: record on the caller's stream, CE stream waits on it.
  EXPECT_EQ(log_.entries[0].stream, kMainStream);
  EXPECT_EQ(log_.entries[1].stream, kCeStream);
  // Second fence: record on the CE stream, caller's stream waits on it.
  EXPECT_EQ(log_.entries[4].stream, kCeStream);
  EXPECT_EQ(log_.entries[5].stream, kMainStream);
  // Both waits are plain ordering waits.
  EXPECT_EQ(log_.entries[1].flags, 0u);
  EXPECT_EQ(log_.entries[5].flags, 0u);
  // Stream aside, the launchers must be handed this comm and this plan.
  for (const auto& e : log_.entries) {
    if (e.which != LaunchLog::kProxyWait && e.which != LaunchLog::kCeWait) continue;
    EXPECT_EQ(e.comm, comm_.get());
    EXPECT_EQ(e.plan, plan_.get());
  }
}

// Opening cudaEventRecord fails, so CUDACHECKGOTO unwinds before either launcher.
TEST_F(RmaWaitSignalTest, BothProxyAndCe_FirstEventRecordFails_NoLaunches) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook record(g_hipEventRecord,
                    [](hipEvent_t, hipStream_t) { return hipErrorInvalidValue; });

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclUnhandledCudaError);
  EXPECT_EQ(record.calls, 1);
  EXPECT_TRUE(log_.entries.empty());
}

// Opening cudaStreamWaitEvent fails, after the record already succeeded.
TEST_F(RmaWaitSignalTest, BothProxyAndCe_FirstStreamWaitFails_NoLaunches) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook wait(g_hipStreamWaitEvent,
                  [](hipStream_t, hipEvent_t, unsigned int) { return hipErrorInvalidValue; });

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclUnhandledCudaError);
  EXPECT_EQ(wait.calls, 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kProxyWait), 0);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCeWait), 0);
}

// Proxy launcher fails; CE is never reached, since proxy is launched first.
TEST_F(RmaWaitSignalTest, BothProxyAndCe_ProxyLaunchFails_CeNeverLaunches) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyWaitLaunch,
                   [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclInternalError; });

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclInternalError);
  EXPECT_EQ(proxy.calls, 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCeWait), 0);
  // Opening fence ran; the closing pair is unreachable on this path too.
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kStreamWait), 1);
}

// DEFECT PINNED, not endorsed: a CE launch failure skips the closing join, so the
// caller's stream is left unordered against work rma_ce.cc may already have put on
// ceStream (rma_ce.cc:451-463 can fail after ncclCuStreamBatchMemOp submitted).
TEST_F(RmaWaitSignalTest, BothProxyAndCe_CeLaunchFails_SkipsClosingFence) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook ce(g_rmaCeWaitLaunch,
                [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclInternalError; });

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclInternalError);
  EXPECT_EQ(ce.calls, 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kProxyWait), 1);
  // Only the opening fence ran; the closing record/wait pair is unreachable.
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kStreamWait), 1);
}

// Closing cudaEventRecord fails while the opening one succeeds (needs per-call control).
TEST_F(RmaWaitSignalTest, BothProxyAndCe_ClosingEventRecordFails) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  int calls = 0;
  ScopedHook record(g_hipEventRecord, [&calls](hipEvent_t, hipStream_t) {
    return ++calls == 2 ? hipErrorInvalidValue : hipSuccess;
  });

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclUnhandledCudaError);
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCeWait), 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kStreamWait), 1);  // closing wait not reached
}

// Closing cudaStreamWaitEvent fails -- the last CUDACHECKGOTO in the function.
TEST_F(RmaWaitSignalTest, BothProxyAndCe_ClosingStreamWaitFails) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  int calls = 0;
  ScopedHook wait(g_hipStreamWaitEvent, [&calls](hipStream_t, hipEvent_t, unsigned int) {
    return ++calls == 2 ? hipErrorInvalidValue : hipSuccess;
  });

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclUnhandledCudaError);
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCeWait), 1);
  // Without this the test cannot tell a skipped closing record from a present
  // one: the injected hook fires on the closing wait either way.
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 2);
}

// ncclInProgress from proxy is non-fatal; CE still runs and CE's success overwrites ret.
TEST_F(RmaWaitSignalTest, BothProxyAndCe_ProxyLaunchInProgress_ContinuesAndReportsSuccess) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyWaitLaunch,
                   [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclInProgress; });

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclSuccess);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCeWait), 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 2);
  EXPECT_EQ(log_.CountOf(LaunchLog::kStreamWait), 2);
}

// ncclInProgress from the last launcher survives, as nothing reassigns ret after it.
TEST_F(RmaWaitSignalTest, BothProxyAndCe_CeLaunchInProgress_ReturnsInProgress) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook ce(g_rmaCeWaitLaunch,
                [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclInProgress; });

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclInProgress);
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 2);
}

// Proxy-only path forwards ncclInProgress to the caller.
TEST_F(RmaWaitSignalTest, ProxyOnly_LaunchInProgress_ReturnsInProgress) {
  SetCounts(2, 0);
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyWaitLaunch,
                   [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclInProgress; });

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclInProgress);
}

// CE-only path forwards ncclInProgress to the caller.
TEST_F(RmaWaitSignalTest, CeOnly_LaunchInProgress_ReturnsInProgress) {
  SetCounts(0, 3);
  AllHooks hooks(log_);
  ScopedHook ce(g_rmaCeWaitLaunch,
                [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclInProgress; });

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclInProgress);
}

// Proxy-only: no fencing at all, and the launcher gets the caller's stream.
TEST_F(RmaWaitSignalTest, ProxyOnly_LaunchesOnCallerStreamWithoutFencing) {
  SetCounts(/*proxy=*/2, /*ce=*/0);
  AllHooks hooks(log_);

  ASSERT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclSuccess);

  EXPECT_EQ(log_.Sequence(), (std::vector<LaunchLog::Which>{LaunchLog::kProxyWait}));
  EXPECT_EQ(log_.StreamOf(LaunchLog::kProxyWait), kMainStream);
}

// Proxy-only launcher failure propagates unchanged.
TEST_F(RmaWaitSignalTest, ProxyOnly_LaunchFails_Propagates) {
  SetCounts(2, 0);
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyWaitLaunch,
                   [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclSystemError; });

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclSystemError);
  EXPECT_EQ(proxy.calls, 1);
}

// CE-only: the launcher gets the caller's stream, not ceStream.
TEST_F(RmaWaitSignalTest, CeOnly_LaunchesOnCallerStreamNotCeStream) {
  SetCounts(/*proxy=*/0, /*ce=*/3);
  AllHooks hooks(log_);

  ASSERT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclSuccess);

  EXPECT_EQ(log_.Sequence(), (std::vector<LaunchLog::Which>{LaunchLog::kCeWait}));
  EXPECT_EQ(log_.StreamOf(LaunchLog::kCeWait), kMainStream);
}

// CE-only launcher failure propagates unchanged.
TEST_F(RmaWaitSignalTest, CeOnly_LaunchFails_Propagates) {
  SetCounts(0, 3);
  AllHooks hooks(log_);
  ScopedHook ce(g_rmaCeWaitLaunch,
                [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclSystemError; });

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclSystemError);
  EXPECT_EQ(ce.calls, 1);
}

// Neither counter positive: every arm falls through to a no-op success.
TEST_F(RmaWaitSignalTest, NoTasks_IsANoOp) {
  SetCounts(0, 0);
  AllHooks hooks(log_);

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclSuccess);
  EXPECT_TRUE(log_.entries.empty());
}

// --- ncclRmaPut (rma.cc:57) ------------------------------------------------
//
// Same four-arm shape as ncclRmaWaitSignal, covered separately rather than by a
// shared parameterised body: the two functions are duplicated source, so one
// test driving both would still pass if a copy called the other's launchers.

class RmaPutTest : public RmaTestBase {
protected:
  ncclRmaArgs args_{};

  void SetUp() override {
    RmaTestBase::SetUp();
    plan_->rmaArgs = &args_;
    args_.func = ncclFuncPutSignal;
  }

  void SetCounts(int proxy, int ce) {
    args_.nRmaTasksProxy = proxy;
    args_.nRmaTasksCe = ce;
  }
};

// Both counters positive: proxy on the caller's stream, CE on ceStream, fenced either side.
TEST_F(RmaPutTest, BothProxyAndCe_FencesAndLaunchesOnSeparateStreams) {
  SetCounts(1, 1);
  AllHooks hooks(log_);

  ASSERT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSuccess);

  ASSERT_EQ(log_.Sequence(),
            (std::vector<LaunchLog::Which>{LaunchLog::kEventRecord, LaunchLog::kStreamWait,
                                           LaunchLog::kProxyPut, LaunchLog::kCePut,
                                           LaunchLog::kEventRecord, LaunchLog::kStreamWait}));
  EXPECT_EQ(log_.StreamOf(LaunchLog::kProxyPut), kMainStream);
  EXPECT_EQ(log_.StreamOf(LaunchLog::kCePut), kCeStream);
  // First fence: record on the caller's stream, CE stream waits on it.
  EXPECT_EQ(log_.entries[0].stream, kMainStream);
  EXPECT_EQ(log_.entries[1].stream, kCeStream);
  // Second fence: record on the CE stream, caller's stream waits on it.
  EXPECT_EQ(log_.entries[4].stream, kCeStream);
  EXPECT_EQ(log_.entries[5].stream, kMainStream);
  // Both waits are plain ordering waits.
  EXPECT_EQ(log_.entries[1].flags, 0u);
  EXPECT_EQ(log_.entries[5].flags, 0u);
  // Stream aside, the launchers must be handed this comm and this plan.
  for (const auto& e : log_.entries) {
    if (e.which != LaunchLog::kProxyPut && e.which != LaunchLog::kCePut) continue;
    EXPECT_EQ(e.comm, comm_.get());
    EXPECT_EQ(e.plan, plan_.get());
  }
}

// Opening cudaEventRecord fails, so CUDACHECKGOTO unwinds before either launcher.
TEST_F(RmaPutTest, BothProxyAndCe_FirstEventRecordFails_NoLaunches) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook record(g_hipEventRecord,
                    [](hipEvent_t, hipStream_t) { return hipErrorInvalidValue; });

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclUnhandledCudaError);
  EXPECT_EQ(record.calls, 1);
  EXPECT_TRUE(log_.entries.empty());
}

// Opening cudaStreamWaitEvent fails, after the record already succeeded.
TEST_F(RmaPutTest, BothProxyAndCe_FirstStreamWaitFails_NoLaunches) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook wait(g_hipStreamWaitEvent,
                  [](hipStream_t, hipEvent_t, unsigned int) { return hipErrorInvalidValue; });

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclUnhandledCudaError);
  EXPECT_EQ(wait.calls, 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kProxyPut), 0);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCePut), 0);
}

// Proxy launcher fails; CE is never reached, since proxy is launched first.
TEST_F(RmaPutTest, BothProxyAndCe_ProxyLaunchFails_CeNeverLaunches) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyPutLaunch,
                   [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclInternalError; });

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclInternalError);
  EXPECT_EQ(proxy.calls, 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCePut), 0);
  // Opening fence ran; the closing pair is unreachable on this path too.
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kStreamWait), 1);
}

// DEFECT PINNED, not endorsed: as on the WaitSignal path, a CE launch failure skips
// the closing join, leaving the caller's stream unordered against work rma_ce.cc may
// already have submitted to ceStream.
TEST_F(RmaPutTest, BothProxyAndCe_CeLaunchFails_SkipsClosingFence) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook ce(g_rmaCePutLaunch,
                [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclInternalError; });

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclInternalError);
  EXPECT_EQ(ce.calls, 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kProxyPut), 1);
  // Only the opening fence ran; the closing record/wait pair is unreachable.
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kStreamWait), 1);
}

// Closing cudaEventRecord fails while the opening one succeeds (needs per-call control).
TEST_F(RmaPutTest, BothProxyAndCe_ClosingEventRecordFails) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  int calls = 0;
  ScopedHook record(g_hipEventRecord, [&calls](hipEvent_t, hipStream_t) {
    return ++calls == 2 ? hipErrorInvalidValue : hipSuccess;
  });

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclUnhandledCudaError);
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCePut), 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kStreamWait), 1);  // closing wait not reached
}

// Closing cudaStreamWaitEvent fails -- the last CUDACHECKGOTO in the function.
TEST_F(RmaPutTest, BothProxyAndCe_ClosingStreamWaitFails) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  int calls = 0;
  ScopedHook wait(g_hipStreamWaitEvent, [&calls](hipStream_t, hipEvent_t, unsigned int) {
    return ++calls == 2 ? hipErrorInvalidValue : hipSuccess;
  });

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclUnhandledCudaError);
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCePut), 1);
  // Without this the test cannot tell a skipped closing record from a present one.
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 2);
}

// ncclInProgress from proxy is non-fatal; CE still runs and CE's success overwrites ret.
TEST_F(RmaPutTest, BothProxyAndCe_ProxyLaunchInProgress_ContinuesAndReportsSuccess) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyPutLaunch,
                   [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclInProgress; });

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSuccess);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCePut), 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 2);
  EXPECT_EQ(log_.CountOf(LaunchLog::kStreamWait), 2);
}

// ncclInProgress from the last launcher survives, as nothing reassigns ret after it.
TEST_F(RmaPutTest, BothProxyAndCe_CeLaunchInProgress_ReturnsInProgress) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook ce(g_rmaCePutLaunch,
                [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclInProgress; });

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclInProgress);
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 2);
}

// Proxy-only: no fencing at all, and the launcher gets the caller's stream.
TEST_F(RmaPutTest, ProxyOnly_LaunchesOnCallerStreamWithoutFencing) {
  SetCounts(4, 0);
  AllHooks hooks(log_);

  ASSERT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSuccess);

  EXPECT_EQ(log_.Sequence(), (std::vector<LaunchLog::Which>{LaunchLog::kProxyPut}));
  EXPECT_EQ(log_.StreamOf(LaunchLog::kProxyPut), kMainStream);
}

// Proxy-only launcher failure propagates unchanged.
TEST_F(RmaPutTest, ProxyOnly_LaunchFails_Propagates) {
  SetCounts(4, 0);
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyPutLaunch,
                   [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclSystemError; });

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSystemError);
  EXPECT_EQ(proxy.calls, 1);
}

// Proxy-only path forwards ncclInProgress to the caller.
TEST_F(RmaPutTest, ProxyOnly_LaunchInProgress_ReturnsInProgress) {
  SetCounts(4, 0);
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyPutLaunch,
                   [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclInProgress; });

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclInProgress);
}

// CE-only: the launcher gets the caller's stream, not ceStream.
TEST_F(RmaPutTest, CeOnly_LaunchesOnCallerStreamNotCeStream) {
  SetCounts(0, 4);
  AllHooks hooks(log_);

  ASSERT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSuccess);

  EXPECT_EQ(log_.Sequence(), (std::vector<LaunchLog::Which>{LaunchLog::kCePut}));
  EXPECT_EQ(log_.StreamOf(LaunchLog::kCePut), kMainStream);
}

// CE-only launcher failure propagates unchanged.
TEST_F(RmaPutTest, CeOnly_LaunchFails_Propagates) {
  SetCounts(0, 4);
  AllHooks hooks(log_);
  ScopedHook ce(g_rmaCePutLaunch,
                [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclSystemError; });

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSystemError);
  EXPECT_EQ(ce.calls, 1);
}

// CE-only path forwards ncclInProgress to the caller.
TEST_F(RmaPutTest, CeOnly_LaunchInProgress_ReturnsInProgress) {
  SetCounts(0, 4);
  AllHooks hooks(log_);
  ScopedHook ce(g_rmaCePutLaunch,
                [](ncclComm*, ncclKernelPlan*, hipStream_t) { return ncclInProgress; });

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclInProgress);
}

// Neither counter positive: every arm falls through to a no-op success.
TEST_F(RmaPutTest, NoTasks_IsANoOp) {
  SetCounts(0, 0);
  AllHooks hooks(log_);

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSuccess);
  EXPECT_TRUE(log_.entries.empty());
}

}  // namespace
