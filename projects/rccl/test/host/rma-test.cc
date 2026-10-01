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
#include <deque>
#include <string>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/dev_runtime_micro_fakes.h"
#include "fakes/rma_fakes.h"

#include "nccl.h"
#include "comm.h"
#include "rma/rma.h"

// rma.cc now declares NCCL_PARAM(RMADisable); the generated body caches its
// value in a function-local static, so it must be redirected before the unit is
// included or the first read freezes it for the process.
#include "fakes/param_redirect.h"

// scheduleRmaTasksToPlan's only failure arms are its two ncclCalloc calls, and
// ncclCalloc is a macro rather than a symbol, so it cannot be faked at link
// time. Route it through a call counter, as dev-runtime-test.cc does.
// TRAP: the substitution is textual and TU-wide, so the index counts every
// ncclCalloc this file reaches, not only the unit under test's.
// TRAP: both statics are TU-local, so ResetRmaFakes() cannot see them; the
// fixture resets them itself.
#include "alloc.h"
static int g_rmaCallocCallIndex = 0;
static int g_rmaCallocFailAt = -1;  // -1 = never fail; otherwise a 0-based index
// Element counts, in call order. An undersized allocation is otherwise invisible:
// the split writes past the end into heap slack, and reading the array back finds
// the values it just wrote, so only the requested count catches it.
static std::vector<size_t> g_rmaCallocCounts;
// Element counts handed to ncclMemoryStackAlloc, in call order. The CE arrays
// come from memScoped rather than ncclCalloc, so without this they could be
// mis-sized in either direction and the read-back would still find its own
// writes.
static std::vector<size_t> g_rmaStackCounts;
// ...and which arena each came from. Sizes alone cannot tell memScoped from
// memPermanent, and the fixture constructs and destructs both the same way, so
// swapping them would leave every count identical.
static std::vector<const void*> g_rmaStackArenas;
// Pool cells come from a backing stack too, but ncclMemoryPoolAlloc reaches it
// through ncclMemoryStack::allocate (utils.h) rather than the macro above, so
// the arena it is handed needs its own record. The two split tasks outlive the
// per-plan scoped arena, so which one they come from is load-bearing.
static std::vector<const void*> g_rmaPoolArenas;
template <typename T>
static ncclResult_t RmaMicroCalloc(const char* file, int line, const char* fn, T** ptr,
                                   size_t nelem) {
  g_rmaCallocCounts.push_back(nelem);
  if (g_rmaCallocCallIndex++ == g_rmaCallocFailAt) return ncclSystemError;
  return ncclCallocDebug(ptr, nelem, file, line, fn, true);
}
#undef ncclCalloc
#define ncclCalloc(...) RmaMicroCalloc(__FILE__, __LINE__, __func__, __VA_ARGS__)

template <typename T>
static T* RmaMicroStackAlloc(struct ncclMemoryStack* me, size_t n = 1) {
  g_rmaStackCounts.push_back(n);
  g_rmaStackArenas.push_back(me);
  return ncclMemoryStackAlloc<T>(me, n);
}
#define ncclMemoryStackAlloc RmaMicroStackAlloc

template <typename T>
static T* RmaMicroPoolAlloc(struct ncclMemoryPool* me, struct ncclMemoryStack* backing) {
  g_rmaPoolArenas.push_back(backing);
  return ncclMemoryPoolAlloc<T>(me, backing);
}
#define ncclMemoryPoolAlloc RmaMicroPoolAlloc

// The split's "proxy side unused" arm frees the two arrays it just allocated.
// Dropping that free is a pure leak: no return value or queue state changes, so
// only a counter can see it. Same textual-substitution caveats as above.
static int g_rmaFreeCalls = 0;
static void RmaMicroFree(void* p) {
  ++g_rmaFreeCalls;
  free(p);
}
#define free(p) RmaMicroFree(p)

#include RMA_CC_PATH

// These overrides are only meaningful for the unit under test; past this point
// the tests' own allocations and cleanup should not land in the counters.
#undef free
#undef ncclCalloc
#undef ncclMemoryStackAlloc
#undef ncclMemoryPoolAlloc

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
  enum Which {
    kProxyPut,
    kCePut,
    kProxyWait,
    kCeWait,
    kEventRecord,
    kStreamWait,
  };

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
  ncclCudaStreamList streamNode_{};
  ncclRma_t rmaVtable_{};  // identity only; ensureRmaProxyReady just null-checks it
  std::unique_ptr<ScopedHook<ncclResult_t(ncclComm*)>> proxyConnect_;

  void SetUp() override {
    // Reset on entry as well as exit: a test that dies mid-body never reaches
    // its TearDown, and these seams are process-wide.
    ResetRmaFakes();
    ResetHipFakes();

    comm_ = std::make_unique<ncclComm>();  // value-initialises the whole object
    comm_->rank = 0;

    // A value-initialised stack has bumper == end == 0, so every allocate()
    // spills; Construct/Destruct is what makes those spills reclaimable.
    ncclMemoryStackConstruct(&comm_->memScoped);
    ncclMemoryStackConstruct(&comm_->memPermanent);
    ncclMemoryPoolConstruct(&comm_->memPool_ncclTaskRma);

    comm_->rmaState.rmaCeState.ceStream = kCeStream;
    comm_->rmaState.rmaCeState.ceEvent = kCeEvent;

    // ensureRmaProxyReady gates every proxy launch on these, so the launch
    // suites need a connected proxy to reach the dispatch they are about.
    comm_->rmaState.rmaProxyState.connected = true;
    comm_->rmaState.rmaProxyState.ncclRma = &rmaVtable_;
    // The seam fails loudly by default; these suites are about what happens
    // after a successful connect, so say so rather than inherit it.
    proxyConnect_ = std::make_unique<ScopedHook<ncclResult_t(ncclComm*)>>(
      g_devrRmaProxyConnectOnce, [](ncclComm*) { return ncclSuccess; });

    // ncclLaunchRma dereferences comm->planner.streams unconditionally.
    streamNode_.next = nullptr;
    streamNode_.stream = kMainStream;
    comm_->planner.streams = &streamNode_;

    plan_ = std::make_unique<ncclKernelPlan>();  // zeroed, so its queues are empty
  }

  void TearDown() override {
    comm_->planner.streams = nullptr;  // borrowed from streamNode_
    ncclMemoryStackDestruct(&comm_->memScoped);
    ncclMemoryStackDestruct(&comm_->memPermanent);
    g_rmaCallocCallIndex = 0;  // TU-local, so Reset*Fakes cannot reach them
    g_rmaCallocFailAt = -1;
    g_rmaCallocCounts.clear();
    g_rmaStackCounts.clear();
    g_rmaStackArenas.clear();
    g_rmaPoolArenas.clear();
    g_rmaFreeCalls = 0;
    proxyConnect_.reset();
    ResetRmaFakes();
    ResetDevRuntimeMicroFakes();
    ResetHipFakes();
  }
};

// Fail the nth call of a seam and succeed otherwise. Hand-rolling this counter
// per test is what MICROTEST_README asks to be a named factory instead.
template <typename... Args>
static std::function<hipError_t(Args...)> FailsOnCall(int nth) {
  auto calls = std::make_shared<int>(0);
  return [calls, nth](Args...) {
    return ++*calls == nth ? hipErrorInvalidValue : hipSuccess;
  };
}

// A launcher that always reports `result`. The 25 call sites below differ only
// in that value, which MICROTEST_README asks to be a named factory.
static std::function<ncclResult_t(ncclComm*, ncclKernelPlan*, hipStream_t)>
LaunchReturns(ncclResult_t result) {
  return [result](ncclComm*, ncclKernelPlan*, hipStream_t) { return result; };
}

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
                   LaunchReturns(ncclInternalError));

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
                LaunchReturns(ncclInternalError));

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
  ScopedHook record(g_hipEventRecord, FailsOnCall<hipEvent_t, hipStream_t>(2));

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclUnhandledCudaError);
  EXPECT_EQ(record.calls, 2);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCeWait), 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kStreamWait), 1);  // closing wait not reached
}

// Closing cudaStreamWaitEvent fails -- the last CUDACHECKGOTO in the function.
TEST_F(RmaWaitSignalTest, BothProxyAndCe_ClosingStreamWaitFails) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook wait(g_hipStreamWaitEvent, FailsOnCall<hipStream_t, hipEvent_t, unsigned int>(2));

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclUnhandledCudaError);
  EXPECT_EQ(wait.calls, 2);
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
                   LaunchReturns(ncclInProgress));

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
                LaunchReturns(ncclInProgress));

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclInProgress);
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 2);
}

// Proxy-only path forwards ncclInProgress to the caller.
TEST_F(RmaWaitSignalTest, ProxyOnly_LaunchInProgress_ReturnsInProgress) {
  SetCounts(2, 0);
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyWaitLaunch,
                   LaunchReturns(ncclInProgress));

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclInProgress);
}

// CE-only path forwards ncclInProgress to the caller.
TEST_F(RmaWaitSignalTest, CeOnly_LaunchInProgress_ReturnsInProgress) {
  SetCounts(0, 3);
  AllHooks hooks(log_);
  ScopedHook ce(g_rmaCeWaitLaunch,
                LaunchReturns(ncclInProgress));

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
                   LaunchReturns(ncclSystemError));

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
                LaunchReturns(ncclSystemError));

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
// Same four-arm shape as ncclRmaWaitSignal, and deliberately a separate suite.
// Not because a shared body would miss a cross-wired launcher -- AllHooks
// records all four and each test compares a whole log_.Sequence() against an
// explicit enum list, so a Put calling the Wait launcher fails the vector either
// way. The reason is that parameterising would take the function under test and
// its expected enums from the same tuple, so one wrong entry would be
// self-consistent and pass. The two functions are duplicated production source
// that can drift apart independently, and these expectations stay literal.

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
                   LaunchReturns(ncclInternalError));

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
                LaunchReturns(ncclInternalError));

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
  ScopedHook record(g_hipEventRecord, FailsOnCall<hipEvent_t, hipStream_t>(2));

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclUnhandledCudaError);
  EXPECT_EQ(record.calls, 2);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCePut), 1);
  EXPECT_EQ(log_.CountOf(LaunchLog::kStreamWait), 1);  // closing wait not reached
}

// Closing cudaStreamWaitEvent fails -- the last CUDACHECKGOTO in the function.
TEST_F(RmaPutTest, BothProxyAndCe_ClosingStreamWaitFails) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook wait(g_hipStreamWaitEvent, FailsOnCall<hipStream_t, hipEvent_t, unsigned int>(2));

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclUnhandledCudaError);
  EXPECT_EQ(wait.calls, 2);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCePut), 1);
  // Without this the test cannot tell a skipped closing record from a present one.
  EXPECT_EQ(log_.CountOf(LaunchLog::kEventRecord), 2);
}

// ncclInProgress from proxy is non-fatal; CE still runs and CE's success overwrites ret.
TEST_F(RmaPutTest, BothProxyAndCe_ProxyLaunchInProgress_ContinuesAndReportsSuccess) {
  SetCounts(1, 1);
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyPutLaunch,
                   LaunchReturns(ncclInProgress));

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
                LaunchReturns(ncclInProgress));

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
                   LaunchReturns(ncclSystemError));

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSystemError);
  EXPECT_EQ(proxy.calls, 1);
}

// Proxy-only path forwards ncclInProgress to the caller.
TEST_F(RmaPutTest, ProxyOnly_LaunchInProgress_ReturnsInProgress) {
  SetCounts(4, 0);
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyPutLaunch,
                   LaunchReturns(ncclInProgress));

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
                LaunchReturns(ncclSystemError));

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSystemError);
  EXPECT_EQ(ce.calls, 1);
}

// CE-only path forwards ncclInProgress to the caller.
TEST_F(RmaPutTest, CeOnly_LaunchInProgress_ReturnsInProgress) {
  SetCounts(0, 4);
  AllHooks hooks(log_);
  ScopedHook ce(g_rmaCePutLaunch,
                LaunchReturns(ncclInProgress));

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclInProgress);
}

// Neither counter positive: every arm falls through to a no-op success.
TEST_F(RmaPutTest, NoTasks_IsANoOp) {
  SetCounts(0, 0);
  AllHooks hooks(log_);

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSuccess);
  EXPECT_TRUE(log_.entries.empty());
}

// --- ncclLaunchRma (rma.cc:90) ---------------------------------------------
//
// The switch on plan->rmaArgs->func, and the only place the stream comes from
// comm->planner.streams rather than the caller. Each case is driven proxy-only
// so it makes exactly one launcher call.

class RmaLaunchTest : public RmaTestBase {
protected:
  ncclRmaArgs args_{};

  void SetUp() override {
    RmaTestBase::SetUp();
    plan_->rmaArgs = &args_;
    args_.nRmaTasksProxy = 1;
    args_.nRmaTasksCe = 0;
  }

  // These arms run proxy-only, where the callee never dereferences comm, so a
  // dispatcher that dropped it would otherwise go unnoticed.
  void ExpectForwardedOurCommAndPlan() {
    ASSERT_EQ(log_.entries.size(), 1u);
    EXPECT_EQ(log_.entries[0].comm, comm_.get());
    EXPECT_EQ(log_.entries[0].plan, plan_.get());
  }
};

// ncclFuncPutSignal routes to ncclRmaPut, on the stream from comm->planner.streams.
TEST_F(RmaLaunchTest, PutSignal_DispatchesToRmaPutOnPlannerStream) {
  args_.func = ncclFuncPutSignal;
  AllHooks hooks(log_);

  ASSERT_EQ(ncclLaunchRma(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(log_.Sequence(), (std::vector<LaunchLog::Which>{LaunchLog::kProxyPut}));
  EXPECT_EQ(log_.StreamOf(LaunchLog::kProxyPut), kMainStream);
  ExpectForwardedOurCommAndPlan();
}

// The stream really is read from the planner, not defaulted: a second node value
// must reach the launcher.
TEST_F(RmaLaunchTest, PutSignal_UsesWhicheverStreamThePlannerHolds) {
  ncclCudaStreamList other{};
  other.stream = kCeStream;  // any stream distinguishable from kMainStream
  comm_->planner.streams = &other;
  args_.func = ncclFuncPutSignal;
  AllHooks hooks(log_);

  ASSERT_EQ(ncclLaunchRma(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(log_.StreamOf(LaunchLog::kProxyPut), kCeStream);
}

// ncclFuncSignal shares the PutSignal arm: a bare signal still launches as a put.
TEST_F(RmaLaunchTest, Signal_DispatchesToRmaPut) {
  args_.func = ncclFuncSignal;
  AllHooks hooks(log_);

  ASSERT_EQ(ncclLaunchRma(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(log_.Sequence(), (std::vector<LaunchLog::Which>{LaunchLog::kProxyPut}));
  ExpectForwardedOurCommAndPlan();
}

// ncclFuncWaitSignal is the only arm reaching ncclRmaWaitSignal.
TEST_F(RmaLaunchTest, WaitSignal_DispatchesToRmaWaitSignal) {
  args_.func = ncclFuncWaitSignal;
  AllHooks hooks(log_);

  ASSERT_EQ(ncclLaunchRma(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(log_.Sequence(), (std::vector<LaunchLog::Which>{LaunchLog::kProxyWait}));
  ExpectForwardedOurCommAndPlan();
}

// The default arm rejects an unsupported func without launching anything.
TEST_F(RmaLaunchTest, UnsupportedFunc_ReturnsInvalidUsageWithoutLaunching) {
  args_.func = ncclFuncAllReduce;
  AllHooks hooks(log_);

  EXPECT_EQ(ncclLaunchRma(comm_.get(), plan_.get()), ncclInvalidUsage);
  EXPECT_TRUE(log_.entries.empty());
}

// A launcher failure surfaces through the switch unchanged.
TEST_F(RmaLaunchTest, DispatchFailure_Propagates) {
  args_.func = ncclFuncPutSignal;
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyPutLaunch,
                   LaunchReturns(ncclInternalError));

  EXPECT_EQ(ncclLaunchRma(comm_.get(), plan_.get()), ncclInternalError);
}

// NCCLCHECKGOTO's ncclInProgress arm: non-blocking status survives the switch.
TEST_F(RmaLaunchTest, PutDispatchInProgress_Propagates) {
  args_.func = ncclFuncPutSignal;
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyPutLaunch,
                   LaunchReturns(ncclInProgress));

  EXPECT_EQ(ncclLaunchRma(comm_.get(), plan_.get()), ncclInProgress);
}

// Same, through the WaitSignal arm.
TEST_F(RmaLaunchTest, WaitSignalDispatchInProgress_Propagates) {
  args_.func = ncclFuncWaitSignal;
  AllHooks hooks(log_);
  ScopedHook proxy(g_rmaProxyWaitLaunch,
                   LaunchReturns(ncclInProgress));

  EXPECT_EQ(ncclLaunchRma(comm_.get(), plan_.get()), ncclInProgress);
}

// --- scheduleRmaTasksToPlan (rma.cc:138) -----------------------------------
//
// Context selection, the WaitSignal LSA split, and the put/signal batching loop.
// Also the only path reaching isLsaAccessible, isRmaPutOrSignal and
// canBatchRmaTasks, which are file-static.

class RmaScheduleTest : public RmaTestBase {
protected:
  std::vector<int> lsaRanks_;
  std::vector<ncclIntruQueue<ncclTaskRma, &ncclTaskRma::next>> ctxQueues_;
  // deque: the task arrays must keep a stable address once handed to a task.
  std::deque<std::vector<int>> waitPeerStore_;
  std::deque<std::vector<int>> waitSignalStore_;
  std::deque<std::vector<int>> waitSignalIdxStore_;

  void SetUp() override {
    RmaTestBase::SetUp();
    SetNumRmaCtx(4);
    SetLsaRanks({0, 1, 2, 3});  // ranks 0-3 local, 4+ remote
  }

  void TearDown() override {
    // Idempotent, and runs even when a fatal assertion cut the test body short.
    ReleaseProxyWaitArrays(plan_.get());
    comm_->planner.rmaTaskQueues = nullptr;  // borrowed from ctxQueues_
    comm_->devrState.lsaRankList = nullptr;  // borrowed from lsaRanks_
    RmaTestBase::TearDown();
  }

  // isLsaAccessible scans this list, so membership is what routes a task to CE.
  void SetLsaRanks(std::vector<int> ranks) {
    lsaRanks_ = std::move(ranks);
    comm_->devrState.lsaSize = static_cast<int>(lsaRanks_.size());
    comm_->devrState.lsaRankList = lsaRanks_.empty() ? nullptr : lsaRanks_.data();
  }

  // Size the per-context queues the way init.cc does.
  void SetNumRmaCtx(int n) {
    comm_->config.numRmaCtx = n;
    // Discards whatever was queued, so the planner's count goes with it --
    // otherwise a second call mid-test leaves nTasksRma counting dropped tasks.
    comm_->planner.nTasksRma = 0;
    ctxQueues_.clear();
    ctxQueues_.resize(n > 0 ? n : 0);
    for (auto& q : ctxQueues_) ncclIntruQueueConstruct(&q);
    comm_->planner.rmaTaskQueues = ctxQueues_.empty() ? nullptr : ctxQueues_.data();
  }

  // Allocate as enqueue.cc does, so the pool free on the WaitSignal path is coherent.
  ncclTaskRma* NewTask(ncclFunc_t func, int ctx) {
    auto* t = ncclMemoryPoolAlloc<ncclTaskRma>(&comm_->memPool_ncclTaskRma, &comm_->memPermanent);
    t->func = func;
    t->ctx = ctx;
    return t;
  }

  ncclTaskRma* EnqueuePut(int ctx, int peer, ncclFunc_t func = ncclFuncPutSignal) {
    ncclTaskRma* t = NewTask(func, ctx);
    t->peer = peer;
    ncclIntruQueueEnqueue(&ctxQueues_[ctx], t);
    comm_->planner.nTasksRma++;
    return t;
  }

  // Signal counts are 10, 11, ... so a mis-paired peer/nsignal split is visible.
  ncclTaskRma* EnqueueWait(int ctx, std::vector<int> peers) {
    ncclTaskRma* t = NewTask(ncclFuncWaitSignal, ctx);
    // Non-default, so a split that zero-fills the field instead of copying it is
    // distinguishable from one that copies correctly.
    t->signalMode = NCCL_SIGNAL;
    t->npeers = static_cast<int>(peers.size());
    waitPeerStore_.push_back(peers);
    waitSignalStore_.emplace_back();
    waitSignalIdxStore_.emplace_back();
    for (size_t i = 0; i < peers.size(); i++) {
      waitSignalStore_.back().push_back(static_cast<int>(10 + i));
      // A third distinct series, so a split that pairs a peer with another
      // peer's signal index is visible rather than coincidentally right.
      waitSignalIdxStore_.back().push_back(static_cast<int>(100 + i));
    }
    t->peers = waitPeerStore_.back().empty() ? nullptr : waitPeerStore_.back().data();
    t->nsignals = waitSignalStore_.back().empty() ? nullptr : waitSignalStore_.back().data();
    t->signalIdxs =
      waitSignalIdxStore_.back().empty() ? nullptr : waitSignalIdxStore_.back().data();
    ncclIntruQueueEnqueue(&ctxQueues_[ctx], t);
    comm_->planner.nTasksRma++;
    return t;
  }

  static std::vector<int> PeersOf(ncclIntruQueue<ncclTaskRma, &ncclTaskRma::next>* q) {
    std::vector<int> out;
    for (ncclTaskRma* t = q->head; t != nullptr; t = t->next) out.push_back(t->peer);
    return out;
  }

  static int QueueLength(ncclIntruQueue<ncclTaskRma, &ncclTaskRma::next>* q) {
    int n = 0;
    for (ncclTaskRma* t = q->head; t != nullptr; t = t->next) ++n;
    return n;
  }

  // The split hands the proxy task ncclCalloc'd arrays and transfers ownership:
  // ncclRmaProxyWaitLaunch frees task->peers/task->nsignals
  // (rma_proxy_launch.cc:719-720), as does descriptor teardown (:269-270). That
  // consumer is faked out here, so the test stands in for it.
  static void ReleaseProxyWaitArrays(ncclKernelPlan* plan) {
    for (ncclTaskRma* t = plan->rmaTaskQueueProxy.head; t != nullptr; t = t->next) {
      if (t->func != ncclFuncWaitSignal) continue;
      // Stands in for the downstream consumer, not for anything
      // scheduleRmaTasksToPlan does. Uncounted: the macro ends at the unit.
      free(t->peers);
      free(t->nsignals);
      free(t->signalIdxs);
      t->peers = nullptr;
      t->nsignals = nullptr;
      t->signalIdxs = nullptr;
    }
  }
};

// numRmaCtx == 0: the scan never runs, and the plan is left unmarked.
TEST_F(RmaScheduleTest, NoRmaContexts_ReturnsSuccessAndLeavesPlanUntouched) {
  SetNumRmaCtx(0);

  EXPECT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);
  EXPECT_FALSE(plan_->isRma);
  EXPECT_EQ(plan_->rmaArgs, nullptr);
}

// Contexts exist but every queue is empty -- same early exit.
TEST_F(RmaScheduleTest, AllContextQueuesEmpty_ReturnsSuccessAndLeavesPlanUntouched) {
  EXPECT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);
  EXPECT_FALSE(plan_->isRma);
  EXPECT_EQ(plan_->rmaArgs, nullptr);
}

// The scan stops at the first non-empty queue; seeding two proves it breaks.
TEST_F(RmaScheduleTest, PicksLowestNonEmptyContext) {
  EnqueuePut(/*ctx=*/2, /*peer=*/0);
  EnqueuePut(/*ctx=*/3, /*peer=*/0);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_TRUE(plan_->isRma);
  // The scan picks ctx 2, but the put/signal arm then drains every context into
  // this one plan, so ctx 3's task comes along and nothing is left queued.
  EXPECT_EQ(PeersOf(&plan_->rmaTaskQueueCe), (std::vector<int>{0, 0}));
  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 2);
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctxQueues_[3]));
  EXPECT_EQ(comm_->planner.nTasksRma, 0);
}

// The scan runs to the last context, not one short of it. Every other test seeds
// a lower context, where an off-by-one upper bound still finds the task.
TEST_F(RmaScheduleTest, PicksTaskOnTheLastContext) {
  EnqueuePut(/*ctx=*/3, /*peer=*/0);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  // Fatal: a scan that stopped short leaves ctx == -1 and rmaArgs null, and the
  // dereference below would crash rather than report.
  ASSERT_TRUE(plan_->isRma);
  ASSERT_NE(plan_->rmaArgs, nullptr);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 1);
  EXPECT_EQ(comm_->planner.nTasksRma, 0);
}

// Every peer LSA-accessible: npeersProxy == 0, so the proxy arrays are freed.
TEST_F(RmaScheduleTest, WaitSignal_AllPeersLsa_ProducesOnlyCeTask) {
  ncclTaskRma* orig = EnqueueWait(/*ctx=*/0, {1, 2, 3});

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->func, ncclFuncWaitSignal);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 1);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksCe, 1);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksProxy, 0);
  ASSERT_TRUE(ncclIntruQueueEmpty(&plan_->rmaTaskQueueProxy));

  ncclTaskRma* ce = plan_->rmaTaskQueueCe.head;
  ASSERT_NE(ce, nullptr);
  EXPECT_EQ(ce->func, ncclFuncWaitSignal);
  EXPECT_EQ(ce->ctx, 0);
  EXPECT_EQ(ce->signalMode, NCCL_SIGNAL);
  // The rmaArgs cell, then both CE arrays sized for the whole peer list.
  EXPECT_EQ(g_rmaStackCounts, (std::vector<size_t>{1, 3, 3, 3}));
  // All three come from the scoped arena, which is released per plan; taking
  // them from memPermanent instead would outlive the plan and leave the counts
  // unchanged.
  EXPECT_EQ(g_rmaStackArenas, (std::vector<const void*>{&comm_->memScoped,
                                                       &comm_->memScoped,
                                                       &comm_->memScoped,
                                                       &comm_->memScoped}));
  ASSERT_EQ(ce->npeers, 3);
  EXPECT_EQ(std::vector<int>(ce->peers, ce->peers + 3), (std::vector<int>{1, 2, 3}));
  EXPECT_EQ(std::vector<int>(ce->nsignals, ce->nsignals + 3), (std::vector<int>{10, 11, 12}));
  EXPECT_EQ(std::vector<int>(ce->signalIdxs, ce->signalIdxs + 3),
            (std::vector<int>{100, 101, 102}));
  EXPECT_EQ(comm_->planner.nTasksRma, 0);
  // All three proxy arrays were allocated and, with no proxy peers, must be
  // released here -- nothing downstream will ever see them.
  EXPECT_EQ(g_rmaFreeCalls, 3);
  // The split consumes the original task, which must return to the pool.
  EXPECT_EQ(reinterpret_cast<void*>(comm_->memPool_ncclTaskRma.head),
            reinterpret_cast<void*>(orig));
}

// No peer LSA-accessible: npeersCe == 0, so the CE arm is skipped.
TEST_F(RmaScheduleTest, WaitSignal_NoPeersLsa_ProducesOnlyProxyTask) {
  EnqueueWait(/*ctx=*/0, {7, 8});

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 1);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksCe, 0);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksProxy, 1);
  EXPECT_TRUE(ncclIntruQueueEmpty(&plan_->rmaTaskQueueCe));

  ncclTaskRma* proxy = plan_->rmaTaskQueueProxy.head;
  ASSERT_NE(proxy, nullptr);
  EXPECT_EQ(proxy->func, ncclFuncWaitSignal);
  EXPECT_EQ(proxy->ctx, 0);
  EXPECT_EQ(proxy->signalMode, NCCL_SIGNAL);
  ASSERT_EQ(proxy->npeers, 2);
  EXPECT_EQ(std::vector<int>(proxy->peers, proxy->peers + 2), (std::vector<int>{7, 8}));
  EXPECT_EQ(std::vector<int>(proxy->nsignals, proxy->nsignals + 2), (std::vector<int>{10, 11}));
  EXPECT_EQ(std::vector<int>(proxy->signalIdxs, proxy->signalIdxs + 2),
            (std::vector<int>{100, 101}));
  // Both arrays must be sized for every peer on the original task.
  EXPECT_EQ(g_rmaCallocCounts, (std::vector<size_t>{2, 2, 2}));
  // Ownership transfers to the task here, so nothing is released: a free() added
  // to this arm would still read {7, 8} back out of unscrubbed heap and pass.
  EXPECT_EQ(g_rmaFreeCalls, 0);
}

// Both arms taken. The interleaved order catches a split that lets the peer and
// nsignal indices drift apart.
TEST_F(RmaScheduleTest, WaitSignal_MixedPeers_SplitsPreservingSignalPairing) {
  EnqueueWait(/*ctx=*/1, {9, 2, 8, 0});  // remote, local, remote, local

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 2);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksCe, 1);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksProxy, 1);

  ncclTaskRma* ce = plan_->rmaTaskQueueCe.head;
  ASSERT_NE(ce, nullptr);
  EXPECT_EQ(ce->func, ncclFuncWaitSignal);
  EXPECT_EQ(ce->ctx, 1);  // the context the task was enqueued on
  EXPECT_EQ(ce->signalMode, NCCL_SIGNAL);
  ASSERT_EQ(ce->npeers, 2);
  EXPECT_EQ(std::vector<int>(ce->peers, ce->peers + 2), (std::vector<int>{2, 0}));
  EXPECT_EQ(std::vector<int>(ce->nsignals, ce->nsignals + 2), (std::vector<int>{11, 13}));
  EXPECT_EQ(std::vector<int>(ce->signalIdxs, ce->signalIdxs + 2),
            (std::vector<int>{101, 103}));

  ncclTaskRma* proxy = plan_->rmaTaskQueueProxy.head;
  ASSERT_NE(proxy, nullptr);
  EXPECT_EQ(proxy->func, ncclFuncWaitSignal);
  EXPECT_EQ(proxy->ctx, 1);
  EXPECT_EQ(proxy->signalMode, NCCL_SIGNAL);
  ASSERT_EQ(proxy->npeers, 2);
  EXPECT_EQ(std::vector<int>(proxy->peers, proxy->peers + 2), (std::vector<int>{9, 8}));
  EXPECT_EQ(std::vector<int>(proxy->nsignals, proxy->nsignals + 2), (std::vector<int>{10, 12}));
  EXPECT_EQ(std::vector<int>(proxy->signalIdxs, proxy->signalIdxs + 2),
            (std::vector<int>{100, 102}));

  // Sized for the whole peer list, not just each side's share of it. This is the
  // case that discriminates: the all-LSA test has npeersCe == npeers, so a
  // mis-size to npeersCe would pass there and fail here.
  EXPECT_EQ(g_rmaCallocCounts, (std::vector<size_t>{4, 4, 4}));
  EXPECT_EQ(g_rmaStackCounts, (std::vector<size_t>{1, 4, 4, 4}));
  EXPECT_EQ(g_rmaFreeCalls, 0);  // both sides in use, so neither array is released
  // Both synthesised tasks are enqueued onto the plan and consumed by the
  // launchers later, so they must come from the permanent arena -- the scoped
  // one is released per plan.
  EXPECT_EQ(g_rmaPoolArenas, (std::vector<const void*>{&comm_->memPermanent,
                                                      &comm_->memPermanent}));
  // The split consumes one queued task even though it emits two.
  EXPECT_EQ(comm_->planner.nTasksRma, 0);

}

// npeers == 0: both arms skipped, so the plan is RMA but carries no work.
TEST_F(RmaScheduleTest, WaitSignal_ZeroPeers_ProducesNoTasks) {
  EnqueueWait(/*ctx=*/0, {});

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_TRUE(plan_->isRma);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 0);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksCe, 0);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksProxy, 0);
  EXPECT_TRUE(ncclIntruQueueEmpty(&plan_->rmaTaskQueueCe));
  EXPECT_TRUE(ncclIntruQueueEmpty(&plan_->rmaTaskQueueProxy));
  EXPECT_EQ(comm_->planner.nTasksRma, 0);
}

// lsaSize == 0: the scan body never runs, so every peer is remote.
TEST_F(RmaScheduleTest, WaitSignal_EmptyLsaTeam_RoutesEveryPeerToProxy) {
  SetLsaRanks({});
  EnqueueWait(/*ctx=*/0, {0, 1});

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasksCe, 0);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksProxy, 1);

}

// A match on the last list entry: a scan stopping one short would misroute it.
TEST_F(RmaScheduleTest, WaitSignal_PeerAtEndOfLsaList_RoutesToCe) {
  SetLsaRanks({5, 6, 7});
  EnqueueWait(/*ctx=*/0, {7});

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasksCe, 1);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksProxy, 0);
}

// First ncclCalloc fails: the fail label runs with both pointers still null.
TEST_F(RmaScheduleTest, WaitSignal_FirstCallocFails_ReturnsError) {
  EnqueueWait(/*ctx=*/0, {7});
  g_rmaCallocFailAt = g_rmaCallocCallIndex;  // fail the next ncclCalloc

  EXPECT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSystemError);
  EXPECT_TRUE(ncclIntruQueueEmpty(&plan_->rmaTaskQueueCe));
  EXPECT_TRUE(ncclIntruQueueEmpty(&plan_->rmaTaskQueueProxy));
  // The fail label frees all three pointers even though none was allocated.
  EXPECT_EQ(g_rmaFreeCalls, 3);
  // The fail label leaves the plan half-built: the task was dequeued at
  // rma.cc:159 and is neither re-queued nor returned to the pool, the decrement
  // at rma.cc:236 is unreachable, and isRma was already set at rma.cc:162.
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctxQueues_[0]));
  EXPECT_EQ(comm_->planner.nTasksRma, 1);
  EXPECT_TRUE(plan_->isRma);
}

// Second ncclCalloc fails: the arm that actually needs the free() pair.
TEST_F(RmaScheduleTest, WaitSignal_SecondCallocFails_FreesFirstAllocation) {
  EnqueueWait(/*ctx=*/0, {7});
  g_rmaCallocFailAt = g_rmaCallocCallIndex + 1;  // peers succeeds, nsignals fails

  EXPECT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSystemError);
  EXPECT_TRUE(ncclIntruQueueEmpty(&plan_->rmaTaskQueueProxy));
  // The name of this test is the assertion: the first allocation must be
  // released by the fail label, not leaked.
  EXPECT_EQ(g_rmaFreeCalls, 3);
  // The fail label leaves the plan half-built: the task was dequeued at
  // rma.cc:159 and is neither re-queued nor returned to the pool, the decrement
  // at rma.cc:236 is unreachable, and isRma was already set at rma.cc:162.
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctxQueues_[0]));
  EXPECT_EQ(comm_->planner.nTasksRma, 1);
  EXPECT_TRUE(plan_->isRma);
}

// lsaAccessible on the put path routes to the CE queue.
TEST_F(RmaScheduleTest, Put_LsaPeer_GoesToCeQueue) {
  EnqueuePut(/*ctx=*/0, /*peer=*/2);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->func, ncclFuncPutSignal);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 1);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksCe, 1);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksProxy, 0);
  EXPECT_EQ(PeersOf(&plan_->rmaTaskQueueCe), (std::vector<int>{2}));
  EXPECT_EQ(comm_->planner.nTasksRma, 0);
}

// A remote peer routes to the proxy queue instead.
TEST_F(RmaScheduleTest, Put_RemotePeer_GoesToProxyQueue) {
  EnqueuePut(/*ctx=*/0, /*peer=*/11);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasksCe, 0);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksProxy, 1);
  EXPECT_EQ(PeersOf(&plan_->rmaTaskQueueProxy), (std::vector<int>{11}));
}

// canBatchRmaTasks: identical funcs batch, so consecutive puts share one plan.
TEST_F(RmaScheduleTest, Batching_ConsecutiveSameFunc_BatchesAll) {
  EnqueuePut(0, 1);
  EnqueuePut(0, 2);
  EnqueuePut(0, 3);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 3);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksCe, 3);
  EXPECT_EQ(PeersOf(&plan_->rmaTaskQueueCe), (std::vector<int>{1, 2, 3}));
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctxQueues_[0]));
  EXPECT_EQ(comm_->planner.nTasksRma, 0);
}

// canBatchRmaTasks: funcs differ but both are put-or-signal, so they still batch.
TEST_F(RmaScheduleTest, Batching_PutSignalThenSignal_BatchesTogether) {
  EnqueuePut(0, 1, ncclFuncPutSignal);
  EnqueuePut(0, 2, ncclFuncSignal);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 2);
  EXPECT_EQ(PeersOf(&plan_->rmaTaskQueueCe), (std::vector<int>{1, 2}));
}

// The reverse pairing, so the predicate is pinned on both arguments.
TEST_F(RmaScheduleTest, Batching_SignalThenPutSignal_BatchesTogether) {
  EnqueuePut(0, 1, ncclFuncSignal);
  EnqueuePut(0, 2, ncclFuncPutSignal);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->func, ncclFuncSignal);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 2);
}

// canBatchRmaTasks falls through to false: a WaitSignal behind a put stops the loop.
TEST_F(RmaScheduleTest, Batching_WaitSignalBehindPut_StopsBatching) {
  EnqueuePut(0, 1);
  EnqueueWait(0, {2});

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 1);
  EXPECT_EQ(PeersOf(&plan_->rmaTaskQueueCe), (std::vector<int>{1}));
  ASSERT_EQ(QueueLength(&ctxQueues_[0]), 1);
  EXPECT_EQ(ctxQueues_[0].head->func, ncclFuncWaitSignal);
  EXPECT_EQ(comm_->planner.nTasksRma, 1);
}

// Each context's queue is drained only up to its first WaitSignal, so per-context
// FIFO holds and WaitSignal stays one-context-per-plan.
TEST_F(RmaScheduleTest, Batching_PutBehindWaitSignal_IsLeftForTheNextPlan) {
  EnqueueWait(0, {2});
  EnqueuePut(0, 1);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->func, ncclFuncWaitSignal);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 1);
  ASSERT_EQ(QueueLength(&ctxQueues_[0]), 1);
  EXPECT_EQ(ctxQueues_[0].head->func, ncclFuncPutSignal);
  EXPECT_EQ(comm_->planner.nTasksRma, 1);
}

// The else branch takes the leading task without checking its func, and the
// drain loop gates only on the tasks it walks -- so a leading non-put/signal
// func does not hold back the puts queued behind it.
TEST_F(RmaScheduleTest, Batching_NonPutSignalLeadFunc_StillTakesFollowingPuts) {
  EnqueuePut(0, 1, ncclFuncAllReduce);
  EnqueuePut(0, 2, ncclFuncPutSignal);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->func, ncclFuncAllReduce);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 2);
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctxQueues_[0]));
  EXPECT_EQ(comm_->planner.nTasksRma, 0);
}

// The drain loop stops at the first task that is not a put or signal, whatever
// the leading task's func was -- matching a func is no longer a reason to batch.
TEST_F(RmaScheduleTest, Batching_StopsAtFirstNonPutSignalTask) {
  EnqueuePut(0, 1, ncclFuncAllReduce);
  EnqueuePut(0, 2, ncclFuncAllReduce);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 1);
  EXPECT_EQ(PeersOf(&plan_->rmaTaskQueueCe), (std::vector<int>{1}));
  ASSERT_EQ(QueueLength(&ctxQueues_[0]), 1);
  EXPECT_EQ(comm_->planner.nTasksRma, 1);
}

// A task's own ctx field no longer gates batching -- the loop walks the queues,
// so a task whose ctx disagrees with the queue it sits in is batched anyway.
TEST_F(RmaScheduleTest, Batching_TaskCtxFieldDoesNotGateBatching) {
  EnqueuePut(0, 1);
  ncclTaskRma* stray = EnqueuePut(0, 2);
  stray->ctx = 3;

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 2);
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctxQueues_[0]));
  EXPECT_EQ(comm_->planner.nTasksRma, 0);
}

// The loop's own accessibility test, distinct from the first task's: one batch
// can straddle both transports.
TEST_F(RmaScheduleTest, Batching_MixedTransports_SplitsAcrossBothQueues) {
  EnqueuePut(0, 1);   // local  -> CE
  EnqueuePut(0, 12);  // remote -> proxy
  EnqueuePut(0, 3);   // local  -> CE
  EnqueuePut(0, 13);  // remote -> proxy

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 4);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksCe, 2);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksProxy, 2);
  EXPECT_EQ(PeersOf(&plan_->rmaTaskQueueCe), (std::vector<int>{1, 3}));
  EXPECT_EQ(PeersOf(&plan_->rmaTaskQueueProxy), (std::vector<int>{12, 13}));
  EXPECT_EQ(comm_->planner.nTasksRma, 0);
}

// The loop's empty-queue exit, as opposed to the canBatchRmaTasks break.
TEST_F(RmaScheduleTest, Batching_DrainsQueueWithoutOverrun) {
  for (int i = 0; i < 8; i++) EnqueuePut(0, i % 4);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 8);
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctxQueues_[0]));
  EXPECT_EQ(comm_->planner.nTasksRma, 0);
}

// One launch covers every context: put/signal tasks are pulled from all of
// them into a single plan, not just the one the scan selected.
TEST_F(RmaScheduleTest, Batching_PullsPutsFromEveryContext) {
  EnqueuePut(1, 1);
  EnqueuePut(1, 2);
  EnqueuePut(2, 3);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 3);
  EXPECT_EQ(PeersOf(&plan_->rmaTaskQueueCe), (std::vector<int>{1, 2, 3}));
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctxQueues_[2]));
  EXPECT_EQ(comm_->planner.nTasksRma, 0);
}

// The debug states that change which arm of a log predicate is taken. INFO is
// `(level >= INFO && (flags & mask)) || level < 0`, and NCCLCHECKGOTO wraps its
// backtrace in `if (ncclDebugNoWarn == 0)`, so covering both arms of each needs
// the mask off and the "debug not yet initialised" level as well as the obvious
// enabled state.
struct DebugState {
  int level;
  uint64_t mask;
  int noWarn;
};
const DebugState kDebugStates[] = {
  {NCCL_LOG_INFO, ~0ULL, 0},  // logging on, backtrace printed
  {NCCL_LOG_INFO, ~0ULL, 1},  // logging on, backtrace suppressed
  {NCCL_LOG_INFO, 0ULL, 0},   // level passes, subsystem mask does not
  {-1, ~0ULL, 0},             // debug not yet initialised
};

// Which call in a repeated seam should fail; 1 = the first call.
struct FailAt {
  int eventRecord = 0;
  int streamWait = 0;
  bool proxyLaunch = false;
  bool ceLaunch = false;
};

// --- debug-logging configuration -------------------------------------------
//
// Everything above runs at the default NCCL_LOG_NONE, which short-circuits both
// the INFO in scheduleRmaTasksToPlan and the backtrace INFO inside
// NCCLCHECKGOTO, so their format strings are never evaluated. That matters:
// rma.cc's plan summary passes six arguments, and a mismatch there would only
// ever fault for a user running NCCL_DEBUG=INFO.
//
// The env var cannot drive this here (the real ncclDebugInit is not linked in),
// so the fakes' debug globals are set directly. They are process-wide, so the
// fixture saves and restores them.

class RmaDebugLoggingTest : public RmaScheduleTest {
protected:
  int savedLevel_ = 0;
  uint64_t savedMask_ = 0;
  int savedNoWarn_ = 0;

  void SetUp() override {
    RmaScheduleTest::SetUp();
    savedLevel_ = ncclDebugLevel;
    savedMask_ = ncclDebugMask;
    savedNoWarn_ = ncclDebugNoWarn;
    ncclDebugLevel = NCCL_LOG_INFO;
    ncclDebugMask = ~0ULL;  // every subsystem, so NCCL_COLL passes the mask test
    // Establish it rather than inherit it: the tests below that do not call
    // SetDebug exist to take NCCLCHECKGOTO's `ncclDebugNoWarn == 0` arm, and
    // none of them asserts on log output, so a stale 1 would silently skip it.
    ncclDebugNoWarn = 0;
  }

  void SetDebug(const DebugState& d) {
    ncclDebugLevel = d.level;
    ncclDebugMask = d.mask;
    ncclDebugNoWarn = d.noWarn;
  }

  // Run one mixed-path call with a chosen check site forced to fail.
  ncclResult_t RunMixed(bool put, const FailAt& f) {
    ScopedHook rec(g_hipEventRecord, FailsOnCall<hipEvent_t, hipStream_t>(f.eventRecord));
    ScopedHook wt(g_hipStreamWaitEvent,
                  FailsOnCall<hipStream_t, hipEvent_t, unsigned int>(f.streamWait));
    using LaunchFn = std::function<ncclResult_t(ncclComm*, ncclKernelPlan*, hipStream_t)>;
    LaunchFn fail = LaunchReturns(ncclInternalError);
    LaunchFn ok = LaunchReturns(ncclSuccess);
    ScopedHook pp(g_rmaProxyPutLaunch, f.proxyLaunch ? fail : ok);
    ScopedHook cp(g_rmaCePutLaunch, f.ceLaunch ? fail : ok);
    ScopedHook pw(g_rmaProxyWaitLaunch, f.proxyLaunch ? fail : ok);
    ScopedHook cw(g_rmaCeWaitLaunch, f.ceLaunch ? fail : ok);

    ncclRmaArgs args{};
    args.func = put ? ncclFuncPutSignal : ncclFuncWaitSignal;
    args.nRmaTasksProxy = 1;
    args.nRmaTasksCe = 1;
    plan_->rmaArgs = &args;
    const ncclResult_t ret = put ? ncclRmaPut(comm_.get(), plan_.get(), kMainStream)
                                 : ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream);
    plan_->rmaArgs = nullptr;  // args is a local; plan_ outlives this frame
    return ret;
  }

  void TearDown() override {
    ncclDebugLevel = savedLevel_;
    ncclDebugMask = savedMask_;
    ncclDebugNoWarn = savedNoWarn_;
    RmaScheduleTest::TearDown();
  }
};

// The plan-summary INFO fires and its six-argument format string is evaluated.
TEST_F(RmaDebugLoggingTest, SchedulePlanSummaryIsLoggedWithoutChangingResult) {
  EnqueuePut(0, 1);
  EnqueuePut(0, 12);

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 2);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksCe, 1);
  EXPECT_EQ(plan_->rmaArgs->nRmaTasksProxy, 1);
}

// The same INFO reached with the split's counters rather than the batch loop's.
TEST_F(RmaDebugLoggingTest, WaitSignalSplitSummaryIsLogged) {
  EnqueueWait(/*ctx=*/0, {2, 9});

  ASSERT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan_.get()), ncclSuccess);

  EXPECT_EQ(plan_->rmaArgs->nRmaTasks, 2);
}

// NCCLCHECKGOTO's `ncclDebugNoWarn == 0` guard, taken on a launcher failure.
TEST_F(RmaDebugLoggingTest, LaunchFailureLogsBacktraceAndStillReturnsError) {
  ncclRmaArgs args{};
  args.func = ncclFuncPutSignal;
  args.nRmaTasksProxy = 1;
  plan_->rmaArgs = &args;
  ScopedHook proxy(g_rmaProxyPutLaunch,
                   LaunchReturns(ncclInternalError));

  EXPECT_EQ(ncclLaunchRma(comm_.get(), plan_.get()), ncclInternalError);
  EXPECT_EQ(proxy.calls, 1);
}

// The other arm of that guard: suppressing the backtrace must not change the error.
TEST_F(RmaDebugLoggingTest, LaunchFailureWithNoWarnSuppressesLogButKeepsError) {
  ncclDebugNoWarn = 1;
  ncclRmaArgs args{};
  args.func = ncclFuncPutSignal;
  args.nRmaTasksProxy = 1;
  plan_->rmaArgs = &args;
  ScopedHook proxy(g_rmaProxyPutLaunch,
                   LaunchReturns(ncclInternalError));

  EXPECT_EQ(ncclLaunchRma(comm_.get(), plan_.get()), ncclInternalError);
}

// Every check site in the two mixed paths, driven to failure under each debug
// state. Only NCCLCHECKGOTO reads the debug state -- its backtrace sits behind
// `ncclDebugNoWarn == 0` and an INFO predicate -- so the launcher rows are what
// the sweep varies. CUDACHECKGOTO logs through ERROR, an unconditional
// ncclDebugLog, so the fence rows run the same code in all four states; they are
// swept anyway to keep the table one shape.
TEST_F(RmaDebugLoggingTest, EveryMixedPathCheckSiteUnwindsUnderEveryDebugState) {
  const FailAt sites[] = {
    {1, 0, false, false}, {2, 0, false, false},
    {0, 1, false, false}, {0, 2, false, false},
    {0, 0, true, false},  {0, 0, false, true},
  };
  for (const DebugState& d : kDebugStates) {
    for (const FailAt& f : sites) {
      SetDebug(d);
      EXPECT_NE(RunMixed(/*put=*/false, f), ncclSuccess);
      EXPECT_NE(RunMixed(/*put=*/true, f), ncclSuccess);
    }
  }
}

// The same sites returning ncclInProgress instead, which NCCLCHECKGOTO must treat
// as non-fatal at each launcher rather than only where a dedicated test checks.
TEST_F(RmaDebugLoggingTest, EveryLauncherInProgressIsNonFatalUnderEveryDebugState) {
  for (const DebugState& d : kDebugStates) {
    SetDebug(d);
    for (bool put : {false, true}) {
      for (bool proxy : {false, true}) {
        LaunchLog log;
        AllHooks hooks(log);
        auto inProgress = LaunchReturns(ncclInProgress);
        ncclRmaArgs args{};
        args.func = put ? ncclFuncPutSignal : ncclFuncWaitSignal;
        args.nRmaTasksProxy = 1;
        args.nRmaTasksCe = 1;
        plan_->rmaArgs = &args;
        if (put && proxy) {
          ScopedHook h(g_rmaProxyPutLaunch, inProgress);
          EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSuccess);
        } else if (put) {
          ScopedHook h(g_rmaCePutLaunch, inProgress);
          EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclInProgress);
        } else if (proxy) {
          ScopedHook h(g_rmaProxyWaitLaunch, inProgress);
          EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclSuccess);
        } else {
          ScopedHook h(g_rmaCeWaitLaunch, inProgress);
          EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclInProgress);
        }
      }
    }
  }
}

// The single-transport arms, which the mixed-path matrix above never reaches.
TEST_F(RmaDebugLoggingTest, SingleTransportArmsUnwindUnderEveryDebugState) {
  using LaunchFn = std::function<ncclResult_t(ncclComm*, ncclKernelPlan*, hipStream_t)>;
  LaunchFn fail = LaunchReturns(ncclInternalError);
  for (const DebugState& d : kDebugStates) {
    for (bool put : {false, true}) {
      for (bool proxyOnly : {false, true}) {
        SetDebug(d);
        ncclRmaArgs args{};
        args.func = put ? ncclFuncPutSignal : ncclFuncWaitSignal;
        args.nRmaTasksProxy = proxyOnly ? 1 : 0;
        args.nRmaTasksCe = proxyOnly ? 0 : 1;
        plan_->rmaArgs = &args;
        ScopedHook pp(g_rmaProxyPutLaunch, fail);
        ScopedHook cp(g_rmaCePutLaunch, fail);
        ScopedHook pw(g_rmaProxyWaitLaunch, fail);
        ScopedHook cw(g_rmaCeWaitLaunch, fail);
        EXPECT_EQ(put ? ncclRmaPut(comm_.get(), plan_.get(), kMainStream)
                      : ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream),
                  ncclInternalError);
      }
    }
  }
}

// ncclLaunchRma's three dispatch arms, failing and in-progress, under each state.
TEST_F(RmaDebugLoggingTest, EveryDispatchArmUnwindsUnderEveryDebugState) {
  const ncclFunc_t funcs[] = {ncclFuncPutSignal, ncclFuncSignal, ncclFuncWaitSignal};
  for (const DebugState& d : kDebugStates) {
    for (ncclFunc_t fn : funcs) {
      for (ncclResult_t r : {ncclInternalError, ncclInProgress}) {
        SetDebug(d);
        ncclRmaArgs args{};
        args.func = fn;
        args.nRmaTasksProxy = 1;
        plan_->rmaArgs = &args;
        auto ret = [r](ncclComm*, ncclKernelPlan*, hipStream_t) { return r; };
        ScopedHook put(g_rmaProxyPutLaunch, ret);
        ScopedHook wait(g_rmaProxyWaitLaunch, ret);
        EXPECT_EQ(ncclLaunchRma(comm_.get(), plan_.get()), r);
      }
    }
  }
}

// Both ncclCalloc arms of the split, under each debug state.
TEST_F(RmaDebugLoggingTest, BothCallocFailuresUnwindUnderEveryDebugState) {
  for (const DebugState& d : kDebugStates) {
    for (int which : {0, 1}) {
      SetDebug(d);
      auto plan = std::make_unique<ncclKernelPlan>();
      SetNumRmaCtx(4);
      EnqueueWait(/*ctx=*/0, {7});
      g_rmaCallocFailAt = g_rmaCallocCallIndex + which;
      EXPECT_EQ(scheduleRmaTasksToPlan(comm_.get(), plan.get()), ncclSystemError);
      g_rmaCallocFailAt = -1;
    }
  }
}

// --- ncclRmaProxyEnabled / ncclRmaInitialized (rma.cc:19, :24) --------------
//
// Both are plain predicates over comm state, so they need no plan or launch
// machinery.

class RmaProxyGatingTest : public RmaTestBase {
protected:
  void SetUp() override {
    RmaTestBase::SetUp();
    // The combination that makes ncclRmaProxyEnabled true, so each test can
    // knock out exactly one term.
    comm_->config.numRmaCtx = 4;
    comm_->globalRmaProxySupport = true;
    comm_->hostRmaSupport = true;
    comm_->rmaState.rmaCeState.initialized = true;
    SetOneLsaTeam(false);
  }

  void TearDown() override {
    comm_->devrState.lsaRankList = nullptr;  // borrowed from lsaRanks_
    RmaTestBase::TearDown();
  }

  // ncclDevrIsOneLsaTeam is dev_runtime.cc's real predicate here, derived from
  // computeLsaSize; a non-zero bigSize short-circuits that to lsaSize, so the
  // team is "one" exactly when lsaSize == nRanks. Move nRanks, not lsaSize, so
  // lsaSize keeps matching lsaRanks_ -- isLsaAccessible walks it by lsaSize.
  void SetOneLsaTeam(bool oneTeam) {
    comm_->devrState.bigSize = 1;
    lsaRanks_ = {0, 1};
    comm_->devrState.lsaSize = static_cast<int>(lsaRanks_.size());
    comm_->devrState.lsaRankList = lsaRanks_.data();
    comm_->nRanks = oneTeam ? 2 : 4;
  }

private:
  std::vector<int> lsaRanks_;
};

// All four terms hold, which is the only way the predicate is true.
TEST_F(RmaProxyGatingTest, ProxyEnabled_AllTermsHold) {
  EXPECT_TRUE(ncclRmaProxyEnabled(comm_.get()));
}

// A single LSA team means every peer is local, so the proxy is not wanted.
TEST_F(RmaProxyGatingTest, ProxyEnabled_OneLsaTeam_IsFalse) {
  SetOneLsaTeam(true);
  EXPECT_FALSE(ncclRmaProxyEnabled(comm_.get()));
}

// No RMA contexts configured.
TEST_F(RmaProxyGatingTest, ProxyEnabled_NoContexts_IsFalse) {
  comm_->config.numRmaCtx = 0;
  EXPECT_FALSE(ncclRmaProxyEnabled(comm_.get()));
}

// The comm-wide proxy support flag is the third term.
TEST_F(RmaProxyGatingTest, ProxyEnabled_NoGlobalProxySupport_IsFalse) {
  comm_->globalRmaProxySupport = false;
  EXPECT_FALSE(ncclRmaProxyEnabled(comm_.get()));
}

// NCCL_RMA_DISABLE is the operator override, read through the param redirect.
TEST_F(RmaProxyGatingTest, ProxyEnabled_RmaDisableSet_IsFalse) {
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deflt) -> int64_t {
    return std::string(env) == "RMA_DISABLE" ? 1 : deflt;
  });
  EXPECT_FALSE(ncclRmaProxyEnabled(comm_.get()));
}

// Initialized needs host RMA, CE set up, and -- only when the proxy is enabled --
// a connected proxy.
TEST_F(RmaProxyGatingTest, Initialized_AllPrerequisitesMet) {
  comm_->rmaState.rmaProxyState.connected = true;
  EXPECT_TRUE(ncclRmaInitialized(comm_.get()));
}

TEST_F(RmaProxyGatingTest, Initialized_NoHostRmaSupport_IsFalse) {
  comm_->hostRmaSupport = false;
  EXPECT_FALSE(ncclRmaInitialized(comm_.get()));
}

TEST_F(RmaProxyGatingTest, Initialized_CeNotInitialized_IsFalse) {
  comm_->rmaState.rmaCeState.initialized = false;
  EXPECT_FALSE(ncclRmaInitialized(comm_.get()));
}

// Proxy enabled but not connected is the one arm that depends on the predicate.
TEST_F(RmaProxyGatingTest, Initialized_ProxyEnabledButNotConnected_IsFalse) {
  comm_->rmaState.rmaProxyState.connected = false;
  EXPECT_FALSE(ncclRmaInitialized(comm_.get()));
}

// ...and with the proxy disabled, connectivity stops mattering.
TEST_F(RmaProxyGatingTest, Initialized_ProxyDisabled_IgnoresConnectedFlag) {
  SetOneLsaTeam(true);
  comm_->rmaState.rmaProxyState.connected = false;
  EXPECT_TRUE(ncclRmaInitialized(comm_.get()));
}

// --- ensureRmaProxyReady (rma.cc:43) ---------------------------------------
//
// Every proxy launch is gated on the proxy being connected. It runs before the
// dispatch both launch suites cover, so its arms live here rather than being
// repeated in each.

class RmaProxyReadyTest : public RmaTestBase {
protected:
  ncclRmaArgs args_{};

  void SetUp() override {
    RmaTestBase::SetUp();
    plan_->rmaArgs = &args_;
    args_.func = ncclFuncPutSignal;
    args_.nRmaTasksProxy = 1;
    args_.nRmaTasksCe = 0;
  }
};

// Proxy work queued but the proxy never connected.
TEST_F(RmaProxyReadyTest, ProxyTasksButNotConnected_ReturnsInvalidUsage) {
  comm_->rmaState.rmaProxyState.connected = false;
  AllHooks hooks(log_);

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclInvalidUsage);
  EXPECT_TRUE(log_.entries.empty());
}

// Connected, but no vtable to dispatch through -- the second half of the check.
TEST_F(RmaProxyReadyTest, ProxyTasksButNoVtable_ReturnsInvalidUsage) {
  comm_->rmaState.rmaProxyState.ncclRma = nullptr;
  AllHooks hooks(log_);

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclInvalidUsage);
  EXPECT_TRUE(log_.entries.empty());
}

// A failing connect attempt surfaces unchanged rather than being reported as a
// readiness problem.
TEST_F(RmaProxyReadyTest, ConnectOnceFails_Propagates) {
  ScopedHook connect(g_devrRmaProxyConnectOnce, [](ncclComm*) { return ncclSystemError; });
  AllHooks hooks(log_);

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSystemError);
  EXPECT_EQ(connect.calls, 1);
  EXPECT_TRUE(log_.entries.empty());
}

// No proxy work means the gate is skipped entirely, so a disconnected proxy is
// irrelevant to a CE-only plan.
TEST_F(RmaProxyReadyTest, CeOnlyPlanSkipsTheGate) {
  args_.nRmaTasksProxy = 0;
  args_.nRmaTasksCe = 1;
  comm_->rmaState.rmaProxyState.connected = false;
  comm_->rmaState.rmaProxyState.ncclRma = nullptr;
  ScopedHook connect(g_devrRmaProxyConnectOnce, [](ncclComm*) { return ncclSystemError; });
  AllHooks hooks(log_);

  EXPECT_EQ(ncclRmaPut(comm_.get(), plan_.get(), kMainStream), ncclSuccess);
  EXPECT_EQ(connect.calls, 0);
  EXPECT_EQ(log_.CountOf(LaunchLog::kCePut), 1);
}

// The same gate fronts the WaitSignal entry point.
TEST_F(RmaProxyReadyTest, WaitSignalIsGatedToo) {
  args_.func = ncclFuncWaitSignal;
  comm_->rmaState.rmaProxyState.connected = false;
  AllHooks hooks(log_);

  EXPECT_EQ(ncclRmaWaitSignal(comm_.get(), plan_.get(), kMainStream), ncclInvalidUsage);
  EXPECT_TRUE(log_.entries.empty());
}

}  // namespace
