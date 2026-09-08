/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/collectives.cc: the UUT is #include'd, so its
// file-static helpers and the ncclXxx_impl entry points are directly callable
// without librccl or a GPU.
//
// LINE-NUMBER BASE: every `collectives.cc:NNNN` citation in these tests refers
// to src/collectives.cc as committed, NOT the hipified copy this TU compiles.
// Hipify inserts one line near the top, so add 1 when navigating
// build/<cfg>/hipify/src/collectives.cc.
//
// ANONYMOUS-NAMESPACE INCLUDE: this binary (rccl-UnitTestsMicroEnqueue) already
// links fakes/collectives_fakes.cc, which DEFINES the name tables and
// ncclFuncToString/ncclAlgoToString/ncclProtoToString that collectives.cc also
// defines. Pulling the real .cc in at file scope would be a multiple-definition
// link error. Wrapping the include in an anonymous namespace gives
// collectives.cc's own definitions internal linkage so they no longer collide,
// while its CALLS to external symbols (ncclEnqueueCheck, the DDA/CE eligibility
// helpers) still resolve against the fakes at link time.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>

#include "fakes/enqueue_fakes.h"
#include "fakes/ce_fakes.h"
#include "fakes/dda_alltoall_fakes.h"

// alloc.h first, so its macros are visible to be #undef'd before collectives.cc's
// transitive includes see them (same ordering enqueue-test.cc relies on).
#include "alloc.h"

#include "fakes/param_redirect.h"  // redirects NCCL_PARAM and both RCCL_PARAM spellings
#include "fakes/nvtx_redirect.h"   // neuter / block nvtx.h before collectives.cc includes it

// Pull collectives.cc's include list in at GLOBAL scope first. Every header
// (system + RCCL) has an include guard, so when collectives.cc re-includes them
// from inside the anonymous namespace below they no-op -- keeping <queue> and
// the rest of the standard library (and their `std` namespace) at global scope
// instead of nested inside the anonymous namespace. Only collectives.cc's own
// definitions then land in the anonymous namespace.
#include "argcheck.h"
#include "collectives.h"
#include "enqueue.h"
#include "graph/topo.h"
#include "nccl.h"
#include "api_trace.h"
#include "nvtx_payload_schemas.h"
#include "device/hierarchical_shuffle.h"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/alltoall/dda_alltoall.h"
#include "algorithms/gin/gin_alltoall.h"
#include "sym_kernels.h"
#include "dev_runtime.h"
#include "ce_coll.h"
#include "alltoallv_meta.h"
#include "strongstream.h"

// COLLECTIVES_CC_PATH is ${hipify}/src/collectives.cc -- collectives.cc is
// basename-unique in the tree, so hipify keeps its name (no _tmp suffix).
namespace {
#include COLLECTIVES_CC_PATH
}  // namespace

namespace {

// Backend selection for AllToAll lives in ncclAlltoAll_impl (collectives.cc): it
// either dispatches a DDA path directly and returns, or falls through to
// ncclEnqueueCheck() where -- for CE -- taskAppend() picks CE vs the kernel. CE
// is therefore NOT observable at this boundary; what IS observable is whether
// DDA serviced the call. The fix for AICOMRCCL-2136 makes ncclAlltoAll_impl
// YIELD the DDA path (fall through to enqueue, where CE runs) when CE will serve
// the call under NCCL_CTA_POLICY_ZERO. So the boundary contract these tests
// assert is: "under the ticket's conditions, DDA is NOT dispatched".
class CollectivesMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetEnqueueFakes();
    ResetCeFakes();
    ResetDdaAlltoAllFakes();

    // MEASURED elsewhere: sizeof(ncclComm) is ~3.8 MB -- heap it, never a stack
    // fixture. Zero-init, then set only the fields ncclAlltoAll_impl reads on the
    // pre-dispatch path.
    comm_ = std::make_unique<ncclComm>();
    std::memset(static_cast<void*>(comm_.get()), 0, sizeof(ncclComm));

    // Invalid magic so the real ncclEnqueueCheck() (linked from enqueue.cc) bails
    // at its first line, CommCheck(), returning ncclInvalidArgument without
    // descending into the group/launch machinery. The fall-through path is thus
    // cheap and side-effect-free; these tests never assert on enqueue's return,
    // only on whether DDA was dispatched first.
    comm_->startMagic = 0;
    comm_->endMagic = 0;
    comm_->revokedFlag = 0;

    comm_->nRanks = 8;
    comm_->nNodes = 1;
    comm_->nChannels = 0;
    comm_->archName = const_cast<char*>(kGfx942_);

    // topo is dereferenced for the pivot-A2A gate; give it a real object with
    // pivot disabled so that branch is never taken.
    topo_ = std::make_unique<ncclTopoSystem>();
    std::memset(static_cast<void*>(topo_.get()), 0, sizeof(ncclTopoSystem));
    topo_->pivotA2AEnabled = false;
    comm_->topo = topo_.get();

    comm_->config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;
  }
  void TearDown() override {
    ResetEnqueueFakes();
    ResetCeFakes();
    ResetDdaAlltoAllFakes();
  }

  ncclResult_t callAlltoAll(size_t count = 512, ncclDataType_t dt = ncclFloat32) {
    // Buffers are never dereferenced on the host path under test; distinct
    // non-null pointers are all that is required.
    return ncclAlltoAll_impl(kSendBuf_, kRecvBuf_, count, dt, comm_.get(), /*stream=*/nullptr);
  }

  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclTopoSystem> topo_;
  static constexpr const char* kGfx942_ = "gfx942";
  void* const kSendBuf_ = reinterpret_cast<void*>(0x1000);
  void* const kRecvBuf_ = reinterpret_cast<void*>(0x2000);
};

// Sanity: with DDA disabled, ncclAlltoAll_impl must not dispatch any DDA path --
// it falls through to enqueue. Anchors the fixture and the "not dispatched"
// observation before the conditional cases below.
TEST_F(CollectivesMicrotest, AlltoAll_DdaDisabled_FallsThroughToEnqueue) {
  g_ddaEnabled = false;
  g_ddaIpcEligible = true;  // eligible, but disabled gate must keep DDA off

  callAlltoAll();

  EXPECT_FALSE(DdaAlltoAllDispatched());
}

// Baseline (pre-fix behaviour, DEFAULT policy): DDA enabled + IPC eligible on
// gfx942 -> DDA IPC services the call. This is the path that is correct when the
// user did NOT ask for zero-CTA / CE.
TEST_F(CollectivesMicrotest, AlltoAll_DdaEligible_DefaultPolicy_TakesDdaIpc) {
  g_ddaEnabled = true;
  g_ddaIpcEligible = true;
  comm_->config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;

  callAlltoAll();

  EXPECT_TRUE(g_ddaIpcDispatched);
}

// AICOMRCCL-2136 regression guard. Ticket conditions: gfx942, 8 ranks, DDA IPC
// eligible AND CE available, with NCCL_CTA_POLICY_ZERO set (the user asked for
// CE / zero-CTA). ncclAlltoAll_impl must YIELD the DDA path so the call falls
// through to enqueue, where CE services it. Before the fix, DDA IPC pre-empts CE
// here and this expectation fails -- which is exactly the reported bug.
TEST_F(CollectivesMicrotest, AlltoAll_CeAvailableAndZeroPolicy_YieldsDdaToCe) {
  // SKIPPED until the fix lands: this asserts the DESIRED post-fix behaviour and
  // currently fails on develop, reproducing AICOMRCCL-2136 (DDA IPC pre-empts CE
  // even when NCCL_CTA_POLICY_ZERO asked for CE). Delete this one line as part of
  // the ncclAlltoAll_impl fix to arm the guard.
  GTEST_SKIP() << "AICOMRCCL-2136: un-skip when the DDA-yields-to-CE fix lands";

  g_ddaEnabled = true;
  g_ddaIpcEligible = true;   // DDA IPC would take it (the trigger #8945 enabled)
  g_ceImplemented = true;    // CE is present...
  g_ceAvailable = true;      // ...and available for these buffers
  comm_->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;

  callAlltoAll();

  EXPECT_FALSE(DdaAlltoAllDispatched())
      << "NCCL_CTA_POLICY_ZERO + CE available: DDA must yield so CE can service "
         "the AllToAll (AICOMRCCL-2136)";
}

}  // namespace
