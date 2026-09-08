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

#include "fakes/enqueue_fakes.h"

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

class CollectivesMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetEnqueueFakes(); }
  void TearDown() override { ResetEnqueueFakes(); }
};

// Absolute-minimal smoke test: prove the TU builds and links, and that the
// anonymous-namespace include actually exposes ncclAlltoAll_impl as a callable
// symbol (not just that collectives.cc compiled). A full behavioural call needs
// a formed comm (topo/archName/channels) and drives the real enqueue.cc linked
// into this binary; that fixture is the next increment. For now, bind the entry
// point so a rename/signature drift breaks this test at compile time.
TEST_F(CollectivesMicrotest, AlltoAll_ImplSymbolIsReachable) {
  auto* fn = &ncclAlltoAll_impl;
  ASSERT_NE(fn, nullptr);
}

}  // namespace
