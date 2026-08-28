/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/enqueue.cc: the UUT is #include'd, so static
// helpers are directly callable.
//
// LINE-NUMBER BASE: every `enqueue.cc:NNNN` citation in these tests refers to the
// HIPIFIED copy (build/hipify/src/enqueue.cc), which is what this TU actually
// compiles. Hipify inserts one line near the top, so these run exactly one higher
// than the same code in src/enqueue.cc -- subtract 1 when navigating the original.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "fakes/enqueue_fakes.h"
#include "../common/LogCapture.hpp"

// alloc.h first, so its macros are visible to be #undef'd before enqueue.cc's
// transitive includes see them.
#include "alloc.h"

#include "fakes/param_redirect.h"

// The real RCCL_PARAM macros cache and declare a pthread_mutex_t global;
// redirect so params stay per-test controllable.
#undef RCCL_PARAM
#define RCCL_PARAM(name, env, deftVal) \
  int64_t rcclParam##name() { return g_loadParam(("RCCL_" env), (deftVal)); }
#undef RCCL_PARAM_NCCL_ALIAS
#define RCCL_PARAM_NCCL_ALIAS(name, env, deftVal) \
  int64_t rcclParam##name() { return g_loadParam(("RCCL_" env), (deftVal)); }

// The NVTX3 range macros expand to NOTHING in this binary, so any guard around
// them shows partial coverage by design.
#if !defined(NVTX_NO_IMPL) && !defined(NVTX_DISABLE)
#include "nvtx.h"  // guarded; enqueue.cc's re-include is a no-op
#undef NCCL_NVTX3_FUNC_RANGE
#define NCCL_NVTX3_FUNC_RANGE
#undef NVTX3_RANGE
#define NVTX3_RANGE(...)
#undef NVTX3_RANGE_ADD_PAYLOAD
#define NVTX3_RANGE_ADD_PAYLOAD(...)
#undef NVTX3_FUNC_WITH_PARAMS
#define NVTX3_FUNC_WITH_PARAMS(...)
#else
#define NCCL_NVTX_H_
#ifndef NCCL_NVTX3_FUNC_RANGE
#define NCCL_NVTX3_FUNC_RANGE
#endif
#endif

// ---------------------------------------------------------------------------
// enqueue.cc:28 includes "common.h", which resolves to src/device/common.h -- a
// DEVICE header. Under --offload-host-only it cannot compile on its own terms
// (extern __shared__ variables; undeclared insert_random_delay_per_warp).
//
// As of this change, the only declaration enqueue.cc uses from it is the six
// ncclDevKernel_Generic_N kernels whose ADDRESSES populate ncclKerns[] at
// enqueue.cc:53; every use is `plan->kernelFn = ncclKerns[i].kernelFn`, an opaque
// void* that is stored, never dereferenced or launched on a host path. So we
// pre-set the device header's own include guard to neuter it and supply those
// six symbols as ordinary host functions.
//
// LIMITS OF THIS SUBSTITUTION -- read before relying on it:
//   * These are host functions, NOT __global__ declarations with the production
//     grid-constant annotation. This binary therefore CANNOT validate the real
//     kernel declarations, their symbols, or the host/device ABI. It exercises
//     host-side table indexing only; kernel-table integrity belongs to a
//     device-linked build.
//   * Neutering the whole header changes the include environment for the rest of
//     the TU. If enqueue.cc later needs another declaration from device/common.h,
//     the build may keep working only because something arrives transitively.
//     A compile error here is the expected signal to revisit this shim -- do not
//     paper over it by adding another local definition.
//
// Pre-setting another header's guard is the same technique init-test.cc uses for
// the nvtx.h / nvtx_stub.h nccl_domain collision.
// ---------------------------------------------------------------------------
#define NCCL_DEVICE_COMMON_H_
// ncclDevKernelArgsDefaultStorage comes from src/include/device.h, which is
// already in scope via enqueue.h -> comm.h. The generated device_table.h is
// NOT needed here and is deliberately not included.
void ncclDevKernel_Generic_1(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_2(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_4(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_8(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_16(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_32(ncclDevKernelArgsDefaultStorage) {}

// ENQUEUE_CC_PATH is ${PROJECT_BINARY_DIR}/hipify/src/enqueue.cc -- enqueue.cc is
// basename-unique in the tree, so hipify keeps its name (no _tmp suffix).
#include ENQUEUE_CC_PATH

class EnqueueMicrotest : public ::testing::Test {
 protected:
  void TearDown() override { ResetEnqueueFakes(); }
};

// ---------------------------------------------------------------------------
// The test bodies, split for reviewability. These are FRAGMENTS, not independent
// files: they share this TU's single anonymous namespace and later ones use
// fixtures from earlier ones, so THE ORDER BELOW IS LOAD-BEARING.
//
// Verified cross-fragment uses:
//   CostTask / ScriptAllTimes / CostTable  defined in batch4, used by batch9
//   RedOpComm                              defined in batch5, used by batch9
//   BatchPlanComm                          defined in batch6, used by batch8, batch9
//
// What each holds (the numbering is historical; contents are what matter):
//   1  ncclFuncTrafficPerByte, ncclTestBudget, the shmem constexprs,
//      calcP2pChannelCount, geteActivationMask / gettaskEventHandle
//   2  hostToDevRedOp
//   3  calcCollChunking
//   4  rcclKernelPackedChannels, ncclGetCollNetSupport, initCollCostTable,
//      updateCollCostTable
//   5  ncclRedOpCreatePreMulSum_impl / ncclRedOpDestroy_impl,
//      rcclEffectiveP2pBatchEnable, getImplicitOrder
//   6  addWorkBatchToPlan
//   7  finishPlan
//   8  waitWorkFifoAvailable, ncclPlanSetDefaultKernel, addProxyOpIfNeeded
//   9  mutation reinforcements + topoGetAlgoInfo interaction tests (covers
//      functions defined across 4, 6 and 7 -- look here for a mutation-driven test)
// ---------------------------------------------------------------------------
#include "tests_batch1.inc"
#include "tests_batch2.inc"
#include "tests_batch3.inc"
#include "tests_batch4.inc"
#include "tests_batch5.inc"
#include "tests_batch6.inc"
#include "tests_batch7.inc"
#include "tests_batch8.inc"
#include "tests_batch9.inc"
