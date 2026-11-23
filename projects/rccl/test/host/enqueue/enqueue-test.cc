/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/enqueue.cc: the UUT is #include'd, so static
// helpers are directly callable.
//
// LINE-NUMBER BASE: every `enqueue.cc:NNNN` citation in these tests refers to
// src/enqueue/enqueue.cc as committed, NOT to the hipified copy this TU compiles.
// Hipify inserts one line near the top, so add 1 when navigating
// build/hipify/src/enqueue/enqueue.cc.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/enqueue_test_deps.h"
#include "fakes/strongstream_stubs.h"

// alloc.h first, so its macros are visible to be #undef'd before enqueue.cc's
// transitive includes see them.
#include "alloc.h"

#include "fakes/param_redirect.h"  // redirects NCCL_PARAM and both RCCL_PARAM spellings

#include "fakes/nvtx_redirect.h"  // neuter / block nvtx.h before enqueue.cc includes it

// ---------------------------------------------------------------------------
// enqueue.cc:28 includes "common.h", which resolves to src/device/common.h -- a
// DEVICE header. Under --offload-host-only it cannot compile on its own terms
// (extern __shared__ variables; undeclared insert_random_delay_per_warp).
//
// As of this change, the only declaration enqueue.cc uses from it is the six
// ncclDevKernel_Generic_N kernels whose ADDRESSES populate ncclKerns[] at
// enqueue.cc:53. On every path this binary covers, the only use is
// `plan->kernelFn = ncclKerns[i].kernelFn` -- an opaque void* that is stored and
// nothing more. That is a statement about COVERAGE, not about enqueue.cc, which
// does dereference these addresses elsewhere: ncclInitKernelsForDevice (:110)
// passes them to cudaFuncGetAttributes (:115) and cudaFuncSetAttribute (:121),
// and ncclLaunchKernel (:2218) reads plan->kernelFn back to launch it. No
// covered path reaches any of those three, which is what makes the substitution
// safe today; a test that reaches one invalidates it. So we pre-set the device
// header's own include guard to neuter it and supply those six symbols as
// ordinary host functions.
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
// ncclDevKernelArgsDefaultStorage comes from src/include/device.h, already in
// scope via fakes/enqueue_test_deps.h -> sym_kernels.h (enqueue.cc itself is only
// included later). The generated device_table.h is NOT needed here.
void ncclDevKernel_Generic_1(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_2(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_4(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_8(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_16(ncclDevKernelArgsDefaultStorage) {}
void ncclDevKernel_Generic_32(ncclDevKernelArgsDefaultStorage) {}

// ENQUEUE_CC_PATH is ${PROJECT_BINARY_DIR}/hipify/src/enqueue/enqueue.cc -- hipify
// keeps the src/enqueue/ directory layout.
#include ENQUEUE_CC_PATH

// Link-only floor for ncclLaunchPrepare tests that stop immediately after plan
// construction. None of these paths executes before the test seam returns.
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm*, struct ncclKernelPlan*) { return ncclInternalError; }
hipError_t hipLaunchHostFunc(hipStream_t, hipHostFn_t, void*) { return hipErrorInvalidValue; }

class EnqueueMicrotest : public ::testing::Test {
 protected:
  // Reset in BOTH, not just TearDown. TearDown alone leaves the first test
  // executed running against static-initialiser state rather than reset state --
  // which test that is depends on --gtest_shuffle. Those two states happen to be
  // identical for every global in the reset closure today, but nothing enforces
  // that, and SetUp costs one line to stop relying on it. It also means a future
  // second fixture in this binary cannot inherit a dirty process.
  void SetUp() override {
    ResetEnqueueTestDeps();
#ifdef RCCL_TEST_ENQUEUE_HOOKS
    g_rcclScheduleCollEstimateVisits = 0;
    g_rcclLaunchPrepareStopAfterPlanning = false;
#endif
  }
  void TearDown() override {
    ResetEnqueueTestDeps();
#ifdef RCCL_TEST_ENQUEUE_HOOKS
    g_rcclScheduleCollEstimateVisits = 0;
    g_rcclLaunchPrepareStopAfterPlanning = false;
#endif
  }
};

// ---------------------------------------------------------------------------
// The tests live in enqueue-test.d/, one file per functional area of enqueue.cc.
// Helpers used by more than one of those files live in enqueue-test-scenes.inc.
// CMake globs enqueue-test.d/*.inc into ENQUEUE_TEST_CASES_INC, so a new file
// joins this binary without a build-system edit. See README.md.
// ---------------------------------------------------------------------------
#include "enqueue-test-scenes.inc"

#include ENQUEUE_TEST_CASES_INC
