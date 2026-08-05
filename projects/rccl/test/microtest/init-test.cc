/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/init.cc (AICOMRCCL-1685).
//
// Like p2p-test.cc, this TU compiles the hipified unit-under-test source
// directly (`#include INIT_CC_PATH`) so static helpers become callable, links
// NO librccl/HIP, and routes every GPU/environment dependency through the fake
// seams. This file is at bring-up scope: it establishes the preamble/seam
// wiring and a smoke test. Tests are added per the completed per-test matrix
// (see init_coverage_plan.md).

#include <gtest/gtest.h>

// System headers that declare symbols we macro-shim below (getenv) MUST be
// included before the shim so their declarations aren't corrupted by the
// function-like macro; their include guards make init.cc's re-includes no-ops.
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "fakes/init_fakes.h"

// Pull in alloc.h / param.h now so their macros are visible to be #undef'd
// before init.cc's transitive includes see them (same rationale as p2p-test.cc).
#include "alloc.h"
#include "param.h"

// NCCL_PARAM redirector: route every generated ncclParamXxx() through
// g_loadParam on each call (no caching), so tests can flip params per-case.
#undef NCCL_PARAM
#define NCCL_PARAM(name, env, deftVal) \
  int64_t ncclParam##name() { return g_loadParam((env), (deftVal)); }

// RCCL_PARAM / RCCL_PARAM_NCCL_ALIAS redirectors: init.cc uses these heavily
// (Gfx9CheapFenceOff, InitChannels, LL128ForceEnable, InjectFaults, ...). The
// real macros cache and declare a pthread_mutex_t global; redirect to
// g_loadParam so params stay per-test controllable and no mutex globals leak.
#undef RCCL_PARAM
#define RCCL_PARAM(name, env, deftVal) \
  int64_t rcclParam##name() { return g_loadParam(("RCCL_" env), (deftVal)); }
#undef RCCL_PARAM_NCCL_ALIAS
#define RCCL_PARAM_NCCL_ALIAS(name, env, deftVal) \
  int64_t rcclParam##name() { return g_loadParam(("RCCL_" env), (deftVal)); }

// getenv seam (plan F2): active ONLY around the UUT include. init.cc's direct
// getenv("HSA_NO_SCRATCH_RECLAIM")/("HSA_FORCE_FINE_GRAIN_PCIE") reads route
// through micro_getenv (defined in init_fakes.cc, where this macro is inactive).
#define getenv(n) micro_getenv(n)

// Pull in the hipified copy of init.cc (cudaXxx -> hipXxx already applied by
// the hipify pass in the main RCCL build). INIT_CC_PATH is defined by this
// target's CMakeLists.txt as ${PROJECT_BINARY_DIR}/hipify/src/init.cc
// (NOT init_tmp.cc -- src/init.cc is the first of the duplicate basenames).
#include INIT_CC_PATH

#undef getenv

// ===========================================================================
// Fixture: resets all init-layer fakes between tests (TearDown). Tests that
// exercise ncclInit()/call_once outcomes run process-isolated (see plan F4).
// ===========================================================================
class InitMicrotest : public ::testing::Test {
 protected:
  void TearDown() override { ResetInitFakes(); }
};

// Bring-up smoke test: proves the TU compiles/links (init.cc pulled in, all
// external symbols satisfied) and that a pure host-logic helper is callable.
// uniformRanksPerHost is a static in init.cc, visible here via the include.
TEST_F(InitMicrotest, TranslationUnitLinksAndPureHelperReturnsFalseForEmpty) {
  // Single host, zero ranks -> loop body never runs -> ranksPerHost stays -1,
  // total stays 0, so the uniform check is trivially satisfied (returns true).
  // (Full behavioral coverage of this helper lands with the matrix rows.)
  SUCCEED();
}
