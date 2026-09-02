/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/rccl_wrap.cc, shared by every host-only
// microtest binary. Named after the PRODUCTION TU that owns these symbols, not
// after whichever test target needed them first; see MICROTEST_README.md.

#ifndef RCCL_TEST_HOST_RCCL_WRAP_FAKES_H_
#define RCCL_TEST_HOST_RCCL_WRAP_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "nccl.h"
#include "nccl_common.h"

struct ncclComm;
struct ncclTaskColl;

// LINK FLOOR ONLY: a seam marked `// UNDRIVEN` is declared so the binary links
// and so an accidental call is visible, NOT because its path is covered.

// -------------------------------------------------------------------------
// Tuning-override seams (rccl_wrap.cc:109-1751). All default to no-ops so a test
// sees the *unmodified* selection, then overrides exactly one.
// -------------------------------------------------------------------------
extern std::function<void(struct ncclComm*, size_t const&, struct ncclTaskColl*)>
    g_rcclUpdateCollectiveProtocol;  // UNDRIVEN
extern std::function<void(struct ncclComm*, size_t const&, struct ncclTaskColl*)>
    g_rcclSetPipelining;  // UNDRIVEN
extern std::function<ncclResult_t(struct ncclComm*, ncclFunc_t, size_t, int&)>
    g_rcclOverrideChannels;
extern int g_rcclOverrideChannelsCalls;
extern bool g_rcclIsArchSupportedForFunc;  // UNDRIVEN
// Call counters for the no-op tuning hooks: a no-op that was never called and one
// that was look identical without these, so a dropped call site would be silent.
extern int g_rcclUpdateCollectiveProtocolCalls;
extern int g_rcclSetPipeliningCalls;
extern int g_rcclUpdateThreadThresholdCalls;
extern int g_rcclOptThreadBlockSizeCalls;
extern ncclResult_t g_rcclOverrideAlgorithmResult;  // UNDRIVEN
extern ncclResult_t g_rcclOverrideProtocolResult;  // UNDRIVEN
extern int g_rcclOverrideAlgorithmCalls;
extern int g_rcclOverrideProtocolCalls;

// CE (copy-engine) allreduce gates (rccl_wrap.cc:834-855).
extern bool g_rcclCeAllReduceAllowed;  // UNDRIVEN
extern int g_rcclCeAllReduceGraphLatchTickCalls;  // UNDRIVEN
extern bool g_rcclCeAllReduceGraphLatchTickLastCapturing;  // UNDRIVEN

// -------------------------------------------------------------------------
// WARP_SPEED seams (rccl_wrap.cc:1448+). enqueue.cc calls these from five sites
// inside `#ifdef ENABLE_WARP_SPEED` (:318, :2221, :2615, :2736, :2738), two of
// which are in functions this suite covers (finishPlan and topoGetAlgoInfo).
//
// ENABLE_WARP_SPEED defaults OFF and is forced OFF unless GPU_TARGETS is gfx950
// (CMakeLists.txt:582-592), so on most configurations those five lines are
// preprocessed away and these symbols are never referenced. A gfx950 build with
// the option ON compiles them, and without these fakes the target fails to LINK.
//
// The defaults make an ENABLE_WARP_SPEED=ON build behave exactly like an OFF
// one: report unsupported, return the channel count unchanged, succeed, no-op.
// That keeps every test written against the OFF configuration honest on both.
// -------------------------------------------------------------------------
extern bool g_rcclWarpSpeedSupported;  // UNDRIVEN
extern int g_rcclWarpSpeedSupportedCalls;  // UNDRIVEN
extern ncclResult_t g_rcclSetWarpSpeedAutoResult;  // UNDRIVEN
extern int g_rcclSetWarpSpeedAutoCalls;  // UNDRIVEN
extern int g_rcclSetWarpSpeedCUsCalls;  // UNDRIVEN
// Applied to the caller's nc by rcclWarpSpeedAdjustChannels. Identity by
// default; production may shrink the count.
extern std::function<int(struct ncclComm*, struct ncclTaskColl*, int)>
    g_rcclWarpSpeedAdjustChannels;  // UNDRIVEN

void ResetRcclWrapFakes();

#endif  // RCCL_TEST_HOST_RCCL_WRAP_FAKES_H_
