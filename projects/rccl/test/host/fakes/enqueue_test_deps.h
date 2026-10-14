/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Aggregation header for rccl-UnitTestsMicroEnqueue: defines no seams, just the per-TU set and their chained reset.
#ifndef RCCL_TEST_HOST_ENQUEUE_TEST_DEPS_H_
#define RCCL_TEST_HOST_ENQUEUE_TEST_DEPS_H_

#include "ce_fakes.h"            // src/ce_coll.cc
#include "comm_fakes.h"          // src/init.cc comm lifecycle
#include "dev_runtime_fakes.h"   // src/dev_runtime.cc
#include "env_fakes.h"           // src/misc/param.cc + getenv interposition
#include "group_fakes.h"         // src/group.cc
#include "hip_fakes.h"           // HIP runtime seams
#include "nccl_fakes.h"          // reusable nccl* seams
#include "nccl_stubs.h"          // core/lifecycle functors; nccl_stubs.cc is linked into this binary too
#include "proxy_fakes.h"         // src/proxy.cc
#include "rccl_wrap_fakes.h"     // src/rccl_wrap.cc
#include "recorder_fakes.h"      // src/recorder.cc
#include "register_stubs.h"      // src/register/coll_reg.cc (hookable entry only)
#include "strongstream_stubs.h"  // src/misc/strongstream.cc
#include "sym_kernels_fakes.h"   // src/sym_kernels.cc
#include "transport_stubs.h"     // src/transport/net.cc (rcclUseAinic)
#include "tuning_fakes.h"        // src/graph/tuning.cc

// TRAP: NCCL_PROTO is latched in a static at updateCollCostTable:2527 and read via raw getenv at topoGetAlgoInfo:2687.
// TRAP: a test that fills the work FIFO must fail the poll or set abortFlag, or waitWorkFifoAvailable spins and hangs.

// taskAppend's AllToAll tests make taskAppend live in this binary. The rccl_wrap.cc entry points it reaches
// are defined in enqueue_test_deps.cc (only this binary compiles enqueue.cc without rccl_wrap.cc). The
// in-place AllToAll helpers, forced CE-Scratch and NCCL_ALGO are seams; the rest return the default-off
// answer or fail loudly.
#include <functional>
#include "rccl_common.h"  // rcclA2aInPlaceSite
struct ncclComm;
extern std::function<bool(const ncclComm*, const void*, const void*)> g_rcclAlltoAllCeScratchInPlace;
extern std::function<ncclResult_t(const ncclComm*, const void*, const void*, size_t)> g_rcclAlltoAllInPlaceGuard;
extern int g_rcclAlltoAllInPlaceGuardCalls;
// What the last guard call was told (site, capturing, registeredCe); see rccl_common.h.
extern rcclA2aInPlaceSite g_rcclAlltoAllInPlaceGuardSite;
extern bool g_rcclAlltoAllInPlaceGuardCapturing;
extern bool g_rcclAlltoAllInPlaceGuardRegisteredCe;
// rcclAlltoAllRegisteredCeRefused: this value for in-place calls, false out of place (gfx1250 when true).
extern bool g_rcclAlltoAllRegisteredCeRefused;
extern bool g_rcclForceCeScratchEnabled;  // rcclForceCeScratchEnabled
extern bool g_rcclNcclAlgoEnvIsSet;       // rcclNcclAlgoEnvIsSet (NCCL_ALGO set)

void ResetEnqueueTestDeps();

#endif  // RCCL_TEST_HOST_ENQUEUE_TEST_DEPS_H_
