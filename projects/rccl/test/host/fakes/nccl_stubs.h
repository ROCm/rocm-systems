/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams nccl_stubs.cc OWNS, declared once so a signature change is a compile error rather than a link mismatch.

#ifndef RCCL_TEST_HOST_NCCL_STUBS_H_
#define RCCL_TEST_HOST_NCCL_STUBS_H_

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "nccl.h"

struct ncclAsyncJob;
struct ncclComm;

// src/group.cc:39. The default reproduces the real ncclGroupDepth == 0 arm: run func, undo on failure, destruct.
extern std::function<ncclResult_t(struct ncclAsyncJob*, ncclResult_t (*)(struct ncclAsyncJob*),
                                  void (*)(struct ncclAsyncJob*), void (*)(void*), struct ncclComm*)>
    g_ncclAsyncLaunch;

// src/init.cc's device bringup reaches src/enqueue.cc through this. Fail-loud by default; script it to reach past it.
#ifndef RCCL_STUBS_OMIT_ncclInitKernelsForDevice
extern std::function<ncclResult_t(int /*cudaArch*/, int /*maxSharedMem*/, size_t* /*maxStackSize*/)>
    g_ncclInitKernelsForDevice;
#endif

// src/misc/coll_trace.cc: comm teardown tears the trace ring down through this.
extern std::function<ncclResult_t(struct ncclComm*)> g_collTraceDestroy;

// src/plugin/tuner.cc: commCleanup unloads the tuner plugin through this.
extern std::function<ncclResult_t(struct ncclComm*)> g_ncclTunerPluginUnload;

// src/misc/mem_manager.cc: commFree releases the single-node size arrays here.
extern std::function<ncclResult_t(void*)> g_ncclMemFree;

// src/symmetric.cc: commFree tears down symmetric-memory resources here.
extern std::function<ncclResult_t(struct ncclComm*)> g_ncclSymkFinalize;

// The public entry point commFree recurses through for hierarchical sub-communicators.
extern std::function<ncclResult_t(ncclComm_t)> g_ncclCommDestroy;

// src/channel.cc: the fake initChannel does NOT allocate ring->userRanks/rankToIndex like the real one;
// callers must supply storage.
extern ncclResult_t g_initChannelResult;
extern int g_initChannelLastId;

// commCleanup() ordering oracle (init.cc:3696). The block is pure ordering + error propagation, so the
// oracle is the call-order log, not any return code. Every teardown step appends its own name in call
// order: "tunerFinalize" (pushed by the test's own ncclTuner_t::finalize), "tunerUnload", and "commFree"
// -- the last pushed by the src/ce_coll.cc ncclCeFinalize fake, which is commFree's FIRST NCCLCHECK and
// therefore marks commFree entry. g_ncclCeFinalizeResult is also the only knob that makes commFree fail;
// note a failing commFree bails before free(comm), so such a test must free it itself.
extern std::vector<std::string> g_cleanupCallOrder;
extern ncclResult_t g_ncclCeFinalizeResult;
// src/plugin/tuner.cc: the comm commCleanup forwarded to ncclTunerPluginUnload.
extern struct ncclComm* g_ncclTunerPluginUnloadLastComm;

// src/mnnvl.cc. The CALL COUNTER, not the result, is the oracle for the :1503-1509 enable/auto/disable logic.
extern ncclResult_t g_ncclMnnvlCheckResult;
extern int g_ncclMnnvlCheckCalls;

// librocm-core's getROCmVersion. Default 1 is != VerSuccess(0), so showVersion()'s runtime-ROCm block is skipped.
extern int g_getROCmVersionResult;
extern unsigned int g_rocmVersionMajor;
extern unsigned int g_rocmVersionMinor;
extern unsigned int g_rocmVersionPatch;

void ResetNcclStubs();

#endif  // RCCL_TEST_HOST_NCCL_STUBS_H_
