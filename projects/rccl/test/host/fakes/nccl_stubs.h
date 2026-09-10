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

void ResetNcclStubs();

#endif  // RCCL_TEST_HOST_NCCL_STUBS_H_
