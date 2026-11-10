/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Externals src/mem_manager.cc reaches, for rccl-UnitTestsMicroMemManager, which compiles it. Distinct from
// mem_manager_fakes.cc, which fakes mem_manager.cc itself. HIP comes from hip_fakes.cc and bootstrap from
// bootstrap_stubs.cc. nccl_fakes.cc cannot link (it defines ncclMemTrack and ncclMemTrackImportFromPeer, which
// mem_manager.cc defines), so its logging and g_loadParam are here too.

#ifndef RCCL_TEST_HOST_MEM_MANAGER_MICRO_FAKES_H_
#define RCCL_TEST_HOST_MEM_MANAGER_MICRO_FAKES_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "comm.h"
#include "group.h"
#include "ipcsocket.h"

extern std::function<int64_t(const char*, int64_t)> g_loadParam;

// proxy, group and comm calls, each routed through a hook a test installs.
extern std::function<ncclResult_t(int /*rank*/, void* /*handle*/, ncclIpcFd*)> g_getFd;
extern std::function<ncclResult_t(ncclComm*)> g_commCheck;
extern std::function<ncclResult_t()> g_groupEnd;
extern std::function<ncclResult_t(ncclAsyncJob*, ncclResult_t (*)(ncclAsyncJob*), void (*)(void*))> g_mgmtEnqueue;
extern std::vector<int> g_closedFds;  // every fd ncclIpcFdClose was handed
extern int64_t g_rearch;               // NCCL_ENQUEUE_REARCH_ENABLE
extern ncclResult_t g_asyncError, g_asyncErrorResult;

#endif  // RCCL_TEST_HOST_MEM_MANAGER_MICRO_FAKES_H_
