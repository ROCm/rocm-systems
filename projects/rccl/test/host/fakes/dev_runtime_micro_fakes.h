/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Controllable seams exposed by fakes/dev_runtime_micro_fakes.cc: the non-HIP
 * externs dev_runtime.cc reaches -- shadow pool, VA space, address map, GIN,
 * RMA proxy, bootstrap. The HIP surface it drives lives in fakes/hip_fakes.h,
 * shared with the other micro suites, and the nccl-wide symbols in
 * fakes/nccl_fakes.h.
 *
 * Distinct from fakes/dev_runtime_fakes.{h,cc}, which fakes two dev_runtime.cc
 * predicates for binaries that do NOT compile that file. This one belongs to
 * rccl-UnitTestsMicro, which compiles dev_runtime.cc itself (dev-runtime-test.cc);
 * the two must never be linked into the same binary.
 *
 * Each hook defaults to the success behaviour the rest of the suite relies on.
 * A test that needs a call to fail installs its own via ScopedHook
 * (test/host/ScopedHook.h), which restores the previous one on scope exit.
 *************************************************************************/

#ifndef RCCL_TEST_HOST_FAKES_DEV_RUNTIME_MICRO_FAKES_H_
#define RCCL_TEST_HOST_FAKES_DEV_RUNTIME_MICRO_FAKES_H_

#include "nccl.h"          // ncclResult_t, for the proxy seam below
#include "gin/gin_host.h"  // NCCL_GIN_MAX_CONNECTIONS, ncclGinWindow_t

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>








// Team shape seen by ncclDevrWorldToLsaRank's symmetric arm. Default to the
// comm's own contiguous stride-1 team; override for strided/offset layouts.
extern std::function<ncclTeam_t(ncclComm_t)> g_devrTeamWorld;

extern std::function<ncclResult_t(void*, int, int, int)> g_devrBootstrapBarrier;
extern std::function<ncclResult_t(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t, void*)>
    g_devrIntruAddressMapInsert;

extern std::function<ncclResult_t(const ncclComm_t, void*, size_t, void**)> g_devrNcclCommRegister;
extern std::function<ncclResult_t(const ncclComm_t, void*)> g_devrNcclCommDeregister;
extern std::function<ncclResult_t(ncclComm_t, ncclWindow_t)> g_devrNcclCommWindowDeregister;

extern std::function<ncclResult_t(void*, void*, int)> g_devrBootstrapAllGather;
extern std::function<ncclResult_t(void*, int*, int, int, int)> g_devrBootstrapIntraNodeBarrier;
extern std::function<ncclResult_t(void*, int*, int, int, void*, int)> g_devrBootstrapIntraNodeAllGather;

extern std::function<ncclResult_t(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS],
                                  ncclGinWindow_t[NCCL_GIN_MAX_CONNECTIONS], int, bool, int)>
    g_devrGinRegister;
extern std::function<ncclResult_t(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS])> g_devrGinDeregister;

extern std::function<ncclResult_t(struct ncclSpace*, int64_t, int64_t, int, int64_t*)> g_devrSpaceAlloc;
extern std::function<ncclResult_t(struct ncclSpace*, int64_t, int64_t)> g_devrSpaceFree;
extern std::function<ncclResult_t(struct ncclDevrMemory*, int)> g_devrPopulateSegmentSizes;

extern std::function<ncclResult_t(struct ncclShadowPool*, size_t, void**, void**, hipStream_t)> g_devrShadowPoolAlloc;
extern std::function<ncclResult_t(struct ncclShadowPool*, void*, hipStream_t)> g_devrShadowPoolFree;
extern std::function<ncclResult_t(struct ncclShadowPool*, void*, void**)> g_devrShadowPoolToHost;

// Resolves a device window to its host record; the default finds nothing.
extern std::function<ncclResult_t(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t, void**)>
    g_devrIntruAddressMapFind;
extern std::function<ncclResult_t(struct ncclDevrState*, struct ncclDevrMemory*, hipStream_t,
                                  struct ncclSegmentWindow**)>
    g_devrAllocAndPopulateSegmentWindows;
extern std::function<ncclResult_t(struct ncclDevrMemory*, struct ncclComm*)> g_devrVerifySegmentLayouts;
extern std::function<ncclResult_t(struct ncclDevrMemory*)> g_devrBuildGinSegmentInfos;

// The CFT seams 2.31 added: the two team accessors, the two sizes
// ncclDevrInitOnce caches, and whether the RMA proxy is in play. Defaults
// describe a comm without CFT and without the proxy; see dev_runtime_micro_fakes.cc.
extern std::function<ncclTeam_t(ncclComm_t, ncclCftTeamMode_t)> g_devrTeamCft;
extern std::function<ncclTeam_t(ncclComm_t)> g_devrTeamCftMultimem;
extern std::function<int(struct ncclComm*)> g_devrComputeCftSize;
extern std::function<int(struct ncclComm*)> g_devrComputeCftMcSize;

extern std::function<ncclResult_t(struct ncclComm*)> g_devrRmaProxyConnectOnce;
extern std::function<ncclResult_t(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS])>
    g_devrRmaProxyRegister;
extern std::function<ncclResult_t(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS])> g_devrRmaProxyDeregister;


// Backs every NCCL_PARAM in the unit under test. dev-runtime-test.cc redefines
// the macro to call this instead of param.h's caching body, so a param's value
// can differ between tests; the default returns the param's own default.
// Takes the bare env name (no "NCCL_" prefix) and that default.

// Restore every seam above to its default. Call from a fixture TearDown so a
// test cannot leak behaviour into the next one.
void ResetDevRuntimeMicroFakes();

#endif  // RCCL_TEST_HOST_FAKES_DEV_RUNTIME_MICRO_FAKES_H_
