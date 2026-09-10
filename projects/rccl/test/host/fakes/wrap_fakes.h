/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// NOT to be confused with fakes/rccl_wrap_fakes.h, which sits on the OPPOSITE
// side of the dependency graph:
//
//   rccl_wrap_fakes.h  fakes the symbols rccl_wrap.cc DEFINES, for targets
//                      that do not compile it (rccl-UnitTestsMicroEnqueue).
//   wrap_fakes.h       (this file) fakes the symbols rccl_wrap.cc DEPENDS ON,
//                      for rccl-UnitTestsMicro, which compiles the real file
//                      and tests it directly.
//
// The two never define the same symbol, which is why both can exist. If you
// arrived here by grep and wanted rccl_wrap.cc's own entry points, you want
// the other header. See MICROTEST_README.md's fakes-ownership table.

#ifndef RCCL_TEST_HOST_WRAP_FAKES_H_
#define RCCL_TEST_HOST_WRAP_FAKES_H_

#include <cstdint>
#include <functional>

#include "nccl.h"  // ncclResult_t
// comm.h transitively provides ncclComm, ncclTaskColl, ncclDevrWindow,
// ncclFunc_t (via sym_kernels.h -> nccl_common.h), ncclSymRegType_t,
// ncclSymkKernelId, and ncclCudaGraph/hipStream_t (via strongstream.h) --
// everything the seam declarations below need. enqueue.h adds
// ncclSimInfo_t and the NCCL_ALGO_*/NCCL_PROTO_* used by getAlgoInfo's
// default (see wrap_fakes.cc). Same two headers wrap_fakes.cc itself
// includes for the same reason.
#include "comm.h"
#include "enqueue.h"

// The env-var test-control API (SetMicroEnv/SetMicroEnvAbsent/ClearMicroEnv)
// is declared by fakes/env_fakes.h, the shared owner of getenv interposition
// for every microtest binary -- include that directly rather than this file
// for those. See MICROTEST_README.md's "Where a fake belongs".

// getFirmwareVersion()'s sole dependency, made settable so a test can script
// a canned firmware response or a failure.
extern std::function<ncclResult_t(uint32_t, uint64_t*)> g_amdSmiGetFirmwareVersion;

// rcclDdaEnabled's ncclParamLaunchOrderImplicit() dependency, made settable
// so a test can drive that disjunct of its 3-way disable guard independently.
extern std::function<int64_t()> g_paramLaunchOrderImplicit;

// ---------------------------------------------------------------------------
// Dispatcher seams: drive rcclSelectAllReduce/AllGather/ReduceScatter,
// rcclHierarchicalAlgoInfo, rcclGetAlgoInfo, rcclGetCollImplInfo, and
// rcclSymkQuery/rcclSymKGetInfo's deep path. See wrap_fakes.cc for each
// default's rationale.
// ---------------------------------------------------------------------------

extern std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, const void*, void*)>
    g_isSymmetricKernelRequested;
extern std::function<ncclResult_t(struct ncclCudaGraph*, hipStream_t, int)> g_cudaGetCapturingGraph;
extern std::function<ncclResult_t(struct ncclComm*, void const*, struct ncclDevrWindow**)> g_devrFindWindow;
extern std::function<bool(struct ncclDevrWindow*)> g_devrWindowHasSysmemSegment;
extern std::function<ncclResult_t(struct ncclDevrWindow*, struct ncclDevrWindow*, ncclSymRegType_t*)>
    g_getSymRegType;

extern std::function<ncclResult_t(struct ncclComm*)> g_symkInitOnce;
extern std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t)> g_symkAvailable;
extern std::function<ncclResult_t(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, size_t, int,
                                  ncclSymRegType_t, float*, ncclSymkKernelId*, int*, int*, bool*)>
    g_symkPickKernel;
extern std::function<bool(int)> g_symkKernelIdIsLL;

extern std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t)> g_ceAvailable;
extern std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, ncclSymRegType_t)>
    g_ceScratchAvailable;
extern std::function<int(ncclDataType_t, size_t)> g_ceLocalReduceBlocks;

extern std::function<bool(const struct ncclComm*, size_t, ncclDataType_t, bool, bool)> g_allReduceShouldTakeDdaPath;

extern std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, int, int, int, ncclSimInfo_t*)>
    g_getAlgoInfo;
extern std::function<int(struct ncclComm*, ncclFunc_t, size_t, ncclDataType_t, int, int)> g_kernelPackedChannels;
extern std::function<ncclResult_t(const ncclComm_t, int*)> g_commCount;
extern std::function<bool()> g_useAinic;
extern std::function<int(struct ncclComm*)> g_pxnDisable;

// --- Per-collective DDA eligibility/blocks (24 hooks total) ---
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t)>
    g_allReduceDdaIpcEligible;
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t)>
    g_allReduceDdaFabricEligible;
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t)>
    g_allReduceDdaFabricLLEligible;
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t)>
    g_allReduceDdaFabricLL128Eligible;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_allReduceDdaIpcBlocks;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_allReduceDdaFabricBlocks;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_allReduceDdaFabricLLBlocks;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_allReduceDdaFabricLL128Blocks;

extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)> g_allGatherDdaIpcEligible;
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)> g_allGatherDdaFabricEligible;
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)> g_allGatherDdaFabricLLEligible;
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)>
    g_allGatherDdaFabricLL128Eligible;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_allGatherDdaIpcBlocks;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_allGatherDdaFabricBlocks;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_allGatherDdaFabricLLBlocks;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_allGatherDdaFabricLL128Blocks;

extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t)>
    g_reduceScatterDdaIpcEligible;
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t)>
    g_reduceScatterDdaFabricEligible;
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t)>
    g_reduceScatterDdaFabricLLEligible;
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t, ncclRedOp_t)>
    g_reduceScatterDdaFabricLL128Eligible;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_reduceScatterDdaIpcBlocks;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_reduceScatterDdaFabricBlocks;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_reduceScatterDdaFabricLLBlocks;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_reduceScatterDdaFabricLL128Blocks;

#endif  // RCCL_TEST_HOST_WRAP_FAKES_H_
