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

// Include canonical fake-owner headers here so wrap-test.cc consumes their
// seams without redeclaring them. comm.h/enqueue.h provide the remaining
// ncclComm, task, datatype, and algorithm types used below.
#include "ce_fakes.h"
#include "comm.h"
#include "dev_runtime_fakes.h"
#include "enqueue.h"
#include "nccl.h"
#include "strongstream_stubs.h"
#include "sym_kernels_fakes.h"
#include "transport_stubs.h"
#include "tuning_fakes.h"

// The env-var test-control API (SetMicroEnv/SetMicroEnvAbsent/ClearMicroEnv)
// is declared by fakes/env_fakes.h, the shared owner of getenv interposition
// for every microtest binary -- include that directly rather than this file
// for those. See MICROTEST_README.md's "Where a fake belongs".

// getFirmwareVersion()'s sole dependency, made settable so a test can script
// a canned firmware response or a failure.
extern std::function<ncclResult_t(uint32_t, uint64_t*)> g_amdSmiGetFirmwareVersion;

// External parameter accessors made settable so tests can drive their
// respective guards independently.
extern std::function<int64_t()> g_paramLaunchOrderImplicit;
extern std::function<int64_t()> g_paramForceCe;

// Deliberate reset opt-out: every mutable hook in this file is installed via
// ScopedHook (or exercised in a forked RUN_ISOLATED_TEST child), so state is
// restored at scope exit or discarded with the child process. This suite has
// no persistent hook state for a ResetWrapFakes() entry point to clean up.

// ---------------------------------------------------------------------------
// Dispatcher seams: drive rcclSelectAllReduce/AllGather/ReduceScatter,
// rcclHierarchicalAlgoInfo, rcclGetAlgoInfo, rcclGetCollImplInfo, and
// rcclSymkQuery/rcclSymKGetInfo's deep path. See wrap_fakes.cc for each
// default's rationale.
// ---------------------------------------------------------------------------

extern std::function<bool(struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t, const void*, void*)>
    g_isSymmetricKernelRequested;
extern std::function<bool(const struct ncclComm*, size_t, ncclDataType_t, bool, bool)> g_allReduceShouldTakeDdaPath;

extern std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, int, int, int, ncclSimInfo_t*)>
    g_getAlgoInfo;
extern std::function<int(struct ncclComm*, ncclFunc_t, size_t, ncclDataType_t, int, int)> g_kernelPackedChannels;
extern std::function<ncclResult_t(const ncclComm_t, int*)> g_commCount;

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
