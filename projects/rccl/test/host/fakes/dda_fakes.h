/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_DDA_FAKES_H_
#define RCCL_TEST_HOST_DDA_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "nccl.h"

struct ncclComm;

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

extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)> g_allToAllDdaIpcEligible;
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)> g_allToAllDdaFabricEligible;
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)> g_allToAllDdaFabricLLEligible;
extern std::function<bool(ncclComm*, const void*, void*, size_t, ncclDataType_t)> g_allToAllDdaFabricLL128Eligible;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_allToAllDdaIpcBlocks;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_allToAllDdaFabricBlocks;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_allToAllDdaFabricLLBlocks;
extern std::function<uint32_t(ncclComm*, size_t, ncclDataType_t)> g_allToAllDdaFabricLL128Blocks;

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

// The launchers. Fail loud until a test installs them.
using DdaLaunchHook = std::function<ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclComm*, hipStream_t)>;
using DdaReduceLaunchHook =
    std::function<ncclResult_t(const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*, hipStream_t)>;
extern DdaLaunchHook g_allGatherDdaIpc, g_allGatherDdaFabric, g_allGatherDdaFabricLL, g_allGatherDdaFabricLL128;
extern DdaLaunchHook g_allToAllDdaIpc, g_allToAllDdaFabric, g_allToAllDdaFabricLL, g_allToAllDdaFabricLL128;
extern DdaReduceLaunchHook g_allReduceDdaIpc, g_allReduceDdaFabric, g_allReduceDdaFabricLL,
    g_allReduceDdaFabricLL128;
extern DdaReduceLaunchHook g_reduceScatterDdaIpc, g_reduceScatterDdaFabric, g_reduceScatterDdaFabricLL,
    g_reduceScatterDdaFabricLL128;

#endif  // RCCL_TEST_HOST_DDA_FAKES_H_
