/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Single definitions of the stub quiet counter and call log declared in anvil_device.hpp.
// Keep this out of the header: IPC and Suite H both include gin_anvil_sdma.h.
// rccl-UnitTestsFixtures device-links this TU under ENABLE_ROCSHMEM_GIN.

#include <hip/hip_runtime.h>

#include "anvil_device.hpp"

namespace sdma_anvil {
__device__ unsigned long long g_sdmaStubQuietCount = 0;
__device__ SdmaStubLog g_sdmaStubLog = {};
__device__ uint64_t* g_sdmaStubMarkDirtyOnQuiet = nullptr;
__device__ uint64_t g_sdmaStubMarkBitOnQuiet = 0;
}
