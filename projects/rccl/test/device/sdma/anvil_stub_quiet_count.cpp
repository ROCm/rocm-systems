/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Single definition of the stub quiet counter declared in anvil_device.hpp.
// Keep this out of the header: IPC and Suite H both include gin_anvil_sdma.h.
// rccl-UnitTestsFixtures device-links this TU under ENABLE_ROCSHMEM_GIN.

#include <hip/hip_runtime.h>

namespace sdma_anvil {
__device__ unsigned long long g_sdmaStubQuietCount = 0;
}
