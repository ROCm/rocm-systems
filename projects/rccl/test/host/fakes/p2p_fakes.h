/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Per-test controllable seams for the fakes layer.
//
// See README.md, "Adding more controllable seams". Tests install per-test
// behaviour by overwriting one of these std::function hooks in a fixture's
// SetUp(), and ResetP2pFakes() in TearDown() restores defaults so tests
// don't contaminate each other.
//
// SCOPE WARNING: the macro shims in p2p-test.cc route EVERY call site of these
// symbols inside the #included p2p.cc through these hooks (not just
// ipcRegisterBuffer), so new tests in the same TU inherit these defaults.
// Prefer defaults that fail loudly, and override per-test in a fixture's
// SetUp() (via ScopedHook) rather than changing a default here.

#ifndef RCCL_TEST_HOST_P2P_FAKES_H_
#define RCCL_TEST_HOST_P2P_FAKES_H_

#include <cstddef>
#include <functional>

#include "nccl.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "nccl_fakes.h"
#include "hip_fakes.h"

extern std::function<ncclResult_t(void** ptr, std::size_t nbytes, hipStream_t)>
    g_fakeCudaCallocAsync;
extern std::function<ncclResult_t(void* dst, void* src, std::size_t nbytes, hipStream_t)>
    g_fakeCudaMemcpyAsync;

// Restore every hook owned by this file (and, transitively, the nccl* and
// HIP hooks) to its default. Call from fixture TearDown().
void ResetP2pFakes();

#endif  // RCCL_TEST_HOST_P2P_FAKES_H_
