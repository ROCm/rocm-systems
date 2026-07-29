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
// SCOPE WARNING -- hook breadth.
//
// The macro shims in p2p-test.cc that route through these hooks
// (`hipMemGetAddressRange`, `hipIpcGetMemHandle`, `ncclCudaCallocAsync`,
// `ncclCudaMemcpyAsync`) take effect for **every** call site of those
// symbols inside the `#include`d p2p.cc -- not just inside
// `ipcRegisterBuffer`. When microtests are added for other functions in
// the same TU (e.g. `ipcDeregisterBuffer`, `ncclIpcLocalRegisterBuffer`,
// `ncclIpcGraphRegisterBuffer`), those new tests will silently inherit
// the default hooks installed here.
//
// Practical consequences:
//   - Setting a hook's default to "return failure loudly" is safe and
//     desirable -- it surfaces unexpected calls from any new unit
//     under test, not just the original one.
//   - Setting a hook's default to a benign success that *happens* to
//     match `ipcRegisterBuffer`'s usage may be wrong for the next
//     unit under test. If you add a new test family, revisit each
//     default and check whether it still makes sense.
//   - When a new test family needs a *different* default, the
//     remedy is to overwrite the hook in its fixture's SetUp() (the
//     ScopedHook helper in p2p-test.cc is the standard pattern), not
//     to change the default here.

#pragma once

#include <cstddef>
#include <functional>

#include "nccl.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "nccl_fakes.h"
#include "hip_fakes.h"

// FakeCudaCallocAsync / FakeCudaMemcpyAsync: targets of the macro shims that
// p2p-test.cc installs over ncclCudaCallocAsync / ncclCudaMemcpyAsync so the
// header-only templates in alloc.h don't reach real HIP runtime. Default
// behaviour is what an honest GPU emulator would do: calloc heap memory and
// memcpy bytes between host pointers. Tests that need to inject failure or
// observe the calls override these hooks; ResetP2pFakes() frees any
// outstanding fake allocations.
extern std::function<ncclResult_t(void** ptr, std::size_t nbytes, hipStream_t)>
    g_fakeCudaCallocAsync;
extern std::function<ncclResult_t(void* dst, void* src, std::size_t nbytes, hipStream_t)>
    g_fakeCudaMemcpyAsync;

// Restore every hook owned by this file (and, transitively, the nccl* and
// HIP hooks) to its default. Call from fixture TearDown().
void ResetP2pFakes();
