/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable HIP seams for the micro-test fakes layer.
//
// The micro-test binary does not link the real HIP runtime (see
// hip_fakes.cc for why). These std::function hooks are the handful of HIP
// calls the unit-under-test drives on its happy path; the macro shims in
// p2p-test.cc route the p2p.cc call sites through them. Tests install
// per-test behaviour by overwriting a hook (see the ScopedHook helper in
// p2p-test.cc) and ResetHipFakes() restores the defaults.
//
// Defaults return hipErrorInvalidValue so any call site a test doesn't
// explicitly opt into surfaces the unexpected call as
// ncclUnhandledCudaError via CUCHECKGOTO.

#pragma once

#include <cstddef>
#include <functional>

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

// hipMemGetAddressRange / hipIpcGetMemHandle: HIP runtime entry points
// reached from ipcRegisterBuffer's fresh-registration arm.
extern std::function<hipError_t(hipDeviceptr_t* /*pbase*/, std::size_t* /*psize*/,
                                hipDeviceptr_t /*dptr*/)>
    g_hipMemGetAddressRange;
extern std::function<hipError_t(hipIpcMemHandle_t* /*handle*/, void* /*devPtr*/)>
    g_hipIpcGetMemHandle;

// hipMemRetainAllocationHandle / hipMemExportToShareableHandle /
// hipMemRelease: the three HIP runtime entry points the cuMem*-export arm
// of ipcRegisterBuffer calls.
//
// Defaults return hipErrorInvalidValue so unexpected call sites surface
// via CUCHECKGOTO; tests that want a happy path install a hook that
// returns hipSuccess (and, for Retain, hands back a sentinel handle).
extern std::function<hipError_t(hipMemGenericAllocationHandle_t* /*handle*/,
                                void* /*addr*/)>
    g_hipMemRetainAllocationHandle;
extern std::function<hipError_t(void* /*shareableHandle*/,
                                hipMemGenericAllocationHandle_t /*handle*/,
                                hipMemAllocationHandleType /*handleType*/,
                                unsigned long long /*flags*/)>
    g_hipMemExportToShareableHandle;
extern std::function<hipError_t(hipMemGenericAllocationHandle_t /*handle*/)>
    g_hipMemRelease;

// Restore the HIP controllable seams above to their defaults. Called by
// ResetP2pFakes(); exposed for tests that only touch HIP hooks.
void ResetHipFakes();
