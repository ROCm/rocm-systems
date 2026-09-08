/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the DDA AllToAll dispatch surface that
// src/collectives.cc (ncclAlltoAll_impl) calls. The real DDA .cu files are GPU
// code and are not linked into the host-only microtest binary, so these host
// doubles both satisfy the link and let a fixture drive the selection decision:
//   - rcclDdaEnabled / *Eligible booleans decide which branch ncclAlltoAll_impl
//     takes;
//   - the dispatch entry points (ncclAllToAllDdaIpc / *Fabric*) record that they
//     were reached, so a test can assert whether DDA actually serviced the call.
//
// Every seam defaults to "DDA not enabled / not eligible" so a test that forgets
// to opt in gets the fall-through (enqueue) path rather than a surprise DDA
// dispatch. Reset via ResetDdaAlltoAllFakes() from the fixture.

#ifndef RCCL_TEST_DDA_ALLTOALL_FAKES_H_
#define RCCL_TEST_DDA_ALLTOALL_FAKES_H_

#include "nccl.h"

// Decision seams (read by ncclAlltoAll_impl before dispatch).
extern bool g_ddaEnabled;             // rcclDdaEnabled(...)
extern bool g_ddaIpcEligible;         // ncclAllToAllDdaIpcEligible(...)
extern bool g_ddaFabricEligible;      // ncclAllToAllDdaFabricEligible(...)
extern bool g_ddaFabricLLEligible;    // ncclAllToAllDdaFabricLLEligible(...)
extern bool g_ddaFabricLL128Eligible; // ncclAllToAllDdaFabricLL128Eligible(...)

// Dispatch observation: which DDA entry point (if any) actually ran.
extern bool g_ddaIpcDispatched;
extern bool g_ddaFabricDispatched;
extern bool g_ddaFabricLLDispatched;
extern bool g_ddaFabricLL128Dispatched;

// True if any DDA dispatch entry point ran.
bool DdaAlltoAllDispatched();

void ResetDdaAlltoAllFakes();

#endif // RCCL_TEST_DDA_ALLTOALL_FAKES_H_
