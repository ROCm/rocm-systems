/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Declarations for DDA IPC comm setup / teardown (see ipc_init.cu).
 * Safe to include from host C++ (.cc); for implementation details see
 * dda_init_detail.h (CUDA/HIP device-related; use from .cu only).
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "nccl.h"

struct ncclComm;

ncclResult_t ncclDdaIpcCommInit(struct ncclComm* comm);
// Runs the setup ncclDdaIpcCommInit deferred, once every rank votes ready; a no-op when nothing
// is deferred. Collective: call only from rank-invariant points (collective type, count, datatype,
// op and comm config, never buffer addresses). capturing: the caller's stream is capturing.
ncclResult_t ncclDdaIpcEnsureInit(struct ncclComm* comm, bool capturing);
ncclResult_t ncclDdaIpcCommClosePeerMappings(struct ncclComm* comm);
ncclResult_t ncclDdaIpcCommFini(struct ncclComm* comm);
