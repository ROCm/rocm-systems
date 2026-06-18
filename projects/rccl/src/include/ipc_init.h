/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Declarations for DDA IPC comm setup / teardown (see ipc_init.cu).
 * Safe to include from host C++ (.cc); for implementation details see
 * ipc_init_detail.cuh (CUDA/HIP device-related; use from .cu only).
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include "nccl.h"

struct ncclComm;

ncclResult_t ncclDdaIpcCommInit(struct ncclComm* comm);
ncclResult_t ncclDdaIpcCommFini(struct ncclComm* comm);
