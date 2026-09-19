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

// True when RCCL_DDA_NRANKS_RELAX=1 (allow 2..8-rank DDA IPC for every DDA IPC
// collective). Default 0. Read per process: must match on every rank of a comm.
bool ncclDdaNranksRelaxEnabled();

// Single source of truth for "is nRanks a supported single-node DDA IPC
// participant count": exactly kDdaNranks always; any 2..kDdaNranks when
// RCCL_DDA_NRANKS_RELAX=1. Shared by the DDA IPC collectives (AllReduce /
// AllGather / ReduceScatter / AllToAll) and by the comm-init gate, which must
// agree with them -- otherwise comm init allocates IPC resources that the
// eligibility gate then refuses.
bool ncclDdaIpcNranksSupported(int nRanks);

// True when nRanks sits in the range where ncclDdaIpcNranksSupported()'s answer
// actually depends on RCCL_DDA_NRANKS_RELAX: 2..kDdaNranks-1. Outside that range
// (exactly kDdaNranks, or too few/many ranks) the supported/unsupported answer
// is the same whether or not the knob is set, so a per-rank mismatch in the env
// var cannot cause ranks to disagree on whether to enter the DDA IPC path. Comm
// init uses this to decide whether checking RCCL_DDA_NRANKS_RELAX agreement
// across ranks is even relevant for a given communicator.
bool ncclDdaNranksRelaxConsensusMatters(int nRanks);

ncclResult_t ncclDdaIpcCommInit(struct ncclComm* comm);
ncclResult_t ncclDdaIpcCommFini(struct ncclComm* comm);
