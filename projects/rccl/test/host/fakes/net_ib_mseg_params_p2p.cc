/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// net_ib's parameter definitions, which IB-MSEG declares extern; the fork binaries link no net_ib unit.
// Hand-copied from src/transport/net_ib/*.cc; NetIbMsegParamCopies (connect-test.cc) fails when they drift.

#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

// src/transport/net_ib/p2p.cc
NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", -2);
NCCL_PARAM(IbReceiverSideMatchingScheme, "IB_RECEIVER_SIDE_MATCHING_SCHEME", -2);
RCCL_PARAM(IbSplitDataThreshold, "IB_SPLIT_DATA_THRESHOLD", 128);
RCCL_PARAM(IbGdrFlushGpuMemNoRelaxedOrdering, "GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", 1);
