/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for the ops-table fault shims owned by
// src/transport/net_ib_cast/net_ib_ops_fault.cc. Defaults fail loudly.

#ifndef RCCL_TEST_HOST_FAKES_NET_IB_OPS_FAULT_FAKES_H_
#define RCCL_TEST_HOST_FAKES_NET_IB_OPS_FAULT_FAKES_H_

#include <functional>

#include "nccl.h"

struct ibv_context;

extern std::function<ncclResult_t(struct ibv_context*)> g_ncclIbOpsFaultInstall;
extern std::function<ncclResult_t(struct ibv_context*)> g_ncclIbOpsFaultRemove;

void ResetNetIbOpsFaultFakes();

#endif  // RCCL_TEST_HOST_FAKES_NET_IB_OPS_FAULT_FAKES_H_
