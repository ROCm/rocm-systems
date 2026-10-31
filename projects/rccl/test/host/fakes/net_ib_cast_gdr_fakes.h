/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Controllable seams for src/transport/net_ib_cast/gdr.cc. Every probe defaults
// to ncclSystemError, i.e. "not supported".

#ifndef RCCL_TEST_HOST_NET_IB_CAST_GDR_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_GDR_FAKES_H_

#include <functional>

#include "nccl.h"

extern std::function<ncclResult_t()> g_ibCastGdrSupport;
extern std::function<ncclResult_t()> g_ibCastPeerMemSupport;
extern std::function<ncclResult_t(int /*dev*/)> g_ibCastDmaBufSupport;

void ResetNetIbCastGdrFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_GDR_FAKES_H_
