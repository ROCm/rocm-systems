/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TRANSPORT_NET_IB_RECOVERY_FAKES_H_
#define RCCL_TEST_HOST_TRANSPORT_NET_IB_RECOVERY_FAKES_H_

#include <functional>

#include "nccl.h"

extern std::function<ncclResult_t()> g_ncclIbPortRecoveryThreadStart;
extern std::function<ncclResult_t()> g_ncclIbPortRecoveryThreadStop;

void ResetTransportNetIbRecoveryFakes();

#endif  // RCCL_TEST_HOST_TRANSPORT_NET_IB_RECOVERY_FAKES_H_
