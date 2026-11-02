/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/qp_sharing.cc fakes. See net_ib_cast_qp_sharing_fakes.h.

#include "net_ib_cast_qp_sharing_fakes.h"

#include "fakes/nccl_fakes.h"
#include "param_redirect.h"
#include "qp_sharing.h"

// Same lines as qp_sharing.cc; test/host/CMakeLists.txt pins both copies.
RCCL_PARAM(IbCastQpSharingEnable, "IB_QP_SHARING_ENABLE", 0);
RCCL_PARAM(IbCastCommNGroups, "IB_COMM_NGROUPS", 4);
RCCL_PARAM(IbCastQpSharingValidatePool, "IB_QP_SHARING_VALIDATE_POOL", 0);

bool IbCastQpSharingGlobalEnable = false;

int g_ibCastValidateSharedQpPoolCalls = 0;
void IbCastValidateSharedQpPool(void) { g_ibCastValidateSharedQpPoolCalls++; }

void ResetNetIbCastQpSharingFakes() {
  IbCastQpSharingGlobalEnable = false;
  g_ibCastValidateSharedQpPoolCalls = 0;
}
