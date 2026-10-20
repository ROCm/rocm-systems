/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// State owned by src/transport/net_ib_cast/qp_sharing.cc, for binaries that do not compile it.
// IbCastQpSharingGlobalEnable itself is declared by qp_sharing.h and starts false, as in production.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_QP_SHARING_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_QP_SHARING_FAKES_H_

void ResetNetIbCastQpSharingFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_QP_SHARING_FAKES_H_
