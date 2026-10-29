/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// A counting stand-in for qp_sharing.cc's IbCastValidateSharedQpPool, which init.cc calls at finalize. The binary
// also compiles the real qp_sharing.cc, so net_ib_cast_qp_sharing_seam.h renames that call to the stand-in below.

#ifndef RCCL_TEST_HOST_NET_IB_CAST_QP_SHARING_FAKES_H_
#define RCCL_TEST_HOST_NET_IB_CAST_QP_SHARING_FAKES_H_

extern int g_ibCastValidateSharedQpPoolCalls;

void micro_IbCastValidateSharedQpPool();

void ResetNetIbCastQpSharingFakes();

#endif  // RCCL_TEST_HOST_NET_IB_CAST_QP_SHARING_FAKES_H_
