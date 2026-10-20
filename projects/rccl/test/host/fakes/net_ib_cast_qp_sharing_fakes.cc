/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/qp_sharing.cc fakes. See net_ib_cast_qp_sharing_fakes.h.

#include "net_ib_cast_qp_sharing_fakes.h"

#include "qp_sharing.h"

bool IbCastQpSharingGlobalEnable = false;

void ResetNetIbCastQpSharingFakes() { IbCastQpSharingGlobalEnable = false; }
