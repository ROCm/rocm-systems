/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// nccl-fastsocket's compat.h only defines the v5 net API below NCCL 2.12, expecting
// the installed headers to provide it from then on; NCCL 2.30 removed v5, so nothing
// declares it. Reuse the example plugin's self-contained header set rather than
// depending on generated nccl.h or the RCCL src tree.

#ifndef RCCL_FASTSOCKET_COMPAT_V5_H_
#define RCCL_FASTSOCKET_COMPAT_V5_H_

// net.h brings err.h (ncclResult_t), common.h, net_device.h and all net_v*.h
// including net_v5.h — everything nccl-fastsocket needs.
#include "net.h"

#endif // RCCL_FASTSOCKET_COMPAT_V5_H_
