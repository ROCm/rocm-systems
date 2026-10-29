/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Routes a unit's calls into the src/transport/net_ib/connect.cc entry points net.cc uses to the micro_*
// trampolines in transport_net_ib_connect_net_fakes.h, for a binary that also compiles the real connect.cc.
// Include it before the #include <UNIT>_CC_PATH and transport_net_ib_connect_net_seam_undef.h after it.
// No include guard on purpose.

// The headers declaring the renamed names, so a rename never rewrites a declaration.
#include "net.h"
#include "transport_net_ib_connect_net_fakes.h"

#define rcclNetP2pPolicy micro_rcclNetP2pPolicy
#define ncclIbSetListenGdrFlushFlag micro_ncclIbSetListenGdrFlushFlag
#define ncclIbGetGpuFlushFlag micro_ncclIbGetGpuFlushFlag
#define ncclIbGetGpuFlushFlagSeq micro_ncclIbGetGpuFlushFlagSeq
#define ncclIbSetGpuFlushFlagHint micro_ncclIbSetGpuFlushFlagHint
