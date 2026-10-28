/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Removes the renames installed by transport_net_ib_connect_net_seam.h. No include guard on purpose.

#undef ncclIbSetGpuFlushFlagHint
#undef ncclIbGetGpuFlushFlagSeq
#undef ncclIbGetGpuFlushFlag
#undef ncclIbSetListenGdrFlushFlag
#undef rcclNetP2pPolicy
