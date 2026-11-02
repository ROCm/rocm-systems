/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Removes the renames installed by transport_p2p_reg_seam.h. No include guard on purpose.

#undef ncclP2pUsesMemcpy
#undef ncclIpcDeregBuffer
#undef ncclIpcGraphRegisterBuffer
#undef ncclIpcLocalRegisterBuffer
