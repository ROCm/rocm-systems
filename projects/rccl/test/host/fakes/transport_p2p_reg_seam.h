/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Routes a unit's calls into the src/transport/p2p.cc registration entry points to the micro_* trampolines in
// transport_p2p_reg_fakes.h, for a binary that also compiles the real p2p.cc. Include it immediately before the
// #include <UNIT>_CC_PATH and transport_p2p_reg_seam_undef.h straight after. No include guard on purpose.

// The headers declaring the renamed names, so a rename never rewrites a declaration.
#include "p2p.h"
#include "transport.h"
#include "transport_p2p_reg_fakes.h"

#define ncclIpcLocalRegisterBuffer micro_ncclIpcLocalRegisterBuffer
#define ncclIpcGraphRegisterBuffer micro_ncclIpcGraphRegisterBuffer
#define ncclIpcDeregBuffer micro_ncclIpcDeregBuffer
#define ncclP2pUsesMemcpy micro_ncclP2pUsesMemcpy
