/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Routes a unit's calls into the src/transport/net_ib_cast/p2p.cc entry points to the micro_* trampolines in
// net_ib_cast_p2p_fakes.h, for a binary that also compiles the real p2p.cc. Include it before the
// #include <UNIT>_CC_PATH and net_ib_cast_p2p_seam_undef.h after it. No include guard on purpose.

// The headers declaring the renamed names, so a rename never rewrites a declaration.
#include "common_cast.h"
#include "net_ib_cast_p2p_fakes.h"

#define IbCastAddEvent micro_IbCastAddEvent
#define IbCastGetRequest micro_IbCastGetRequest
#define IbCastFreeRequest micro_IbCastFreeRequest
#define IbCastIsend micro_IbCastIsend
#define IbCastIrecv micro_IbCastIrecv
#define IbCastIflush micro_IbCastIflush
#define IbCastTest micro_IbCastTest
