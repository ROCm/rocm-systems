/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Routes a unit's calls into src/transport/net_ib_cast/qp_sharing.cc's IbCastValidateSharedQpPool to the counting
// stand-in in net_ib_cast_qp_sharing_fakes.h, for a binary that also compiles the real qp_sharing.cc. Include it
// before the #include <UNIT>_CC_PATH and net_ib_cast_qp_sharing_seam_undef.h after it. No include guard on purpose.

// The headers declaring the renamed names, so a rename never rewrites a declaration.
#include "qp_sharing.h"
#include "net_ib_cast_qp_sharing_fakes.h"

#define IbCastValidateSharedQpPool micro_IbCastValidateSharedQpPool
