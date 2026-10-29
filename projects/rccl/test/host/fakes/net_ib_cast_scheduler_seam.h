/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Routes a unit's calls into the src/transport/net_ib_cast/scheduler.cc entry points to the micro_* trampolines in
// net_ib_cast_scheduler_fakes.h, for a binary that also compiles the real scheduler.cc. Include it before the
// #include <UNIT>_CC_PATH and net_ib_cast_scheduler_seam_undef.h after it. No include guard on purpose.

// The headers declaring the renamed names, so a rename never rewrites a declaration.
#include "common_cast.h"
#include "net_ib_cast_scheduler_fakes.h"

#define rcclUseIbCastQpSched micro_rcclUseIbCastQpSched
#define IbCastQpSchedInitParms micro_IbCastQpSchedInitParms
