/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "hip.h"
#include "hipfile-warnings.h"
#include "state.h"
#include "stats.h"
#include "sys.h"

namespace hipFile {

HipFileInit::HipFileInit()
{
    Context<Hip>::get();
    Context<Sys>::get();
    Context<StatsServer>::get();
    Context<DriverState>::get();
}

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF
HIPFILE_WARN_NO_EXIT_DTOR_OFF
static HipFileInit *hipfile_init = Context<HipFileInit>::get();
HIPFILE_WARN_NO_EXIT_DTOR_ON
HIPFILE_WARN_NO_GLOBAL_CTOR_ON

}
