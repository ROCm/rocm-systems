/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "hip.h"
#include "state.h"
#include "stats.h"
#include "sys.h"

namespace hipFile {

void
hipFileInit()
{
    Context<Hip>::get();
    Context<Sys>::get();
    Context<IStatsServer>::get();
    Context<DriverState>::get();
}

}
