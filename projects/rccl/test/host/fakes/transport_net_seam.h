/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Routes a unit's calls into src/transport/net.cc's rcclUseAinic to the micro_* trampoline in
// transport_stubs.h, for a binary that also compiles the real net.cc. Include it before the
// #include <UNIT>_CC_PATH and transport_net_seam_undef.h after it. No include guard on purpose.

// The headers declaring the renamed name, or calling it from an inline function, so a rename never
// rewrites either.
#include "channel.h"
#include "net.h"
#include "transport_stubs.h"

#define rcclUseAinic micro_rcclUseAinic
