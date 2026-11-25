/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/transport/net_ib_cast/capability.cc fakes. See net_ib_cast_capability_fakes.h.

#include "net_ib_cast_capability_fakes.h"

#include "capability_cast.h"
#include "fail_loud.h"
#include "signature-drift.h"

static bool DefaultIbCastCapUdSupported(const struct ncclIbDev*) {
  FailLoudUnfaked("net_ib_cast_capability_fakes", "IbCastCapUdSupported");
}
std::function<bool(const struct ncclIbDev*)> g_ibCastCapUdSupported = DefaultIbCastCapUdSupported;

ASSERT_HOOK_MATCHES_PROD(g_ibCastCapUdSupported, IbCastCapUdSupported);

bool IbCastCapUdSupported(const struct ncclIbDev* dev) { return g_ibCastCapUdSupported(dev); }

void ResetNetIbCastCapabilityFakes() { g_ibCastCapUdSupported = DefaultIbCastCapUdSupported; }
