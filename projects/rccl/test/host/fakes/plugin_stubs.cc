/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for src/plugin/: plugin_open.cc's library path table and the net/net_v*.cc version
// getters. These satisfy a unit-under-test's link-time symbol closure; the tests linking this floor never call
// the getters (net.cc's getNcclNet[] tables are swapped for fakes before use).

#include "fail_loud.h"
#include "nccl.h"
#include "net.h"
#include "plugin.h"

// src/plugin/plugin_open.cc
char* ncclPluginLibPaths[ncclPluginTypeEnv + 1] = {};

// src/plugin/net/net_v6.cc .. net_v12.cc
#define PLUGIN_NET_STUB(v)                                                                        \
  ncclNet_t* getNcclNet_v##v(void*) { FailLoudUnfaked("plugin_stubs", "getNcclNet_v" #v); }       \
  ncclCollNet_t* getNcclCollNet_v##v(void*) { FailLoudUnfaked("plugin_stubs", "getNcclCollNet_v" #v); }
PLUGIN_NET_STUB(6)
PLUGIN_NET_STUB(7)
PLUGIN_NET_STUB(8)
PLUGIN_NET_STUB(9)
PLUGIN_NET_STUB(10)
PLUGIN_NET_STUB(11)
PLUGIN_NET_STUB(12)
#undef PLUGIN_NET_STUB
