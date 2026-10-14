/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/transport/net_ib/ and net_ib_cast/, for the microtest binaries that do not compile them.

#include <cstdint>

#include "fakes/fail_loud.h"
#include "fakes/nccl_fakes.h"
#include "net.h"

int64_t ncclParamIbQpsPerConn() { return g_loadParam("IB_QPS_PER_CONNECTION", 1); }

// net.cc compares a plugin's address against these.
ncclNet_t ncclNetIb = {"IB"};
ncclNet_t netIbCast = {"IBCast"};
ncclResult_t rcclNetP2pPolicy(void*, int) { FailLoudUnfaked("transport_net_ib_fakes", "rcclNetP2pPolicy"); }
ncclResult_t rcclCastNetP2pPolicy(void*, int) { FailLoudUnfaked("transport_net_ib_fakes", "rcclCastNetP2pPolicy"); }
