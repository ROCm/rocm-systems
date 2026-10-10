/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/transport/net_ib/connect.cc, for the microtest binaries that do not compile it.

#include <cstdint>

#include "fakes/nccl_fakes.h"

int64_t ncclParamIbQpsPerConn() { return g_loadParam("IB_QPS_PER_CONNECTION", 1); }
