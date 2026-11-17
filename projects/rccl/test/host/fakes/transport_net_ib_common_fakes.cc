/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/transport/net_ib/common.cc, for the microtest binaries that do not compile it.

#include "net.h"

// Only its address matters: init.cc compares comm->ncclNet against it.
ncclNet_t ncclNetIb{};
