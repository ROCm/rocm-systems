/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seam for ncclIbScanDefaultPeerMemClients (src/misc/gdr_peermem.cc), whose real
// answer depends on the host's /sys. Fails loud by default.

#ifndef RCCL_TEST_HOST_GDR_PEERMEM_FAKES_H_
#define RCCL_TEST_HOST_GDR_PEERMEM_FAKES_H_

#include <functional>

extern std::function<int()> g_ncclIbScanDefaultPeerMemClients;

void ResetGdrPeermemFakes();

#endif  // RCCL_TEST_HOST_GDR_PEERMEM_FAKES_H_
