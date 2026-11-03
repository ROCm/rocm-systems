/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seam for ncclIbProbeGdrSupport (src/misc/gdr_probe.cc), for binaries that do not
// compile gdr_probe.cc (rccl-UnitTestsMicro does, so it cannot link this). Fails
// loud by default.

#ifndef RCCL_TEST_HOST_GDR_PROBE_FAKES_H_
#define RCCL_TEST_HOST_GDR_PROBE_FAKES_H_

#include <functional>

struct ibv_context;

extern std::function<int(struct ibv_context* /*context*/, int /*relaxedOrderingEnabled*/)> g_ncclIbProbeGdrSupport;

void ResetGdrProbeFakes();

#endif  // RCCL_TEST_HOST_GDR_PROBE_FAKES_H_
