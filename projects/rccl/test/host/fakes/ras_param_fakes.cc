/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ras/ras_param.h"
#include "ras/ras_internal.h"
#include "fakes/ras_param_fakes.h"
#include "fakes/signature-drift.h"

namespace {
int64_t DefaultRasTimeoutFactorNs(int64_t baseSeconds) { return baseSeconds * CLOCK_UNITS_PER_SEC; }
}

std::function<int64_t(int64_t)> g_rasTimeoutFactorNs = DefaultRasTimeoutFactorNs;
ASSERT_HOOK_MATCHES_PROD(g_rasTimeoutFactorNs, rasTimeoutFactorNs);
#undef ASSERT_HOOK_MATCHES_PROD

int64_t rasTimeoutFactorNs(int64_t baseSeconds) { return g_rasTimeoutFactorNs(baseSeconds); }

void ResetRasParamFakes() { g_rasTimeoutFactorNs = DefaultRasTimeoutFactorNs; }

// client.cc is the unit under test; keep its timeout calculations independent
// of NCCL_RAS_TIMEOUT_FACTOR inherited from the test runner's environment.
double rasTimeoutFactorSec(int baseSeconds) { return static_cast<double>(baseSeconds); }
