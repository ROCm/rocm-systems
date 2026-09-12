/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/plugin/env.cc. Distinct from env_fakes,
// which stands in for src/misc/param.cc.

#ifndef RCCL_TEST_HOST_ENV_PLUGIN_FAKES_H_
#define RCCL_TEST_HOST_ENV_PLUGIN_FAKES_H_

#include <functional>

#include "nccl.h"

// ncclInitEnv() latches this behind std::call_once, so only the FIRST call in a process can observe a failure.
extern ncclResult_t g_ncclEnvPluginInitResult;
extern std::function<ncclResult_t()> g_ncclEnvPluginInit;

void ResetEnvPluginFakes();

#endif  // RCCL_TEST_HOST_ENV_PLUGIN_FAKES_H_
