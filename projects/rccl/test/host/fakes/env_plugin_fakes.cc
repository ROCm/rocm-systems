/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/plugin/env.cc fakes. See env_plugin_fakes.h.

#include "env_plugin_fakes.h"

ncclResult_t g_ncclEnvPluginInitResult = ncclSuccess;
// Default reads the plain result seam so both styles work: set g_ncclEnvPluginInitResult, or ScopedHook the functor.
static ncclResult_t DefaultNcclEnvPluginInit() { return g_ncclEnvPluginInitResult; }
std::function<ncclResult_t()> g_ncclEnvPluginInit = DefaultNcclEnvPluginInit;
ncclResult_t ncclEnvPluginInit(void) { return g_ncclEnvPluginInit(); }

void ResetEnvPluginFakes() {
  g_ncclEnvPluginInitResult = ncclSuccess;
  g_ncclEnvPluginInit = DefaultNcclEnvPluginInit;
}
