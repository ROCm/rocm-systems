/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the src/config/collconfig.cc symbols collconfig_fakes.cc defines.

#ifndef RCCL_TEST_HOST_FAKES_COLLCONFIG_FAKES_H_
#define RCCL_TEST_HOST_FAKES_COLLCONFIG_FAKES_H_

#include <functional>

#include "nccl.h"

// ncclParseCollConfig. Fails loud until a test installs it: a faithful parse needs the real collconfig.cc.
extern std::function<ncclResult_t(const ncclCollConfig_t*, ncclCollConfig_t*)> g_parseCollConfig;

void ResetCollConfigFakes();

#endif  // RCCL_TEST_HOST_FAKES_COLLCONFIG_FAKES_H_
