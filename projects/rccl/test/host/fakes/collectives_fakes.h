/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the src/collectives.cc symbols collectives_fakes.cc defines.

#ifndef RCCL_TEST_HOST_FAKES_COLLECTIVES_FAKES_H_
#define RCCL_TEST_HOST_FAKES_COLLECTIVES_FAKES_H_

#include <functional>

struct ncclComm;

// rcclCollectiveMustUseEnqueuePath. Default false: a plain comm in default check mode, neither suspended nor revoked.
extern std::function<bool(struct ncclComm*)> g_collectiveMustUseEnqueuePath;
// ncclCommIsSuspended. Default false.
extern std::function<bool(struct ncclComm*)> g_commIsSuspended;

#endif  // RCCL_TEST_HOST_FAKES_COLLECTIVES_FAKES_H_
