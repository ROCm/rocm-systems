/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// src/tuning/policy_adapter.cc for binaries that compile connect.cc or coll_reg.cc but not the
// execution policy sources. Each stub answers as if no policy rule matched.

#include "policy_adapter.h"

int rcclGetCollectiveExecutionPolicyRequiredChannels(struct ncclComm*, bool) { return 0; }

bool rcclPolicyCollectiveAllowsRegistration(const struct ncclTaskColl*) { return true; }
