/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/group.cc.

#ifndef RCCL_TEST_HOST_GROUP_FAKES_H_
#define RCCL_TEST_HOST_GROUP_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclGroupJob;

extern std::function<ncclResult_t(struct ncclGroupJob*)> g_ncclGroupJobAbort;

void ResetGroupFakes();

#endif  // RCCL_TEST_HOST_GROUP_FAKES_H_
