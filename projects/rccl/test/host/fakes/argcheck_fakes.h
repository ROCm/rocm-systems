/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for the symbols defined by src/misc/argcheck.cc. PtrCheck is a plain accept.

#ifndef RCCL_TEST_HOST_ARGCHECK_FAKES_H_
#define RCCL_TEST_HOST_ARGCHECK_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclComm;

// Defaults accept: fixtures hand-build a valid comm and pass host addresses.
extern std::function<ncclResult_t(struct ncclComm*, const char* /*opname*/, const char* /*ptrname*/)> g_commCheck;
extern std::function<ncclResult_t(const void* /*pointer*/, struct ncclComm*, const char* /*ptrname*/,
                                  const char* /*opname*/)>
    g_cudaPtrCheck;

void ResetArgcheckFakes();

#endif  // RCCL_TEST_HOST_ARGCHECK_FAKES_H_
