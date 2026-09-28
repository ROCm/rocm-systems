/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_FAKES_ROCSHMEM_FAKES_H_
#define RCCL_TEST_HOST_FAKES_ROCSHMEM_FAKES_H_

// The rocSHMEM host API. Reachable only from init.cc's ENABLE_ROCSHMEM arm,
// which also needs rcclParamRocshmemEnabled(); no test turns that on yet.

#include <functional>

#include <rocshmem/rocshmem.hpp>

namespace rocshmem {

// The three calls init.cc checks a result from. Default ROCSHMEM_SUCCESS.
extern std::function<int()> g_rocshmemGetUniqueId;        // UNDRIVEN
extern std::function<int()> g_rocshmemSetAttrUniqueIdArgs;  // UNDRIVEN
extern std::function<int()> g_rocshmemInitAttr;           // UNDRIVEN

}  // namespace rocshmem

void ResetRocshmemFakes();

#endif  // RCCL_TEST_HOST_FAKES_ROCSHMEM_FAKES_H_
