/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_FAKES_ROCSHMEM_FAKES_H_
#define RCCL_TEST_HOST_FAKES_ROCSHMEM_FAKES_H_

// The rocSHMEM host API, faked for host-only microtests. librocshmem.a
// initialises a GPU, so the Init targets cannot link the real one; every
// symbol here is reachable only under ENABLE_ROCSHMEM (init.cc:2943-3007).

#include <functional>

#include <rocshmem/rocshmem.hpp>

namespace rocshmem {

// Seams for the two calls init.cc branches on. Both default to
// ROCSHMEM_SUCCESS so the rocSHMEM arm runs to completion unless a test says
// otherwise; no caller inspects a result from the rest of the API.
extern std::function<int()> g_rocshmemGetUniqueId;
extern std::function<int()> g_rocshmemInitAttr;

// What rocshmem_malloc handed out last, so a test can recognise
// comm->sourceRshmem and comm->destRshmem. Not a real allocation: nothing
// dereferences either.
extern void* g_rocshmemMallocReturn;

}  // namespace rocshmem

void ResetRocshmemFakes();

#endif  // RCCL_TEST_HOST_FAKES_ROCSHMEM_FAKES_H_
