/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// NVTX redirector shared by the microtest TUs that textually #include a
// production .cc which itself includes nvtx.h (init.cc, group.cc, ...).
//
// Two builds have to be satisfied. With NVTX compiled in, nvtx.h is pulled in
// here once and its range macros are neutered, so the unit-under-test's own
// re-include is a no-op and the ranges expand to nothing. With NVTX disabled
// (the standard microtest build: -DNVTX_DISABLE -DNVTX_NO_IMPL) nvtx_stub.h has
// already defined struct nccl_domain, so nvtx.h's guard is pre-set to keep the
// unit-under-test's include from redefining it, and the one macro the guarded
// sources still reference is supplied.
//
// Include this before the unit-under-test, i.e. immediately above the
// #include <UUT>_CC_PATH line.

#ifndef RCCL_TEST_HOST_NVTX_REDIRECT_H_
#define RCCL_TEST_HOST_NVTX_REDIRECT_H_

// The NVTX3 range macros expand to NOTHING in this binary, so any guard around them shows partial coverage by design.
#if !defined(NVTX_NO_IMPL) && !defined(NVTX_DISABLE)
#include "nvtx.h"  // guarded; the unit-under-test's re-include is a no-op
#undef NCCL_NVTX3_FUNC_RANGE
#define NCCL_NVTX3_FUNC_RANGE
#undef NVTX3_RANGE
#define NVTX3_RANGE(...)
#undef NVTX3_RANGE_ADD_PAYLOAD
#define NVTX3_RANGE_ADD_PAYLOAD(...)
#undef NVTX3_FUNC_WITH_PARAMS
#define NVTX3_FUNC_WITH_PARAMS(...)
#else
// nvtx_stub.h already defines nccl_domain; pre-set nvtx.h's guard so the unit-under-test cannot redefine it.
#define NCCL_NVTX_H_
#ifndef NCCL_NVTX3_FUNC_RANGE
#define NCCL_NVTX3_FUNC_RANGE
#endif
#endif

#endif  // RCCL_TEST_HOST_NVTX_REDIRECT_H_
