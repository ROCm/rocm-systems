/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Window-shape queries defined by src/dev_runtime.cc. Default to "not
// registered" so the NVLS/CollNet registration arms stay off unless a test asks.

#ifndef RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_
#define RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclComm;
struct ncclDevrWindow;

extern std::function<ncclResult_t(struct ncclComm*, void const*, struct ncclDevrWindow**)> g_devrFindWindow;
extern std::function<bool(struct ncclDevrWindow*)> g_devrWindowHasSysmemSegment;
extern bool g_devrWindowIsMultiSegment;  // UNDRIVEN
extern bool g_devrWindowHasSysmemSegmentValue;  // UNDRIVEN

void ResetDevRuntimeFakes();

#endif  // RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_
