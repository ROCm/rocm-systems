/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams for src/dev_runtime.cc, for targets that do not compile the real file: window-shape
// queries default to "not registered" so NVLS/CollNet registration arms stay off unless a test
// asks, and ncclDevrInitOnce defaults to success.

#ifndef RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_
#define RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_

#include <functional>

#include "nccl.h"

struct ncclComm;
struct ncclDevrWindow;

extern std::function<ncclResult_t(struct ncclComm*, void const*, struct ncclDevrWindow**)> g_devrFindWindow;
extern std::function<bool(struct ncclDevrWindow*)> g_devrWindowHasSysmemSegment;
extern bool g_devrWindowIsMultiSegment;
extern bool g_devrWindowHasSysmemSegmentValue;

// src/dev_runtime.cc's ncclDevrInitOnce: generous default (ncclSuccess), matching a comm that
// never sets symmetricSupport with a null peerInfo (dev_runtime.cc's own real error condition).
extern std::function<ncclResult_t(struct ncclComm*)> g_devrInitOnce;

void ResetDevRuntimeFakes();

#endif  // RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_
