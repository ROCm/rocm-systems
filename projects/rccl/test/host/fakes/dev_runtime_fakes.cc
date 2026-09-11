/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See dev_runtime_fakes.h. ncclDevrFindWindow / ncclDevrIsOneLsaTeam remain in
// nccl_stubs.cc as part of that fail-loud floor; move them here when one of them
// needs a seam.

#include "dev_runtime_fakes.h"

struct ncclDevrWindow;

bool g_devrWindowIsMultiSegment = false;
bool g_devrWindowHasSysmemSegment = false;

bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow*) { return g_devrWindowIsMultiSegment; }
bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow*) { return g_devrWindowHasSysmemSegment; }

void ResetDevRuntimeFakes() {
  g_devrWindowIsMultiSegment = false;
  g_devrWindowHasSysmemSegment = false;
}
