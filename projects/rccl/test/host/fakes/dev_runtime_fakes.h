/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Window-shape queries defined by src/dev_runtime.cc. Default to "not
// registered" so the NVLS/CollNet registration arms stay off unless a test asks.

#ifndef RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_
#define RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_

extern bool g_devrWindowIsMultiSegment;  // UNDRIVEN
extern bool g_devrWindowHasSysmemSegment;  // UNDRIVEN

void ResetDevRuntimeFakes();

#endif  // RCCL_TEST_HOST_DEV_RUNTIME_FAKES_H_
