/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "DriverContext.hh"
#include <hip_test_common.hh>

DriverContext::DriverContext() {
  HIP_CHECK(hipInit(0));
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipDevicePrimaryCtxRetain(&ctx, device));
  HIP_CHECK(hipCtxPushCurrent(ctx));
}

DriverContext::~DriverContext() {
  HIP_CHECK(hipCtxPopCurrent(&ctx));
  HIP_CHECK(hipDevicePrimaryCtxRelease(device));
}
