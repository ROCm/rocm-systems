// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__HIP_DEVICE_COMPILE__)
// ROCR keeps process-lifetime runtime state allocated after hipDeviceReset().
// Ignore only allocations whose stack includes the external HSA runtime while
// retaining LeakSanitizer coverage for rocjitsu and this test executable.
extern "C" __attribute__((visibility("default"))) const char *__lsan_default_suppressions() {
  return "leak:libhsa-runtime64.so\n";
}
#endif
