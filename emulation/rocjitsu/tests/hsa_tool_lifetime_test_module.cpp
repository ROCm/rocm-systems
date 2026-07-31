// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/hsa_tool_lifetime.h"

#if defined(__GNUC__) || defined(__clang__)
#define RJ_TEST_EXPORT __attribute__((visibility("default")))
#else
#define RJ_TEST_EXPORT
#endif

extern "C" RJ_TEST_EXPORT bool rj_test_retain_hsa_tool_dso() {
  return rocjitsu::hooks::retain_hsa_tool_dso();
}
