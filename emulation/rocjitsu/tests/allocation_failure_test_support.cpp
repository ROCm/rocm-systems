// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include "allocation_failure_test_support.h"

#include <new>

namespace rocjitsu::test_support {
thread_local ScopedAllocationFailure *allocation_failure = nullptr;

ScopedAllocationFailure::ScopedAllocationFailure(size_t minimum_bytes)
    : minimum_bytes_(minimum_bytes), previous_(allocation_failure) {
  allocation_failure = this;
}
ScopedAllocationFailure::~ScopedAllocationFailure() { allocation_failure = previous_; }
} // namespace rocjitsu::test_support

extern "C" void *__real__Znwm(size_t bytes);
extern "C" void *__wrap__Znwm(size_t bytes) {
  if (auto *failure = rocjitsu::test_support::allocation_failure; failure && failure->reject(bytes))
    throw std::bad_alloc();
  return __real__Znwm(bytes);
}
