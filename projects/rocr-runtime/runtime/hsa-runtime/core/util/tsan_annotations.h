/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// ThreadSanitizer annotations for custom synchronization primitives.
//
// HSA signals provide synchronization semantics that TSAN doesn't understand
// natively. These annotations inform TSAN about happens-before relationships
// established by signal operations, eliminating false positive data race
// reports when using async memory operations.

#ifndef HSA_RUNTIME_CORE_UTIL_TSAN_ANNOTATIONS_H_
#define HSA_RUNTIME_CORE_UTIL_TSAN_ANNOTATIONS_H_

// Check for ThreadSanitizer - GCC/Clang define __SANITIZE_THREAD__ or
// __has_feature(thread_sanitizer)
#if defined(__SANITIZE_THREAD__)
#define ROCR_TSAN_ENABLED 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define ROCR_TSAN_ENABLED 1
#endif
#endif

#ifdef ROCR_TSAN_ENABLED

#include <sanitizer/tsan_interface.h>

namespace rocr {

// Signal that data has been published and is ready to be consumed.
// Call this AFTER writing data and BEFORE signaling completion.
// The addr parameter should be the signal's value address.
inline void TsanRelease(void* addr) {
  __tsan_release(addr);
}

// Signal that published data has been observed and will now be read.
// Call this AFTER the wait completes and BEFORE reading the data.
// The addr parameter should be the signal's value address.
inline void TsanAcquire(void* addr) {
  __tsan_acquire(addr);
}

}  // namespace rocr

#else  // !ROCR_TSAN_ENABLED

namespace rocr {

inline void TsanRelease(void*) {}
inline void TsanAcquire(void*) {}

}  // namespace rocr

#endif  // ROCR_TSAN_ENABLED

#endif  // HSA_RUNTIME_CORE_UTIL_TSAN_ANNOTATIONS_H_
