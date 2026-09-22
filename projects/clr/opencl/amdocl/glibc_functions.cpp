/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#if defined(__linux__)

#include <string.h>

#if defined(__cplusplus)
extern "C" {
#endif  // __cplusplus

#if defined(_LP64)
asm(".symver memcpy, memcpy@GLIBC_2.2.5");
void* __wrap_memcpy(void* dest, const void* src, size_t n) { return memcpy(dest, src, n); }
#endif  // _LP64

#if defined(__cplusplus)
}
#endif  // __cplusplus)

#endif  // __linux__
