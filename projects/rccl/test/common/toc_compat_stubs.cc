/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ppc64le TOC references survive --gc-sections, so host microtests that link
// dev_runtime.cc or the debug/alloc inlines but not every defining .cc file
// fail to link. Provide weak fallbacks; the real definitions override them
// where linked. Added to the link on ppc64le only (see the test CMake).

#if defined(__powerpc64__) || defined(__PPC64__)

#include <cstdint>

#include "dev_runtime.h"  // struct ncclDevCommCompat
#include "alloc.h"        // struct allocationTracker, MAX_ALLOC_TRACK_NGPU

__attribute__((weak)) struct ncclDevCommCompat ncclDevCommCompat_v22902 = {};
__attribute__((weak)) struct ncclDevCommCompat ncclDevCommCompat_v22907 = {};
__attribute__((weak)) struct ncclDevCommCompat ncclDevCommCompat_v23000 = {};
__attribute__((weak)) struct ncclDevCommCompat ncclDevCommCompat_v23100 = {};
__attribute__((weak)) struct ncclDevCommCompat ncclDevCommCompat_v23204 = {};
__attribute__((weak)) struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};
__attribute__((weak)) std::uint32_t ncclDebugLevelMask = 0;
__attribute__((weak)) std::uint64_t ncclDebugMask = 0;

#endif  // ppc64le
