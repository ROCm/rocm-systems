/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile.h"

#include <cstddef>

struct HipFileDataOps {
    static void assertMemoryRegionsMatch(void *mem1, hoff_t mem1_offset, void *mem2, hoff_t mem2_offset,
                                         size_t region_size);
    static void assertFileAndMemoryRegionsMatch(void *mem, hoff_t mem_offset, int fd, hoff_t fd_offset,
                                                size_t region_size);
    static void assertZeroedMemRegion(void *mem, hoff_t mem_offset, size_t region_size);
    static void assertZeroedFileRegion(int fd, hoff_t fd_offset, size_t region_size);
    static void randomizeMemoryRegion(void *mem, hoff_t offset, size_t region_size);
    static void zeroMemoryRegion(void *mem, hoff_t offset, size_t region_size);
    static void zeroFileRegion(int fd, size_t size, hoff_t offset = 0);
    static void randomizeFileRegion(int fd, size_t size, hoff_t offset = 0);
};
