/*
 * Copyright (C) 2016-2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include <gtest/gtest.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string>
#include "amdp2ptest.h"
#include "RDMAUtil.hpp"

void LocalMemoryAccess::Open() {
    fd = open(AMDP2PTEST_DEVICE_PATH, O_RDWR);
}

void LocalMemoryAccess::Close() {
    close(fd);
    fd = -1;
}

int LocalMemoryAccess::GetPages(uint64_t gpu_va_addr, uint64_t size) {
    struct AMDRDMA_IOCTL_GET_PAGES_PARAM param = {0};

    if (fd <= 0)
        return -1;

    param.addr = gpu_va_addr;
    param.length = size;

    return ioctl(fd, AMD2P2PTEST_IOCTL_GET_PAGES, &param);
}

void *LocalMemoryAccess::MMap(uint64_t offset, size_t size) {
    void *gpuAddr;

    if (fd <= 0)
        return NULL;

    gpuAddr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, offset);
    return gpuAddr;
}

void LocalMemoryAccess::UnMap(void *offset, size_t size) {
    munmap(offset, size);
}
