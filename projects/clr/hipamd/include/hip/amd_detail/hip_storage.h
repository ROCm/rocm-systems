/* Copyright (c) 2025 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */
#pragma once

#include <hip/hip_runtime.h>

typedef struct hipAmdFileHandle_t {
  /*
   * file handle for AIS read & write. Linux will use fd.
   * pad is keep the size consistent across different platforms.
   */
  union {
    void* handle;
    int fd;
    uint8_t pad[8];
  };
} hipAmdFileHandle_t;

/**
 * @brief Read data from a file to device memory.
 *
 * Reads data from a file at the specified offset into a device memory buffer.
 * The device memory pointer must be accessible from the host and point to
 * a valid allocation.
 *
 * @param[IN] handle: Handle of the file to read from.
 * @param[IN] devicePtr: Device memory buffer pointer to store the read data.
 * @param[IN] size: Size in bytes of the data to read.
 * @param[IN] file_offset: Offset in bytes into the file where data will be read from.
 * @param[IN/OUT] size_copied: Actual number of bytes copied.
 * @param[IN/OUT] status: Additional status if any.
 */
hipError_t hipAmdFileRead(hipAmdFileHandle_t handle, void* devicePtr, uint64_t size, int64_t file_offset,
                       uint64_t* size_copied, int32_t* status);

/**
 * Write data from device memory to a file.
 *
 * Writes data from device memory buffer to a file at the specified offset.
 * The device memory pointer must be accessible from the host and point to
 * a valid allocation.
 *
 * @param[IN] handle: Handle of the file to write to.
 * @param[IN] devicePtr: Device memory buffer pointer containing data to write.
 * @param[IN] size: Size in bytes of the data to write.
 * @param[IN] file_offset: Offset in bytes into the file where data will be written.
 * @param[IN/OUT] size_copied: Actual number of bytes written.
 * @param[IN/OUT] status: Additional status if any.
 */
hipError_t hipAmdFileWrite(hipAmdFileHandle_t handle, void* devicePtr, uint64_t size, int64_t file_offset,
                        uint64_t* size_copied, int32_t* status);
