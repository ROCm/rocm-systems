/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
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
