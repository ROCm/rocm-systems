/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile.h"
#include "io.h"
#include "memcpy-kernel.h"

#include <cerrno>
#include <cstdint>

namespace hipFile {

__global__ void
hipFileMemcpyKernel(AsyncOpFallback *op)
{
    // Expects to be called with a single block
    const size_t  start_idx                  = threadIdx.x;
    const ssize_t bytes_transferred_internal = op->bytes_transferred_internal;

    if (bytes_transferred_internal < 0)
        return;

    IoType io_type = op->io_type;

    size_t *size_p = std::get_if<size_t>(&op->size);
    if (!size_p) {
        if (start_idx == 0) {
            op->bytes_transferred_internal = -hipFileInvalidValue;
        }
        return;
    }
    const size_t size = io_type == IoType::Read ? static_cast<size_t>(bytes_transferred_internal) : *size_p;

    const hoff_t *buffer_offset_p = std::get_if<const hoff_t>(&op->buffer_offset);
    if (!buffer_offset_p) {
        if (start_idx == 0) {
            op->bytes_transferred_internal = -hipFileInvalidValue;
        }
        return;
    }
    const hoff_t buffer_offset = *buffer_offset_p;

    size_t *const cpu_buffer = reinterpret_cast<size_t *>(op->bounce_buffer_dev_ptr);
    if (!cpu_buffer) {
        if (start_idx == 0) {
            op->bytes_transferred_internal = -hipFileInvalidValue;
        }
        return;
    }

    void *const gpu_buffer = op->gpu_buffer;
    if (!gpu_buffer) {
        if (start_idx == 0) {
            op->bytes_transferred_internal = -hipFileInvalidValue;
        }
        return;
    }
    size_t *const gpu_buffer_region = reinterpret_cast<size_t *>(reinterpret_cast<uintptr_t>(op->gpu_buffer) +
                                                                 static_cast<uintptr_t>(buffer_offset));
    const size_t  num_chunks        = size / sizeof(size_t);

    switch (io_type) {
        case IoType::Read:
            for (size_t idx = start_idx; idx < num_chunks; idx += blockDim.x) {
                gpu_buffer_region[idx] = cpu_buffer[idx];
            }
            break;
        case IoType::Write:
            for (size_t idx = start_idx; idx < num_chunks; idx += blockDim.x) {
                cpu_buffer[idx] = gpu_buffer_region[idx];
            }
            break;
        default:
            if (start_idx == 0)
                op->bytes_transferred_internal = -hipFileInvalidValue;
            return;
    }

    if (start_idx == 0) {
        uint8_t *const cpu_buffer_bytes        = reinterpret_cast<uint8_t *>(cpu_buffer);
        uint8_t *const gpu_buffer_region_bytes = reinterpret_cast<uint8_t *>(gpu_buffer_region);
        const size_t   remaining_start         = num_chunks * sizeof(size_t);

        switch (io_type) {
            case IoType::Read:
                for (size_t idx = remaining_start; idx < size; idx++) {
                    gpu_buffer_region_bytes[idx] = cpu_buffer_bytes[idx];
                }
                break;
            case IoType::Write:
                for (size_t idx = remaining_start; idx < size; idx++) {
                    cpu_buffer_bytes[idx] = gpu_buffer_region_bytes[idx];
                }
                break;
            default:
                op->bytes_transferred_internal = -hipFileInvalidValue;
                return;
        }
    }
}

}
