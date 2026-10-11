/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "io-data-modification-kernel.h"

#include <hip/hip_runtime.h>

namespace hipFileTest {

// Grid-stride integer verify+modify kernel.
//
// This kernel verifies:
//    - bytes within [device_buffer_start, device_buffer_start + device_buffer_size)
//      but outside of [device_buffer_start + data_start, device_buffer_start + data_start + n)
//      are untouched (remain equal to slack_sentinel this buffer was initialized
//      with by hipMemcpy).
//    - bytes (interpreted as integers) within [device_buffer_start + data_start,
//      device_buffer_start + data_start + n) are initially equal to the expected
//      pattern (buffer + data_start)[i] = expect_base + i.
//
// This kernel then modifies the data by:
//    - (buffer + data_start)[i] = expect_base + i iff i % modify_stride != 0.
//                               = (expect_base + i)*2 iff i % modify_stride == 0.
static __global__ void
verifyAndModify(int32_t *device_buffer_start, size_t device_buffer_size, size_t data_start, size_t n,
                int32_t expect_base, int32_t *first_bad_idx, int32_t slack_sentinel,
                int32_t *first_bad_slack_idx, size_t modify_stride)
{
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t start  = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = start; i < device_buffer_size; i += stride) {
        const bool in_data = (data_start <= i) && (i < data_start + n);
        if (in_data) {
            const size_t d = i - data_start;
            if (device_buffer_start[i] != expect_base + static_cast<int32_t>(d)) {
                atomicCAS(first_bad_idx, -1, static_cast<int32_t>(d));
            }
            if (d % modify_stride == 0) {
                device_buffer_start[i] = device_buffer_start[i] * 2;
            }
        }
        else {
            if (device_buffer_start[i] != slack_sentinel) {
                atomicCAS(first_bad_slack_idx, -1, static_cast<int32_t>(i));
            }
        }
    }
}

hipError_t
launchVerifyAndModify(int32_t *device_buffer_start, size_t device_buffer_size, size_t data_start, size_t n,
                      int32_t expect_base, int32_t *first_bad_idx, int32_t slack_sentinel,
                      int32_t *first_bad_slack_idx, dim3 grid, dim3 workgroup, size_t modify_stride)
{
    verifyAndModify<<<grid, workgroup>>>(device_buffer_start, device_buffer_size, data_start, n, expect_base,
                                         first_bad_idx, slack_sentinel, first_bad_slack_idx, modify_stride);
    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) {
        return launch_err;
    }
    return hipDeviceSynchronize();
}

// Grid-stride byte-wise verify+modify kernel.
//
// This kernel verifies:
//    - bytes within [buf, buf + buf_size) but outside of [buf + data_start, buf + data_start + n)
//      are untouched (remain equal to slack_sentinel this buffer was initialized
//      with by hipMemcpy).
//    - bytes within [device_buffer_start + data_start, device_buffer_start + data_start + n)
//      are initially equal to the expected pattern (buf + data_start)[i] = entry.
//
// This kernel then modifies the data by:
//    - (buf + data_start)[i] = entry iff i % modify_stride != 0.
//                            = modified iff i % modify_stride == 0.
static __global__ void
verifyAndModifyBytes(uint8_t *buf, size_t buf_size, size_t data_start, size_t n, uint8_t entry,
                     uint8_t modified, int32_t *first_bad_idx, uint8_t slack_sentinel,
                     int32_t *first_bad_slack_idx, size_t modify_stride)
{
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t start  = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = start; i < buf_size; i += stride) {
        const bool in_data = (data_start <= i) && (i < data_start + n);
        if (in_data) {
            const size_t d = i - data_start;
            if (buf[i] != entry) {
                atomicCAS(first_bad_idx, -1, static_cast<int32_t>(d));
            }
            if (d % modify_stride == 0) {
                buf[i] = modified;
            }
        }
        else {
            if (buf[i] != slack_sentinel) {
                atomicCAS(first_bad_slack_idx, -1, static_cast<int32_t>(i));
            }
        }
    }
}

hipError_t
launchVerifyAndModifyBytes(uint8_t *buf, size_t buf_size, size_t data_start, size_t n, uint8_t entry,
                           uint8_t modified, int32_t *first_bad_idx, uint8_t slack_sentinel,
                           int32_t *first_bad_slack_idx, dim3 grid, dim3 workgroup, size_t modify_stride)
{
    verifyAndModifyBytes<<<grid, workgroup>>>(buf, buf_size, data_start, n, entry, modified, first_bad_idx,
                                              slack_sentinel, first_bad_slack_idx, modify_stride);
    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) {
        return launch_err;
    }
    return hipDeviceSynchronize();
}

} // namespace hipFileTest
