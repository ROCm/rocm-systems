/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>

namespace hipFileTest {

// Launches the grid-stride integer verify+modify kernel and synchronizes the device.
//
// device_buffer_start: start of the whole device allocation.
// device_buffer_size:  element count of the whole allocation.
// data_start:          element index of the start of the data within the allocation; must satisfy
//                      data_start + n <= device_buffer_size.
// n:                   data element count.
// expect_base:         value the kernel expects at data index 0.
// first_bad_idx:       device int, init -1; first data index not matching the pattern.
// slack_sentinel:      value every non-data allocation element must still hold.
// first_bad_slack_idx: device int, init -1; first non-data allocation index != sentinel.
// modify_stride:       only data elements whose index is a multiple of this are
//                      doubled; verify-in still covers the whole data region. Must be
//                      >= 1; pass 1 to double every element.
hipError_t launchVerifyAndModify(int32_t *device_buffer_start, size_t device_buffer_size, size_t data_start,
                                 size_t n, int32_t expect_base, int32_t *first_bad_idx,
                                 int32_t slack_sentinel, int32_t *first_bad_slack_idx, dim3 grid,
                                 dim3 workgroup, size_t modify_stride);

// Launches the grid-stride byte-granular verify+modify kernel and synchronizes the device.
//
// buf:                 start of the whole device allocation (in BYTES).
// buf_size:            byte count of the whole allocation.
// data_start:          byte index of the start of the Data within the allocation; must satisfy
//                      data_start + n <= buf_size.
// n:                   Data byte count.
// entry:               value every Data byte must hold on entry.
// modified:            value qualifying Data bytes are set to.
// first_bad_idx:       device int32, init -1; first Data index not equal to `entry`.
// slack_sentinel:      value every device sentinel region byte must still hold.
// first_bad_slack_idx: device int32, init -1; first device sentinel region allocation index != sentinel.
// modify_stride:       only Data bytes whose data-index is a multiple of this are set to
//                      `modified`; verify-in still covers the whole Data region. Must be
//                      >= 1; pass 1 to modify every byte.
hipError_t launchVerifyAndModifyBytes(uint8_t *buf, size_t buf_size, size_t data_start, size_t n,
                                      uint8_t entry, uint8_t modified, int32_t *first_bad_idx,
                                      uint8_t slack_sentinel, int32_t *first_bad_slack_idx, dim3 grid,
                                      dim3 workgroup, size_t modify_stride);

} // namespace hipFileTest
