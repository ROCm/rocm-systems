/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>

namespace hip {
inline namespace stream {

/* Empty kernel to ensure work finishes on the stream quickly */
__global__ void empty_kernel();

const hipStream_t nullStream = nullptr;
const hipStream_t streamPerThread = hipStreamPerThread;

// Checks stream for valid values of flags and priority
bool checkStream(hipStream_t stream);

// Checks stream for valid flags and a particular value of priority
bool checkStreamPriorityAndFlags(hipStream_t stream, int priority,
                                 unsigned int flags = hipStreamDefault);

}  // namespace stream
}  // namespace hip
