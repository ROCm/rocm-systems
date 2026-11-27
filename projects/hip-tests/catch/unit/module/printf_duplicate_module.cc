/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

// Built device-only with -mprintf-kind=buffered: the code object's amdhsa.printf metadata then
// lists the repeated format string twice, with the same hash. A linked executable's code object
// has the duplicate merged, so this needs a module.
extern "C" __global__ void printf_duplicate_format_string(int* count) {
  count[0] = printf("Duplicate format string: %d\n", 1);
  count[1] = printf("Duplicate format string: %d\n", 2);
}
