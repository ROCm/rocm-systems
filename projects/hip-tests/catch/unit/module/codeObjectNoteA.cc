/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// One of two objects linked into a code object with two metadata notes (see CMakeLists.txt)

#include <hip/hip_runtime.h>

extern "C" __global__ void note_a(int* out) { out[0] = 11; }
