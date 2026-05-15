/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

__global__ void kernel() { asm volatile("v_nop" ::: "memory"); }

// This test case compiles with --offload-arch=amdgcnspirv to verify SPIRV mode
HIP_TEST_CASE(Unit_test_spirv_mode) { kernel<<<1, 32>>>(); }
