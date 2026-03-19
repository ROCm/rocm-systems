/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

__global__ void StaticAssertPassKernel1() {
  static_assert(sizeof(int) < sizeof(long), "[StaticAssertPassKernel1]");
}

__global__ void StaticAssertPassKernel2() { static_assert(10 > 5, "[StaticAssertPassKernel2]"); }

__global__ void StaticAssertFailKernel1() {
  static_assert(sizeof(int) > sizeof(long), "[StaticAssertFailKernel1]");
}

__global__ void StaticAssertFailKernel2() { static_assert(10 < 5, "[StaticAssertFailKernel2]"); }
