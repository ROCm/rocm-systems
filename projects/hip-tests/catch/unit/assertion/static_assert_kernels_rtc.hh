/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/*
Positive and negative kernels used for the static_assert Test Cases that are using RTC.
*/

static constexpr auto kStaticAssert_Positive{
    R"(
    __global__ void StaticAssertPassKernel1() {
      static_assert(sizeof(int) < sizeof(long), "[StaticAssertPassKernel1]");
    }

    __global__ void StaticAssertPassKernel2() {
      static_assert(10 > 5, "[StaticAssertPassKernel2]");
    }

    __global__ void StaticAssertFailKernel1() {
      static_assert(sizeof(int) > sizeof(long), "[StaticAssertFailKernel1]");
    }

    __global__ void StaticAssertFailKernel2() {
      static_assert(10 < 5, "[StaticAssertFailKernel2]");
    }
  )"};

static constexpr auto kStaticAssert_Negative{
    R"(
    __global__ void StaticAssertErrorKernel1() {
      const int tid = threadIdx.x + blockIdx.x * blockDim.x;
      static_assert(tid % 2 == 1, "[StaticAssertErrorKernel1]");
    }

    __global__ void StaticAssertErrorKernel2() {
      int tid = threadIdx.x + blockIdx.x * blockDim.x;
      static_assert(++tid > 2, "[StaticAssertErrorKernel2]");
    }
  )"};
