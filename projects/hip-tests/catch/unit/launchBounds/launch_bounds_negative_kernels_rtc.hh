/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/*
Negative kernels used for the launch bounds negative Test Cases that are using RTC.
*/

static constexpr auto kMaxThreadsZero{
    R"(
    __launch_bounds__(0) __global__ void SumKernel(int* sum) {
      const int tid = threadIdx.x + blockIdx.x * blockDim.x;
      atomicAdd(sum, tid);
    }
  )"};

static constexpr auto kMaxThreadsNegative{
    R"(
    __launch_bounds__(-1) __global__ void SumKernel(int* sum) {
      const int tid = threadIdx.x + blockIdx.x * blockDim.x;
      atomicAdd(sum, tid);
    }
  )"};

static constexpr auto kMinWarpsNegative{
    R"(
    __launch_bounds__(128, -1) __global__ void SumKernel(int* sum) {
      const int tid = threadIdx.x + blockIdx.x * blockDim.x;
      atomicAdd(sum, tid);
    }
  )"};

static constexpr auto kMaxThreadsNotInt{
    R"(
    __launch_bounds__(1.5) __global__ void SumKernel(int* sum) {
      const int tid = threadIdx.x + blockIdx.x * blockDim.x;
      atomicAdd(sum, tid);
    }
  )"};

static constexpr auto kMinWarpsNotInt{
    R"(
    __launch_bounds__(128, 1.5) __global__ void SumKernel(int* sum) {
      const int tid = threadIdx.x + blockIdx.x * blockDim.x;
      atomicAdd(sum, tid);
    }
  )"};
