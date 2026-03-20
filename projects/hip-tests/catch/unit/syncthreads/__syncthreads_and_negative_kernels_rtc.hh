/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

static constexpr auto kSyncthreadsAndSource{
    R"(
        struct Dummy {
          __device__ Dummy() {}
          __device__ ~Dummy() {}
        };

        __global__ void __syncthreads_and_v1(int* predicate) {
          int result = __syncthreads_and(predicate);
        }

        __global__ void __syncthreads_and_v2(Dummy predicate) {
          int result = __syncthreads_and(predicate);
        }
    )"};