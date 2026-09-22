/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

static constexpr auto kSyncthreadsOrSource{
    R"(
        struct Dummy {
          __device__ Dummy() {}
          __device__ ~Dummy() {}
        };

        __global__ void __syncthreads_or_v1(int* predicate) {
          int result = __syncthreads_or(predicate);
        }

        __global__ void __syncthreads_or_v2(Dummy predicate) {
          int result = __syncthreads_or(predicate);
        }
    )"};