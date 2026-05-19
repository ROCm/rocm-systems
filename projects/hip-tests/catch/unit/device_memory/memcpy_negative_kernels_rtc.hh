/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

static constexpr auto kMemcpyParam{
    R"(
        struct Dummy {
          __device__ Dummy() {}
          __device__ ~Dummy() {}
        };
        __global__ void memcpy_n1(int* dst, const int src, size_t size) { memcpy(dst, src, size); }
        __global__ void memcpy_n2(int dst, const int* src, size_t size) { memcpy(dst, src, size); }
        __global__ void memcpy_n3(int* dst, const int* src, size_t* size) { memcpy(dst, src, size); }
        __global__ void memcpy_n8(int* dst, const int* src, Dummy size) { memcpy(dst, src, size); }
    )"};