/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

static constexpr auto kPrintfParam{
    R"(
        struct Dummy {
          __device__ Dummy() {}
          __device__ ~Dummy() {}
        };
        __global__ void printf_n1(int* p) { printf(p); }
        __global__ void printf_n2(unsigned int* p) { printf(p); }
        __global__ void printf_n3(short* p) { printf(p); }
        __global__ void printf_n4(long* p) { printf(p); }
        __global__ void printf_n5(unsigned long* p) { printf(p); }
        __global__ void printf_n6(long long* p) { printf(p); }
        __global__ void printf_n7(unsigned long long* p) { printf(p); }
        __global__ void printf_n8(float* p) { printf(p); }
        __global__ void printf_n9(double* p) { printf(p); }
        __global__ void printf_n10(long double* p) { printf(p); }
        __global__ void printf_n11(Dummy* p) { printf(p); }
    )"};