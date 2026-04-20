/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

static constexpr auto kHipHcAdd8pkBasic{
    R"(
        struct Dummy {
          __device__ Dummy() {}
          __device__ ~Dummy() {}
        };

        __global__ void hip_hc_add8pk_char_n1(char4* out, char in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_char_n2(char4* out, char4 in1, char in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_char_n3(char* out, char4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_int_n1(char4* out, int in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_int_n2(char4* out, char4 in1, int in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_int_n3(int* out, char4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_long_n1(char4* out, long in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_long_n2(char4* out, char4 in1, long in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_long_n3(long* out, char4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_float_n1(char4* out, float in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_float_n2(char4* out, char4 in1, float in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_float_n3(float* out, char4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_dummy_n1(char4* out, Dummy in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_dummy_n2(char4* out, char4 in1, Dummy in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_dummy_n3(Dummy* out, char4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
    )"};

static constexpr auto kHipHcAdd8pkVector{
    R"(
        __global__ void hip_hc_add8pk_char4_n1(char4* out, char4 in1, char4 in2) { out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_char4_n2(char4* out, char4 in1, char4 in2) { *out = __hip_hc_add8pk(&in1, in2); }
        __global__ void hip_hc_add8pk_char4_n3(char4* out, char4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, &in2); }
        __global__ void hip_hc_add8pk_char2_n1(char4* out, char2 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_char2_n2(char4* out, char4 in1, char2 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_char2_n3(char2* out, char4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_int4_n1(char4* out, int4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_int4_n2(char4* out, char4 in1, int4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_int4_n3(int4* out, char4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_long4_n1(char4* out, long4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_long4_n2(char4* out, char4 in1, long4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_long4_n3(long4* out, char4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_float4_n1(char4* out, float4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_float4_n2(char4* out, char4 in1, float4 in2) { *out = __hip_hc_add8pk(in1, in2); }
        __global__ void hip_hc_add8pk_float4_n3(float4* out, char4 in1, char4 in2) { *out = __hip_hc_add8pk(in1, in2); }
    )"};

static constexpr auto kHipHcSub8pkBasic{
    R"(
        struct Dummy {
          __device__ Dummy() {}
          __device__ ~Dummy() {}
        };

        __global__ void hip_hc_sub8pk_char_n1(char4* out, char in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_char_n2(char4* out, char4 in1, char in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_char_n3(char* out, char4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_int_n1(char4* out, int in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_int_n2(char4* out, char4 in1, int in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_int_n3(int* out, char4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_long_n1(char4* out, long in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_long_n2(char4* out, char4 in1, long in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_long_n3(long* out, char4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_float_n1(char4* out, float in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_float_n2(char4* out, char4 in1, float in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_float_n3(float* out, char4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_dummy_n1(char4* out, Dummy in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_dummy_n2(char4* out, char4 in1, Dummy in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_dummy_n3(Dummy* out, char4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
    )"};

static constexpr auto kHipHcSub8pkVector{
    R"(
        __global__ void hip_hc_sub8pk_char4_n1(char4* out, char4 in1, char4 in2) { out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_char4_n2(char4* out, char4 in1, char4 in2) { *out = __hip_hc_sub8pk(&in1, in2); }
        __global__ void hip_hc_sub8pk_char4_n3(char4* out, char4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, &in2); }
        __global__ void hip_hc_sub8pk_char2_n1(char4* out, char2 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_char2_n2(char4* out, char4 in1, char2 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_char2_n3(char2* out, char4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_int4_n1(char4* out, int4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_int4_n2(char4* out, char4 in1, int4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_int4_n3(int4* out, char4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_long4_n1(char4* out, long4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_long4_n2(char4* out, char4 in1, long4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_long4_n3(long4* out, char4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_float4_n1(char4* out, float4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_float4_n2(char4* out, char4 in1, float4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
        __global__ void hip_hc_sub8pk_float4_n3(float4* out, char4 in1, char4 in2) { *out = __hip_hc_sub8pk(in1, in2); }
    )"};

static constexpr auto kHipHcMul8pkBasic{
    R"(
        struct Dummy {
          __device__ Dummy() {}
          __device__ ~Dummy() {}
        };

        __global__ void hip_hc_mul8pk_char_n1(char4* out, char in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_char_n2(char4* out, char4 in1, char in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_char_n3(char* out, char4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_int_n1(char4* out, int in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_int_n2(char4* out, char4 in1, int in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_int_n3(int* out, char4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_long_n1(char4* out, long in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_long_n2(char4* out, char4 in1, long in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_long_n3(long* out, char4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_float_n1(char4* out, float in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_float_n2(char4* out, char4 in1, float in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_float_n3(float* out, char4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_dummy_n1(char4* out, Dummy in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_dummy_n2(char4* out, char4 in1, Dummy in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_dummy_n3(Dummy* out, char4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
    )"};

static constexpr auto kHipHcMul8pkVector{
    R"(
        __global__ void hip_hc_mul8pk_char4_n1(char4* out, char4 in1, char4 in2) { out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_char4_n2(char4* out, char4 in1, char4 in2) { *out = __hip_hc_mul8pk(&in1, in2); }
        __global__ void hip_hc_mul8pk_char4_n3(char4* out, char4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, &in2); }
        __global__ void hip_hc_mul8pk_char2_n1(char4* out, char2 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_char2_n2(char4* out, char4 in1, char2 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_char2_n3(char2* out, char4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_int4_n1(char4* out, int4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_int4_n2(char4* out, char4 in1, int4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_int4_n3(int4* out, char4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_long4_n1(char4* out, long4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_long4_n2(char4* out, char4 in1, long4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_long4_n3(long4* out, char4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_float4_n1(char4* out, float4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_float4_n2(char4* out, char4 in1, float4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
        __global__ void hip_hc_mul8pk_float4_n3(float4* out, char4 in1, char4 in2) { *out = __hip_hc_mul8pk(in1, in2); }
    )"};