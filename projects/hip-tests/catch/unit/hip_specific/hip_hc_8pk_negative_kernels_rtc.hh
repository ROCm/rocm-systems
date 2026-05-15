/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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