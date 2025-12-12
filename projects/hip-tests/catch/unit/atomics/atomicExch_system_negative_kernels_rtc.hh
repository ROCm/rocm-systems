/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

static constexpr auto kAtomicExchSystemInt{
    R"(
        struct Dummy {
          __device__ Dummy() {}
          __device__ ~Dummy() {}
        };

        __global__ void atomicExch_system_int_n1(int* p, int v) { atomicExch_system(p, p); }
        __global__ void atomicExch_system_int_n2(int* p, int v) { atomicExch_system(&p, v); }
        __global__ void atomicExch_system_int_n3(char* p, int v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_int_n4(short* p, int v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_int_n5(long* p, int v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_int_n6(long long* p, int v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_int_n7(Dummy* p, int v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_int_n8(int* p, Dummy v) { atomicExch_system(p, v); }
    )"};

static constexpr auto kAtomicExchSystemUnsignedInt{
    R"(
        struct Dummy {
          __device__ Dummy() {}
          __device__ ~Dummy() {}
        };

        __global__ void atomicExch_system_unsigned_int_n1(unsigned int* p, unsigned int v) {
          atomicExch_system(p, p);
        }
        __global__ void atomicExch_system_unsigned_int_n2(unsigned int* p, unsigned int v) {
          atomicExch_system(&p, v);
        }
        __global__ void atomicExch_system_unsigned_int_n3(char* p, unsigned int v) {
          atomicExch_system(p, v);
        }
        __global__ void atomicExch_system_unsigned_int_n4(short* p, unsigned int v) {
          atomicExch_system(p, v);
        }
        __global__ void atomicExch_system_unsigned_int_n5(long* p, unsigned int v) {
          atomicExch_system(p, v);
        }
        __global__ void atomicExch_system_unsigned_int_n6(long long* p, unsigned int v) {
          atomicExch_system(p, v);
        }
        __global__ void atomicExch_system_unsigned_int_n7(Dummy* p, unsigned int v) {
          atomicExch_system(p, v);
        }
        __global__ void atomicExch_system_unsigned_int_n8(unsigned int* p, Dummy v) {
          atomicExch_system(p, v);
        }
    )"};

static constexpr auto kAtomicExchSystemULL{
    R"(
        struct Dummy {
          __device__ Dummy() {}
          __device__ ~Dummy() {}
        };

        __global__ void atomicExch_system_unsigned_long_long_n1(unsigned long long* p,
                                                                unsigned long long v) {
          atomicExch_system(p, p);
        }
        __global__ void atomicExch_system_unsigned_long_long_n2(unsigned long long* p,
                                                                unsigned long long v) {
          atomicExch_system(&p, v);
        }
        __global__ void atomicExch_system_unsigned_long_long_n3(char* p, unsigned long long v) {
          atomicExch_system(p, v);
        }
        __global__ void atomicExch_system_unsigned_long_long_n4(short* p, unsigned long long v) {
          atomicExch_system(p, v);
        }
        __global__ void atomicExch_system_unsigned_long_long_n5(long* p, unsigned long long v) {
          atomicExch_system(p, v);
        }
        __global__ void atomicExch_system_unsigned_long_long_n6(long long* p, unsigned long long v) {
          atomicExch_system(p, v);
        }
        __global__ void atomicExch_system_unsigned_long_long_n7(Dummy* p, unsigned long long v) {
          atomicExch_system(p, v);
        }
        __global__ void atomicExch_system_unsigned_long_long_n8(unsigned long long* p, Dummy v) {
          atomicExch_system(p, v);
        }
    )"};

static constexpr auto kAtomicExchSystemFloat{
    R"(
        struct Dummy {
          __device__ Dummy() {}
          __device__ ~Dummy() {}
        };

        __global__ void atomicExch_system_float_n1(float* p, float v) { atomicExch_system(p, p); }
        __global__ void atomicExch_system_float_n2(float* p, float v) { atomicExch_system(&p, v); }
        __global__ void atomicExch_system_float_n3(char* p, float v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_float_n4(short* p, float v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_float_n5(long* p, float v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_float_n6(long long* p, float v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_float_n7(Dummy* p, float v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_float_n8(float* p, Dummy v) { atomicExch_system(p, v); }
    )"};

static constexpr auto kAtomicExchSystemDouble{
    R"(
        struct Dummy {
          __device__ Dummy() {}
          __device__ ~Dummy() {}
        };

        __global__ void atomicExch_system_double_n1(double* p, double v) { atomicExch_system(p, p); }
        __global__ void atomicExch_system_double_n2(double* p, double v) { atomicExch_system(&p, v); }
        __global__ void atomicExch_system_double_n3(char* p, double v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_double_n4(short* p, double v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_double_n5(long* p, double v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_double_n6(long long* p, double v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_double_n7(Dummy* p, double v) { atomicExch_system(p, v); }
        __global__ void atomicExch_system_double_n8(double* p, Dummy v) { atomicExch_system(p, v); }
    )"};