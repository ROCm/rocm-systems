/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/*
Negative kernels used for the atomics negative Test Cases that are using RTC.
*/

static constexpr auto kAtomicInc_uint{
    R"(
    __global__ void atomicInc_uint_v1(unsigned int* address, unsigned int* result) {
      *result = atomicInc(&address, 1234);
    }

    __global__ void atomicInc_uint_v2(unsigned int* address, unsigned int* result) {
      *result = atomicInc(address, address);
    }

    __global__ void atomicInc_uint_v3(unsigned int* address, unsigned int* result) {
      *result = atomicInc(1234, 1234);
    }

    class Dummy {
     public:
      __device__ Dummy() {}
      __device__ ~Dummy() {}
    };

    __global__ void atomicInc_uint_v4(Dummy* address, unsigned int* result) {
      *result = atomicInc(address, 1234);
    }

    __global__ void atomicInc_uint_v5(char* address, unsigned int* result) {
      *result = atomicInc(address, 1234);
    }

    __global__ void atomicInc_uint_v6(short* address, unsigned int* result) {
      *result = atomicInc(address, 1234);
    }

    __global__ void atomicInc_uint_v7(long* address, unsigned int* result) {
      *result = atomicInc(address, 1234);
    }

    __global__ void atomicInc_uint_v8(long long* address, unsigned int* result) {
      *result = atomicInc(address, 1234);
    }
  )"};