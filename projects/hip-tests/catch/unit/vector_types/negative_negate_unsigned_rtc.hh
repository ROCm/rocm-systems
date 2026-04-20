/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

static constexpr auto kNegateUnsignedChar{R"(
  __global__ void NegateDevice(uchar1* vector_dev_ptr) {
    uchar1 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(uchar1& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(uchar2* vector_dev_ptr) {
    uchar2 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(uchar2& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(uchar3* vector_dev_ptr) {
    uchar3 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(uchar3& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(uchar4* vector_dev_ptr) {
    uchar4 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(uchar4& vector_host) { vector_host = -vector_host; }
)"};

static constexpr auto kNegateUnsignedShort{R"(
  __global__ void NegateDevice(ushort1* vector_dev_ptr) {
    ushort1 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(ushort1& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(ushort2* vector_dev_ptr) {
    ushort2 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(ushort2& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(ushort3* vector_dev_ptr) {
    ushort3 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(ushort3& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(ushort4* vector_dev_ptr) {
    ushort4 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(ushort4& vector_host) { vector_host = -vector_host; }
)"};

static constexpr auto kNegateUnsignedInt{R"(
  __global__ void NegateDevice(uint1* vector_dev_ptr) {
    uint1 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(uint1& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(uint2* vector_dev_ptr) {
    uint2 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(uint2& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(uint3* vector_dev_ptr) {
    uint3 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(uint3& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(uint4* vector_dev_ptr) {
    uint4 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(uint4& vector_host) { vector_host = -vector_host; }
)"};

static constexpr auto kNegateUnsignedLong{R"(
  __global__ void NegateDevice(ulong1* vector_dev_ptr) {
    ulong1 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(ulong1& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(ulong2* vector_dev_ptr) {
    ulong2 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(ulong2& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(ulong3* vector_dev_ptr) {
    ulong3 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(ulong3& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(ulong4* vector_dev_ptr) {
    ulong4 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(ulong4& vector_host) { vector_host = -vector_host; }
)"};

static constexpr auto kNegateUnsignedLongLong{R"(
  __global__ void NegateDevice(ulonglong1* vector_dev_ptr) {
    ulonglong1 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(ulonglong1& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(ulonglong2* vector_dev_ptr) {
    ulonglong2 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(ulonglong2& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(ulonglong3* vector_dev_ptr) {
    ulonglong3 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(ulonglong3& vector_host) { vector_host = -vector_host; }

  __global__ void NegateDevice(ulonglong4* vector_dev_ptr) {
    ulonglong4 vector_dev = *vector_dev_ptr;
    vector_dev = -vector_dev;
  }
  void NegateHost(ulonglong4& vector_host) { vector_host = -vector_host; }
)"};
