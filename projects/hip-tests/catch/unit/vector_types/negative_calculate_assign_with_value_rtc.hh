/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

static constexpr auto kCalculateAssignChar{R"(
  __global__ void CalculateAssignDevice(char1* vector_dev_ptr, decltype(char1().x) value) {
    char1 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(char1& vector_host, decltype(char1().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(uchar1* vector_dev_ptr, decltype(uchar1().x) value) {
    uchar1 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(uchar1& vector_host, decltype(uchar1().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(char2* vector_dev_ptr, decltype(char2().x) value) {
    char2 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(char2& vector_host, decltype(char2().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(uchar2* vector_dev_ptr, decltype(uchar2().x) value) {
    uchar2 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(uchar2& vector_host, decltype(uchar2().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(char3* vector_dev_ptr, decltype(char3().x) value) {
    char3 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(char3& vector_host, decltype(char3().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(uchar3* vector_dev_ptr, decltype(uchar3().x) value) {
    uchar3 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(uchar3& vector_host, decltype(uchar3().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(char4* vector_dev_ptr, decltype(char4().x) value) {
    char4 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(char4& vector_host, decltype(char4().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(uchar4* vector_dev_ptr, decltype(uchar4().x) value) {
    uchar4 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(uchar4& vector_host, decltype(uchar4().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }
)"};

static constexpr auto kCalculateAssignShort{R"(
  __global__ void CalculateAssignDevice(short1* vector_dev_ptr, decltype(short1().x) value) {
    short1 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(short1& vector_host, decltype(short1().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(ushort1* vector_dev_ptr, decltype(ushort1().x) value) {
    ushort1 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(ushort1& vector_host, decltype(ushort1().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(short2* vector_dev_ptr, decltype(short2().x) value) {
    short2 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(short2& vector_host, decltype(short2().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(ushort2* vector_dev_ptr, decltype(ushort2().x) value) {
    ushort2 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(ushort2& vector_host, decltype(ushort2().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(short3* vector_dev_ptr, decltype(short3().x) value) {
    short3 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(short3& vector_host, decltype(short3().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(ushort3* vector_dev_ptr, decltype(ushort3().x) value) {
    ushort3 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(ushort3& vector_host, decltype(ushort3().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(short4* vector_dev_ptr, decltype(short4().x) value) {
    short4 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(short4& vector_host, decltype(short4().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(ushort4* vector_dev_ptr, decltype(ushort4().x) value) {
    ushort4 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(ushort4& vector_host, decltype(ushort4().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }
)"};

static constexpr auto kCalculateAssignInt{R"(
  __global__ void CalculateAssignDevice(int1* vector_dev_ptr, decltype(int1().x) value) {
    int1 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(int1& vector_host, decltype(int1().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(uint1* vector_dev_ptr, decltype(uint1().x) value) {
    uint1 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(uint1& vector_host, decltype(uint1().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(int2* vector_dev_ptr, decltype(int2().x) value) {
    int2 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(int2& vector_host, decltype(int2().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(uint2* vector_dev_ptr, decltype(uint2().x) value) {
    uint2 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(uint2& vector_host, decltype(uint2().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(int3* vector_dev_ptr, decltype(int3().x) value) {
    int3 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(int3& vector_host, decltype(int3().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(uint3* vector_dev_ptr, decltype(uint3().x) value) {
    uint3 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(uint3& vector_host, decltype(uint3().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(int4* vector_dev_ptr, decltype(int4().x) value) {
    int4 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(int4& vector_host, decltype(int4().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(uint4* vector_dev_ptr, decltype(uint4().x) value) {
    uint4 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(uint4& vector_host, decltype(uint4().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }
)"};

static constexpr auto kCalculateAssignLong{R"(
  __global__ void CalculateAssignDevice(long1* vector_dev_ptr, decltype(long1().x) value) {
    long1 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(long1& vector_host, decltype(long1().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(ulong1* vector_dev_ptr, decltype(ulong1().x) value) {
    ulong1 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(ulong1& vector_host, decltype(ulong1().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(long2* vector_dev_ptr, decltype(long2().x) value) {
    long2 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(long2& vector_host, decltype(long2().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(ulong2* vector_dev_ptr, decltype(ulong2().x) value) {
    ulong2 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(ulong2& vector_host, decltype(ulong2().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(long3* vector_dev_ptr, decltype(long3().x) value) {
    long3 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(long3& vector_host, decltype(long3().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(ulong3* vector_dev_ptr, decltype(ulong3().x) value) {
    ulong3 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(ulong3& vector_host, decltype(ulong3().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(long4* vector_dev_ptr, decltype(long4().x) value) {
    long4 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(long4& vector_host, decltype(long4().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(ulong4* vector_dev_ptr, decltype(ulong4().x) value) {
    ulong4 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(ulong4& vector_host, decltype(ulong4().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }
)"};

static constexpr auto kCalculateAssignLongLong{R"(
  __global__ void CalculateAssignDevice(longlong1* vector_dev_ptr, decltype(longlong1().x) value) {
    longlong1 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(longlong1& vector_host, decltype(longlong1().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(ulonglong1* vector_dev_ptr, decltype(ulonglong1().x) value) {
    ulonglong1 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(ulonglong1& vector_host, decltype(ulonglong1().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(longlong2* vector_dev_ptr, decltype(longlong2().x) value) {
    longlong2 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(longlong2& vector_host, decltype(longlong2().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(ulonglong2* vector_dev_ptr, decltype(ulonglong2().x) value) {
    ulonglong2 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(ulonglong2& vector_host, decltype(ulonglong2().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(longlong3* vector_dev_ptr, decltype(longlong3().x) value) {
    longlong3 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(longlong3& vector_host, decltype(longlong3().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(ulonglong3* vector_dev_ptr, decltype(ulonglong3().x) value) {
    ulonglong3 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(ulonglong3& vector_host, decltype(ulonglong3().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(longlong4* vector_dev_ptr, decltype(longlong4().x) value) {
    longlong4 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(longlong4& vector_host, decltype(longlong4().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }

  __global__ void CalculateAssignDevice(ulonglong4* vector_dev_ptr, decltype(ulonglong4().x) value) {
    ulonglong4 vector_dev = *vector_dev_ptr;
    vector_dev %= value;
    vector_dev ^= value;
    vector_dev |= value;
    vector_dev &= value;
    vector_dev >>= value;
    vector_dev <<= value;
  }
  void CalculateAssignHost(ulonglong4& vector_host, decltype(ulonglong4().x) value) {
    vector_host %= value;
    vector_host ^= value;
    vector_host |= value;
    vector_host &= value;
    vector_host >>= value;
    vector_host <<= value;
  }
)"};
