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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
