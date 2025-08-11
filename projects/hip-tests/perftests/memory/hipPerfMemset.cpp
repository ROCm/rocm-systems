/*
 Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

/* HIT_START
 * BUILD: %t %s ../../src/test_common.cpp
 * TEST: %t
 * HIT_END
 */

#include "test_common.h"
#include <iostream>
#include <chrono>

static unsigned int sizeList[] = {
  256, 512, 1024, 2048, 4096, 8192,
};

static unsigned int eleNumList[] = {
    0x100, 0x400, 0x1000, 0x4000, 0x10000, 0x20000, 0x40000, 0x80000, 0x100000,
    0x200000, 0x400000, 0x800000, 0x1000000
};

typedef struct _dataType {
char memsetval = 0x42;
char memsetD8val = 0xDE;
int16_t memsetD16val = 0xDEAD;
int memsetD32val = 0xDEADBEEF;
}dataType;

#define NUM_ITER 1000

enum MemsetType {
  hipMemsetTypeDefault,
  hipMemsetTypeD8,
  hipMemsetTypeD16,
  hipMemsetTypeD32,
  hipMemsetTypeMax

};

using namespace std;

class hipPerfMemset {
  private:
    uint64_t     bufSize_;
    unsigned int num_elements_;
    unsigned int testNumEle_;
    unsigned int _numSubTests = 0;
    unsigned int _numSubTests2D = 0;
    unsigned int _numSubTests3D = 0;
    unsigned int num_sizes_ =0;

  public:
    hipPerfMemset() {
    num_elements_ = sizeof(eleNumList) / sizeof(unsigned int);
    _numSubTests = num_elements_ * hipMemsetTypeMax;

    num_sizes_ = sizeof(sizeList) / sizeof(unsigned int);
    _numSubTests2D = num_sizes_;
    _numSubTests3D = _numSubTests2D;
    };

    ~hipPerfMemset() {};

    void open(int deviceID);

    template<typename T>
    void run1D(unsigned int test, T memsetval, enum MemsetType type, bool async);

    template<typename T>
    void run2D(unsigned int test, T memsetval, enum MemsetType type, bool async);

    template<typename T>
    void run3D(unsigned int test, T memsetval, enum MemsetType type, bool async);

    uint getNumTests() {
      return _numSubTests;
    }

    uint getNumTests2D() {
      return _numSubTests2D;
    }
    uint getNumTests3D() {
      return _numSubTests3D;
    }
};


void hipPerfMemset::open(int deviceId) {
  int nGpu = 0;
  HIPCHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    failed("No GPU!");
  }

  HIPCHECK(hipSetDevice(deviceId));
  hipDeviceProp_t props = {0};
  HIPCHECK(hipGetDeviceProperties(&props, deviceId));
  std::cout << "info: running on bus " << "0x" << props.pciBusID << " " << props.name
            << " with " << props.multiProcessorCount << " CUs" << " and device id: " << deviceId
            << std::endl;
}

template<typename T>
void hipPerfMemset::run1D(unsigned int test, T memsetval, enum MemsetType type, bool async) {

  T * A_h;
  T * A_d;

  testNumEle_ = eleNumList[test % num_elements_];

  bufSize_ = testNumEle_ * sizeof(uint32_t);

  HIPCHECK(hipMalloc(&A_d, bufSize_));

  A_h = reinterpret_cast<T*> (malloc(bufSize_));

  hipStream_t stream;
  HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  // Warm-up
  if (async) {
    HIPCHECK(hipMemsetAsync((void *)A_d, memsetval, bufSize_, stream));
    HIPCHECK(hipStreamSynchronize(stream));
  } else {
    HIPCHECK(hipMemset((void *)A_d, memsetval, bufSize_));
    HIPCHECK(hipDeviceSynchronize());
  }

  auto start = chrono::high_resolution_clock::now();
  for (uint i = 0; i < NUM_ITER; i++) {
    if (type == hipMemsetTypeDefault && !async) {
      HIPCHECK(hipMemset((void *)A_d, memsetval, bufSize_));
    }
    else if (type == hipMemsetTypeDefault && async) {
      HIPCHECK(hipMemsetAsync(A_d, memsetval, bufSize_, stream));
    }
    else if (type == hipMemsetTypeD8 && !async){
      HIPCHECK(hipMemsetD8((hipDeviceptr_t)A_d, memsetval, bufSize_));
    }
    else if (type == hipMemsetTypeD8 && async) {
      HIPCHECK(hipMemsetD8Async((hipDeviceptr_t)A_d, memsetval, bufSize_, stream));
    }
    else if (type == hipMemsetTypeD16 && !async) {
      HIPCHECK(hipMemsetD16((hipDeviceptr_t)A_d, memsetval, bufSize_/sizeof(T)));
    }
    else if (type == hipMemsetTypeD16 && async) {
      HIPCHECK(hipMemsetD16Async((hipDeviceptr_t)A_d, memsetval, bufSize_/sizeof(T), stream));
    }
    else if (type == hipMemsetTypeD32 && !async) {
      HIPCHECK(hipMemsetD32((hipDeviceptr_t)A_d, memsetval, bufSize_/sizeof(T)));
    }
    else if (type == hipMemsetTypeD32 && async) {
      HIPCHECK(hipMemsetD32Async((hipDeviceptr_t)A_d, memsetval, bufSize_/sizeof(T), stream));
    }
  }
  if (async) {
    HIPCHECK(hipStreamSynchronize(stream));
  } else {
    HIPCHECK(hipDeviceSynchronize());
  }

  auto end = chrono::high_resolution_clock::now();

  HIPCHECK(hipMemcpy(A_h, A_d, bufSize_, hipMemcpyDeviceToHost) );

  for (int i = 0; i < bufSize_ / sizeof(T); i++) {
    if (A_h[i] != memsetval) {
      cout << "mismatch at index " << i << " computed: " << static_cast<int> (A_h[i])
           << ", memsetval: " << static_cast<int> (memsetval) << endl;
      break;
    }
  }

  HIPCHECK(hipFree(A_d));
  free(A_h);

  auto diff = std::chrono::duration<double>(end - start);
  auto sec = diff.count();

  auto perf = static_cast<double>((bufSize_ * NUM_ITER * (double)(1e-09)) / sec);

  cout <<  "[" << setw(2) << test << "] " << setw(5) << bufSize_/1024 << " Kb " << setw(4)
       << " typeSize " << (int)sizeof(T) << " : " << setw(7) << perf << " GB/s " << endl;
}

template<typename T>
void hipPerfMemset::run2D(unsigned int test, T memsetval, enum MemsetType type, bool async) {

  bufSize_ = sizeList[test % num_sizes_];

  size_t numH = bufSize_;
  size_t numW = bufSize_;
  size_t pitch_A;
  size_t width = numW * sizeof(char);
  size_t sizeElements = width * numH;
  size_t elements = numW* numH;

  T * A_h;
  T * A_d;

  HIPCHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width ,
                          numH));
  A_h = reinterpret_cast<char*>(malloc(sizeElements));

  for (size_t i=0; i < elements; i++) {
    A_h[i] = 1;
  }

  hipStream_t stream;
  HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

  // Warm-up
  if (async) {
    HIPCHECK(hipMemset2DAsync(A_d, pitch_A, memsetval, numW, numH, stream));
    HIPCHECK(hipStreamSynchronize(stream));
  } else {
    HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, numW, numH));
    HIPCHECK(hipDeviceSynchronize());
  }

  auto start = chrono::steady_clock::now();

  for (uint i = 0; i < NUM_ITER; i++) {
    if (type == hipMemsetTypeDefault && !async) {
    HIPCHECK(hipMemset2D(A_d, pitch_A, memsetval, numW, numH));
    }
    else if (type == hipMemsetTypeDefault && async) {
      HIPCHECK(hipMemset2DAsync(A_d, pitch_A, memsetval, numW, numH, stream));
    }
  }

  if (async) {
    HIPCHECK(hipStreamSynchronize(stream));
  } else {
    HIPCHECK(hipDeviceSynchronize());
  }

  auto end = chrono::steady_clock::now();

  HIPCHECK(hipMemcpy2D(A_h, width, A_d, pitch_A, numW, numH,
                       hipMemcpyDeviceToHost));

  for (int i=0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      cout << "mismatch at index " << i << " computed: " << static_cast<int> (A_h[i])
           << ", memsetval: " << static_cast<int> (memsetval) << endl;
      break;
    }
  }

  chrono::duration<double> diff = end - start;

  auto sec = diff.count();

  auto perf = static_cast<double>((sizeElements* NUM_ITER * (double)(1e-09)) / sec);

  cout << " hipPerf2DMemset" << (async ? "Async" : "     ") << "[" << test << "] "
       << "  " << "(GB/s) for " << setw(5) << bufSize_
       << " x " << setw(5) << bufSize_ << " bytes : " << setw(7) << perf <<  endl;

  HIPCHECK(hipStreamDestroy(stream));
  HIPCHECK(hipFree(A_d));
  free(A_h);
}

template<typename T>
void hipPerfMemset::run3D(unsigned int test, T memsetval, enum MemsetType type, bool async) {

    bufSize_ = sizeList[test % num_sizes_];

    size_t numH = bufSize_;
    size_t numW = bufSize_;
    size_t depth = 10;
    size_t width = numW * sizeof(char);
    size_t sizeElements = width * numH * depth;
    size_t elements = numW* numH* depth;

    hipStream_t stream;
    HIPCHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    T *A_h;

    hipExtent extent = make_hipExtent(width, numH, depth);
    hipPitchedPtr devPitchedPtr;

    HIPCHECK(hipMalloc3D(&devPitchedPtr, extent));
    A_h = (char*)malloc(sizeElements);
    HIPASSERT(A_h != NULL);

    for (size_t i=0; i<elements; i++) {
        A_h[i] = 1;
    }

  // Warm-up
  if (async) {
    HIPCHECK(hipMemset3DAsync( devPitchedPtr, memsetval, extent, stream));
    HIPCHECK(hipStreamSynchronize(stream));
  } else {
    HIPCHECK(hipMemset3D( devPitchedPtr, memsetval, extent));
    HIPCHECK(hipDeviceSynchronize());
  }
   auto start = chrono::steady_clock::now();

   for (uint i = 0; i < NUM_ITER; i++) {
     if (type == hipMemsetTypeDefault && !async) {
       HIPCHECK(hipMemset3D( devPitchedPtr, memsetval, extent));
     }
     else if (type == hipMemsetTypeDefault && async) {
       HIPCHECK(hipMemset3DAsync(devPitchedPtr, memsetval, extent, stream));
     }
   }

  if (async) {
    HIPCHECK(hipStreamSynchronize(stream));
  } else {
    HIPCHECK(hipDeviceSynchronize());
  }

  auto end = chrono::steady_clock::now();

  hipMemcpy3DParms myparms = {0};
  myparms.srcPos = make_hipPos(0,0,0);
  myparms.dstPos = make_hipPos(0,0,0);
  myparms.dstPtr = make_hipPitchedPtr(A_h, width , numW, numH);
  myparms.srcPtr = devPitchedPtr;
  myparms.extent = extent;

  myparms.kind = hipMemcpyDeviceToHost;

  HIPCHECK(hipMemcpy3D(&myparms));

  for (int i=0; i<elements; i++) {
    if (A_h[i] != memsetval) {
      cout << "mismatch at index " << i << " computed: " << static_cast<int> (A_h[i])
           << ", memsetval: " << static_cast<int> (memsetval) << endl;
      break;
      }
  }

  chrono::duration<double> diff = end - start;

  auto sec = diff.count();

  auto perf = static_cast<double>((sizeElements * NUM_ITER * (double)(1e-09)) / sec);

  cout << " hipPerf3DMemset" << (async ? "Async" : "     ") << "[" << test << "] " << "  "
       <<  "(GB/s) for " << setw(5) << bufSize_ << " x " << setw(5)
       << bufSize_  << " x " << depth << " bytes : " << setw(7) << perf <<  endl;
  HIPCHECK(hipFree(devPitchedPtr.ptr));
  free(A_h);
}

int main() {
  hipPerfMemset hipPerfMemset;

  dataType pattern;
  int deviceId = 0;
  hipPerfMemset.open(deviceId);
  MemsetType type;

  int numTests = hipPerfMemset.getNumTests();
  int numTests2D = hipPerfMemset.getNumTests2D();
  int numTests3D = hipPerfMemset.getNumTests3D();


  cout << "--------------------- 1D buffer -------------------" << endl;
  bool async= false;
  for (uint i = 0; i < 2 ; i++) {
    cout << endl;

    for (auto testCase = 0; testCase < numTests; testCase++) {
      if (testCase < sizeof(eleNumList) / sizeof(uint32_t)) {
        cout << "API: hipMemsetD8" << (async ? "Async " : "      ");
        hipPerfMemset.run1D(testCase, pattern.memsetval, hipMemsetTypeD8, async);
      }

      else if (testCase < 2 * sizeof(eleNumList) / sizeof(uint32_t)) {
        cout << "API: hipMemsetD16" << (async ? "Async" : "     ");
        hipPerfMemset.run1D(testCase,pattern.memsetD16val, hipMemsetTypeD16, async);
      }

      else if (testCase < 3 * sizeof(eleNumList) / sizeof(uint32_t)) {
        cout << "API: hipMemsetD32" << (async ? "Async" : "     ");
        hipPerfMemset.run1D(testCase,pattern.memsetD32val, hipMemsetTypeD32, async);
      }

      else {
        cout << "API: hipMemset" << (async ? "Async   " : "        ");
        hipPerfMemset.run1D(testCase,pattern.memsetval, hipMemsetTypeDefault, async);
      }
    }
    async = true;
  }

  cout << endl;
  cout << "------------------ 2D buffer arrays ---------------" << endl;

  async = false;
  for (uint i = 0; i < 2; i++) {
    cout << endl;
    for (uint test = 0; test < numTests2D; test++) {
      hipPerfMemset.run2D(test, pattern.memsetval, hipMemsetTypeDefault, async);
    }
    async = true;
  }

  cout << endl;
  cout << "------------------ 3D buffer arrays ---------------" << endl;

  async = false;
  for (uint i = 0; i < 2; i++) {
    cout << endl;
    for (uint test =0; test < numTests3D; test++) {
      hipPerfMemset.run3D(test, pattern.memsetval, hipMemsetTypeDefault, async);
    }
    async = true;
  }

  passed();
}
