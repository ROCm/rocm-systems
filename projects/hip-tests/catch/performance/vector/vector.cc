/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>
#include <performance_common.hh>
#include "hip_test_common.hh"
#include "resource_guards.hh"

// This benchmark tries to compare the performance of using HIP_Vector::operator[] as opposed to
// to use the C++ native array operator[] on the raw data pointer (get_native_vector).
// Although the operator's implementation is trivial, there could be overhead depending how well
// the compiler is able to optimize away the function call
static constexpr int NumVectorElems = 4;
using TestType = unsigned int;
using VectorType = uint4;
static constexpr size_t NumVectors = 32 * 1024 * 1024;
static constexpr int BlockSize = 256;
static constexpr int GridSize = 128 * 1024;

// uses HIP_Vector::operator[]
__global__ void kernelSquareBracket(VectorType* output, unsigned int value) {
  int pos = threadIdx.x + blockIdx.x * BlockSize;
  __shared__ VectorType buffer[BlockSize];
  // intentionally get to 64KB LDS usage
  __shared__ unsigned char dummy[64 * 1024 - sizeof(buffer)];
  VectorType tmp;

  for (int j = 0; j < NumVectorElems; j++) {
    buffer[threadIdx.x][j] = pos + value + j;
  }

  dummy[0] = 1;

  for (int m = 1; m < sizeof(dummy); m++) {
    dummy[m] = dummy[m - 1] + value;
  }

  tmp = buffer[threadIdx.x];
  __syncthreads();

  for (int i = 0; i < BlockSize; i++) {
    for (int j = 0; j < NumVectorElems; j++) {
      tmp[j] += 1;
      for (int k = 0; k < 1000; k++) tmp[j] ^= i + j + value;
    }
  }

  output[pos] = tmp + dummy[sizeof(dummy) - 1];
}

// uses get_native_vector
__global__ void kernelGetNativeVector(VectorType* output, unsigned int value) {
  int pos = threadIdx.x + blockIdx.x * BlockSize;
  __shared__ VectorType buffer[BlockSize];
    // intentionally get to 64KB LDS usage
  __shared__ unsigned char dummy[64 * 1024 - sizeof(buffer)];
  VectorType tmp;

  for (int j = 0; j < NumVectorElems; j++) {
    get_native_vector(buffer[threadIdx.x])[j] = pos + value + j;
  }

  dummy[0] = 1;

  for (int m = 1; m < sizeof(dummy); m++) {
    dummy[m] = dummy[m - 1] + value;
  }

  tmp = buffer[threadIdx.x];
  __syncthreads();

  for (int i = 0; i < BlockSize; i++) {
    for (int j = 0; j < NumVectorElems; j++) {
      get_native_vector(tmp)[j] += 1;
      for (int k = 0; k < 1000; k++) get_native_vector(tmp)[j] ^= i + j + value;
    }
  }

  for (int j = 0; j < NumVectorElems; j++) {
    get_native_vector(output[pos])[j] = tmp[j] + dummy[sizeof(dummy) - 1];
  }

}

struct SquareBracketBenchmark : public Benchmark<SquareBracketBenchmark> {
  void operator()(VectorType* output, unsigned int value) {
    dim3 blockDim { BlockSize };
    dim3 gridDim { GridSize };
    TIMED_SECTION(kTimerTypeEvent) {
      kernelSquareBracket<<<gridDim, blockDim, 0, 0>>>(output, value);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
    }
  }
};

struct GetNativeVectorBenchmark : public Benchmark<GetNativeVectorBenchmark> {
  void operator()(VectorType* output, unsigned int value) {
    dim3 blockDim{BlockSize};
    dim3 gridDim{GridSize};

    TIMED_SECTION(kTimerTypeEvent) {
      kernelGetNativeVector<<<gridDim, blockDim, 0, 0>>>(output, value);
      HIP_CHECK(hipGetLastError());
      HIP_CHECK(hipDeviceSynchronize());
    }
  }
};


TEST_CASE("Performance_Hip_Vec_Access")
{
  LinearAllocGuard<VectorType> d_outputBracket(LinearAllocs::hipMalloc, sizeof(VectorType) * NumVectors);
  LinearAllocGuard<VectorType> d_outputNative(LinearAllocs::hipMalloc, sizeof(VectorType) * NumVectors);
  unsigned int value = rand();
  SquareBracketBenchmark benchmark1;
  GetNativeVectorBenchmark benchmark2;

  printf("--- square bracket ---\n");
  benchmark1.Run(d_outputBracket.ptr(), value);
  printf("--- get_native_vector ---\n");
  benchmark2.Run(d_outputNative.ptr(), value);

  {
    LinearAllocGuard<VectorType> outputBracket(LinearAllocs::malloc, d_outputBracket.size_bytes());
    LinearAllocGuard<VectorType> outputNative(LinearAllocs::malloc, d_outputNative.size_bytes());
    printf("Checking results...\n");

    HIP_CHECK(hipMemcpy(outputBracket.ptr(), d_outputBracket.ptr(), d_outputBracket.size_bytes(), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(outputNative.ptr(), d_outputNative.ptr(), d_outputNative.size_bytes(), hipMemcpyDeviceToHost));
    REQUIRE(!memcmp(outputBracket.ptr(), outputNative.ptr(), outputBracket.size_bytes()));
  }
}
