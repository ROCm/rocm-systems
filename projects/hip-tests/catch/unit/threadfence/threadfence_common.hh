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

enum class ThreadfenceScope { kBlock, kDevice, kSystem };

template <ThreadfenceScope scope> __device__ void Threadfence() {
  if constexpr (scope == ThreadfenceScope::kBlock) {
    __threadfence_block();
  } else if constexpr (scope == ThreadfenceScope::kDevice) {
    __threadfence();
  } else if constexpr (scope == ThreadfenceScope::kSystem) {
    __threadfence_system();
  }
}

static constexpr int kInitVal1 = 1, kInitVal2 = 2;
static constexpr int kSetVal1 = 10, kSetVal2 = 20;

template <ThreadfenceScope scope> __host__ __device__ void Write(volatile int* in) {
  in[0] = kSetVal1;
#ifdef __HIP_DEVICE_COMPILE__
  Threadfence<scope>();
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
  in[1] = kSetVal2;
}

template <ThreadfenceScope scope>
__host__ __device__ void Read(volatile int* out, volatile int* in) {
  out[1] = in[1];
#ifdef __HIP_DEVICE_COMPILE__
  Threadfence<scope>();
#else
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
  out[0] = in[0];
}

template <ThreadfenceScope scope, bool use_shared_mem>
__device__ void ThreadfenceTest(int* out, int* in) {
  if constexpr (scope == ThreadfenceScope::kBlock || use_shared_mem) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      Write<scope>(in);
    } else if (threadIdx.x == 1 && blockIdx.x == 0) {
      Read<scope>(out, in);
    }
  } else if constexpr (scope == ThreadfenceScope::kDevice) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      Write<scope>(in);
    } else if (threadIdx.x == 0 && blockIdx.x == 1) {
      Read<scope>(out, in);
    }
  }
}

template <ThreadfenceScope scope, bool use_shared_mem>
__global__ void ThreadfenceTestKernel(int* out, int* in) {
  extern __shared__ int shared_mem[];

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int *out_mem = out, *in_mem = in;

  if constexpr (use_shared_mem) {
    if (tid == 0) {
      in_mem = &shared_mem[0];
      out_mem = &shared_mem[2];

      in_mem[0] = in[0];
      in_mem[1] = in[1];
    }

    __syncthreads();
  }

  ThreadfenceTest<scope, use_shared_mem>(out_mem, in_mem);

  if constexpr (use_shared_mem) {
    __syncthreads();

    if (tid == 0) {
      out[0] = out_mem[0];
      out[1] = out_mem[1];
    }
  }
}