/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Kernels whose metadata hipCodeObjectMetadata.cc checks against values that follow from this
// source: argument offsets and sizes, static LDS, launch bounds, scratch, and printf strings
// (compiled with -mprintf-kind=buffered so the formats are in amdhsa.printf).

#include <hip/hip_runtime.h>

//! 8 bytes, 4-byte aligned
struct Pair {
  int a;
  char b;
};

//! 24 bytes, 8-byte aligned
struct Big {
  double x[3];
};

// Explicit arguments at offsets c:0 s:2 i:4 l:8 f:16 d:24 out:32 p:40 b:48 (72 bytes). Thread 0
// writes back what it received.
extern "C" __global__ void args_layout(char c, short s, int i, long long l, float f, double d,
                                       long long* out, Pair p, Big b) {
  if (threadIdx.x != 0) return;
  out[0] = c;
  out[1] = s;
  out[2] = i;
  out[3] = l;
  out[4] = static_cast<long long>(f * 1000.0f);
  out[5] = static_cast<long long>(d * 1000.0);
  out[6] = p.a;
  out[7] = p.b;
  out[8] = static_cast<long long>(b.x[0] + b.x[1] * 10.0 + b.x[2] * 100.0);
}

// 1024 bytes of static LDS; at most 128 threads per block
extern "C" __global__ void __launch_bounds__(128) shared_static(int* out) {
  __shared__ int buf[256];
  buf[threadIdx.x] = threadIdx.x;
  buf[threadIdx.x + 128] = 1;
  __syncthreads();
  out[threadIdx.x] = buf[127 - threadIdx.x] + buf[threadIdx.x + 128];
}

// No arguments at all
extern "C" __global__ void no_args() {}

// A dynamically indexed private array: at least 256 bytes of scratch per thread
extern "C" __global__ void scratch(int* out, int n) {
  volatile int a[64];
  for (int k = 0; k < 64; ++k) a[k] = k * n;
  out[threadIdx.x] = a[(threadIdx.x + n) & 63];
}

// The target the code object was built for, e.g. "gfx1100" or "gfx11-generic", so a test can
// tell which entry of a fat binary the runtime loaded
extern "C" __global__ void target_id(char* out) {
#if defined(__HIP_DEVICE_COMPILE__)
  const char id[] = __amdgcn_target_id__;
  for (unsigned k = 0; k < sizeof(id); ++k) out[k] = id[k];
#endif
}

// Two printf formats, so the printf IDs and their argument sizes are exercised
extern "C" __global__ void print_values(int v, double d) {
  if (threadIdx.x != 0) return;
  printf("metadata printf %d %.2f %s\n", v, d, "ok");
  printf("metadata second %ld\n", static_cast<long>(v) * 3);
}
