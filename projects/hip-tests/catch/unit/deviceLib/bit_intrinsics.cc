/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip_test_common.hh>
#include <hip/device_functions.h>

#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <stdint.h>

// Common constants
#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define THREADS_PER_BLOCK_Z 1

#define NUM_TESTS 65
#define HI_INT 0xfacefeed
#define LO_INT 0xdeadbeef

// Helper function to get and log device properties
static void logDeviceProperties(hipDeviceProp_t* devProp) {
  HIP_CHECK(hipGetDeviceProperties(devProp, 0));
  INFO("System minor : " << devProp->minor);
  INFO("System major : " << devProp->major);
  INFO("agent prop name : " << devProp->name);
  INFO("hip Device prop succeeded");
}

// Helper structure to manage memory for 2D grid kernel tests
struct GridTestMemory {
  unsigned int* hostA;
  unsigned int* hostB;
  unsigned long long int* hostC;
  unsigned long long int* hostD;
  unsigned int* deviceA;
  unsigned int* deviceB;
  unsigned long long int* deviceC;
  unsigned long long int* deviceD;
  int num;

  GridTestMemory(int n) : num(n) {
    hostA = (unsigned int*)malloc(num * sizeof(unsigned int));
    hostB = (unsigned int*)malloc(num * sizeof(unsigned int));
    hostC = (unsigned long long int*)malloc(num * sizeof(unsigned long long int));
    hostD = (unsigned long long int*)malloc(num * sizeof(unsigned long long int));
    HIP_CHECK(hipMalloc((void**)&deviceA, num * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void**)&deviceB, num * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void**)&deviceC, num * sizeof(unsigned long long int)));
    HIP_CHECK(hipMalloc((void**)&deviceD, num * sizeof(unsigned long long int)));
  }

  ~GridTestMemory() {
    HIP_CHECK(hipFree(deviceA));
    HIP_CHECK(hipFree(deviceB));
    HIP_CHECK(hipFree(deviceC));
    HIP_CHECK(hipFree(deviceD));
    free(hostA);
    free(hostB);
    free(hostC);
    free(hostD);
  }
};


// CPU implementations for verification
template <typename T> T bitreverse(T num) {
  T count = sizeof(num) * 8 - 1;
  T reverse_num = num;

  num >>= 1;
  while (num) {
    reverse_num <<= 1;
    reverse_num |= num & 1;
    num >>= 1;
    count--;
  }
  reverse_num <<= count;
  return reverse_num;
}

unsigned int firstbit_u32(unsigned int a) {
  if (a == 0) {
    return 32;
  }
  unsigned int pos = 0;
  while ((int)a > 0) {
    a <<= 1;
    pos++;
  }
  return pos;
}

unsigned int firstbit_u64(unsigned long long int a) {
  if (a == 0) {
    return 64;
  }
  unsigned int pos = 0;
  while ((long long int)a > 0) {
    a <<= 1;
    pos++;
  }
  return pos;
}

template <typename T> unsigned lastbit(T a) {
  if (a == 0) return 0;
  unsigned pos = 1;
  while ((a & 1) != 1) {
    a >>= 1;
    pos++;
  }
  return pos;
}

template <typename T> unsigned int popcountCPU(T value) {
  unsigned int ret = 0;
  while (value) {
    if (value & 0x1) ++ret;
    value >>= 1;
  }
  return ret;
}

static unsigned int cpu_funnelshift_l(unsigned int lo, unsigned int hi, unsigned int shift) {
  uint64_t val = hi;
  val <<= 32;
  val |= lo;
  val <<= (shift & 31);
  val >>= 32;
  return val & 0xffffffff;
}

static unsigned int cpu_funnelshift_lc(unsigned int lo, unsigned int hi, unsigned int shift) {
  uint64_t val = hi;
  val <<= 32;
  val |= lo;
  if (shift > 32) shift = 32;
  val <<= shift;
  val >>= 32;
  return val & 0xffffffff;
}

static unsigned int cpu_funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift) {
  uint64_t val = hi;
  val <<= 32;
  val |= lo;
  val >>= (shift & 31);
  return val & 0xffffffff;
}

static unsigned int cpu_funnelshift_rc(unsigned int lo, unsigned int hi, unsigned int shift) {
  uint64_t val = hi;
  val <<= 32;
  val |= lo;
  if (shift > 32) shift = 32;
  val >>= shift;
  return val & 0xffffffff;
}

// Enum to specify which bit intrinsic to use
enum class BitIntrinsic {
  BREV,
  CLZ,
  FFS,
  POPC
};

// Unified kernel function that accepts the bit intrinsic as a template parameter
template <BitIntrinsic Intrinsic>
__global__ void bit_intrinsic_kernel(unsigned int* a, unsigned int* b, unsigned long long* c,
                                      unsigned long long int* d, int width, int height) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int i = y * width + x;
  if (i < (width * height)) {
    if constexpr (Intrinsic == BitIntrinsic::BREV) {
      a[i] = __brev(b[i]);
      c[i] = __brevll(d[i]);
    } else if constexpr (Intrinsic == BitIntrinsic::CLZ) {
      a[i] = __clz(b[i]);
      c[i] = __clzll(d[i]);
    } else if constexpr (Intrinsic == BitIntrinsic::FFS) {
      a[i] = __ffs(b[i]);
      c[i] = __ffsll(d[i]);
    } else if constexpr (Intrinsic == BitIntrinsic::POPC) {
      a[i] = __popc(b[i]);
      c[i] = __popcll(d[i]);
    }
  }
}

__device__ void test_ambiguity() {
  short s = 1;
  unsigned short us = 2;
  float f = 3;
  int i = 4;
  unsigned int ui = 5;
  __clz(f);
  __clz(s);
  __clz(us);
  __clzll(f);
  __clzll(i);
  __clzll(ui);
}

__global__ void gpu_ballot(unsigned int* device_ballot, unsigned Num_Warps_per_Block,
                           unsigned pshift) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int warp_num = threadIdx.x >> pshift;

#ifdef __HIP_PLATFORM_AMD__
  atomicAdd(&device_ballot[warp_num + blockIdx.x * Num_Warps_per_Block],
            __popcll(__ballot(tid - 245)));
#else
  atomicAdd(&device_ballot[warp_num + blockIdx.x * Num_Warps_per_Block],
            __popc(__ballot(tid - 245)));
#endif
}

__global__ void funnelshift_kernel(unsigned int* l_out, unsigned int* lc_out, unsigned int* r_out,
                                   unsigned int* rc_out) {
  for (int i = 0; i < NUM_TESTS; i++) {
    l_out[i] = __funnelshift_l(LO_INT, HI_INT, i);
    lc_out[i] = __funnelshift_lc(LO_INT, HI_INT, i);
    r_out[i] = __funnelshift_r(LO_INT, HI_INT, i);
    rc_out[i] = __funnelshift_rc(LO_INT, HI_INT, i);
  }
}

__global__ void mbcnt_kernel(unsigned int* mbcnt_lo, unsigned int* mbcnt_hi, unsigned int* lane_id) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  mbcnt_lo[x] = __builtin_amdgcn_mbcnt_lo(0xFFFFFFFF, 0);
  mbcnt_hi[x] = __builtin_amdgcn_mbcnt_hi(0xFFFFFFFF, 0);
  lane_id[x] = __lane_id();
}

TEST_CASE("Unit_bit_intrinsics") {
  hipDeviceProp_t devProp;
  logDeviceProperties(&devProp);

  SECTION("brev - bit reverse") {
    constexpr int WIDTH = 32;
    constexpr int HEIGHT = 32;
    constexpr int NUM = WIDTH * HEIGHT;

    GridTestMemory mem(NUM);

    // Initialize input data
    for (int i = 0; i < NUM; i++) {
      mem.hostB[i] = i;
      mem.hostD[i] = i;
    }

    // Copy input to device
    HIP_CHECK(hipMemcpy(mem.deviceB, mem.hostB, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(mem.deviceD, mem.hostD, NUM * sizeof(unsigned long long int),
                        hipMemcpyHostToDevice));

    // Launch kernel
    std::cout << "kernel launch\n";
    hipLaunchKernelGGL(bit_intrinsic_kernel<BitIntrinsic::BREV>,
                       dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, mem.deviceA, mem.deviceB,
                       mem.deviceC, mem.deviceD, WIDTH, HEIGHT);
    std::cout << "after kernel launch\n";
    HIP_CHECK(hipGetLastError());

    // Copy results back
    std::cout << "before memcpy\n";
    HIP_CHECK(hipMemcpy(mem.hostA, mem.deviceA, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(mem.hostC, mem.deviceC, NUM * sizeof(unsigned long long int),
                        hipMemcpyDeviceToHost));
    std::cout << "after memcpy\n";

    // Verify results
    int errors = 0;
    for (int i = 0; i < NUM; i++) {
      if (mem.hostA[i] != bitreverse(mem.hostB[i])) {
        errors++;
      }
    }
    for (int i = 0; i < NUM; i++) {
      if (mem.hostC[i] != bitreverse(mem.hostD[i])) {
        errors++;
      }
    }

    REQUIRE(errors == 0);
  }

  SECTION("clz - count leading zeros") {
    constexpr int WIDTH = 8;
    constexpr int HEIGHT = 8;
    constexpr int NUM = WIDTH * HEIGHT;

    GridTestMemory mem(NUM);

    // Initialize input data
    for (unsigned int i = 0; i < NUM; i++) {
      mem.hostB[i] = 419430 * i;
      mem.hostD[i] = i;
    }

    // Copy input to device
    HIP_CHECK(hipMemcpy(mem.deviceB, mem.hostB, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(mem.deviceD, mem.hostD, NUM * sizeof(unsigned long long int),
                        hipMemcpyHostToDevice));

    // Launch kernel
    hipLaunchKernelGGL(bit_intrinsic_kernel<BitIntrinsic::CLZ>,
                       dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, mem.deviceA, mem.deviceB,
                       mem.deviceC, mem.deviceD, WIDTH, HEIGHT);
    HIP_CHECK(hipGetLastError());

    // Copy results back
    HIP_CHECK(hipMemcpy(mem.hostA, mem.deviceA, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(mem.hostC, mem.deviceC, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    for (unsigned int i = 0; i < NUM; i++) {
      if (mem.hostA[i] != firstbit_u32(mem.hostB[i])) {
        INFO("Match Failed: " << mem.hostA[i] << " - " << firstbit_u32(mem.hostB[i]));
        errors++;
      }
    }
    for (unsigned int i = 0; i < NUM; i++) {
      if (mem.hostC[i] != firstbit_u64(mem.hostD[i])) {
        INFO("Match Failed: " << mem.hostC[i] << " - " << firstbit_u64(mem.hostD[i]));
        errors++;
      }
    }

    REQUIRE(errors == 0);
  }

  SECTION("ffs - find first set") {
    constexpr int WIDTH = 8;
    constexpr int HEIGHT = 8;
    constexpr int NUM = WIDTH * HEIGHT;

    GridTestMemory mem(NUM);

    // Initialize input data
    for (int i = 0; i < NUM; i++) {
      mem.hostB[i] = i;
      mem.hostD[i] = 1099511627776 + i;
    }

    // Copy input to device
    HIP_CHECK(hipMemcpy(mem.deviceB, mem.hostB, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(mem.deviceD, mem.hostD, NUM * sizeof(unsigned long long int),
                        hipMemcpyHostToDevice));

    // Launch kernel
    hipLaunchKernelGGL(bit_intrinsic_kernel<BitIntrinsic::FFS>,
                       dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, mem.deviceA, mem.deviceB,
                       mem.deviceC, mem.deviceD, WIDTH, HEIGHT);
    HIP_CHECK(hipGetLastError());

    // Copy results back
    HIP_CHECK(hipMemcpy(mem.hostA, mem.deviceA, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(mem.hostC, mem.deviceC, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    for (int i = 0; i < NUM; i++) {
      if (mem.hostA[i] != lastbit(mem.hostB[i])) {
        INFO("Mismatch: " << mem.hostA[i] << " - " << lastbit(mem.hostB[i]));
        errors++;
      }
    }
    for (int i = 0; i < NUM; i++) {
      if (mem.hostC[i] != lastbit(mem.hostD[i])) {
        INFO("Mismatch: " << mem.hostC[i] << " - " << lastbit(mem.hostD[i]));
        errors++;
      }
    }

    REQUIRE(errors == 0);
  }

  SECTION("popc - population count") {
    constexpr int WIDTH = 16;
    constexpr int HEIGHT = 16;
    constexpr int NUM = WIDTH * HEIGHT;

    GridTestMemory mem(NUM);

    // Initialize input data
    for (int i = 0; i < NUM; i++) {
      mem.hostB[i] = i;
      mem.hostD[i] = 1099511627776 - i;
    }

    // Copy input to device
    HIP_CHECK(hipMemcpy(mem.deviceB, mem.hostB, NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(mem.deviceD, mem.hostD, NUM * sizeof(unsigned long long int),
                        hipMemcpyHostToDevice));

    // Launch kernel
    hipLaunchKernelGGL(bit_intrinsic_kernel<BitIntrinsic::POPC>,
                       dim3(WIDTH / THREADS_PER_BLOCK_X, HEIGHT / THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, mem.deviceA, mem.deviceB,
                       mem.deviceC, mem.deviceD, WIDTH, HEIGHT);
    HIP_CHECK(hipGetLastError());

    // Copy results back
    HIP_CHECK(hipMemcpy(mem.hostA, mem.deviceA, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(mem.hostC, mem.deviceC, NUM * sizeof(unsigned int), hipMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    for (int i = 0; i < NUM; i++) {
      if (mem.hostA[i] != popcountCPU(mem.hostB[i])) {
        errors++;
      }
    }
    for (int i = 0; i < NUM; i++) {
      if (mem.hostC[i] != popcountCPU(mem.hostD[i])) {
        errors++;
      }
    }

    REQUIRE(errors == 0);
  }

  SECTION("ballot") {
    unsigned warpSize, pshift;
    warpSize = devProp.warpSize;

    int w = warpSize;
    pshift = 0;
    while (w >>= 1) ++pshift;

    unsigned int Num_Threads_per_Block = 512;
    unsigned int Num_Blocks_per_Grid = 1;
    unsigned int Num_Warps_per_Block = Num_Threads_per_Block / warpSize;
    unsigned int Num_Warps_per_Grid = (Num_Threads_per_Block * Num_Blocks_per_Grid) / warpSize;
    unsigned int* host_ballot = (unsigned int*)malloc(Num_Warps_per_Grid * sizeof(unsigned int));
    unsigned int* device_ballot;
    HIP_CHECK(hipMalloc((void**)&device_ballot, Num_Warps_per_Grid * sizeof(unsigned int)));
    int divergent_count = 0;
    for (unsigned i = 0; i < Num_Warps_per_Grid; i++) host_ballot[i] = 0;

    HIP_CHECK(hipMemcpy(device_ballot, host_ballot, Num_Warps_per_Grid * sizeof(unsigned int),
                        hipMemcpyHostToDevice));

    hipLaunchKernelGGL(gpu_ballot, dim3(Num_Blocks_per_Grid), dim3(Num_Threads_per_Block), 0, 0,
                       device_ballot, Num_Warps_per_Block, pshift);

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(host_ballot, device_ballot, Num_Warps_per_Grid * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));

    for (unsigned i = 0; i < Num_Warps_per_Grid; i++) {
      if ((host_ballot[i] == 0) || (host_ballot[i] / warpSize == warpSize)) {
        INFO("Warp: " << i << " is convergent - predicate true for " << (host_ballot[i] / warpSize)
                      << " threads");
      } else {
        INFO("Warp: " << i << " is divergent - predicate true for " << (host_ballot[i] / warpSize)
                      << " threads");
        divergent_count++;
      }
    }

    HIP_CHECK(hipFree(device_ballot));
    free(host_ballot);
    REQUIRE(divergent_count == 1);
  }

  SECTION("funnelshift") {
    // Allocate host and device memory
    unsigned int* host_l_output = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
    unsigned int* host_lc_output = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
    unsigned int* host_r_output = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
    unsigned int* host_rc_output = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));

    unsigned int* golden_l = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
    unsigned int* golden_lc = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
    unsigned int* golden_r = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));
    unsigned int* golden_rc = (unsigned int*)calloc(NUM_TESTS, sizeof(unsigned int));

    unsigned int* device_l_output;
    unsigned int* device_lc_output;
    unsigned int* device_r_output;
    unsigned int* device_rc_output;

    // Compute golden values
    for (int i = 0; i < NUM_TESTS; i++) {
      golden_l[i] = cpu_funnelshift_l(LO_INT, HI_INT, i);
      golden_lc[i] = cpu_funnelshift_lc(LO_INT, HI_INT, i);
      golden_r[i] = cpu_funnelshift_r(LO_INT, HI_INT, i);
      golden_rc[i] = cpu_funnelshift_rc(LO_INT, HI_INT, i);
    }

    HIP_CHECK(hipMalloc((void**)&device_l_output, NUM_TESTS * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void**)&device_lc_output, NUM_TESTS * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void**)&device_r_output, NUM_TESTS * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void**)&device_rc_output, NUM_TESTS * sizeof(unsigned int)));

    hipLaunchKernelGGL(funnelshift_kernel, dim3(1), dim3(1), 0, 0, device_l_output, device_lc_output,
                       device_r_output, device_rc_output);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(host_l_output, device_l_output, NUM_TESTS * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_lc_output, device_lc_output, NUM_TESTS * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_r_output, device_r_output, NUM_TESTS * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_rc_output, device_rc_output, NUM_TESTS * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));

    // Verify results
    printf("HI val: 0x%x\n", HI_INT);
    printf("LO val: 0x%x\n", LO_INT);

    auto verify = [](unsigned int* host, unsigned int* golden, const char* name) {
      for (int i = 0; i < NUM_TESTS; i++) {
        if (host[i] != golden[i]) {
          INFO("Mismatch " << name << ": " << host[i] << " - " << golden[i]);
          return false;
        }
      }
      return true;
    };

    REQUIRE(verify(host_l_output, golden_l, "l"));
    REQUIRE(verify(host_lc_output, golden_lc, "lc"));
    REQUIRE(verify(host_r_output, golden_r, "r"));
    REQUIRE(verify(host_rc_output, golden_rc, "rc"));

    // Cleanup
    HIP_CHECK(hipFree(device_l_output));
    HIP_CHECK(hipFree(device_lc_output));
    HIP_CHECK(hipFree(device_r_output));
    HIP_CHECK(hipFree(device_rc_output));

    free(host_l_output);
    free(host_lc_output);
    free(host_r_output);
    free(host_rc_output);
    free(golden_l);
    free(golden_lc);
    free(golden_r);
    free(golden_rc);
  }

#if HT_AMD
  SECTION("mbcnt - mask bit count") {
    unsigned int* device_mbcnt_lo;
    unsigned int* device_mbcnt_hi;
    unsigned int* device_lane_id;

    constexpr unsigned int num_waves_per_block = 2;
    const unsigned int wave_size = devProp.warpSize;
    const unsigned int num_threads_per_block = wave_size * num_waves_per_block;
    const unsigned int num_blocks = 2;
    const unsigned int num_threads = num_threads_per_block * num_blocks;
    const size_t buffer_size = num_threads * sizeof(unsigned int);

    HIP_CHECK(hipMalloc((void**)&device_mbcnt_lo, buffer_size));
    HIP_CHECK(hipMalloc((void**)&device_mbcnt_hi, buffer_size));
    HIP_CHECK(hipMalloc((void**)&device_lane_id, buffer_size));

    hipLaunchKernelGGL(mbcnt_kernel, dim3(num_blocks), dim3(num_threads_per_block), 0, 0,
                       device_mbcnt_lo, device_mbcnt_hi, device_lane_id);
    HIP_CHECK(hipGetLastError());
    unsigned int* host_mbcnt_lo = (unsigned int*)malloc(buffer_size);
    unsigned int* host_mbcnt_hi = (unsigned int*)malloc(buffer_size);
    unsigned int* host_lane_id = (unsigned int*)malloc(buffer_size);

    HIP_CHECK(hipMemcpy(host_mbcnt_lo, device_mbcnt_lo, buffer_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_mbcnt_hi, device_mbcnt_hi, buffer_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(host_lane_id, device_lane_id, buffer_size, hipMemcpyDeviceToHost));

    int mbcnt_lo_errors = 0;
    int mbcnt_hi_errors = 0;
    int lane_id_errors = 0;
    for (unsigned int i = 0; i < num_threads; i++) {
      unsigned int this_lane_id = i % wave_size;
      unsigned int this_mbcnt_lo = this_lane_id >= 32 ? 32 : this_lane_id;
      unsigned int this_mbcnt_hi = this_lane_id < 32 ? 0 : (this_lane_id - 32);

      if (host_mbcnt_lo[i] != this_mbcnt_lo) mbcnt_lo_errors++;

      if (host_mbcnt_hi[i] != this_mbcnt_hi) mbcnt_hi_errors++;

      if (host_lane_id[i] != this_lane_id) lane_id_errors++;
    }

    HIP_CHECK(hipFree(device_mbcnt_lo));
    HIP_CHECK(hipFree(device_mbcnt_hi));
    HIP_CHECK(hipFree(device_lane_id));

    free(host_mbcnt_lo);
    free(host_mbcnt_hi);
    free(host_lane_id);

    REQUIRE(mbcnt_lo_errors == 0);
    REQUIRE(mbcnt_hi_errors == 0);
    REQUIRE(lane_id_errors == 0);
  }
#endif
}
