/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
This testcase verifies the CopyBufferLocal and CopyBufferLocalNonTemporal kernels
Production kernels in rocclr/device/blitcl.cpp:
  - __amd_rocclr_copyBuffer     (temporal/normal stores)
  - __amd_rocclr_copyBufferNT   (non-temporal stores for streaming writes)
*/

#include <hip_test_common.hh>
#include <cstdlib>
#include <cstring>

// Runtime flag to select between temporal and non-temporal kernel implementations
// Default is false (normal copy). Set via environment variable USE_NONTEMPORAL=1
static bool useNonTemporalKernel() {
  const char* env = std::getenv("USE_NONTEMPORAL");
  return (env != nullptr && std::strcmp(env, "1") == 0);
}

// CopyBufferLocalNonTemporal kernel - uses non-temporal stores for streaming writes
// This kernel copies data from src buffer to dst buffer with NT hints
// Production kernel: __amd_rocclr_copyBufferNT in rocclr/device/blitcl.cpp
__global__ void CopyBufferLocalNonTemporal(unsigned char* src, unsigned char* dst, unsigned long size,
                                unsigned int remainder, unsigned int aligned_size,
                                unsigned long end_ptr, unsigned int next_chunk,
                                unsigned int workgroup_size) {
  unsigned int l = threadIdx.x;
  unsigned int g = blockIdx.x;
  unsigned long id = (g * workgroup_size + l);
  unsigned long id_remainder = id;

  if (aligned_size == sizeof(unsigned long)) {
    unsigned long* srcD = (unsigned long*)(src);
    unsigned long* dstD = (unsigned long*)(dst);
    while ((unsigned long)(&dstD[id]) < end_ptr) {
      __builtin_nontemporal_store(srcD[id], &dstD[id]);
      id += next_chunk;
    }
  } else {
    unsigned int* srcD = (unsigned int*)(src);
    unsigned int* dstD = (unsigned int*)(dst);
    while ((unsigned long)(&dstD[id]) < end_ptr) {
      __builtin_nontemporal_store(srcD[id], &dstD[id]);
      id += next_chunk;
    }
  }
  if ((remainder != 0) && (id_remainder == 0)) {
    for (unsigned long i = size - remainder; i < size; ++i) {
      dst[i] = src[i];
    }
  }
}

// CopyBufferLocal kernel - copied from __amd_rocclr_copyBuffer
// This kernel copies data from src buffer to dst buffer
// Production kernel: __amd_rocclr_copyBuffer in rocclr/device/blitcl.cpp
__global__ void CopyBufferLocal(unsigned char* src, unsigned char* dst, unsigned long size,
                                unsigned int remainder, unsigned int aligned_size,
                                unsigned long end_ptr, unsigned int next_chunk,
                                unsigned int workgroup_size) {
  unsigned int l = threadIdx.x;
  unsigned int g = blockIdx.x;
  unsigned long id = (g * workgroup_size + l);
  unsigned long id_remainder = id;

  if (aligned_size == sizeof(ulong2)) {
    ulong2* srcD = (ulong2*)(src);
    ulong2* dstD = (ulong2*)(dst);
    while ((unsigned long)(&dstD[id]) < end_ptr) {
      dstD[id] = srcD[id];
      id += next_chunk;
    }
  } else {
    unsigned int* srcD = (unsigned int*)(src);
    unsigned int* dstD = (unsigned int*)(dst);
    while ((unsigned long)(&dstD[id]) < end_ptr) {
      dstD[id] = srcD[id];
      id += next_chunk;
    }
  }
  if ((remainder != 0) && (id_remainder == 0)) {
    for (unsigned long i = size - remainder; i < size; ++i) {
      dst[i] = src[i];
    }
  }
}

/*
This testcase verifies CopyBufferLocal kernel
1. Allocates device memory for src and dst buffers
2. Fills src buffer with known pattern
3. Launches kernel to copy src to dst
4. Copies result back to host and validates
*/
TEST_CASE("Unit_CopyBufferLocal_Basic") {
  constexpr size_t kBufferSize = 1024 * 1024;  // 1 MB buffer
  constexpr unsigned int kWorkgroupSize = 256;
  constexpr unsigned int kNumWorkgroups = 64;
  constexpr unsigned int aligned_size = sizeof(unsigned int);
  constexpr unsigned int total_threads = kWorkgroupSize * kNumWorkgroups;
  constexpr unsigned int next_chunk = total_threads;

  // Calculate size in terms of aligned elements
  constexpr size_t num_aligned_elements = kBufferSize / aligned_size;
  constexpr unsigned int remainder = kBufferSize % aligned_size;

  // Allocate host memory
  unsigned char* h_src = new unsigned char[kBufferSize];
  unsigned char* h_dst = new unsigned char[kBufferSize];
  REQUIRE(h_src != nullptr);
  REQUIRE(h_dst != nullptr);

  // Initialize source with known pattern
  for (size_t i = 0; i < kBufferSize; ++i) {
    h_src[i] = static_cast<unsigned char>(i & 0xFF);
  }
  memset(h_dst, 0, kBufferSize);

  // Allocate device memory
  unsigned char* d_src;
  unsigned char* d_dst;
  HIP_CHECK(hipMalloc(&d_src, kBufferSize));
  HIP_CHECK(hipMalloc(&d_dst, kBufferSize));

  // Copy source to device and initialize dst to zero
  HIP_CHECK(hipMemcpy(d_src, h_src, kBufferSize, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_dst, 0, kBufferSize));

  // Calculate end pointer (for aligned portion)
  unsigned long end_ptr = (unsigned long)d_dst + (num_aligned_elements * aligned_size);

  // Launch the kernel - select between temporal and non-temporal based on USE_NONTEMPORAL env var
  if (useNonTemporalKernel()) {
    INFO("Using CopyBufferLocalNonTemporal kernel (non-temporal stores)");
    CopyBufferLocalNonTemporal<<<dim3(kNumWorkgroups), dim3(kWorkgroupSize), 0, 0>>>(
        d_src, d_dst, kBufferSize, remainder, aligned_size, end_ptr, next_chunk, kWorkgroupSize);
  } else {
    INFO("Using CopyBufferLocal kernel (normal stores)");
    CopyBufferLocal<<<dim3(kNumWorkgroups), dim3(kWorkgroupSize), 0, 0>>>(
        d_src, d_dst, kBufferSize, remainder, aligned_size, end_ptr, next_chunk, kWorkgroupSize);
  }

  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Copy results back to host
  HIP_CHECK(hipMemcpy(h_dst, d_dst, kBufferSize, hipMemcpyDeviceToHost));

  // Verify results
  size_t errors = 0;
  for (size_t i = 0; i < kBufferSize; ++i) {
    if (h_dst[i] != h_src[i]) {
      errors++;
    }
  }

  REQUIRE(errors == 0);

  // Cleanup
  HIP_CHECK(hipFree(d_src));
  HIP_CHECK(hipFree(d_dst));
  delete[] h_src;
  delete[] h_dst;
}
