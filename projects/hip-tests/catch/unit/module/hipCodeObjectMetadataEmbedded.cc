/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Fat binaries embedded in the test executable: the runtime gets them through
// __hipRegisterFatBinary, with no size known. CMake builds this file for many processors besides
// the device's own and the generic targets, so the runtime has to pick the device's entry out of
// a multi-ASIC fat binary. It is built twice: as it is, and with --offload-compress through a
// wrapper generated in the build directory that defines CO_MD_EMBEDDED_COMPRESSED (compression
// is set per translation unit). The kernels are in an anonymous namespace, so both translation
// units link into one executable: two registered fat binaries in one process.

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

#include <memory>
#include <string>

namespace {

//! 8 bytes, 4-byte aligned
struct EmbeddedPair {
  int a;
  char b;
};

//! 24 bytes, 8-byte aligned
struct EmbeddedBig {
  double x[3];
};

// Thread 0 writes back what it received
__global__ void embedded_args_layout(char c, short s, int i, long long l, float f, double d,
                                     long long* out, EmbeddedPair p, EmbeddedBig b) {
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
__global__ void __launch_bounds__(128) embedded_shared_static(int* out) {
  __shared__ int buf[256];
  buf[threadIdx.x] = threadIdx.x;
  buf[threadIdx.x + 128] = 1;
  __syncthreads();
  out[threadIdx.x] = buf[127 - threadIdx.x] + buf[threadIdx.x + 128];
}

// A dynamically indexed private array: at least 256 bytes of scratch per thread
__global__ void embedded_scratch(int* out, int n) {
  volatile int a[64];
  for (int k = 0; k < 64; ++k) a[k] = k * n;
  out[threadIdx.x] = a[(threadIdx.x + n) & 63];
}

// The target the loaded code object was built for, e.g. "gfx1100" or "gfx11-generic"
__global__ void embedded_target_id(char* out) {
#if defined(__HIP_DEVICE_COMPILE__)
  const char id[] = __amdgcn_target_id__;
  for (unsigned k = 0; k < sizeof(id); ++k) out[k] = id[k];
#endif
}

//! hipMalloc'd memory, freed on every exit: HIP_CHECK and REQUIRE throw
template <typename T> std::unique_ptr<T, hipError_t (*)(void*)> EmbeddedAlloc(size_t bytes) {
  T* ptr = nullptr;
  HIP_CHECK(hipMalloc(&ptr, bytes));
  return std::unique_ptr<T, hipError_t (*)(void*)>(ptr, hipFree);
}

//! The device's processor without features, e.g. "gfx1100"
std::string EmbeddedDeviceProcessor() {
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  const std::string arch = prop.gcnArchName;
  return arch.substr(0, arch.find(':'));
}

}  // namespace

/**
 * @addtogroup ModuleTest
 * @{
 */

// One body for both builds; the yaml tags need the literal name in HIP_TEST_CASE
#if defined(CO_MD_EMBEDDED_COMPRESSED)
/**
 * Test Description
 * ------------------------
 * - The compressed (--offload-compress) multi-ASIC fat binary embedded in the executable: the
 *   runtime decompresses it without a known size, loads the device's native entry, and the
 *   kernels' attributes, arguments and LDS match the source.
 */
HIP_TEST_CASE(Unit_CodeObjectMetadata_EmbeddedCompressedFatBinary) {
#else
/**
 * Test Description
 * ------------------------
 * - The uncompressed multi-ASIC fat binary embedded in the executable (other processors, the
 *   generic targets and the device's own): the runtime loads the device's native entry, and the
 *   kernels' attributes, arguments and LDS match the source.
 */
HIP_TEST_CASE(Unit_CodeObjectMetadata_EmbeddedFatBinary) {
#endif
  SECTION("the device's native entry is loaded, not another processor's or a generic one") {
    constexpr size_t kSize = 64;
    const auto buffer = EmbeddedAlloc<char>(kSize);
    HIP_CHECK(hipMemset(buffer.get(), 0, kSize));
    embedded_target_id<<<1, 1>>>(buffer.get());
    HIP_CHECK(hipGetLastError());
    char host[kSize] = {};
    HIP_CHECK(hipMemcpy(host, buffer.get(), kSize, hipMemcpyDeviceToHost));
    host[kSize - 1] = '\0';
    const std::string target = host;
    INFO("loaded " << target);
    CHECK(target.substr(0, target.find(':')) == EmbeddedDeviceProcessor());
  }
  SECTION("attributes from the metadata") {
    hipFuncAttributes attr{};
    HIP_CHECK(hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(embedded_shared_static)));
    CHECK(attr.sharedSizeBytes == 1024);
    CHECK(attr.maxThreadsPerBlock == 128);
    HIP_CHECK(hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(embedded_scratch)));
    CHECK(attr.localSizeBytes >= 256);
    CHECK(attr.sharedSizeBytes == 0);
  }
  SECTION("arguments as the kernel receives them") {
    const auto buffer = EmbeddedAlloc<long long>(9 * sizeof(long long));
    HIP_CHECK(hipMemset(buffer.get(), 0, 9 * sizeof(long long)));
    embedded_args_layout<<<1, 64>>>(-7, 1234, -123456, 0x123456789abcll, 1.5f, -2.25, buffer.get(),
                                    EmbeddedPair{42, 'x'}, EmbeddedBig{{1.0, 2.0, 3.0}});
    HIP_CHECK(hipGetLastError());
    long long host[9] = {};
    HIP_CHECK(hipMemcpy(host, buffer.get(), sizeof(host), hipMemcpyDeviceToHost));
    const long long want[9] = {-7, 1234, -123456, 0x123456789abcll, 1500, -2250, 42, 'x', 321};
    for (int k = 0; k < 9; ++k) {
      INFO("argument " << k);
      CHECK(host[k] == want[k]);
    }
  }
  SECTION("static LDS as the kernel uses it") {
    const auto buffer = EmbeddedAlloc<int>(128 * sizeof(int));
    embedded_shared_static<<<1, 128>>>(buffer.get());
    HIP_CHECK(hipGetLastError());
    int host[128] = {};
    HIP_CHECK(hipMemcpy(host, buffer.get(), sizeof(host), hipMemcpyDeviceToHost));
    for (int t = 0; t < 128; ++t) {
      INFO("thread " << t);
      CHECK(host[t] == 127 - t + 1);
    }
  }
}

/**
 * End doxygen group ModuleTest.
 * @}
 */
