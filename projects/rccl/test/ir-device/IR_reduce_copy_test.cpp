/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// [D1] Typed ReduceCopy thunks of librccl_device.bc (NCCL 2.32.3 bindings/ir,
// "expose typed ReduceCopy APIs"): a consumer kernel calls the purely local
// ncclLocalReduceSumCopy_{I8,F32,BF16} and the host checks the N->M sums.
//
// Linked into IR_test.exe by tests/conftest.py. Each thunk also gets a weak
// no-op definition here, so the device link still succeeds against a bitcode
// library that does not export it (the library's definition wins when it
// does). Without the thunk the destinations stay zero and the check fails.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include <nccl.h>
#include <nccl_device_wrapper.h>

constexpr int kNSrc = 3, kNDst = 2;
constexpr size_t kCount = 1000;  // not a multiple of any pack/unroll width: covers the tails

#define LOCAL_REDUCE_SUM_COPY(suffix, T)                                                    \
  extern "C" __device__ __attribute__((weak)) void ncclLocalReduceSumCopy_##suffix(         \
      ncclCoopAny, int, T*, size_t, int, T*, size_t, size_t) {}                            \
  static __device__ void localReduceSumCopy(ncclCoopAny coop, T* src, T* dst) {             \
    ncclLocalReduceSumCopy_##suffix(coop, kNSrc, src, kCount, kNDst, dst, kCount, kCount); \
  }
LOCAL_REDUCE_SUM_COPY(I8, int8_t)
LOCAL_REDUCE_SUM_COPY(F32, float)
LOCAL_REDUCE_SUM_COPY(BF16, __hip_bfloat16)

template <typename T>
__global__ void k_local_reduce_sum_copy(T* src, T* dst) {
  ncclCoopAny coop;
  ncclCoopAnyInitCta(&coop);
  localReduceSumCopy(coop, src, dst);
}

/* Small integers are exact in every type here. bf16 is the high half of an
 * fp32 (bytes 2..3 on a little-endian host), so the host needs no HIP bf16 support. */
template <typename T>
static void put(unsigned char* p, int v) {
  if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    float f = (float)v;
    std::memcpy(p, (char*)&f + 2, 2);
  } else {
    T t = (T)v;
    std::memcpy(p, &t, sizeof t);
  }
}
template <typename T>
static int get(const unsigned char* p) {
  if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    float f = 0;
    std::memcpy((char*)&f + 2, p, 2);
    return (int)f;
  } else {
    T t;
    std::memcpy(&t, p, sizeof t);
    return (int)t;
  }
}

template <typename T>
static void checkLocalReduceSumCopy(const char* suffix) {
  SCOPED_TRACE(std::string("ncclLocalReduceSumCopy_") + suffix);
  std::vector<unsigned char> src(kNSrc * kCount * sizeof(T)), dst(kNDst * kCount * sizeof(T));
  for (int s = 0; s < kNSrc; ++s)
    for (size_t i = 0; i < kCount; ++i) put<T>(&src[(s * kCount + i) * sizeof(T)], (int)((i + s) % 7));
  void *dSrc = nullptr, *dDst = nullptr;
  ASSERT_EQ(hipMalloc(&dSrc, src.size()), hipSuccess);
  ASSERT_EQ(hipMalloc(&dDst, dst.size()), hipSuccess);
  ASSERT_EQ(hipMemcpy(dSrc, src.data(), src.size(), hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemset(dDst, 0, dst.size()), hipSuccess);
  k_local_reduce_sum_copy<T><<<1, 256>>>((T*)dSrc, (T*)dDst);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  ASSERT_EQ(hipMemcpy(dst.data(), dDst, dst.size(), hipMemcpyDeviceToHost), hipSuccess);
  ASSERT_EQ(hipFree(dSrc), hipSuccess);
  ASSERT_EQ(hipFree(dDst), hipSuccess);

  int wrong = 0;  // every expected sum is >= 3, so a no-op leaves all of them wrong
  for (int d = 0; d < kNDst; ++d)
    for (size_t i = 0; i < kCount; ++i) {
      int want = 0;
      for (int s = 0; s < kNSrc; ++s) want += (int)((i + s) % 7);
      wrong += get<T>(&dst[(d * kCount + i) * sizeof(T)]) != want;
    }
  EXPECT_EQ(wrong, 0) << "wrong sums out of " << kNDst * kCount
                      << " (all wrong: the weak fallback ran, librccl_device.bc lacks the thunk)";
}

TEST(IRReduceCopyTest, D1_LocalReduceSumCopyTyped) {
  /* The thunks are linked in after these kernels are compiled, so the kernels
   * run on the runtime's dynamic stack limit (1 KiB by default). At -O0 the I8
   * thunk alone takes a ~31 KiB frame, so ask for close to the gfx9 maximum.
   * Run it with scratch reclaim on (tests/test_ir_device.py sets
   * HSA_NO_SCRATCH_RECLAIM=0): without reclaim ROCr maps that stack for every
   * wave slot of the GPU, more than MI300X's scratch aperture holds. */
  if (hipDeviceSetLimit(hipLimitStackSize, 120 * 1024) != hipSuccess) {
    (void)hipGetLastError();
    ASSERT_EQ(hipDeviceSetLimit(hipLimitStackSize, 64 * 1024), hipSuccess);
  }
  checkLocalReduceSumCopy<int8_t>("I8");
  checkLocalReduceSumCopy<float>("F32");
  checkLocalReduceSumCopy<__hip_bfloat16>("BF16");
}
