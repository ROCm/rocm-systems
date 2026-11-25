/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// [D1] Typed ReduceCopy thunks of librccl_device.bc: a kernel calls the local
// ncclLocalReduceSumCopy_{I8,F32,BF16} and the host checks the N->M sums. Each
// thunk gets a weak no-op definition here, so the link still succeeds when the
// bitcode lacks it; the destinations then stay zero and the check fails.

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
// The strides are also the displacements, which pick the pack width: keep them
// 16-byte aligned for every type and give the count the odd value, so the
// copy runs its packed loops and then a scalar tail. The two strides differ so
// a thunk that swaps srcDispl and dstDispl gets the sums wrong.
constexpr size_t kSrcStride = 1024, kDstStride = 2048, kCount = 1001;

#define LOCAL_REDUCE_SUM_COPY(suffix, T) \
  extern "C" __device__ __attribute__((weak)) void ncclLocalReduceSumCopy_##suffix( \
      ncclCoopAny, int, T*, size_t, int, T*, size_t, size_t) {} \
  static __device__ void localReduceSumCopy(ncclCoopAny coop, T* src, T* dst) { \
    ncclLocalReduceSumCopy_##suffix(coop, kNSrc, src, kSrcStride, kNDst, dst, kDstStride, kCount); \
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
  std::vector<unsigned char> src(kNSrc * kSrcStride * sizeof(T)), dst(kNDst * kDstStride * sizeof(T));
  // Elements past kCount are 1 in every source, so a copy that overruns
  // kCount writes 3 into destination padding that must stay 0.
  for (int s = 0; s < kNSrc; ++s) {
    for (size_t i = 0; i < kSrcStride; ++i) {
      put<T>(&src[(s * kSrcStride + i) * sizeof(T)], i < kCount ? (int)((i + s) % 7) : 1);
    }
  }
  // Use a lambda so hipFree always runs regardless of which ASSERT fires.
  void *dSrc = nullptr, *dDst = nullptr;
  auto cleanup = [&] { (void)hipFree(dSrc); (void)hipFree(dDst); };
  ASSERT_EQ(hipMalloc(&dSrc, src.size()), hipSuccess) << (cleanup(), "");
  ASSERT_EQ(hipMalloc(&dDst, dst.size()), hipSuccess) << (cleanup(), "");
  ASSERT_EQ(hipMemcpy(dSrc, src.data(), src.size(), hipMemcpyHostToDevice), hipSuccess) << (cleanup(), "");
  ASSERT_EQ(hipMemset(dDst, 0, dst.size()), hipSuccess) << (cleanup(), "");
  k_local_reduce_sum_copy<T><<<1, 256>>>((T*)dSrc, (T*)dDst);
  ASSERT_EQ(hipGetLastError(), hipSuccess) << (cleanup(), "");
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess) << (cleanup(), "");
  ASSERT_EQ(hipMemcpy(dst.data(), dDst, dst.size(), hipMemcpyDeviceToHost), hipSuccess) << (cleanup(), "");
  cleanup();

  int wrong = 0, overrun = 0;  // every expected sum is >= 3, so a no-op leaves all of them wrong
  for (int d = 0; d < kNDst; ++d) {
    for (size_t i = 0; i < kDstStride; ++i) {
      int got = get<T>(&dst[(d * kDstStride + i) * sizeof(T)]);
      if (i >= kCount) {
        overrun += got != 0;
        continue;
      }
      int want = 0;
      for (int s = 0; s < kNSrc; ++s) want += (int)((i + s) % 7);
      wrong += got != want;
    }
  }
  EXPECT_EQ(wrong, 0) << "wrong sums out of " << kNDst * kCount
                      << " (all wrong: the weak fallback ran, librccl_device.bc lacks the thunk)";
  EXPECT_EQ(overrun, 0) << "destination elements past count " << kCount << " were written";
}

/* Stack probe: the indirect call gives this kernel a dynamic stack, so the
 * runtime sizes its scratch from hipLimitStackSize exactly as it does for the
 * thunk kernels, whose callees are only linked in after they are compiled. */
__device__ int stackProbeLeaf(int x) { return x + 1; }
__device__ int (*volatile stackProbeFn)(int) = stackProbeLeaf;
__global__ void k_stack_probe(int* out) { out[threadIdx.x] = stackProbeFn((int)threadIdx.x); }

/* The thunks need more than the 1 KiB default stack: on gfx950 D1 passes at
 * every size from 25 to 120 KiB and faults or hangs at most sizes below that.
 * The runtime maps scratch for whatever is asked, so ask for 32 KiB, a little
 * above the need. Too large a request can fail at dispatch (120 KiB on
 * gfx1151), and a failed dispatch can leave the device unusable for the rest
 * of the process, so there is one probe and no retry: its result goes into
 * every D1 failure message instead. */
constexpr size_t kStackBytes = 32 * 1024;

class IRReduceCopyTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    size_t def = 0, limit = 0;
    (void)hipDeviceGetLimit(&def, hipLimitStackSize);
    hipError_t err = hipSuccess;
    if (def < kStackBytes) err = hipDeviceSetLimit(hipLimitStackSize, kStackBytes);
    int* dOut = nullptr;
    if (err == hipSuccess) err = hipMalloc(&dOut, 256 * sizeof(int));
    if (err == hipSuccess) {
      k_stack_probe<<<1, 256>>>(dOut);
      err = hipGetLastError();
      if (err == hipSuccess) err = hipDeviceSynchronize();
      (void)hipFree(dOut);
    }
    (void)hipGetLastError();
    (void)hipDeviceGetLimit(&limit, hipLimitStackSize);
    stackReport_ = "stack limit " + std::to_string(limit / 1024) + " KiB, probe " + hipGetErrorName(err);
    printf("[IRReduceCopyTest] %s\n", stackReport_.c_str());
  }
  static std::string stackReport_;
};
std::string IRReduceCopyTest::stackReport_;

TEST_F(IRReduceCopyTest, D1_LocalReduceSumCopy_I8) {
  SCOPED_TRACE(stackReport_);
  checkLocalReduceSumCopy<int8_t>("I8");
}
TEST_F(IRReduceCopyTest, D1_LocalReduceSumCopy_F32) {
  SCOPED_TRACE(stackReport_);
  checkLocalReduceSumCopy<float>("F32");
}
TEST_F(IRReduceCopyTest, D1_LocalReduceSumCopy_BF16) {
  SCOPED_TRACE(stackReport_);
  checkLocalReduceSumCopy<__hip_bfloat16>("BF16");
}
