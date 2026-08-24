/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for LL protocol FIFO 128-bit vector load/store helpers (prims_ll.h).
// Mirrors loadLLLine/storeLLLine so we can exercise them without pulling in the
// full Primitives<> template.

#include "DeviceTestBase.hpp"

#include "device.h"
#include "rccl_ptr.h"

namespace RcclUnitTesting
{

namespace
{

// Keep in sync with projects/rccl/src/device/prims_ll.h
inline __device__ void loadLLLine(const union ncclLLFifoLine* src, union ncclLLFifoLine& dst) {
  union {
    v4u v;
    uint64_t u64[2];
  } u;
  u.v = __builtin_nontemporal_load((v4u_gptr)src->v);
  dst.v[0] = u.u64[0];
  dst.v[1] = u.u64[1];
}

inline __device__ void storeLLLine(union ncclLLFifoLine* dst, const union ncclLLFifoLine& src) {
#if !RCCL_HAVE_GLOBAL_DWORDX4_BUILTINS && (defined(__gfx1200__) || defined(__gfx1201__))
  __hip_atomic_store((u64_gptr)dst->v, src.v[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store((u64_gptr)dst->v + 1, src.v[1], __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  union {
    v4u v;
    uint64_t u64[2];
  } u;
  u.u64[0] = src.v[0];
  u.u64[1] = src.v[1];
  *((v4u_gptr)dst->v) = u.v;
#endif
}

inline __host__ __device__ void makeFifoLine(union ncclLLFifoLine& line, uint64_t payload, uint32_t flag) {
  line.data1 = static_cast<uint32_t>(payload);
  line.data2 = static_cast<uint32_t>(payload >> 32);
  line.flag1 = flag;
  line.flag2 = flag;
}

} // namespace

class LLFifoVectorTest : public DeviceTestBase
{
};

__global__ void kernelFifoLayoutRoundtrip(const union ncclLLFifoLine* src, union ncclLLFifoLine* dst) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    storeLLLine(dst, *src);
    loadLLLine(dst, *dst);
  }
}

TEST_F(LLFifoVectorTest, FifoLineLayoutRoundtrip) {
  union ncclLLFifoLine h_in {};
  makeFifoLine(h_in, 0x0123456789ABCDEFULL, 0xA5A5A5A5u);

  DeviceBuffer<ncclLLFifoLine> d_in(1), d_out(1);
  d_in.upload(h_in);

  kernelFifoLayoutRoundtrip<<<1, 1>>>(d_in.ptr, d_out.ptr);
  syncAndCheck();

  union ncclLLFifoLine h_out = d_out.download();
  EXPECT_EQ(h_out.data1, h_in.data1);
  EXPECT_EQ(h_out.data2, h_in.data2);
  EXPECT_EQ(h_out.flag1, h_in.flag1);
  EXPECT_EQ(h_out.flag2, h_in.flag2);
  EXPECT_EQ(h_out.v[0], h_in.v[0]);
  EXPECT_EQ(h_out.v[1], h_in.v[1]);
}

__global__ void kernelFifoFlagPoll(union ncclLLFifoLine* fifo, uint32_t expectFlag, int* ready) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  union ncclLLFifoLine published {};
  makeFifoLine(published, 0xDEADBEEFCAFEBABEULL, expectFlag);
  storeLLLine(fifo, published);

  union ncclLLFifoLine observed {};
  for (int spins = 0; spins < 32; ++spins) {
    loadLLLine(fifo, observed);
    if (observed.flag1 == expectFlag && observed.flag2 == expectFlag) {
      *ready = 1;
      return;
    }
  }
  *ready = 0;
}

TEST_F(LLFifoVectorTest, FifoLineFlagPollSeesMatchingFlags) {
  DeviceBuffer<ncclLLFifoLine> d_fifo(1);
  DeviceBuffer<int> d_ready(1);
  d_ready.zero();

  const uint32_t flag = 0x12345678u;
  kernelFifoFlagPoll<<<1, 1>>>(d_fifo.ptr, flag, d_ready.ptr);
  syncAndCheck();

  EXPECT_EQ(d_ready.download(), 1);

  union ncclLLFifoLine h_line = d_fifo.download();
  EXPECT_EQ(h_line.flag1, flag);
  EXPECT_EQ(h_line.flag2, flag);
  EXPECT_EQ(h_line.data1, 0xCAFEBABEu);
  EXPECT_EQ(h_line.data2, 0xDEADBEEFu);
}

__global__ void kernelFifoZeroCleanup(union ncclLLFifoLine* fifo) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  union ncclLLFifoLine zero {};
  makeFifoLine(zero, 0, 0x2u);
  storeLLLine(fifo, zero);
}

TEST_F(LLFifoVectorTest, FifoLineZeroCleanup) {
  DeviceBuffer<ncclLLFifoLine> d_fifo(1);
  union ncclLLFifoLine dirty {};
  makeFifoLine(dirty, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFu);
  d_fifo.upload(dirty);

  kernelFifoZeroCleanup<<<1, 1>>>(d_fifo.ptr);
  syncAndCheck();

  union ncclLLFifoLine h_line = d_fifo.download();
  EXPECT_EQ(h_line.data1, 0u);
  EXPECT_EQ(h_line.data2, 0u);
  EXPECT_EQ(h_line.flag1, 2u);
  EXPECT_EQ(h_line.flag2, 2u);
}

} // namespace RcclUnitTesting
