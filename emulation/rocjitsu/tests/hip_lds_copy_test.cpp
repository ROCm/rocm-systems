// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_lds_copy_test.cpp
/// @brief Small HIP kernels that force rocjitsu's CDNA4->CDNA3 virtual-LDS path.

#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

namespace {

constexpr int kBlockSize = 256;
constexpr int kTrB16BlockSize = 128;
constexpr int kBlocks = 64;
constexpr int kSharedWords = 17024;
constexpr int kHighBase = kSharedWords - kBlockSize;
constexpr int kHighVectorBase = kSharedWords - kBlockSize * 4;
constexpr int kLargeDsByteOffset = 0x1400;
static_assert(kSharedWords * static_cast<int>(sizeof(uint32_t)) > 64 * 1024);

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

using U32x4 = uint32_t __attribute__((ext_vector_type(4)));
using U32x2 = uint32_t __attribute__((ext_vector_type(2)));

constexpr uint32_t pack_u16_pair(uint32_t lo, uint32_t hi) {
  return (lo & 0xffffu) | ((hi & 0xffffu) << 16);
}

constexpr uint32_t tr_b16_halfword_value(int thread, int halfword) {
  return (0x1200u + static_cast<uint32_t>(thread) * 0x11u + static_cast<uint32_t>(halfword)) &
         0xffffu;
}

__global__ void lds_copy_low_kernel(const uint32_t *__restrict__ in, uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const uint32_t addr =
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)) + tid * sizeof(uint32_t);
  uint32_t value = in[tid];

  // HIP C++ shared-memory accesses may compile to flat LDS instructions through
  // src_shared_base. The current virtual-LDS implementation rewrites DS
  // opcodes, so these tiny tests use inline DS instructions deliberately.
  asm volatile("ds_write_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(addr)
               : "memory");
  out[tid] = value;
}

__global__ void lds_reverse_high_kernel(const uint32_t *__restrict__ in,
                                        uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
  const uint32_t store_addr = base + (kHighBase + tid) * sizeof(uint32_t);
  const uint32_t load_addr = base + (kHighBase + (kBlockSize - 1 - tid)) * sizeof(uint32_t);
  uint32_t value = in[tid];

  // Touch the end of a >64 KiB static LDS allocation. This is the smallest
  // check that the translated DS offset is based on the virtual backing buffer,
  // not the host hardware LDS aperture.
  asm volatile("ds_write_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(store_addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(load_addr)
               : "memory");
  out[tid] = value;
}

__global__ void lds_copy_high_kernel(const uint32_t *__restrict__ in, uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)) +
                        (kHighBase + tid) * sizeof(uint32_t);
  uint32_t value = in[tid];

  // Keep the first high-offset virtual-LDS check intentionally boring: each
  // lane reads back the word it wrote. Reverse and cross-wave tests below then
  // isolate producer/consumer ordering once this baseline is known-good.
  asm volatile("ds_write_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(addr)
               : "memory");
  out[tid] = value;
}

__global__ void lds_multi_block_high_kernel(const uint32_t *__restrict__ in,
                                            uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const int index = blockIdx.x * blockDim.x + tid;
  const uint32_t addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds)) +
                        (kHighBase + tid) * sizeof(uint32_t);
  uint32_t value = in[index];

  // Every workgroup writes the same high LDS offsets. Virtual LDS must offset
  // the backing pointer by workgroup id so concurrent workgroups do not alias.
  asm volatile("ds_write_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(addr)
               : "memory");
  out[index] = value;
}

__global__ void lds_cross_wave_exchange_high_kernel(const uint32_t *__restrict__ in,
                                                    uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
  const uint32_t store_addr = base + (kHighBase + tid) * sizeof(uint32_t);
  const uint32_t load_addr = base + (kHighBase + (tid ^ 64)) * sizeof(uint32_t);
  uint32_t value = in[tid];

  // This is the smallest LDS producer/consumer check that forces traffic across
  // wavefronts. Matmul-style tiles rely on one wave observing data another wave
  // wrote before the block barrier, so a virtual-LDS global-memory replacement
  // must preserve both the data and the synchronization semantics here.
  asm volatile("ds_write_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(store_addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(load_addr)
               : "memory");
  out[tid] = value;
}

__global__ void lds_b128_reverse_high_kernel(const uint32_t *__restrict__ in,
                                             uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const int source = tid * 4;
  const int target = (kBlockSize - 1 - tid) * 4;
  const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
  const uint32_t store_addr = base + (kHighVectorBase + source) * sizeof(uint32_t);
  const uint32_t load_addr = base + (kHighVectorBase + target) * sizeof(uint32_t);
  U32x4 value = {in[source + 0], in[source + 1], in[source + 2], in[source + 3]};

  // The fp16 Tensile repro uses b128 LDS traffic. Keep this fixture as small as
  // possible while covering the same vector-width lowering path.
  asm volatile("ds_write_b128 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(store_addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b128 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(load_addr)
               : "memory");

  out[source + 0] = value[0];
  out[source + 1] = value[1];
  out[source + 2] = value[2];
  out[source + 3] = value[3];
}

__global__ void lds_b128_reverse_immediate_offset_kernel(const uint32_t *__restrict__ in,
                                                         uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const int source = tid * 4;
  const int target = (kBlockSize - 1 - tid) * 4;
  const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
  const uint32_t store_addr = base + source * sizeof(uint32_t);
  const uint32_t load_addr = base + target * sizeof(uint32_t);
  U32x4 value = {in[source + 0], in[source + 1], in[source + 2], in[source + 3]};

  static_assert(kLargeDsByteOffset + kBlockSize * 4 * static_cast<int>(sizeof(uint32_t)) <=
                kSharedWords * static_cast<int>(sizeof(uint32_t)));

  // Tensile kernels commonly keep a small vector LDS address in a VGPR and put
  // the tile displacement in the DS instruction's immediate offset. When that
  // offset is larger than CDNA3 global-memory's immediate range, virtual-LDS
  // lowering must materialize an address in a scratch VGPR and restore every
  // clobbered register afterwards.
  asm volatile("ds_write_b128 %0, %1 offset:5120\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(store_addr), "v"(value)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b128 %0, %1 offset:5120\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(load_addr)
               : "memory");

  out[source + 0] = value[0];
  out[source + 1] = value[1];
  out[source + 2] = value[2];
  out[source + 3] = value[3];
}

__global__ void lds_read_b64_tr_b16_kernel(uint32_t *__restrict__ out) {
  __shared__ volatile uint32_t lds[kSharedWords];
  const int tid = threadIdx.x;
  const uint32_t base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(lds));
  const uint32_t addr = base + tid * sizeof(U32x2);
  U32x2 raw = {pack_u16_pair(tr_b16_halfword_value(tid, 0), tr_b16_halfword_value(tid, 1)),
               pack_u16_pair(tr_b16_halfword_value(tid, 2), tr_b16_halfword_value(tid, 3))};
  U32x2 value = {};

  // `ds_read_b64_tr_b16` is a distinct Tensile-shaped LDS read: it reads a
  // per-lane b64 footprint and returns a 4x16-lane halfword transpose through
  // the DS crossbar. Keep this fixture all-lane and full-wave, matching the ISA
  // contract assumed by the lowering.
  asm volatile("ds_write_b64 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               :
               : "v"(addr), "v"(raw)
               : "memory");
  __syncthreads();
  asm volatile("ds_read_b64_tr_b16 %0, %1\n"
               "s_waitcnt lgkmcnt(0)\n"
               : "=v"(value)
               : "v"(addr)
               : "memory");

  out[tid * 2 + 0] = value[0];
  out[tid * 2 + 1] = value[1];
}

std::vector<uint32_t> make_input(size_t count, uint32_t salt) {
  std::vector<uint32_t> values(count);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = 0x9e370000u ^ salt ^ static_cast<uint32_t>(i * 2654435761u);
  return values;
}

void run_unary_kernel(void (*kernel)(const uint32_t *, uint32_t *),
                      const std::vector<uint32_t> &expected_input,
                      const std::vector<uint32_t> &expected_output) {
  ASSERT_EQ(expected_input.size(), expected_output.size());
  const size_t bytes = expected_input.size() * sizeof(uint32_t);

  uint32_t *in = nullptr;
  uint32_t *out = nullptr;
  HIP_ASSERT(hipMalloc(&in, bytes));
  HIP_ASSERT(hipMalloc(&out, bytes));
  HIP_ASSERT(hipMemcpy(in, expected_input.data(), bytes, hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(out, 0x5a, bytes));

  kernel<<<expected_input.size() / kBlockSize, kBlockSize>>>(in, out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> actual(expected_output.size());
  HIP_ASSERT(hipMemcpy(actual.data(), out, bytes, hipMemcpyDeviceToHost));
  (void)hipFree(in);
  (void)hipFree(out);

  uint32_t mismatches = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] == expected_output[i])
      continue;
    if (mismatches < 8) {
      ADD_FAILURE() << "mismatch at i=" << i << ": got=0x" << std::hex << actual[i]
                    << " expected=0x" << expected_output[i] << std::dec;
    }
    ++mismatches;
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " mismatches";
}

void run_output_kernel(void (*kernel)(uint32_t *), const std::vector<uint32_t> &expected_output,
                       int threads) {
  const size_t bytes = expected_output.size() * sizeof(uint32_t);

  uint32_t *out = nullptr;
  HIP_ASSERT(hipMalloc(&out, bytes));
  HIP_ASSERT(hipMemset(out, 0x5a, bytes));

  kernel<<<1, threads>>>(out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> actual(expected_output.size());
  HIP_ASSERT(hipMemcpy(actual.data(), out, bytes, hipMemcpyDeviceToHost));
  (void)hipFree(out);

  uint32_t mismatches = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] == expected_output[i])
      continue;
    if (mismatches < 8) {
      ADD_FAILURE() << "mismatch at i=" << i << ": got=0x" << std::hex << actual[i]
                    << " expected=0x" << expected_output[i] << std::dec;
    }
    ++mismatches;
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " mismatches";
}

} // namespace

TEST(HipLdsCopyDbtTest, LargeStaticLdsCopyLowOffset) {
  const std::vector<uint32_t> input = make_input(kBlockSize, 0x101u);
  run_unary_kernel(lds_copy_low_kernel, input, input);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsReverseHighOffset) {
  const std::vector<uint32_t> input = make_input(kBlockSize, 0x202u);
  std::vector<uint32_t> expected(input.rbegin(), input.rend());
  run_unary_kernel(lds_reverse_high_kernel, input, expected);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsCopyHighOffset) {
  const std::vector<uint32_t> input = make_input(kBlockSize, 0x707u);
  run_unary_kernel(lds_copy_high_kernel, input, input);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsMultiBlockHighOffset) {
  const std::vector<uint32_t> input = make_input(kBlocks * kBlockSize, 0x303u);
  run_unary_kernel(lds_multi_block_high_kernel, input, input);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsCrossWaveExchangeHighOffset) {
  const std::vector<uint32_t> input = make_input(kBlockSize, 0x606u);
  std::vector<uint32_t> expected(input.size());
  for (int tid = 0; tid < kBlockSize; ++tid)
    expected[tid] = input[tid ^ 64];
  run_unary_kernel(lds_cross_wave_exchange_high_kernel, input, expected);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsB128ReverseHighOffset) {
  const std::vector<uint32_t> input = make_input(kBlockSize * 4, 0x404u);
  std::vector<uint32_t> expected(input.size());
  for (int tid = 0; tid < kBlockSize; ++tid) {
    const int source = tid * 4;
    const int target = (kBlockSize - 1 - tid) * 4;
    for (int lane = 0; lane < 4; ++lane)
      expected[source + lane] = input[target + lane];
  }
  run_unary_kernel(lds_b128_reverse_high_kernel, input, expected);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsB128ReverseImmediateOffset) {
  const std::vector<uint32_t> input = make_input(kBlockSize * 4, 0x505u);
  std::vector<uint32_t> expected(input.size());
  for (int tid = 0; tid < kBlockSize; ++tid) {
    const int source = tid * 4;
    const int target = (kBlockSize - 1 - tid) * 4;
    for (int lane = 0; lane < 4; ++lane)
      expected[source + lane] = input[target + lane];
  }
  run_unary_kernel(lds_b128_reverse_immediate_offset_kernel, input, expected);
}

TEST(HipLdsCopyDbtTest, LargeStaticLdsReadB64TrB16) {
  std::vector<uint32_t> expected(kTrB16BlockSize * 2);
  for (int tid = 0; tid < kTrB16BlockSize; ++tid) {
    const int wave_base = (tid / 64) * 64;
    const int lane = tid & 63;
    const int halfword = lane & 3;
    const int source_base = wave_base + (lane & 0x30) + ((lane & 0x0c) >> 2);
    const uint32_t h0 = tr_b16_halfword_value(source_base + 0, halfword);
    const uint32_t h1 = tr_b16_halfword_value(source_base + 4, halfword);
    const uint32_t h2 = tr_b16_halfword_value(source_base + 8, halfword);
    const uint32_t h3 = tr_b16_halfword_value(source_base + 12, halfword);
    expected[tid * 2 + 0] = pack_u16_pair(h0, h1);
    expected[tid * 2 + 1] = pack_u16_pair(h2, h3);
  }
  run_output_kernel(lds_read_b64_tr_b16_kernel, expected, kTrB16BlockSize);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}
