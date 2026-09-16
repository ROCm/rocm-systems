// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// A fixed two-wave fault with independently selectable workgroup and LDS cell.
// Only one lane in each wave accesses LDS, so retained-address diversity cannot
// explain a miss. The ordered control differs by exactly one uniform barrier.
#include "consan_device_test_support.h"
#include <vector>

namespace {

struct State {
  uint32_t cell;
  uint32_t ordered;
  uint32_t observed[256];
  uint32_t control[256 * 128];
};

__global__ void sampled_selection_for_consan(State *state) {
  __shared__ uint32_t words[2];
  const uint32_t tid = threadIdx.x;
  const uint32_t wave = warpSize;
  volatile uint32_t *address = &words[state->cell];
  if (tid == 0)
    consan_device_test::store_lds_word_without_cave(address, 0x12345678u);
  if (state->ordered)
    __syncthreads();
  if (tid == wave)
    consan_device_test::store_lds_word_without_cave(address, 0x87654321u);
  __syncthreads();
  if (tid == 0)
    state->observed[blockIdx.x] = consan_device_test::load_lds_word_without_cave(address);
  state->control[blockIdx.x * 128u + tid] = 0x63000000u | (blockIdx.x * 128u + tid);
}

void run_selection(uint32_t cell, bool ordered, uint32_t workgroups = 256) {
  State state{};
  state.cell = cell;
  state.ordered = ordered;
  const uint32_t threads = 2u * consan_device_test::active_wave_size();
  State *device = nullptr;
  HIP_ASSERT(hipMalloc(&device, sizeof(state)));
  HIP_ASSERT(hipMemcpy(device, &state, sizeof(state), hipMemcpyHostToDevice));
  // Sampled rotates the workgroup residue by dispatch identity. Covering all
  // 256 x coordinates guarantees a selected workgroup at the production rate.
  sampled_selection_for_consan<<<workgroups, threads>>>(device);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  HIP_ASSERT(hipMemcpy(&state, device, sizeof(state), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device));
  for (uint32_t group = 0; group < workgroups; ++group) {
    for (uint32_t tid = 0; tid < threads; ++tid)
      EXPECT_EQ(state.control[group * 128u + tid], 0x63000000u | (group * 128u + tid));
    if (ordered)
      EXPECT_EQ(state.observed[group], 0x87654321u);
    // Output corruption is not required for the unordered write/write race.
    else
      EXPECT_TRUE(state.observed[group] == 0x12345678u || state.observed[group] == 0x87654321u);
  }
}

TEST(ConSanDeviceSampledSelectionTest, RaceCellZero) { run_selection(0, false); }
TEST(ConSanDeviceSampledSelectionTest, RaceCellOne) { run_selection(1, false); }
TEST(ConSanDeviceSampledSelectionTest, OrderedCellZero) { run_selection(0, true); }

TEST(ConSanDeviceSampledSelectionTest, RaceCellOneSingleWorkgroup) { run_selection(1, false, 1); }
TEST(ConSanDeviceSampledSelectionTest, RaceCellZeroSingleWorkgroup) { run_selection(0, false, 1); }

} // namespace

namespace {

// Keep the broadcast inside the same basic block as the access, including
// partial EXEC cases. Values differ by lane: uniform addresses are not benign.
template <int Mode> __global__ void sampled_uniform_for_consan(uint32_t *output) {
  __shared__ uint32_t words[128];
  const uint32_t lane = threadIdx.x;
  uint32_t base =
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(words)));
  if constexpr (Mode == 7)
    base += sizeof(uint32_t); // Cell 1: excluded by high, included by max.
  if constexpr (Mode == 3 || Mode == 4) {
    if (lane == 0)
      consan_device_test::store_lds_word_without_cave(words, 7);
    __syncthreads();
  }
  if constexpr (Mode == 6) {
    uint64_t saved_exec;
    uint32_t address;
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__)
#define RJ_EMPTY_STORE "ds_write_b32 %1, %2\n\t"
#else
#define RJ_EMPTY_STORE "ds_store_b32 %1, %2\n\t"
#endif
    asm volatile("s_and_saveexec_b64 %0, 0\n\t"
                 "v_mov_b32 %1, %3\n\t" RJ_EMPTY_STORE "s_mov_b64 exec, %0\n\t"
                 : "=&s"(saved_exec), "=&v"(address)
                 : "v"(lane + 1), "s"(base)
                 : "memory", "scc");
#undef RJ_EMPTY_STORE
  } else if constexpr (Mode == 4) {
    atomicAdd(words, 1u);
  } else if constexpr (Mode == 3) {
    output[blockIdx.x * 128 + lane] = consan_device_test::load_lds_word_without_cave(words);
  } else if constexpr (Mode == 2) {
    consan_device_test::store_lds_word_without_cave(words + lane, lane + 1);
  } else {
    if ((Mode == 0 || Mode == 7) || (Mode == 1 && lane == 0) ||
        (Mode == 5 && (lane == 0 || lane == 3))) {
      uint32_t address;
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__)
#define RJ_UNIFORM_STORE "ds_write_b32 %0, %1\n\t"
#else
#define RJ_UNIFORM_STORE "ds_store_b32 %0, %1\n\t"
#endif
      asm volatile("v_mov_b32 %0, %2\n\t" RJ_UNIFORM_STORE
                   : "=&v"(address)
                   : "v"(lane + 1), "s"(base)
                   : "memory");
#undef RJ_UNIFORM_STORE
    }
  }
  __syncthreads();
  if constexpr (Mode == 6) {
    output[blockIdx.x * 128 + lane] = lane;
  } else if constexpr (Mode != 3) {
    output[blockIdx.x * 128 + lane] = consan_device_test::load_lds_word_without_cave(
        words + (Mode == 2 ? lane : (Mode == 7 ? 1 : 0)));
  }
}

template <int Mode> void run_uniform(uint32_t groups = 1, uint32_t waves = 1) {
  const uint32_t wave = waves * consan_device_test::active_wave_size();
  uint32_t *device = nullptr;
  HIP_ASSERT(hipMalloc(&device, groups * 128u * sizeof(uint32_t)));
  sampled_uniform_for_consan<Mode><<<groups, wave>>>(device);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint32_t> values(groups * 128u);
  HIP_ASSERT(
      hipMemcpy(values.data(), device, values.size() * sizeof(uint32_t), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device));
  for (uint32_t group = 0; group < groups; ++group) {
    for (uint32_t lane = 0; lane < wave; ++lane) {
      const uint32_t value = values[group * 128 + lane];
      if constexpr (Mode == 0 || Mode == 7) {
        EXPECT_GE(value, 1u);
        EXPECT_LE(value, wave);
      } else if constexpr (Mode == 1) {
        EXPECT_EQ(value, 1u);
      } else if constexpr (Mode == 2) {
        EXPECT_EQ(value, lane + 1u);
      } else if constexpr (Mode == 3) {
        EXPECT_EQ(value, 7u);
      } else if constexpr (Mode == 4) {
        EXPECT_EQ(value, 7u + wave);
      } else if constexpr (Mode == 6) {
        EXPECT_EQ(value, lane);
      } else {
        EXPECT_TRUE(value == 1u || value == 4u);
      }
    }
  }
}

TEST(ConSanDeviceSampledUniformTest, EmptyExec) { run_uniform<6>(); }
TEST(ConSanDeviceSampledUniformTest, CrossWave) { run_uniform<0>(1, 2); }
TEST(ConSanDeviceSampledUniformTest, DifferentValues) { run_uniform<0>(); }
TEST(ConSanDeviceSampledUniformTest, DefaultSelected) { run_uniform<0>(256); }
TEST(ConSanDeviceSampledUniformTest, SingleWriter) { run_uniform<1>(); }
TEST(ConSanDeviceSampledUniformTest, Disjoint) { run_uniform<2>(); }
TEST(ConSanDeviceSampledUniformTest, ReadOnly) { run_uniform<3>(); }
TEST(ConSanDeviceSampledUniformTest, Atomic) { run_uniform<4>(); }
TEST(ConSanDeviceSampledUniformTest, PartialExec) { run_uniform<5>(); }

} // namespace

TEST(ConSanDeviceSampledUniformTest, NonzeroCell) { run_uniform<7>(); }

// Identical data must be proven for every word; one varying word is enough
// to retain the intra-wave diagnostic. Multiple waves must still conflict.
template <bool VaryLast> __global__ void uniform_value_b128(uint32_t *out, uint32_t seed) {
  __shared__ alignas(16) uint32_t words[4];
  uint32_t base =
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(words)));
  const uint32_t a = __builtin_amdgcn_readfirstlane(seed + 11);
  const uint32_t b = __builtin_amdgcn_readfirstlane(seed + 22);
  const uint32_t c = __builtin_amdgcn_readfirstlane(seed + 33);
  const uint32_t d = __builtin_amdgcn_readfirstlane(seed + 44);
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1250__)
#define RJ_VALUE_STORE "ds_write_b128 %0, v[20:23]\n\t"
#else
#define RJ_VALUE_STORE "ds_store_b128 %0, v[20:23]\n\t"
#endif
  uint32_t addr;
  if constexpr (VaryLast) {
    asm volatile("v_mov_b32 %0, %1\n\t"
                 "v_mov_b32 v20, %2\n\t"
                 "v_mov_b32 v21, %3\n\t"
                 "v_mov_b32 v22, %4\n\t"
                 "v_mov_b32 v23, %5\n\t" RJ_VALUE_STORE
                 : "=&v"(addr)
                 : "s"(base), "s"(a), "s"(b), "s"(c), "v"(threadIdx.x + d)
                 : "v20", "v21", "v22", "v23", "memory");
  } else {
    asm volatile("v_mov_b32 %0, %1\n\t"
                 "v_mov_b32 v20, %2\n\t"
                 "v_mov_b32 v21, %3\n\t"
                 "v_mov_b32 v22, %4\n\t"
                 "v_mov_b32 v23, %5\n\t" RJ_VALUE_STORE
                 : "=&v"(addr)
                 : "s"(base), "s"(a), "s"(b), "s"(c), "s"(d)
                 : "v20", "v21", "v22", "v23", "memory");
  }
#undef RJ_VALUE_STORE
  __syncthreads();
  if (threadIdx.x == 0)
    for (unsigned i = 0; i < 4; ++i)
      out[i] = words[i];
}

template <bool VaryLast> void run_uniform_value(unsigned waves) {
  uint32_t *device = nullptr;
  HIP_ASSERT(hipMalloc(&device, 4 * sizeof(uint32_t)));
  uniform_value_b128<VaryLast><<<1, waves * consan_device_test::active_wave_size()>>>(device, 19);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  uint32_t result[4];
  HIP_ASSERT(hipMemcpy(result, device, sizeof(result), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device));
  EXPECT_EQ(result[0], 30u);
  EXPECT_EQ(result[1], 41u);
  EXPECT_EQ(result[2], 52u);
  if constexpr (!VaryLast)
    EXPECT_EQ(result[3], 63u);
}

TEST(ConSanDeviceUniformValueTest, IdenticalWords) { run_uniform_value<false>(1); }
TEST(ConSanDeviceUniformValueTest, VaryingLastWord) { run_uniform_value<true>(1); }
TEST(ConSanDeviceUniformValueTest, CrossWave) { run_uniform_value<false>(2); }
