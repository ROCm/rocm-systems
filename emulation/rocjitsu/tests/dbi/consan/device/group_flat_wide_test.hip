// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Group-FLAT subword and wide LDS publication, distilled from framework
/// kernels that address shared storage through generic pointers.

#include "consan_device_test_support.h"

namespace {

#if defined(__gfx942__) || defined(__gfx950__)
#define RJ_GROUP_FLAT_ADD "v_add_u32 v0, v0, %[offset]\n\t"
#define RJ_GROUP_FLAT_SA_WAIT
#define RJ_GROUP_FLAT_SET_VCC "v_cmp_eq_u32_e32 vcc, %[value], %[value]\n\t"
#define RJ_GROUP_FLAT_WAIT "s_waitcnt vmcnt(0)\n\t"
#define RJ_GROUP_FLAT_LOAD_B16 "flat_load_ushort v20, v[0:1]\n\t"
#define RJ_GROUP_FLAT_LOAD_B64 "flat_load_dwordx2 v[20:21], v[0:1]\n\t"
#define RJ_GROUP_FLAT_LOAD_B128 "flat_load_dwordx4 v[20:23], v[0:1]\n\t"
#define RJ_GROUP_FLAT_STORE_B16 "flat_store_short v[0:1], %[value]\n\t"
#define RJ_GROUP_FLAT_STORE_B64 "flat_store_dwordx2 v[0:1], v[20:21]\n\t"
#define RJ_GROUP_FLAT_STORE_B128 "flat_store_dwordx4 v[0:1], v[20:23]\n\t"
#else
#define RJ_GROUP_FLAT_ADD "v_add_nc_u32 v0, v0, %[offset]\n\t"
#if defined(__gfx1100__)
#define RJ_GROUP_FLAT_SA_WAIT
#define RJ_GROUP_FLAT_SET_VCC "v_cmp_eq_u32_e32 vcc_lo, %[value], %[value]\n\t"
#define RJ_GROUP_FLAT_WAIT "s_waitcnt vmcnt(0)\n\t"
#else
#define RJ_GROUP_FLAT_SA_WAIT "s_wait_alu depctr_sa_sdst(0)\n\t"
#define RJ_GROUP_FLAT_SET_VCC "v_cmp_eq_u32_e32 %[value], %[value]\n\t"
#define RJ_GROUP_FLAT_WAIT "s_wait_loadcnt 0\n\t"
#endif
#define RJ_GROUP_FLAT_LOAD_B16 "flat_load_u16 v20, v[0:1]\n\t"
#define RJ_GROUP_FLAT_LOAD_B64 "flat_load_b64 v[20:21], v[0:1]\n\t"
#define RJ_GROUP_FLAT_LOAD_B128 "flat_load_b128 v[20:23], v[0:1]\n\t"
#define RJ_GROUP_FLAT_STORE_B16 "flat_store_b16 v[0:1], %[value]\n\t"
#define RJ_GROUP_FLAT_STORE_B64 "flat_store_b64 v[0:1], v[20:21]\n\t"
#define RJ_GROUP_FLAT_STORE_B128 "flat_store_b128 v[0:1], v[20:23]\n\t"
#endif

constexpr uint32_t kMaximumThreads = 128u;

struct DeviceState {
  uint32_t synchronize;
  uint32_t output[7];
  uint32_t vcc_survived;
  uint32_t control[kMaximumThreads];
};

__device__ __forceinline__ void group_flat_store_b16(uint32_t offset, uint32_t value) {
  asm volatile(
      "s_mov_b64 s[14:15], src_shared_base\n\t"
      "v_mov_b32 v0, s14\n\t"
      "v_mov_b32 v1, s15\n\t" RJ_GROUP_FLAT_ADD RJ_GROUP_FLAT_STORE_B16 RJ_CONSAN_DEVICE_LARGE_CAVE
      :
      : [offset] "v"(offset), [value] "v"(value)
      : "s14", "s15", "v0", "v1", "vcc", "memory");
}

__device__ __forceinline__ void group_flat_store_b64(uint32_t offset, uint2 value) {
  register uint32_t value0 asm("v20") = value.x;
  register uint32_t value1 asm("v21") = value.y;
  asm volatile(
      "s_mov_b64 s[14:15], src_shared_base\n\t"
      "v_mov_b32 v0, s14\n\t"
      "v_mov_b32 v1, s15\n\t" RJ_GROUP_FLAT_ADD RJ_GROUP_FLAT_STORE_B64 RJ_CONSAN_DEVICE_LARGE_CAVE
      :
      : [offset] "v"(offset), "v"(value0), "v"(value1)
      : "s14", "s15", "v0", "v1", "v20", "v21", "vcc", "memory");
}

__device__ __forceinline__ uint32_t group_flat_store_b128(uint32_t offset, uint4 value) {
  register uint32_t value0 asm("v20") = value.x;
  register uint32_t value1 asm("v21") = value.y;
  register uint32_t value2 asm("v22") = value.z;
  register uint32_t value3 asm("v23") = value.w;
  uint32_t vcc_survived = 0u;
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t" RJ_GROUP_FLAT_SA_WAIT
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t" RJ_GROUP_FLAT_ADD RJ_GROUP_FLAT_STORE_B128
                   RJ_CONSAN_DEVICE_LARGE_CAVE RJ_GROUP_FLAT_SET_VCC "s_cbranch_vccz 1f\n\t"
               "v_mov_b32 %[survived], 1\n\t"
               "1:\n\t"
               : [survived] "+v"(vcc_survived)
               : [offset] "v"(offset), [value] "v"(value0), "v"(value1), "v"(value2), "v"(value3)
               : "s14", "s15", "v0", "v1", "vcc", "memory");
  return vcc_survived;
}

__device__ __forceinline__ uint4 group_flat_load_b128(uint32_t offset) {
  register uint32_t value0 asm("v20") = 0u;
  register uint32_t value1 asm("v21") = 0u;
  register uint32_t value2 asm("v22") = 0u;
  register uint32_t value3 asm("v23") = 0u;
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t" RJ_GROUP_FLAT_SA_WAIT
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t" RJ_GROUP_FLAT_ADD RJ_GROUP_FLAT_LOAD_B128
                   RJ_CONSAN_DEVICE_LARGE_CAVE RJ_GROUP_FLAT_WAIT
               : "+v"(value0), "+v"(value1), "+v"(value2), "+v"(value3)
               : [offset] "v"(offset)
               : "s14", "s15", "v0", "v1", "vcc", "memory");
  return make_uint4(value0, value1, value2, value3);
}

__device__ __forceinline__ uint2 group_flat_load_b64(uint32_t offset) {
  register uint32_t value0 asm("v20") = 0u;
  register uint32_t value1 asm("v21") = 0u;
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t" RJ_GROUP_FLAT_SA_WAIT
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t" RJ_GROUP_FLAT_ADD RJ_GROUP_FLAT_LOAD_B64
                   RJ_CONSAN_DEVICE_LARGE_CAVE RJ_GROUP_FLAT_WAIT
               : "+v"(value0), "+v"(value1)
               : [offset] "v"(offset)
               : "s14", "s15", "v0", "v1", "vcc", "memory");
  return make_uint2(value0, value1);
}

__device__ __forceinline__ uint32_t group_flat_load_b16(uint32_t offset) {
  register uint32_t value asm("v20") = 0u;
  asm volatile("s_mov_b64 s[14:15], src_shared_base\n\t" RJ_GROUP_FLAT_SA_WAIT
               "v_mov_b32 v0, s14\n\t"
               "v_mov_b32 v1, s15\n\t" RJ_GROUP_FLAT_ADD RJ_GROUP_FLAT_LOAD_B16
                   RJ_CONSAN_DEVICE_LARGE_CAVE RJ_GROUP_FLAT_WAIT
               : "+v"(value)
               : [offset] "v"(offset)
               : "s14", "s15", "v0", "v1", "vcc", "memory");
  return value;
}

__global__ void group_flat_wide_for_consan(DeviceState *state) {
  __shared__ __align__(16) uint8_t storage[32];
  // Keep the compiler-visible LDS object in the kernel descriptor. The
  // group-FLAT assembly addresses it through src_shared_base, so without an
  // explicit operand the compiler can otherwise remove the allocation and
  // emit a zero-byte group segment even though the instructions use LDS.
  asm volatile("" : : "v"(storage) : "memory");
  const uint32_t tid = threadIdx.x;
  const uint32_t wave_size = warpSize;

  if (tid == 0u) {
    state->vcc_survived =
        group_flat_store_b128(0u, make_uint4(0x10203040u, 0x50607080u, 0x90a0b0c0u, 0xd0e0f001u));
    group_flat_store_b64(16u, make_uint2(0x13579bdfu, 0x2468ace0u));
    group_flat_store_b16(24u, 0x0000beefu);
  }
  if (state->synchronize != 0u)
    __syncthreads();

  if (tid == wave_size) {
    const uint4 wide = group_flat_load_b128(0u);
    const uint2 pair = group_flat_load_b64(16u);
    state->output[0] = wide.x;
    state->output[1] = wide.y;
    state->output[2] = wide.z;
    state->output[3] = wide.w;
    state->output[4] = pair.x;
    state->output[5] = pair.y;
    state->output[6] = group_flat_load_b16(24u);
  }
  state->control[tid] = 0xf1a70000u | tid;
}

void run_group_flat_wide(bool synchronize) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  DeviceState initial{};
  initial.synchronize = synchronize ? 1u : 0u;
  DeviceState *device_state = nullptr;
  HIP_ASSERT(hipMalloc(&device_state, sizeof(DeviceState)));
  HIP_ASSERT(hipMemcpy(device_state, &initial, sizeof(DeviceState), hipMemcpyHostToDevice));
  group_flat_wide_for_consan<<<1, thread_count>>>(device_state);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  DeviceState result{};
  HIP_ASSERT(hipMemcpy(&result, device_state, sizeof(DeviceState), hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(device_state));
  for (uint32_t tid = 0; tid < thread_count; ++tid)
    EXPECT_EQ(result.control[tid], 0xf1a70000u | tid) << "thread=" << tid;
  EXPECT_EQ(result.vcc_survived, 1u);
  if (!synchronize)
    return;
  constexpr uint32_t expected[] = {0x10203040u, 0x50607080u, 0x90a0b0c0u, 0xd0e0f001u,
                                   0x13579bdfu, 0x2468ace0u, 0x0000beefu};
  for (uint32_t i = 0; i < 7u; ++i)
    EXPECT_EQ(result.output[i], expected[i]) << "element=" << i;
}

TEST(ConSanDeviceGroupFlatWideTest, Correct) { run_group_flat_wide(true); }
TEST(ConSanDeviceGroupFlatWideTest, Incorrect) { run_group_flat_wide(false); }

} // namespace
