/*
 * Manual HIP device-coverage PoC.
 *
 * These explicit record_edge calls stand in for hooks that a future AMDGCN
 * instrumentation backend would insert at basic-block entries.
 */

#include <hip/hip_runtime.h>

#include <stdint.h>

#define ROCM_POC_MAP_SIZE 65536u
#define ROCM_POC_DEVICE_START (ROCM_POC_MAP_SIZE / 2u)
#define ROCM_POC_DEVICE_SLOTS (ROCM_POC_MAP_SIZE / 2u)

enum ManualBasicBlockId : uint32_t {
  BB_KERNEL_ENTRY = 0x8f123001u,
  BB_SELECT_ONE = 0x8f123101u,
  BB_SELECT_TWO = 0x8f123201u,
  BB_SELECT_THREE = 0x8f123301u,
  BB_SELECT_OTHER = 0x8f123f01u,
  BB_KERNEL_EXIT = 0x8f123ff1u,
};

__device__ inline __attribute__((always_inline)) uint64_t flat_workitem_id() {
  return threadIdx.x + blockIdx.x * blockDim.x +
         threadIdx.y * blockDim.x * gridDim.x +
         blockIdx.y * blockDim.x * blockDim.y * gridDim.x +
         threadIdx.z * blockDim.x * blockDim.y * gridDim.x * gridDim.y +
         blockIdx.z * blockDim.x * blockDim.y * blockDim.z * gridDim.x *
             gridDim.y;
}

__device__ inline __attribute__((always_inline)) void record_edge(
    uint32_t bb_id, uint32_t *coverage, uint32_t *previous_bb,
    uint32_t previous_bb_entries) {
  const uint64_t tid = flat_workitem_id();
  if (tid >= previous_bb_entries) {
    return;
  }

  const uint32_t previous = previous_bb[tid];
  const uint32_t edge =
      ROCM_POC_DEVICE_START + ((previous ^ bb_id) % ROCM_POC_DEVICE_SLOTS);

  previous_bb[tid] = bb_id >> 1;
  atomicAdd(&coverage[edge], 1u);
}

extern "C" __global__ void manual_coverage_kernel(
    int select, uint32_t *result, uint32_t *coverage, uint32_t *previous_bb,
    uint32_t previous_bb_entries) {
  const uint64_t tid = flat_workitem_id();

  record_edge(BB_KERNEL_ENTRY, coverage, previous_bb, previous_bb_entries);

  uint32_t value = 0;
  if (select == 1) {
    record_edge(BB_SELECT_ONE, coverage, previous_bb, previous_bb_entries);
    value = 11;
  } else if (select == 2) {
    record_edge(BB_SELECT_TWO, coverage, previous_bb, previous_bb_entries);
    value = 22;
  } else if (select == 3) {
    record_edge(BB_SELECT_THREE, coverage, previous_bb, previous_bb_entries);
    value = 33;
  } else {
    record_edge(BB_SELECT_OTHER, coverage, previous_bb, previous_bb_entries);
    value = 99;
  }

  if (tid == 0) {
    *result = value;
  }

  record_edge(BB_KERNEL_EXIT, coverage, previous_bb, previous_bb_entries);
}
