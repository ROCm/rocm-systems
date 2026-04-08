#include <stdint.h>

#define QUEUE_BASE_ADDR_OFF    8
#define QUEUE_DOORBELL_OFF     16
#define QUEUE_SIZE_OFF         24
#define QUEUE_WRITE_ID_OFF     56
#define QUEUE_READ_ID_OFF      128

extern "C" __attribute__((amdgpu_flat_work_group_size(1, 1)))
__global__ void __amd_rocclr_graphSchedulerHIP(
    const uint8_t* __restrict__ cmd_buffer,
    uint32_t total_packet_count,
    uint8_t* __restrict__ queue_ptr) {

  uint8_t* qptr = queue_ptr ? queue_ptr
                             : (uint8_t*)__builtin_amdgcn_queue_ptr();

  uint8_t* base = *(uint8_t**)(qptr + QUEUE_BASE_ADDR_OFF);
  uint32_t q_size = *(uint32_t*)(qptr + QUEUE_SIZE_OFF);
  uint32_t q_mask = q_size - 1;
  volatile uint64_t* wr_id = (volatile uint64_t*)(qptr + QUEUE_WRITE_ID_OFF);
  volatile uint64_t* rd_id = (volatile uint64_t*)(qptr + QUEUE_READ_ID_OFF);
  volatile int64_t* doorbell = *(volatile int64_t**)(qptr + QUEUE_DOORBELL_OFF);

  // Reserve slots
  uint64_t wr = __atomic_fetch_add(wr_id, (uint64_t)total_packet_count,
                                    __ATOMIC_RELAXED);

  // Wait for space
  while (wr + total_packet_count - *rd_id > q_size) {}

  // Save first packet header, set to INVALID in command buffer
  uint32_t saved_header = *(volatile uint32_t*)cmd_buffer;
  *(volatile uint32_t*)cmd_buffer = 0;

  // Split copy into contiguous regions for wrap-around
  uint32_t start_slot = (uint32_t)(wr & q_mask);
  uint32_t slots_before_wrap = q_size - start_slot;
  uint32_t first_count = (total_packet_count <= slots_before_wrap)
                          ? total_packet_count : slots_before_wrap;
  uint32_t second_count = total_packet_count - first_count;

  // Region 1: bulk copy to queue
  __builtin_memcpy(base + start_slot * 64, cmd_buffer, first_count * 64);

  // Region 2: wrap-around copy from slot 0
  if (second_count > 0) {
    __builtin_memcpy(base, cmd_buffer + first_count * 64, second_count * 64);
  }

  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");

  // Flip first header — opens the gate for CP
  __atomic_store_n((uint32_t*)(base + start_slot * 64),
                   saved_header, __ATOMIC_RELEASE);

  // Ring doorbell
  __atomic_store_n(doorbell, (int64_t)(wr + total_packet_count - 1),
                   __ATOMIC_RELEASE);

  // Restore command buffer header for next launch
  *(volatile uint32_t*)cmd_buffer = saved_header;
}
