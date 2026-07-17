// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hsa_invalid_packet_test.cpp
/// @brief End-to-end ROCr queue callback test for an unsupported AQL packet.

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include <hsa/hsa.h>
RJ_DIAGNOSTIC_POP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

namespace {

struct alignas(64) UnknownVendorPacket {
  uint16_t header;
  uint8_t amd_format;
  uint8_t reserved[53];
  hsa_signal_t completion_signal;
};
static_assert(sizeof(UnknownVendorPacket) == 64);
static_assert(offsetof(UnknownVendorPacket, completion_signal) == 56);

std::atomic<uint32_t> callback_count{0};
std::atomic<hsa_status_t> callback_status{HSA_STATUS_SUCCESS};

hsa_status_t find_gpu(hsa_agent_t agent, void *data) {
  hsa_device_type_t type{};
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS &&
      type == HSA_DEVICE_TYPE_GPU)
    *static_cast<hsa_agent_t *>(data) = agent;
  return HSA_STATUS_SUCCESS;
}

void queue_error(hsa_status_t status, hsa_queue_t *, void *) {
  callback_status.store(status, std::memory_order_relaxed);
  callback_count.fetch_add(1, std::memory_order_release);
}

} // namespace

int main() {
  if (hsa_init() != HSA_STATUS_SUCCESS)
    return 1;

  hsa_agent_t gpu{};
  if (hsa_iterate_agents(find_gpu, &gpu) != HSA_STATUS_SUCCESS || gpu.handle == 0)
    return 2;

  uint32_t queue_size = 0;
  if (hsa_agent_get_info(gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size) != HSA_STATUS_SUCCESS)
    return 3;

  hsa_queue_t *queue = nullptr;
  if (hsa_queue_create(gpu, queue_size, HSA_QUEUE_TYPE_MULTI, queue_error, nullptr, UINT32_MAX,
                       UINT32_MAX, &queue) != HSA_STATUS_SUCCESS)
    return 4;

  hsa_signal_t completion{};
  if (hsa_signal_create(1, 0, nullptr, &completion) != HSA_STATUS_SUCCESS)
    return 5;

  const uint64_t index = hsa_queue_add_write_index_relaxed(queue, 1);
  auto *packet =
      static_cast<UnknownVendorPacket *>(queue->base_address) + (index & (queue->size - 1));
  std::memset(packet, 0, sizeof(*packet));
  packet->amd_format = 0xff;
  packet->completion_signal = completion;
  uint16_t header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
  __atomic_store_n(reinterpret_cast<uint16_t *>(packet), header, __ATOMIC_RELEASE);
  hsa_signal_store_relaxed(queue->doorbell_signal, index);

  for (uint32_t i = 0; i < 500 && callback_count.load(std::memory_order_acquire) == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  const bool passed =
      callback_count.load(std::memory_order_acquire) == 1 &&
      callback_status.load(std::memory_order_relaxed) == HSA_STATUS_ERROR_INVALID_PACKET_FORMAT &&
      hsa_signal_load_scacquire(completion) == 1 &&
      hsa_queue_load_read_index_scacquire(queue) == index;
  hsa_queue_destroy(queue);
  hsa_signal_destroy(completion);
  hsa_shut_down();
  return passed ? 0 : 6;
}
