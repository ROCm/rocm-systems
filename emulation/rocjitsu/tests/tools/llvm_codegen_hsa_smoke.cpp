// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string_view>
#include <vector>

#define HSA_CHECK(call)                                                                            \
  do {                                                                                             \
    const hsa_status_t status = (call);                                                            \
    if (status != HSA_STATUS_SUCCESS) {                                                            \
      const char *description = nullptr;                                                           \
      hsa_status_string(status, &description);                                                     \
      std::fprintf(stderr, "%s: %#x (%s)\n", #call, status,                                        \
                   description ? description : "unknown");                                         \
      return 1;                                                                                    \
    }                                                                                              \
  } while (false)

int main(int argc, char **argv) {
  if (argc != 3 ||
      (std::string_view(argv[1]) != "execute" && std::string_view(argv[1]) != "reject")) {
    std::fprintf(stderr, "Usage: %s execute|reject CODE_OBJECT\n", argv[0]);
    return 2;
  }
  std::ifstream file(argv[2], std::ios::binary);
  if (!file)
    return 2;
  const std::vector<char> image{std::istreambuf_iterator<char>(file), {}};
  HSA_CHECK(hsa_init());
  struct Agents {
    hsa_agent_t cpu{}, gpu{};
  } agents;
  HSA_CHECK(hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) {
        auto &agents = *static_cast<Agents *>(data);
        hsa_device_type_t type{};
        const auto status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        if (status != HSA_STATUS_SUCCESS)
          return status;
        if (type == HSA_DEVICE_TYPE_GPU)
          agents.gpu = agent;
        else if (type == HSA_DEVICE_TYPE_CPU)
          agents.cpu = agent;
        return HSA_STATUS_SUCCESS;
      },
      &agents));
  if (!agents.cpu.handle || !agents.gpu.handle)
    return 1;

  hsa_code_object_reader_t reader{};
  hsa_executable_t executable{};
  HSA_CHECK(hsa_code_object_reader_create_from_memory(image.data(), image.size(), &reader));
  HSA_CHECK(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                      nullptr, &executable));
  const auto load =
      hsa_executable_load_agent_code_object(executable, agents.gpu, reader, nullptr, nullptr);
  if (std::string_view(argv[1]) == "reject") {
    if (load != HSA_STATUS_ERROR_INVALID_ISA && load != HSA_STATUS_ERROR_INVALID_ISA_NAME &&
        load != HSA_STATUS_ERROR_INVALID_CODE_OBJECT &&
        load != HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS) {
      std::fprintf(stderr, "Expected incompatible object rejection, got %#x\n", load);
      return 1;
    }
    HSA_CHECK(hsa_executable_destroy(executable));
    HSA_CHECK(hsa_code_object_reader_destroy(reader));
    HSA_CHECK(hsa_shut_down());
    return 0;
  }
  HSA_CHECK(load);
  HSA_CHECK(hsa_executable_freeze(executable, nullptr));
  hsa_executable_symbol_t symbol{};
  uint64_t kernel_object = 0;
  uint32_t group_size = 0, private_size = 0;
  HSA_CHECK(hsa_executable_get_symbol_by_name(executable, "smoke.kd", &agents.gpu, &symbol));
  HSA_CHECK(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                           &kernel_object));
  HSA_CHECK(hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_size));
  HSA_CHECK(hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_size));

  hsa_amd_memory_pool_t pool{};
  HSA_CHECK(hsa_amd_agent_iterate_memory_pools(
      agents.cpu,
      [](hsa_amd_memory_pool_t candidate, void *data) {
        hsa_amd_segment_t segment{};
        hsa_amd_memory_pool_get_info(candidate, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
        if (segment != HSA_AMD_SEGMENT_GLOBAL)
          return HSA_STATUS_SUCCESS;
        uint32_t flags = 0;
        hsa_amd_memory_pool_get_info(candidate, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT)
          *static_cast<hsa_amd_memory_pool_t *>(data) = candidate;
        return HSA_STATUS_SUCCESS;
      },
      &pool));
  if (!pool.handle)
    return 1;
  constexpr size_t count = 128;
  void *allocation = nullptr;
  HSA_CHECK(hsa_amd_memory_pool_allocate(pool, 2 * count * sizeof(uint32_t) + 256, 0, &allocation));
  HSA_CHECK(hsa_amd_agents_allow_access(1, &agents.gpu, nullptr, allocation));
  auto *input = static_cast<uint32_t *>(allocation);
  auto *output = input + count;
  auto *arguments = reinterpret_cast<void **>(output + count);
  std::memset(allocation, 0, 2 * count * sizeof(uint32_t) + 256);
  for (size_t i = 0; i < count; ++i)
    input[i] = static_cast<uint32_t>(3 * i);
  arguments[0] = input;
  arguments[1] = output;

  hsa_queue_t *queue = nullptr;
  hsa_signal_t completion{};
  HSA_CHECK(hsa_queue_create(agents.gpu, 128, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX,
                             UINT32_MAX, &queue));
  HSA_CHECK(hsa_signal_create(1, 0, nullptr, &completion));
  const uint64_t index = hsa_queue_add_write_index_relaxed(queue, 1);
  auto *packet = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
                 (index & (queue->size - 1));
  std::memset(packet, 0, sizeof(*packet));
  packet->setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  packet->workgroup_size_x = 64;
  packet->workgroup_size_y = packet->workgroup_size_z = 1;
  packet->grid_size_x = count;
  packet->grid_size_y = packet->grid_size_z = 1;
  packet->kernel_object = kernel_object;
  packet->group_segment_size = group_size;
  packet->private_segment_size = private_size;
  packet->kernarg_address = arguments;
  packet->completion_signal = completion;
  const uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
                          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
  __atomic_store_n(&packet->header, header, __ATOMIC_RELEASE);
  hsa_signal_store_screlease(queue->doorbell_signal, index);
  if (hsa_signal_wait_scacquire(completion, HSA_SIGNAL_CONDITION_EQ, 0, 5'000'000'000ULL,
                                HSA_WAIT_STATE_BLOCKED) != 0) {
    std::fprintf(stderr, "Kernel completion timed out\n");
    return 1;
  }
  for (size_t i = 0; i < count; ++i) {
    if (output[i] != input[i] + 7) {
      std::fprintf(stderr, "Element %zu: got %u, expected %u\n", i, output[i], input[i] + 7);
      return 1;
    }
  }
  HSA_CHECK(hsa_signal_destroy(completion));
  HSA_CHECK(hsa_queue_destroy(queue));
  HSA_CHECK(hsa_amd_memory_pool_free(allocation));
  HSA_CHECK(hsa_executable_destroy(executable));
  HSA_CHECK(hsa_code_object_reader_destroy(reader));
  HSA_CHECK(hsa_shut_down());
  std::puts("128 kernel results matched through HSA");
  return 0;
}
