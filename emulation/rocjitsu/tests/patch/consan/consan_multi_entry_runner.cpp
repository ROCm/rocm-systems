// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/target.h"
#include "rocjitsu/hooks/consan/api.h"
#include "rocjitsu/hooks/consan/testing.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

struct Agents {
  hsa_agent_t cpu = {};
  hsa_agent_t gpu = {};
  std::string_view target;
};

Agents agents(std::string_view target) {
  Agents found = {.target = target};
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) {
        auto &result = *static_cast<Agents *>(data);
        hsa_device_type_t type = {};
        if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
          return HSA_STATUS_SUCCESS;
        if (type == HSA_DEVICE_TYPE_CPU)
          result.cpu = agent;
        if (type == HSA_DEVICE_TYPE_GPU) {
          hsa_isa_t isa = {};
          uint32_t bytes = 0;
          if (hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa) == HSA_STATUS_SUCCESS &&
              hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &bytes) == HSA_STATUS_SUCCESS &&
              bytes > 0 && bytes <= 256) {
            std::array<char, 256> name = {};
            if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name.data()) == HSA_STATUS_SUCCESS) {
              std::string_view identity(name.data());
              constexpr std::string_view prefix = "amdgcn-amd-amdhsa--";
              if (identity.starts_with(prefix)) {
                identity.remove_prefix(prefix.size());
                identity = identity.substr(0, identity.find(':'));
                if (identity == result.target)
                  result.gpu = agent;
              }
            }
          }
        }
        return HSA_STATUS_SUCCESS;
      },
      &found);
  return found;
}

hsa_amd_memory_pool_t fine_pool(hsa_agent_t cpu) {
  hsa_amd_memory_pool_t found = {};
  hsa_amd_agent_iterate_memory_pools(
      cpu,
      [](hsa_amd_memory_pool_t pool, void *data) {
        hsa_amd_segment_t segment = {};
        uint32_t flags = 0;
        bool allowed = false;
        if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) ==
                HSA_STATUS_SUCCESS &&
            segment == HSA_AMD_SEGMENT_GLOBAL &&
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags) ==
                HSA_STATUS_SUCCESS &&
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                                         &allowed) == HSA_STATUS_SUCCESS &&
            allowed &&
            (flags & (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
                      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) ==
                (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
                 HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
          *static_cast<hsa_amd_memory_pool_t *>(data) = pool;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &found);
  return found;
}

struct Entry {
  const char *name;
  uint32_t argument_bytes;
  uint32_t group_bytes;
  uint64_t kernel = 0;
  uint32_t alignment = 0;
};

int run(int argc, char **argv) {
  if (argc != 3 ||
      (std::string_view(argv[2]) != "baseline" && std::string_view(argv[2]) != "instrumented"))
    return 3;
  const bool instrumented = std::string_view(argv[2]) == "instrumented";
  if (hsa_init() != HSA_STATUS_SUCCESS)
    return 20;
  // Agent enumeration requires a live HSA runtime.
  const Agents initialized = agents("gfx1250");
  if (!initialized.cpu.handle || !initialized.gpu.handle)
    return 77;
  void *plugin = nullptr;
  rj_consan_finish_fn finish = nullptr;
  rj_consan_drain_fn drain = nullptr;
  rj_consan_statistics_fn statistics = nullptr;
  if (instrumented) {
    const char *path = std::getenv("RJ_CONSAN_PLUGIN_PATH");
    if (path)
      plugin = ::dlopen(path, RTLD_NOW | RTLD_NOLOAD);
    if (plugin) {
      finish = reinterpret_cast<rj_consan_finish_fn>(::dlsym(plugin, "rj_consan_finish"));
      drain = reinterpret_cast<rj_consan_drain_fn>(::dlsym(plugin, "rj_consan_drain"));
      statistics =
          reinterpret_cast<rj_consan_statistics_fn>(::dlsym(plugin, "rj_consan_get_statistics"));
    }
    if (!finish || !drain || !statistics || drain() != 0)
      return 3;
  }
  const auto require = [](hsa_status_t status, const char *operation) {
    if (status == HSA_STATUS_SUCCESS)
      return true;
    std::cerr << operation << " failed: " << uint32_t(status) << '\n';
    return false;
  };
  hsa_executable_t executable = {};
  hsa_code_object_reader_t reader = {};
  hsa_queue_t *queue = nullptr;
  hsa_signal_t signal = {};
  void *arguments = nullptr;
  void *output = nullptr;
  int result = 0;
  uint32_t loads = 0;
  uint32_t completed = 0;
  std::array<Entry, 3> entries = {
      Entry{.name = "mk_a.kd", .argument_bytes = 16, .group_bytes = 8},
      Entry{.name = "mk_b.kd", .argument_bytes = 32, .group_bytes = 256},
      Entry{.name = "mk_c.kd", .argument_bytes = 24, .group_bytes = 64}};
  const int descriptor = ::open(argv[1], O_RDONLY | O_CLOEXEC);
  if (descriptor < 0 ||
      !require(hsa_code_object_reader_create_from_file(descriptor, &reader), "reader") ||
      !require(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                         nullptr, &executable),
               "executable") ||
      !require(hsa_executable_load_agent_code_object(executable, initialized.gpu, reader, nullptr,
                                                     nullptr),
               "whole-image load"))
    result = 20;
  else
    ++loads;
  if (descriptor >= 0)
    ::close(descriptor);
  if (result == 0 && !require(hsa_executable_freeze(executable, nullptr), "freeze"))
    result = 20;
  for (Entry &entry : entries) {
    if (result != 0)
      break;
    hsa_executable_symbol_t symbol = {};
    uint32_t argument_bytes = 0, group_bytes = 0, private_bytes = 0;
    if (!require(
            hsa_executable_get_symbol_by_name(executable, entry.name, &initialized.gpu, &symbol),
            "symbol") ||
        !require(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                                &entry.kernel),
                 "kernel object") ||
        !require(
            hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &argument_bytes),
            "kernarg") ||
        !require(hsa_executable_symbol_get_info(
                     symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
                     &entry.alignment),
                 "kernarg alignment") ||
        !require(hsa_executable_symbol_get_info(
                     symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_bytes),
                 "group") ||
        !require(
            hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &private_bytes),
            "private") ||
        entry.kernel == 0 || argument_bytes != entry.argument_bytes || entry.alignment != 16 ||
        group_bytes != entry.group_bytes || private_bytes != 0) {
      std::cerr << "entry ABI mismatch: " << entry.name << " arguments=" << argument_bytes
                << " alignment=" << entry.alignment << " group=" << group_bytes
                << " private=" << private_bytes << " kernel=" << entry.kernel << '\n';
      result = 20;
    }
  }
  if (result == 0 &&
      (entries[0].kernel == entries[1].kernel || entries[0].kernel == entries[2].kernel ||
       entries[1].kernel == entries[2].kernel))
    result = 20;
  const hsa_amd_memory_pool_t pool = fine_pool(initialized.cpu);
  if (result == 0 &&
      (!pool.handle ||
       !require(hsa_amd_memory_pool_allocate(pool, 32, 0, &arguments), "arguments") ||
       !require(hsa_amd_memory_pool_allocate(pool, 16, 0, &output), "output") ||
       !require(hsa_amd_agents_allow_access(1, &initialized.gpu, nullptr, arguments),
                "args access") ||
       !require(hsa_amd_agents_allow_access(1, &initialized.gpu, nullptr, output), "out access") ||
       !require(hsa_queue_create(initialized.gpu, 128, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                 0, &queue),
                "queue")))
    result = 20;
  const std::array<uint32_t, 4> order = {0, 1, 2, 0};
  const std::array<uint32_t, 4> seeds = {7, 23, 55, 39};
  constexpr uint32_t kGuard = 0xdeadbeef;
  for (uint32_t ordinal = 0; result == 0 && ordinal < order.size(); ++ordinal) {
    const uint32_t index = order[ordinal];
    const Entry &entry = entries[index];
    const uint32_t seed = seeds[ordinal];
    const uint32_t bias = 100;
    if (!require(hsa_signal_create(1, 0, nullptr, &signal), "dedicated completion")) {
      result = 21;
      break;
    }
    std::memset(arguments, 0, 32);
    if (index == 0) {
      std::memcpy(arguments, &seed, sizeof(seed));
      std::memcpy(static_cast<uint8_t *>(arguments) + 8, &output, sizeof(output));
    } else {
      std::memcpy(arguments, &output, sizeof(output));
      std::memcpy(static_cast<uint8_t *>(arguments) + 16, &seed, sizeof(seed));
      if (index == 1)
        std::memcpy(static_cast<uint8_t *>(arguments) + 24, &bias, sizeof(bias));
    }
    std::fill_n(static_cast<uint32_t *>(output), 4, kGuard);
    const uint64_t write_index = hsa_queue_add_write_index_relaxed(queue, 1);
    auto *packet = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
                   (write_index & (queue->size - 1));
    std::memset(packet, 0, sizeof(*packet));
    packet->setup = 1;
    packet->workgroup_size_x = packet->grid_size_x = 64;
    packet->workgroup_size_y = packet->workgroup_size_z = 1;
    packet->grid_size_y = packet->grid_size_z = 1;
    packet->group_segment_size = entry.group_bytes;
    packet->kernel_object = entry.kernel;
    packet->kernarg_address = arguments;
    packet->completion_signal = signal;
    const uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                            (1u << HSA_PACKET_HEADER_BARRIER) |
                            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n(&packet->header, header, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(queue->doorbell_signal, write_index);
    hsa_signal_value_t value = 1;
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (value == 1 && std::chrono::steady_clock::now() < deadline)
      value = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, 1000000,
                                        HSA_WAIT_STATE_BLOCKED);
    if (value == 1) {
      std::cerr << "completion timeout; retaining live dispatch resources\n";
      ::_exit(24);
    }
    if (value != 0) {
      result = 21;
      break;
    }
    ++completed;
    std::array<uint32_t, 4> actual = {};
    std::memcpy(actual.data(), output, sizeof(actual));
    const uint32_t expected_first = seed + (index == 1 ? bias : 0);
    const uint32_t expected_second = expected_first + (index == 2 ? 2 : 1);
    const std::array<uint32_t, 4> expected = {expected_first, expected_second, kGuard, kGuard};
    if (actual != expected)
      result = 23;
    std::cout << "{\"ordinal\":" << ordinal + 1 << ",\"entry\":\"" << entry.name
              << "\",\"native\":" << (index == 2 ? "true" : "false")
              << ",\"kernarg_bytes\":" << entry.argument_bytes
              << ",\"kernarg_alignment\":" << entry.alignment
              << ",\"group_bytes\":" << entry.group_bytes << ",\"completion\":" << value
              << ",\"output\":[" << actual[0] << ',' << actual[1] << ',' << actual[2] << ','
              << actual[3] << "]}\n";
    if (drain && drain() != 0)
      result = 22;
    if (!require(hsa_signal_destroy(signal), "completion release"))
      result = 22;
    signal = {};
  }
  const int finish_status = finish ? finish() : 0;
  if (finish_status != 0 && result == 0)
    result = 22;
  const rj_consan_statistics stats = statistics ? statistics() : rj_consan_statistics{};
  if (signal.handle && !require(hsa_signal_destroy(signal), "signal cleanup") && result == 0)
    result = 22;
  if (queue && !require(hsa_queue_destroy(queue), "queue cleanup") && result == 0)
    result = 22;
  if (reader.handle && !require(hsa_code_object_reader_destroy(reader), "reader cleanup") &&
      result == 0)
    result = 22;
  if (executable.handle && !require(hsa_executable_destroy(executable), "executable cleanup") &&
      result == 0)
    result = 22;
  if (arguments && !require(hsa_amd_memory_pool_free(arguments), "arguments cleanup") &&
      result == 0)
    result = 22;
  if (output && !require(hsa_amd_memory_pool_free(output), "output cleanup") && result == 0)
    result = 22;
  if (!require(hsa_shut_down(), "shutdown") && result == 0)
    result = 22;
  if (plugin)
    ::dlclose(plugin);
  std::cout << "{\"app_status\":" << result << ",\"finish_status\":" << finish_status
            << ",\"loads\":" << loads << ",\"completed\":" << completed
            << ",\"submitted\":" << stats.submitted << ",\"forwarded\":" << stats.forwarded
            << ",\"copied\":" << stats.copied << ",\"written\":" << stats.written
            << ",\"reused\":" << stats.reused << ",\"native_submitted\":" << stats.native_submitted
            << ",\"native_completed\":" << stats.native_completed << "}\n";
  return result;
}
} // namespace

int main(int argc, char **argv) { return run(argc, argv); }
