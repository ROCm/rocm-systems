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
        if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS) {
          return HSA_STATUS_SUCCESS;
        }
        if (type == HSA_DEVICE_TYPE_CPU) {
          result.cpu = agent;
        }
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
                if (identity == result.target) {
                  result.gpu = agent;
                }
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
  if ((argc < 3 || argc > 6) ||
      (std::string_view(argv[2]) != "baseline" && std::string_view(argv[2]) != "instrumented" &&
       std::string_view(argv[2]) != "native-application")) {
    return 3;
  }
  const bool instrumented = std::string_view(argv[2]) == "instrumented";
  const bool native_application = std::string_view(argv[2]) == "native-application";
  bool application_completion = false;
  uint32_t cluster_size = 0;
  uint32_t workgroups = 1;
  std::vector<std::string_view> images = {argv[1]};
  for (int index = 3; index < argc; ++index) {
    const std::string_view option(argv[index]);
    constexpr std::string_view second_prefix = "--second-image=";
    if (option.starts_with(second_prefix)) {
      if (images.size() != 1 || option.size() == second_prefix.size() || native_application)
        return 3;
      images.push_back(option.substr(second_prefix.size()));
      continue;
    }
    const std::string_view prefix =
        option.starts_with("--application-workgroups=") ? "--application-workgroups="
        : option.starts_with("--trace-workgroups=")     ? "--trace-workgroups="
                                                        : "--cluster-size=";
    if (!option.starts_with(prefix))
      return 3;
    const std::string_view count = option.substr(prefix.size());
    uint32_t parsed_count = 0;
    const std::from_chars_result parsed =
        std::from_chars(count.data(), count.data() + count.size(), parsed_count);
    if (parsed.ec != std::errc{} || parsed.ptr != count.data() + count.size())
      return 3;
    if (prefix == "--cluster-size=") {
      if (cluster_size != 0 || (parsed_count != 1 && parsed_count != 2))
        return 3;
      cluster_size = parsed_count;
    } else {
      if (application_completion || native_application || parsed_count == 0 || parsed_count > 512)
        return 3;
      application_completion = true;
      workgroups = parsed_count;
    }
  }
  if (cluster_size && (!application_completion || workgroups % cluster_size != 0))
    return 3;
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    return 20;
  }
  // Agent enumeration requires a live HSA runtime.
  const Agents initialized = agents("gfx1250");
  if (!initialized.cpu.handle || !initialized.gpu.handle) {
    return 77;
  }
  void *plugin = nullptr;
  rj_consan_finish_fn finish = nullptr;
  rj_consan_drain_fn drain = nullptr;
  rj_consan_statistics_fn statistics = nullptr;
  if (instrumented || native_application) {
    const char *path = std::getenv("RJ_CONSAN_PLUGIN_PATH");
    if (path) {
      plugin = ::dlopen(path, RTLD_NOW | RTLD_NOLOAD);
    }
    if (plugin) {
      finish = reinterpret_cast<rj_consan_finish_fn>(::dlsym(plugin, "rj_consan_finish"));
      drain = reinterpret_cast<rj_consan_drain_fn>(::dlsym(plugin, "rj_consan_drain"));
      statistics =
          reinterpret_cast<rj_consan_statistics_fn>(::dlsym(plugin, "rj_consan_get_statistics"));
    }
    if (!finish || !drain || !statistics || drain() != 0) {
      return 3;
    }
  }
  const auto require = [](hsa_status_t status, const char *operation) {
    if (status == HSA_STATUS_SUCCESS) {
      return true;
    }
    std::cerr << operation << " failed: " << uint32_t(status) << '\n';
    return false;
  };
  uint32_t initializations = 1;
  uint32_t shutdowns = 0;
  if (native_application) {
    if (!require(hsa_init(), "second HSA reference"))
      return 20;
    ++initializations;
  }
  std::array<hsa_executable_t, 2> executables = {};
  std::array<hsa_code_object_reader_t, 2> readers = {};
  hsa_queue_t *queue = nullptr;
  std::vector<hsa_queue_t *> native_queues;
  hsa_signal_t signal = {};
  void *arguments = nullptr;
  void *output = nullptr;
  int result = 0;
  uint32_t loads = 0;
  uint32_t completed = 0;
  uint32_t queues_created = 0, queues_destroyed = 0;
  const std::array<Entry, 3> original_entries = {
      Entry{.name = "mk_a.kd", .argument_bytes = 16, .group_bytes = 8},
      Entry{.name = "mk_b.kd", .argument_bytes = 32, .group_bytes = 256},
      Entry{.name = "mk_c.kd", .argument_bytes = 24, .group_bytes = 64}};
  std::array<std::array<Entry, 3>, 2> image_entries = {original_entries, original_entries};
  if (native_application) {
    constexpr uint32_t kNativeQueues = 17;
    for (uint32_t index = 0; index < kNativeQueues; ++index) {
      hsa_amd_queue_create_desc_t descriptor = {};
      descriptor.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
      descriptor.engine_type = HSA_AMD_QUEUE_ENGINE_COMPUTE;
      descriptor.queue_size_bytes = 128 * sizeof(hsa_kernel_dispatch_packet_t);
      descriptor.priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
      descriptor.engine.compute.type = HSA_QUEUE_TYPE_SINGLE;
      const hsa_status_t status = hsa_amd_queue_create(initialized.gpu, &descriptor, 1);
      const bool distinct =
          descriptor.queue != nullptr && std::find(native_queues.begin(), native_queues.end(),
                                                   descriptor.queue) == native_queues.end();
      if (distinct)
        native_queues.push_back(descriptor.queue);
      if (!require(status, "AMD queue before image load") || !distinct) {
        result = 20;
        break;
      }
      ++queues_created;
      queue = descriptor.queue;
    }
  }
  for (size_t image = 0; image < images.size() && result == 0; ++image) {
    const int descriptor = result == 0 ? ::open(images[image].data(), O_RDONLY | O_CLOEXEC) : -1;
    if (result == 0 &&
        (descriptor < 0 ||
         !require(hsa_code_object_reader_create_from_file(descriptor, &readers[image]), "reader") ||
         !require(hsa_executable_create_alt(HSA_PROFILE_FULL,
                                            HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                            &executables[image]),
                  "executable") ||
         !require(hsa_executable_load_agent_code_object(executables[image], initialized.gpu,
                                                        readers[image], nullptr, nullptr),
                  "whole-image load")))
      result = 20;
    else if (result == 0)
      ++loads;
    if (descriptor >= 0)
      ::close(descriptor);
    if (result == 0 && !require(hsa_executable_freeze(executables[image], nullptr), "freeze"))
      result = 20;
    for (Entry &entry : image_entries[image]) {
      if (result != 0)
        break;
      hsa_executable_symbol_t symbol = {};
      uint32_t argument_bytes = 0, group_bytes = 0, private_bytes = 0;
      if (!require(hsa_executable_get_symbol_by_name(executables[image], entry.name,
                                                     &initialized.gpu, &symbol),
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
    if (result == 0 && (image_entries[image][0].kernel == image_entries[image][1].kernel ||
                        image_entries[image][0].kernel == image_entries[image][2].kernel ||
                        image_entries[image][1].kernel == image_entries[image][2].kernel))
      result = 20;
  }
  const hsa_amd_memory_pool_t pool = fine_pool(initialized.cpu);
  if (result == 0 &&
      (!pool.handle ||
       !require(hsa_amd_memory_pool_allocate(pool, 32, 0, &arguments), "arguments") ||
       !require(hsa_amd_memory_pool_allocate(pool, 16, 0, &output), "output") ||
       !require(hsa_amd_agents_allow_access(1, &initialized.gpu, nullptr, arguments),
                "args access") ||
       !require(hsa_amd_agents_allow_access(1, &initialized.gpu, nullptr, output), "out access") ||
       (!native_application &&
        !require(hsa_queue_create(initialized.gpu, 128, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0,
                                  0, &queue),
                 "queue")))) {
    result = 20;
  }
  if (!native_application && queue)
    ++queues_created;
  if (result == 0 && native_application) {
    if (!require(hsa_shut_down(), "intermediate HSA shutdown")) {
      result = 22;
    } else {
      ++shutdowns;
    }
  }
  const std::array<uint32_t, 4> order = {0, 1, 2, 0};
  const std::array<uint32_t, 4> seeds = {7, 23, 55, 39};
  constexpr uint32_t kGuard = 0xdeadbeef;
  for (uint32_t ordinal = 0; result == 0 && ordinal < order.size() * images.size(); ++ordinal) {
    const size_t image = ordinal % images.size();
    const size_t invocation = ordinal / images.size();
    const uint32_t index = order[invocation];
    const Entry &entry = image_entries[image][index];
    const uint32_t seed = seeds[invocation];
    const uint32_t bias = 100;
    if (!native_application &&
        !require(hsa_signal_create(1, 0, nullptr, &signal), "dedicated completion")) {
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
      if (index == 1) {
        std::memcpy(static_cast<uint8_t *>(arguments) + 24, &bias, sizeof(bias));
      }
    }
    std::fill_n(static_cast<uint32_t *>(output), 4, kGuard);
    const uint64_t write_index = hsa_queue_add_write_index_relaxed(queue, 1);
    auto *packet = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
                   (write_index & (queue->size - 1));
    std::memset(packet, 0, sizeof(*packet));
    packet->setup = 1;
    packet->workgroup_size_x = 64;
    packet->grid_size_x = 64 * workgroups;
    packet->workgroup_size_y = packet->workgroup_size_z = 1;
    packet->grid_size_y = packet->grid_size_z = 1;
    packet->group_segment_size = entry.group_bytes;
    packet->kernel_object = entry.kernel;
    packet->kernarg_address = arguments;
    packet->completion_signal = application_completion ? hsa_signal_t{} : signal;
    uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                      (1u << HSA_PACKET_HEADER_BARRIER) |
                      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    if (cluster_size != 0) {
      hsa_amd_ext_kernel_dispatch_packet_t extended = {};
      extended.amd_format = HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH;
      extended.setup = packet->setup;
      extended.workgroup_size_x = packet->workgroup_size_x;
      extended.workgroup_size_y = extended.workgroup_size_z = 1;
      extended.cluster_count_x = workgroups / cluster_size;
      extended.cluster_count_y = extended.cluster_count_z = 1;
      extended.cluster_size_x = cluster_size;
      extended.cluster_size_y = extended.cluster_size_z = 1;
      extended.group_segment_size = packet->group_segment_size;
      extended.kernel_object = packet->kernel_object;
      extended.kernarg_address = packet->kernarg_address;
      extended.completion_signal = packet->completion_signal;
      std::memcpy(packet, &extended, sizeof(extended));
      header = (header & ~uint16_t{0xff}) | HSA_PACKET_TYPE_VENDOR_SPECIFIC;
    }
    __atomic_store_n(&packet->header, header, __ATOMIC_RELEASE);
    uint64_t published_index = write_index;
    if (application_completion) {
      // The fixture application waits through its own following barrier. ConSan
      // sees the original null-completion dispatch and adds no dependency packet.
      published_index = hsa_queue_add_write_index_relaxed(queue, 1);
      auto *barrier = static_cast<hsa_barrier_and_packet_t *>(queue->base_address) +
                      (published_index & (queue->size - 1));
      std::memset(barrier, 0, sizeof(*barrier));
      barrier->completion_signal = signal;
      const uint16_t barrier_header = (header & ~uint16_t{0xff}) | HSA_PACKET_TYPE_BARRIER_AND;
      __atomic_store_n(&barrier->header, barrier_header, __ATOMIC_RELEASE);
    }
    hsa_signal_store_screlease(queue->doorbell_signal, published_index);
    hsa_signal_value_t value = native_application ? 0 : 1;
    if (native_application && drain() != 0) {
      std::cerr << "native drain failed; retaining live dispatch resources\n";
      ::_exit(22);
    }
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (value == 1 && std::chrono::steady_clock::now() < deadline) {
      value = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, 1000000,
                                        HSA_WAIT_STATE_BLOCKED);
    }
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
    if (actual != expected) {
      result = 23;
    }
    std::cout << "{";
    if (images.size() > 1)
      std::cout << "\"image\":" << image << ",";
    std::cout << "\"ordinal\":" << ordinal + 1 << ",\"entry\":\"" << entry.name
              << "\",\"native\":" << (native_application || index == 2 ? "true" : "false")
              << ",\"kernarg_bytes\":" << entry.argument_bytes
              << ",\"kernarg_alignment\":" << entry.alignment
              << ",\"group_bytes\":" << entry.group_bytes << ",\"completion\":" << value
              << ",\"output\":[" << actual[0] << ',' << actual[1] << ',' << actual[2] << ','
              << actual[3] << "]}\n";
    if (drain && !native_application && drain() != 0) {
      result = 22;
    }
    if (signal.handle && !require(hsa_signal_destroy(signal), "completion release")) {
      result = 22;
    }
    signal = {};
  }
  const int finish_status = finish && !native_application ? finish() : 0;
  if (finish_status != 0 && result == 0) {
    result = 22;
  }
  const rj_consan_statistics stats = statistics ? statistics() : rj_consan_statistics{};
  if (signal.handle && !require(hsa_signal_destroy(signal), "signal cleanup") && result == 0) {
    result = 22;
  }
  if (native_application) {
    for (hsa_queue_t *created : native_queues) {
      if (require(hsa_queue_destroy(created), "native queue cleanup"))
        ++queues_destroyed;
      else if (result == 0)
        result = 22;
    }
  } else if (queue) {
    if (require(hsa_queue_destroy(queue), "queue cleanup"))
      ++queues_destroyed;
    else if (result == 0)
      result = 22;
  }
  for (size_t image = 0; image < images.size(); ++image) {
    if (readers[image].handle &&
        !require(hsa_code_object_reader_destroy(readers[image]), "reader cleanup") && result == 0)
      result = 22;
    if (executables[image].handle &&
        !require(hsa_executable_destroy(executables[image]), "executable cleanup") && result == 0)
      result = 22;
  }
  if (arguments && !require(hsa_amd_memory_pool_free(arguments), "arguments cleanup") &&
      result == 0) {
    result = 22;
  }
  if (output && !require(hsa_amd_memory_pool_free(output), "output cleanup") && result == 0) {
    result = 22;
  }
  while (shutdowns < initializations) {
    if (!require(hsa_shut_down(), "shutdown")) {
      if (result == 0)
        result = 22;
      break;
    }
    ++shutdowns;
  }
  if (plugin)
    ::dlclose(plugin);
  std::cout << "{\"app_status\":" << result << ",\"finish_status\":";
  if (native_application)
    std::cout << "null";
  else
    std::cout << finish_status;
  std::cout << ",\"initializations\":" << initializations << ",\"shutdowns\":" << shutdowns
            << ",\"queues_created\":" << queues_created
            << ",\"queues_destroyed\":" << queues_destroyed << ",\"loads\":" << loads
            << ",\"completed\":" << completed << ",\"submitted\":" << stats.submitted
            << ",\"forwarded\":" << stats.forwarded << ",\"copied\":" << stats.copied
            << ",\"written\":" << stats.written << ",\"reused\":" << stats.reused
            << ",\"native_submitted\":" << stats.native_submitted
            << ",\"native_completed\":" << stats.native_completed << "}\n";
  return result;
}
} // namespace

int main(int argc, char **argv) { return run(argc, argv); }
