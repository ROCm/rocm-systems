// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/target.h"
#include "rocjitsu/hooks/consan/api.h"
#include "rocjitsu/hooks/consan/testing.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct Agents {
  hsa_agent_t cpu{}, gpu{};
  std::string_view target;
};

Agents agents(std::string_view target) {
  Agents found{{}, {}, target};
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) {
        auto &result = *static_cast<Agents *>(data);
        hsa_device_type_t type{};
        if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
          return HSA_STATUS_SUCCESS;
        if (type == HSA_DEVICE_TYPE_CPU)
          result.cpu = agent;
        if (type == HSA_DEVICE_TYPE_GPU) {
          hsa_isa_t isa{};
          uint32_t bytes = 0;
          if (hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa) == HSA_STATUS_SUCCESS &&
              hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &bytes) == HSA_STATUS_SUCCESS &&
              bytes > 0 && bytes <= 256) {
            std::array<char, 256> name{};
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
  hsa_amd_memory_pool_t found{};
  hsa_amd_agent_iterate_memory_pools(
      cpu,
      [](hsa_amd_memory_pool_t pool, void *data) {
        hsa_amd_segment_t segment{};
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

int run(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: consan_hsa_runner ORIGINAL ENTRY LAUNCHES defined|observe [--baseline] "
                 "[--bad-geometry] [--profiling] [--reuse-signal] [--target GPU] [--wave-size N] "
                 "[--threads N] [--test-statistics] [--batch-barrier-and|--batch-barrier-or] "
                 "[--separate-barrier]\n";
    return 3;
  }
  uint32_t launches = 0;
  const std::from_chars_result parsed =
      std::from_chars(argv[3], argv[3] + std::strlen(argv[3]), launches);
  if (parsed.ec != std::errc{} || *parsed.ptr != '\0' || launches == 0 || launches > 16)
    return 3;
  const bool defined = std::string_view(argv[4]) == "defined";
  if (!defined && std::string_view(argv[4]) != "observe")
    return 3;
  std::string_view target = "gfx1100";
  uint32_t threads = 64, wave_size = 32;
  bool baseline = false, bad_geometry = false, profiling = false, reuse_signal = false;
  bool test_statistics = false, separate_barrier = false;
  uint16_t barrier_type = 0;
  for (int index = 5; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "--baseline")
      baseline = true;
    else if (option == "--bad-geometry")
      bad_geometry = true;
    else if (option == "--profiling")
      profiling = true;
    else if (option == "--reuse-signal")
      reuse_signal = true;
    else if (option == "--test-statistics")
      test_statistics = true;
    else if (option == "--batch-barrier-and" || option == "--batch-barrier-or")
      barrier_type = option == "--batch-barrier-and" ? HSA_PACKET_TYPE_BARRIER_AND
                                                     : HSA_PACKET_TYPE_BARRIER_OR;
    else if (option == "--separate-barrier")
      separate_barrier = true;
    else if (option == "--target" && index + 1 < argc)
      target = argv[++index];
    else if ((option == "--threads" || option == "--wave-size") && index + 1 < argc) {
      const char *value = argv[++index];
      uint32_t &number = option == "--threads" ? threads : wave_size;
      const std::from_chars_result conversion =
          std::from_chars(value, value + std::strlen(value), number);
      if (conversion.ec != std::errc{} || *conversion.ptr != '\0')
        return 3;
    } else
      return 3;
  }
  const rocjitsu::consan::TargetProfile *profile = rocjitsu::consan::target_profile(target);
  if (profile == nullptr || !rocjitsu::consan::supports_wave(*profile, wave_size) || threads == 0 ||
      threads > 1024)
    return 3;
  const uint32_t wave_count = (threads + wave_size - 1) / wave_size;
  const size_t output_bytes = std::max(wave_count, 2u) * sizeof(uint32_t);
  if (baseline && (bad_geometry || profiling || reuse_signal || test_statistics || barrier_type))
    return 3;
  if (separate_barrier && (!barrier_type || !bad_geometry))
    return 3;
  if (hsa_init() != HSA_STATUS_SUCCESS)
    return 77;
  const Agents hardware = agents(target);
  void *plugin = nullptr;
  rj_consan_finish_fn finish = nullptr;
  rj_consan_drain_fn drain = nullptr;
  rj_consan_statistics_fn statistics = nullptr;
  if (!baseline) {
    const char *path = std::getenv("CONSAN_PLUGIN_PATH");
    if (path)
      plugin = ::dlopen(path, RTLD_NOW | RTLD_NOLOAD);
    if (plugin) {
      finish = reinterpret_cast<rj_consan_finish_fn>(::dlsym(plugin, "rj_consan_finish"));
      drain = reinterpret_cast<rj_consan_drain_fn>(::dlsym(plugin, "rj_consan_drain"));
      if (test_statistics)
        statistics =
            reinterpret_cast<rj_consan_statistics_fn>(::dlsym(plugin, "rj_consan_get_statistics"));
    }
    if (!finish || !drain || drain() != 0) {
      std::cerr << "required ConSan plugin was not active before original load\n";
      return 3;
    }
    if (test_statistics && !statistics) {
      std::cerr << "required ConSan test statistics interface is unavailable\n";
      return 3;
    }
  }
  if (!hardware.cpu.handle || !hardware.gpu.handle) {
    if (finish)
      (void)finish();
    (void)hsa_shut_down();
    if (plugin)
      ::dlclose(plugin);
    return 77;
  }

  hsa_executable_t executable{};
  hsa_code_object_reader_t reader{};
  hsa_queue_t *queue = nullptr;
  hsa_signal_t signal{};
  hsa_signal_t barrier_signal{};
  hsa_signal_value_t barrier_completion = 1;
  void *arguments = nullptr, *output = nullptr;
  int result = 0;
  uint32_t completed = 0;
  int profiling_status = 0;
  std::vector<uint32_t> last_output(std::max(wave_count, 2u));
  const auto require = [&](hsa_status_t status, const char *operation) {
    if (status == HSA_STATUS_SUCCESS)
      return true;
    std::cerr << operation << " failed with HSA status " << uint32_t(status) << '\n';
    return false;
  };
  const int descriptor = ::open(argv[1], O_RDONLY | O_CLOEXEC);
  if (descriptor < 0 ||
      !require(hsa_code_object_reader_create_from_file(descriptor, &reader), "original reader") ||
      !require(hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                         nullptr, &executable),
               "executable create") ||
      !require(
          hsa_executable_load_agent_code_object(executable, hardware.gpu, reader, nullptr, nullptr),
          "original load through plugin") ||
      !require(hsa_executable_freeze(executable, nullptr), "freeze/binding"))
    result = 20;
  if (descriptor >= 0 && ::close(descriptor) != 0 && result == 0)
    result = 20;
  uint64_t kernel = 0;
  uint32_t kernarg_bytes = 0, group_bytes = 0;
  hsa_executable_symbol_t symbol{};
  const std::string name = std::string(argv[2]) + ".kd";
  if (result == 0 &&
      (!require(hsa_executable_get_symbol_by_name(executable, name.c_str(), &hardware.gpu, &symbol),
                "kernel symbol") ||
       !require(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                               &kernel),
                "kernel object") ||
       !require(hsa_executable_symbol_get_info(
                    symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_bytes),
                "original argument ABI") ||
       !require(hsa_executable_symbol_get_info(
                    symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &group_bytes),
                "LDS extent") ||
       kernarg_bytes != 16 || kernel == 0))
    result = 20;
  const hsa_amd_memory_pool_t pool = fine_pool(hardware.cpu);
  if (result == 0 &&
      (!pool.handle ||
       !require(hsa_amd_memory_pool_allocate(pool, 16, 0, &arguments), "original arguments") ||
       !require(hsa_amd_memory_pool_allocate(pool, output_bytes, 0, &output),
                "output allocation") ||
       !require(hsa_amd_agents_allow_access(1, &hardware.gpu, nullptr, arguments),
                "argument access") ||
       !require(hsa_amd_agents_allow_access(1, &hardware.gpu, nullptr, output), "output access") ||
       !require(hsa_queue_create(hardware.gpu, 128, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0,
                                 &queue),
                "controlled queue")))
    result = 20;
  if (result == 0 && profiling) {
    profiling_status = hsa_amd_profiling_set_profiler_enabled(queue, 1);
    if (profiling_status == HSA_STATUS_SUCCESS)
      result = 21;
  }
  if (const char *maps = std::getenv("CONSAN_PROCESS_MAPS")) {
    std::ifstream source("/proc/self/maps");
    std::ofstream destination(maps);
    destination << source.rdbuf();
    destination.close();
    if (destination.fail() && result == 0)
      result = 22;
  }
  for (uint32_t index = 0; result == 0 && index < launches; ++index) {
    if (signal.handle == 0) {
      if (!require(hsa_signal_create(1, 0, nullptr, &signal), "dedicated signal")) {
        result = 21;
        break;
      }
    } else
      hsa_signal_store_screlease(signal, 1);
    const uint32_t input = 7 + 16 * index;
    std::memset(arguments, 0, 16);
    std::memcpy(arguments, &input, 4);
    std::memcpy(static_cast<uint8_t *>(arguments) + 8, &output, 8);
    std::memset(output, 0, output_bytes);
    if (barrier_type &&
        !require(hsa_signal_create(1, 0, nullptr, &barrier_signal), "barrier completion")) {
      result = 21;
      break;
    }
    const uint64_t packet_count = barrier_type ? 2 : 1;
    const uint64_t write_index = hsa_queue_add_write_index_relaxed(queue, packet_count);
    auto *packet = static_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address) +
                   (write_index & (queue->size - 1));
    std::memset(packet, 0, sizeof(*packet));
    packet->setup = 1;
    packet->workgroup_size_x = static_cast<uint16_t>(threads);
    packet->workgroup_size_y = packet->workgroup_size_z = 1;
    packet->grid_size_x = bad_geometry ? 2 * threads : threads;
    packet->grid_size_y = packet->grid_size_z = 1;
    packet->group_segment_size = group_bytes;
    packet->kernel_object = kernel;
    packet->kernarg_address = arguments;
    packet->completion_signal = signal;
    const uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                            (1u << HSA_PACKET_HEADER_BARRIER) |
                            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    if (barrier_type) {
      auto *barrier = reinterpret_cast<hsa_barrier_and_packet_t *>(queue->base_address) +
                      ((write_index + 1) & (queue->size - 1));
      std::memset(barrier, 0, sizeof(*barrier));
      barrier->dep_signal[0] = signal;
      barrier->completion_signal = barrier_signal;
      barrier->header =
          separate_barrier
              ? uint16_t{HSA_PACKET_TYPE_INVALID}
              : barrier_type | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    }
    // Keep the second header invalid for the separate-callback control.
    __atomic_store_n(reinterpret_cast<uint16_t *>(packet), header, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(queue->doorbell_signal,
                               separate_barrier ? write_index : write_index + packet_count - 1);
    if (separate_barrier) {
      if (hsa_signal_load_scacquire(signal) != -1) {
        std::cerr << "separate dispatch did not report rejection; retaining live HSA resources\n";
        ::_exit(24);
      }
      auto *barrier = reinterpret_cast<hsa_barrier_and_packet_t *>(queue->base_address) +
                      ((write_index + 1) & (queue->size - 1));
      const uint16_t barrier_header =
          barrier_type | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
      __atomic_store_n(&barrier->header, barrier_header, __ATOMIC_RELEASE);
      hsa_signal_store_screlease(queue->doorbell_signal, write_index + 1);
    }
    if (barrier_type) {
      const std::chrono::steady_clock::time_point deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (barrier_completion == 1 && std::chrono::steady_clock::now() < deadline)
        barrier_completion = hsa_signal_wait_scacquire(barrier_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                                       1000000, HSA_WAIT_STATE_BLOCKED);
      if (barrier_completion != -1) {
        std::cerr << "batch barrier did not report failure; retaining live HSA resources\n";
        ::_exit(24);
      }
    }
    hsa_signal_value_t value = 1;
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (value == 1 && std::chrono::steady_clock::now() < deadline)
      value = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, 1000000,
                                        HSA_WAIT_STATE_BLOCKED);
    if (value == 1) {
      std::cerr << "dispatch timed out without completion proof; retaining live GPU allocations\n";
      ::_exit(24);
    }
    if (value != 0) {
      result = 21;
      break;
    }
    ++completed;
    std::memcpy(last_output.data(), output, output_bytes);
    std::cout << "launch=" << index + 1 << " input=" << input << " output=" << last_output[0] << ','
              << last_output[1] << '\n';
    if (defined) {
      for (uint32_t wave = 0; wave < wave_count; ++wave)
        if (last_output[wave] != input + wave)
          result = 23;
    }
    if (drain && drain() != 0)
      result = 22;
    if (!reuse_signal) {
      if (!require(hsa_signal_destroy(signal), "completed signal destroy") && result == 0)
        result = 22;
      signal = {};
    }
  }
  // This explicit finish is before signal/queue/executable teardown and HSA
  // shutdown, so GPU and capture last users are still inside HSA's lifetime.
  const int finish_status = finish ? finish() : 0;
  if (finish_status != 0 && result == 0)
    result = 22;
  const rj_consan_statistics stats = statistics ? statistics() : rj_consan_statistics{};
  if (signal.handle && !require(hsa_signal_destroy(signal), "signal cleanup") && result == 0)
    result = 22;
  if (barrier_signal.handle &&
      !require(hsa_signal_destroy(barrier_signal), "barrier signal cleanup") && result == 0)
    result = 22;
  if (queue && !require(hsa_queue_destroy(queue), "queue cleanup") && result == 0)
    result = 22;
  if (reader.handle && !require(hsa_code_object_reader_destroy(reader), "reader cleanup") &&
      result == 0)
    result = 22;
  if (executable.handle && !require(hsa_executable_destroy(executable), "executable cleanup") &&
      result == 0)
    result = 22;
  if (arguments && !require(hsa_amd_memory_pool_free(arguments), "argument cleanup") && result == 0)
    result = 22;
  if (output && !require(hsa_amd_memory_pool_free(output), "output cleanup") && result == 0)
    result = 22;
  if (!require(hsa_shut_down(), "HSA shutdown") && result == 0)
    result = 22;
  if (plugin)
    ::dlclose(plugin);
  std::cout << "{\"app_status\":" << result << ",\"finish_status\":" << finish_status
            << ",\"completed\":" << completed;
  if (barrier_type)
    std::cout << ",\"barrier_completion\":" << barrier_completion;
  if (statistics)
    std::cout << ",\"submitted\":" << stats.submitted << ",\"forwarded\":" << stats.forwarded
              << ",\"copied\":" << stats.copied << ",\"written\":" << stats.written
              << ",\"reused\":" << stats.reused;
  std::cout << ",\"profiling_status\":" << profiling_status << "}\n";
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "HSA runner tool failure: " << error.what() << '\n';
    // An exception is not completion; process teardown reclaims its HSA
    // resources rather than freeing allocations a GPU may still be using.
    ::_exit(3);
  }
}
