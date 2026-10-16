// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Shared scaffolding for the HSA vector_scalar_add benchmarks.
//
// Both benchmarks dispatch the same design through the same API and differ only in which hsaco
// they load -- PDI+insts or full-ELF -- so everything up to that choice lives here. They used to
// carry a copy each, and the copies drifted: output verification, teardown ordering and failure
// handling were all fixed in one and wrong in the other.
//
// Header-only and gtest-free, so it can be included from a google-benchmark translation unit.

#ifndef ROCRTST_SUITES_AIE_PERFORMANCE_AIE_HSA_HARNESS_H_
#define ROCRTST_SUITES_AIE_PERFORMANCE_AIE_HSA_HARNESS_H_

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_amd_aie.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

namespace aie_bench {

/// Elements the vector_scalar_add design operates on, and the buffer size that implies.
constexpr std::size_t N = 1024;
constexpr std::size_t DATA_SIZE = N * sizeof(std::uint32_t);

/// Number of kernargs, followed by the same number of sizes.
constexpr std::size_t NUM_KERNARGS = 2;
constexpr std::size_t KERNARG_ENTRIES = 2 * NUM_KERNARGS;

/// A pool allocation handed out in bump-pointer fashion, so a benchmark can price argument
/// preparation without pricing a pool allocation per dispatch.
class HsaBumpAllocator {
  void* base_ = nullptr;
  std::size_t size_ = 0;
  std::size_t offset_ = 0;

 public:
  HsaBumpAllocator(hsa_amd_memory_pool_t pool, std::size_t size) : size_(size) {
    if (hsa_amd_memory_pool_allocate(pool, size, 0, &base_) != HSA_STATUS_SUCCESS) {
      throw std::bad_alloc();
    }
  }

  ~HsaBumpAllocator() {
    if (base_) hsa_amd_memory_pool_free(base_);
  }

  HsaBumpAllocator(const HsaBumpAllocator&) = delete;
  HsaBumpAllocator& operator=(const HsaBumpAllocator&) = delete;

  template <typename T> T* allocate(std::size_t count, std::size_t alignment = alignof(T)) {
    std::size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
    std::size_t new_offset = aligned + count * sizeof(T);
    if (new_offset > size_) throw std::bad_alloc();
    offset_ = new_offset;
    return reinterpret_cast<T*>(static_cast<char*>(base_) + aligned);
  }

  void reset() { offset_ = 0; }
};

inline hsa_status_t find_aie_agent(hsa_agent_t agent, void* data) {
  hsa_device_type_t type{};
  if (auto s = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type); s != HSA_STATUS_SUCCESS) {
    return s;
  }
  if (type == HSA_DEVICE_TYPE_AIE) {
    *static_cast<hsa_agent_t*>(data) = agent;
    return HSA_STATUS_INFO_BREAK;
  }
  return HSA_STATUS_SUCCESS;
}

/// Finds a global memory pool matching the given flags and allocatability.
struct find_pool_data {
  hsa_amd_memory_pool_global_flag_t expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  bool expected_allocatable = true;
  hsa_amd_memory_pool_t pool{};
};

inline hsa_status_t find_memory_pool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment{};
  if (auto s = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
      s != HSA_STATUS_SUCCESS || segment != HSA_AMD_SEGMENT_GLOBAL) {
    return s;
  }

  hsa_amd_memory_pool_global_flag_t flags{};
  if (auto s = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
      s != HSA_STATUS_SUCCESS) {
    return s;
  }

  auto& d = *static_cast<find_pool_data*>(data);
  if ((flags & d.expected_flags) == 0) return HSA_STATUS_SUCCESS;

  std::size_t alloc_rec_granule = 0;
  if (auto s = hsa_amd_memory_pool_get_info(
          pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, &alloc_rec_granule);
      s != HSA_STATUS_SUCCESS) {
    return s;
  }

  if (d.expected_allocatable != (alloc_rec_granule != 0)) return HSA_STATUS_SUCCESS;

  d.pool = pool;
  return HSA_STATUS_INFO_BREAK;
}

/// Reads a file whole. Throws rather than returning an empty vector, so a missing file and an
/// empty one are distinguishable, and rejects a short read instead of leaving a zero-padded tail
/// for the code object reader to trip over.
inline std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Cannot open " + path.string());
  const auto size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<std::uint8_t> data(size);
  f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
  if (static_cast<std::size_t>(f.gcount()) != size) {
    throw std::runtime_error("Short read loading " + path.string());
  }
  return data;
}

/// Writes one packet into the next queue slot and fills its kernargs. Does not ring the doorbell,
/// so a caller can batch several packets into one command chain.
inline std::uint64_t dispatch_packet(std::uint64_t kernel_object, void* input, void* output,
                                     std::uint64_t* kernargs, hsa_queue_t* q) {
  kernargs[0] = reinterpret_cast<std::uint64_t>(input);
  kernargs[1] = reinterpret_cast<std::uint64_t>(output);
  kernargs[2] = DATA_SIZE;  // input size
  kernargs[3] = DATA_SIZE;  // output size

  auto* queue = static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(q->base_address);
  const auto mask = q->size - 1;

  const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
  while (wr_idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {
    // spin until a slot frees up
  }

  // Built here and stored into the ring in one assignment: writing the READY header into the
  // live slot first would let a processor still draining the previous doorbell see a ready packet
  // whose kernel object has not been written yet.
  hsa_amd_aie_kernel_dispatch_packet_t pkt{};
  pkt.header = (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  pkt.opcode = HSA_AMD_AIE_PACKET_OPCODE_KMQ;
  pkt.count = 24;
  pkt.completion_signal.handle = 0;
  pkt.kernel_object_low = kernel_object & 0xFFFFFFFF;
  pkt.kernel_object_high = kernel_object >> 32;
  pkt.num_kernargs = NUM_KERNARGS;
  pkt.kernarg_address = kernargs;

  queue[wr_idx & mask] = pkt;
  return wr_idx;
}

/// Everything a benchmark needs: an initialized runtime, an agent, a queue, a loaded hsaco and
/// per-dispatch buffers. Set up outside the timed loop, so the loop measures only dispatch and
/// wait, and released in reverse order by the destructor -- notably freeing pool memory before
/// hsa_shut_down, which a plain scope-exit ordering gets wrong.
///
/// Throws on any failure so a benchmark body can report it through SkipWithError; google-benchmark
/// does not catch, so a body that lets one escape terminates the whole run.
class HsaHarness {
 public:
  /// @param num_dispatches per-dispatch input/output buffers to allocate
  /// @param hsaco_path hsaco to load
  /// @param kernel_name symbol to take the kernel object from
  /// @param required_arch agent name this hsaco needs, or nullptr when any AIE agent will do.
  /// Full-ELF dispatch is aie2p only; PDI+insts runs on either.
  HsaHarness(std::int32_t num_dispatches, const std::filesystem::path& hsaco_path,
             const char* kernel_name, const char* required_arch = nullptr) {
    try {
      Acquire(num_dispatches, hsaco_path, kernel_name, required_arch);
    } catch (...) {
      // A throwing constructor does not run its own destructor, so anything already acquired
      // would leak -- including the hsa_init refcount, once per benchmark case.
      Release();
      throw;
    }
  }

  ~HsaHarness() { Release(); }

  HsaHarness(const HsaHarness&) = delete;
  HsaHarness& operator=(const HsaHarness&) = delete;

  /// Confirms the kernel ran; otherwise the numbers time a no-op dispatch. Input is [1..N], so
  /// element i must come back as i + 2.
  bool verify() const {
    for (auto* out : outputs) {
      for (std::size_t i = 0; i < N; ++i) {
        if (out[i] != i + 2) return false;
      }
    }
    return true;
  }

  hsa_agent_t agent{};
  hsa_amd_memory_pool_t data_pool{};
  hsa_amd_memory_pool_t kernarg_pool{};
  hsa_queue_t* queue = nullptr;
  // The application does not parse the payload, place blobs, or patch anything: it loads an hsaco
  // and holds the kernel object handle the loader published. The runtime does the rest, which is
  // what these benchmarks are here to price.
  hsa_executable_t executable{};
  hsa_code_object_reader_t reader{};
  std::uint64_t kernel_object = 0;
  std::vector<std::uint32_t*> inputs;
  std::vector<std::uint32_t*> outputs;

 private:
  void Acquire(std::int32_t num_dispatches, const std::filesystem::path& hsaco_path,
               const char* kernel_name, const char* required_arch) {
    if (hsa_init() != HSA_STATUS_SUCCESS) throw std::runtime_error("hsa_init failed");
    initialized_ = true;

    if (hsa_iterate_agents(find_aie_agent, &agent) != HSA_STATUS_INFO_BREAK) {
      throw std::runtime_error("No AIE agent found");
    }

    if (required_arch != nullptr) {
      char agent_name[64] = {};
      if (hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name) != HSA_STATUS_SUCCESS ||
          std::strcmp(agent_name, required_arch) != 0) {
        throw std::runtime_error(std::string("this benchmark needs a ") + required_arch +
                                 " agent, found '" + agent_name + "'");
      }
    }

    // data_memory: coarse-grained, allocatable (for tensor data)
    find_pool_data data_pool_data{};
    data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    data_pool_data.expected_allocatable = true;
    if (hsa_amd_agent_iterate_memory_pools(agent, find_memory_pool, &data_pool_data) !=
        HSA_STATUS_INFO_BREAK) {
      throw std::runtime_error("No allocatable coarse-grained pool found");
    }
    data_pool = data_pool_data.pool;

    // kernarg_memory: KERNARG_INIT, allocatable; falls back to the data pool
    find_pool_data kernarg_pool_data{};
    kernarg_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
    kernarg_pool_data.expected_allocatable = true;
    if (hsa_amd_agent_iterate_memory_pools(agent, find_memory_pool, &kernarg_pool_data) !=
        HSA_STATUS_INFO_BREAK) {
      kernarg_pool_data.pool = data_pool;
    }
    kernarg_pool = kernarg_pool_data.pool;

    std::uint32_t min_queue_size = 0;
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size) !=
        HSA_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to get min queue size");
    }
    if (hsa_queue_create(agent, min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0,
                         &queue) != HSA_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create HSA queue");
    }

    LoadHsaco(hsaco_path, kernel_name);

    inputs.resize(num_dispatches, nullptr);
    outputs.resize(num_dispatches, nullptr);
    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      if (hsa_amd_memory_pool_allocate(data_pool, DATA_SIZE, 0,
                                       reinterpret_cast<void**>(&inputs[i])) !=
              HSA_STATUS_SUCCESS ||
          hsa_amd_memory_pool_allocate(data_pool, DATA_SIZE, 0,
                                       reinterpret_cast<void**>(&outputs[i])) !=
              HSA_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to allocate I/O buffers");
      }
      std::iota(inputs[i], inputs[i] + N, 1);
      std::fill_n(outputs[i], N, 0);
    }
  }

  /// Loads the hsaco and takes the kernel object. Each failure past the reader releases what has
  /// already been created, so the throw leaves nothing behind for Release() to guess at.
  void LoadHsaco(const std::filesystem::path& hsaco_path, const char* kernel_name) {
    const auto image = read_file(hsaco_path);
    if (hsa_code_object_reader_create_from_memory(image.data(), image.size(), &reader) !=
        HSA_STATUS_SUCCESS) {
      reader = {};
      throw std::runtime_error("Failed to create a code object reader");
    }
    if (hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                  nullptr, &executable) != HSA_STATUS_SUCCESS) {
      executable = {};
      throw std::runtime_error("Failed to create an executable");
    }
    if (hsa_executable_load_agent_code_object(executable, agent, reader, nullptr, nullptr) !=
        HSA_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to load the hsaco");
    }
    if (hsa_executable_freeze(executable, nullptr) != HSA_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to freeze the executable");
    }
    hsa_executable_symbol_t symbol{};
    if (hsa_executable_get_symbol_by_name(executable, kernel_name, &agent, &symbol) !=
        HSA_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("Kernel not found in hsaco: ") + kernel_name);
    }
    if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                       &kernel_object) != HSA_STATUS_SUCCESS ||
        kernel_object == 0) {
      throw std::runtime_error("Kernel object is not available");
    }
  }

  void Release() {
    for (auto* p : outputs) {
      if (p) hsa_amd_memory_pool_free(p);
    }
    for (auto* p : inputs) {
      if (p) hsa_amd_memory_pool_free(p);
    }
    // Destroying the executable frees the blobs the loader placed, including the control code the
    // application used to own.
    if (executable.handle) hsa_executable_destroy(executable);
    if (reader.handle) hsa_code_object_reader_destroy(reader);
    if (queue) hsa_queue_destroy(queue);
    if (initialized_) hsa_shut_down();

    outputs.clear();
    inputs.clear();
    executable = {};
    reader = {};
    kernel_object = 0;
    queue = nullptr;
    initialized_ = false;
  }

  bool initialized_ = false;
};

}  // namespace aie_bench

#endif  // ROCRTST_SUITES_AIE_PERFORMANCE_AIE_HSA_HARNESS_H_
