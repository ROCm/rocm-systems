// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Benchmark for vector_scalar_add kernel dispatch via HSA (ROCR) using full-ELF
// kernels.
//
// The comparison point is vector_scalar_add_hsa.cpp, which dispatches the same
// design through the PDI + instruction sequence path. Two costs differ:
//
//  - Device time. A full-ELF runtime sequence starts with a load_pdi, so the
//    array is reprogrammed on every dispatch. The PDI path programs it once,
//    when the hardware context is configured.
//  - Host time. The application patches each argument address into the control
//    code before enqueuing, work the firmware does itself on the PDI path. This
//    shows up in the CPU-time column, separate from wall time.

#include <benchmark/benchmark.h>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_amd_aie.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

namespace {

const std::filesystem::path g_hsaco_path = STRINGIFY(DEFAULT_ELF_HSACO_PATH);
const char* const g_kernel_name = DEFAULT_ELF_KERNEL_NAME;

constexpr std::size_t N = 1024;
constexpr std::size_t DATA_SIZE = N * sizeof(std::uint32_t);

// Number of kernargs, followed by the same number of sizes.
constexpr std::size_t NUM_KERNARGS = 2;
constexpr std::size_t KERNARG_ENTRIES = 2 * NUM_KERNARGS;

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

hsa_status_t find_aie_agent(hsa_agent_t agent, void* data) {
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

// Finds a global memory pool matching the given flags and allocatability.
struct find_pool_data {
  hsa_amd_memory_pool_global_flag_t expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  bool expected_allocatable = true;
  hsa_amd_memory_pool_t pool{};
};

hsa_status_t find_memory_pool(hsa_amd_memory_pool_t pool, void* data) {
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

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Cannot open file: " + path.string());
  const auto size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<std::uint8_t> data(size);
  f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
  return data;
}

// Write a full-ELF packet into the next queue slot. Does not ring the doorbell,
// so a caller can batch several packets into one command chain.
//
// `ctrl_code` must already hold this dispatch's patched control code.
std::uint64_t dispatch_packet(std::uint64_t kernel_object, void* input, void* output,
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

  auto* pkt = queue + (wr_idx & mask);
  *pkt = {};
  pkt->header = (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  pkt->opcode = HSA_AMD_AIE_PACKET_OPCODE_KMQ;
  pkt->count = 24;
  pkt->completion_signal.handle = 0;
  pkt->kernel_object_low = kernel_object & 0xFFFFFFFF;
  pkt->kernel_object_high = kernel_object >> 32;
  pkt->num_kernargs = NUM_KERNARGS;
  pkt->kernarg_address = kernargs;

  return wr_idx;
}

// Everything a benchmark needs: an initialized runtime, an agent, a queue, a
// loaded ELF and per-dispatch buffers. Set up outside the timed loop, so the
// loop measures only sync, dispatch and wait.
struct ElfHarness {
  hsa_agent_t agent{};
  hsa_amd_memory_pool_t dev_pool{};
  hsa_amd_memory_pool_t data_pool{};
  hsa_amd_memory_pool_t kernarg_pool{};
  hsa_queue_t* queue = nullptr;
  // The application no longer parses the ELF, places the PDI, or allocates and patches a control
  // code per dispatch: it loads an hsaco and holds the kernel object handle the loader published.
  // The runtime does the rest, which is what this benchmark is here to price.
  hsa_executable_t executable{};
  hsa_code_object_reader_t reader{};
  std::uint64_t kernel_object = 0;
  std::vector<std::uint32_t*> inputs;
  std::vector<std::uint32_t*> outputs;

  explicit ElfHarness(std::int32_t num_dispatches) {
    try {
      Acquire(num_dispatches);
    } catch (...) {
      // A throwing constructor does not run its own destructor, so anything already acquired
      // would leak -- including the hsa_init refcount, once per benchmark case.
      Release();
      throw;
    }
  }

  // Throws on failure so a benchmark body can report it through SkipWithError.
  void Acquire(std::int32_t num_dispatches) {
    if (hsa_init() != HSA_STATUS_SUCCESS) throw std::runtime_error("hsa_init failed");
    initialized_ = true;

    if (hsa_iterate_agents(find_aie_agent, &agent) != HSA_STATUS_INFO_BREAK) {
      throw std::runtime_error("No AIE agent found");
    }

    // Full-ELF dispatch is aie2p only; aie2 has neither the firmware command nor the
    // preemption support it is built on.
    char agent_name[64] = {};
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name) != HSA_STATUS_SUCCESS ||
        std::strcmp(agent_name, "aie2p") != 0) {
      throw std::runtime_error(std::string("full-ELF dispatch needs an aie2p agent, found '") +
                               agent_name + "'");
    }

    find_pool_data dev_pool_data{};
    dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    dev_pool_data.expected_allocatable = false;
    hsa_amd_agent_iterate_memory_pools(agent, find_memory_pool, &dev_pool_data);
    dev_pool = dev_pool_data.pool;

    find_pool_data data_pool_data{};
    data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    data_pool_data.expected_allocatable = true;
    hsa_amd_agent_iterate_memory_pools(agent, find_memory_pool, &data_pool_data);
    data_pool = data_pool_data.pool;

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

    // Load the hsaco and take the kernel object. Everything the application used to do by hand --
    // parse the ELF, place the PDI, allocate and patch control code -- now happens behind this.
    const auto image = read_file(g_hsaco_path);
    if (image.empty()) {
      throw std::runtime_error("Failed to read hsaco: " + g_hsaco_path.string());
    }
    if (hsa_code_object_reader_create_from_memory(image.data(), image.size(), &reader) !=
        HSA_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create a code object reader");
    }
    if (hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                  nullptr, &executable) != HSA_STATUS_SUCCESS) {
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
    if (hsa_executable_get_symbol_by_name(executable, g_kernel_name, &agent, &symbol) !=
        HSA_STATUS_SUCCESS) {
      throw std::runtime_error(std::string("Kernel not found in hsaco: ") + g_kernel_name);
    }
    if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                       &kernel_object) != HSA_STATUS_SUCCESS ||
        kernel_object == 0) {
      throw std::runtime_error("Kernel object is not available");
    }

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

  ~ElfHarness() { Release(); }

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

  ElfHarness(const ElfHarness&) = delete;
  ElfHarness& operator=(const ElfHarness&) = delete;

  // Confirm the kernel ran; otherwise the numbers time a no-op dispatch.
  // Input is [1..N], so element i must come back as i + 2.
  bool verify() const {
    for (auto* out : outputs) {
      for (std::size_t i = 0; i < N; ++i) {
        if (out[i] != i + 2) return false;
      }
    }
    return true;
  }

 private:
  bool initialized_ = false;
};

}  // namespace

// Dispatch N full-ELF packets per iteration with the kernargs prepared up front.
// The counterpart of VectorScalarAddHSA.
static void VectorScalarAddHSAELF(benchmark::State& state) {
  const std::int32_t num_dispatches = state.range(0);
  try {
    ElfHarness h(num_dispatches);

    std::vector<std::uint64_t*> kernargs(num_dispatches, nullptr);
    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      if (hsa_amd_memory_pool_allocate(h.kernarg_pool, KERNARG_ENTRIES * sizeof(std::uint64_t), 0,
                                       reinterpret_cast<void**>(&kernargs[i])) !=
          HSA_STATUS_SUCCESS) {
        state.SkipWithError("Failed to allocate kernarg buffer");
        return;
      }
    }

    for (auto _ : state) {
      std::uint64_t last_wr_idx = 0;
      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        last_wr_idx =
            dispatch_packet(h.kernel_object, h.inputs[i], h.outputs[i], kernargs[i], h.queue);
      }

      // The doorbell store submits and waits, so the whole batch is timed.
      hsa_signal_store_screlease(h.queue->doorbell_signal, last_wr_idx);

      benchmark::ClobberMemory();
    }

    if (!h.verify()) state.SkipWithError("Incorrect kernel output");

    for (auto* p : kernargs) {
      if (p) hsa_amd_memory_pool_free(p);
    }
  } catch (const std::exception& e) {
    // Google Benchmark does not catch, so without this a missing artifact or an
    // unsupported agent terminates the whole run instead of failing one case.
    state.SkipWithError(e.what());
  }
}

// Same, but the kernargs are carved out of a bump allocator inside the loop, so
// the numbers include the cost of preparing arguments per dispatch. The
// counterpart of VectorScalarAddHSAAllocKernargs.
static void VectorScalarAddHSAELFAllocKernargs(benchmark::State& state) {
  const std::int32_t num_dispatches = state.range(0);
  try {
    ElfHarness h(num_dispatches);

    HsaBumpAllocator kernarg_alloc(h.kernarg_pool,
                                   num_dispatches * KERNARG_ENTRIES * sizeof(std::uint64_t));

    for (auto _ : state) {
      kernarg_alloc.reset();
      std::uint64_t last_wr_idx = 0;
      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        auto* kernargs = kernarg_alloc.allocate<std::uint64_t>(KERNARG_ENTRIES);
        last_wr_idx =
            dispatch_packet(h.kernel_object, h.inputs[i], h.outputs[i], kernargs, h.queue);
      }

      hsa_signal_store_screlease(h.queue->doorbell_signal, last_wr_idx);

      benchmark::ClobberMemory();
    }

    if (!h.verify()) state.SkipWithError("Incorrect kernel output");
  } catch (const std::exception& e) {
    state.SkipWithError(e.what());
  }
}

BENCHMARK(VectorScalarAddHSAELF)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1, 32);
BENCHMARK(VectorScalarAddHSAELFAllocKernargs)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1, 32);
