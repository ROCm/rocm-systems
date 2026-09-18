// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Benchmark for vector_scalar_add kernel dispatch via HSA (ROCR).

#include <benchmark/benchmark.h>

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
#include <utility>
#include <vector>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

// ---------------------------------------------------------------------------
// HSA helpers
// ---------------------------------------------------------------------------

namespace {

const std::filesystem::path g_hsaco_path = STRINGIFY(DEFAULT_HSACO_PATH);
constexpr const char* g_kernel_name = DEFAULT_HSACO_KERNEL_NAME;

constexpr std::size_t N = 1024;
constexpr std::size_t DATA_SIZE = N * sizeof(std::uint32_t);

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

  HsaBumpAllocator(HsaBumpAllocator&& other) noexcept
      : base_(std::exchange(other.base_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        offset_(std::exchange(other.offset_, 0)) {}

  HsaBumpAllocator& operator=(HsaBumpAllocator&& other) noexcept {
    if (this != &other) {
      if (base_) hsa_amd_memory_pool_free(base_);
      base_ = std::exchange(other.base_, nullptr);
      size_ = std::exchange(other.size_, 0);
      offset_ = std::exchange(other.offset_, 0);
    }
    return *this;
  }

  template <typename T> T* allocate(std::size_t count, std::size_t alignment = alignof(T)) {
    std::size_t aligned = (offset_ + alignment - 1) & ~(alignment - 1);
    std::size_t new_offset = aligned + count * sizeof(T);
    if (new_offset > size_) throw std::bad_alloc();
    offset_ = new_offset;
    return reinterpret_cast<T*>(static_cast<char*>(base_) + aligned);
  }

  void reset() { offset_ = 0; }
};

// Agent discovery: find AIE agents
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

// Memory pool discovery
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

  bool allocatable = (alloc_rec_granule != 0);
  if (d.expected_allocatable != allocatable) return HSA_STATUS_SUCCESS;

  d.pool = pool;
  return HSA_STATUS_INFO_BREAK;
}

// Reads the hsaco into host memory. Unlike the PDI and instruction blobs this replaces, the image
// is only handed to the code object reader -- the loader places the payloads in device memory
// itself, which is part of the work this benchmark now prices.
std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Cannot open " + path.string());
  const auto size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);
  std::vector<std::uint8_t> data(size);
  f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
  // Without this a truncated file yields a zero-padded tail, and the code object reader reports a
  // malformed image rather than the short read that caused it.
  if (static_cast<std::size_t>(f.gcount()) != size) {
    throw std::runtime_error("Short read loading " + path.string());
  }
  return data;
}

// Loads `g_hsaco_path` onto `agent` and returns the kernel object the loader published. The
// executable and reader are handed back so the caller can destroy them after the run.
std::uint64_t load_hsaco(hsa_agent_t agent, hsa_executable_t* executable,
                         hsa_code_object_reader_t* reader) {
  const auto image = read_file(g_hsaco_path);
  if (hsa_code_object_reader_create_from_memory(image.data(), image.size(), reader) !=
      HSA_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create a code object reader");
  }
  // From here on the caller has nothing to release yet, so each failure releases what this
  // function has already created rather than leaking it into the throw.
  if (hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                executable) != HSA_STATUS_SUCCESS) {
    hsa_code_object_reader_destroy(*reader);
    *reader = {};
    throw std::runtime_error("Failed to create an executable");
  }
  const auto fail = [&](const char* what) {
    hsa_executable_destroy(*executable);
    hsa_code_object_reader_destroy(*reader);
    *executable = {};
    *reader = {};
    throw std::runtime_error(what);
  };
  if (hsa_executable_load_agent_code_object(*executable, agent, *reader, nullptr, nullptr) !=
      HSA_STATUS_SUCCESS) {
    fail("Failed to load the hsaco");
  }
  if (hsa_executable_freeze(*executable, nullptr) != HSA_STATUS_SUCCESS) {
    fail("Failed to freeze the executable");
  }
  hsa_executable_symbol_t symbol{};
  if (hsa_executable_get_symbol_by_name(*executable, g_kernel_name, &agent, &symbol) !=
      HSA_STATUS_SUCCESS) {
    fail("Kernel not found in hsaco");
  }
  std::uint64_t kernel_object = 0;
  if (hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                     &kernel_object) != HSA_STATUS_SUCCESS ||
      kernel_object == 0) {
    fail("Kernel object is not available");
  }
  return kernel_object;
}

// Dispatch packet
std::uint64_t dispatch_packet(std::uint64_t kernel_object, void* input, void* output,
                              uint64_t* kernargs, hsa_queue_t* q) {
  auto* queue = static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(q->base_address);
  const auto mask = q->size - 1;

  // Find slot in the queue
  const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
  // Wait if queue is full
  while (wr_idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {
    // spin
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
  pkt.num_kernargs = 2;
  pkt.kernarg_address = kernargs;

  queue[wr_idx & mask] = pkt;
  return wr_idx;
}

// Everything a benchmark needs: an initialized runtime, an agent, a queue, a loaded hsaco and
// per-dispatch buffers. Set up outside the timed loop, and released in reverse order by the
// destructor -- notably before hsa_shut_down, which a plain scope-exit ordering got wrong.
struct PdiHarness {
  hsa_agent_t agent{};
  hsa_amd_memory_pool_t data_pool{};
  hsa_amd_memory_pool_t kernarg_pool{};
  hsa_queue_t* queue = nullptr;
  hsa_executable_t executable{};
  hsa_code_object_reader_t reader{};
  std::uint64_t kernel_object = 0;
  std::vector<std::uint32_t*> inputs;
  std::vector<std::uint32_t*> outputs;

  explicit PdiHarness(std::int32_t num_dispatches) {
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

    kernel_object = load_hsaco(agent, &executable, &reader);

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

  ~PdiHarness() { Release(); }

  void Release() {
    for (auto* p : outputs) {
      if (p) hsa_amd_memory_pool_free(p);
    }
    for (auto* p : inputs) {
      if (p) hsa_amd_memory_pool_free(p);
    }
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

  PdiHarness(const PdiHarness&) = delete;
  PdiHarness& operator=(const PdiHarness&) = delete;

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

// Dispatch N PDI+insts packets per iteration with the kernargs prepared up front.
static void VectorScalarAddHSA(benchmark::State& state) {
  const std::int32_t num_dispatches = state.range(0);
  try {
    PdiHarness h(num_dispatches);

    std::vector<std::uint64_t*> kernargs(num_dispatches, nullptr);
    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      if (hsa_amd_memory_pool_allocate(h.kernarg_pool, 4 * sizeof(std::uint64_t), 0,
                                       reinterpret_cast<void**>(&kernargs[i])) !=
          HSA_STATUS_SUCCESS) {
        state.SkipWithError("Failed to allocate kernarg buffer");
        return;
      }
      kernargs[i][0] = reinterpret_cast<std::uint64_t>(h.inputs[i]);
      kernargs[i][1] = reinterpret_cast<std::uint64_t>(h.outputs[i]);
      kernargs[i][2] = DATA_SIZE;  // input size
      kernargs[i][3] = DATA_SIZE;  // output size
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
    // Google Benchmark does not catch, so without this a missing artifact or a failed allocation
    // terminates the whole run instead of failing one case.
    state.SkipWithError(e.what());
  }
}

// Same, but the kernargs are carved out of a bump allocator inside the loop, so the numbers
// include the cost of preparing arguments per dispatch.
static void VectorScalarAddHSAAllocKernargs(benchmark::State& state) {
  const std::int32_t num_dispatches = state.range(0);
  try {
    PdiHarness h(num_dispatches);

    // Declared after the harness, so it is destroyed before it -- its pool memory must be freed
    // while the runtime is still up.
    HsaBumpAllocator kernarg_alloc(h.kernarg_pool, num_dispatches * 4 * sizeof(std::uint64_t));

    for (auto _ : state) {
      kernarg_alloc.reset();
      std::uint64_t last_wr_idx = 0;
      for (std::int32_t i = 0; i < num_dispatches; ++i) {
        auto* kernargs = kernarg_alloc.allocate<std::uint64_t>(4);
        kernargs[0] = reinterpret_cast<std::uint64_t>(h.inputs[i]);
        kernargs[1] = reinterpret_cast<std::uint64_t>(h.outputs[i]);
        kernargs[2] = DATA_SIZE;  // input size
        kernargs[3] = DATA_SIZE;  // output size

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

BENCHMARK(VectorScalarAddHSA)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1, 32);
BENCHMARK(VectorScalarAddHSAAllocKernargs)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1, 32);
