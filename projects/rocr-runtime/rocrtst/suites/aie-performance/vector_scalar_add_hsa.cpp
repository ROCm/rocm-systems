// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Benchmark for vector_scalar_add kernel dispatch via HSA (ROCR).

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
#include <utility>
#include <vector>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

// ---------------------------------------------------------------------------
// HSA helpers
// ---------------------------------------------------------------------------

namespace {

const std::filesystem::path g_pdi_path = STRINGIFY(DEFAULT_PDI_PATH);
const std::filesystem::path g_insts_path = STRINGIFY(DEFAULT_INSTS_PATH);

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

// File loaders: read binary files into HSA device memory
void load_binary(hsa_amd_memory_pool_t pool, const std::filesystem::path& path, void** buf,
                 std::size_t& size_out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Cannot open file: " + path.string());

  auto size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);

  if (auto s = hsa_amd_memory_pool_allocate(pool, size, 0, buf); s != HSA_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to allocate buffer for: " + path.string());
  }

  f.read(static_cast<char*>(*buf), size);
  size_out = size;
}

// Dispatch packet
std::uint64_t dispatch_packet(void* pdi_buf, void* insts_buf, std::size_t insts_size, void* input,
                              void* output, uint64_t* kernargs, hsa_queue_t* q) {
  auto* queue = static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(q->base_address);
  const auto mask = q->size - 1;

  // Find slot in the queue
  const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
  // Wait if queue is full
  while (wr_idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {
    // spin
  }
  const std::uint64_t pkt_idx = wr_idx & mask;

  // Write packet
  auto* pkt = queue + pkt_idx;
  *pkt = {};
  pkt->header = (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  pkt->opcode = HSA_AMD_AIE_PACKET_OPCODE_KMQ;
  pkt->count = 24;
  pkt->completion_signal.handle = 0;
  pkt->insts_addr_low = reinterpret_cast<std::uintptr_t>(insts_buf) & 0xFFFFFFFF;
  pkt->insts_addr_high = reinterpret_cast<std::uintptr_t>(insts_buf) >> 32;
  pkt->num_kernargs = 2;
  pkt->kernarg_address = kernargs;
  pkt->insts_size = insts_size;
  pkt->pdi_addr = pdi_buf;

  return wr_idx;
}

}  // namespace

static void VectorScalarAddHSA(benchmark::State& state) {
  // --- Initialize HSA runtime ---
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    state.SkipWithError("hsa_init failed");
    return;
  }

  // --- Find AIE agent ---
  hsa_agent_t aie_agent{};
  if (hsa_iterate_agents(find_aie_agent, &aie_agent) != HSA_STATUS_INFO_BREAK) {
    state.SkipWithError("No AIE agent found");
    hsa_shut_down();
    return;
  }

  // --- Discover memory pools (dev / data / kernarg) ---
  // dev_memory: coarse-grained, non-allocatable (for PDI and instructions)
  find_pool_data dev_pool_data{};
  dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  dev_pool_data.expected_allocatable = false;
  hsa_amd_agent_iterate_memory_pools(aie_agent, find_memory_pool, &dev_pool_data);

  // data_memory: coarse-grained, allocatable (for tensor data)
  find_pool_data data_pool_data{};
  data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  data_pool_data.expected_allocatable = true;
  hsa_amd_agent_iterate_memory_pools(aie_agent, find_memory_pool, &data_pool_data);

  // kernarg_memory: KERNARG_INIT, allocatable (for payloads); fallback to data
  find_pool_data kernarg_pool_data{};
  kernarg_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
  kernarg_pool_data.expected_allocatable = true;
  auto ka_status =
      hsa_amd_agent_iterate_memory_pools(aie_agent, find_memory_pool, &kernarg_pool_data);
  if (ka_status != HSA_STATUS_INFO_BREAK) {
    // fallback to data pool
    kernarg_pool_data.pool = data_pool_data.pool;
  }

  auto dev_pool = dev_pool_data.pool;
  auto data_pool = data_pool_data.pool;
  auto kernarg_pool = kernarg_pool_data.pool;

  // --- Create queue (uses min_queue_size) ---
  std::uint32_t min_queue_size = 0;
  if (hsa_agent_get_info(aie_agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size) !=
      HSA_STATUS_SUCCESS) {
    state.SkipWithError("Failed to get min queue size");
    hsa_shut_down();
    return;
  }

  hsa_queue_t* queue = nullptr;
  if (hsa_queue_create(aie_agent, min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0,
                       &queue) != HSA_STATUS_SUCCESS) {
    state.SkipWithError("Failed to create HSA queue");
    hsa_shut_down();
    return;
  }

  // --- Load PDI and instructions into dev_memory ---
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  load_binary(dev_pool, g_pdi_path, &pdi_buf, pdi_size);

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  load_binary(dev_pool, g_insts_path, &insts_buf, insts_size);

  const std::int32_t num_dispatches = state.range(0);

  // --- Allocate I/O buffers in data_memory ---
  std::vector<std::uint32_t*> inputs(num_dispatches, nullptr);
  std::vector<std::uint32_t*> outputs(num_dispatches, nullptr);
  for (std::int32_t i = 0; i < num_dispatches; ++i) {
    if (hsa_amd_memory_pool_allocate(data_pool, DATA_SIZE, 0,
                                     reinterpret_cast<void**>(&inputs[i])) != HSA_STATUS_SUCCESS) {
      state.SkipWithError("Failed to allocate input buffer");
      return;
    }
    if (hsa_amd_memory_pool_allocate(data_pool, DATA_SIZE, 0,
                                     reinterpret_cast<void**>(&outputs[i])) != HSA_STATUS_SUCCESS) {
      state.SkipWithError("Failed to allocate output buffer");
      return;
    }
    // Initialize input: [1, 2, ..., 1024]
    std::iota(inputs[i], inputs[i] + N, 1);
    // Initialize output: all zeros
    std::fill_n(outputs[i], N, 0);
  }

  // Allocate kernargs from kernarg_memory
  std::vector<uint64_t*> kernargs_vec(num_dispatches, nullptr);
  for (std::int32_t i = 0; i < num_dispatches; ++i) {
    if (hsa_amd_memory_pool_allocate(kernarg_pool, 4 * sizeof(uint64_t), 0,
                                     reinterpret_cast<void**>(&kernargs_vec[i])) !=
        HSA_STATUS_SUCCESS) {
      state.SkipWithError("Failed to allocate payload buffer");
      return;
    }

    // Set args
    kernargs_vec[i][0] = reinterpret_cast<uint64_t>(inputs[i]);
    kernargs_vec[i][1] = reinterpret_cast<uint64_t>(outputs[i]);
    kernargs_vec[i][2] = DATA_SIZE;  // input size
    kernargs_vec[i][3] = DATA_SIZE;  // output size
  }

  // --- Benchmark loop ---
  for (auto _ : state) {
    std::uint64_t last_wr_idx = 0;
    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      // Dispatch HSA packet
      last_wr_idx = dispatch_packet(pdi_buf, insts_buf, insts_size, inputs[i], outputs[i],
                                    kernargs_vec[i], queue);
    }

    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, last_wr_idx);

    benchmark::ClobberMemory();
  }

  // --- Teardown ---
  hsa_queue_destroy(queue);
  for (std::int32_t i = 0; i < num_dispatches; ++i) {
    hsa_amd_memory_pool_free(outputs[i]);
    hsa_amd_memory_pool_free(inputs[i]);
    hsa_amd_memory_pool_free(kernargs_vec[i]);
  }
  hsa_amd_memory_pool_free(pdi_buf);
  hsa_amd_memory_pool_free(insts_buf);
  hsa_shut_down();
}

static void VectorScalarAddHSAAllocKernargs(benchmark::State& state) {
  // --- Initialize HSA runtime ---
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    state.SkipWithError("hsa_init failed");
    return;
  }

  // --- Find AIE agent ---
  hsa_agent_t aie_agent{};
  if (hsa_iterate_agents(find_aie_agent, &aie_agent) != HSA_STATUS_INFO_BREAK) {
    state.SkipWithError("No AIE agent found");
    hsa_shut_down();
    return;
  }

  // --- Discover memory pools (dev / data / kernarg) ---
  // dev_memory: coarse-grained, non-allocatable (for PDI and instructions)
  find_pool_data dev_pool_data{};
  dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  dev_pool_data.expected_allocatable = false;
  hsa_amd_agent_iterate_memory_pools(aie_agent, find_memory_pool, &dev_pool_data);

  // data_memory: coarse-grained, allocatable (for tensor data)
  find_pool_data data_pool_data{};
  data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  data_pool_data.expected_allocatable = true;
  hsa_amd_agent_iterate_memory_pools(aie_agent, find_memory_pool, &data_pool_data);

  // kernarg_memory: KERNARG_INIT, allocatable (for payloads); fallback to data
  find_pool_data kernarg_pool_data{};
  kernarg_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
  kernarg_pool_data.expected_allocatable = true;
  auto ka_status =
      hsa_amd_agent_iterate_memory_pools(aie_agent, find_memory_pool, &kernarg_pool_data);
  if (ka_status != HSA_STATUS_INFO_BREAK) {
    // fallback to data pool
    kernarg_pool_data.pool = data_pool_data.pool;
  }

  auto dev_pool = dev_pool_data.pool;
  auto data_pool = data_pool_data.pool;
  auto kernarg_pool = kernarg_pool_data.pool;

  // --- Create queue (uses min_queue_size) ---
  std::uint32_t min_queue_size = 0;
  if (hsa_agent_get_info(aie_agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size) !=
      HSA_STATUS_SUCCESS) {
    state.SkipWithError("Failed to get min queue size");
    hsa_shut_down();
    return;
  }

  hsa_queue_t* queue = nullptr;
  if (hsa_queue_create(aie_agent, min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0,
                       &queue) != HSA_STATUS_SUCCESS) {
    state.SkipWithError("Failed to create HSA queue");
    hsa_shut_down();
    return;
  }

  // --- Load PDI and instructions into dev_memory ---
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  load_binary(dev_pool, g_pdi_path, &pdi_buf, pdi_size);

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  load_binary(dev_pool, g_insts_path, &insts_buf, insts_size);

  const std::int32_t num_dispatches = state.range(0);

  // --- Allocate I/O buffers in data_memory ---
  std::vector<std::uint32_t*> inputs(num_dispatches, nullptr);
  std::vector<std::uint32_t*> outputs(num_dispatches, nullptr);
  for (std::int32_t i = 0; i < num_dispatches; ++i) {
    if (hsa_amd_memory_pool_allocate(data_pool, DATA_SIZE, 0,
                                     reinterpret_cast<void**>(&inputs[i])) != HSA_STATUS_SUCCESS) {
      state.SkipWithError("Failed to allocate input buffer");
      return;
    }
    if (hsa_amd_memory_pool_allocate(data_pool, DATA_SIZE, 0,
                                     reinterpret_cast<void**>(&outputs[i])) != HSA_STATUS_SUCCESS) {
      state.SkipWithError("Failed to allocate output buffer");
      return;
    }
    // Initialize input: [1, 2, ..., 1024]
    std::iota(inputs[i], inputs[i] + N, 1);
    // Initialize output: all zeros
    std::fill_n(outputs[i], N, 0);
  }

  // --- Bump allocator for kernargs ---
  HsaBumpAllocator kernarg_alloc(kernarg_pool, num_dispatches * 4 * sizeof(uint64_t));

  // --- Benchmark loop ---
  std::vector<uint64_t*> kernargs_vec(num_dispatches, nullptr);
  for (auto _ : state) {
    kernarg_alloc.reset();
    std::uint64_t last_wr_idx = 0;
    for (std::int32_t i = 0; i < num_dispatches; ++i) {
      kernargs_vec[i] = kernarg_alloc.allocate<uint64_t>(4);
      kernargs_vec[i][0] = reinterpret_cast<uint64_t>(inputs[i]);
      kernargs_vec[i][1] = reinterpret_cast<uint64_t>(outputs[i]);
      kernargs_vec[i][2] = DATA_SIZE;  // input size
      kernargs_vec[i][3] = DATA_SIZE;  // output size

      // Dispatch HSA packet
      last_wr_idx = dispatch_packet(pdi_buf, insts_buf, insts_size, inputs[i], outputs[i],
                                    kernargs_vec[i], queue);
    }

    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, last_wr_idx);

    benchmark::ClobberMemory();
  }

  // --- Teardown ---
  hsa_queue_destroy(queue);
  for (std::int32_t i = 0; i < num_dispatches; ++i) {
    hsa_amd_memory_pool_free(outputs[i]);
    hsa_amd_memory_pool_free(inputs[i]);
  }
  hsa_amd_memory_pool_free(pdi_buf);
  hsa_amd_memory_pool_free(insts_buf);
  hsa_shut_down();
}

BENCHMARK(VectorScalarAddHSA)->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1, 32);
BENCHMARK(VectorScalarAddHSAAllocKernargs)
    ->Unit(benchmark::kMicrosecond)
    ->RangeMultiplier(2)
    ->Range(1, 32);
