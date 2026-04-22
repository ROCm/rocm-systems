// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Benchmark for vector_scalar_add kernel dispatch via HSA (ROCR).
// Measures combined dispatch overhead (synchronous).

#include <benchmark/benchmark.h>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>

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

// Dispatch packet submission
void dispatch_packet(hsa_queue_t* queue, hsa_amd_aie_ert_start_kernel_data_t* payload,
                     std::size_t payload_dwords) {
  const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queue, 1);

  // Wait if queue is full
  while (wr_idx - hsa_queue_load_read_index_scacquire(queue) >= queue->size) {
    // spin
  }

  const auto mask = queue->size - 1;
  const std::uint64_t packet_id = wr_idx & mask;
  auto* pkt = reinterpret_cast<hsa_amd_aie_ert_packet_t*>(queue->base_address) + packet_id;
  pkt->header.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
  pkt->header.AmdFormat = HSA_AMD_PACKET_TYPE_AIE_ERT;
  pkt->state = HSA_AMD_AIE_ERT_STATE_NEW;
  pkt->count = payload_dwords;
  pkt->opcode = HSA_AMD_AIE_ERT_START_CU;
  pkt->payload_data = reinterpret_cast<std::uint64_t>(payload);

  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
}

}  // namespace

static void VectorScalarAddHSANoAlloc(benchmark::State& state) {
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
  std::uint32_t insts_dwords = static_cast<std::uint32_t>(insts_size / sizeof(std::uint32_t));

  // --- Allocate I/O buffers in data_memory ---
  std::uint32_t* input = nullptr;
  std::uint32_t* output = nullptr;
  if (hsa_amd_memory_pool_allocate(data_pool, DATA_SIZE, 0, reinterpret_cast<void**>(&input)) !=
      HSA_STATUS_SUCCESS) {
    state.SkipWithError("Failed to allocate input buffer");
  }
  if (hsa_amd_memory_pool_allocate(data_pool, DATA_SIZE, 0, reinterpret_cast<void**>(&output)) !=
      HSA_STATUS_SUCCESS) {
    state.SkipWithError("Failed to allocate output buffer");
  }

  // Initialize input: [0, 1, 2, ..., 1023]
  std::iota(input, input + N, std::uint32_t{0});

  // --- Build payload ---
  // For 1 source + 1 destination:
  //   packet_dwords = 3 (insts) + (1+1)*3 (tensors) = 9
  constexpr std::size_t NUM_SRC = 1;
  constexpr std::size_t PACKET_DWORDS = 3 + (NUM_SRC + 1) * 3;  // = 9

  // Allocate payload from kernarg_memory (64 bytes)
  hsa_amd_aie_ert_start_kernel_data_t* payload = nullptr;
  if (hsa_amd_memory_pool_allocate(kernarg_pool, 64, 0, reinterpret_cast<void**>(&payload)) !=
      HSA_STATUS_SUCCESS) {
    state.SkipWithError("Failed to allocate payload buffer");
  }

  // --- Benchmark loop: dispatch is synchronous ---
  for (auto _ : state) {
    // Create HSA packet

    // PDI pointer
    payload->pdi_addr = pdi_buf;
    // Transaction opcode (not counted in packet_dwords)
    payload->data[0] = 0x3;
    payload->data[1] = 0x0;
    // Instructions: 3 dwords
    payload->data[2] =
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(insts_buf) & 0xFFFFFFFF);
    payload->data[3] =
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(insts_buf) >> 32);
    payload->data[4] = insts_dwords;

    // Source address: 2 dwords
    payload->data[5] =
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(input) & 0xFFFFFFFF);
    payload->data[6] = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(input) >> 32);
    // Destination address: 2 dwords
    payload->data[7] =
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(output) & 0xFFFFFFFF);
    payload->data[8] = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(output) >> 32);
    // Sizes: 1 dword per tensor (source, then destination)
    payload->data[9] = static_cast<std::uint32_t>(DATA_SIZE);   // input
    payload->data[10] = static_cast<std::uint32_t>(DATA_SIZE);  // output

    dispatch_packet(queue, payload, PACKET_DWORDS);

    benchmark::ClobberMemory();
  }

  // --- Teardown ---
  hsa_queue_destroy(queue);
  hsa_amd_memory_pool_free(output);
  hsa_amd_memory_pool_free(input);
  hsa_amd_memory_pool_free(payload);
  hsa_amd_memory_pool_free(pdi_buf);
  hsa_amd_memory_pool_free(insts_buf);
  hsa_shut_down();
}

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
  std::uint32_t insts_dwords = static_cast<std::uint32_t>(insts_size / sizeof(std::uint32_t));

  // --- Allocate I/O buffers in data_memory ---
  std::uint32_t* input = nullptr;
  std::uint32_t* output = nullptr;
  if (hsa_amd_memory_pool_allocate(data_pool, DATA_SIZE, 0, reinterpret_cast<void**>(&input)) !=
      HSA_STATUS_SUCCESS) {
    state.SkipWithError("Failed to allocate input buffer");
  }
  if (hsa_amd_memory_pool_allocate(data_pool, DATA_SIZE, 0, reinterpret_cast<void**>(&output)) !=
      HSA_STATUS_SUCCESS) {
    state.SkipWithError("Failed to allocate output buffer");
  }

  // Initialize input: [0, 1, 2, ..., 1023]
  std::iota(input, input + N, std::uint32_t{0});

  // --- Build payload ---
  // For 1 source + 1 destination:
  //   packet_dwords = 3 (insts) + (1+1)*3 (tensors) = 9
  constexpr std::size_t NUM_SRC = 1;
  constexpr std::size_t PACKET_DWORDS = 3 + (NUM_SRC + 1) * 3;  // = 9

  // --- Benchmark loop: dispatch is synchronous ---
  for (auto _ : state) {
    // Allocate payload from kernarg_memory (64 bytes)
    hsa_amd_aie_ert_start_kernel_data_t* payload = nullptr;
    if (hsa_amd_memory_pool_allocate(kernarg_pool, 64, 0, reinterpret_cast<void**>(&payload)) !=
        HSA_STATUS_SUCCESS) {
      state.SkipWithError("Failed to allocate payload buffer");
    }

    // Create HSA packet

    // PDI pointer
    payload->pdi_addr = pdi_buf;
    // Transaction opcode (not counted in packet_dwords)
    payload->data[0] = 0x3;
    payload->data[1] = 0x0;
    // Instructions: 3 dwords
    payload->data[2] =
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(insts_buf) & 0xFFFFFFFF);
    payload->data[3] =
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(insts_buf) >> 32);
    payload->data[4] = insts_dwords;
    // Source address: 2 dwords
    payload->data[5] =
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(input) & 0xFFFFFFFF);
    payload->data[6] = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(input) >> 32);
    // Destination address: 2 dwords
    payload->data[7] =
        static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(output) & 0xFFFFFFFF);
    payload->data[8] = static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(output) >> 32);
    // Sizes: 1 dword per tensor (source, then destination)
    payload->data[9] = static_cast<std::uint32_t>(DATA_SIZE);   // input
    payload->data[10] = static_cast<std::uint32_t>(DATA_SIZE);  // output

    dispatch_packet(queue, payload, PACKET_DWORDS);

    // Free payload after dispatch to include allocation overhead in the benchmark
    hsa_amd_memory_pool_free(payload);

    benchmark::ClobberMemory();
  }

  // --- Teardown ---
  hsa_queue_destroy(queue);
  hsa_amd_memory_pool_free(output);
  hsa_amd_memory_pool_free(input);
  hsa_amd_memory_pool_free(pdi_buf);
  hsa_amd_memory_pool_free(insts_buf);
  hsa_shut_down();
}

BENCHMARK(VectorScalarAddHSANoAlloc)->Unit(benchmark::kMicrosecond);
BENCHMARK(VectorScalarAddHSA)->Unit(benchmark::kMicrosecond);
