/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025-2026, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

static const std::filesystem::path kPdiPath = STRINGIFY(DEFAULT_PDI_PATH);
static const std::filesystem::path kInstsPath = STRINGIFY(DEFAULT_INSTS_PATH);

static constexpr std::size_t N = 1024;

namespace {

// ---------------------------------------------------------------------------
// Agent discovery
// ---------------------------------------------------------------------------

template <hsa_device_type_t DeviceType>
hsa_status_t discover_agents(hsa_agent_t agent, void* data) {
  if (!data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_device_type_t device_type = {};
  const auto status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  if (device_type == DeviceType) {
    auto* const agents = static_cast<std::vector<hsa_agent_t>*>(data);
    agents->push_back(agent);
  }

  return HSA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Memory pool discovery
// ---------------------------------------------------------------------------

struct find_pool_data {
  hsa_amd_memory_pool_global_flag_t expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  bool expected_allocatable = true;
  hsa_amd_memory_pool_t pool{};
};

hsa_status_t find_memory_pool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment{};
  auto s = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if (s != HSA_STATUS_SUCCESS) {
    return s;
  }
  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_amd_memory_pool_global_flag_t flags{};
  if (auto s = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
      s != HSA_STATUS_SUCCESS) {
    return s;
  }

  auto& d = *static_cast<find_pool_data*>(data);
  if ((flags & d.expected_flags) == 0) {
    return HSA_STATUS_SUCCESS;
  }

  std::size_t alloc_rec_granule = 0;
  if (auto s = hsa_amd_memory_pool_get_info(
          pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, &alloc_rec_granule);
      s != HSA_STATUS_SUCCESS) {
    return s;
  }

  const bool allocatable = (alloc_rec_granule != 0);
  if (d.expected_allocatable != allocatable) {
    return HSA_STATUS_SUCCESS;
  }

  d.pool = pool;
  return HSA_STATUS_INFO_BREAK;
}

// ---------------------------------------------------------------------------
// Binary loader
// ---------------------------------------------------------------------------

bool load_binary(hsa_amd_memory_pool_t pool, const std::filesystem::path& path, void** buf,
                 std::size_t& size_out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    return false;
  }

  const auto size = static_cast<std::size_t>(f.tellg());
  f.seekg(0);

  if (hsa_amd_memory_pool_allocate(pool, size, 0, buf) != HSA_STATUS_SUCCESS) {
    return false;
  }

  f.read(static_cast<char*>(*buf), static_cast<std::streamsize>(size));
  if (static_cast<std::size_t>(f.gcount()) != size) {
    hsa_amd_memory_pool_free(*buf);
    *buf = nullptr;
    return false;
  }
  size_out = size;
  return true;
}

// ---------------------------------------------------------------------------
// AIE packet submission
// ---------------------------------------------------------------------------

std::uint64_t submit_aie_packet(hsa_queue_t* queue, hsa_amd_aie_ert_start_kernel_data_t* payload,
                                std::uint32_t payload_dwords, hsa_signal_t completion_signal) {
  hsa_amd_aie_ert_packet_t pkt{};
  pkt.header.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
  pkt.header.AmdFormat = HSA_AMD_PACKET_TYPE_AIE_ERT;
  pkt.state = HSA_AMD_AIE_ERT_STATE_NEW;
  pkt.count = payload_dwords;
  pkt.opcode = HSA_AMD_AIE_ERT_START_CU;
  pkt.completion_signal = completion_signal;
  pkt.payload_data = reinterpret_cast<std::uint64_t>(payload);

  const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(queue, 1);

  while (wr_idx - hsa_queue_load_read_index_scacquire(queue) >= queue->size) {
  }

  const std::uint64_t packet_id = wr_idx % queue->size;
  *(static_cast<hsa_amd_aie_ert_packet_t*>(queue->base_address) + packet_id) = pkt;

  return wr_idx;
}

// ---------------------------------------------------------------------------
// Payload builder
// ---------------------------------------------------------------------------

void create_aie_vector_scalar_payload(hsa_amd_aie_ert_start_kernel_data_t* payload, void* pdi_buf,
                                      void* insts_buf, std::uint32_t insts_dwords, void* input,
                                      void* output, std::uint32_t size) {
  payload->pdi_addr = pdi_buf;

  // Transaction opcode
  payload->data[0] = 0x3;
  payload->data[1] = 0x0;

  std::size_t idx = 2;

  // Instructions: pointer (lo/hi) + dword count
  payload->data[idx + 0] =
      static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(insts_buf) & 0xFFFFFFFF);
  payload->data[idx + 1] =
      static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(insts_buf) >> 32);
  payload->data[idx + 2] = insts_dwords;
  idx += 3;

  // Source tensor: pointer (lo/hi)
  payload->data[idx + 0] =
      static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(input) & 0xFFFFFFFF);
  payload->data[idx + 1] =
      static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(input) >> 32);
  idx += 2;

  // Destination tensor: pointer (lo/hi)
  payload->data[idx + 0] =
      static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(output) & 0xFFFFFFFF);
  payload->data[idx + 1] =
      static_cast<std::uint32_t>(reinterpret_cast<std::uintptr_t>(output) >> 32);
  idx += 2;

  // Sizes: 1 dword per tensor
  payload->data[idx + 0] = size;
  payload->data[idx + 1] = size;
}

}  // namespace

// ===========================================================================
// Tests
// ===========================================================================

TEST(Dispatch, QueueCreate) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  std::uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);
  ASSERT_GT(min_queue_size, 0u);

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(queue, nullptr);

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, QueueMinMaxSize) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  std::uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);
  EXPECT_GT(min_queue_size, 0u);

  std::uint32_t max_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MAX_SIZE, &max_queue_size),
            HSA_STATUS_SUCCESS);
  EXPECT_GT(max_queue_size, 0u);

  EXPECT_LE(min_queue_size, max_queue_size);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, DevPoolDiscovery) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  find_pool_data dev_pool_data{};
  dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  dev_pool_data.expected_allocatable = false;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &dev_pool_data),
      HSA_STATUS_INFO_BREAK);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, KernargPoolDiscovery) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  // Try KERNARG_INIT pool first
  find_pool_data kernarg_pool_data{};
  kernarg_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
  kernarg_pool_data.expected_allocatable = true;
  auto ka_status =
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &kernarg_pool_data);

  if (ka_status != HSA_STATUS_INFO_BREAK) {
    // Fall back to allocatable coarse-grained pool
    find_pool_data data_pool_data{};
    data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    data_pool_data.expected_allocatable = true;
    ASSERT_EQ(
        hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &data_pool_data),
        HSA_STATUS_INFO_BREAK);
    kernarg_pool_data.pool = data_pool_data.pool;
  }

  // Verify we can allocate from the discovered pool
  void* test_buf = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool_data.pool, 64, 0, &test_buf),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(test_buf, nullptr);
  EXPECT_EQ(hsa_amd_memory_pool_free(test_buf), HSA_STATUS_SUCCESS);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, LoadPDI) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  find_pool_data dev_pool_data{};
  dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  dev_pool_data.expected_allocatable = false;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &dev_pool_data),
      HSA_STATUS_INFO_BREAK);

  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool_data.pool, kPdiPath, &pdi_buf, pdi_size));
  EXPECT_NE(pdi_buf, nullptr);
  EXPECT_GT(pdi_size, 0u);

  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, LoadInstructions) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  find_pool_data dev_pool_data{};
  dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  dev_pool_data.expected_allocatable = false;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &dev_pool_data),
      HSA_STATUS_INFO_BREAK);

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool_data.pool, kInstsPath, &insts_buf, insts_size));
  EXPECT_NE(insts_buf, nullptr);
  EXPECT_GT(insts_size, 0u);
  EXPECT_EQ(insts_size % sizeof(std::uint32_t), 0u);

  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, SingleDispatch) {
  constexpr std::size_t payload_size = 64;

  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  // --- Discover AIE agent ---
  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  // --- Discover memory pools ---
  // dev pool: coarse-grained, non-allocatable (for PDI and instructions)
  find_pool_data dev_pool_data{};
  dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  dev_pool_data.expected_allocatable = false;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &dev_pool_data),
      HSA_STATUS_INFO_BREAK);

  // data pool: coarse-grained, allocatable (for tensor data)
  find_pool_data data_pool_data{};
  data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  data_pool_data.expected_allocatable = true;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &data_pool_data),
      HSA_STATUS_INFO_BREAK);

  // kernarg pool: KERNARG_INIT, allocatable; falls back to data pool
  find_pool_data kernarg_pool_data{};
  kernarg_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
  kernarg_pool_data.expected_allocatable = true;
  if (hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool,
                                         &kernarg_pool_data) != HSA_STATUS_INFO_BREAK) {
    kernarg_pool_data.pool = data_pool_data.pool;
  }

  auto dev_pool = dev_pool_data.pool;
  auto data_pool = data_pool_data.pool;
  auto kernarg_pool = kernarg_pool_data.pool;

  // --- Create queue ---
  std::uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load PDI and instructions ---
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, kPdiPath, &pdi_buf, pdi_size));

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, kInstsPath, &insts_buf, insts_size));
  const auto insts_dwords = static_cast<std::uint32_t>(insts_size / sizeof(std::uint32_t));

  // --- Allocate I/O buffers ---
  const std::size_t data_size = N * sizeof(std::uint32_t);

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, data_size, 0, reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(
      hsa_amd_memory_pool_allocate(data_pool, data_size, 0, reinterpret_cast<void**>(&output)),
      HSA_STATUS_SUCCESS);

  std::iota(input, input + N, 0);
  std::fill_n(output, N, 0);

  // --- Create payload ---
  constexpr std::uint32_t NUM_SRC = 1;
  constexpr std::uint32_t PACKET_DWORDS = 3 + (NUM_SRC + 1) * 3;  // = 9

  hsa_amd_aie_ert_start_kernel_data_t* payload = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, payload_size, 0,
                                         reinterpret_cast<void**>(&payload)),
            HSA_STATUS_SUCCESS);

  create_aie_vector_scalar_payload(payload, pdi_buf, insts_buf, insts_dwords, input, output,
                                   static_cast<std::uint32_t>(data_size));

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Submit packet ---
  const auto wr_idx = submit_aie_packet(queue, payload, PACKET_DWORDS, signal);

  // --- Ring doorbell ---
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  // --- Wait for completion ---
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // --- Verify output: output[i] == input[i] + 1 ---
  for (std::size_t i = 0; i < N; ++i) {
    EXPECT_EQ(output[i], static_cast<std::uint32_t>(i + 1)) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(payload), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, MultiDispatch) {
  constexpr std::size_t payload_size = 64;

  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  // --- Discover AIE agent ---
  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  // --- Discover memory pools ---
  find_pool_data dev_pool_data{};
  dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  dev_pool_data.expected_allocatable = false;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &dev_pool_data),
      HSA_STATUS_INFO_BREAK);

  find_pool_data data_pool_data{};
  data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  data_pool_data.expected_allocatable = true;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &data_pool_data),
      HSA_STATUS_INFO_BREAK);

  find_pool_data kernarg_pool_data{};
  kernarg_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
  kernarg_pool_data.expected_allocatable = true;
  if (hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool,
                                         &kernarg_pool_data) != HSA_STATUS_INFO_BREAK) {
    kernarg_pool_data.pool = data_pool_data.pool;
  }

  auto dev_pool = dev_pool_data.pool;
  auto data_pool = data_pool_data.pool;
  auto kernarg_pool = kernarg_pool_data.pool;

  // --- Create queue ---
  std::uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);
  const std::uint32_t total_num_dispatches = 100;

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load PDI and instructions ---
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, kPdiPath, &pdi_buf, pdi_size));

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, kInstsPath, &insts_buf, insts_size));
  const auto insts_dwords = static_cast<std::uint32_t>(insts_size / sizeof(std::uint32_t));

  // --- Allocate I/O buffers ---
  const std::size_t data_size = N * sizeof(std::uint32_t);
  const auto total_data_size = data_size * total_num_dispatches;

  std::uint32_t* input = nullptr;
  ASSERT_EQ(
      hsa_amd_memory_pool_allocate(data_pool, total_data_size, 0, reinterpret_cast<void**>(&input)),
      HSA_STATUS_SUCCESS);
  std::iota(input, input + N * total_num_dispatches, 0);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_data_size, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::fill_n(output, N * total_num_dispatches, 0);

  // --- Allocate storage for packet payload ---
  constexpr std::uint32_t NUM_SRC = 1;
  constexpr std::uint32_t PACKET_DWORDS = 3 + (NUM_SRC + 1) * 3;

  void* payload = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, (payload_size * total_num_dispatches), 0,
                                         &payload),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Dispatch loop ---
  for (std::uint32_t iter = 0; iter < total_num_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto input_ptr = input + iter * N;
    auto output_ptr = output + iter * N;
    auto payload_ptr = reinterpret_cast<hsa_amd_aie_ert_start_kernel_data_t*>(
        static_cast<std::byte*>(payload) + (payload_size * iter));

    // Create payload for this dispatch
    create_aie_vector_scalar_payload(payload_ptr, pdi_buf, insts_buf, insts_dwords, input_ptr,
                                     output_ptr, static_cast<std::uint32_t>(data_size));

    // Submit
    const auto wr_idx = submit_aie_packet(queue, payload_ptr, PACKET_DWORDS, signal);

    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  }

  // Wait for completion of all dispatches
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify output: output == input[i] + 1
  for (std::size_t i = 0; i < N * total_num_dispatches; ++i) {
    EXPECT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(payload), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, MultiDispatchAsync) {
  constexpr std::size_t payload_size = 64;

  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  // --- Discover AIE agent ---
  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  // --- Discover memory pools ---
  find_pool_data dev_pool_data{};
  dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  dev_pool_data.expected_allocatable = false;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &dev_pool_data),
      HSA_STATUS_INFO_BREAK);

  find_pool_data data_pool_data{};
  data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  data_pool_data.expected_allocatable = true;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &data_pool_data),
      HSA_STATUS_INFO_BREAK);

  find_pool_data kernarg_pool_data{};
  kernarg_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
  kernarg_pool_data.expected_allocatable = true;
  if (hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool,
                                         &kernarg_pool_data) != HSA_STATUS_INFO_BREAK) {
    kernarg_pool_data.pool = data_pool_data.pool;
  }

  auto dev_pool = dev_pool_data.pool;
  auto data_pool = data_pool_data.pool;
  auto kernarg_pool = kernarg_pool_data.pool;

  // --- Create queue ---
  std::uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);
  const std::uint32_t total_num_dispatches = 2;

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load PDI and instructions ---
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, kPdiPath, &pdi_buf, pdi_size));

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, kInstsPath, &insts_buf, insts_size));
  const auto insts_dwords = static_cast<std::uint32_t>(insts_size / sizeof(std::uint32_t));

  // --- Allocate I/O buffers ---
  const std::size_t data_size = N * sizeof(std::uint32_t);
  const auto total_data_size = data_size * total_num_dispatches;

  std::uint32_t* input = nullptr;
  ASSERT_EQ(
      hsa_amd_memory_pool_allocate(data_pool, total_data_size, 0, reinterpret_cast<void**>(&input)),
      HSA_STATUS_SUCCESS);
  std::iota(input, input + N * total_num_dispatches, 0);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_data_size, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::fill_n(output, N * total_num_dispatches, 0);

  // --- Allocate storage for packet payload ---
  constexpr std::uint32_t NUM_SRC = 1;
  constexpr std::uint32_t PACKET_DWORDS = 3 + (NUM_SRC + 1) * 3;

  void* payload = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, (payload_size * total_num_dispatches), 0,
                                         &payload),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Dispatch loop ---
  std::uint64_t last_wr_idx = 0;
  for (std::uint32_t iter = 0; iter < total_num_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto input_ptr = input + iter * N;
    auto output_ptr = output + iter * N;
    auto payload_ptr = reinterpret_cast<hsa_amd_aie_ert_start_kernel_data_t*>(
        static_cast<std::byte*>(payload) + (payload_size * iter));

    // Create payload for this dispatch
    create_aie_vector_scalar_payload(payload_ptr, pdi_buf, insts_buf, insts_dwords, input_ptr,
                                     output_ptr, static_cast<std::uint32_t>(data_size));

    // Submit
    last_wr_idx = submit_aie_packet(queue, payload_ptr, PACKET_DWORDS, signal);
  }

  // Ring doorbell
  hsa_signal_store_screlease(queue->doorbell_signal, last_wr_idx);

  // Wait for completion of all dispatches
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify output: output == input[i] + 1
  for (std::size_t i = 0; i < N * total_num_dispatches; ++i) {
    EXPECT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(payload), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, MultiDispatchWrapAround) {
  constexpr std::size_t payload_size = 64;

  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  // --- Discover AIE agent ---
  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  // --- Discover memory pools ---
  find_pool_data dev_pool_data{};
  dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  dev_pool_data.expected_allocatable = false;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &dev_pool_data),
      HSA_STATUS_INFO_BREAK);

  find_pool_data data_pool_data{};
  data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  data_pool_data.expected_allocatable = true;
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &data_pool_data),
      HSA_STATUS_INFO_BREAK);

  find_pool_data kernarg_pool_data{};
  kernarg_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
  kernarg_pool_data.expected_allocatable = true;
  if (hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool,
                                         &kernarg_pool_data) != HSA_STATUS_INFO_BREAK) {
    kernarg_pool_data.pool = data_pool_data.pool;
  }

  auto dev_pool = dev_pool_data.pool;
  auto data_pool = data_pool_data.pool;
  auto kernarg_pool = kernarg_pool_data.pool;

  // --- Create queue ---
  std::uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);
  const std::uint32_t initial_dispatches = min_queue_size / 2;
  const std::uint32_t num_dispatches = 10 * min_queue_size;
  const std::uint32_t total_num_dispatches = initial_dispatches + num_dispatches;

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load PDI and instructions ---
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, kPdiPath, &pdi_buf, pdi_size));

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, kInstsPath, &insts_buf, insts_size));
  const auto insts_dwords = static_cast<std::uint32_t>(insts_size / sizeof(std::uint32_t));

  // --- Allocate I/O buffers ---
  const std::size_t data_size = N * sizeof(std::uint32_t);

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, (data_size * total_num_dispatches), 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + (N * total_num_dispatches), 0);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, (data_size * total_num_dispatches), 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::fill_n(output, N * total_num_dispatches, 0);

  // --- Allocate storage for packet payload ---
  constexpr std::uint32_t NUM_SRC = 1;
  constexpr std::uint32_t PACKET_DWORDS = 3 + (NUM_SRC + 1) * 3;

  void* payload = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, (payload_size * total_num_dispatches), 0,
                                         &payload),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Initial Packets to set-up for wrap-around ---
  for (std::uint32_t iter = 0; iter < initial_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto input_ptr = input + iter * N;
    auto output_ptr = output + iter * N;
    auto payload_ptr = reinterpret_cast<hsa_amd_aie_ert_start_kernel_data_t*>(
        reinterpret_cast<std::byte*>(payload) + payload_size * iter);

    // Create packet
    create_aie_vector_scalar_payload(payload_ptr, pdi_buf, insts_buf, insts_dwords, input_ptr,
                                     output_ptr, static_cast<std::uint32_t>(data_size));

    // Submit
    const auto wr_idx = submit_aie_packet(queue, payload_ptr, PACKET_DWORDS, signal);

    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  }

  // --- Dispatch loop ---
  for (std::uint32_t iter = 0; iter < num_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto offset = initial_dispatches + iter;
    auto input_ptr = input + offset * N;
    auto output_ptr = output + offset * N;
    auto payload_ptr = reinterpret_cast<hsa_amd_aie_ert_start_kernel_data_t*>(
        reinterpret_cast<std::byte*>(payload) + payload_size * offset);

    // Create packet
    create_aie_vector_scalar_payload(payload_ptr, pdi_buf, insts_buf, insts_dwords, input_ptr,
                                     output_ptr, static_cast<std::uint32_t>(data_size));

    // Submit
    const auto wr_idx = submit_aie_packet(queue, payload_ptr, PACKET_DWORDS, signal);

    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  }

  // Wait for completion of all dispatches
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify output: output[i] == input[i] + 1
  for (std::size_t i = 0; i < N * total_num_dispatches; ++i) {
    EXPECT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(payload), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}
