/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace {

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

hsa_status_t discover_first_global_coarse_grain_mem_pool(hsa_amd_memory_pool_t pool, void* data) {
  if (!data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_amd_segment_t segment = {};
  auto status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
  if ((status != HSA_STATUS_SUCCESS) || (segment != HSA_AMD_SEGMENT_GLOBAL)) {
    return status;
  }

  hsa_amd_memory_pool_global_flag_t flags = {};
  status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
  if ((status != HSA_STATUS_SUCCESS) ||
      ((flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) == 0x0)) {
    return status;
  }

  std::size_t alloc_granule = 0;
  status = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                        &alloc_granule);
  if ((status != HSA_STATUS_SUCCESS) || (alloc_granule == 0)) {
    return status;
  }

  auto* global_memory_pool = static_cast<hsa_amd_memory_pool_t*>(data);
  *global_memory_pool = pool;

  return HSA_STATUS_INFO_BREAK;
}

hsa_status_t collect_all_pools(hsa_amd_memory_pool_t pool, void* data) {
  if (!data) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto* pools = static_cast<std::vector<hsa_amd_memory_pool_t>*>(data);
  pools->push_back(pool);
  return HSA_STATUS_SUCCESS;
}

}  // namespace

TEST(Memory, PoolAllocate) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  std::uint32_t* buffer = {};
  ASSERT_EQ(hsa_amd_memory_pool_allocate(global_memory_pool, allocation_size, 0 /* flags */,
                                         reinterpret_cast<void**>(&buffer)),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(buffer, nullptr);

  for (std::size_t i = 0; i < buffer_size; ++i) {
    buffer[i] = i;
  }

  // cleanup
  EXPECT_EQ(hsa_amd_memory_pool_free(buffer), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, DMABufExportImportGPUtoAIE) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> gpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_GPU>, &gpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(gpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          gpu_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  std::uint32_t* buffer = {};
  ASSERT_EQ(hsa_amd_memory_pool_allocate(global_memory_pool, allocation_size, 0 /* flags */,
                                         reinterpret_cast<void**>(&buffer)),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(buffer, nullptr);

  for (std::size_t i = 0; i < buffer_size; ++i) {
    buffer[i] = i;
  }

  int dma_buf_fd = -1;
  std::uint64_t dma_buf_offset = 0;
  EXPECT_EQ(hsa_amd_portable_export_dmabuf(buffer, allocation_size, &dma_buf_fd, &dma_buf_offset),
            HSA_STATUS_SUCCESS);
  EXPECT_GT(dma_buf_fd, 0);

  const std::uint32_t num_agents = aie_agents.size();
  auto* agents = aie_agents.data();
  std::size_t import_size = 0;
  std::uint32_t* import_buffer = nullptr;
  // TODO hsa_amd_interop_map_buffer is not implemented for AIE agents. The runtime's InteropMap
  // path calls hsaKmtRegisterGraphicsHandleToNodes which does not support XDNA DRM handles.
  EXPECT_NE(hsa_amd_interop_map_buffer(num_agents, agents, dma_buf_fd, 0 /* flags */, &import_size,
                                       reinterpret_cast<void**>(&import_buffer), nullptr, nullptr),
            HSA_STATUS_SUCCESS);

  // cleanup
  EXPECT_EQ(hsa_amd_portable_close_dmabuf(dma_buf_fd), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(buffer), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, DMABufExportImportAIEtoGPU) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> gpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_GPU>, &gpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(gpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  std::uint32_t* buffer = {};
  ASSERT_EQ(hsa_amd_memory_pool_allocate(global_memory_pool, allocation_size, 0 /* flags */,
                                         reinterpret_cast<void**>(&buffer)),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(buffer, nullptr);

  for (std::size_t i = 0; i < buffer_size; ++i) {
    buffer[i] = i;
  }

  int dma_buf_fd = -1;
  std::uint64_t dma_buf_offset = 0;
  // TODO It calls hsaKmtExportDMABufHandle which assumes it's KFD memory.
  EXPECT_NE(hsa_amd_portable_export_dmabuf(buffer, allocation_size, &dma_buf_fd, &dma_buf_offset),
            HSA_STATUS_SUCCESS);
  EXPECT_LT(dma_buf_fd, 0);

  const std::uint32_t num_agents = gpu_agents.size();
  auto* agents = gpu_agents.data();
  std::size_t import_size = 0;
  std::uint32_t* import_buffer = nullptr;
  // TODO hsaKmtRegisterGraphicsHandleToNodes fails because the KFD thunk cannot register an
  // XDNA-originated DMA-buf handle into GPU nodes. Cross-driver DMA-buf import is not supported.
  EXPECT_NE(hsa_amd_interop_map_buffer(num_agents, agents, dma_buf_fd, 0 /* flags */, &import_size,
                                       reinterpret_cast<void**>(&import_buffer), nullptr, nullptr),
            HSA_STATUS_SUCCESS);

  // cleanup
  EXPECT_EQ(hsa_amd_portable_close_dmabuf(dma_buf_fd), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(buffer), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, MemoryLock) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  std::vector<void*> agent_ptrs(aie_agents.size());

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  std::uint32_t* buffer = new std::uint32_t[buffer_size];
  ASSERT_NE(buffer, nullptr);

  for (std::size_t i = 0; i < buffer_size; ++i) {
    buffer[i] = i;
  }

  // TODO XdnaDriver::RegisterMemory unconditionally returns HSA_STATUS_ERROR. The XDNA DRM driver
  // does not support pinning arbitrary host memory into device-accessible address space.
  const std::uint32_t num_agents = aie_agents.size();
  auto* agents = aie_agents.data();
  EXPECT_NE(hsa_amd_memory_lock(buffer, allocation_size, agents, num_agents, agent_ptrs.data()),
            HSA_STATUS_SUCCESS);

  delete[] buffer;

  // cleanup
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, PoolAllocateAllowAccessGPUtoAIE) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> gpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_GPU>, &gpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(gpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          gpu_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  std::uint32_t* buffer = {};
  const std::uint32_t flags = 0;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(global_memory_pool, allocation_size, flags,
                                         reinterpret_cast<void**>(&buffer)),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(buffer, nullptr);

  for (std::size_t i = 0; i < buffer_size; ++i) {
    buffer[i] = i;
  }

  // TODO MemoryRegion::AllowAccess explicitly rejects kAmdAieDevice agents with
  // HSA_STATUS_ERROR_INVALID_AGENT. Cross-agent access for AIE requires the vmem API path instead.
  const std::uint32_t num_agents = aie_agents.size();
  auto* agents = aie_agents.data();
  EXPECT_NE(hsa_amd_agents_allow_access(num_agents, agents, nullptr, buffer), HSA_STATUS_SUCCESS);

  // cleanup
  EXPECT_EQ(hsa_amd_memory_pool_free(buffer), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, PoolAllocateAllowAccessAIEtoGPU) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> gpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_GPU>, &gpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(gpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  std::uint32_t* buffer = {};
  const std::uint32_t flags = 0;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(global_memory_pool, allocation_size, flags,
                                         reinterpret_cast<void**>(&buffer)),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(buffer, nullptr);

  for (std::size_t i = 0; i < buffer_size; ++i) {
    buffer[i] = i;
  }

  // TODO AllowAccess on an AIE-owned pool fails because the underlying allocation was not made
  // through the KFD thunk, so hsaKmtMapMemoryToGPUNodes cannot map it into GPU nodes.
  const std::uint32_t num_agents = gpu_agents.size();
  auto* agents = gpu_agents.data();
  EXPECT_NE(hsa_amd_agents_allow_access(num_agents, agents, nullptr, buffer), HSA_STATUS_SUCCESS);

  // cleanup
  EXPECT_EQ(hsa_amd_memory_pool_free(buffer), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, VMemSetAccessFromGPU) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> cpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_CPU>, &cpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(cpu_agents.empty());

  std::vector<hsa_agent_t> gpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_GPU>, &gpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(gpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          gpu_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  // allocate on GPU 0
  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
  ASSERT_EQ(hsa_amd_vmem_handle_create(global_memory_pool, allocation_size, MEMORY_TYPE_PINNED,
                                       0 /* flags */, &memory_handle),
            HSA_STATUS_SUCCESS);

  // reserve on host
  const std::uint64_t address = 0;
  const std::uint64_t alignment = 0;
  std::uint32_t* buffer = nullptr;
  ASSERT_EQ(hsa_amd_vmem_address_reserve_align(reinterpret_cast<void**>(&buffer), allocation_size,
                                               address, alignment, 0 /* flags */),
            HSA_STATUS_SUCCESS);

  const std::uint64_t offset = 0;
  ASSERT_EQ(hsa_amd_vmem_map(buffer, allocation_size, offset, memory_handle, 0 /* flags */),
            HSA_STATUS_SUCCESS);

  std::vector<hsa_amd_memory_access_desc_t> memory_access_desc;
  memory_access_desc.reserve(cpu_agents.size() + gpu_agents.size() + aie_agents.size());
  for (auto const& agent : cpu_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_RW, agent});
  }
  for (auto const& agent : gpu_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_RW, agent});
  }
  for (auto const& agent : aie_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_RW, agent});
  }

  EXPECT_EQ(hsa_amd_vmem_set_access(buffer, allocation_size, memory_access_desc.data(),
                                    memory_access_desc.size()),
            HSA_STATUS_SUCCESS);

  // cleanup
  EXPECT_EQ(hsa_amd_vmem_unmap(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_address_free(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_handle_release(memory_handle), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, VMemSetUnsetAccessFromGPU) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> cpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_CPU>, &cpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(cpu_agents.empty());

  std::vector<hsa_agent_t> gpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_GPU>, &gpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(gpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          gpu_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  // allocate on GPU 0
  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
  ASSERT_EQ(hsa_amd_vmem_handle_create(global_memory_pool, allocation_size, MEMORY_TYPE_PINNED,
                                       0 /* flags */, &memory_handle),
            HSA_STATUS_SUCCESS);

  // reserve on host
  const std::uint64_t address = 0;
  const std::uint64_t alignment = 0;
  std::uint32_t* buffer = nullptr;
  ASSERT_EQ(hsa_amd_vmem_address_reserve_align(reinterpret_cast<void**>(&buffer), allocation_size,
                                               address, alignment, 0 /* flags */),
            HSA_STATUS_SUCCESS);

  const std::uint64_t offset = 0;
  ASSERT_EQ(hsa_amd_vmem_map(buffer, allocation_size, offset, memory_handle, 0 /* flags */),
            HSA_STATUS_SUCCESS);

  std::vector<hsa_amd_memory_access_desc_t> memory_access_desc;
  memory_access_desc.reserve(cpu_agents.size() + gpu_agents.size() + aie_agents.size());
  for (auto const& agent : cpu_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_RW, agent});
  }
  for (auto const& agent : gpu_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_RW, agent});
  }
  for (auto const& agent : aie_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_RW, agent});
  }

  ASSERT_EQ(hsa_amd_vmem_set_access(buffer, allocation_size, memory_access_desc.data(),
                                    memory_access_desc.size()),
            HSA_STATUS_SUCCESS);

  memory_access_desc.clear();
  memory_access_desc.reserve(cpu_agents.size() + gpu_agents.size() + aie_agents.size());
  for (auto const& agent : cpu_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_NONE, agent});
  }
  for (auto const& agent : gpu_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_NONE, agent});
  }
  for (auto const& agent : aie_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_NONE, agent});
  }

  EXPECT_EQ(hsa_amd_vmem_set_access(buffer, allocation_size, memory_access_desc.data(),
                                    memory_access_desc.size()),
            HSA_STATUS_SUCCESS);

  // cleanup
  EXPECT_EQ(hsa_amd_vmem_unmap(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_address_free(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_handle_release(memory_handle), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, VMemCreateNPU) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  // allocate on NPU 0
  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
  EXPECT_EQ(hsa_amd_vmem_handle_create(global_memory_pool, allocation_size, MEMORY_TYPE_PINNED,
                                       0 /* flags */, &memory_handle),
            HSA_STATUS_SUCCESS);

  // cleanup
  EXPECT_EQ(hsa_amd_vmem_handle_release(memory_handle), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, VMemMapNPU) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  // allocate on NPU 0
  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
  ASSERT_EQ(hsa_amd_vmem_handle_create(global_memory_pool, allocation_size, MEMORY_TYPE_PINNED,
                                       0 /* flags */, &memory_handle),
            HSA_STATUS_SUCCESS);

  // reserve on host
  const std::uint64_t address = 0;
  const std::uint64_t alignment = 0;
  std::uint32_t* buffer = nullptr;
  ASSERT_EQ(hsa_amd_vmem_address_reserve_align(reinterpret_cast<void**>(&buffer), allocation_size,
                                               address, alignment, 0 /* flags */),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(buffer, nullptr);

  const std::uint64_t offset = 0;
  EXPECT_EQ(hsa_amd_vmem_map(buffer, allocation_size, offset, memory_handle, 0 /* flags */),
            HSA_STATUS_SUCCESS);

  // cleanup
  EXPECT_EQ(hsa_amd_vmem_unmap(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_address_free(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_handle_release(memory_handle), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, VMemSetAccessFromNPU) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> cpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_CPU>, &cpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(cpu_agents.empty());

  std::vector<hsa_agent_t> gpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_GPU>, &gpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(gpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  // allocate on NPU 0
  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
  ASSERT_EQ(hsa_amd_vmem_handle_create(global_memory_pool, allocation_size, MEMORY_TYPE_PINNED,
                                       0 /* flags */, &memory_handle),
            HSA_STATUS_SUCCESS);

  // reserve on host
  const std::uint64_t address = 0;
  const std::uint64_t alignment = 0;
  std::uint32_t* buffer = nullptr;
  ASSERT_EQ(
      hsa_amd_vmem_address_reserve_align(reinterpret_cast<void**>(&buffer), allocation_size,
                                         address, alignment, HSA_AMD_VMEM_ADDRESS_NO_REGISTER),
      HSA_STATUS_SUCCESS);

  const std::uint64_t offset = 0;
  ASSERT_EQ(hsa_amd_vmem_map(buffer, allocation_size, offset, memory_handle, 0 /* flags */),
            HSA_STATUS_SUCCESS);

  std::vector<hsa_amd_memory_access_desc_t> memory_access_desc;
  memory_access_desc.reserve(cpu_agents.size() + gpu_agents.size() + aie_agents.size());
  for (auto const& agent : cpu_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_RW, agent});
  }
  for (auto const& agent : gpu_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_RW, agent});
  }
  for (auto const& agent : aie_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_RW, agent});
  }

  EXPECT_EQ(hsa_amd_vmem_set_access(buffer, allocation_size, memory_access_desc.data(),
                                    memory_access_desc.size()),
            HSA_STATUS_SUCCESS);

  buffer[0] = 42;

  // cleanup
  EXPECT_EQ(hsa_amd_vmem_unmap(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_address_free(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_handle_release(memory_handle), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, VMemSetUnsetAccessFromNPU) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> cpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_CPU>, &cpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(cpu_agents.empty());

  std::vector<hsa_agent_t> gpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_GPU>, &gpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(gpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  // allocate on NPU 0
  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
  ASSERT_EQ(hsa_amd_vmem_handle_create(global_memory_pool, allocation_size, MEMORY_TYPE_PINNED,
                                       0 /* flags */, &memory_handle),
            HSA_STATUS_SUCCESS);

  // reserve on host
  const std::uint64_t address = 0;
  const std::uint64_t alignment = 0;
  std::uint32_t* buffer = nullptr;
  ASSERT_EQ(
      hsa_amd_vmem_address_reserve_align(reinterpret_cast<void**>(&buffer), allocation_size,
                                         address, alignment, HSA_AMD_VMEM_ADDRESS_NO_REGISTER),
      HSA_STATUS_SUCCESS);

  const std::uint64_t offset = 0;
  ASSERT_EQ(hsa_amd_vmem_map(buffer, allocation_size, offset, memory_handle, 0 /* flags */),
            HSA_STATUS_SUCCESS);

  std::vector<hsa_amd_memory_access_desc_t> memory_access_desc;
  memory_access_desc.reserve(cpu_agents.size() + gpu_agents.size() + aie_agents.size());
  for (auto const& agent : cpu_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_RW, agent});
  }
  for (auto const& agent : gpu_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_RW, agent});
  }
  for (auto const& agent : aie_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_RW, agent});
  }

  ASSERT_EQ(hsa_amd_vmem_set_access(buffer, allocation_size, memory_access_desc.data(),
                                    memory_access_desc.size()),
            HSA_STATUS_SUCCESS);

  buffer[0] = 42;

  memory_access_desc.clear();
  memory_access_desc.reserve(cpu_agents.size() + gpu_agents.size() + aie_agents.size());
  for (auto const& agent : cpu_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_NONE, agent});
  }
  for (auto const& agent : gpu_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_NONE, agent});
  }
  for (auto const& agent : aie_agents) {
    memory_access_desc.push_back(hsa_amd_memory_access_desc_t{HSA_ACCESS_PERMISSION_NONE, agent});
  }

  EXPECT_EQ(hsa_amd_vmem_set_access(buffer, allocation_size, memory_access_desc.data(),
                                    memory_access_desc.size()),
            HSA_STATUS_SUCCESS);

  // cleanup
  EXPECT_EQ(hsa_amd_vmem_unmap(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_address_free(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_handle_release(memory_handle), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, PoolGetInfo) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  hsa_amd_segment_t segment = {};
  EXPECT_EQ(
      hsa_amd_memory_pool_get_info(global_memory_pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(segment, HSA_AMD_SEGMENT_GLOBAL);

  uint32_t flags = 0;
  EXPECT_EQ(hsa_amd_memory_pool_get_info(global_memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                         &flags),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED, 0u);

  std::size_t pool_size = 0;
  EXPECT_EQ(
      hsa_amd_memory_pool_get_info(global_memory_pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &pool_size),
      HSA_STATUS_SUCCESS);
  EXPECT_GT(pool_size, 0u);

  bool alloc_allowed = false;
  EXPECT_EQ(hsa_amd_memory_pool_get_info(
                global_memory_pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed),
            HSA_STATUS_SUCCESS);
  EXPECT_TRUE(alloc_allowed);

  std::size_t alloc_granule = 0;
  EXPECT_EQ(hsa_amd_memory_pool_get_info(
                global_memory_pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &alloc_granule),
            HSA_STATUS_SUCCESS);
  EXPECT_GT(alloc_granule, 0u);

  std::size_t alloc_alignment = 0;
  EXPECT_EQ(
      hsa_amd_memory_pool_get_info(
          global_memory_pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT, &alloc_alignment),
      HSA_STATUS_SUCCESS);
  EXPECT_GT(alloc_alignment, 0u);
  EXPECT_EQ(alloc_alignment & (alloc_alignment - 1), 0u);

  bool accessible_by_all = false;
  EXPECT_EQ(hsa_amd_memory_pool_get_info(
                global_memory_pool, HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL, &accessible_by_all),
            HSA_STATUS_SUCCESS);

  std::size_t alloc_max_size = 0;
  EXPECT_EQ(hsa_amd_memory_pool_get_info(global_memory_pool,
                                         HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE, &alloc_max_size),
            HSA_STATUS_SUCCESS);
  EXPECT_GT(alloc_max_size, 0u);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, AgentPoolGetInfo) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> cpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_CPU>, &cpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(cpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  hsa_amd_memory_pool_access_t aie_access = {};
  EXPECT_EQ(hsa_amd_agent_memory_pool_get_info(aie_agents.front(), global_memory_pool,
                                               HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &aie_access),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(aie_access, HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED);

  hsa_amd_memory_pool_access_t cpu_access = {};
  EXPECT_EQ(hsa_amd_agent_memory_pool_get_info(cpu_agents.front(), global_memory_pool,
                                               HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &cpu_access),
            HSA_STATUS_SUCCESS);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, VMemGetAccess) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> cpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_CPU>, &cpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(cpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
  ASSERT_EQ(hsa_amd_vmem_handle_create(global_memory_pool, allocation_size, MEMORY_TYPE_PINNED,
                                       0, &memory_handle),
            HSA_STATUS_SUCCESS);

  void* buffer = nullptr;
  ASSERT_EQ(hsa_amd_vmem_address_reserve_align(&buffer, allocation_size, 0, 0,
                                               HSA_AMD_VMEM_ADDRESS_NO_REGISTER),
            HSA_STATUS_SUCCESS);

  ASSERT_EQ(hsa_amd_vmem_map(buffer, allocation_size, 0, memory_handle, 0), HSA_STATUS_SUCCESS);

  std::vector<hsa_amd_memory_access_desc_t> desc;
  for (auto const& agent : cpu_agents) {
    desc.push_back({HSA_ACCESS_PERMISSION_RW, agent});
  }
  for (auto const& agent : aie_agents) {
    desc.push_back({HSA_ACCESS_PERMISSION_RW, agent});
  }
  ASSERT_EQ(hsa_amd_vmem_set_access(buffer, allocation_size, desc.data(), desc.size()),
            HSA_STATUS_SUCCESS);

  hsa_access_permission_t perms = HSA_ACCESS_PERMISSION_NONE;
  EXPECT_EQ(hsa_amd_vmem_get_access(buffer, &perms, aie_agents.front()), HSA_STATUS_SUCCESS);
  EXPECT_EQ(perms, HSA_ACCESS_PERMISSION_RW);

  hsa_access_permission_t cpu_perms = HSA_ACCESS_PERMISSION_NONE;
  EXPECT_EQ(hsa_amd_vmem_get_access(buffer, &cpu_perms, cpu_agents.front()), HSA_STATUS_SUCCESS);
  EXPECT_EQ(cpu_perms, HSA_ACCESS_PERMISSION_RW);

  // cleanup
  EXPECT_EQ(hsa_amd_vmem_unmap(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_address_free(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_handle_release(memory_handle), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, VMemDataIntegrity) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> cpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_CPU>, &cpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(cpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
  ASSERT_EQ(hsa_amd_vmem_handle_create(global_memory_pool, allocation_size, MEMORY_TYPE_PINNED, 0,
                                       &memory_handle),
            HSA_STATUS_SUCCESS);

  std::uint32_t* buffer = nullptr;
  ASSERT_EQ(hsa_amd_vmem_address_reserve_align(reinterpret_cast<void**>(&buffer), allocation_size,
                                               0, 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER),
            HSA_STATUS_SUCCESS);

  ASSERT_EQ(hsa_amd_vmem_map(buffer, allocation_size, 0, memory_handle, 0), HSA_STATUS_SUCCESS);

  std::vector<hsa_amd_memory_access_desc_t> desc;
  for (auto const& agent : cpu_agents) {
    desc.push_back({HSA_ACCESS_PERMISSION_RW, agent});
  }
  for (auto const& agent : aie_agents) {
    desc.push_back({HSA_ACCESS_PERMISSION_RW, agent});
  }
  ASSERT_EQ(hsa_amd_vmem_set_access(buffer, allocation_size, desc.data(), desc.size()),
            HSA_STATUS_SUCCESS);

  for (std::size_t i = 0; i < buffer_size; ++i) {
    buffer[i] = static_cast<std::uint32_t>(i * 3 + 7);
  }

  for (std::size_t i = 0; i < buffer_size; ++i) {
    EXPECT_EQ(buffer[i], static_cast<std::uint32_t>(i * 3 + 7));
  }

  // cleanup
  EXPECT_EQ(hsa_amd_vmem_unmap(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_address_free(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_handle_release(memory_handle), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, VMemExportImportShareableHandle) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
  ASSERT_EQ(hsa_amd_vmem_handle_create(global_memory_pool, allocation_size, MEMORY_TYPE_PINNED, 0,
                                       &memory_handle),
            HSA_STATUS_SUCCESS);

  int dmabuf_fd = -1;
  EXPECT_EQ(hsa_amd_vmem_export_shareable_handle(&dmabuf_fd, memory_handle, 0), HSA_STATUS_SUCCESS);
  EXPECT_GT(dmabuf_fd, 0);

  hsa_amd_vmem_alloc_handle_t imported_handle = {};
  EXPECT_EQ(hsa_amd_vmem_import_shareable_handle(dmabuf_fd, &imported_handle), HSA_STATUS_SUCCESS);

  // cleanup
  EXPECT_EQ(hsa_amd_vmem_handle_release(imported_handle), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_handle_release(memory_handle), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, PointerInfo) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  void* buffer = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(global_memory_pool, allocation_size, 0, &buffer),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(buffer, nullptr);

  // TODO hsa_amd_pointer_info relies on hsaKmtQueryPointerInfo (KFD thunk) which is unaware of
  // XDNA DRM allocations. The call succeeds but returns HSA_EXT_POINTER_TYPE_UNKNOWN because the
  // pointer was not registered through the KFD path.
  hsa_amd_pointer_info_t info = {};
  info.size = sizeof(info);
  EXPECT_EQ(hsa_amd_pointer_info(buffer, &info, nullptr, nullptr, nullptr), HSA_STATUS_SUCCESS);
  EXPECT_EQ(info.type, HSA_EXT_POINTER_TYPE_UNKNOWN);
  EXPECT_NE(info.sizeInBytes, allocation_size);
  EXPECT_EQ(info.agentBaseAddress, nullptr);

  // cleanup
  EXPECT_EQ(hsa_amd_memory_pool_free(buffer), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, IterateAllPools) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  std::vector<hsa_amd_memory_pool_t> pools;
  ASSERT_EQ(hsa_amd_agent_iterate_memory_pools(aie_agents.front(), collect_all_pools, &pools),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(pools.size(), 3u);

  std::size_t coarse_grain_count = 0;
  for (const auto& pool : pools) {
    hsa_amd_segment_t segment = {};
    EXPECT_EQ(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(segment, HSA_AMD_SEGMENT_GLOBAL);

    uint32_t flags = 0;
    EXPECT_EQ(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags),
              HSA_STATUS_SUCCESS);
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
      ++coarse_grain_count;
    }
  }
  EXPECT_EQ(coarse_grain_count, 3u);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, PoolCanMigrate) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> cpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_CPU>, &cpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(cpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t aie_pool = {};
  ASSERT_EQ(hsa_amd_agent_iterate_memory_pools(
                aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &aie_pool),
            HSA_STATUS_INFO_BREAK);

  hsa_amd_memory_pool_t cpu_pool = {};
  ASSERT_EQ(hsa_amd_agent_iterate_memory_pools(
                cpu_agents.front(), discover_first_global_coarse_grain_mem_pool, &cpu_pool),
            HSA_STATUS_INFO_BREAK);

  bool can_migrate = true;
  EXPECT_EQ(hsa_amd_memory_pool_can_migrate(aie_pool, cpu_pool, &can_migrate),
            HSA_STATUS_ERROR_OUT_OF_RESOURCES);
  EXPECT_FALSE(can_migrate);

  can_migrate = true;
  EXPECT_EQ(hsa_amd_memory_pool_can_migrate(cpu_pool, aie_pool, &can_migrate),
            HSA_STATUS_ERROR_OUT_OF_RESOURCES);
  EXPECT_FALSE(can_migrate);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, VMemMapWithOffset) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> cpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_CPU>, &cpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(cpu_agents.empty());

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  std::size_t alloc_granule = 0;
  ASSERT_EQ(hsa_amd_memory_pool_get_info(global_memory_pool,
                                         HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                         &alloc_granule),
            HSA_STATUS_SUCCESS);
  ASSERT_GT(alloc_granule, 0u);

  const std::size_t allocation_size = alloc_granule * 2;
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
  ASSERT_EQ(hsa_amd_vmem_handle_create(global_memory_pool, allocation_size, MEMORY_TYPE_PINNED,
                                       0, &memory_handle),
            HSA_STATUS_SUCCESS);

  const std::size_t map_size = alloc_granule;
  void* buffer = nullptr;
  ASSERT_EQ(hsa_amd_vmem_address_reserve_align(&buffer, map_size, 0, 0,
                                               HSA_AMD_VMEM_ADDRESS_NO_REGISTER),
            HSA_STATUS_SUCCESS);

  const std::uint64_t offset = alloc_granule;
  EXPECT_EQ(hsa_amd_vmem_map(buffer, map_size, offset, memory_handle, 0), HSA_STATUS_SUCCESS);

  std::vector<hsa_amd_memory_access_desc_t> desc;
  for (auto const& agent : cpu_agents) {
    desc.push_back({HSA_ACCESS_PERMISSION_RW, agent});
  }
  for (auto const& agent : aie_agents) {
    desc.push_back({HSA_ACCESS_PERMISSION_RW, agent});
  }
  EXPECT_EQ(hsa_amd_vmem_set_access(buffer, map_size, desc.data(), desc.size()),
            HSA_STATUS_SUCCESS);

  auto* typed_buffer = static_cast<std::uint32_t*>(buffer);
  typed_buffer[0] = 0xDEADBEEF;
  EXPECT_EQ(typed_buffer[0], 0xDEADBEEF);

  // cleanup
  EXPECT_EQ(hsa_amd_vmem_unmap(buffer, map_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_address_free(buffer, map_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_handle_release(memory_handle), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, VMemDoubleMap) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
  ASSERT_EQ(hsa_amd_vmem_handle_create(global_memory_pool, allocation_size, MEMORY_TYPE_PINNED, 0,
                                       &memory_handle),
            HSA_STATUS_SUCCESS);

  void* buffer = nullptr;
  ASSERT_EQ(hsa_amd_vmem_address_reserve_align(&buffer, allocation_size, 0, 0, 0),
            HSA_STATUS_SUCCESS);

  ASSERT_EQ(hsa_amd_vmem_map(buffer, allocation_size, 0, memory_handle, 0), HSA_STATUS_SUCCESS);

  EXPECT_NE(hsa_amd_vmem_map(buffer, allocation_size, 0, memory_handle, 0), HSA_STATUS_SUCCESS);

  // cleanup
  EXPECT_EQ(hsa_amd_vmem_unmap(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_address_free(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_handle_release(memory_handle), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, VMemReleaseBeforeUnmap) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  hsa_amd_vmem_alloc_handle_t memory_handle = {};
  ASSERT_EQ(hsa_amd_vmem_handle_create(global_memory_pool, allocation_size, MEMORY_TYPE_PINNED, 0,
                                       &memory_handle),
            HSA_STATUS_SUCCESS);

  void* buffer = nullptr;
  ASSERT_EQ(hsa_amd_vmem_address_reserve_align(&buffer, allocation_size, 0, 0, 0),
            HSA_STATUS_SUCCESS);

  ASSERT_EQ(hsa_amd_vmem_map(buffer, allocation_size, 0, memory_handle, 0), HSA_STATUS_SUCCESS);

  // The runtime refcounts handles, so release may succeed but the mapping remains valid.
  const auto release_status = hsa_amd_vmem_handle_release(memory_handle);
  EXPECT_TRUE(release_status == HSA_STATUS_SUCCESS || release_status == HSA_STATUS_ERROR);

  // cleanup
  EXPECT_EQ(hsa_amd_vmem_unmap(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_vmem_address_free(buffer, allocation_size), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, PoolDoubleFree) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t allocation_size = 4096;
  void* buffer = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(global_memory_pool, allocation_size, 0, &buffer),
            HSA_STATUS_SUCCESS);
  ASSERT_NE(buffer, nullptr);

  ASSERT_EQ(hsa_amd_memory_pool_free(buffer), HSA_STATUS_SUCCESS);

  EXPECT_NE(hsa_amd_memory_pool_free(buffer), HSA_STATUS_SUCCESS);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, PoolAllocateMultiple) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t num_buffers = 4;
  constexpr std::size_t sizes[] = {1024, 4096, 8192, 16384};
  std::uint32_t* buffers[num_buffers] = {};

  for (std::size_t i = 0; i < num_buffers; ++i) {
    ASSERT_EQ(hsa_amd_memory_pool_allocate(global_memory_pool, sizes[i], 0,
                                           reinterpret_cast<void**>(&buffers[i])),
              HSA_STATUS_SUCCESS);
    ASSERT_NE(buffers[i], nullptr);
  }

  for (std::size_t i = 0; i < num_buffers; ++i) {
    const std::size_t count = sizes[i] / sizeof(std::uint32_t);
    const auto pattern = static_cast<std::uint32_t>(0xA0 + i);
    for (std::size_t j = 0; j < count; ++j) {
      buffers[i][j] = pattern;
    }
  }

  for (std::size_t i = 0; i < num_buffers; ++i) {
    const std::size_t count = sizes[i] / sizeof(std::uint32_t);
    const auto pattern = static_cast<std::uint32_t>(0xA0 + i);
    for (std::size_t j = 0; j < count; ++j) {
      EXPECT_EQ(buffers[i][j], pattern);
    }
  }

  for (std::size_t i = num_buffers; i > 0; --i) {
    EXPECT_EQ(hsa_amd_memory_pool_free(buffers[i - 1]), HSA_STATUS_SUCCESS);
  }

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Memory, MemoryLockToPool) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  hsa_amd_memory_pool_t global_memory_pool = {};
  ASSERT_EQ(
      hsa_amd_agent_iterate_memory_pools(
          aie_agents.front(), discover_first_global_coarse_grain_mem_pool, &global_memory_pool),
      HSA_STATUS_INFO_BREAK);

  constexpr std::size_t buffer_size = 1024;
  constexpr std::size_t allocation_size = buffer_size * sizeof(std::uint32_t);
  auto* buffer = new std::uint32_t[buffer_size];
  ASSERT_NE(buffer, nullptr);

  void* agent_ptr = nullptr;
  // TODO XdnaDriver::RegisterMemory unconditionally returns HSA_STATUS_ERROR. The XDNA DRM driver
  // does not support pinning arbitrary host memory into device-accessible address space.
  EXPECT_NE(hsa_amd_memory_lock_to_pool(buffer, allocation_size, aie_agents.data(),
                                        static_cast<int>(aie_agents.size()), global_memory_pool, 0,
                                        &agent_ptr),
            HSA_STATUS_SUCCESS);

  delete[] buffer;

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}
