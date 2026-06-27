/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_amd_aie.h"

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

namespace {

// ---------------------------------------------------------------------------
// Agent discovery
// ---------------------------------------------------------------------------

// HSA agent iteration callback: appends every agent of type DeviceType to the
// std::vector<hsa_agent_t>* stored in `data`.
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

// Parameters and result for find_memory_pool: describes what kind of pool to
// look for and receives the first matching pool handle.
struct find_pool_data {
  hsa_amd_memory_pool_global_flag_t expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
  bool expected_allocatable = true;
  hsa_amd_memory_pool_t pool{};
};

// HSA memory-pool iteration callback: stops at the first global pool that
// matches the flags and allocatability recorded in the find_pool_data* stored
// in `data`, storing the result there and returning HSA_STATUS_INFO_BREAK.
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

// Open `path` for binary reading and report its size. On success the returned
// stream is positioned at the start; on failure it is in a failed state, so the
// caller can test it with `operator bool`.
std::ifstream open_binary(const std::filesystem::path& path, std::size_t* size_out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (f) {
    *size_out = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
  }
  return f;
}

// Read exactly `size` bytes from `f` into `dst`. Returns false on short read.
bool read_exact(std::ifstream& f, void* dst, std::size_t size) {
  f.read(static_cast<char*>(dst), static_cast<std::streamsize>(size));
  return static_cast<std::size_t>(f.gcount()) == size;
}

testing::AssertionResult load_binary(hsa_amd_memory_pool_t pool, const std::filesystem::path& path,
                                     void** buf, std::size_t& size_out) {
  std::size_t size = 0;
  auto f = open_binary(path, &size);
  if (!f) {
    return testing::AssertionFailure() << "failed to open '" << path << "'";
  }

  if (hsa_amd_memory_pool_allocate(pool, size, 0, buf) != HSA_STATUS_SUCCESS) {
    return testing::AssertionFailure()
        << "failed to allocate " << size << " bytes for '" << path << "'";
  }

  if (!read_exact(f, *buf, size)) {
    hsa_amd_memory_pool_free(*buf);
    *buf = nullptr;
    return testing::AssertionFailure()
        << "short read loading '" << path << "' (expected " << size << " bytes)";
  }
  size_out = size;
  return testing::AssertionSuccess();
}

// ---------------------------------------------------------------------------
// Virtual memory (vmem) allocation
// ---------------------------------------------------------------------------

// Owns a vmem allocation: the physical handle, the reserved virtual address, and
// the mapped size. All three are needed to fully release the buffer via vmem_free.
struct vmem_buffer {
  hsa_amd_vmem_alloc_handle_t handle{};
  void* va = nullptr;
  std::size_t size = 0;
};

// Allocate a buffer through the vmem API and make it accessible to every agent
// in `agents`. The allocation is created from `pool` (a coarse-grained,
// allocatable global pool), reserved, mapped, and granted RW access.
testing::AssertionResult vmem_allocate(hsa_amd_memory_pool_t pool, std::size_t size,
                                       const std::vector<hsa_agent_t>& agents, vmem_buffer* out) {
  // The vmem API requires the allocation size to be a multiple of the pool's
  // allocation granule (page size); unlike hsa_amd_memory_pool_allocate it does
  // not round up internally. Round the request up to the granule.
  std::size_t granule = 0;
  if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                   &granule) != HSA_STATUS_SUCCESS ||
      granule == 0) {
    return testing::AssertionFailure() << "failed to query pool allocation granule";
  }
  size = ((size + granule - 1) / granule) * granule;

  vmem_buffer buf{};
  buf.size = size;

  if (hsa_amd_vmem_handle_create(pool, size, MEMORY_TYPE_PINNED, 0, &buf.handle) !=
      HSA_STATUS_SUCCESS) {
    return testing::AssertionFailure()
        << "hsa_amd_vmem_handle_create failed for " << size << " bytes";
  }
  if (hsa_amd_vmem_address_reserve_align(&buf.va, size, 0, 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER) !=
      HSA_STATUS_SUCCESS) {
    hsa_amd_vmem_handle_release(buf.handle);
    return testing::AssertionFailure()
        << "hsa_amd_vmem_address_reserve_align failed for " << size << " bytes";
  }
  if (hsa_amd_vmem_map(buf.va, size, 0, buf.handle, 0) != HSA_STATUS_SUCCESS) {
    hsa_amd_vmem_address_free(buf.va, size);
    hsa_amd_vmem_handle_release(buf.handle);
    return testing::AssertionFailure() << "hsa_amd_vmem_map failed for " << size << " bytes";
  }

  std::vector<hsa_amd_memory_access_desc_t> desc;
  desc.reserve(agents.size());
  for (const auto& agent : agents) {
    desc.push_back({HSA_ACCESS_PERMISSION_RW, agent});
  }
  if (hsa_amd_vmem_set_access(buf.va, size, desc.data(), desc.size()) != HSA_STATUS_SUCCESS) {
    hsa_amd_vmem_unmap(buf.va, size);
    hsa_amd_vmem_address_free(buf.va, size);
    hsa_amd_vmem_handle_release(buf.handle);
    return testing::AssertionFailure()
        << "hsa_amd_vmem_set_access failed for " << agents.size() << " agents";
  }

  *out = buf;
  return testing::AssertionSuccess();
}

// Release a vmem buffer: unmap, free the address range, release the handle.
// All steps are attempted even if an earlier one fails, so a single buffer
// cannot leak the rest of its resources.
testing::AssertionResult vmem_free(const vmem_buffer& buf) {
  std::vector<const char*> failures;
  if (hsa_amd_vmem_unmap(buf.va, buf.size) != HSA_STATUS_SUCCESS) {
    failures.push_back("hsa_amd_vmem_unmap");
  }
  if (hsa_amd_vmem_address_free(buf.va, buf.size) != HSA_STATUS_SUCCESS) {
    failures.push_back("hsa_amd_vmem_address_free");
  }
  if (hsa_amd_vmem_handle_release(buf.handle) != HSA_STATUS_SUCCESS) {
    failures.push_back("hsa_amd_vmem_handle_release");
  }
  if (failures.empty()) {
    return testing::AssertionSuccess();
  }

  auto result = testing::AssertionFailure() << "vmem free failed:";
  for (const auto* f : failures) {
    result << ' ' << f;
  }
  return result;
}

#if 0
// Load a file into a freshly vmem-allocated buffer accessible to `agents`.
// Disabled: PDI/instructions must live in the dev heap, which is incompatible
// with the vmem reserve+map path (see docs/bug-vmem-map-dev-heap.md). Kept here,
// guarded out, so the intended vmem code path is preserved for when the
// runtime/driver gains dev-heap vmem support.
testing::AssertionResult load_binary_vmem(hsa_amd_memory_pool_t pool,
                                          const std::vector<hsa_agent_t>& agents,
                                          const std::filesystem::path& path, vmem_buffer* out) {
  std::size_t size = 0;
  auto f = open_binary(path, &size);
  if (!f) {
    return testing::AssertionFailure() << "failed to open '" << path << "'";
  }

  if (auto r = vmem_allocate(pool, size, agents, out); !r) {
    return testing::AssertionFailure() << "loading '" << path << "': " << r.message();
  }

  if (!read_exact(f, out->va, size)) {
    vmem_free(*out);
    *out = {};
    return testing::AssertionFailure()
        << "short read loading '" << path << "' (expected " << size << " bytes)";
  }
  return testing::AssertionSuccess();
}
#endif

// ---------------------------------------------------------------------------
// AIE packet submission
// ---------------------------------------------------------------------------

// Compile-time constants and dispatch helper for the vector-scalar-add AIE
// kernel: adds 1 to every element of a uint32 array of element_count entries.
struct aie_vector_scalar_kernel {
  static const std::filesystem::path pdiPath;
  static const std::filesystem::path instsPath;

  // Number of elements in the input and output buffers for the vector-scalar add kernel.
  static constexpr std::size_t element_count = 1024;
  static constexpr std::size_t element_bytes = element_count * sizeof(std::uint32_t);

  // Number of kernargs
  static constexpr std::size_t num_kernargs = 2;
  // Number of kernarg sizes (for this kernel, all kernargs are pointer+size pairs)
  static constexpr std::size_t num_kernarg_sizes = num_kernargs;
  // Kernargs and sizes
  static constexpr std::size_t num_kernargs_sizes = num_kernargs + num_kernarg_sizes;
  // Buffer size for kernargs: 2 pointers + 2 sizes
  static constexpr std::size_t kernarg_bytes = num_kernargs_sizes * sizeof(uint64_t);

  /**
   * @brief Create a AIE packet payload for vector-scalar add.
   *
   * @param pdi_buf buffer containing the PDI for this packet
   * @param insts_buf buffer containing the instruction sequence for this packet
   * @param insts_size size of the instruction sequence in bytes
   * @param input source buffer for the packet
   * @param output destination buffer for the packet
   * @param kernargs pointer to the kernel arguments buffer
   * @param completion_signal signal to be used for completion notification
   * @param pkt_payload packet payload
   * @param q HSA queue to which the packet will be submitted
   */
  static std::uint64_t dispatch_packet(void* pdi_buf, void* insts_buf, std::uint32_t insts_size,
                                       void* input, void* output, uint64_t* kernargs,
                                       hsa_signal_t completion_signal, hsa_queue_t* q) {
    kernargs[0] = reinterpret_cast<uint64_t>(input);
    kernargs[1] = reinterpret_cast<uint64_t>(output);
    kernargs[2] = element_bytes;  // input size in bytes
    kernargs[3] = element_bytes;  // output size in bytes

    hsa_amd_aie_kernel_dispatch_packet_t pkt{};
    pkt.header = (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    pkt.opcode = HSA_AMD_AIE_PACKET_OPCODE_KMQ;
    pkt.count = 24;
    pkt.completion_signal = completion_signal;
    pkt.insts_addr_low = reinterpret_cast<std::uintptr_t>(insts_buf) & 0xFFFFFFFF;
    pkt.insts_addr_high = reinterpret_cast<std::uintptr_t>(insts_buf) >> 32;
    pkt.num_kernargs = num_kernargs;
    pkt.kernarg_address = kernargs;
    pkt.insts_size = insts_size;
    pkt.pdi_addr = pdi_buf;

    auto* queue = static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(q->base_address);

    const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
    while (wr_idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {
      // wait for available slot - if it hangs here, then the doorbell was not rung, or the packet
      // was not processed for some reason
    }

    const std::uint64_t pkt_idx = wr_idx % q->size;
    queue[pkt_idx] = pkt;

    return wr_idx;
  }
};
const std::filesystem::path aie_vector_scalar_kernel::pdiPath = STRINGIFY(DEFAULT_PDI_PATH);
const std::filesystem::path aie_vector_scalar_kernel::instsPath = STRINGIFY(DEFAULT_INSTS_PATH);

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
  ASSERT_TRUE(
      load_binary(dev_pool_data.pool, aie_vector_scalar_kernel::pdiPath, &pdi_buf, pdi_size));
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
  ASSERT_TRUE(
      load_binary(dev_pool_data.pool, aie_vector_scalar_kernel::instsPath, &insts_buf, insts_size));
  EXPECT_NE(insts_buf, nullptr);
  EXPECT_GT(insts_size, 0u);
  EXPECT_EQ(insts_size % sizeof(std::uint32_t), 0u);

  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

class DispatchTest : public ::testing::Test {
 protected:
  std::vector<hsa_agent_t> aie_agents;
  hsa_amd_memory_pool_t dev_pool{};
  hsa_amd_memory_pool_t data_pool{};
  hsa_amd_memory_pool_t kernarg_pool{};
  std::uint32_t min_queue_size = 0;

  void SetUp() override {
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

    // --- Discover AIE agent ---
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
    dev_pool = dev_pool_data.pool;

    // data pool: coarse-grained, allocatable (for tensor data)
    find_pool_data data_pool_data{};
    data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    data_pool_data.expected_allocatable = true;
    ASSERT_EQ(
        hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &data_pool_data),
        HSA_STATUS_INFO_BREAK);
    data_pool = data_pool_data.pool;

    // kernarg pool: KERNARG_INIT, allocatable; falls back to data pool
    find_pool_data kernarg_pool_data{};
    kernarg_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
    kernarg_pool_data.expected_allocatable = true;
    if (hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool,
                                           &kernarg_pool_data) != HSA_STATUS_INFO_BREAK) {
      kernarg_pool_data.pool = data_pool_data.pool;
    }
    kernarg_pool = kernarg_pool_data.pool;

    // --- Get queue size info ---
    ASSERT_EQ(
        hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
        HSA_STATUS_SUCCESS);
  }

  void TearDown() override { ASSERT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS); }
};

TEST_F(DispatchTest, SingleDispatch) {
  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load PDI and instructions ---
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::pdiPath, &pdi_buf, pdi_size));

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::instsPath, &insts_buf, insts_size));

  // --- Allocate I/O buffers ---
  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);

  std::iota(input, input + aie_vector_scalar_kernel::element_count, 0);
  std::fill_n(output, aie_vector_scalar_kernel::element_count, 0);

  // --- Create payload ---
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Dispatch packet ---
  const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
      pdi_buf, insts_buf, insts_size, input, output, kernargs, signal, queue);

  // --- Ring doorbell ---
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  // --- Wait for completion ---
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // --- Verify output: output[i] == input[i] + 1 ---
  for (std::size_t i = 0; i < aie_vector_scalar_kernel::element_count; ++i) {
    EXPECT_EQ(output[i], static_cast<std::uint32_t>(i + 1)) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, SingleDispatchVMem) {
  // Same as SingleDispatch, but the I/O buffers and kernargs go through the vmem
  // API. PDI and instructions stay on plain pool allocation because they must
  // live in the dev heap, which is incompatible with the vmem reserve+map path
  // (see docs/bug-vmem-map-dev-heap.md). The vmem buffers are made accessible to
  // both the AIE agent (for execution) and the CPU agent (for filling inputs /
  // verifying outputs).
  std::vector<hsa_agent_t> cpu_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_CPU>, &cpu_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(cpu_agents.empty());

  std::vector<hsa_agent_t> access_agents;
  access_agents.insert(access_agents.end(), cpu_agents.begin(), cpu_agents.end());
  access_agents.insert(access_agents.end(), aie_agents.begin(), aie_agents.end());

  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load PDI and instructions ---
  // These must come from the dev pool, which is the XDNA dev heap. The dev heap
  // is incompatible with the vmem reserve+map path (see
  // docs/bug-vmem-map-dev-heap.md), so they use plain pool allocation while the
  // I/O buffers and kernargs below go through the vmem API. The vmem variant is
  // preserved behind `#if 0` for when the runtime/driver gains dev-heap vmem
  // support.
#if 0
  vmem_buffer pdi{};
  ASSERT_TRUE(load_binary_vmem(dev_pool, access_agents, aie_vector_scalar_kernel::pdiPath, &pdi));
  void* const pdi_buf = pdi.va;

  vmem_buffer insts{};
  ASSERT_TRUE(
      load_binary_vmem(dev_pool, access_agents, aie_vector_scalar_kernel::instsPath, &insts));
  void* const insts_buf = insts.va;
  const std::size_t insts_size = insts.size;
#else
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::pdiPath, &pdi_buf, pdi_size));

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::instsPath, &insts_buf, insts_size));
#endif

  // --- Allocate I/O buffers ---
  vmem_buffer input{};
  ASSERT_TRUE(
      vmem_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, access_agents, &input));

  vmem_buffer output{};
  ASSERT_TRUE(
      vmem_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, access_agents, &output));

  auto* input_data = static_cast<std::uint32_t*>(input.va);
  auto* output_data = static_cast<std::uint32_t*>(output.va);
  std::iota(input_data, input_data + aie_vector_scalar_kernel::element_count, 0);
  std::fill_n(output_data, aie_vector_scalar_kernel::element_count, 0);

  // --- Create payload ---
  vmem_buffer kernargs{};
  ASSERT_TRUE(
      vmem_allocate(data_pool, aie_vector_scalar_kernel::kernarg_bytes, access_agents, &kernargs));

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Dispatch packet ---
  const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
      pdi_buf, insts_buf, insts_size, input.va, output.va, static_cast<std::uint64_t*>(kernargs.va),
      signal, queue);

  // --- Ring doorbell ---
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  // --- Wait for completion ---
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // --- Verify output: output[i] == input[i] + 1 ---
  for (std::size_t i = 0; i < aie_vector_scalar_kernel::element_count; ++i) {
    EXPECT_EQ(output_data[i], static_cast<std::uint32_t>(i + 1)) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_TRUE(vmem_free(kernargs));
  EXPECT_TRUE(vmem_free(output));
  EXPECT_TRUE(vmem_free(input));
#if 0
  EXPECT_TRUE(vmem_free(insts));
  EXPECT_TRUE(vmem_free(pdi));
#else
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
#endif
}

TEST_F(DispatchTest, MultiDispatch) {
  const std::uint32_t total_num_dispatches = 100;

  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load PDI and instructions ---
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::pdiPath, &pdi_buf, pdi_size));

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::instsPath, &insts_buf, insts_size));

  // --- Allocate I/O buffers ---
  const auto total_element_count = aie_vector_scalar_kernel::element_count * total_num_dispatches;
  const auto total_element_size = aie_vector_scalar_kernel::element_bytes * total_num_dispatches;

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + total_element_count, 0);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::fill_n(output, total_element_count, 0);

  // --- Allocate storage for packet payload ---
  const auto total_kernargs_bytes = aie_vector_scalar_kernel::kernarg_bytes * total_num_dispatches;
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, total_kernargs_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Dispatch loop ---
  for (std::uint32_t iter = 0; iter < total_num_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto input_ptr = input + iter * aie_vector_scalar_kernel::element_count;
    auto output_ptr = output + iter * aie_vector_scalar_kernel::element_count;
    auto kernarg_ptr = kernargs + iter * aie_vector_scalar_kernel::num_kernargs_sizes;

    // Dispatch packet
    const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        pdi_buf, insts_buf, insts_size, input_ptr, output_ptr, kernarg_ptr, signal, queue);

    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  }

  // Wait for completion of all dispatches
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify output: output == input[i] + 1
  for (std::size_t i = 0; i < total_element_count; ++i) {
    EXPECT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, MultiDispatchAsync) {
  const std::uint32_t total_num_dispatches = 40;

  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load PDI and instructions ---
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::pdiPath, &pdi_buf, pdi_size));

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::instsPath, &insts_buf, insts_size));

  // --- Allocate I/O buffers ---
  const auto total_element_count = aie_vector_scalar_kernel::element_count * total_num_dispatches;
  const auto total_element_size = aie_vector_scalar_kernel::element_bytes * total_num_dispatches;

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + total_element_count, 0);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::fill_n(output, total_element_count, 0);

  // --- Allocate storage for packet payload ---
  const auto total_kernarg_bytes = aie_vector_scalar_kernel::kernarg_bytes * total_num_dispatches;
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, total_kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Dispatch loop ---
  std::uint64_t last_wr_idx = 0;
  for (std::uint32_t iter = 0; iter < total_num_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto input_ptr = input + iter * aie_vector_scalar_kernel::element_count;
    auto output_ptr = output + iter * aie_vector_scalar_kernel::element_count;
    auto kernarg_ptr = kernargs + iter * aie_vector_scalar_kernel::num_kernargs_sizes;

    // Dispatch packet
    last_wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        pdi_buf, insts_buf, insts_size, input_ptr, output_ptr, kernarg_ptr, signal, queue);
  }

  // Ring doorbell
  hsa_signal_store_screlease(queue->doorbell_signal, last_wr_idx);

  // Wait for completion of all dispatches
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify output: output == input[i] + 1
  for (std::size_t i = 0; i < total_element_count; ++i) {
    EXPECT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, MultiDispatchWrapAround) {
  const std::uint32_t initial_dispatches = min_queue_size / 2;
  const std::uint32_t num_dispatches = 10 * min_queue_size;
  const std::uint32_t total_num_dispatches = initial_dispatches + num_dispatches;

  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load PDI and instructions ---
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::pdiPath, &pdi_buf, pdi_size));

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::instsPath, &insts_buf, insts_size));

  // --- Allocate I/O buffers ---
  const auto total_element_count = aie_vector_scalar_kernel::element_count * total_num_dispatches;
  const auto total_element_size = aie_vector_scalar_kernel::element_bytes * total_num_dispatches;

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + total_element_count, 0);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::fill_n(output, total_element_count, 0);

  // --- Allocate storage for kernel arguments ---
  const auto total_kernarg_bytes = aie_vector_scalar_kernel::kernarg_bytes * total_num_dispatches;
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, total_kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Initial Packets to set-up for wrap-around ---
  for (std::uint32_t iter = 0; iter < initial_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto input_ptr = input + iter * aie_vector_scalar_kernel::element_count;
    auto output_ptr = output + iter * aie_vector_scalar_kernel::element_count;
    auto kernarg_ptr = kernargs + iter * aie_vector_scalar_kernel::num_kernargs_sizes;

    // Dispatch packet
    const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        pdi_buf, insts_buf, insts_size, input_ptr, output_ptr, kernarg_ptr, signal, queue);

    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  }

  // --- Dispatch loop ---
  for (std::uint32_t iter = 0; iter < num_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto offset = initial_dispatches + iter;
    auto input_ptr = input + offset * aie_vector_scalar_kernel::element_count;
    auto output_ptr = output + offset * aie_vector_scalar_kernel::element_count;
    auto kernarg_ptr = kernargs + offset * aie_vector_scalar_kernel::num_kernargs_sizes;

    // Dispatch packet
    const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        pdi_buf, insts_buf, insts_size, input_ptr, output_ptr, kernarg_ptr, signal, queue);
    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  }

  // Wait for completion of all dispatches
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify output: output[i] == input[i] + 1
  for (std::size_t i = 0; i < total_element_count; ++i) {
    EXPECT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, MultiDispatchWrapAroundAsync) {
  const std::uint32_t initial_dispatches = min_queue_size - 1;
  const std::uint32_t num_dispatches = 40;
  const std::uint32_t total_num_dispatches = initial_dispatches + num_dispatches;

  // --- Create queue ---
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  // --- Load PDI and instructions ---
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::pdiPath, &pdi_buf, pdi_size));

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::instsPath, &insts_buf, insts_size));

  // --- Allocate I/O buffers ---
  const auto total_element_count = aie_vector_scalar_kernel::element_count * total_num_dispatches;
  const auto total_element_size = aie_vector_scalar_kernel::element_bytes * total_num_dispatches;

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + total_element_count, 0);

  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::fill_n(output, total_element_count, 0);

  // --- Allocate storage for kernel arguments ---
  const auto total_kernarg_bytes = aie_vector_scalar_kernel::kernarg_bytes * total_num_dispatches;
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, total_kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  // --- Create completion signal ---
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Initial Packets to set-up for wrap-around ---
  for (std::uint32_t iter = 0; iter < initial_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto input_ptr = input + iter * aie_vector_scalar_kernel::element_count;
    auto output_ptr = output + iter * aie_vector_scalar_kernel::element_count;
    auto kernarg_ptr = kernargs + iter * aie_vector_scalar_kernel::num_kernargs_sizes;

    // Dispatch packet
    const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        pdi_buf, insts_buf, insts_size, input_ptr, output_ptr, kernarg_ptr, signal, queue);

    // Ring doorbell
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  }

  // --- Dispatch loop ---
  std::uint64_t last_wr_idx = 0;
  for (std::uint32_t iter = 0; iter < num_dispatches; ++iter) {
    SCOPED_TRACE(iter);

    auto offset = initial_dispatches + iter;
    auto input_ptr = input + offset * aie_vector_scalar_kernel::element_count;
    auto output_ptr = output + offset * aie_vector_scalar_kernel::element_count;
    auto kernarg_ptr = kernargs + offset * aie_vector_scalar_kernel::num_kernargs_sizes;

    // Dispatch packet
    last_wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        pdi_buf, insts_buf, insts_size, input_ptr, output_ptr, kernarg_ptr, signal, queue);
  }

  // Ring doorbell
  hsa_signal_store_screlease(queue->doorbell_signal, last_wr_idx);

  // Wait for completion of all dispatches
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify output: output[i] == input[i] + 1
  for (std::size_t i = 0; i < total_element_count; ++i) {
    EXPECT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
}
