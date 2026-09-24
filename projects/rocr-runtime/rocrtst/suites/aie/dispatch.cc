/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "aie_full_elf.h"
#include "aie_test_env.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_amd_aie.h"

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)

namespace {

using namespace aie_test;  // NOLINT -- test translation unit

// ---------------------------------------------------------------------------
// Agent discovery
// ---------------------------------------------------------------------------

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
//
// Note this keys on RUNTIME_ALLOC_REC_GRANULE where memory.cc's equivalent keys on
// RUNTIME_ALLOC_GRANULE. The two very likely select the same pool, but that has not been
// established, so the two files keep their own predicate rather than sharing one.
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

// Header fields every AIE dispatch packet carries, whatever kernel it names. The per-kernel
// helpers below differ only in the fields they fill in on top of this.
hsa_amd_aie_kernel_dispatch_packet_t make_aie_packet(hsa_signal_t completion_signal) {
  hsa_amd_aie_kernel_dispatch_packet_t pkt{};
  pkt.header = (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  pkt.opcode = HSA_AMD_AIE_PACKET_OPCODE_KMQ;
  pkt.count = 24;
  pkt.completion_signal = completion_signal;
  return pkt;
}

// Claim the next queue slot and write `pkt` into it, returning its write index. Does not ring the
// doorbell, so a caller can batch several packets into one command chain.
std::uint64_t enqueue_aie_packet(hsa_queue_t* q, const hsa_amd_aie_kernel_dispatch_packet_t& pkt) {
  auto* queue = static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(q->base_address);

  const std::uint64_t wr_idx = hsa_queue_add_write_index_relaxed(q, 1);
  while (wr_idx - hsa_queue_load_read_index_scacquire(q) >= q->size) {
    // wait for available slot - if it hangs here, then the doorbell was not rung, or the packet
    // was not processed for some reason
  }

  queue[wr_idx % q->size] = pkt;
  return wr_idx;
}

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

    auto pkt = make_aie_packet(completion_signal);
    pkt.insts_addr_low = reinterpret_cast<std::uintptr_t>(insts_buf) & 0xFFFFFFFF;
    pkt.insts_addr_high = reinterpret_cast<std::uintptr_t>(insts_buf) >> 32;
    pkt.num_kernargs = num_kernargs;
    pkt.kernarg_address = kernargs;
    pkt.insts_size = insts_size;
    pkt.pdi_addr = pdi_buf;

    return enqueue_aie_packet(q, pkt);
  }
};
const std::filesystem::path aie_vector_scalar_kernel::pdiPath = STRINGIFY(DEFAULT_PDI_PATH);
const std::filesystem::path aie_vector_scalar_kernel::instsPath = STRINGIFY(DEFAULT_INSTS_PATH);

// Compile-time constants and dispatch helper for the vector-scalar-mul AIE kernel: multiplies
// every element of a uint32 array by `scale`, in place.
//
// Deliberately unlike aie_vector_scalar_kernel in the two ways a test can observe: the result
// says which design ran, and the single in/out buffer means one kernarg instead of two, so the
// two kernels' commands are not the same shape either.
struct aie_vector_scalar_mul_kernel {
  static const std::filesystem::path pdiPath;
  static const std::filesystem::path instsPath;

  // Injected by the build from VSMUL_SCALE, the same value it passes to the design script.
  static constexpr std::uint32_t scale = MUL_SCALE;

  // Number of elements in the in/out buffer for the vector-scalar mul kernel.
  static constexpr std::size_t element_count = 1024;
  static constexpr std::size_t element_bytes = element_count * sizeof(std::uint32_t);

  // Number of kernargs: one buffer, read and written.
  static constexpr std::size_t num_kernargs = 1;
  // uint64_t slots one dispatch's kernargs occupy: every kernarg is an address plus a size, so
  // two per argument. Deliberately not spelled as a `num_kernargs` + `num_kernarg_sizes` pair --
  // those two names differ by one character and mistyping the stride silently halves it.
  static constexpr std::size_t num_kernargs_sizes = 2 * num_kernargs;
  // Buffer size for kernargs: 1 pointer + 1 size
  static constexpr std::size_t kernarg_bytes = num_kernargs_sizes * sizeof(uint64_t);

  /**
   * @brief Create an AIE packet payload for vector-scalar mul.
   *
   * @param pdi_buf buffer containing the PDI for this packet
   * @param insts_buf buffer containing the instruction sequence for this packet
   * @param insts_size size of the instruction sequence in bytes
   * @param inout buffer read and written by the packet
   * @param kernargs pointer to the kernel arguments buffer
   * @param completion_signal signal to be used for completion notification
   * @param q HSA queue to which the packet will be submitted
   */
  static std::uint64_t dispatch_packet(void* pdi_buf, void* insts_buf, std::uint32_t insts_size,
                                       void* inout, uint64_t* kernargs,
                                       hsa_signal_t completion_signal, hsa_queue_t* q) {
    kernargs[0] = reinterpret_cast<uint64_t>(inout);
    kernargs[1] = element_bytes;  // in/out size in bytes

    auto pkt = make_aie_packet(completion_signal);
    pkt.insts_addr_low = reinterpret_cast<std::uintptr_t>(insts_buf) & 0xFFFFFFFF;
    pkt.insts_addr_high = reinterpret_cast<std::uintptr_t>(insts_buf) >> 32;
    pkt.num_kernargs = num_kernargs;
    pkt.kernarg_address = kernargs;
    pkt.insts_size = insts_size;
    pkt.pdi_addr = pdi_buf;

    return enqueue_aie_packet(q, pkt);
  }
};
const std::filesystem::path aie_vector_scalar_mul_kernel::pdiPath = STRINGIFY(MUL_PDI_PATH);
const std::filesystem::path aie_vector_scalar_mul_kernel::instsPath = STRINGIFY(MUL_INSTS_PATH);

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

// The dispatch tests' environment. The runtime and the AIE agent come from the shared base in
// aie_test_env.h; the pools and the queue sizing below are specific to this binary.
class DispatchTest : public aie_test::AieTestBase {
 protected:
  hsa_amd_memory_pool_t dev_pool{};
  hsa_amd_memory_pool_t data_pool{};
  hsa_amd_memory_pool_t kernarg_pool{};
  std::uint32_t min_queue_size = 0;

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(AieTestBase::SetUp());
    ASSERT_FALSE(aie_agents.empty());

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

    ASSERT_EQ(
        hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
        HSA_STATUS_SUCCESS);
  }
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

TEST(Dispatch, AgentInfo) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  char name[64] = {};
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_NAME, name), HSA_STATUS_SUCCESS);
  EXPECT_GT(std::strlen(name), 0u);

  char product_name[64] = {};
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(),
                               static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_PRODUCT_NAME),
                               product_name),
            HSA_STATUS_SUCCESS);

  char vendor_name[64] = {};
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_VENDOR_NAME, vendor_name),
            HSA_STATUS_SUCCESS);
  EXPECT_STREQ(vendor_name, "AMD");

  // HSA_AMD_AGENT_INFO_UUID is documented (hsa_ext_amd.h) as an Ascii string with a
  // maximum of 21 chars including NUL. Query into an exactly 21-byte buffer flanked by
  // guard bytes to catch any write past it.
  struct {
    char guard_before[8];
    char uuid[21];
    char guard_after[8];
  } uuid_buf;
  std::memset(uuid_buf.guard_before, 0xAB, sizeof(uuid_buf.guard_before));
  std::memset(uuid_buf.uuid, 0xCD, sizeof(uuid_buf.uuid));
  std::memset(uuid_buf.guard_after, 0xAB, sizeof(uuid_buf.guard_after));

  char guard_expected[8];
  std::memset(guard_expected, 0xAB, sizeof(guard_expected));

  ASSERT_EQ(
      hsa_agent_get_info(aie_agents.front(), static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_UUID),
                         uuid_buf.uuid),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(std::memcmp(uuid_buf.guard_before, guard_expected, sizeof(guard_expected)), 0);
  EXPECT_EQ(std::memcmp(uuid_buf.guard_after, guard_expected, sizeof(guard_expected)), 0);
  EXPECT_STREQ(uuid_buf.uuid, "AIE-XX");

  hsa_agent_feature_t feature{};
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_FEATURE, &feature),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(feature, HSA_AGENT_FEATURE_AGENT_DISPATCH);

  hsa_queue_type32_t queue_type{};
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_TYPE, &queue_type),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(queue_type, HSA_QUEUE_TYPE_SINGLE);

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST(Dispatch, CreateDestroyMultipleQueues) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);

  std::vector<hsa_agent_t> aie_agents;
  ASSERT_EQ(hsa_iterate_agents(discover_agents<HSA_DEVICE_TYPE_AIE>, &aie_agents),
            HSA_STATUS_SUCCESS);
  ASSERT_FALSE(aie_agents.empty());

  std::uint32_t min_queue_size = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUE_MIN_SIZE, &min_queue_size),
            HSA_STATUS_SUCCESS);
  ASSERT_GT(min_queue_size, 0u);

  std::uint32_t max_queues = 0;
  ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_QUEUES_MAX, &max_queues),
            HSA_STATUS_SUCCESS);
  ASSERT_GT(max_queues, 0u);

  const std::uint32_t num_queues = std::min(max_queues, 4u);
  std::vector<hsa_queue_t*> queues(num_queues, nullptr);

  for (std::uint32_t i = 0; i < num_queues; ++i) {
    SCOPED_TRACE(i);
    ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                               nullptr, 0, 0, &queues[i]),
              HSA_STATUS_SUCCESS);
    ASSERT_NE(queues[i], nullptr);
  }

  for (std::uint32_t i = 0; i < num_queues; ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(hsa_queue_destroy(queues[i]), HSA_STATUS_SUCCESS);
  }

  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, DestroyQueueWithPendingPacket) {
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

  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // --- Dispatch and ring the doorbell, then tear the queue down without waiting ---
  const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
      pdi_buf, insts_buf, insts_size, input, output, kernargs, signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  // AieAqlQueue::Inactivate destroys the kernel-mode queue; it does not drain the ring,
  // so this races tear-down against the in-flight packet. The destroy must still return
  // cleanly and leave the hardware detached from the buffers freed below.
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);

  // --- Cleanup ---
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, ConcurrentQueuesIndependentExecution) {
  constexpr std::uint32_t num_queues = 2;
  constexpr std::uint32_t num_rounds = 20;

  // --- Create queues ---
  hsa_queue_t* queues[num_queues] = {};
  for (std::uint32_t q = 0; q < num_queues; ++q) {
    SCOPED_TRACE(q);
    ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                               nullptr, 0, 0, &queues[q]),
              HSA_STATUS_SUCCESS);
  }

  // --- Load PDI and instructions (shared by both queues) ---
  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::pdiPath, &pdi_buf, pdi_size));

  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::instsPath, &insts_buf, insts_size));

  // --- Per-queue I/O buffers, so a cross-queue bleed shows up as a wrong value ---
  const auto total_element_count = aie_vector_scalar_kernel::element_count * num_rounds;
  const auto total_element_size = aie_vector_scalar_kernel::element_bytes * num_rounds;
  const auto total_kernarg_bytes = aie_vector_scalar_kernel::kernarg_bytes * num_rounds;

  std::uint32_t* input[num_queues] = {};
  std::uint32_t* output[num_queues] = {};
  uint64_t* kernargs[num_queues] = {};
  hsa_signal_t signals[num_queues] = {};

  for (std::uint32_t q = 0; q < num_queues; ++q) {
    SCOPED_TRACE(q);
    ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                           reinterpret_cast<void**>(&input[q])),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, total_element_size, 0,
                                           reinterpret_cast<void**>(&output[q])),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, total_kernarg_bytes, 0,
                                           reinterpret_cast<void**>(&kernargs[q])),
              HSA_STATUS_SUCCESS);

    // Distinct value ranges per queue.
    std::iota(input[q], input[q] + total_element_count, q * total_element_count);
    std::fill_n(output[q], total_element_count, 0);

    ASSERT_EQ(hsa_signal_create(num_rounds, 0, nullptr, &signals[q]), HSA_STATUS_SUCCESS);
  }

  // --- Interleave submissions so both queues are in flight at the same time ---
  for (std::uint32_t iter = 0; iter < num_rounds; ++iter) {
    for (std::uint32_t q = 0; q < num_queues; ++q) {
      SCOPED_TRACE(testing::Message() << "queue " << q << " iter " << iter);

      auto* input_ptr = input[q] + iter * aie_vector_scalar_kernel::element_count;
      auto* output_ptr = output[q] + iter * aie_vector_scalar_kernel::element_count;
      auto* kernarg_ptr = kernargs[q] + iter * aie_vector_scalar_kernel::num_kernargs_sizes;

      const auto wr_idx =
          aie_vector_scalar_kernel::dispatch_packet(pdi_buf, insts_buf, insts_size, input_ptr,
                                                    output_ptr, kernarg_ptr, signals[q], queues[q]);
      hsa_signal_store_screlease(queues[q]->doorbell_signal, wr_idx);
    }
  }

  // --- Wait and verify each queue saw only its own dispatches ---
  for (std::uint32_t q = 0; q < num_queues; ++q) {
    SCOPED_TRACE(q);
    hsa_signal_wait_scacquire(signals[q], HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);
    for (std::size_t i = 0; i < total_element_count; ++i) {
      EXPECT_EQ(output[q][i], input[q][i] + 1) << "mismatch at index " << i;
    }
  }

  // --- Cleanup ---
  for (std::uint32_t q = 0; q < num_queues; ++q) {
    EXPECT_EQ(hsa_signal_destroy(signals[q]), HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_queue_destroy(queues[q]), HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_amd_memory_pool_free(kernargs[q]), HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_amd_memory_pool_free(output[q]), HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_amd_memory_pool_free(input[q]), HSA_STATUS_SUCCESS);
  }
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
}


// ===========================================================================
// Full-ELF dispatch
//
// The application, not the runtime, reads the ELF: it extracts the PDI and the
// control code, allocates both from the device pool, and patches its argument
// addresses into the control code. The packet then names those buffers. The one
// thing the application cannot compute is the PDI's device address, so it passes
// the offset of that patch site and the runtime writes it -- and that non-zero
// offset is also what tells the runtime this is a full-ELF dispatch.
//
// Supported on aie2p only.
// ===========================================================================

namespace {

// Owns a device-pool allocation so a test body can bail out with ASSERT_* without leaking.
class pool_buffer {
 public:
  pool_buffer() = default;
  // Takes ownership of a pointer already allocated from a pool.
  explicit pool_buffer(void* p) : ptr_(p) {}
  pool_buffer(const pool_buffer&) = delete;
  pool_buffer& operator=(const pool_buffer&) = delete;
  pool_buffer(pool_buffer&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  pool_buffer& operator=(pool_buffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  ~pool_buffer() { reset(); }

  // Releases anything already held: the class has value semantics, so a caller may reasonably
  // reuse one buffer, and overwriting ptr_ would leak the previous allocation silently.
  hsa_status_t allocate(hsa_amd_memory_pool_t pool, std::size_t size) {
    reset();
    return hsa_amd_memory_pool_allocate(pool, size, 0, &ptr_);
  }

  void reset() {
    if (ptr_ != nullptr) hsa_amd_memory_pool_free(ptr_);
    ptr_ = nullptr;
  }

  template <typename T> T* as() const { return static_cast<T*>(ptr_); }
  void* get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

// Load a file straight into a pool_buffer, so ownership never passes through a raw pointer the
// caller has to remember to release.
testing::AssertionResult load_binary(hsa_amd_memory_pool_t pool, const std::filesystem::path& path,
                                     pool_buffer* out, std::size_t& size_out) {
  void* p = nullptr;
  if (auto r = load_binary(pool, path, &p, size_out); !r) {
    // Leave nothing half-built: a caller that loads several artifacts in sequence should not see
    // a stale size paired with an empty buffer.
    out->reset();
    size_out = 0;
    return r;
  }
  *out = pool_buffer(p);
  return testing::AssertionSuccess();
}

// A PDI-path design's two artifacts, loaded into the device heap and owned for the caller's scope.
struct kernel_artifacts {
  pool_buffer pdi;
  pool_buffer insts;
  std::size_t pdi_size = 0;
  std::size_t insts_size = 0;

  testing::AssertionResult load(hsa_amd_memory_pool_t dev_pool,
                                const std::filesystem::path& pdi_path,
                                const std::filesystem::path& insts_path) {
    if (auto r = load_binary(dev_pool, pdi_path, &pdi, pdi_size); !r) return r;
    return load_binary(dev_pool, insts_path, &insts, insts_size);
  }

  // The add design, which most PDI-path tests use.
  testing::AssertionResult load_add(hsa_amd_memory_pool_t dev_pool) {
    return load(dev_pool, aie_vector_scalar_kernel::pdiPath, aie_vector_scalar_kernel::instsPath);
  }

  // The mul design, used by the interleaving tests.
  testing::AssertionResult load_mul(hsa_amd_memory_pool_t dev_pool) {
    return load(dev_pool, aie_vector_scalar_mul_kernel::pdiPath,
                aie_vector_scalar_mul_kernel::instsPath);
  }
};

// Post-dispatch checks for the two designs.
void VerifyAdd(const std::uint32_t* in, const std::uint32_t* out, std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    ASSERT_EQ(out[i], in[i] + 1) << "add mismatch at index " << i;
  }
}

// The mul kernel works in place, so there is no untouched input to compare against and the
// expected value has to be reconstructed from the element's position in the iota that seeded the
// buffer. `base_index` is that position for inout[0]; it has no default because a slice of a
// larger buffer that quietly assumed 0 would compare against another chunk's expectations.
void VerifyMul(const std::uint32_t* inout, std::size_t count, std::size_t base_index) {
  for (std::size_t i = 0; i < count; ++i) {
    ASSERT_EQ(inout[i],
              static_cast<std::uint32_t>(base_index + i) * aie_vector_scalar_mul_kernel::scale)
        << "mul mismatch at index " << (base_index + i);
  }
}

// Dispatch one add-kernel packet against `pdi`, wait for it, and check the whole output buffer.
// Both PDI-cache tests walk the cache one PDI at a time exactly like this; call it under
// ASSERT_NO_FATAL_FAILURE so a mismatch stops the caller too.
void DispatchAddAndVerify(hsa_queue_t* queue, void* pdi, const kernel_artifacts& artifacts,
                          std::uint32_t* in, std::uint32_t* out, std::uint64_t* kernargs) {
  std::fill_n(out, aie_vector_scalar_kernel::element_count, 0);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
      pdi, artifacts.insts.get(), artifacts.insts_size, in, out, kernargs, signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

  VerifyAdd(in, out, aie_vector_scalar_kernel::element_count);
}

// The mul counterpart. Seeds the in-place buffer itself, since the kernel overwrites what it read
// and a second run over a stale buffer would be checking the wrong expectations.
void DispatchMulAndVerify(hsa_queue_t* queue, const kernel_artifacts& artifacts,
                          std::uint32_t* inout, std::uint64_t* kernargs) {
  std::iota(inout, inout + aie_vector_scalar_mul_kernel::element_count, 0);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  const auto wr_idx = aie_vector_scalar_mul_kernel::dispatch_packet(
      artifacts.pdi.get(), artifacts.insts.get(), artifacts.insts_size, inout, kernargs, signal,
      queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

  VerifyMul(inout, aie_vector_scalar_mul_kernel::element_count, 0);
}

// The full-ELF build of the same vector-scalar-add kernel used above.
struct aie_full_elf_kernel {
  static const std::filesystem::path elfPath;
  static constexpr const char* kernel_name = DEFAULT_ELF_KERNEL_NAME;

  static constexpr std::size_t element_count = 1024;
  static constexpr std::size_t element_bytes = element_count * sizeof(std::uint32_t);

  static constexpr std::size_t num_kernargs = 2;
  static constexpr std::size_t num_kernargs_sizes = 2 * num_kernargs;
  static constexpr std::size_t kernarg_bytes = num_kernargs_sizes * sizeof(std::uint64_t);

  // The runtime requires a full-ELF control code to sit on a 16 KiB boundary in device memory;
  // see the pdi_patch_offset documentation in hsa_ext_amd_aie.h.
  static constexpr std::size_t ctrl_code_alignment = 16384;

  // Rounds strictly past `p` to the next control-code boundary, so the result is always at a
  // non-zero offset into the allocation even when the pool already returned an aligned pointer.
  static std::uint8_t* align_ctrl_code_past(std::uint8_t* p) {
    const auto addr = reinterpret_cast<std::uintptr_t>(p);
    return p + (ctrl_code_alignment - (addr % ctrl_code_alignment));
  }

  // Write a full-ELF packet into the next queue slot. Does not ring the doorbell,
  // so a caller can batch several packets into one command chain.
  //
  // `ctrl_code` must already hold this dispatch's patched control code; because the
  // arguments live in the control code rather than in the command, two dispatches
  // with different buffers need two control-code buffers.
  static std::uint64_t dispatch_packet(
      void* ctrl_code, std::size_t ctrl_code_size, void* pdi, std::uint64_t pdi_patch_offset,
      void* input, void* output, std::uint64_t* kernargs, hsa_signal_t completion_signal,
      hsa_queue_t* q,
      const std::function<void(hsa_amd_aie_kernel_dispatch_packet_t&)>& mutate = {}) {
    kernargs[0] = reinterpret_cast<std::uint64_t>(input);
    kernargs[1] = reinterpret_cast<std::uint64_t>(output);
    kernargs[2] = element_bytes;  // input size in bytes
    kernargs[3] = element_bytes;  // output size in bytes

    auto pkt = make_aie_packet(completion_signal);
    pkt.insts_addr_low = reinterpret_cast<std::uintptr_t>(ctrl_code) & 0xFFFFFFFF;
    pkt.insts_addr_high = reinterpret_cast<std::uintptr_t>(ctrl_code) >> 32;
    pkt.insts_size = ctrl_code_size;
    pkt.num_kernargs = num_kernargs;
    pkt.kernarg_address = kernargs;
    pkt.pdi_addr = pdi;
    pkt.pdi_patch_offset = pdi_patch_offset;

    // Applied last, so a test can corrupt exactly one field of a packet that is otherwise known
    // to be good.
    if (mutate) mutate(pkt);

    return enqueue_aie_packet(q, pkt);
  }
};
const std::filesystem::path aie_full_elf_kernel::elfPath = STRINGIFY(DEFAULT_ELF_PATH);

// The full-ELF build of the vector-scalar-mul kernel. Same shape as above, but one argument
// instead of two, since the design reads and writes a single buffer.
struct aie_full_elf_mul_kernel {
  static const std::filesystem::path elfPath;
  // Neither design names its aie.device or its runtime sequence, so aiecc gives both the same
  // default name.
  static constexpr const char* kernel_name = DEFAULT_ELF_KERNEL_NAME;

  static constexpr std::size_t element_count = 1024;
  static constexpr std::size_t element_bytes = element_count * sizeof(std::uint32_t);

  static constexpr std::size_t num_kernargs = 1;
  static constexpr std::size_t num_kernargs_sizes = 2 * num_kernargs;
  static constexpr std::size_t kernarg_bytes = num_kernargs_sizes * sizeof(std::uint64_t);

  static std::uint64_t dispatch_packet(void* ctrl_code, std::size_t ctrl_code_size, void* pdi,
                                       std::uint64_t pdi_patch_offset, void* inout,
                                       std::uint64_t* kernargs, hsa_signal_t completion_signal,
                                       hsa_queue_t* q) {
    kernargs[0] = reinterpret_cast<std::uint64_t>(inout);
    kernargs[1] = element_bytes;  // in/out size in bytes

    auto pkt = make_aie_packet(completion_signal);
    pkt.insts_addr_low = reinterpret_cast<std::uintptr_t>(ctrl_code) & 0xFFFFFFFF;
    pkt.insts_addr_high = reinterpret_cast<std::uintptr_t>(ctrl_code) >> 32;
    pkt.insts_size = ctrl_code_size;
    pkt.num_kernargs = num_kernargs;
    pkt.kernarg_address = kernargs;
    pkt.pdi_addr = pdi;
    pkt.pdi_patch_offset = pdi_patch_offset;

    return enqueue_aie_packet(q, pkt);
  }
};
const std::filesystem::path aie_full_elf_mul_kernel::elfPath = STRINGIFY(MUL_ELF_PATH);

// Rewrite the relocation at `index` in .rela.dyn to reference symbol `sym` with relocation type
// `type`. Lets a test build a malformed variant of the real ELF in memory -- the entries are
// fixed size, so this is an in-place edit that shifts nothing.
bool mutate_relocation(std::vector<std::uint8_t>& image, std::size_t index, std::uint32_t sym,
                       std::uint32_t type) {
  if (image.size() < sizeof(Elf32_Ehdr)) return false;
  Elf32_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  if (ehdr.e_shoff + ehdr.e_shnum * sizeof(Elf32_Shdr) > image.size()) return false;

  Elf32_Shdr shstrtab{};
  std::memcpy(&shstrtab, image.data() + ehdr.e_shoff + ehdr.e_shstrndx * sizeof(Elf32_Shdr),
              sizeof(shstrtab));

  for (std::uint32_t i = 0; i < ehdr.e_shnum; ++i) {
    Elf32_Shdr sh{};
    std::memcpy(&sh, image.data() + ehdr.e_shoff + i * sizeof(Elf32_Shdr), sizeof(sh));
    const auto* name =
        reinterpret_cast<const char*>(image.data() + shstrtab.sh_offset + sh.sh_name);
    if (std::strcmp(name, ".rela.dyn") != 0) continue;
    if ((index + 1) * sizeof(Elf32_Rela) > sh.sh_size) return false;

    const std::size_t entry = sh.sh_offset + index * sizeof(Elf32_Rela);
    const std::uint32_t r_info = (sym << 8) | (type & 0xFF);
    std::memcpy(image.data() + entry + offsetof(Elf32_Rela, r_info), &r_info, sizeof(r_info));
    return true;
  }
  return false;
}


}  // namespace

class FullElfDispatchTest : public DispatchTest {
 protected:
  aie_full_elf::Kernel kernel;
  // The PDI is fixed for the life of the kernel, so one copy serves every dispatch.
  pool_buffer pdi;

  void SetUp() override {
    DispatchTest::SetUp();
    if (::testing::Test::HasFatalFailure()) return;

    // Full-ELF dispatch is aie2p only; aie2 has neither the firmware command nor the
    // preemption support it is built on.
    char agent_name[64] = {};
    ASSERT_EQ(hsa_agent_get_info(aie_agents.front(), HSA_AGENT_INFO_NAME, agent_name),
              HSA_STATUS_SUCCESS);
    if (std::strcmp(agent_name, "aie2p") != 0) {
      GTEST_SKIP() << "full-ELF dispatch needs an aie2p agent, found '" << agent_name << "'";
    }

    // The build skips the full ELF when the toolchain cannot produce one.
    if (!std::filesystem::exists(aie_full_elf_kernel::elfPath)) {
      GTEST_SKIP() << "full ELF was not built: " << aie_full_elf_kernel::elfPath;
    }

    ASSERT_NO_THROW({
      auto kernels = aie_full_elf::ParseFile(aie_full_elf_kernel::elfPath.string());
      auto it = kernels.find(aie_full_elf_kernel::kernel_name);
      ASSERT_NE(it, kernels.end()) << "kernel not in ELF: " << aie_full_elf_kernel::kernel_name;
      kernel = std::move(it->second);
    });

    // PDI and control code are fetched by the NPU directly, so they have to live in the device
    // heap; the data buffers do not.
    ASSERT_TRUE(kernel.has_pdi_patch) << "full ELF has no PDI to load";
    ASSERT_EQ(pdi.allocate(dev_pool, kernel.pdi.size()), HSA_STATUS_SUCCESS);
    std::memcpy(pdi.get(), kernel.pdi.data(), kernel.pdi.size());
  }

  void TearDown() override {
    // Release before the base class shuts the runtime down: a pool_buffer member outlives
    // TearDown(), and freeing after hsa_shut_down() is a no-op whose error nothing can see.
    pdi.reset();
    DispatchTest::TearDown();
  }

  // Allocate a control-code buffer for `design` and fill it with `arg_addrs` patched in. Takes the
  // kernel rather than reading the fixture's, so a test carrying a second design patches it the
  // same way instead of open-coding allocate-then-write.
  testing::AssertionResult make_ctrl_code(const aie_full_elf::Kernel& design, pool_buffer* out,
                                          const std::vector<std::uint64_t>& arg_addrs) {
    if (out->allocate(dev_pool, design.ctrl_code.size()) != HSA_STATUS_SUCCESS) {
      return testing::AssertionFailure() << "failed to allocate control code";
    }
    try {
      aie_full_elf::WriteControlCode(design, out->get(), design.ctrl_code.size(), arg_addrs);
    } catch (const std::exception& e) {
      return testing::AssertionFailure() << "patching control code: " << e.what();
    }
    return testing::AssertionSuccess();
  }

  // The fixture's own design, which most full-ELF tests use.
  testing::AssertionResult make_ctrl_code(pool_buffer* out,
                                          const std::vector<std::uint64_t>& arg_addrs) {
    return make_ctrl_code(kernel, out, arg_addrs);
  }

  // Dispatch `num_dispatches` full-ELF packets with distinct buffer pairs as a
  // single command chain, and check every output. Each dispatch gets its own
  // control-code buffer, since that is where its arguments live.
  void RunChain(hsa_queue_t* queue, std::uint32_t num_dispatches) {
    const std::size_t total_elements = aie_full_elf_kernel::element_count * num_dispatches;

    pool_buffer input, output, kernargs;
    ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes * num_dispatches),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes * num_dispatches),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes * num_dispatches),
              HSA_STATUS_SUCCESS);

    auto* in = input.as<std::uint32_t>();
    auto* out = output.as<std::uint32_t>();
    std::iota(in, in + total_elements, 0);
    std::fill_n(out, total_elements, 0);

    // Each dispatch carries its arguments in its own control code.
    std::vector<pool_buffer> ctrl_codes(num_dispatches);
    for (std::uint32_t i = 0; i < num_dispatches; ++i) {
      SCOPED_TRACE(i);
      ASSERT_TRUE(make_ctrl_code(
          &ctrl_codes[i],
          {reinterpret_cast<std::uint64_t>(in + i * aie_full_elf_kernel::element_count),
           reinterpret_cast<std::uint64_t>(out + i * aie_full_elf_kernel::element_count)}));
    }

    hsa_signal_t signal{};
    ASSERT_EQ(hsa_signal_create(num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

    // Enqueue every packet before ringing the doorbell, so the runtime submits them
    // as one chain rather than one at a time.
    std::uint64_t wr_idx = 0;
    for (std::uint32_t i = 0; i < num_dispatches; ++i) {
      wr_idx = aie_full_elf_kernel::dispatch_packet(
          ctrl_codes[i].get(), kernel.ctrl_code.size(), pdi.get(), kernel.pdi_patch_offset,
          in + i * aie_full_elf_kernel::element_count, out + i * aie_full_elf_kernel::element_count,
          kernargs.as<std::uint64_t>() + i * aie_full_elf_kernel::num_kernargs_sizes, signal,
          queue);
    }
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);

    for (std::size_t i = 0; i < total_elements; ++i) {
      ASSERT_EQ(out[i], in[i] + 1) << "mismatch at index " << i;
    }

    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  }
};

TEST_F(FullElfDispatchTest, ElfParse) {
  // The vector-scalar-add design has one PDI and two buffer arguments.
  EXPECT_EQ(kernel.name, aie_full_elf_kernel::kernel_name);
  EXPECT_FALSE(kernel.pdi.empty());
  EXPECT_FALSE(kernel.ctrl_code.empty());
  EXPECT_TRUE(kernel.has_pdi_patch);
  EXPECT_EQ(kernel.num_args(), aie_full_elf_kernel::num_kernargs);
  // The PDI address is 8 bytes, so its patch site has to fit in the control code.
  EXPECT_LE(kernel.pdi_patch_offset + sizeof(std::uint64_t), kernel.ctrl_code.size());
}

TEST_F(FullElfDispatchTest, ElfParseRejectsGarbage) {
  std::vector<std::uint8_t> garbage(1024, 0xAB);
  EXPECT_THROW(aie_full_elf::Parse(garbage.data(), garbage.size()), std::runtime_error);

  // A truncated but otherwise valid ELF must not be walked off the end of.
  std::size_t size = 0;
  auto f = open_binary(aie_full_elf_kernel::elfPath, &size);
  ASSERT_TRUE(static_cast<bool>(f));
  std::vector<std::uint8_t> bytes(size);
  ASSERT_TRUE(read_exact(f, bytes.data(), size));
  EXPECT_THROW(aie_full_elf::Parse(bytes.data(), bytes.size() / 2), std::runtime_error);
}

TEST_F(FullElfDispatchTest, ElfParseRejectsSecondPdiPatchSite) {
  // A Kernel carries one PDI patch offset. If an ELF asked for two, keeping only one would leave
  // the other load_pdi pointing at a placeholder and the dispatch would run against a bogus PDI
  // address -- so the reader has to refuse rather than pick one. Turn the first argument
  // relocation into a second PDI relocation to provoke it: symbol 1 is .pdi.1 and type 8 is
  // address_64, matching the relocation the ELF already has at index 0.
  std::size_t size = 0;
  auto f = open_binary(aie_full_elf_kernel::elfPath, &size);
  ASSERT_TRUE(static_cast<bool>(f));
  std::vector<std::uint8_t> image(size);
  ASSERT_TRUE(read_exact(f, image.data(), size));

  // Unmodified, it parses.
  ASSERT_NO_THROW(aie_full_elf::Parse(image.data(), image.size()));

  ASSERT_TRUE(mutate_relocation(image, 1, /*sym=*/1, /*type=*/8));
  // Assert on the reason, not just that something threw: this ELF encodes the patch scheme in
  // r_info, but an ABI-version-1 ELF encodes it in the addend, where rewriting r_info alone would
  // leave the scheme unchanged and trip a different check.
  try {
    aie_full_elf::Parse(image.data(), image.size());
    FAIL() << "a second PDI patch site was accepted";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("more than one PDI patch site"), std::string::npos)
        << "rejected for the wrong reason: " << e.what();
  }
}

TEST_F(FullElfDispatchTest, ElfWriteControlCodeChecksItsInputs) {
  // The arguments live in the control code rather than in the packet, so nothing downstream can
  // notice a short argument list -- the dispatch would run against whatever the ELF's
  // placeholder happened to be. The check has to happen here.
  std::vector<std::uint8_t> scratch(kernel.ctrl_code.size());
  const std::vector<std::uint64_t> args(kernel.num_args(), 0x1000);
  ASSERT_GT(kernel.num_args(), 0u);

  EXPECT_NO_THROW(aie_full_elf::WriteControlCode(kernel, scratch.data(), scratch.size(), args));

  const std::vector<std::uint64_t> too_few(kernel.num_args() - 1, 0x1000);
  EXPECT_THROW(aie_full_elf::WriteControlCode(kernel, scratch.data(), scratch.size(), too_few),
               std::runtime_error);

  // Likewise a destination that cannot hold the control code.
  EXPECT_THROW(aie_full_elf::WriteControlCode(kernel, scratch.data(), scratch.size() - 1, args),
               std::runtime_error);
}

TEST_F(FullElfDispatchTest, ElfSingleDispatch) {
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  pool_buffer input, output, kernargs;
  ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);

  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  std::iota(in, in + aie_full_elf_kernel::element_count, 0);
  std::fill_n(out, aie_full_elf_kernel::element_count, 0);

  pool_buffer ctrl_code;
  ASSERT_TRUE(make_ctrl_code(
      &ctrl_code, {reinterpret_cast<std::uint64_t>(in), reinterpret_cast<std::uint64_t>(out)}));

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
      ctrl_code.get(), kernel.ctrl_code.size(), pdi.get(), kernel.pdi_patch_offset, in, out,
      kernargs.as<std::uint64_t>(), signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  for (std::size_t i = 0; i < aie_full_elf_kernel::element_count; ++i) {
    ASSERT_EQ(out[i], static_cast<std::uint32_t>(i + 1)) << "mismatch at index " << i;
  }

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, ElfFullQueueDispatch) {
  // A chain as long as the queue allows. A full-ELF command occupies 60 bytes of
  // the driver's 4 KiB chain buffer, so 68 would fit; the queue tops out first.
  // A one-packet queue would leave the chain path untested rather than failing.
  ASSERT_GE(min_queue_size, 2u) << "queue too small to batch a chain";

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  RunChain(queue, min_queue_size);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, ElfRepatch) {
  // Reusing one control-code buffer across dispatches with different arguments has
  // to give the right answer every time. The shim-DMA patch scheme adds to the
  // buffer descriptor already in the control code, so patching over the previous
  // result instead of over a pristine copy gets the second dispatch wrong while
  // the first still looks fine.
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  pool_buffer ctrl_code;
  ASSERT_EQ(ctrl_code.allocate(dev_pool, kernel.ctrl_code.size()), HSA_STATUS_SUCCESS);

  constexpr std::uint32_t rounds = 4;
  for (std::uint32_t round = 0; round < rounds; ++round) {
    SCOPED_TRACE(round);

    // Fresh buffers each round, so every dispatch patches a different address into
    // the same control-code buffer.
    pool_buffer input, output, kernargs;
    ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
    ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
    ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes),
              HSA_STATUS_SUCCESS);

    auto* in = input.as<std::uint32_t>();
    auto* out = output.as<std::uint32_t>();
    std::iota(in, in + aie_full_elf_kernel::element_count, round * 1000);
    std::fill_n(out, aie_full_elf_kernel::element_count, 0);

    ASSERT_NO_THROW(aie_full_elf::WriteControlCode(
        kernel, ctrl_code.get(), kernel.ctrl_code.size(),
        {reinterpret_cast<std::uint64_t>(in), reinterpret_cast<std::uint64_t>(out)}));

    hsa_signal_t signal{};
    ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

    const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
        ctrl_code.get(), kernel.ctrl_code.size(), pdi.get(), kernel.pdi_patch_offset, in, out,
        kernargs.as<std::uint64_t>(), signal, queue);
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);

    for (std::size_t i = 0; i < aie_full_elf_kernel::element_count; ++i) {
      ASSERT_EQ(out[i], in[i] + 1) << "mismatch at index " << i;
    }
    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  }

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, ElfBuffersAtAllocationOffset) {
  // Neither buffer has to sit at the start of its allocation, so the runtime has to add the
  // offset into the allocation when it derives their device addresses. Every other test passes
  // allocation bases, where that arithmetic is a no-op. The control code still has to land on a
  // 16 KiB boundary; the PDI is deliberately left unaligned to show it has no such requirement.
  constexpr std::size_t pdi_offset = 4096;

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  pool_buffer input, output, kernargs, ctrl_alloc, pdi_alloc;
  ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(ctrl_alloc.allocate(dev_pool,
                                aie_full_elf_kernel::ctrl_code_alignment + kernel.ctrl_code.size()),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(pdi_alloc.allocate(dev_pool, pdi_offset + kernel.pdi.size()), HSA_STATUS_SUCCESS);

  auto* ctrl_code = aie_full_elf_kernel::align_ctrl_code_past(ctrl_alloc.as<std::uint8_t>());
  ASSERT_GT(ctrl_code, ctrl_alloc.as<std::uint8_t>());
  auto* pdi_at_offset = pdi_alloc.as<std::uint8_t>() + pdi_offset;

  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  std::iota(in, in + aie_full_elf_kernel::element_count, 0);
  std::fill_n(out, aie_full_elf_kernel::element_count, 0);

  // Poison the bytes before each buffer: a runtime that resolved the allocation base instead of
  // the pointer it was given would point the hardware at this.
  std::fill_n(ctrl_alloc.as<std::uint8_t>(),
              static_cast<std::size_t>(ctrl_code - ctrl_alloc.as<std::uint8_t>()), 0xA5);
  std::fill_n(pdi_alloc.as<std::uint8_t>(), pdi_offset, 0xA5);

  std::memcpy(pdi_at_offset, kernel.pdi.data(), kernel.pdi.size());
  ASSERT_NO_THROW(aie_full_elf::WriteControlCode(
      kernel, ctrl_code, kernel.ctrl_code.size(),
      {reinterpret_cast<std::uint64_t>(in), reinterpret_cast<std::uint64_t>(out)}));

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
      ctrl_code, kernel.ctrl_code.size(), pdi_at_offset, kernel.pdi_patch_offset, in, out,
      kernargs.as<std::uint64_t>(), signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  for (std::size_t i = 0; i < aie_full_elf_kernel::element_count; ++i) {
    ASSERT_EQ(out[i], static_cast<std::uint32_t>(i + 1)) << "mismatch at index " << i;
  }

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, ElfNoPdiCeiling) {
  // The PDI + instruction sequence path can hold at most 32 distinct PDIs per
  // hardware context, because each one costs a compute-unit slot. Full-ELF loads
  // the PDI from the control code instead, so there is no such ceiling: use more
  // than 32 separate PDI copies on one queue.
  constexpr std::uint32_t num_pdis = 40;

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  std::vector<pool_buffer> pdis(num_pdis);
  for (std::uint32_t i = 0; i < num_pdis; ++i) {
    SCOPED_TRACE(i);
    ASSERT_EQ(pdis[i].allocate(dev_pool, kernel.pdi.size()), HSA_STATUS_SUCCESS);
    std::memcpy(pdis[i].get(), kernel.pdi.data(), kernel.pdi.size());
  }

  pool_buffer input, output, kernargs;
  ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  std::iota(in, in + aie_full_elf_kernel::element_count, 0);

  pool_buffer ctrl_code;
  ASSERT_TRUE(make_ctrl_code(
      &ctrl_code, {reinterpret_cast<std::uint64_t>(in), reinterpret_cast<std::uint64_t>(out)}));

  for (std::uint32_t i = 0; i < num_pdis; ++i) {
    SCOPED_TRACE(i);
    std::fill_n(out, aie_full_elf_kernel::element_count, 0);

    hsa_signal_t signal{};
    ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
    const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
        ctrl_code.get(), kernel.ctrl_code.size(), pdis[i].get(), kernel.pdi_patch_offset, in, out,
        kernargs.as<std::uint64_t>(), signal, queue);
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);
    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

    for (std::size_t e = 0; e < aie_full_elf_kernel::element_count; ++e) {
      ASSERT_EQ(out[e], in[e] + 1) << "PDI " << i << " mismatch at index " << e;
    }
  }

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// ===========================================================================
// Tests that a rejected dispatch is refused rather than executed.
//
// A rejected dispatch is reported through the queue's error callback and nowhere
// else. The doorbell store is void, so it cannot return a status; and the packet
// never reached the device, so its completion signal is never released. A blocking
// wait on that signal would therefore hang forever -- these tests must not wait on
// it, and instead assert the signal still holds its initial value.
//
// Submission is synchronous on the calling thread for KMQ queues, so the callback
// has already fired by the time the doorbell store returns.
//
// A queue that takes a rejected packet is suspended and submits nothing further, by
// design: the failed packets are deliberately left unconsumed rather than retired.
// So "the refusal did not poison anything" is checked on a *fresh* queue, never by
// reusing the suspended one.
// ===========================================================================

// Asserts the dispatch was refused: reported once through the error callback, and its completion
// signal left untouched because the packet never executed.
void ExpectRejected(const aie_test::dispatch_error& err, hsa_queue_t* queue, hsa_signal_t signal,
                    hsa_signal_value_t initial) {
  EXPECT_TRUE(err.invoked.load(std::memory_order_acquire))
      << "a rejected dispatch did not report through the queue error callback";
  EXPECT_NE(err.status, HSA_STATUS_SUCCESS);
  EXPECT_EQ(err.source, queue);
  EXPECT_EQ(hsa_signal_load_scacquire(signal), initial)
      << "completion signal fired for a packet that never executed";
  // The refused packet is still in the ring: only packets that ran are consumed, so the read
  // index cannot have caught up with the write index.
  EXPECT_LT(hsa_queue_load_read_index_scacquire(queue), hsa_queue_load_write_index_scacquire(queue))
      << "a refused packet was consumed from the ring";
}

TEST_F(FullElfDispatchTest, ElfMisalignedControlCodeRejected) {
  // A control code that is not 16 KiB aligned is accepted by the hardware and then never
  // completes, so the runtime rejects it up front rather than letting the dispatch hang.
  dispatch_error err;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
            HSA_STATUS_SUCCESS);

  pool_buffer input, output, kernargs, ctrl_alloc, pdi;
  ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(ctrl_alloc.allocate(
                dev_pool, 2 * aie_full_elf_kernel::ctrl_code_alignment + kernel.ctrl_code.size()),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(pdi.allocate(dev_pool, kernel.pdi.size()), HSA_STATUS_SUCCESS);

  // Deliberately one page past a 16 KiB boundary.
  auto* ctrl_code = aie_full_elf_kernel::align_ctrl_code_past(ctrl_alloc.as<std::uint8_t>()) + 4096;

  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  std::iota(in, in + aie_full_elf_kernel::element_count, 0);
  constexpr std::uint32_t sentinel = 0xD0D0D0D0;
  std::fill_n(out, aie_full_elf_kernel::element_count, sentinel);
  std::memcpy(pdi.get(), kernel.pdi.data(), kernel.pdi.size());
  ASSERT_NO_THROW(aie_full_elf::WriteControlCode(
      kernel, ctrl_code, kernel.ctrl_code.size(),
      {reinterpret_cast<std::uint64_t>(in), reinterpret_cast<std::uint64_t>(out)}));

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
      ctrl_code, kernel.ctrl_code.size(), pdi.as<std::uint8_t>(), kernel.pdi_patch_offset, in, out,
      kernargs.as<std::uint64_t>(), signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  ExpectRejected(err, queue, signal, 1);

  for (std::size_t i = 0; i < aie_full_elf_kernel::element_count; ++i) {
    ASSERT_EQ(out[i], sentinel) << "rejected dispatch wrote output at index " << i;
  }

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, ElfHostOnlyControlCodeRejected) {
  // The NPU fetches the control code directly, so it has to come from the device
  // pool. A host-only allocation has no device address and would not be reachable.
  dispatch_error err;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
            HSA_STATUS_SUCCESS);

  pool_buffer input, output, kernargs, ctrl_code;
  ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  // data_pool, not dev_pool: this is the mistake being tested.
  ASSERT_EQ(ctrl_code.allocate(data_pool, kernel.ctrl_code.size()), HSA_STATUS_SUCCESS);

  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  std::iota(in, in + aie_full_elf_kernel::element_count, 0);
  constexpr std::uint32_t sentinel = 0xD0D0D0D0;
  std::fill_n(out, aie_full_elf_kernel::element_count, sentinel);

  ASSERT_NO_THROW(aie_full_elf::WriteControlCode(
      kernel, ctrl_code.get(), kernel.ctrl_code.size(),
      {reinterpret_cast<std::uint64_t>(in), reinterpret_cast<std::uint64_t>(out)}));

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
      ctrl_code.get(), kernel.ctrl_code.size(), pdi.get(), kernel.pdi_patch_offset, in, out,
      kernargs.as<std::uint64_t>(), signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  ExpectRejected(err, queue, signal, 1);

  for (std::size_t i = 0; i < aie_full_elf_kernel::element_count; ++i) {
    ASSERT_EQ(out[i], sentinel) << "rejected dispatch wrote output at index " << i;
  }

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(FullElfDispatchTest, ElfMalformedPacketsRejected) {
  // Each case corrupts exactly one field of a packet that is otherwise known good, so a failure
  // points at one validation rather than at "the packet was bad somehow". Either way the output
  // buffer must come back untouched and the refusal must reach the error callback.
  //
  // Most cases are refused while the batch is being built, before anything is submitted. "PDI not
  // in device memory" is the exception: the buffer is a registered BO, so it resolves, and the
  // runtime builds a device address for it anyway -- the hardware cannot fetch it and the command
  // is only caught by the driver's ~6 s watchdog. That is a missing validation, not an intended
  // path; it is what makes this test slow.
  pool_buffer host_pdi;
  ASSERT_EQ(host_pdi.allocate(data_pool, kernel.pdi.size()), HSA_STATUS_SUCCESS);

  const std::size_t ctrl_code_size = kernel.ctrl_code.size();
  using packet_t = hsa_amd_aie_kernel_dispatch_packet_t;

  struct malformed_case {
    const char* name;
    std::function<void(packet_t&)> mutate;
  };
  const std::vector<malformed_case> cases = {
      {"null control code",
       [](packet_t& p) {
         p.insts_addr_low = 0;
         p.insts_addr_high = 0;
       }},
      {"zero insts_size", [](packet_t& p) { p.insts_size = 0; }},
      {"insts_size past the end of the allocation",
       [ctrl_code_size](packet_t& p) { p.insts_size = ctrl_code_size + 1024 * 1024; }},
      {"null PDI", [](packet_t& p) { p.pdi_addr = nullptr; }},
      // The NPU fetches the PDI itself, so a host-only allocation has no address it can use.
      {"PDI not in device memory", [&host_pdi](packet_t& p) { p.pdi_addr = host_pdi.get(); }},
      // The patch site is 8 bytes and has to lie wholly inside the control code.
      {"PDI patch site past the end of the control code",
       [ctrl_code_size](packet_t& p) { p.pdi_patch_offset = ctrl_code_size - 4; }},
      {"PDI patch site misaligned", [](packet_t& p) { p.pdi_patch_offset += 1; }},
      {"unrecognised opcode", [](packet_t& p) { p.opcode = 0xFF; }},
  };

  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);

    // A fresh queue per case: a rejected packet suspends its queue, and reusing one would let an
    // earlier case mask a later one.
    dispatch_error err;
    hsa_queue_t* queue = nullptr;
    ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
              HSA_STATUS_SUCCESS);

    pool_buffer input, output, kernargs, ctrl_code;
    ASSERT_EQ(input.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
    ASSERT_EQ(output.allocate(data_pool, aie_full_elf_kernel::element_bytes), HSA_STATUS_SUCCESS);
    ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes),
              HSA_STATUS_SUCCESS);

    auto* in = input.as<std::uint32_t>();
    auto* out = output.as<std::uint32_t>();
    std::iota(in, in + aie_full_elf_kernel::element_count, 0);
    constexpr std::uint32_t sentinel = 0xD0D0D0D0;
    std::fill_n(out, aie_full_elf_kernel::element_count, sentinel);

    ASSERT_TRUE(make_ctrl_code(
        &ctrl_code, {reinterpret_cast<std::uint64_t>(in), reinterpret_cast<std::uint64_t>(out)}));

    hsa_signal_t signal{};
    ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
    const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
        ctrl_code.get(), ctrl_code_size, pdi.get(), kernel.pdi_patch_offset, in, out,
        kernargs.as<std::uint64_t>(), signal, queue, c.mutate);
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
    ExpectRejected(err, queue, signal, 1);

    for (std::size_t i = 0; i < aie_full_elf_kernel::element_count; ++i) {
      ASSERT_EQ(out[i], sentinel) << "rejected dispatch wrote output at index " << i;
    }

    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
    EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  }
}

TEST_F(DispatchTest, PdiCacheRolledBackOnFailedBatch) {
  // Building a command records its PDI in the queue's cache, but the hardware context is only
  // reconfigured to match after the whole batch is built. A batch that fails in between must not
  // leave the cache claiming compute units the context never got -- otherwise the next submission
  // finds the PDI "cached", skips the reconfigure, and dispatches against a context that cannot
  // run it. Submit a batch whose second packet is malformed, then check a fresh queue still works.
  dispatch_error err;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
            HSA_STATUS_SUCCESS);

  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::pdiPath, &pdi_buf, pdi_size));
  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::instsPath, &insts_buf, insts_size));

  std::uint32_t* input = nullptr;
  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, 0,
                                         reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool, aie_vector_scalar_kernel::element_bytes, 0,
                                         reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes, 0,
                                         reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + aie_vector_scalar_kernel::element_count, 0);
  std::fill_n(output, aie_vector_scalar_kernel::element_count, 0);

  // Two packets in one batch: the first introduces the PDI, the second is rejected because it
  // declares far more kernel arguments than its kernarg buffer holds.
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(2, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  aie_vector_scalar_kernel::dispatch_packet(pdi_buf, insts_buf, insts_size, input, output, kernargs,
                                            signal, queue);
  auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(pdi_buf, insts_buf, insts_size, input,
                                                          output, kernargs, signal, queue);
  auto* ring = static_cast<hsa_amd_aie_kernel_dispatch_packet_t*>(queue->base_address);
  ring[wr_idx % queue->size].num_kernargs = 2000;
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  // The batch is refused while it is still being built, so neither packet runs and the signal
  // keeps both of its counts.
  ExpectRejected(err, queue, signal, 2);
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

  // A well-formed dispatch on a fresh queue must still work: if the cache kept the rejected
  // batch's entry, this one skips the reconfigure and runs against an unconfigured context.
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  wr_idx = aie_vector_scalar_kernel::dispatch_packet(pdi_buf, insts_buf, insts_size, input, output,
                                                     kernargs, signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  for (std::size_t i = 0; i < aie_vector_scalar_kernel::element_count; ++i) {
    ASSERT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
}

// A kernel argument's declared size is not inert bookkeeping: the runtime clflushes
// [ptr, ptr + size) before the dispatch and again when the packet retires. A size larger than the
// argument's allocation would walk that flush off the end of the mapping and fault the host
// process, so the packet has to be refused while the batch is being built -- before either flush.
//
// Shared by both dispatch paths (AddKernargBOs), so the PDI path is enough to cover it and this
// test needs no Peano toolchain.
TEST_F(DispatchTest, KernargSizeExceedsBufferRejected) {
  dispatch_error err;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
            HSA_STATUS_SUCCESS);

  kernel_artifacts add;
  ASSERT_TRUE(add.load_add(dev_pool));

  pool_buffer input, output, kernargs;
  ASSERT_EQ(input.allocate(data_pool, aie_vector_scalar_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_vector_scalar_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  auto* args = kernargs.as<std::uint64_t>();
  std::iota(in, in + aie_vector_scalar_kernel::element_count, 0);

  // The rejected dispatch must not write, so the output keeps the sentinel.
  constexpr std::uint32_t sentinel = 0xD0D0D0D0;
  std::fill_n(out, aie_vector_scalar_kernel::element_count, sentinel);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
      add.pdi.get(), add.insts.get(), add.insts_size, in, out, args, signal, queue);

  // dispatch_packet fills the sizes, so overstate one of them afterwards. A gigabyte is far past
  // any rounding the pool may apply to the element_bytes request, so the rejection cannot be an
  // artifact of the allocation being larger than asked for.
  args[aie_vector_scalar_kernel::num_kernargs] = 1ULL << 30;

  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  ExpectRejected(err, queue, signal, 1);
  for (std::size_t i = 0; i < aie_vector_scalar_kernel::element_count; ++i) {
    ASSERT_EQ(out[i], sentinel) << "rejected dispatch wrote output at index " << i;
  }

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// The ceiling DispatchTest.PdiCacheHoldsThirtyTwo walks up to, seen from the other side. A
// 33rd distinct PDI has no compute-unit slot left -- the CU mask is 32 bits -- so the runtime
// refuses the packet rather than dispatching it against a context that cannot select it.
//
// The refusal happens while the batch is still being built, before anything is submitted, so
// nothing runs and the output stays as the test left it.
TEST_F(DispatchTest, PdiCacheRejectsThirtyThree) {
  constexpr std::uint32_t cache_capacity = 32;

  dispatch_error err;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
            HSA_STATUS_SUCCESS);

  kernel_artifacts add;
  ASSERT_TRUE(add.load_add(dev_pool));

  // One more than the cache holds. Identical bytes, distinct BOs: the cache keys on the handle.
  std::vector<pool_buffer> pdis(cache_capacity + 1);
  for (std::uint32_t i = 0; i < pdis.size(); ++i) {
    SCOPED_TRACE(i);
    ASSERT_EQ(pdis[i].allocate(dev_pool, add.pdi_size), HSA_STATUS_SUCCESS);
    std::memcpy(pdis[i].get(), add.pdi.get(), add.pdi_size);
  }

  pool_buffer input, output, kernargs;
  ASSERT_EQ(input.allocate(data_pool, aie_vector_scalar_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_vector_scalar_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  auto* args = kernargs.as<std::uint64_t>();
  std::iota(in, in + aie_vector_scalar_kernel::element_count, 0);

  // Fill the cache. Each of these has to succeed, or the test is not measuring what it claims.
  for (std::uint32_t i = 0; i < cache_capacity; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(DispatchAddAndVerify(queue, pdis[i].get(), add, in, out, args));
  }

  // The 33rd: rejected, so the output keeps the sentinel.
  constexpr std::uint32_t sentinel = 0xD0D0D0D0;
  std::fill_n(out, aie_vector_scalar_kernel::element_count, sentinel);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(1, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  const auto wr_idx = aie_vector_scalar_kernel::dispatch_packet(
      pdis[cache_capacity].get(), add.insts.get(), add.insts_size, in, out, args, signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  ExpectRejected(err, queue, signal, 1);
  for (std::size_t e = 0; e < aie_vector_scalar_kernel::element_count; ++e) {
    ASSERT_EQ(out[e], sentinel) << "rejected dispatch wrote output at index " << e;
  }
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

  // Hitting the ceiling refuses the packet that could not be placed; it does not take the device
  // or the runtime with it. The refused queue is suspended by design, so this is checked on a
  // fresh one -- which starts with an empty cache and so has room for the PDI again.
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NO_FATAL_FAILURE(DispatchAddAndVerify(queue, pdis[0].get(), add, in, out, args));

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// Grouping dispatches by mode is the caller's job. A batch whose packets do not agree is
// refused outright rather than split: splitting would mean rebuilding the hardware context with
// packets of the same batch still in flight, which the two modes' incompatible CU configurations
// make unsafe. Switching between *batches* is supported -- see ModeSwitchAlternatingBatches.
TEST_F(FullElfDispatchTest, MixedModeBatchRejected) {
  static_assert(aie_vector_scalar_kernel::element_count == aie_full_elf_kernel::element_count);
  constexpr std::size_t n = aie_vector_scalar_kernel::element_count;

  dispatch_error err;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_GE(queue->size, 2u);

  kernel_artifacts add;
  ASSERT_TRUE(add.load_add(dev_pool));

  pool_buffer input, output, pdi_kernargs, elf_kernargs, ctrl_code;
  ASSERT_EQ(input.allocate(data_pool, aie_vector_scalar_kernel::element_bytes * 2),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_vector_scalar_kernel::element_bytes * 2),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(pdi_kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(elf_kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);

  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  std::iota(in, in + n * 2, 0);
  constexpr std::uint32_t sentinel = 0xD0D0D0D0;
  std::fill_n(out, n * 2, sentinel);

  ASSERT_TRUE(make_ctrl_code(
      &ctrl_code,
      {reinterpret_cast<std::uint64_t>(in + n), reinterpret_cast<std::uint64_t>(out + n)}));

  // One PDI packet and one full-ELF packet staged together and released with a single doorbell.
  // The first fixes the batch's mode, the second contradicts it, so neither runs.
  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(2, 0, nullptr, &signal), HSA_STATUS_SUCCESS);
  aie_vector_scalar_kernel::dispatch_packet(add.pdi.get(), add.insts.get(), add.insts_size, in, out,
                                            pdi_kernargs.as<std::uint64_t>(), signal, queue);
  const auto wr_idx = aie_full_elf_kernel::dispatch_packet(
      ctrl_code.get(), kernel.ctrl_code.size(), pdi.get(), kernel.pdi_patch_offset, in + n, out + n,
      elf_kernargs.as<std::uint64_t>(), signal, queue);
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  // Neither packet runs, so the signal keeps both of its counts.
  ExpectRejected(err, queue, signal, 2);

  for (std::size_t i = 0; i < n * 2; ++i) {
    ASSERT_EQ(out[i], sentinel) << "rejected batch wrote output at index " << i;
  }
  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

  // The refusal does not take the device or the runtime with it: a well-formed batch still runs.
  // The refused queue is suspended by design, so this runs on a fresh one.
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_NO_FATAL_FAILURE(
      DispatchAddAndVerify(queue, add.pdi.get(), add, in, out, pdi_kernargs.as<std::uint64_t>()));

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// The PDI + instruction sequence path packs a wider chain slot than full-ELF, so
// its chains are shorter. The driver packs each command into a 4 KiB buffer, giving
// floor(4096 / (52 + 4 * arg_cnt)) commands per chain: 44 for this kernel's two
// arguments, against a 64-packet queue. The runtime splits an oversized batch
// across several chains rather than letting the driver reject it.
//
// Full-ELF needs no such split: its commands take 60 bytes, so 68 fit and the queue
// tops out first.
TEST_F(DispatchTest, PdiChainSplit) {
  const std::uint32_t total_num_dispatches = 64;

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_GE(min_queue_size, total_num_dispatches);

  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::pdiPath, &pdi_buf, pdi_size));
  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::instsPath, &insts_buf, insts_size));

  const auto total_element_count = aie_vector_scalar_kernel::element_count * total_num_dispatches;
  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(
                data_pool, aie_vector_scalar_kernel::element_bytes * total_num_dispatches, 0,
                reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(
                data_pool, aie_vector_scalar_kernel::element_bytes * total_num_dispatches, 0,
                reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  std::iota(input, input + total_element_count, 0);
  std::fill_n(output, total_element_count, 0);

  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(
                kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes * total_num_dispatches, 0,
                reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // One doorbell for the whole batch, so the runtime sees all of them at once.
  std::uint64_t wr_idx = 0;
  for (std::uint32_t i = 0; i < total_num_dispatches; ++i) {
    wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        pdi_buf, insts_buf, insts_size, input + i * aie_vector_scalar_kernel::element_count,
        output + i * aie_vector_scalar_kernel::element_count,
        kernargs + i * aie_vector_scalar_kernel::num_kernargs_sizes, signal, queue);
  }
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  for (std::size_t i = 0; i < total_element_count; ++i) {
    ASSERT_EQ(output[i], input[i] + 1) << "mismatch at index " << i;
  }

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
}

// A batch too large for one chain is split and submitted as several chains back to back. When a
// later chain fails, the packets in the chains before it have already run on the device and
// written their output, so their completion signals have to fire: the failure is reported once,
// through the queue callback, and a waiter on an earlier packet has no other way to learn that its
// own dispatch succeeded.
//
// The failure is induced with an instruction sequence in host memory. It resolves as a registered
// BO, so the batch builds normally, and unlike the PDI it takes no part in configuring the
// hardware context -- so the failure lands at the device, mid-batch, instead of before submission.
// The device has no address it can fetch those instructions from, and only the driver's ~6 s
// watchdog catches it, which is what makes this test slow.
//
// The last packet is the failing one, so the split point does not have to be hardcoded: whatever
// it is, at least one whole chain ran. The check that every retired packet also produced correct
// output is the load-bearing one -- it is what would catch the runtime crediting a packet that
// never executed.
TEST_F(DispatchTest, PartiallyFailedBatchRetiresCompletedPackets) {
  constexpr std::uint32_t total_num_dispatches = 64;
  constexpr std::uint32_t n = aie_vector_scalar_kernel::element_count;

  dispatch_error err;
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(create_queue_with_error_callback(aie_agents.front(), min_queue_size, &err, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_GE(min_queue_size, total_num_dispatches);

  void* pdi_buf = nullptr;
  std::size_t pdi_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::pdiPath, &pdi_buf, pdi_size));
  void* insts_buf = nullptr;
  std::size_t insts_size = 0;
  ASSERT_TRUE(load_binary(dev_pool, aie_vector_scalar_kernel::instsPath, &insts_buf, insts_size));
  // A device-resident instruction sequence the NPU cannot execute: right size and right pool, so
  // everything host-side accepts it, and the device faults on the contents.
  void* bad_insts = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(dev_pool, insts_size, 0, &bad_insts), HSA_STATUS_SUCCESS);
  std::memset(bad_insts, 0xFF, insts_size);

  std::uint32_t* input = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool,
                                         aie_vector_scalar_kernel::element_bytes *
                                             total_num_dispatches,
                                         0, reinterpret_cast<void**>(&input)),
            HSA_STATUS_SUCCESS);
  std::uint32_t* output = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(data_pool,
                                         aie_vector_scalar_kernel::element_bytes *
                                             total_num_dispatches,
                                         0, reinterpret_cast<void**>(&output)),
            HSA_STATUS_SUCCESS);
  uint64_t* kernargs = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(
                kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes * total_num_dispatches, 0,
                reinterpret_cast<void**>(&kernargs)),
            HSA_STATUS_SUCCESS);

  const std::size_t total_element_count = n * total_num_dispatches;
  std::iota(input, input + total_element_count, 0);
  constexpr std::uint32_t sentinel = 0xD0D0D0D0;
  std::fill_n(output, total_element_count, sentinel);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(total_num_dispatches, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // One doorbell for the whole batch, so the runtime splits it itself.
  std::uint64_t wr_idx = 0;
  for (std::uint32_t i = 0; i < total_num_dispatches; ++i) {
    void* insts = (i == total_num_dispatches - 1) ? bad_insts : insts_buf;
    wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        pdi_buf, insts, insts_size, input + i * n, output + i * n,
        kernargs + i * aie_vector_scalar_kernel::num_kernargs_sizes, signal, queue);
  }
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);

  EXPECT_TRUE(err.invoked.load(std::memory_order_acquire))
      << "a partially failed batch did not report through the queue error callback";
  EXPECT_NE(err.status, HSA_STATUS_SUCCESS);

  // Partial: the chains that ran are retired, the failing one is not.
  const hsa_signal_value_t remaining = hsa_signal_load_scacquire(signal);
  ASSERT_LT(remaining, total_num_dispatches)
      << "no packet was retired, so a waiter on a dispatch that did run would block forever";
  ASSERT_GT(remaining, 0) << "the failing packet was credited with a completion it never reached";

  const std::uint32_t retired = total_num_dispatches - static_cast<std::uint32_t>(remaining);

  // The ring agrees with the signal: the packets that ran are consumed, the failing one and
  // everything behind it are not. This queue started empty, so the read index is the count.
  EXPECT_EQ(hsa_queue_load_read_index_scacquire(queue), retired)
      << "read index does not match the packets that were completed";
  EXPECT_LT(hsa_queue_load_read_index_scacquire(queue), hsa_queue_load_write_index_scacquire(queue))
      << "the failing packet was consumed from the ring";

  for (std::uint32_t d = 0; d < retired; ++d) {
    SCOPED_TRACE(d);
    for (std::size_t e = 0; e < n; ++e) {
      ASSERT_EQ(output[d * n + e], input[d * n + e] + 1)
          << "retired a packet that did not run, at element " << e;
    }
  }
  // The failing dispatch never ran, so it wrote nothing.
  for (std::size_t e = 0; e < n; ++e) {
    ASSERT_EQ(output[(total_num_dispatches - 1) * n + e], sentinel)
        << "the failing dispatch wrote output at element " << e;
  }

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(kernargs), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(output), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(input), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(bad_insts), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(insts_buf), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_amd_memory_pool_free(pdi_buf), HSA_STATUS_SUCCESS);
}

// The PDI + instruction sequence path gives every distinct PDI a compute-unit slot in the queue's
// hardware context, and the CU mask driving it is 32 bits wide, so 32 distinct PDIs is the
// ceiling. Each new PDI tears the context down and rebuilds it with one more CU configured, so
// walking all the way to the ceiling checks that the rebuild keeps producing a context that can
// still run the kernel -- not just that the cache accepted the entry.
//
// Counterpart to FullElfDispatchTest.ElfNoPdiCeiling, which shows the full-ELF path has no such
// limit because it loads the PDI from the control code instead of spending a CU slot on it.
TEST_F(DispatchTest, PdiCacheHoldsThirtyTwo) {
  constexpr std::uint32_t num_pdis = 32;

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  kernel_artifacts add;
  ASSERT_TRUE(add.load_add(dev_pool));

  // Distinct BOs holding identical PDI bytes: the cache keys on the BO handle, so these count as
  // 32 separate PDIs even though the design is the same one every time.
  std::vector<pool_buffer> pdis(num_pdis);
  for (std::uint32_t i = 0; i < num_pdis; ++i) {
    SCOPED_TRACE(i);
    ASSERT_EQ(pdis[i].allocate(dev_pool, add.pdi_size), HSA_STATUS_SUCCESS);
    std::memcpy(pdis[i].get(), add.pdi.get(), add.pdi_size);
  }

  pool_buffer input, output, kernargs;
  ASSERT_EQ(input.allocate(data_pool, aie_vector_scalar_kernel::element_bytes), HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_vector_scalar_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  auto* args = kernargs.as<std::uint64_t>();
  std::iota(in, in + aie_vector_scalar_kernel::element_count, 0);

  // One dispatch per PDI, each waited on before the next: every iteration adds a cache entry and
  // forces a context rebuild, and the result shows the rebuilt context still runs the kernel.
  for (std::uint32_t i = 0; i < num_pdis; ++i) {
    SCOPED_TRACE(i);
    ASSERT_NO_FATAL_FAILURE(DispatchAddAndVerify(queue, pdis[i].get(), add, in, out, args));
  }

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

TEST_F(DispatchTest, MulSingleDispatch) {
  // The second kernel on its own. Without this, a failure in the interleaving tests below cannot
  // be told apart from the new design simply not working.
  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  kernel_artifacts mul;
  ASSERT_TRUE(mul.load_mul(dev_pool));

  pool_buffer inout, kernargs;
  ASSERT_EQ(inout.allocate(data_pool, aie_vector_scalar_mul_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(kernargs.allocate(kernarg_pool, aie_vector_scalar_mul_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);

  ASSERT_NO_FATAL_FAILURE(
      DispatchMulAndVerify(queue, mul, inout.as<std::uint32_t>(), kernargs.as<std::uint64_t>()));

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// Two different designs on one queue. Each holds a compute-unit slot in the shared hardware
// context and is selected per command by its CU mask, so getting this wrong shows up as one
// kernel's packets running the other's design -- which the differing results make visible.
//
// The two also carry different argument counts (two kernargs against one), so the commands are
// not even the same size in the chain the driver packs them into.
TEST_F(DispatchTest, InterleavedKernels) {
  constexpr std::uint32_t num_pairs = 4;
  // One stride `n` walks both designs' buffers below, which is only valid while they agree.
  static_assert(aie_vector_scalar_kernel::element_count ==
                aie_vector_scalar_mul_kernel::element_count);

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  // enqueue_aie_packet spins until a slot frees and the doorbell is only rung after the whole
  // batch is staged, so a ring smaller than the batch would hang rather than fail.
  ASSERT_GE(queue->size, 2 * num_pairs);

  kernel_artifacts add, mul;
  ASSERT_TRUE(add.load_add(dev_pool));
  ASSERT_TRUE(mul.load_mul(dev_pool));

  constexpr std::size_t n = aie_vector_scalar_kernel::element_count;
  pool_buffer add_in, add_out, add_kernargs, mul_inout, mul_kernargs;
  ASSERT_EQ(add_in.allocate(data_pool, aie_vector_scalar_kernel::element_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(add_out.allocate(data_pool, aie_vector_scalar_kernel::element_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(
      add_kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes * num_pairs),
      HSA_STATUS_SUCCESS);
  ASSERT_EQ(mul_inout.allocate(data_pool, aie_vector_scalar_mul_kernel::element_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(
      mul_kernargs.allocate(kernarg_pool, aie_vector_scalar_mul_kernel::kernarg_bytes * num_pairs),
      HSA_STATUS_SUCCESS);

  auto* ain = add_in.as<std::uint32_t>();
  auto* aout = add_out.as<std::uint32_t>();
  auto* mio = mul_inout.as<std::uint32_t>();
  std::iota(ain, ain + n * num_pairs, 0);
  std::fill_n(aout, n * num_pairs, 0);
  std::iota(mio, mio + n * num_pairs, 0);

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(2 * num_pairs, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  // Alternate the two designs in the ring and ring the doorbell once, so the runtime sees the
  // whole mixed batch at one time and packs it into one chain.
  std::uint64_t wr_idx = 0;
  for (std::uint32_t i = 0; i < num_pairs; ++i) {
    wr_idx = aie_vector_scalar_kernel::dispatch_packet(
        add.pdi.get(), add.insts.get(), add.insts_size, ain + i * n, aout + i * n,
        add_kernargs.as<std::uint64_t>() + i * aie_vector_scalar_kernel::num_kernargs_sizes, signal,
        queue);
    wr_idx = aie_vector_scalar_mul_kernel::dispatch_packet(
        mul.pdi.get(), mul.insts.get(), mul.insts_size, mio + i * n,
        mul_kernargs.as<std::uint64_t>() + i * aie_vector_scalar_mul_kernel::num_kernargs_sizes,
        signal, queue);
  }
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  ASSERT_NO_FATAL_FAILURE(VerifyAdd(ain, aout, n * num_pairs));
  ASSERT_NO_FATAL_FAILURE(VerifyMul(mio, n * num_pairs, 0));

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// The same two designs alternating across separate doorbells rather than in one chain. The first
// round introduces both PDIs and rebuilds the hardware context twice; every round after that
// finds them cached and must skip the rebuild, still selecting the right one per packet.
TEST_F(DispatchTest, InterleavedKernelsSeparateBatches) {
  constexpr std::uint32_t num_rounds = 4;
  static_assert(aie_vector_scalar_kernel::element_count ==
                aie_vector_scalar_mul_kernel::element_count);

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);

  kernel_artifacts add, mul;
  ASSERT_TRUE(add.load_add(dev_pool));
  ASSERT_TRUE(mul.load_mul(dev_pool));

  constexpr std::size_t n = aie_vector_scalar_kernel::element_count;
  pool_buffer add_in, add_out, add_kernargs, mul_inout, mul_kernargs;
  ASSERT_EQ(add_in.allocate(data_pool, aie_vector_scalar_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(add_out.allocate(data_pool, aie_vector_scalar_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(add_kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(mul_inout.allocate(data_pool, aie_vector_scalar_mul_kernel::element_bytes),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(mul_kernargs.allocate(kernarg_pool, aie_vector_scalar_mul_kernel::kernarg_bytes),
            HSA_STATUS_SUCCESS);

  auto* ain = add_in.as<std::uint32_t>();
  auto* aout = add_out.as<std::uint32_t>();
  auto* mio = mul_inout.as<std::uint32_t>();
  std::iota(ain, ain + n, 0);

  for (std::uint32_t round = 0; round < num_rounds; ++round) {
    SCOPED_TRACE(round);
    ASSERT_NO_FATAL_FAILURE(DispatchAddAndVerify(queue, add.pdi.get(), add, ain, aout,
                                                 add_kernargs.as<std::uint64_t>()));
    ASSERT_NO_FATAL_FAILURE(
        DispatchMulAndVerify(queue, mul, mio, mul_kernargs.as<std::uint64_t>()));
  }

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// Adds the second design's full ELF on top of FullElfDispatchTest's.
class FullElfInterleaveTest : public FullElfDispatchTest {
 protected:
  aie_full_elf::Kernel mul_kernel;
  pool_buffer mul_pdi;

  void SetUp() override {
    FullElfDispatchTest::SetUp();
    if (::testing::Test::HasFatalFailure() || ::testing::Test::IsSkipped()) return;

    if (!std::filesystem::exists(aie_full_elf_mul_kernel::elfPath)) {
      GTEST_SKIP() << "full ELF was not built: " << aie_full_elf_mul_kernel::elfPath;
    }
    // ParseFile throws on a malformed ELF; the lookup failure is a plain assertion. Keeping the
    // two separate matters -- an ASSERT_* inside the ASSERT_NO_THROW block would only return from
    // that block, and setup would carry on against an empty Kernel.
    std::map<std::string, aie_full_elf::Kernel> kernels;
    ASSERT_NO_THROW(kernels = aie_full_elf::ParseFile(aie_full_elf_mul_kernel::elfPath.string()));
    auto it = kernels.find(aie_full_elf_mul_kernel::kernel_name);
    ASSERT_NE(it, kernels.end()) << "kernel not in ELF: " << aie_full_elf_mul_kernel::kernel_name;
    mul_kernel = std::move(it->second);
    ASSERT_TRUE(mul_kernel.has_pdi_patch) << "full ELF has no PDI to load";
    ASSERT_EQ(mul_pdi.allocate(dev_pool, mul_kernel.pdi.size()), HSA_STATUS_SUCCESS);
    std::memcpy(mul_pdi.get(), mul_kernel.pdi.data(), mul_kernel.pdi.size());
  }

  void TearDown() override {
    // Same ordering reason as the base class: release before the runtime shuts down.
    mul_pdi.reset();
    FullElfDispatchTest::TearDown();
  }
};

// The full-ELF counterpart of DispatchTest.InterleavedKernels. This path has no PDI cache and no
// CU masks -- each packet points at its own control code, which carries both its arguments and
// the PDI it loads -- so what is being checked is the other half of interleaving: that alternating
// designs in one chain each run against their own control code and PDI rather than the previous
// packet's.
TEST_F(FullElfInterleaveTest, ElfInterleavedKernels) {
  constexpr std::uint32_t num_pairs = 4;
  constexpr std::size_t n = aie_full_elf_kernel::element_count;
  static_assert(aie_full_elf_kernel::element_count == aie_full_elf_mul_kernel::element_count);

  // The differing argument counts are what make the two designs distinguishable at the ABI level,
  // so assert it rather than assume the ELFs are what this test thinks they are.
  ASSERT_EQ(kernel.num_args(), aie_full_elf_kernel::num_kernargs);
  ASSERT_EQ(mul_kernel.num_args(), aie_full_elf_mul_kernel::num_kernargs);

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  // enqueue_aie_packet spins until a slot frees and the doorbell is only rung after the whole
  // batch is staged, so a ring smaller than the batch would hang rather than fail.
  ASSERT_GE(queue->size, 2 * num_pairs);

  pool_buffer add_in, add_out, add_kernargs, mul_inout, mul_kernargs;
  ASSERT_EQ(add_in.allocate(data_pool, aie_full_elf_kernel::element_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(add_out.allocate(data_pool, aie_full_elf_kernel::element_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(add_kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(mul_inout.allocate(data_pool, aie_full_elf_mul_kernel::element_bytes * num_pairs),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(mul_kernargs.allocate(kernarg_pool, aie_full_elf_mul_kernel::kernarg_bytes * num_pairs),
            HSA_STATUS_SUCCESS);

  auto* ain = add_in.as<std::uint32_t>();
  auto* aout = add_out.as<std::uint32_t>();
  auto* mio = mul_inout.as<std::uint32_t>();
  std::iota(ain, ain + n * num_pairs, 0);
  std::fill_n(aout, n * num_pairs, 0);
  std::iota(mio, mio + n * num_pairs, 0);

  // Arguments live in the control code, so every dispatch needs its own copy.
  std::vector<pool_buffer> add_ctrl(num_pairs), mul_ctrl(num_pairs);
  for (std::uint32_t i = 0; i < num_pairs; ++i) {
    SCOPED_TRACE(i);
    ASSERT_TRUE(make_ctrl_code(&add_ctrl[i],
                               {reinterpret_cast<std::uint64_t>(ain + i * n),
                                reinterpret_cast<std::uint64_t>(aout + i * n)}));
    ASSERT_TRUE(
        make_ctrl_code(mul_kernel, &mul_ctrl[i], {reinterpret_cast<std::uint64_t>(mio + i * n)}));
  }

  hsa_signal_t signal{};
  ASSERT_EQ(hsa_signal_create(2 * num_pairs, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

  std::uint64_t wr_idx = 0;
  for (std::uint32_t i = 0; i < num_pairs; ++i) {
    wr_idx = aie_full_elf_kernel::dispatch_packet(
        add_ctrl[i].get(), kernel.ctrl_code.size(), pdi.get(), kernel.pdi_patch_offset, ain + i * n,
        aout + i * n,
        add_kernargs.as<std::uint64_t>() + i * aie_full_elf_kernel::num_kernargs_sizes, signal,
        queue);
    wr_idx = aie_full_elf_mul_kernel::dispatch_packet(
        mul_ctrl[i].get(), mul_kernel.ctrl_code.size(), mul_pdi.get(), mul_kernel.pdi_patch_offset,
        mio + i * n,
        mul_kernargs.as<std::uint64_t>() + i * aie_full_elf_mul_kernel::num_kernargs_sizes, signal,
        queue);
  }
  hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
  hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  ASSERT_NO_FATAL_FAILURE(VerifyAdd(ain, aout, n * num_pairs));
  ASSERT_NO_FATAL_FAILURE(VerifyMul(mio, n * num_pairs, 0));

  EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);
  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}

// Modes alternate across batches, several packets at a time. Each batch is one mode, rung and
// waited on before the next is staged, so the queue is idle whenever the runtime rebuilds the
// hardware context for the incoming mode.
//
// Each batch carries several packets, so the switch is checked with a populated PDI cache to drop
// rather than with a single packet on either side.
TEST_F(FullElfDispatchTest, ModeSwitchAlternatingBatches) {
  static_assert(aie_vector_scalar_kernel::element_count == aie_full_elf_kernel::element_count);
  constexpr std::size_t n = aie_vector_scalar_kernel::element_count;
  constexpr std::uint32_t pkts_per_batch = 2;
  // true = PDI + instruction sequence, false = full ELF. Starts and ends on different modes so
  // both switch directions are exercised twice.
  constexpr bool batch_is_pdi[] = {true, false, true, false};

  hsa_queue_t* queue = nullptr;
  ASSERT_EQ(hsa_queue_create(aie_agents.front(), min_queue_size, HSA_QUEUE_TYPE_SINGLE, nullptr,
                             nullptr, 0, 0, &queue),
            HSA_STATUS_SUCCESS);
  ASSERT_GE(queue->size, pkts_per_batch);

  kernel_artifacts add;
  ASSERT_TRUE(add.load_add(dev_pool));

  pool_buffer input, output, pdi_kernargs, elf_kernargs;
  ASSERT_EQ(input.allocate(data_pool, aie_vector_scalar_kernel::element_bytes * pkts_per_batch),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(output.allocate(data_pool, aie_vector_scalar_kernel::element_bytes * pkts_per_batch),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(
      pdi_kernargs.allocate(kernarg_pool, aie_vector_scalar_kernel::kernarg_bytes * pkts_per_batch),
      HSA_STATUS_SUCCESS);
  ASSERT_EQ(
      elf_kernargs.allocate(kernarg_pool, aie_full_elf_kernel::kernarg_bytes * pkts_per_batch),
      HSA_STATUS_SUCCESS);

  auto* in = input.as<std::uint32_t>();
  auto* out = output.as<std::uint32_t>();
  std::iota(in, in + n * pkts_per_batch, 0);

  for (std::uint32_t b = 0; b < std::size(batch_is_pdi); ++b) {
    SCOPED_TRACE(b);
    std::fill_n(out, n * pkts_per_batch, 0);

    // A full-ELF dispatch carries its arguments in its control code, so each packet needs its own.
    std::vector<pool_buffer> ctrl_codes(pkts_per_batch);
    if (!batch_is_pdi[b]) {
      for (std::uint32_t i = 0; i < pkts_per_batch; ++i) {
        ASSERT_TRUE(make_ctrl_code(&ctrl_codes[i],
                                   {reinterpret_cast<std::uint64_t>(in + i * n),
                                    reinterpret_cast<std::uint64_t>(out + i * n)}));
      }
    }

    hsa_signal_t signal{};
    ASSERT_EQ(hsa_signal_create(pkts_per_batch, 0, nullptr, &signal), HSA_STATUS_SUCCESS);

    std::uint64_t wr_idx = 0;
    for (std::uint32_t i = 0; i < pkts_per_batch; ++i) {
      wr_idx = batch_is_pdi[b]
          ? aie_vector_scalar_kernel::dispatch_packet(
                add.pdi.get(), add.insts.get(), add.insts_size, in + i * n, out + i * n,
                pdi_kernargs.as<std::uint64_t>() + i * aie_vector_scalar_kernel::num_kernargs_sizes,
                signal, queue)
          : aie_full_elf_kernel::dispatch_packet(
                ctrl_codes[i].get(), kernel.ctrl_code.size(), pdi.get(), kernel.pdi_patch_offset,
                in + i * n, out + i * n,
                elf_kernargs.as<std::uint64_t>() + i * aie_full_elf_kernel::num_kernargs_sizes,
                signal, queue);
    }
    hsa_signal_store_screlease(queue->doorbell_signal, wr_idx);
    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);
    EXPECT_EQ(hsa_signal_destroy(signal), HSA_STATUS_SUCCESS);

    // Both designs add one, so a batch is checked the same way whichever mode ran it.
    ASSERT_NO_FATAL_FAILURE(VerifyAdd(in, out, n * pkts_per_batch));
  }

  EXPECT_EQ(hsa_queue_destroy(queue), HSA_STATUS_SUCCESS);
}
