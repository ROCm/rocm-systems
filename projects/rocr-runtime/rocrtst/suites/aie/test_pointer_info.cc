/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"

#include "aie_test_env.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace {

using namespace aie_test;  // NOLINT -- test translation unit

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

/**
 * @brief HSA memory-pool iteration callback that finds the first global pool matching a
 * find_pool_data.
 *
 * @param pool memory pool being visited
 * @param data find_pool_data* with the flags and allocatability to match; receives the pool
 * @return HSA_STATUS_INFO_BREAK on a match, HSA_STATUS_SUCCESS to keep iterating, or the error
 * from a failed pool query
 */
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
  if (allocatable != d.expected_allocatable) {
    return HSA_STATUS_SUCCESS;
  }

  d.pool = pool;
  return HSA_STATUS_INFO_BREAK;
}

}  // namespace

// The pointer-info tests' environment. The runtime and the AIE agent come from the shared base in
// aie_test_env.h; the pools below are specific to this binary.
class PointerInfoTest : public aie_test::AieTestBase {
 protected:
  hsa_amd_memory_pool_t dev_pool{};
  hsa_amd_memory_pool_t data_pool{};

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(AieTestBase::SetUp());
    ASSERT_FALSE(aie_agents.empty());

    // dev pool: coarse-grained, non-allocatable. aie4 (UMQ) has no such pool at all -- it has no
    // device heap, so there is nothing non-allocatable to find -- so this is left
    // default-constructed (has_dev_pool() false) instead of asserted; pointer_info_pool() falls
    // back to data_pool for those tests that need a device-memory pool but do not care which one.
    find_pool_data dev_pool_data{};
    dev_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    dev_pool_data.expected_allocatable = false;
    if (hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &dev_pool_data) ==
        HSA_STATUS_INFO_BREAK) {
      dev_pool = dev_pool_data.pool;
    }

    // data pool: coarse-grained, allocatable (for tensor data)
    find_pool_data data_pool_data{};
    data_pool_data.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED;
    data_pool_data.expected_allocatable = true;
    ASSERT_EQ(
        hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &data_pool_data),
        HSA_STATUS_INFO_BREAK);
    data_pool = data_pool_data.pool;
  }

  /**
   * @brief Whether this agent has a dev pool (see the comment on dev_pool's discovery in SetUp).
   *
   * @return true if dev_pool was found
   */
  bool has_dev_pool() const { return dev_pool.handle != 0; }

  /**
   * @brief A device-memory pool for tests that only care about pointer-info behavior on device
   * memory, not on the specific pool's own allocatable/non-allocatable flag.
   *
   * aie2p's dev_pool exercises the discoverable non-allocatable pool; aie4 (UMQ) has no such pool
   * (see dev_pool's discovery comment in SetUp), so these tests run against data_pool there
   * instead -- the same underlying device memory, from whichever pool this agent actually has.
   *
   * @return dev_pool if this agent has one, otherwise data_pool
   */
  hsa_amd_memory_pool_t pointer_info_pool() const { return has_dev_pool() ? dev_pool : data_pool; }
};

// ---------------------------------------------------------------------------
// hsa_amd_pointer_info on AIE allocations
// ---------------------------------------------------------------------------
// Applications patch agentBaseAddress into a full-ELF design's control code.

TEST_F(PointerInfoTest, ReportsDeviceAddress) {
  constexpr std::size_t kSize = 4096;
  void* ptr = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(pointer_info_pool(), kSize, 0, &ptr), HSA_STATUS_SUCCESS);

  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  EXPECT_EQ(hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr), HSA_STATUS_SUCCESS);

  EXPECT_EQ(info.type, HSA_EXT_POINTER_TYPE_HSA);
  EXPECT_EQ(info.hostBaseAddress, ptr);
  EXPECT_EQ(info.sizeInBytes, kSize);
  EXPECT_EQ(info.agentOwner.handle, aie_agents.front().handle);
  EXPECT_NE(info.agentBaseAddress, nullptr);
  if (has_dev_pool()) {
    // Dev-pool memory has a device address distinct from its host address.
    EXPECT_NE(info.agentBaseAddress, ptr);
  } else {
    // Measured directly on this pool: aie4's data pool has no DDR aperture offset, so the device
    // address is the host address itself (see XdnaDriver::TranslateUmqPackets's
    // PatchShimDma57Aie4 comment, which relies on the same measurement).
    EXPECT_EQ(info.agentBaseAddress, ptr);
  }

  EXPECT_EQ(hsa_amd_memory_pool_free(ptr), HSA_STATUS_SUCCESS);
}

TEST_F(PointerInfoTest, ReportsAccessibleAgent) {
  constexpr std::size_t kSize = 4096;
  void* ptr = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(pointer_info_pool(), kSize, 0, &ptr), HSA_STATUS_SUCCESS);

  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  std::uint32_t num_agents = 0;
  hsa_agent_t* agents = nullptr;
  EXPECT_EQ(hsa_amd_pointer_info(ptr, &info, std::malloc, &num_agents, &agents),
            HSA_STATUS_SUCCESS);

  // Only the owning agent can access the allocation.
  ASSERT_EQ(num_agents, 1u);
  ASSERT_NE(agents, nullptr);
  EXPECT_EQ(agents[0].handle, aie_agents.front().handle);
  std::free(agents);

  EXPECT_EQ(hsa_amd_memory_pool_free(ptr), HSA_STATUS_SUCCESS);
}

TEST_F(PointerInfoTest, ResolvesInteriorPointer) {
  constexpr std::size_t kSize = 4096;
  constexpr std::size_t kOffset = 256;
  void* ptr = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(pointer_info_pool(), kSize, 0, &ptr), HSA_STATUS_SUCCESS);

  hsa_amd_pointer_info_t base{};
  base.size = sizeof(base);
  hsa_amd_pointer_info_t inside{};
  inside.size = sizeof(inside);
  ASSERT_EQ(hsa_amd_pointer_info(ptr, &base, nullptr, nullptr, nullptr), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_pointer_info(static_cast<std::uint8_t*>(ptr) + kOffset, &inside, nullptr,
                                 nullptr, nullptr),
            HSA_STATUS_SUCCESS);

  // An interior pointer reports the allocation that contains it.
  EXPECT_EQ(inside.type, base.type);
  EXPECT_EQ(inside.hostBaseAddress, base.hostBaseAddress);
  EXPECT_EQ(inside.agentBaseAddress, base.agentBaseAddress);
  EXPECT_EQ(inside.sizeInBytes, base.sizeInBytes);

  EXPECT_EQ(hsa_amd_memory_pool_free(ptr), HSA_STATUS_SUCCESS);
}

TEST_F(PointerInfoTest, UnknownAfterFree) {
  constexpr std::size_t kSize = 4096;
  void* ptr = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(pointer_info_pool(), kSize, 0, &ptr), HSA_STATUS_SUCCESS);
  ASSERT_EQ(hsa_amd_memory_pool_free(ptr), HSA_STATUS_SUCCESS);

  // A freed allocation is unknown.
  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  EXPECT_EQ(hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr), HSA_STATUS_SUCCESS);
  EXPECT_EQ(info.type, HSA_EXT_POINTER_TYPE_UNKNOWN);

  // An output struct with no size is rejected.
  hsa_amd_pointer_info_t unsized{};
  EXPECT_EQ(hsa_amd_pointer_info(ptr, &unsized, nullptr, nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST_F(PointerInfoTest, UnknownForForeignPointers) {
  // None of these are HSA allocations.
  void* heap = std::malloc(4096);
  ASSERT_NE(heap, nullptr);
  int on_stack = 0;
  static int in_bss = 0;

  struct {
    const char* what;
    const void* ptr;
  } cases[] = {
      {"heap", heap},
      {"stack", &on_stack},
      {"bss", &in_bss},
      {"function", reinterpret_cast<const void*>(&std::malloc)},
      {"garbage", reinterpret_cast<const void*>(std::uintptr_t{1})},
  };

  for (const auto& c : cases) {
    hsa_amd_pointer_info_t info{};
    info.size = sizeof(info);
    EXPECT_EQ(hsa_amd_pointer_info(c.ptr, &info, nullptr, nullptr, nullptr), HSA_STATUS_SUCCESS)
        << c.what;
    EXPECT_EQ(info.type, HSA_EXT_POINTER_TYPE_UNKNOWN) << c.what;
    EXPECT_EQ(info.agentBaseAddress, nullptr) << c.what;
  }

  // A null pointer is rejected.
  hsa_amd_pointer_info_t null_info{};
  null_info.size = sizeof(null_info);
  EXPECT_EQ(hsa_amd_pointer_info(nullptr, &null_info, nullptr, nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);

  std::free(heap);
}

TEST_F(PointerInfoTest, UnknownJustPastAnAllocation) {
  constexpr std::size_t kSize = 4096;
  void* ptr = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(pointer_info_pool(), kSize, 0, &ptr), HSA_STATUS_SUCCESS);

  // One past the end is outside the allocation.
  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  EXPECT_EQ(hsa_amd_pointer_info(static_cast<std::uint8_t*>(ptr) + kSize, &info, nullptr, nullptr,
                                 nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(info.type, HSA_EXT_POINTER_TYPE_UNKNOWN);

  // The last byte is inside.
  hsa_amd_pointer_info_t last{};
  last.size = sizeof(last);
  EXPECT_EQ(hsa_amd_pointer_info(static_cast<std::uint8_t*>(ptr) + kSize - 1, &last, nullptr,
                                 nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(last.type, HSA_EXT_POINTER_TYPE_HSA);
  EXPECT_EQ(last.hostBaseAddress, ptr);

  EXPECT_EQ(hsa_amd_memory_pool_free(ptr), HSA_STATUS_SUCCESS);
}

TEST_F(PointerInfoTest, UnknownInTheAlignmentPadding) {
  // Allocations are rounded up to the region's granularity, and the padding past the requested
  // size is not part of the allocation. The size is chosen so that there is padding.
  constexpr std::size_t kRequested = 100;
  void* ptr = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(pointer_info_pool(), kRequested, 0, &ptr),
            HSA_STATUS_SUCCESS);

  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  ASSERT_EQ(hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr), HSA_STATUS_SUCCESS);
  ASSERT_EQ(info.type, HSA_EXT_POINTER_TYPE_HSA);
  // The reported size is the requested size.
  EXPECT_EQ(info.sizeInBytes, kRequested);

  // The last requested byte is inside.
  hsa_amd_pointer_info_t last{};
  last.size = sizeof(last);
  EXPECT_EQ(hsa_amd_pointer_info(static_cast<std::uint8_t*>(ptr) + kRequested - 1, &last, nullptr,
                                 nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(last.type, HSA_EXT_POINTER_TYPE_HSA);
  EXPECT_EQ(last.hostBaseAddress, ptr);

  // The first byte of padding is outside.
  hsa_amd_pointer_info_t padding{};
  padding.size = sizeof(padding);
  EXPECT_EQ(hsa_amd_pointer_info(static_cast<std::uint8_t*>(ptr) + kRequested, &padding, nullptr,
                                 nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(padding.type, HSA_EXT_POINTER_TYPE_UNKNOWN);

  EXPECT_EQ(hsa_amd_memory_pool_free(ptr), HSA_STATUS_SUCCESS);
}

TEST_F(PointerInfoTest, ReportsPoolFlags) {
  // The allocation reports the same global flags as its pool.
  constexpr std::size_t kSize = 4096;
  const hsa_amd_memory_pool_t pool = pointer_info_pool();
  void* ptr = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(pool, kSize, 0, &ptr), HSA_STATUS_SUCCESS);

  std::uint32_t pool_flags = 0;
  ASSERT_EQ(hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &pool_flags),
            HSA_STATUS_SUCCESS);

  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  ASSERT_EQ(hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr), HSA_STATUS_SUCCESS);

  EXPECT_EQ(info.global_flags, pool_flags);
  EXPECT_TRUE(info.registered);

  EXPECT_EQ(hsa_amd_memory_pool_free(ptr), HSA_STATUS_SUCCESS);
}

TEST_F(PointerInfoTest, ReportsKernargPoolFlags) {
  // A kernarg pool allocation reports KERNARG_INIT.
  find_pool_data kernarg{};
  kernarg.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT;
  kernarg.expected_allocatable = true;
  if (hsa_amd_agent_iterate_memory_pools(aie_agents.front(), find_memory_pool, &kernarg) !=
      HSA_STATUS_INFO_BREAK) {
    GTEST_SKIP() << "no allocatable kernarg pool on this agent";
  }

  void* ptr = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(kernarg.pool, 4096, 0, &ptr), HSA_STATUS_SUCCESS);

  std::uint32_t pool_flags = 0;
  ASSERT_EQ(hsa_amd_memory_pool_get_info(kernarg.pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                         &pool_flags),
            HSA_STATUS_SUCCESS);

  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  ASSERT_EQ(hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr), HSA_STATUS_SUCCESS);
  ASSERT_EQ(info.type, HSA_EXT_POINTER_TYPE_HSA);
  EXPECT_EQ(info.global_flags, pool_flags);
  EXPECT_TRUE(info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT);

  EXPECT_EQ(hsa_amd_memory_pool_free(ptr), HSA_STATUS_SUCCESS);
}

TEST_F(PointerInfoTest, ReportsPerAllocationFlags) {
  // Flags passed to the allocation are reported in alloc_flags.
  constexpr std::size_t kSize = 4096;
  void* ptr = nullptr;
  const hsa_amd_memory_pool_flag_t kFlag = HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG;
  if (hsa_amd_memory_pool_allocate(pointer_info_pool(), kSize, kFlag, &ptr) != HSA_STATUS_SUCCESS) {
    GTEST_SKIP() << "pool does not accept the executable allocation flag";
  }

  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  ASSERT_EQ(hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr), HSA_STATUS_SUCCESS);
  ASSERT_EQ(info.type, HSA_EXT_POINTER_TYPE_HSA);
  EXPECT_TRUE(info.alloc_flags & HSA_AMD_POINTER_INFO_ALLOC_FLAG_EXECUTABLE);

  EXPECT_EQ(hsa_amd_memory_pool_free(ptr), HSA_STATUS_SUCCESS);
}

TEST_F(PointerInfoTest, LeavesNonAieAllocationsAlone) {
  // libhsakmt only registers a CPU-pool allocation as a real KFD buffer object
  // (hsakmt_fmm_allocate_host picking hsakmt_is_dgpu's branch) when at least one discrete-GPU KFD
  // node exists; with none, hsa_amd_pointer_info reports it as HSA_EXT_POINTER_TYPE_RESERVED_ADDR
  // instead of HSA_EXT_POINTER_TYPE_HSA. That is a libhsakmt classification gap on GPU-less
  // systems, unrelated to AIE, so this test needs a GPU present rather than trying to work around
  // it here.
  std::vector<hsa_agent_t> gpu_agents;
  ASSERT_EQ(hsa_iterate_agents(aie_test::discover_agents<HSA_DEVICE_TYPE_GPU>, &gpu_agents),
            HSA_STATUS_SUCCESS);
  if (gpu_agents.empty()) GTEST_SKIP() << "no GPU agent; see libhsakmt's hsakmt_is_dgpu gap";

  // A CPU pool allocation is owned by the CPU agent.
  std::vector<hsa_agent_t> cpu_agents;
  ASSERT_EQ(hsa_iterate_agents(aie_test::discover_agents<HSA_DEVICE_TYPE_CPU>, &cpu_agents),
            HSA_STATUS_SUCCESS);
  if (cpu_agents.empty()) GTEST_SKIP() << "no CPU agent";

  find_pool_data sys_pool{};
  sys_pool.expected_flags = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED;
  sys_pool.expected_allocatable = true;
  if (hsa_amd_agent_iterate_memory_pools(cpu_agents.front(), find_memory_pool, &sys_pool) !=
      HSA_STATUS_INFO_BREAK) {
    GTEST_SKIP() << "no allocatable fine-grained CPU pool";
  }

  void* ptr = nullptr;
  ASSERT_EQ(hsa_amd_memory_pool_allocate(sys_pool.pool, 4096, 0, &ptr), HSA_STATUS_SUCCESS);

  hsa_amd_pointer_info_t info{};
  info.size = sizeof(info);
  EXPECT_EQ(hsa_amd_pointer_info(ptr, &info, nullptr, nullptr, nullptr), HSA_STATUS_SUCCESS);
  EXPECT_EQ(info.type, HSA_EXT_POINTER_TYPE_HSA);
  // System memory is accessed at its host address.
  EXPECT_EQ(info.agentBaseAddress, ptr);
  EXPECT_EQ(info.agentOwner.handle, cpu_agents.front().handle);

  EXPECT_EQ(hsa_amd_memory_pool_free(ptr), HSA_STATUS_SUCCESS);
}
