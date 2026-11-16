/* Copyright (c) 2026 Advanced Micro Devices, Inc. */
/* SPDX-License-Identifier: MIT */

#include <atomic>
#include <thread>

#include <gtest/gtest.h>
#include "abce_device.h"
#include "abce_test_harness.hpp"

namespace {

bool MapOne() {
  abce_test::Mapper mapper(abce_test::IsaByName("gfx942"));
  const auto metadata = mapper.NewMetadata();
  abce_plan_t plan{};
  return mapper.Map({abce_test::MakeDeviceToDevice(0x1000, 0x2000, 64)}, metadata, &plan) ==
             ABCE_STATUS_OK &&
         plan.num_frames == 1 && mapper.WalkFrame(plan, 0).clean;
}

struct MapAtThreadExit {
  std::atomic<int>* result;
  ~MapAtThreadExit() { result->store(MapOne() ? 1 : -1); }
};

TEST(ExternalAbi, MapFromCppThreadLocalDestructor) {
  std::atomic<int> result{0};
  std::thread worker([&] {
    // Register this destructor before first use of the provider's TLS. A Rust
    // thread_local Vec would already be destroyed when this destructor maps.
    thread_local MapAtThreadExit at_exit{&result};
    (void)at_exit;
    EXPECT_TRUE(MapOne());
  });
  worker.join();
  EXPECT_EQ(result.load(), 1);
}

TEST(ExternalAbi, InitializesDeviceDescriptorWithoutDeviceImplementation) {
  EXPECT_EQ(abce_host_abi_version(), ABCE_HOST_ABI_VERSION);
  abce_builder_t builder{};
  ASSERT_EQ(abce_builder_initialize({9, 4, 2}, nullptr, &builder), ABCE_STATUS_OK);
  abce_device_ring_t ring{};
  abce_device_engine_t engine{};
  ASSERT_EQ(abce_device_engine_initialize(&builder, &ring, 3, true, &engine), ABCE_STATUS_OK);
  EXPECT_EQ(engine.engine_id, 3u);
  EXPECT_TRUE(engine.device_atomic_support);
  abce_device_handle_t handle{};
  handle.abi_version = ABCE_DEVICE_ABI_VERSION;
  handle.engine = &engine;
  EXPECT_TRUE(abce_device_handle_valid(&handle));
  ++handle.abi_version;
  EXPECT_FALSE(abce_device_handle_valid(&handle));
}

TEST(ExternalAbi, PolicyCallbacksHaveThePublicFunctionAddress) {
  EXPECT_EQ(abce_sdma_policy_base_vtable.rank_h2d, &abce_sdma_policy_base_rank_all);
  EXPECT_EQ(abce_sdma_policy_base_vtable.rank_d2h, &abce_sdma_policy_base_rank_all);
  EXPECT_EQ(abce_sdma_policy_base_vtable.rank_p2p, &abce_sdma_policy_base_rank_p2p);
  EXPECT_EQ(abce_sdma_policy_base_vtable.rank_local_d2d, &abce_sdma_policy_base_rank_all);
  EXPECT_EQ(abce_sdma_policy_gfx90a_vtable.rank_local_d2d, &abce_sdma_policy_base_rank_all);
  EXPECT_EQ(abce_sdma_policy_gfx94x_vtable.rank_local_d2d, &abce_sdma_policy_base_rank_all);
}

}  // namespace
