// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file xcd_distribution_test.cpp
/// @brief How a single AQL dispatch is spread across the XCDs of a multi-XCD SoC.

#include "aql_queue.h"
#include "test_paths.h"

#include "embedded_schema.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

const std::string CONFIG_PATH = test::config_path("gfx950_cdna4.json");

constexpr uint32_t kTotalXcds = 8;
constexpr uint32_t kCusPerXcd = 32; // 4 SEs x 8 CUs
constexpr uint32_t kTotalCus = kTotalXcds * kCusPerXcd;
constexpr uint64_t kKdAddr = 0x10000;
constexpr uint32_t kWavefrontSize = 64;

/// A loaded gfx950 SoC plus a trivial s_endpgm kernel resident in GPU memory.
struct XcdDistributionFixture {
  config::LoadedConfig loaded;
  std::unique_ptr<simdojo::SimulationEngine> engine;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *memory = nullptr;

  XcdDistributionFixture() : loaded(config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema)) {
    soc = loaded.soc();
    memory = loaded.memory();
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    engine->create();

    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t kd{};
    kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                    ((256 / 8) - 1)); // CDNA4 VGPR granularity is 8
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    ((104 / 8) - 1));
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
    memory->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), kKdAddr);
    memory->write32(kKdAddr + sizeof(kernel_descriptor_t),
                    build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
  }
};

} // namespace

// One HW queue is placed on one XCD by SoC::assign_queue_cp(), which is the
// selector the KFD CREATE_QUEUE path uses. Every workgroup of a dispatch on that
// queue is therefore placed by that one XCD's command processor, on that one
// XCD's compute units.
//
// This pins CURRENT behavior, which is a fidelity gap: a real multi-XCD part in
// SPX mode spreads one dispatch over every XCD of the partition. The topology
// here advertises 256 CUs, so a single-queue application reaches 1/8 of them.
// The follow-on fan-out work flips this expectation to an even spread; see
// SingleQueueGridSpreadsOverAllXcds below.
TEST(XcdDistributionTest, SingleQueueGridLandsOnOneXcd) {
  XcdDistributionFixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);

  auto *cp = fx.soc->assign_queue_cp();
  ASSERT_NE(cp, nullptr);
  test::AqlQueue queue(fx.memory, cp);
  queue.dispatch(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), kTotalCus);

  size_t xcds_used = 0;
  for (auto count : counts)
    xcds_used += count > 0 ? 1 : 0;
  EXPECT_EQ(xcds_used, 1u) << "expected the whole grid confined to one XCD";
}

// A queue marked for fan-out spreads each dispatch over every XCD, round-robin
// one workgroup at a time. The permutation is part of the contract: kernels that
// swizzle their workgroup index for cache locality assume workgroup i runs on XCD
// i % num_xcds.
TEST(XcdDistributionTest, FanoutQueueGridSpreadsOverAllXcds) {
  XcdDistributionFixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);

  auto *cp = fx.soc->assign_queue_cp();
  ASSERT_NE(cp, nullptr);
  test::AqlQueue queue(fx.memory, cp, test::AqlQueue::DEFAULT_RING_ADDR,
                       test::AqlQueue::DEFAULT_RING_SIZE, test::AqlQueue::DEFAULT_READ_PTR_ADDR,
                       test::AqlQueue::DEFAULT_WRITE_PTR_ADDR,
                       test::AqlQueue::DEFAULT_DOORBELL_ADDR, /*xcd_fanout=*/true);
  queue.dispatch(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), kTotalCus);
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    EXPECT_EQ(counts[xi], kTotalCus / kTotalXcds) << "xcd" << xi;
}

// The split must not depend on which XCD the queue landed on: rank is the XCD's
// own index, so the workgroup-to-XCD mapping is the same for every queue.
TEST(XcdDistributionTest, FanoutIsIndependentOfOwningXcd) {
  XcdDistributionFixture fx;

  // Rotate the assignment so the queue is owned by an XCD other than xcd0.
  for (uint32_t i = 0; i < 3; ++i)
    ASSERT_NE(fx.soc->assign_queue_cp(), nullptr);
  auto *cp = fx.soc->assign_queue_cp();
  ASSERT_NE(cp, nullptr);
  ASSERT_NE(cp, fx.soc->xcd(0)->command_processor());

  test::AqlQueue queue(fx.memory, cp, test::AqlQueue::DEFAULT_RING_ADDR,
                       test::AqlQueue::DEFAULT_RING_SIZE, test::AqlQueue::DEFAULT_READ_PTR_ADDR,
                       test::AqlQueue::DEFAULT_WRITE_PTR_ADDR,
                       test::AqlQueue::DEFAULT_DOORBELL_ADDR, /*xcd_fanout=*/true);
  queue.dispatch(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    EXPECT_EQ(counts[xi], kTotalCus / kTotalXcds) << "xcd" << xi;
}

// A grid with fewer workgroups than XCDs is not split: an empty share is
// indistinguishable from a barrier packet and would retire the dispatch early.
TEST(XcdDistributionTest, GridSmallerThanXcdCountIsNotSplit) {
  XcdDistributionFixture fx;

  auto *cp = fx.soc->assign_queue_cp();
  ASSERT_NE(cp, nullptr);
  test::AqlQueue queue(fx.memory, cp, test::AqlQueue::DEFAULT_RING_ADDR,
                       test::AqlQueue::DEFAULT_RING_SIZE, test::AqlQueue::DEFAULT_READ_PTR_ADDR,
                       test::AqlQueue::DEFAULT_WRITE_PTR_ADDR,
                       test::AqlQueue::DEFAULT_DOORBELL_ADDR, /*xcd_fanout=*/true);
  constexpr uint32_t kWgs = kTotalXcds - 1;
  queue.dispatch(kKdAddr, kWgs * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), kWgs);
  size_t xcds_used = 0;
  for (auto count : counts)
    xcds_used += count > 0 ? 1 : 0;
  EXPECT_EQ(xcds_used, 1u);
}

// assign_queue_cp() rotates queues across XCDs, so N queues do reach N XCDs.
// This is the only XCD spreading that exists today, and it only helps an
// application that opens more than one HW queue.
TEST(XcdDistributionTest, QueuesRotateAcrossXcds) {
  XcdDistributionFixture fx;

  constexpr uint32_t kWgsPerQueue = kCusPerXcd;
  std::vector<std::unique_ptr<test::AqlQueue>> queues;
  for (uint32_t qi = 0; qi < kTotalXcds; ++qi) {
    auto *cp = fx.soc->assign_queue_cp();
    ASSERT_NE(cp, nullptr);
    uint64_t ring = 0xF0000000ULL + qi * 0x100000ULL;
    queues.push_back(std::make_unique<test::AqlQueue>(fx.memory, cp, ring, 4096, ring + 0x10000,
                                                      ring + 0x10008, ring + 0x10010));
    queues.back()->dispatch(kKdAddr, kWgsPerQueue * kWavefrontSize, kWavefrontSize);
  }

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    EXPECT_EQ(counts[xi], kWgsPerQueue) << "xcd" << xi;
}
