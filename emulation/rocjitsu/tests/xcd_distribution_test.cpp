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
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
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

// Records the interleaving of workgroup dispatch and completion across all XCDs.
class WorkgroupOrderPlugin : public ExecutionPlugin {
public:
  WorkgroupOrderPlugin() : ExecutionPlugin("xcd-wg-order") {}

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t, uint32_t, uint32_t,
                                   std::span<amdgpu::Wavefront *>) override {
    if (first_dispatched_.find(dispatch_id) == first_dispatched_.end())
      first_dispatched_[dispatch_id] = step_;
    ++step_;
  }

  void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t) override {
    last_completed_[dispatch_id] = step_++;
  }

  /// Step at which the first workgroup of @p dispatch_id was placed on any XCD.
  uint64_t first_dispatched(uint32_t dispatch_id) const {
    auto it = first_dispatched_.find(dispatch_id);
    return it == first_dispatched_.end() ? UINT64_MAX : it->second;
  }
  /// Step at which the last workgroup of @p dispatch_id retired on any XCD.
  uint64_t last_completed(uint32_t dispatch_id) const {
    auto it = last_completed_.find(dispatch_id);
    return it == last_completed_.end() ? UINT64_MAX : it->second;
  }
  size_t dispatch_count() const { return first_dispatched_.size(); }

  /// Dispatch ids in the order their first workgroup was placed.
  std::vector<uint32_t> dispatch_ids() const {
    std::vector<uint32_t> ids;
    for (const auto &[id, step] : first_dispatched_)
      ids.push_back(id);
    std::sort(ids.begin(), ids.end(),
              [&](uint32_t a, uint32_t b) { return first_dispatched(a) < first_dispatched(b); });
    return ids;
  }

private:
  uint64_t step_ = 0;
  std::map<uint32_t, uint64_t> first_dispatched_;
  std::map<uint32_t, uint64_t> last_completed_;
};

} // namespace

// Fan-out is a property of the queue, not of the device. A queue registered
// directly against one command processor, as a test that wants that command
// processor's CUs to itself does, keeps the whole grid on that one XCD. The KFD
// path opts in; see FanoutQueueGridSpreadsOverAllXcds for that behavior.
TEST(XcdDistributionTest, QueueWithoutFanoutKeepsGridOnOneXcd) {
  XcdDistributionFixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);

  auto *cp = fx.soc->assign_queue_owner_cp();
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

  auto *cp = fx.soc->assign_queue_owner_cp();
  ASSERT_NE(cp, nullptr);
  auto queue = test::make_fanout_queue(fx.memory, cp);
  queue->dispatch(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

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
    ASSERT_NE(fx.soc->assign_queue_owner_cp(), nullptr);
  auto *cp = fx.soc->assign_queue_owner_cp();
  ASSERT_NE(cp, nullptr);
  ASSERT_NE(cp, fx.soc->xcd(0)->command_processor());

  auto queue = test::make_fanout_queue(fx.memory, cp);
  queue->dispatch(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    EXPECT_EQ(counts[xi], kTotalCus / kTotalXcds) << "xcd" << xi;
}

// A grid with fewer workgroups than XCDs still reaches one workgroup per XCD for
// as far as it goes; the remaining XCDs take an empty share. The empty shares are
// what keep every XCD's view of the queue in step, so ordering still works.
TEST(XcdDistributionTest, GridSmallerThanXcdCountSpreadsOnePerXcd) {
  XcdDistributionFixture fx;

  auto *cp = fx.soc->assign_queue_owner_cp();
  ASSERT_NE(cp, nullptr);
  auto queue = test::make_fanout_queue(fx.memory, cp);
  constexpr uint32_t kWgs = kTotalXcds - 3;
  queue->dispatch(kKdAddr, kWgs * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), kWgs);
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    EXPECT_EQ(counts[xi], xi < kWgs ? 1u : 0u) << "xcd" << xi;
}

// A dispatch too small to reach every XCD still has to hold up a following
// barrier'd packet on the XCDs it never touched. Those XCDs only know the packet
// exists because fan-out gives them an empty share of it, so this is the case
// that breaks if empty shares are skipped as an optimization.
TEST(XcdDistributionTest, BarrierAfterSmallDispatchStillOrders) {
  XcdDistributionFixture fx;

  auto plugin = std::make_unique<WorkgroupOrderPlugin>();
  auto *order = plugin.get();
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::move(plugin)));
  fx.soc->set_plugin_group(group);

  auto *cp = fx.soc->assign_queue_owner_cp();
  ASSERT_NE(cp, nullptr);
  auto queue = test::make_fanout_queue(fx.memory, cp);

  // Two workgroups: only two of the eight XCDs run any of it.
  queue->dispatch(kKdAddr, 2 * kWavefrontSize, kWavefrontSize);
  queue->dispatch_with_barrier(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto ids = order->dispatch_ids();
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_GT(order->first_dispatched(ids[1]), order->last_completed(ids[0]))
      << "an XCD with no share of the first dispatch started the barrier'd one early";
}

// Fan-out copies a dispatch id onto peer XCDs, and completion bookkeeping looks
// entries up by that id. If two XCDs could mint the same id, a peer holding both
// a shard of one dispatch and its own dispatch with the same id would credit
// workgroup completions to whichever it found first. Drive this through real
// dispatches on queues owned by different XCDs rather than the id counter alone.
TEST(XcdDistributionTest, DispatchIdsAreDisjointAcrossXcds) {
  XcdDistributionFixture fx;

  auto plugin = std::make_unique<WorkgroupOrderPlugin>();
  auto *order = plugin.get();
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::move(plugin)));
  fx.soc->set_plugin_group(group);

  // One queue per XCD, so every XCD mints ids for dispatches of its own while
  // also holding shards minted by the other seven.
  std::vector<std::unique_ptr<test::AqlQueue>> queues;
  for (uint32_t qi = 0; qi < kTotalXcds; ++qi) {
    auto *cp = fx.soc->assign_queue_owner_cp();
    ASSERT_NE(cp, nullptr);
    uint64_t ring = 0xF0000000ULL + qi * 0x100000ULL;
    // Distinct queue ids: a fan-out queue is replicated onto every XCD, and each
    // CP routes an incoming shard back by (queue_id, process_id).
    queues.push_back(test::make_fanout_queue(fx.memory, cp, /*queue_id=*/qi + 1, ring));
    queues.back()->dispatch(kKdAddr, kTotalXcds * kWavefrontSize, kWavefrontSize);
  }

  fx.engine->run();

  // Every dispatch must be distinguishable; a collision would have merged two of
  // them into one id and lost a completion.
  EXPECT_EQ(order->dispatch_count(), kTotalXcds);
  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}),
            uint64_t{kTotalXcds} * kTotalXcds);
}

// Two fanned-out dispatches, the second carrying the AQL barrier bit. The barrier
// means no later packet starts until every preceding packet has completed, which
// for a fanned-out dispatch is a property of the whole grid: an XCD that finished
// its own share of the first dispatch must still not begin the second while a
// peer is running.
//
// Resource pressure cannot show this — these workgroups retire immediately and
// never fill the device — so observe the interleaving directly. Since fan-out
// gives every shard of a dispatch the same dispatch id, "first workgroup of the
// second dispatch placed anywhere" must come after "last workgroup of the first
// dispatch retired anywhere".
TEST(XcdDistributionTest, BarrierBitWaitsForEveryXcdsShare) {
  XcdDistributionFixture fx;

  auto plugin = std::make_unique<WorkgroupOrderPlugin>();
  auto *order = plugin.get();
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::move(plugin)));
  fx.soc->set_plugin_group(group);

  auto *cp = fx.soc->assign_queue_owner_cp();
  ASSERT_NE(cp, nullptr);
  auto queue = test::make_fanout_queue(fx.memory, cp);

  queue->dispatch(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);
  queue->dispatch_with_barrier(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), uint64_t{2} * kTotalCus);
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    EXPECT_EQ(counts[xi], 2 * (kTotalCus / kTotalXcds)) << "xcd" << xi;

  // Fan-out shares one dispatch id per dispatch, so exactly two ids appear.
  auto ids = order->dispatch_ids();
  ASSERT_EQ(ids.size(), 2u) << "expected exactly two distinct dispatch ids";
  EXPECT_GT(order->first_dispatched(ids[1]), order->last_completed(ids[0]))
      << "an XCD began the barrier'd dispatch before every XCD retired the previous one";
}

// Queue ownership still rotates across XCDs. With fan-out that no longer decides
// where the work runs, but it does spread ring reads and completion signalling.
TEST(XcdDistributionTest, QueuesRotateAcrossXcds) {
  XcdDistributionFixture fx;

  constexpr uint32_t kWgsPerQueue = kCusPerXcd;
  std::vector<std::unique_ptr<test::AqlQueue>> queues;
  for (uint32_t qi = 0; qi < kTotalXcds; ++qi) {
    auto *cp = fx.soc->assign_queue_owner_cp();
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
