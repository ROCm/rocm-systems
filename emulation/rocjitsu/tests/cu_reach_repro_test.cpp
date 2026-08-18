// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cu_reach_repro_test.cpp
/// @brief Minimal reproducer: how many compute units does one dispatch reach?
///
/// @details gfx950 advertises 256 compute units to the guest, but a HW queue is
/// owned by one XCD's command processor, and before XCD fan-out that command
/// processor placed every workgroup of every dispatch on its own XCD's 32 CUs.
/// A single-queue application therefore used one eighth of the device it was
/// told it had, and nothing in the tree measured it.
///
/// This measures the thing the claim is about -- the number of *distinct
/// compute units* that ran a wavefront -- rather than the per-XCD workgroup
/// histogram the fan-out tests assert on.
///
/// Reaching a compute unit takes more than being offered to it. Placement is
/// first-fit: a command processor offers a workgroup to its shader engines in
/// order and a shader engine round-robins over its own CUs, so a workgroup that
/// costs nothing piles up on the first engine and the census undercounts the
/// hardware the dispatch could have used. Each workgroup here asks for enough
/// LDS that a CU can hold exactly one, which is what makes the grid spread as
/// far as the placement policy allows and makes the count mean something.
///
/// It is deliberately self-contained: it builds its own ring and packet instead
/// of using tests/aql_queue.h, and it asks for queue fan-out through a
/// `requires` probe, so the same source compiles and runs on a tree from before
/// the fix (where it reports 32 and fails) and on one with it (256, passes).
///
/// Run it with:
///   ctest --test-dir build -R CuReachRepro --output-on-failure
/// or
///   ./build/tests/rocjitsu_tests --gtest_filter='CuReachRepro.*'

#include "test_paths.h"

#include "embedded_schema.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"

#ifndef HSA_LARGE_MODEL
#define HSA_LARGE_MODEL 1
#endif

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
#include "hsa/hsa.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

const std::string CONFIG_PATH = test::config_path("gfx950_cdna4.json");

constexpr uint32_t kTotalXcds = 8;
constexpr uint32_t kCusPerXcd = 32; // 4 shader engines x 8 CUs
constexpr uint32_t kTotalCus = kTotalXcds * kCusPerXcd;
constexpr uint32_t kWavefrontSize = 64;

/// A CU holds 160 KiB of LDS, so this much per workgroup means one at a time.
constexpr uint32_t kLdsPerWorkgroup = 96u * 1024u;

constexpr uint64_t kKdAddr = 0x10000;
constexpr uint64_t kRingAddr = 0xF0000000ULL;
constexpr uint32_t kRingSize = 4096; // 64 packets
constexpr uint64_t kReadPtrAddr = 0xF0010000ULL;
constexpr uint64_t kWritePtrAddr = 0xF0010008ULL;
constexpr uint64_t kDoorbellAddr = 0xF0010010ULL;

/// @brief Records the identity of every compute unit that ran a wavefront.
///
/// @details `Wavefront::cu().id()` is a simdojo ComponentID, unique across the
/// whole SoC, so the size of this set is the number of physical compute units
/// the run touched, whichever XCD they belong to. Workgroup dispatch is not a
/// hot hook and fan-out places workgroups from several engine threads, so this
/// locks rather than relying on the serial-hot-hook contract.
class CuCensusPlugin : public ExecutionPlugin {
public:
  CuCensusPlugin() : ExecutionPlugin("cu-census") {}

  void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t, uint32_t,
                                   std::span<amdgpu::Wavefront *> wavefronts) override {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto *wf : wavefronts) {
      if (wf != nullptr)
        cus_.insert(static_cast<uint32_t>(wf->cu().id()));
    }
  }

  std::size_t distinct_cus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cus_.size();
  }

private:
  mutable std::mutex mutex_;
  std::set<uint32_t> cus_;
};

/// @brief A loaded gfx950 SoC with a trivial `s_endpgm` kernel in GPU memory.
struct Fixture {
  config::LoadedConfig loaded;
  std::unique_ptr<simdojo::SimulationEngine> engine;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *memory = nullptr;
  std::shared_ptr<ExecutionPluginGroup> group;
  CuCensusPlugin *census = nullptr;

  Fixture() : loaded(config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema)) {
    soc = loaded.soc();
    memory = loaded.memory();
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    engine->create();

    auto plugin = std::make_unique<CuCensusPlugin>();
    census = plugin.get();
    group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    group->add(std::move(plugin));
    soc->set_plugin_group(group);

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

/// @brief Ask for the fan-out a real compute queue gets, on trees that have it.
///
/// @details The KFD queue-creation path sets `HwQueue::xcd_fanout` on compute
/// queues, and that flag is what makes a dispatch reach the whole device. A
/// tree from before that work has no such field, and this reproducer still has
/// to build and run there, because the run without the fix is the evidence.
/// Templated so the missing-member branch is discarded rather than merely not
/// taken. @returns Whether the field existed.
template <typename Queue> bool request_xcd_fanout(Queue &q) {
  if constexpr (requires { q.xcd_fanout = true; }) {
    q.xcd_fanout = true;
    return true;
  } else {
    return false;
  }
}

/// @brief Workgroups placed per XCD, where the SoC can say.
///
/// @details The per-XCD histogram arrived with the fan-out work, so a tree from
/// before it cannot answer; templated for the same reason as
/// request_xcd_fanout. @returns The histogram, or empty when unavailable.
template <typename Soc> std::vector<uint64_t> xcd_histogram(const Soc &soc) {
  if constexpr (requires { soc.dispatched_workgroups_per_xcd(); }) {
    return soc.dispatched_workgroups_per_xcd();
  } else {
    return {};
  }
}

/// @brief Register one compute queue on xcd0's command processor.
///
/// @details Deliberately not `SoC::assign_queue_owner_cp()`, whose spelling is
/// newer than the bug. @param fx The SoC to register the queue on. @param
/// fanout Whether to ask for the whole device. @returns Whether fan-out could
/// be asked for at all.
bool register_queue(Fixture &fx, bool fanout) {
  auto *cp = fx.soc->xcd(0)->command_processor();
  EXPECT_NE(cp, nullptr);

  const uint64_t zero = 0;
  fx.memory->load_image(reinterpret_cast<const uint8_t *>(&zero), 8, kReadPtrAddr);
  fx.memory->load_image(reinterpret_cast<const uint8_t *>(&zero), 8, kWritePtrAddr);
  fx.memory->load_image(reinterpret_cast<const uint8_t *>(&zero), 8, kDoorbellAddr);

  amdgpu::HwQueue hw{};
  hw.queue_id = 1;
  hw.ring_base_va = kRingAddr;
  hw.ring_size = kRingSize;
  hw.read_ptr_va = kReadPtrAddr;
  hw.write_ptr_va = kWritePtrAddr;
  hw.doorbell_va = kDoorbellAddr;
  const bool available = fanout && request_xcd_fanout(hw);
  cp->register_queue(std::move(hw));
  return available;
}

/// @brief Write one kernel dispatch packet into the ring and ring the doorbell.
void dispatch_grid(Fixture &fx, uint32_t grid_size_x) {
  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 1;
  pkt.workgroup_size_x = static_cast<uint16_t>(kWavefrontSize);
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = grid_size_x;
  pkt.grid_size_y = 1;
  pkt.grid_size_z = 1;
  pkt.group_segment_size = kLdsPerWorkgroup;
  pkt.kernel_object = kKdAddr;

  fx.memory->load_image(reinterpret_cast<const uint8_t *>(&pkt), 64, kRingAddr);
  const uint64_t write_idx = 1;
  fx.memory->load_image(reinterpret_cast<const uint8_t *>(&write_idx), 8, kWritePtrAddr);
  fx.memory->load_image(reinterpret_cast<const uint8_t *>(&write_idx), 8, kDoorbellAddr);

  auto *cp = fx.soc->xcd(0)->command_processor();
  cp->engine()->schedule_event_now(cp->doorbell_event());
}

/// @brief Say what the grid reached, so a failing run reports what the tree does.
void report(const char *what, std::size_t reached, const std::vector<uint64_t> &histogram) {
  std::cout << "[ cu-reach ] " << what << ": " << kTotalCus << " workgroups reached " << reached
            << " of " << kTotalCus << " compute units";
  if (!histogram.empty()) {
    std::size_t xcds = 0;
    for (auto count : histogram)
      xcds += count > 0 ? 1 : 0;
    std::cout << " across " << xcds << " of " << kTotalXcds << " XCDs; workgroups per XCD:";
    for (auto count : histogram)
      std::cout << ' ' << count;
  }
  std::cout << '\n';
}

} // namespace

// The reproducer. One queue, one dispatch, one workgroup per compute unit the
// device advertises. Every one of those workgroups should get a compute unit of
// its own. Before XCD fan-out they all landed on the 32 CUs of the single XCD
// that owned the queue, and this reports 32.
TEST(CuReachRepro, OneDispatchReachesEveryComputeUnit) {
  Fixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);

  const bool fanout = register_queue(fx, /*fanout=*/true);
  dispatch_grid(fx, kTotalCus * kWavefrontSize);
  fx.engine->run();

  const std::size_t reached = fx.census->distinct_cus();
  report(fanout ? "fan-out queue" : "fan-out unavailable on this tree", reached,
         xcd_histogram(*fx.soc));

  EXPECT_EQ(reached, kTotalCus) << "one dispatch reached " << reached << " of the " << kTotalCus
                                << " compute units this device advertises"
                                << (fanout ? ""
                                           : "; this tree has no queue fan-out, which is the bug");
}

// The control. A queue that does not ask for the whole device stays on the XCD
// that owns it, which is the behaviour every queue used to have. It holds on
// both sides of the fix, and it is what says the census is measuring where
// workgroups ran rather than how many there were: if this one ever reports more
// than one XCD's worth of compute units, stop believing the other test.
TEST(CuReachRepro, QueueWithoutFanoutIsCappedAtOneXcd) {
  Fixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);

  register_queue(fx, /*fanout=*/false);
  dispatch_grid(fx, kTotalCus * kWavefrontSize);
  fx.engine->run();

  const std::size_t reached = fx.census->distinct_cus();
  report("single-XCD queue", reached, xcd_histogram(*fx.soc));

  EXPECT_EQ(reached, kCusPerXcd);
}

// The small-grid case, which is the one the fan-out hardening changes. A grid
// with fewer workgroups than the device has XCDs used to be left whole, on the
// reasoning that it was too small to be worth splitting -- so the four
// workgroups of a four-workgroup grid shared one XCD's compute units while the
// other seven XCDs sat idle. There is nothing to gain by keeping them there.
TEST(CuReachRepro, GridSmallerThanTheXcdCountStillLeavesOneXcd) {
  Fixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);

  constexpr uint32_t kWorkgroups = 4; // fewer than the 8 XCDs
  register_queue(fx, /*fanout=*/true);
  dispatch_grid(fx, kWorkgroups * kWavefrontSize);
  fx.engine->run();

  const auto histogram = xcd_histogram(*fx.soc);
  if (histogram.empty())
    GTEST_SKIP() << "this tree cannot report where workgroups were placed";

  std::size_t xcds = 0;
  for (auto count : histogram)
    xcds += count > 0 ? 1 : 0;
  std::cout << "[ cu-reach ] small grid: " << kWorkgroups << " workgroups reached "
            << fx.census->distinct_cus() << " compute units across " << xcds << " XCDs;"
            << " workgroups per XCD:";
  for (auto count : histogram)
    std::cout << ' ' << count;
  std::cout << '\n';

  EXPECT_EQ(xcds, kWorkgroups);
  EXPECT_EQ(fx.census->distinct_cus(), kWorkgroups);
}
