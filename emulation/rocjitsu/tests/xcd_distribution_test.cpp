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
#include "rocjitsu/vm/amdgpu/partitioning.h"
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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

const std::string CONFIG_PATH = test::config_path("gfx950_mi355x.json");

constexpr uint32_t kTotalXcds = 8;
constexpr uint32_t kCusPerXcd = 36; // 4 SEs x 9 CUs
constexpr uint32_t kTotalCus = kTotalXcds * kCusPerXcd;
constexpr uint64_t kKdAddr = 0x10000;
constexpr uint64_t kScratchKdAddr = 0x20000;
constexpr uint64_t kDefaultScratchPool = 0x1'0000'0000ULL;
constexpr uint32_t kWavefrontSize = 64;

/// How many worker threads the fixture's engine runs on.
enum class Threading {
  /// One thread drives every XCD, as the config ships. Any ordering between two
  /// XCDs is then a property of the single drain loop rather than of the code
  /// under test.
  Single,
  /// One thread per XCD, via the XCD-aware partitioning policy. This is what puts
  /// two command processors on genuinely opposing threads, so the cross-CP inbox
  /// lock ordering and the cross-thread shard handoff are actually exercised.
  ThreadPerXcd,
};

/// A loaded gfx950 SoC plus a trivial s_endpgm kernel resident in GPU memory.
struct XcdDistributionFixture {
  config::LoadedConfig loaded;
  std::unique_ptr<simdojo::SimulationEngine> engine;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *memory = nullptr;

  explicit XcdDistributionFixture(Threading threading = Threading::Single)
      : loaded(config::load_config(CONFIG_PATH, rocjitsu::kEmbeddedSchema)) {
    soc = loaded.soc();
    memory = loaded.memory();
    if (threading == Threading::ThreadPerXcd)
      loaded.engine_config.num_threads =
          amdgpu::clamp_xcd_partition_count(soc, static_cast<uint32_t>(soc->num_xcds()));
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    if (loaded.engine_config.num_threads > 1 &&
        !amdgpu::partition_topology_by_xcds(engine->topology(), soc,
                                            loaded.engine_config.num_threads))
      ADD_FAILURE() << "multi-threaded topology requires per-XCD partitioning";
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

void install_scratch_kernel(amdgpu::GpuMemory &memory, uint32_t private_bytes) {
  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                  ((256 / 8) - 1));
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  ((104 / 8) - 1));
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  kd.private_segment_fixed_size = private_bytes;
  memory.load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), kScratchKdAddr);
  memory.write32(kScratchKdAddr + sizeof(kernel_descriptor_t),
                 build_s_endpgm(ROCJITSU_CODE_ARCH_CDNA4));
}

/// Index of the XCD whose command processor is @p cp.
uint32_t assigned_xcd_index(const SoC &soc, const amdgpu::CommandProcessor *cp) {
  for (uint32_t xi = 0; xi < soc.num_xcds(); ++xi)
    if (soc.xcd(xi)->command_processor() == cp)
      return xi;
  ADD_FAILURE() << "command processor does not belong to this SoC";
  return 0;
}

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

/// Records the grid-wide dispatch callbacks alongside workgroup completion.
///
/// @details Ordering comes from the shared counter rather than from wall clock:
/// the lock that guards it also serializes the callbacks, so the steps are a
/// total order consistent with the order the plugin actually observed them, even
/// with one thread per XCD.
class ExecutionSpanPlugin : public ExecutionPlugin {
public:
  struct Event {
    uint32_t dispatch_id;
    uint64_t step;
  };

  ExecutionSpanPlugin() : ExecutionPlugin("xcd-exec-span") {}

  void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    begins_.push_back({dispatch_id, step_++});
  }

  void onAmdgpuWorkgroupCompleted(uint32_t, uint32_t) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++workgroups_completed_;
    last_workgroup_step_ = step_++;
  }

  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ends_.push_back({dispatch_id, step_++});
  }

  std::vector<Event> begins() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return begins_;
  }
  std::vector<Event> ends() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ends_;
  }
  uint32_t workgroups_completed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return workgroups_completed_;
  }
  /// Step at which the last workgroup anywhere on the device retired.
  uint64_t last_workgroup_step() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_workgroup_step_;
  }

private:
  mutable std::mutex mutex_;
  uint64_t step_ = 0;
  uint32_t workgroups_completed_ = 0;
  uint64_t last_workgroup_step_ = 0;
  std::vector<Event> begins_;
  std::vector<Event> ends_;
};

/// Records whether each dispatched wave retained the queue's address-space identity.
class AddressSpaceIdentityPlugin : public ExecutionPlugin {
public:
  explicit AddressSpaceIdentityPlugin(amdgpu::AddressSpaceHandle expected)
      : ExecutionPlugin("xcd-address-space-identity"), expected_(expected) {}

  void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t, uint32_t, uint32_t,
                                   std::span<amdgpu::Wavefront *> waves) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++workgroups_;
    for (const auto *wave : waves)
      retained_ = retained_ && wave->address_space() == expected_;
  }

  bool retained() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return retained_;
  }

  uint32_t workgroups() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return workgroups_;
  }

private:
  amdgpu::AddressSpaceHandle expected_;
  mutable std::mutex mutex_;
  bool retained_ = true;
  uint32_t workgroups_ = 0;
};

/// Identity translator and transport adapter used only to put the ordinary
/// simulator backing behind a real generation-checked GpuVm handle.
class SparseGpuVmAccess final : public amdgpu::AddressSpaceTranslator,
                                public amdgpu::PhysicalMemoryAccess {
public:
  struct HostWindow {
    uint64_t base = 0;
    size_t size = 0;
    size_t extent = amdgpu::GpuMemory::PAGE_SIZE;
  };

  explicit SparseGpuVmAccess(amdgpu::GpuMemory &memory,
                             std::optional<HostWindow> host_window = std::nullopt)
      : memory_(&memory), host_window_(host_window),
        host_bytes_(host_window ? host_window->size : 0) {}

  bool provision_host_window(size_t size) {
    if (!host_window_ || size > host_window_->extent)
      return false;
    host_window_->size = size;
    host_bytes_.resize(size);
    return true;
  }

  amdgpu::VmTranslationResult translate(uint64_t address, std::size_t size,
                                        amdgpu::VmAccessKind) const override {
    if (size == 0 || size - 1 > std::numeric_limits<uint64_t>::max() - address)
      return {.outcome = amdgpu::VmAccessOutcome::Malformed, .translation = {}};
    if (host_window_ && address >= host_window_->base &&
        address - host_window_->base < host_window_->extent) {
      const uint64_t offset = address - host_window_->base;
      if (offset > host_window_->size || size > host_window_->size - static_cast<size_t>(offset))
        return {.outcome = amdgpu::VmAccessOutcome::Faulted, .translation = {}};
    }
    return {
        .outcome = amdgpu::VmAccessOutcome::Complete,
        .translation = {.domain = amdgpu::VmMemoryDomain::Local,
                        .address = address,
                        .contiguous_bytes =
                            amdgpu::GpuMemory::PAGE_SIZE - (address & amdgpu::GpuMemory::PAGE_MASK),
                        .mtype = amdgpu::Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

  amdgpu::VmAccessOutcome read(amdgpu::VmMemoryDomain domain, uint64_t address,
                               std::span<std::byte> bytes) override {
    if (domain != amdgpu::VmMemoryDomain::Local)
      return amdgpu::VmAccessOutcome::Malformed;
    memory_->read_block(
        address, std::span<uint8_t>(reinterpret_cast<uint8_t *>(bytes.data()), bytes.size()));
    return amdgpu::VmAccessOutcome::Complete;
  }

  amdgpu::VmAccessOutcome write(amdgpu::VmMemoryDomain domain, uint64_t address,
                                std::span<const std::byte> bytes) override {
    if (domain != amdgpu::VmMemoryDomain::Local)
      return amdgpu::VmAccessOutcome::Malformed;
    memory_->write_block(
        address,
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
    return amdgpu::VmAccessOutcome::Complete;
  }

  amdgpu::AtomicLoadResult atomic_load(amdgpu::VmMemoryDomain domain, uint64_t address,
                                       uint32_t width) override {
    if (domain != amdgpu::VmMemoryDomain::Local)
      return {.outcome = amdgpu::VmAccessOutcome::Malformed};
    if (!valid_atomic(address, width))
      return {.outcome = amdgpu::VmAccessOutcome::Malformed};
    uint64_t observed = 0;
    const bool complete = memory_->atomic_modify(
        address, width, [&](const uint8_t *bytes) { std::memcpy(&observed, bytes, width); });
    return {.outcome =
                complete ? amdgpu::VmAccessOutcome::Complete : amdgpu::VmAccessOutcome::Faulted,
            .value = observed};
  }

  amdgpu::VmAccessOutcome atomic_store(amdgpu::VmMemoryDomain domain, uint64_t address,
                                       uint32_t width, uint64_t value) override {
    if (domain != amdgpu::VmMemoryDomain::Local)
      return amdgpu::VmAccessOutcome::Malformed;
    if (!valid_atomic(address, width))
      return amdgpu::VmAccessOutcome::Malformed;
    const bool complete = memory_->atomic_modify(
        address, width, [&](uint8_t *bytes) { std::memcpy(bytes, &value, width); });
    return complete ? amdgpu::VmAccessOutcome::Complete : amdgpu::VmAccessOutcome::Faulted;
  }

  amdgpu::AtomicCompareExchangeResult compare_exchange(amdgpu::VmMemoryDomain domain,
                                                       uint64_t address, uint32_t width,
                                                       uint64_t expected,
                                                       uint64_t desired) override {
    if (domain != amdgpu::VmMemoryDomain::Local)
      return {.outcome = amdgpu::VmAccessOutcome::Malformed};
    if (!valid_atomic(address, width))
      return {.outcome = amdgpu::VmAccessOutcome::Malformed};
    uint64_t observed = 0;
    bool exchanged = false;
    const bool complete = memory_->atomic_modify(address, width, [&](uint8_t *bytes) {
      std::memcpy(&observed, bytes, width);
      exchanged = observed == expected;
      if (exchanged)
        std::memcpy(bytes, &desired, width);
    });
    return {.outcome =
                complete ? amdgpu::VmAccessOutcome::Complete : amdgpu::VmAccessOutcome::Faulted,
            .observed = observed,
            .exchanged = exchanged};
  }

  std::byte *resolve_host_pointer(amdgpu::VmMemoryDomain domain, uint64_t address,
                                  std::size_t size) const override {
    if (domain != amdgpu::VmMemoryDomain::Local || !host_window_ || address < host_window_->base)
      return nullptr;
    const uint64_t offset = address - host_window_->base;
    if (offset > host_window_->size || size > host_window_->size - static_cast<size_t>(offset))
      return nullptr;
    return const_cast<std::byte *>(host_bytes_.data()) + offset;
  }

private:
  static bool valid_atomic(uint64_t address, uint32_t width) {
    return (width == sizeof(uint32_t) || width == sizeof(uint64_t)) && (address & (width - 1)) == 0;
  }

  amdgpu::GpuMemory *memory_;
  std::optional<HostWindow> host_window_;
  std::vector<std::byte> host_bytes_;
};

} // namespace

// Fan-out is a property of the queue, not of the device. A queue registered
// directly against one command processor, as a test that wants that command
// processor's CUs to itself does, keeps the whole grid on that one XCD. Opting
// in is what spreads it; see FanoutQueueGridSpreadsOverAllXcds.
TEST(XcdDistributionTest, QueueWithoutFanoutKeepsGridOnOneXcd) {
  XcdDistributionFixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);
  ASSERT_EQ(fx.soc->all_cus().size(), kTotalCus);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
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

  // ...and it must be the XCD the queue was actually assigned to. Cardinality
  // alone would still pass if the grid ran on the wrong one.
  EXPECT_EQ(counts[assigned_xcd_index(*fx.soc, cp)], kTotalCus);
}

// A queue marked for fan-out spreads each dispatch over every XCD, round-robin
// one workgroup at a time. The permutation is part of the contract: kernels that
// swizzle their workgroup index for cache locality assume workgroup i runs on XCD
// i % num_xcds.
TEST(XcdDistributionTest, FanoutQueueGridSpreadsOverAllXcds) {
  XcdDistributionFixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
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

TEST(XcdDistributionTest, FanoutPreservesAddressSpaceIdentityThroughEveryWave) {
  XcdDistributionFixture fx;
  auto vm_access = std::make_shared<SparseGpuVmAccess>(*fx.memory);
  const amdgpu::AddressSpaceHandle address_space =
      fx.soc->gpu_vm().register_translated(/*vmid=*/0, vm_access, vm_access);
  ASSERT_TRUE(address_space);

  auto plugin = std::make_unique<AddressSpaceIdentityPlugin>(address_space);
  auto *identity = plugin.get();
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::move(plugin)));
  fx.soc->set_plugin_group(group);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  auto queue = test::make_fanout_queue(fx.memory, cp, /*queue_id=*/1,
                                       test::AqlQueue::DEFAULT_RING_ADDR, address_space);

  for (uint32_t xi = 0; xi < kTotalXcds; ++xi) {
    EXPECT_EQ(fx.soc->xcd(xi)->command_processor()->queue_address_space_for_test(
                  /*queue_id=*/1, /*process_id=*/0),
              address_space)
        << "xcd" << xi << " lost the queue address-space identity";
  }

  queue->dispatch(kKdAddr, kTotalXcds * kWavefrontSize, kWavefrontSize);
  fx.engine->run();

  EXPECT_EQ(identity->workgroups(), kTotalXcds);
  EXPECT_TRUE(identity->retained());
  const auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi) {
    EXPECT_EQ(counts[xi], 1u) << "xcd" << xi << " did not execute its address-space shard";
    EXPECT_EQ(fx.soc->xcd(xi)->command_processor()->first_accepted_address_space_for_test(
                  /*queue_id=*/1, /*process_id=*/0),
              address_space)
        << "xcd" << xi << " lost the dispatch address-space identity";
  }
}

// The split must not depend on which XCD the queue landed on: rank is the XCD's
// own index, so the workgroup-to-XCD mapping is the same for every queue.
TEST(XcdDistributionTest, FanoutIsIndependentOfOwningXcd) {
  XcdDistributionFixture fx;

  // Ask for an ordinal that lands the queue on an XCD other than xcd0.
  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/3);
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

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
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

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
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

// A one-workgroup grid is assigned to xcd0 even when another XCD owns the
// queue.  The owner therefore has an empty kernel shard, but it still has to
// publish that share, advance past the packet, and wait for grid-wide retirement
// before it exposes the completion signal or admits a barrier'd follower.
TEST(XcdDistributionTest, EmptyOwnerShardCompletesAndOrdersBarrieredFollower) {
  XcdDistributionFixture fx(Threading::ThreadPerXcd);

  auto plugin = std::make_unique<WorkgroupOrderPlugin>();
  auto *order = plugin.get();
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::move(plugin)));
  fx.soc->set_plugin_group(group);

  constexpr uint64_t kSignalAddr = 0x68000;
  constexpr uint32_t kSignalValueOffset = 8;
  fx.memory->write64(kSignalAddr + kSignalValueOffset, 1);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/3);
  ASSERT_NE(cp, nullptr);
  ASSERT_EQ(assigned_xcd_index(*fx.soc, cp), 3u);
  auto queue = test::make_fanout_queue(fx.memory, cp);

  hsa_kernel_dispatch_packet_t first{};
  first.header = HSA_PACKET_TYPE_KERNEL_DISPATCH | (1u << HSA_PACKET_HEADER_BARRIER);
  first.setup = 1;
  first.workgroup_size_x = kWavefrontSize;
  first.workgroup_size_y = 1;
  first.workgroup_size_z = 1;
  first.grid_size_x = kWavefrontSize;
  first.grid_size_y = 1;
  first.grid_size_z = 1;
  first.kernel_object = kKdAddr;
  first.completion_signal.handle = kSignalAddr;
  queue->submit(first);
  queue->dispatch_with_barrier(kKdAddr, kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  EXPECT_EQ(fx.memory->read64(kSignalAddr + kSignalValueOffset), 0u)
      << "the empty owner shard prevented grid-wide completion publication";

  const auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  EXPECT_EQ(counts[0], 2u);
  for (uint32_t xi = 1; xi < kTotalXcds; ++xi)
    EXPECT_EQ(counts[xi], 0u) << "xcd" << xi << " unexpectedly received a workgroup";

  const auto ids = order->dispatch_ids();
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_GT(order->first_dispatched(ids[1]), order->last_completed(ids[0]))
      << "the barrier'd follower started before the one-workgroup grid retired";
}

// A barrier packet whose dependency is still unsatisfied stops the owner from
// reading further, and that is the only thing holding the dispatch behind it
// back on every XCD. Barrier packets are never handed to peers -- only kernel
// shards are -- so a peer's replica would see Kernel/Kernel with nothing between
// them. If shards were issued at fetch time rather than gated on the owner
// getting past the barrier, every peer would run the second dispatch while the
// barrier was still stalled.
TEST(XcdDistributionTest, BarrierOnPriorDispatchOrdersEveryXcd) {
  XcdDistributionFixture fx;

  auto plugin = std::make_unique<WorkgroupOrderPlugin>();
  auto *order = plugin.get();
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::move(plugin)));
  fx.soc->set_plugin_group(group);

  // The barrier waits on the first dispatch's own completion signal, so it clears
  // only once that dispatch has retired device-wide. amd_signal_t::value lives 8
  // bytes into the signal object.
  constexpr uint64_t kDepSignalAddr = 0x60000;
  constexpr uint32_t kSignalValueOffset = 8;
  fx.memory->write64(kDepSignalAddr + kSignalValueOffset, 1);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  auto queue = test::make_fanout_queue(fx.memory, cp);

  hsa_kernel_dispatch_packet_t first{};
  first.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  first.setup = 1;
  first.workgroup_size_x = kWavefrontSize;
  first.workgroup_size_y = 1;
  first.workgroup_size_z = 1;
  first.grid_size_x = kTotalCus * kWavefrontSize;
  first.grid_size_y = 1;
  first.grid_size_z = 1;
  first.kernel_object = kKdAddr;
  first.completion_signal.handle = kDepSignalAddr;
  queue->submit(first);

  queue->barrier_and(kDepSignalAddr);
  queue->dispatch(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  // Both dispatches ran, spread over every XCD.
  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), 2 * kTotalCus);

  auto ids = order->dispatch_ids();
  ASSERT_EQ(ids.size(), 2u);
  EXPECT_GT(order->first_dispatched(ids[1]), order->last_completed(ids[0]))
      << "an XCD started the post-barrier dispatch before the barrier's dependency cleared";
}

// A packet that runs no shader has no grid to split, so every XCD gets the entry
// itself rather than a share of it. It is still one packet owed one completion:
// the copies carry no signal, or an IB would decrement its signal once per XCD.
TEST(XcdDistributionTest, FanoutReplicatesNonKernelPacketsButSignalsThemOnce) {
  XcdDistributionFixture fx;

  // amd_signal_t::value lives 8 bytes into the signal object. Not 1, so a signal
  // written rather than decremented is also visible.
  constexpr uint64_t kSignalAddr = 0x70000;
  constexpr uint32_t kSignalValueOffset = 8;
  constexpr uint64_t kInitialValue = 5;
  fx.memory->write64(kSignalAddr + kSignalValueOffset, kInitialValue);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  auto queue = test::make_fanout_queue(fx.memory, cp);
  queue->pm4_ib(kSignalAddr);

  fx.engine->run();

  EXPECT_EQ(fx.memory->read64(kSignalAddr + kSignalValueOffset), kInitialValue - 1)
      << "a replicated packet must still signal once, not once per XCD";
}

// The replication itself. Every XCD must accept its share of the kernel and then
// the following IB.
//
// The acceptance count and first two packet kinds are read from each CP AFTER the
// run. Together they prove that the kernel and then the IB reached each queue
// without requiring them to coexist there: with one thread per XCD, a peer may
// legitimately retire its kernel shard before the owner fans out the IB behind it.
TEST(XcdDistributionTest, EveryXcdAcceptsNonKernelPacketsInQueueOrder) {
  XcdDistributionFixture fx(Threading::ThreadPerXcd);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  auto queue = test::make_fanout_queue(fx.memory, cp);

  queue->dispatch(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);
  queue->pm4_ib();

  fx.engine->run();

  for (uint32_t xi = 0; xi < kTotalXcds; ++xi) {
    auto *xcd_cp = fx.soc->xcd(xi)->command_processor();
    EXPECT_EQ(xcd_cp->accepted_entry_count_for_test(/*queue_id=*/1, /*process_id=*/0), 2u)
        << "xcd" << xi << " did not accept the kernel share and the IB behind it";
    const auto kinds =
        xcd_cp->first_accepted_entry_kinds_for_test(/*queue_id=*/1, /*process_id=*/0);
    EXPECT_EQ(kinds[0], amdgpu::DispatchPacketKind::Kernel)
        << "xcd" << xi << " did not accept the kernel share first";
    EXPECT_EQ(kinds[1], amdgpu::DispatchPacketKind::NonKernel)
        << "xcd" << xi << " did not accept the IB after the kernel share";
  }
}

// The mix that makes the replication load-bearing: a barrier'd kernel sitting
// behind a packet that runs no shader. A peer reads its ordering from the entries
// ahead of the barrier'd packet in its own queue, so unless the IB is placed there
// too the peer decides against a shorter prefix than the owner's.
TEST(XcdDistributionTest, BarrierdKernelBehindAnIbOrdersOnEveryXcd) {
  XcdDistributionFixture fx(Threading::ThreadPerXcd);

  auto plugin = std::make_unique<WorkgroupOrderPlugin>();
  auto *order = plugin.get();
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::move(plugin)));
  fx.soc->set_plugin_group(group);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  auto queue = test::make_fanout_queue(fx.memory, cp);

  queue->dispatch(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);
  queue->pm4_ib();
  queue->dispatch_with_barrier(kKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), uint64_t{2} * kTotalCus);

  auto ids = order->dispatch_ids();
  ASSERT_EQ(ids.size(), 2u) << "the IB runs no shader, so only the two kernels report workgroups";
  EXPECT_GT(order->first_dispatched(ids[1]), order->last_completed(ids[0]))
      << "an XCD started the barrier'd kernel while a peer still ran the one before the IB";
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
    auto *cp = fx.soc->assign_queue_owner_cp(qi);
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

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
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
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);
  ASSERT_EQ(fx.soc->all_cus().size(), kTotalCus);

  constexpr uint32_t kWgsPerQueue = kCusPerXcd;
  std::vector<std::unique_ptr<test::AqlQueue>> queues;
  for (uint32_t qi = 0; qi < kTotalXcds; ++qi) {
    auto *cp = fx.soc->assign_queue_owner_cp(qi);
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

// The counter is documented as a lifetime running total, so a second dispatch on
// the same queue must add to it rather than replace it. Both other tests submit
// one packet per command processor, so they would still pass if the counter were
// reset or overwritten per packet; this samples the histogram around the second
// dispatch and checks the delta as well as the accumulated total.
TEST(XcdDistributionTest, CounterAccumulatesAcrossDispatchesOnOneQueue) {
  XcdDistributionFixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);
  ASSERT_EQ(fx.soc->all_cus().size(), kTotalCus);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  const uint32_t xi = assigned_xcd_index(*fx.soc, cp);

  constexpr uint32_t kFirstWgs = kCusPerXcd;
  constexpr uint32_t kSecondWgs = kCusPerXcd / 2;

  test::AqlQueue queue(fx.memory, cp);
  queue.dispatch(kKdAddr, kFirstWgs * kWavefrontSize, kWavefrontSize);
  queue.dispatch(kKdAddr, kSecondWgs * kWavefrontSize, kWavefrontSize);
  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(counts.size(), kTotalXcds);
  EXPECT_EQ(counts[xi], kFirstWgs + kSecondWgs)
      << "counter must accumulate across packets, not restart per packet";
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}),
            uint64_t{kFirstWgs} + kSecondWgs);
}

// The id classes must stay disjoint across the 32-bit wrap, which
// DispatchIdsAreDisjointAcrossXcds cannot reach: it is ~2^29 dispatches per XCD
// away. The case that breaks is a non-power-of-two XCD count, which XcdShard
// explicitly permits -- letting the counter run off the end of the type keeps
// the classes disjoint only when the count divides 2^32. With stride 3, rank 0
// would step 4294967295 -> 2 and collide with rank 1's class.
TEST(XcdDistributionTest, DispatchIdClassesStayDisjointAcrossTheWrap) {
  using amdgpu::CommandProcessor;
  constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();

  for (uint32_t stride : {1u, 3u, 5u, 6u, 7u, 8u}) {
    // Walk each rank's class from just below the top, through the wrap, and on
    // past its restart, collecting every id any rank can produce there.
    std::map<uint32_t, uint32_t> owner_of_id;
    for (uint32_t rank = 0; rank < stride; ++rank) {
      const uint32_t base = 1 + rank;

      // Fast-forward to the last id of this class without enumerating it.
      const uint32_t last = base + ((kMax - base) / stride) * stride;
      ASSERT_LE(last, kMax);
      ASSERT_GT(last, kMax - stride) << "last really is the final id of the class";

      uint32_t id = last;
      for (int step = 0; step < 4; ++step) {
        EXPECT_EQ(id % stride, base % stride)
            << "stride " << stride << " rank " << rank << " left its residue class";
        auto [it, inserted] = owner_of_id.emplace(id, rank);
        EXPECT_TRUE(inserted || it->second == rank)
            << "stride " << stride << ": id " << id << " minted by rank " << rank << " and rank "
            << it->second;
        id = CommandProcessor::step_dispatch_id(id, base, stride);
      }
      // The wrap returns to the class's own base, never past the end of it.
      EXPECT_EQ(CommandProcessor::step_dispatch_id(last, base, stride), base)
          << "stride " << stride << " rank " << rank;
    }
  }
}

// Every other fan-out test leaves completion_signal at zero, so none of them
// enters fire_signal at all: the owner-only signalling rule is unexercised, and
// the barrier and plugin assertions would still pass if the user-visible signal
// were never written, or were written once per XCD.
//
// A fanned-out grid is split eight ways, so a signal fired per share would land
// eight decrements instead of one. Check the value itself, not just that it
// moved.
//
// Two independent mechanisms keep it at one: fan_out_dispatch() zeroes each
// peer's completion_signal, and drain_completions() fires only for the shard
// that is not a peer. Defeating either alone still yields one decrement, so this
// pins the property rather than either implementation of it.
void run_fanout_completion_signal(Threading threading) {
  XcdDistributionFixture fx(threading);
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);

  // amd_signal_t::value lives 8 bytes into the signal object.
  constexpr uint64_t kSignalAddr = 0x50000;
  constexpr uint32_t kSignalValueOffset = 8;
  constexpr uint64_t kInitialValue = 5;
  fx.memory->write64(kSignalAddr + kSignalValueOffset, kInitialValue);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  test::AqlQueue queue(fx.memory, cp, test::AqlQueue::DEFAULT_RING_ADDR,
                       test::AqlQueue::DEFAULT_RING_SIZE, test::AqlQueue::DEFAULT_READ_PTR_ADDR,
                       test::AqlQueue::DEFAULT_WRITE_PTR_ADDR,
                       test::AqlQueue::DEFAULT_DOORBELL_ADDR, /*xcd_fanout=*/true);

  hsa_kernel_dispatch_packet_t pkt{};
  pkt.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  pkt.setup = 1;
  pkt.workgroup_size_x = kWavefrontSize;
  pkt.workgroup_size_y = 1;
  pkt.workgroup_size_z = 1;
  pkt.grid_size_x = kTotalCus * kWavefrontSize;
  pkt.grid_size_y = 1;
  pkt.grid_size_z = 1;
  pkt.kernel_object = kKdAddr;
  pkt.completion_signal.handle = kSignalAddr;
  queue.submit(pkt);

  fx.engine->run();

  // The grid really was spread, or this would prove nothing about fan-out.
  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), kTotalCus);
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    ASSERT_EQ(counts[xi], kTotalCus / kTotalXcds) << "xcd" << xi;

  EXPECT_EQ(fx.memory->read64(kSignalAddr + kSignalValueOffset), kInitialValue - 1)
      << "the dispatch must decrement its completion signal once, not once per XCD";
}

TEST(XcdDistributionTest, FanoutFiresTheCompletionSignalExactlyOnce) {
  run_fanout_completion_signal(Threading::Single);
}

// The same, with one engine thread per XCD. On a single thread the XCD that
// completes the grid and the XCD that owns the signal are driven by the same
// drain loop, so the wake that rouses a parked owner is never actually needed.
// Here they are on different threads and it is.
TEST(XcdDistributionTest, FanoutFiresTheCompletionSignalExactlyOnceThreaded) {
  run_fanout_completion_signal(Threading::ThreadPerXcd);
}

// The two ordering regressions above run on a single engine thread, where every
// XCD is driven by one drain loop and two command processors never actually run
// at the same time. That leaves the properties they check resting on an
// accident of the drain order: the cross-CP inbox lock ordering, the
// cross-thread shard handoff, and wake_all_xcds() rousing a peer parked behind a
// barrier are all unexercised.
//
// Re-run both with one thread per XCD. ExecutionPluginGroup already serializes
// the callbacks WorkgroupOrderPlugin records, so the observations stay valid.

TEST(XcdDistributionTest, DispatchIdsAreDisjointAcrossXcdsThreaded) {
  XcdDistributionFixture fx(Threading::ThreadPerXcd);
  ASSERT_GT(fx.loaded.engine_config.num_threads, 1u) << "fixture did not actually go concurrent";

  auto plugin = std::make_unique<WorkgroupOrderPlugin>();
  auto *order = plugin.get();
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::move(plugin)));
  fx.soc->set_plugin_group(group);

  // One queue per XCD, so the XCDs are opposing owners: each mints ids for its
  // own dispatches while concurrently holding shards minted by the other seven.
  std::vector<std::unique_ptr<test::AqlQueue>> queues;
  for (uint32_t qi = 0; qi < kTotalXcds; ++qi) {
    auto *cp = fx.soc->assign_queue_owner_cp(qi);
    ASSERT_NE(cp, nullptr);
    uint64_t ring = 0xF0000000ULL + qi * 0x100000ULL;
    queues.push_back(test::make_fanout_queue(fx.memory, cp, /*queue_id=*/qi + 1, ring));
    queues.back()->dispatch(kKdAddr, kTotalXcds * kWavefrontSize, kWavefrontSize);
  }

  fx.engine->run();

  EXPECT_EQ(order->dispatch_count(), kTotalXcds)
      << "two XCDs minted the same dispatch id and their completions merged";
  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}),
            uint64_t{kTotalXcds} * kTotalXcds);
}

TEST(XcdDistributionTest, BarrierBitWaitsForEveryXcdsShareThreaded) {
  XcdDistributionFixture fx(Threading::ThreadPerXcd);
  ASSERT_GT(fx.loaded.engine_config.num_threads, 1u) << "fixture did not actually go concurrent";

  auto plugin = std::make_unique<WorkgroupOrderPlugin>();
  auto *order = plugin.get();
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::move(plugin)));
  fx.soc->set_plugin_group(group);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
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

  auto ids = order->dispatch_ids();
  ASSERT_EQ(ids.size(), 2u) << "expected exactly two distinct dispatch ids";
  EXPECT_GT(order->first_dispatched(ids[1]), order->last_completed(ids[0]))
      << "an XCD began the barrier'd dispatch while a peer still ran the previous one";
}

// One matched begin/end pair is owed per packet, not per share, and the pair
// spans the whole grid rather than the owner's part of it. The hard case is a
// grid smaller than the XCD count whose owner draws an empty share: begin cannot
// be pinned to the XCD that read the packet, because that XCD never places a
// workgroup, so it is claimed by whichever XCD places the grid's first one --
// while end stays with the owner. Threaded, so the two are genuinely different
// CPs on different threads and the claim is a real race rather than a formality.
TEST(XcdDistributionTest, FanoutReportsOneExecutionPairWhenTheOwnerShareIsEmpty) {
  XcdDistributionFixture fx(Threading::ThreadPerXcd);
  ASSERT_GT(fx.loaded.engine_config.num_threads, 1u) << "fixture did not actually go concurrent";

  auto plugin = std::make_unique<ExecutionSpanPlugin>();
  auto *span = plugin.get();
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  ASSERT_TRUE(group->add(std::move(plugin)));
  fx.soc->set_plugin_group(group);

  // Fewer workgroups than XCDs, on a queue owned by an XCD past the end of the
  // grid, so the owner's share really is empty.
  constexpr uint32_t kWgs = 2;
  static_assert(kWgs < kTotalXcds, "the owner can only draw an empty share on a small grid");
  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/3);
  ASSERT_NE(cp, nullptr);
  const uint32_t owner_xcd = assigned_xcd_index(*fx.soc, cp);
  ASSERT_GE(owner_xcd, kWgs) << "the owning XCD must fall outside the grid for this test";

  auto queue = test::make_fanout_queue(fx.memory, cp);
  queue->dispatch(kKdAddr, kWgs * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  ASSERT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), kWgs);
  ASSERT_EQ(counts[owner_xcd], 0u)
      << "the owning XCD was supposed to draw an empty share, so this proves nothing";

  auto begins = span->begins();
  auto ends = span->ends();
  ASSERT_EQ(begins.size(), 1u) << "a fanned-out dispatch owes exactly one execution begin";
  ASSERT_EQ(ends.size(), 1u) << "a fanned-out dispatch owes exactly one execution end";
  EXPECT_EQ(begins.front().dispatch_id, ends.front().dispatch_id)
      << "the begin and end must name the same dispatch";
  EXPECT_EQ(span->workgroups_completed(), kWgs);
  EXPECT_LT(begins.front().step, ends.front().step);
  EXPECT_GT(ends.front().step, span->last_workgroup_step())
      << "execution end was reported before the grid's last workgroup retired";
}

// Scratch is the one resource a shard cannot size from its own share. A wave's
// scratch slot is indexed by its *grid-wide* workgroup id, so the high-index
// workgroups a peer XCD runs address the far end of the pool. Sizing the
// allocation from an XCD's own share would leave that end unbacked, and the
// allocator would then be re-entered mid-dispatch, remapping the pool VA out
// from under waves already spilling into it.
//
// Every fan-out test elsewhere uses a kernel with no scratch, so none of them
// reaches this path at all. Record what the allocator is actually asked for.
TEST(XcdDistributionTest, FanoutSizesScratchForTheWholeGridNotOneShare) {
  XcdDistributionFixture fx;
  ASSERT_EQ(fx.soc->num_xcds(), kTotalXcds);

  constexpr uint32_t kPrivateBytes = 64;
  constexpr uint64_t kPerWave = uint64_t{kPrivateBytes} * kWavefrontSize;
  constexpr uint64_t kWavesPerWg = 1;
  constexpr uint64_t kGridWide = kPerWave * kTotalCus * kWavesPerWg;
  install_scratch_kernel(*fx.memory, kPrivateBytes);

  auto vm_access = std::make_shared<SparseGpuVmAccess>(
      *fx.memory,
      SparseGpuVmAccess::HostWindow{.base = kDefaultScratchPool, .size = 0, .extent = kGridWide});
  constexpr uint32_t kProcessId = 17;
  const amdgpu::AddressSpaceHandle address_space =
      fx.soc->gpu_vm().register_translated(kProcessId, vm_access, vm_access);
  ASSERT_TRUE(address_space);

  struct ScratchRequest {
    uint64_t gpu_va;
    size_t size;
  };
  std::vector<ScratchRequest> requests;
  std::mutex requests_mutex;
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi) {
    fx.soc->xcd(xi)->command_processor()->set_scratch_backing_allocator(
        [&, vm_access](uint32_t process_id, uint64_t gpu_va, size_t size) -> bool {
          {
            std::lock_guard<std::mutex> lock(requests_mutex);
            requests.push_back({gpu_va, size});
          }
          EXPECT_EQ(process_id, kProcessId);
          EXPECT_EQ(gpu_va, kDefaultScratchPool);
          return vm_access->provision_host_window(size);
        });
  }

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  auto queue = test::make_fanout_queue(
      fx.memory, cp, /*queue_id=*/1, test::AqlQueue::DEFAULT_RING_ADDR, address_space, kProcessId);
  queue->dispatch(kScratchKdAddr, kTotalCus * kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  // The grid really was spread, or the sizing claim below is untested.
  auto counts = fx.soc->dispatched_workgroups_per_xcd();
  for (uint32_t xi = 0; xi < kTotalXcds; ++xi)
    ASSERT_EQ(counts[xi], kTotalCus / kTotalXcds) << "xcd" << xi;

  ASSERT_FALSE(requests.empty()) << "the scratch path was never reached";

  // Every request is sized against the whole grid, not the requesting XCD's
  // share. This is the claim that matters: a wave's slot is indexed by its
  // grid-wide workgroup id, so a share-sized pool would be an eighth as large
  // and the workgroups above it would spill past the end of it.
  const uint64_t share_wide = kPerWave * (kTotalCus / kTotalXcds) * kWavesPerWg;
  for (const auto &request : requests) {
    EXPECT_EQ(request.size, kGridWide)
        << "scratch was sized from a shard's share rather than from the whole grid";
    EXPECT_NE(request.size, share_wide) << "sized from this XCD's own share";
  }

  EXPECT_EQ(requests.size(), 1u)
      << "the shared GPUVM mapping should prevent per-XCD scratch reprovisioning";
}

TEST(XcdDistributionTest, ScratchAllocatorFailureStopsBeforeWaveAdmission) {
  XcdDistributionFixture fx;
  constexpr uint32_t kPrivateBytes = 64;
  install_scratch_kernel(*fx.memory, kPrivateBytes);

  auto vm_access = std::make_shared<SparseGpuVmAccess>(
      *fx.memory, SparseGpuVmAccess::HostWindow{.base = kDefaultScratchPool, .size = 0});
  const amdgpu::AddressSpaceHandle address_space =
      fx.soc->gpu_vm().register_translated(/*vmid=*/17, vm_access, vm_access);
  ASSERT_TRUE(address_space);

  uint32_t allocator_calls = 0;
  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  cp->set_scratch_backing_allocator([&](uint32_t, uint64_t, size_t) {
    ++allocator_calls;
    return false;
  });
  test::AqlQueue queue(fx.memory, cp, test::AqlQueue::DEFAULT_RING_ADDR,
                       test::AqlQueue::DEFAULT_RING_SIZE, test::AqlQueue::DEFAULT_READ_PTR_ADDR,
                       test::AqlQueue::DEFAULT_WRITE_PTR_ADDR,
                       test::AqlQueue::DEFAULT_DOORBELL_ADDR, /*xcd_fanout=*/false,
                       /*queue_id=*/1, address_space, /*process_id=*/17);
  queue.dispatch(kScratchKdAddr, kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  EXPECT_EQ(allocator_calls, 1u);
  const auto counts = fx.soc->dispatched_workgroups_per_xcd();
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), 0u)
      << "a failed scratch allocation admitted a wavefront";
}

TEST(XcdDistributionTest, ScratchRequiresTheCompleteWaveSlice) {
  XcdDistributionFixture fx;
  constexpr uint32_t kPrivateBytes = 64;
  install_scratch_kernel(*fx.memory, kPrivateBytes);

  // The first byte resolves, which is enough for the old point probe, while the
  // rest of this wave's 4 KiB scratch slice is deliberately absent.
  auto vm_access = std::make_shared<SparseGpuVmAccess>(
      *fx.memory, SparseGpuVmAccess::HostWindow{.base = kDefaultScratchPool, .size = 1});
  const amdgpu::AddressSpaceHandle address_space =
      fx.soc->gpu_vm().register_translated(/*vmid=*/17, vm_access, vm_access);
  ASSERT_TRUE(address_space);

  auto *cp = fx.soc->assign_queue_owner_cp(/*queue_ordinal=*/0);
  ASSERT_NE(cp, nullptr);
  test::AqlQueue queue(fx.memory, cp, test::AqlQueue::DEFAULT_RING_ADDR,
                       test::AqlQueue::DEFAULT_RING_SIZE, test::AqlQueue::DEFAULT_READ_PTR_ADDR,
                       test::AqlQueue::DEFAULT_WRITE_PTR_ADDR,
                       test::AqlQueue::DEFAULT_DOORBELL_ADDR, /*xcd_fanout=*/false,
                       /*queue_id=*/1, address_space, /*process_id=*/17);
  queue.dispatch(kScratchKdAddr, kWavefrontSize, kWavefrontSize);

  fx.engine->run();

  const auto counts = fx.soc->dispatched_workgroups_per_xcd();
  EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), uint64_t{0}), 0u)
      << "a one-byte scratch mapping was accepted for a complete wave";
}
