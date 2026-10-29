// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "config_test_util.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/compute_queue_binding_factory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/soc.h"
#include "simdojo/sim/simulation.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {
namespace {

class Pm4QueueMemory final : public AddressSpaceTranslator, public PhysicalMemoryAccess {
public:
  Pm4QueueMemory() : bytes_(0x4000) {}

  VmTranslationResult translate(uint64_t address, std::size_t size, VmAccessKind) const override {
    if (size == 0 || address > bytes_.size() || size > bytes_.size() - address)
      return {.outcome = VmAccessOutcome::Faulted, .translation = {}};
    return {
        .outcome = VmAccessOutcome::Complete,
        .translation = {.domain = VmMemoryDomain::System,
                        .address = address,
                        .contiguous_bytes = bytes_.size() - address,
                        .mtype = Mtype::RW,
                        .permissions = {.readable = true, .writable = true, .executable = true}}};
  }

  VmAccessOutcome read(VmMemoryDomain, uint64_t address, std::span<std::byte> bytes) override {
    if (unavailable_read_ && address == *unavailable_read_)
      return VmAccessOutcome::Unavailable;
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::ranges::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(address), bytes.size(),
                        bytes.begin());
    return VmAccessOutcome::Complete;
  }

  VmAccessOutcome write(VmMemoryDomain, uint64_t address,
                        std::span<const std::byte> bytes) override {
    if (write_failure_ && address == write_failure_->first)
      return write_failure_->second;
    if (address > bytes_.size() || bytes.size() > bytes_.size() - address)
      return VmAccessOutcome::Faulted;
    std::ranges::copy(bytes, bytes_.begin() + static_cast<std::ptrdiff_t>(address));
    return VmAccessOutcome::Complete;
  }

  AtomicLoadResult atomic_load(VmMemoryDomain, uint64_t address, uint32_t width) override {
    if (unavailable_read_ && address == *unavailable_read_)
      return {.outcome = VmAccessOutcome::Unavailable};
    if ((width != 4 && width != 8) || address % width != 0 || address > bytes_.size() ||
        width > bytes_.size() - address) {
      return {.outcome = VmAccessOutcome::Malformed};
    }
    uint64_t value = 0;
    std::memcpy(&value, bytes_.data() + address, width);
    return {.outcome = VmAccessOutcome::Complete, .value = value};
  }

  VmAccessOutcome atomic_store(VmMemoryDomain, uint64_t address, uint32_t width,
                               uint64_t value) override {
    if (unavailable_store_ && address == *unavailable_store_)
      return VmAccessOutcome::Unavailable;
    if (faulted_store_ && address == *faulted_store_)
      return VmAccessOutcome::Faulted;
    if ((width != 4 && width != 8) || address % width != 0 || address > bytes_.size() ||
        width > bytes_.size() - address) {
      return VmAccessOutcome::Malformed;
    }
    std::memcpy(bytes_.data() + address, &value, width);
    return VmAccessOutcome::Complete;
  }

  AtomicCompareExchangeResult compare_exchange(VmMemoryDomain, uint64_t, uint32_t, uint64_t,
                                               uint64_t) override {
    return {.outcome = VmAccessOutcome::Malformed};
  }

  template <typename T> void store(uint64_t address, T value) {
    std::memcpy(bytes_.data() + address, &value, sizeof(value));
  }

  template <typename T> T load(uint64_t address) const {
    T value{};
    std::memcpy(&value, bytes_.data() + address, sizeof(value));
    return value;
  }

  void fail_write(uint64_t address, VmAccessOutcome outcome) {
    write_failure_ = std::pair{address, outcome};
  }
  void make_read_unavailable(uint64_t address) { unavailable_read_ = address; }
  void make_store_unavailable(uint64_t address) { unavailable_store_ = address; }
  void make_store_faulted(uint64_t address) { faulted_store_ = address; }
  void make_available() {
    unavailable_read_.reset();
    unavailable_store_.reset();
    faulted_store_.reset();
    write_failure_.reset();
  }

private:
  std::vector<std::byte> bytes_;
  std::optional<std::pair<uint64_t, VmAccessOutcome>> write_failure_;
  std::optional<uint64_t> unavailable_read_;
  std::optional<uint64_t> unavailable_store_;
  std::optional<uint64_t> faulted_store_;
};

class Pm4ComputeQueueTest : public ::testing::Test {
protected:
  void SetUp() override {
    memory = std::make_shared<Pm4QueueMemory>();
    address_space = gpu_vm.register_translated(1, memory, memory);
    ASSERT_TRUE(address_space);
    memory->store<uint32_t>(kReadPointer, 0);
    config::LoadedConfig loaded;
    ASSERT_TRUE(test::load_valid_config_from_string(R"({"max_ticks":100000,"num_threads":1,
      "vm":{"arch":"cdna3"},"topology":{"root":{"name":"soc","type":"soc","children":[
      {"name":"vram","type":"gpu_memory"},{"name":"xcd0","type":"xcd","children":[
      {"name":"l2","type":"l2_cache"},{"name":"cp","type":"command_processor"},
      {"name":"se0","type":"shader_engine","children":[{"name":"cu0","type":"compute_unit"}]}]}]},
      "links":[{"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2},
      {"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10}]}})",
                                                    loaded));
    cp = loaded.soc()->xcd(0)->command_processor();
    l2 = loaded.soc()->xcd(0)->l2_cache();
    gpu_memory = loaded.soc()->memory();
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    engine->create();
    cp->set_gpu_vm(&gpu_vm, address_space);
  }

  uint64_t attach(Pm4PacketCallbacks callbacks, uint64_t ring = kRing,
                  uint64_t read_pointer = kReadPointer, uint32_t ring_bytes = kRingBytes) {
    return cp->register_queue({.address_space = address_space,
                               .queue_id = next_queue_id++,
                               .ring_base_va = ring,
                               .ring_size = ring_bytes,
                               .read_ptr_va = read_pointer,
                               .packet_format = QueuePacketFormat::Pm4,
                               .packet_callbacks = std::move(callbacks)});
  }

  void service() { (void)engine->step(); }

  static constexpr uint64_t kRing = 0x100;
  static constexpr uint32_t kRingBytes = 64;
  static constexpr uint64_t kReadPointer = 0x200;

  GpuVm gpu_vm;
  std::shared_ptr<Pm4QueueMemory> memory;
  AddressSpaceHandle address_space;
  std::unique_ptr<simdojo::SimulationEngine> engine;
  CommandProcessor *cp = nullptr;
  L2Cache *l2 = nullptr;
  GpuMemory *gpu_memory = nullptr;
  uint32_t next_queue_id = 1;
};

TEST_F(Pm4ComputeQueueTest, RejectsSubmissionsForUnknownQueueOrProcess) {
  ASSERT_TRUE(
      cp->register_drm_queue({.address_space = address_space, .process_id = 7, .queue_id = 1}));
  constexpr uint32_t kOutput = 0x300;
  const std::array words{0xc0033700u, 0x100u, kOutput, 0u, 0xdeadbeefu};
  for (size_t word_index = 0; word_index < words.size(); ++word_index)
    memory->store<uint32_t>(kRing + word_index * 4, words[word_index]);

  uint32_t completions = 0;
  for (const auto &[queue_id, process_id] : {std::pair{2u, 7u}, std::pair{1u, 8u}}) {
    Pm4Submission submission;
    submission.buffers.push_back({.address = kRing, .dwords = words.size()});
    submission.complete = [&](bool) { ++completions; };
    EXPECT_FALSE(cp->submit_pm4(queue_id, process_id, std::move(submission)));
  }

  service();
  service();
  EXPECT_EQ(completions, 0u);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
}

TEST_F(Pm4ComputeQueueTest, RejectsSubmissionsForNativeRingQueue) {
  const auto registration = cp->register_queue({.address_space = address_space,
                                                .process_id = 7,
                                                .queue_id = 1,
                                                .ring_base_va = kRing,
                                                .ring_size = kRingBytes,
                                                .read_ptr_va = kReadPointer,
                                                .packet_format = QueuePacketFormat::Pm4});
  ASSERT_NE(registration, 0u);
  constexpr uint32_t kOutput = 0x300;
  const std::array words{0xc0033700u, 0x100u, kOutput, 0u, 0xdeadbeefu};
  for (size_t word_index = 0; word_index < words.size(); ++word_index)
    memory->store<uint32_t>(kRing + word_index * 4, words[word_index]);

  uint32_t completions = 0;
  Pm4Submission submission;
  submission.buffers.push_back({.address = kRing, .dwords = words.size()});
  submission.complete = [&](bool) { ++completions; };
  EXPECT_FALSE(cp->submit_pm4(1, 7, std::move(submission)));

  service();
  service();
  EXPECT_EQ(completions, 0u);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
}

TEST_F(Pm4ComputeQueueTest, SubmissionDefersEffectsUntilTheCpServicesTheQueue) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t writes = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);

  EXPECT_EQ(cp->notify_pm4_queue_doorbell(registration, 3), QueueSubmissionStatus::Accepted);
  EXPECT_EQ(writes, 0u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);

  service();
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
}

TEST_F(Pm4ComputeQueueTest, NativeRingResumesAfterPublishingAnInitiallyUnreadyGart) {
  constexpr uint64_t aperture = 0x10000;
  constexpr uint64_t page_table = 0x1000;
  constexpr uint64_t physical_page = 0x2000;
  // One readable/writable system page, also marked as a leaf PDE.
  constexpr uint64_t page_flags = (uint64_t{1} << 63) | 0x63;
  memory->store<uint64_t>(page_table, physical_page | page_flags);
  memory->store<uint32_t>(physical_page + kRing, 0xc0017900);
  memory->store<uint32_t>(physical_page + kRing + 4, 0x40);
  memory->store<uint32_t>(physical_page + kRing + 8, 0xdeadbeef);
  address_space = gpu_vm.initialize_gart_address_space();
  ASSERT_TRUE(address_space);
  ASSERT_FALSE(gpu_vm.lookup(address_space)->ready);
  uint32_t writes = 0;
  const auto registration = attach({.write_uconfig_register =
                                        [&](uint64_t, uint32_t value) {
                                          EXPECT_EQ(value, 0xdeadbeefu);
                                          ++writes;
                                          return Pm4RegisterWriteStatus::Complete;
                                        }},
                                   aperture + kRing, aperture + kReadPointer);
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 3), QueueSubmissionStatus::Accepted);

  service();
  EXPECT_EQ(writes, 0u);
  EXPECT_EQ(memory->load<uint32_t>(physical_page + kReadPointer), 0u);
  ASSERT_TRUE(gpu_vm.publish_gart(
      {.page_table_base = page_table, .aperture_start = aperture, .aperture_end = aperture + 0xfff},
      memory));
  service();
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(physical_page + kReadPointer), 3u);
  service();
  EXPECT_EQ(writes, 1u);
}

TEST_F(Pm4ComputeQueueTest, SubmittedStreamResumesAfterPublishingAnInitiallyUnreadyGart) {
  constexpr uint64_t aperture = 0x10000;
  constexpr uint64_t page_table = 0x1000;
  constexpr uint64_t physical_page = 0x2000;
  constexpr uint64_t page_flags = (uint64_t{1} << 63) | 0x63;
  memory->store<uint64_t>(page_table, physical_page | page_flags);
  memory->store<uint32_t>(physical_page + kRing, 0xc0033700);
  memory->store<uint32_t>(physical_page + kRing + 4, 0x100);
  memory->store<uint32_t>(physical_page + kRing + 8, aperture + kReadPointer);
  memory->store<uint32_t>(physical_page + kRing + 12, 0);
  memory->store<uint32_t>(physical_page + kRing + 16, 0xdeadbeef);
  address_space = gpu_vm.initialize_gart_address_space();
  ASSERT_TRUE(address_space);
  ASSERT_TRUE(
      cp->register_drm_queue({.address_space = address_space, .process_id = 1, .queue_id = 1}));
  uint32_t completions = 0;
  Pm4Submission submission;
  submission.buffers.push_back({.address = aperture + kRing, .dwords = 5});
  submission.complete = [&](bool success) {
    EXPECT_TRUE(success);
    ++completions;
  };
  ASSERT_TRUE(cp->submit_pm4(1, 1, std::move(submission)));

  service();
  EXPECT_EQ(completions, 0u);
  EXPECT_EQ(memory->load<uint32_t>(physical_page + kReadPointer), 0u);
  ASSERT_TRUE(gpu_vm.publish_gart(
      {.page_table_base = page_table, .aperture_start = aperture, .aperture_end = aperture + 0xfff},
      memory));
  service();
  EXPECT_EQ(completions, 1u);
  EXPECT_EQ(memory->load<uint32_t>(physical_page + kReadPointer), 0xdeadbeefu);
  service();
  EXPECT_EQ(completions, 1u);
}

TEST_F(Pm4ComputeQueueTest, BlockedSubmissionsKeepStallRecheckWorkBounded) {
  bool ready = false;
  uint32_t completed = 0;
  for (uint32_t id : {1u, 2u}) {
    ASSERT_TRUE(cp->register_drm_queue({.address_space = address_space, .queue_id = id}));
    Pm4Submission submission;
    submission.ready = [&](util::DiagnosticEmitter) -> util::FailureOr<bool> { return ready; };
    submission.complete = [&](bool success) {
      EXPECT_TRUE(success);
      ++completed;
    };
    ASSERT_TRUE(cp->submit_pm4(id, 0, std::move(submission)));
  }

  // A persistent dependency must back off even when several queues and several
  // service sites request a retry. Duplicate retries would multiply at each tick
  // and prevent later shader or dependency-completion events from progressing.
  while (engine->context(cp->partition_id()).current_tick() < 20000u)
    ASSERT_TRUE(engine->step());
  EXPECT_LT(cp->doorbell_handle_count_for_test(), 64u);
  EXPECT_EQ(completed, 0u);

  ready = true;
  // Readiness alone must be observed by the coalesced timer.
  for (uint32_t i = 0; i < 8 && completed != 2; ++i)
    service();
  EXPECT_EQ(completed, 2u);
}

TEST_F(Pm4ComputeQueueTest, FailedDependencyCancelsQueuedSubmissionsOnce) {
  ASSERT_TRUE(cp->register_drm_queue({.address_space = address_space, .queue_id = 1}));
  uint32_t completions = 0;
  for (uint32_t i = 0; i < 2; ++i) {
    Pm4Submission submission;
    submission.ready = [](util::DiagnosticEmitter emit_error) -> util::FailureOr<bool> {
      return emit_error.emit() << "PM4 submission dependency failed";
    };
    submission.complete = [&](bool success) {
      EXPECT_FALSE(success);
      ++completions;
    };
    ASSERT_TRUE(cp->submit_pm4(1, 0, std::move(submission)));
  }
  EXPECT_NO_THROW(service());
  EXPECT_EQ(completions, 2u);
  service();
  EXPECT_EQ(completions, 2u);
}

class Pm4FlushPlugin final : public ExecutionPlugin {
public:
  Pm4FlushPlugin() : ExecutionPlugin("pm4-flush") {}
  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &) override {
    ++processed;
    if (on_processed)
      on_processed();
  }
  void onAmdgpuDispatchExecutionBegin(uint32_t id) override { begins.push_back(id); }
  void onAmdgpuDispatchExecutionEnd(uint32_t id) override { ends.push_back(id); }
  void onAmdgpuWorkgroupCompleted(uint32_t, uint32_t) override {
    if (on_completed)
      on_completed();
  }
  std::function<void()> on_processed;
  std::function<void()> on_completed;
  uint32_t processed = 0;
  std::vector<uint32_t> begins, ends;
};

class Pm4DispatchFlushTest : public Pm4ComputeQueueTest {
protected:
  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(Pm4ComputeQueueTest::SetUp());
    l2->set_backing_memory(gpu_memory);
    l2->set_gpu_vm(&gpu_vm);
    auto owned_plugin = std::make_unique<Pm4FlushPlugin>();
    plugin = owned_plugin.get();
    auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    ASSERT_TRUE(group->add(std::move(owned_plugin)));
    cp->set_plugin_group(group);
  }

  void inject_flush_failure() {
    std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
    dirty.fill(0xab);
    ASSERT_EQ(l2->writeback_line(kDirtyLine, dirty.data(), Mtype::RW, 1),
              VmAccessOutcome::Complete);
    memory->fail_write(kDirtyLine, VmAccessOutcome::Faulted);
  }

  void submit_dispatch(bool begun) {
    // SET_SH_REG sets 1x1x1 threads and the program address; dispatch then
    // precedes a register callback that must not execute after either flush gate.
    const std::array words{0xc0037600u,
                           kPm4ComputeNumThreadX,
                           1u,
                           1u,
                           1u,
                           0xc0017600u,
                           kPm4ComputePgmLo,
                           0x20u,
                           0xc0031500u,
                           begun ? 1u : 0u,
                           1u,
                           1u,
                           1u,
                           0xc0017900u,
                           0x40u,
                           0xdeadbeefu};
    memory->store<uint32_t>(0x2000, 0xbf810000u); // s_endpgm
    for (size_t i = 0; i < words.size(); ++i)
      memory->store<uint32_t>(kRing + i * 4, words[i]);
    ASSERT_TRUE(cp->register_drm_queue(
        {.address_space = address_space,
         .queue_id = 1,
         .packet_callbacks = {.write_uconfig_register = [&](uint64_t, uint32_t) {
           ++successors;
           return Pm4RegisterWriteStatus::Complete;
         }}}));
    Pm4Submission submission;
    submission.buffers.push_back({.address = kRing, .dwords = words.size()});
    submission.complete = [&](bool success) {
      EXPECT_FALSE(success);
      ++completions;
    };
    ASSERT_TRUE(cp->submit_pm4(1, 0, std::move(submission)));
  }

  void expect_cancelled() {
    for (uint32_t i = 0; i < 100 && !completions; ++i)
      service();
    EXPECT_EQ(completions, 1u);
    EXPECT_EQ(successors, 0u);
    service();
    EXPECT_EQ(completions, 1u);
  }

  static constexpr uint64_t kDirtyLine = 0x1000;
  Pm4FlushPlugin *plugin = nullptr;
  uint32_t completions = 0, successors = 0;
};

TEST_F(Pm4DispatchFlushTest, AdmissionFlushFailurePreventsDispatchAndSuccessor) {
  ASSERT_NO_FATAL_FAILURE(inject_flush_failure());
  ASSERT_NO_FATAL_FAILURE(submit_dispatch(false));
  expect_cancelled();
  EXPECT_EQ(plugin->processed, 0u);
}

TEST_F(Pm4DispatchFlushTest, RetirementFlushFailurePreventsSuccessor) {
  plugin->on_processed = [&] { inject_flush_failure(); };
  ASSERT_NO_FATAL_FAILURE(submit_dispatch(false));
  expect_cancelled();
  EXPECT_EQ(plugin->processed, 1u);
  EXPECT_TRUE(plugin->begins.empty());
  EXPECT_TRUE(plugin->ends.empty());
}

TEST_F(Pm4DispatchFlushTest, RetirementFlushFailureEndsCompletedDispatch) {
  plugin->on_completed = [&] { inject_flush_failure(); };
  ASSERT_NO_FATAL_FAILURE(submit_dispatch(true));
  expect_cancelled();
  EXPECT_EQ(plugin->processed, 1u);
  ASSERT_EQ(plugin->begins.size(), 1u);
  EXPECT_EQ(plugin->ends, plugin->begins);
}

TEST_F(Pm4ComputeQueueTest, ForeignFlushFailureDoesNotBlockFreshQueues) {
  l2->set_backing_memory(gpu_memory);
  l2->set_gpu_vm(&gpu_vm);
  constexpr uint64_t kDirtyLine = 0x1000;
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0xab);
  ASSERT_EQ(l2->writeback_line(kDirtyLine, dirty.data(), Mtype::RW, 1), VmAccessOutcome::Complete);
  memory->fail_write(kDirtyLine, VmAccessOutcome::Faulted);
  auto healthy_memory = std::make_shared<Pm4QueueMemory>();
  const auto healthy_space = gpu_vm.register_translated(2, healthy_memory, healthy_memory);
  ASSERT_TRUE(healthy_space);
  constexpr uint32_t kOutput = 0x300;
  const std::array words{0xc0033700u, 0x100u, kOutput, 0u, 0xdeadbeefu};
  for (size_t i = 0; i < words.size(); ++i)
    healthy_memory->store<uint32_t>(kRing + i * 4, words[i]);
  uint32_t completions = 0;
  for (uint32_t id : {1u, 2u}) {
    ASSERT_TRUE(cp->register_drm_queue({.address_space = healthy_space, .queue_id = id}));
    healthy_memory->store<uint32_t>(kOutput, 0);
    Pm4Submission submission;
    submission.buffers.push_back({.address = kRing, .dwords = words.size()});
    submission.complete = [&](bool success) {
      EXPECT_TRUE(success);
      ++completions;
    };
    ASSERT_TRUE(cp->submit_pm4(id, 0, std::move(submission)));
    for (uint32_t i = 0; i < 10 && completions != id; ++i)
      service();
    EXPECT_EQ(completions, id);
    EXPECT_EQ(healthy_memory->load<uint32_t>(kOutput), 0xdeadbeefu);
    EXPECT_EQ(memory->load<uint32_t>(kDirtyLine), 0u);
  }
  // Scoped maintenance must retain the failed owner's dirty bytes for retry.
  memory->make_available();
  EXPECT_EQ(l2->flush_address_space(1), VmAccessOutcome::Complete);
  EXPECT_EQ(memory->load<uint32_t>(kDirtyLine), 0xababababu);
}

class Pm4QueueCacheFlushFailureTest : public Pm4ComputeQueueTest,
                                      public ::testing::WithParamInterface<VmAccessOutcome> {};

TEST_P(Pm4QueueCacheFlushFailureTest, CancellationCompletesEvenWhenCleanupFlushAlsoFails) {
  l2->set_backing_memory(gpu_memory);
  l2->set_gpu_vm(&gpu_vm);
  constexpr uint64_t kDirtyLine = 0x1000;
  constexpr uint32_t kOutput = 0x300;
  const std::array words{0xc0033700u, 0x100u, kOutput, 0u, 0xdeadbeefu};
  for (size_t i = 0; i < words.size(); ++i)
    memory->store<uint32_t>(kRing + i * 4, words[i]);
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  dirty.fill(0xab);
  ASSERT_EQ(l2->writeback_line(kDirtyLine, dirty.data(), Mtype::RW, 1), VmAccessOutcome::Complete);
  memory->fail_write(kDirtyLine, GetParam());
  ASSERT_TRUE(cp->register_drm_queue({.address_space = address_space, .queue_id = 1}));
  uint32_t completions = 0;
  for (uint32_t i = 0; i < 2; ++i) {
    Pm4Submission submission;
    submission.buffers.push_back({.address = kRing, .dwords = words.size()});
    submission.complete = [&](bool success) {
      EXPECT_FALSE(success);
      ++completions;
    };
    ASSERT_TRUE(cp->submit_pm4(1, 0, std::move(submission)));
  }
  EXPECT_NO_THROW(service());
  EXPECT_EQ(completions, 2u);
  EXPECT_EQ(memory->load<uint32_t>(kOutput), 0u);
  EXPECT_EQ(memory->load<uint32_t>(kDirtyLine), 0u);
  service();
  EXPECT_EQ(completions, 2u);
}

INSTANTIATE_TEST_SUITE_P(NonComplete, Pm4QueueCacheFlushFailureTest,
                         ::testing::Values(VmAccessOutcome::Unavailable, VmAccessOutcome::Faulted,
                                           VmAccessOutcome::Malformed));

TEST_F(Pm4ComputeQueueTest, ProducerCompletionWakesEarlierConsumerPromptly) {
  bool fence = false;
  uint32_t consumer_done = 0, producer_done = 0;
  ASSERT_TRUE(cp->register_drm_queue({.address_space = address_space, .queue_id = 1}));
  ASSERT_TRUE(cp->register_drm_queue({.address_space = address_space, .queue_id = 2}));
  Pm4Submission consumer;
  consumer.ready = [&](util::DiagnosticEmitter) -> util::FailureOr<bool> { return fence; };
  consumer.complete = [&](bool success) {
    EXPECT_TRUE(success);
    ++consumer_done;
  };
  ASSERT_TRUE(cp->submit_pm4(1, 0, std::move(consumer)));
  auto tick = [&] { return engine->context(cp->partition_id()).current_tick(); };
  while (tick() < 20000)
    ASSERT_TRUE(engine->step());
  Pm4Submission producer;
  producer.complete = [&](bool success) {
    EXPECT_TRUE(success);
    fence = true;
    ++producer_done;
  };
  const auto start = tick();
  ASSERT_TRUE(cp->submit_pm4(2, 0, std::move(producer)));
  ASSERT_TRUE(engine->step());
  ASSERT_EQ(producer_done, 1u);
  ASSERT_EQ(consumer_done, 0u);
  for (unsigned i = 0; i < 8 && consumer_done == 0; ++i)
    ASSERT_TRUE(engine->step());
  EXPECT_EQ(consumer_done, 1u);
  EXPECT_LE(tick() - start, 1u);
  const auto passes = cp->doorbell_handle_count_for_test();
  unsigned stale_steps = 0;
  while (engine->step())
    ASSERT_LT(++stale_steps, 20u);
  // Completing both submissions may request one final retry, but superseded
  // long deadlines must not enter the CP handler after all work is retired.
  EXPECT_LE(cp->doorbell_handle_count_for_test() - passes, 1u);
}

TEST_F(Pm4ComputeQueueTest, BoundsEachQueueServiceTurn) {
  constexpr uint64_t kLargeRing = 0x400;
  constexpr uint64_t kLargeReadPointer = 0xe00;
  constexpr uint32_t kPacketCount = 257;
  constexpr uint32_t kLargeRingBytes = 512 * sizeof(uint32_t);
  for (uint32_t packet = 0; packet < kPacketCount; ++packet)
    memory->store<uint32_t>(kLargeRing + packet * sizeof(uint32_t), 0xffff1000);
  memory->store<uint32_t>(kLargeReadPointer, 0);

  ASSERT_TRUE(cp->register_drm_queue({.address_space = address_space, .queue_id = 99}));
  Pm4Submission blocked;
  blocked.ready = [](util::DiagnosticEmitter) -> util::FailureOr<bool> { return false; };
  ASSERT_TRUE(cp->submit_pm4(99, 0, std::move(blocked)));
  auto tick = [&] { return engine->context(cp->partition_id()).current_tick(); };
  while (tick() < 20000)
    ASSERT_TRUE(engine->step());
  const auto start = tick();

  const auto registration = attach({}, kLargeRing, kLargeReadPointer, kLargeRingBytes);
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, kPacketCount),
            QueueSubmissionStatus::Accepted);

  service();
  EXPECT_EQ(memory->load<uint32_t>(kLargeReadPointer), 256u);

  service();
  EXPECT_EQ(memory->load<uint32_t>(kLargeReadPointer), kPacketCount);
  EXPECT_LE(tick() - start, 1u);
}

TEST_F(Pm4ComputeQueueTest, RetainsItsAddressSpaceUntilDetach) {
  const uint64_t registration = attach({});
  ASSERT_NE(registration, 0u);

  EXPECT_FALSE(gpu_vm.unregister_address_space(address_space));
  EXPECT_TRUE(cp->unregister_pm4_queue_registration(registration));
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST_F(Pm4ComputeQueueTest, CursorPublicationRetryDoesNotReplayThePacketEffect) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t writes = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 3), QueueSubmissionStatus::Accepted);
  memory->make_store_unavailable(kReadPointer);

  service();
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);

  memory->make_available();
  service();
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
}

TEST_F(Pm4ComputeQueueTest, RootReplacementDoesNotMovePendingCursorPublication) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t writes = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 3), QueueSubmissionStatus::Accepted);
  memory->make_store_unavailable(kReadPointer);

  service();
  EXPECT_EQ(writes, 1u);

  auto replacement = std::make_shared<Pm4QueueMemory>();
  replacement->store<uint32_t>(kReadPointer, 99);
  ASSERT_TRUE(gpu_vm.replace_translated(address_space, replacement, replacement));
  memory->make_available();

  service();
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_EQ(replacement->load<uint32_t>(kReadPointer), 99u);
}

TEST_F(Pm4ComputeQueueTest, RootReplacementDoesNotMoveBlockedPacketFetch) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t written_value = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t value) {
    written_value = value;
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 3), QueueSubmissionStatus::Accepted);
  memory->make_read_unavailable(kRing);

  service();
  EXPECT_EQ(written_value, 0u);

  auto replacement = std::make_shared<Pm4QueueMemory>();
  replacement->store<uint32_t>(kRing, 0xc0017900);
  replacement->store<uint32_t>(kRing + 4, 0x40);
  replacement->store<uint32_t>(kRing + 8, 0x12345678);
  replacement->store<uint32_t>(kReadPointer, 99);
  ASSERT_TRUE(gpu_vm.replace_translated(address_space, replacement, replacement));
  memory->make_available();

  service();
  EXPECT_EQ(written_value, 0xdeadbeefu);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_EQ(replacement->load<uint32_t>(kReadPointer), 99u);
}

TEST_F(Pm4ComputeQueueTest, LaterDoorbellUsesFreshSnapshotAfterBlockedTransaction) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0x11111111);
  std::vector<uint32_t> values;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t value) {
    values.push_back(value);
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 3), QueueSubmissionStatus::Accepted);
  memory->make_store_unavailable(kReadPointer);

  service();
  ASSERT_EQ(values.size(), 1u);
  EXPECT_EQ(values[0], 0x11111111u);

  auto replacement = std::make_shared<Pm4QueueMemory>();
  replacement->store<uint32_t>(kRing + 12, 0xc0017900);
  replacement->store<uint32_t>(kRing + 16, 0x40);
  replacement->store<uint32_t>(kRing + 20, 0x22222222);
  replacement->store<uint32_t>(kReadPointer, 3);
  ASSERT_TRUE(gpu_vm.replace_translated(address_space, replacement, replacement));
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 6), QueueSubmissionStatus::Accepted);
  memory->make_available();

  service();
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  service();
  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values[1], 0x22222222u);
  EXPECT_EQ(replacement->load<uint32_t>(kReadPointer), 6u);
}

TEST_F(Pm4ComputeQueueTest, PublishesEarlierProgressBeforeRetryingALaterFetch) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  memory->store<uint32_t>(kRing + 12, 0xffff1000);
  uint32_t writes = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 4), QueueSubmissionStatus::Accepted);
  memory->make_read_unavailable(kRing + 12);

  service();
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);

  memory->make_available();
  service();
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 4u);
}

TEST_F(Pm4ComputeQueueTest, RejectsInvalidRingsAtRegistration) {
  EXPECT_EQ(cp->register_pm4_queue({.address_space = address_space,
                                    .ring_base = kRing + 1,
                                    .ring_size_bytes = kRingBytes,
                                    .consumer_pointer_address = kReadPointer,
                                    .initial_consumer_cursor = std::nullopt,
                                    .packet_callbacks = {}}),
            0u);
  EXPECT_EQ(cp->register_pm4_queue({.address_space = address_space,
                                    .ring_base = kRing,
                                    .ring_size_bytes = kRingBytes - 1,
                                    .consumer_pointer_address = kReadPointer,
                                    .initial_consumer_cursor = std::nullopt,
                                    .packet_callbacks = {}}),
            0u);
  EXPECT_EQ(cp->register_pm4_queue({.address_space = address_space,
                                    .ring_base = kRing,
                                    .ring_size_bytes = kRingBytes,
                                    .consumer_pointer_address = kReadPointer + sizeof(uint16_t),
                                    .initial_consumer_cursor = std::nullopt,
                                    .packet_callbacks = {}}),
            0u);
}

TEST_F(Pm4ComputeQueueTest, NormalizesWrappedNativeProducerBeforeRetainingProgress) {
  constexpr uint32_t kRingDwords = 4;
  memory->store<uint64_t>(kReadPointer, 3);
  memory->store<uint32_t>(kRing + 3 * sizeof(uint32_t), 0xc0017900);
  memory->store<uint32_t>(kRing, 0x40);
  memory->store<uint32_t>(kRing + sizeof(uint32_t), 0xdeadbeef);

  uint32_t writes = 0;
  const auto registration = cp->register_pm4_queue({
      .address_space = address_space,
      .ring_base = kRing,
      .ring_size_bytes = kRingDwords * sizeof(uint32_t),
      .consumer_pointer_address = kReadPointer,
      .initial_consumer_cursor = 3,
      .packet_callbacks = {.write_uconfig_register =
                               [&](uint64_t, uint32_t) {
                                 ++writes;
                                 return Pm4RegisterWriteStatus::Complete;
                               }},
  });
  ASSERT_NE(registration, 0u);

  EXPECT_EQ(cp->notify_pm4_queue_doorbell(registration, 2), QueueSubmissionStatus::Accepted);
  service();
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint64_t>(kReadPointer), 2u);
  service();
  EXPECT_EQ(writes, 1u);
}

TEST_F(Pm4ComputeQueueTest, BlockedQueuesRetryIndependently) {
  constexpr uint64_t kOtherRing = 0x300;
  constexpr uint64_t kOtherReadPointer = 0x380;
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 1);
  memory->store<uint32_t>(kOtherRing, 0xc0017900);
  memory->store<uint32_t>(kOtherRing + 4, 0x40);
  memory->store<uint32_t>(kOtherRing + 8, 2);
  memory->store<uint32_t>(kOtherReadPointer, 0);

  bool first_blocked = true;
  bool second_blocked = true;
  uint32_t first_attempts = 0;
  uint32_t second_attempts = 0;
  const auto first = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++first_attempts;
    return first_blocked ? Pm4RegisterWriteStatus::Blocked : Pm4RegisterWriteStatus::Complete;
  }});
  const auto second = attach({.write_uconfig_register =
                                  [&](uint64_t, uint32_t) {
                                    ++second_attempts;
                                    return second_blocked ? Pm4RegisterWriteStatus::Blocked
                                                          : Pm4RegisterWriteStatus::Complete;
                                  }},
                             kOtherRing, kOtherReadPointer);
  ASSERT_NE(first, 0u);
  ASSERT_NE(second, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(first, 3), QueueSubmissionStatus::Accepted);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(second, 3), QueueSubmissionStatus::Accepted);

  service();
  EXPECT_GE(first_attempts, 1u);
  EXPECT_GE(second_attempts, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
  EXPECT_EQ(memory->load<uint32_t>(kOtherReadPointer), 0u);

  const uint32_t first_blocked_attempts = first_attempts;
  first_blocked = false;
  service();
  EXPECT_EQ(first_attempts, first_blocked_attempts + 1);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_EQ(memory->load<uint32_t>(kOtherReadPointer), 0u);
  second_blocked = false;
  service();
  EXPECT_EQ(first_attempts, first_blocked_attempts + 1);
  EXPECT_EQ(memory->load<uint32_t>(kOtherReadPointer), 3u);
}

TEST_F(Pm4ComputeQueueTest, DetachWaitsForActiveCommandProcessorService) {
  using namespace std::chrono_literals;
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  std::promise<void> callback_entered;
  std::promise<void> release_callback;
  std::shared_future<void> release = release_callback.get_future().share();
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    callback_entered.set_value();
    release.wait();
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 3), QueueSubmissionStatus::Accepted);

  auto service_task = std::async(std::launch::async, [&] { service(); });
  ASSERT_EQ(callback_entered.get_future().wait_for(1s), std::future_status::ready);
  auto detach = std::async(std::launch::async,
                           [&] { return cp->unregister_pm4_queue_registration(registration); });

  EXPECT_EQ(detach.wait_for(20ms), std::future_status::timeout);
  release_callback.set_value();
  service_task.get();
  EXPECT_TRUE(detach.get());
}

TEST_F(Pm4ComputeQueueTest, ReconfigureWaitsForActiveService) {
  using namespace std::chrono_literals;
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  std::promise<void> callback_entered;
  std::promise<void> release_callback;
  std::shared_future<void> release = release_callback.get_future().share();
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    callback_entered.set_value();
    release.wait();
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 3), QueueSubmissionStatus::Accepted);

  auto service_task = std::async(std::launch::async, [&] { service(); });
  ASSERT_EQ(callback_entered.get_future().wait_for(1s), std::future_status::ready);
  auto reconfigure = std::async(std::launch::async, [&] {
    return cp->update_pm4_queue_registration(
        registration,
        {.ring_base_address = kRing, .ring_size_bytes = kRingBytes, .scheduling_percentage = 100});
  });

  EXPECT_EQ(reconfigure.wait_for(20ms), std::future_status::timeout);
  release_callback.set_value();
  service_task.get();
  EXPECT_EQ(reconfigure.get(), QueueReconfigureStatus::Applied);
}

TEST_F(Pm4ComputeQueueTest, GracefulDetachWaitsForPendingCursorPublication) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t writes = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++writes;
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 3), QueueSubmissionStatus::Accepted);
  memory->make_store_unavailable(kReadPointer);

  service();
  EXPECT_EQ(cp->prepare_unregister_pm4_queue_registration(registration),
            QueuePrepareCloseStatus::Busy);
  EXPECT_EQ(cp->registered_pm4_queue_count_for_test(), 1u);
  EXPECT_EQ(writes, 1u);

  memory->make_available();
  service();
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_EQ(cp->prepare_unregister_pm4_queue_registration(registration),
            QueuePrepareCloseStatus::Ready);
  EXPECT_EQ(cp->registered_pm4_queue_count_for_test(), 0u);
}

TEST_F(Pm4ComputeQueueTest, GracefulDetachWaitsForBlockedCommandStream) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t attempts = 0;
  const auto registration = attach({.write_uconfig_register = [&](uint64_t, uint32_t) {
    ++attempts;
    return Pm4RegisterWriteStatus::Blocked;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 3), QueueSubmissionStatus::Accepted);

  service();
  EXPECT_EQ(attempts, 1u);
  EXPECT_EQ(cp->prepare_unregister_pm4_queue_registration(registration),
            QueuePrepareCloseStatus::Busy);
  EXPECT_EQ(cp->registered_pm4_queue_count_for_test(), 1u);
  EXPECT_TRUE(cp->unregister_pm4_queue_registration(registration));
}

TEST_F(Pm4ComputeQueueTest, GracefulDetachReportsFaultedPublicationWithoutRetryingForever) {
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  const auto registration = attach({.write_uconfig_register = [](uint64_t, uint32_t) {
    return Pm4RegisterWriteStatus::Complete;
  }});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 3), QueueSubmissionStatus::Accepted);
  memory->make_store_faulted(kReadPointer);

  service();
  EXPECT_EQ(cp->prepare_unregister_pm4_queue_registration(registration),
            QueuePrepareCloseStatus::Faulted);
  EXPECT_EQ(cp->registered_pm4_queue_count_for_test(), 1u);
  EXPECT_TRUE(cp->unregister_pm4_queue_registration(registration));
  EXPECT_EQ(cp->registered_pm4_queue_count_for_test(), 0u);
}

TEST_F(Pm4ComputeQueueTest, DrainedQueueCanBeDisabledAndReconfigured) {
  constexpr uint64_t kReplacementRing = 0x300;
  memory->store<uint32_t>(kRing, 0xffff1000);
  const auto registration = attach({});
  ASSERT_NE(registration, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(registration, 1), QueueSubmissionStatus::Accepted);
  service();

  EXPECT_EQ(
      cp->update_pm4_queue_registration(
          registration, {.ring_base_address = 0, .ring_size_bytes = 0, .scheduling_percentage = 0}),
      QueueReconfigureStatus::Disabled);
  memory->store<uint32_t>(kReplacementRing, 0xffff1000);
  EXPECT_EQ(cp->update_pm4_queue_registration(registration, {.ring_base_address = kReplacementRing,
                                                             .ring_size_bytes = kRingBytes,
                                                             .scheduling_percentage = 100}),
            QueueReconfigureStatus::Applied);
  EXPECT_EQ(cp->notify_pm4_queue_doorbell(registration, 1), QueueSubmissionStatus::Accepted);
  service();
  EXPECT_EQ(cp->prepare_unregister_pm4_queue_registration(registration),
            QueuePrepareCloseStatus::Ready);
}

TEST_F(Pm4ComputeQueueTest, InitialCursorRetryKeepsTheOriginalVmSnapshot) {
  memory->store<uint32_t>(kRing, 0xffff1000);
  memory->make_read_unavailable(kReadPointer);
  const auto id = attach({});
  ASSERT_NE(id, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(id, 1), QueueSubmissionStatus::Accepted);
  service();
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
  EXPECT_EQ(cp->prepare_unregister_pm4_queue_registration(id), QueuePrepareCloseStatus::Busy);
  auto replacement = std::make_shared<Pm4QueueMemory>();
  replacement->store<uint32_t>(kReadPointer, 99);
  ASSERT_TRUE(gpu_vm.replace_translated(address_space, replacement, replacement));
  memory->make_available();
  service();
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 1u);
  EXPECT_EQ(replacement->load<uint32_t>(kReadPointer), 99u);
}

TEST_F(Pm4ComputeQueueTest, InitialCursorRetryIsReportedWithoutAnEngine) {
  CommandProcessor standalone("standalone");
  standalone.set_gpu_vm(&gpu_vm);
  memory->make_read_unavailable(kReadPointer);
  const auto id = standalone.register_pm4_queue({.address_space = address_space,
                                                 .ring_base = kRing,
                                                 .ring_size_bytes = kRingBytes,
                                                 .consumer_pointer_address = kReadPointer});
  ASSERT_NE(id, 0u);
  EXPECT_EQ(standalone.notify_pm4_queue_doorbell(id, 1), QueueSubmissionStatus::Retry);
}

TEST_F(Pm4ComputeQueueTest, ChangedRingWaitsForANewDoorbell) {
  constexpr uint64_t replacement_ring = 0x300;
  memory->store<uint32_t>(kRing, 0xffff1000);
  memory->store<uint32_t>(replacement_ring, 0xffff1000);
  const auto id = attach({});
  ASSERT_NE(id, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(id, 1), QueueSubmissionStatus::Accepted);
  service();
  ASSERT_EQ(memory->load<uint32_t>(kReadPointer), 1u);
  ASSERT_EQ(cp->update_pm4_queue_registration(id, {}), QueueReconfigureStatus::Disabled);
  memory->store<uint32_t>(kReadPointer, 0);
  ASSERT_EQ(cp->update_pm4_queue_registration(id, {.ring_base_address = replacement_ring,
                                                   .ring_size_bytes = kRingBytes,
                                                   .scheduling_percentage = 100}),
            QueueReconfigureStatus::Applied);
  service();
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
  EXPECT_FALSE(cp->queue_faulted_for_test(1, 0));
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(id, 1), QueueSubmissionStatus::Accepted);
  service();
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 1u);
}

TEST_F(Pm4ComputeQueueTest, SameRingRuntimeResumePreservesSubmittedProgress) {
  memory->store<uint32_t>(kRing, 0xffff1000);
  const auto id = attach({});
  ASSERT_NE(id, 0u);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(id, 1), QueueSubmissionStatus::Accepted);
  ASSERT_EQ(cp->update_pm4_queue_registration(id, {}), QueueReconfigureStatus::Disabled);
  service();
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
  ASSERT_EQ(cp->update_pm4_queue_registration(id, {.ring_base_address = kRing,
                                                   .ring_size_bytes = kRingBytes,
                                                   .scheduling_percentage = 100}),
            QueueReconfigureStatus::Applied);
  service();
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 1u);
}

TEST_F(Pm4ComputeQueueTest, DebugResumeServicesTheExistingDoorbell) {
  memory->store<uint32_t>(kRing, 0xffff1000);
  const auto id = attach({});
  ASSERT_NE(id, 0u);
  cp->set_queue_debug_suspended(1, 0, true);
  ASSERT_EQ(cp->notify_pm4_queue_doorbell(id, 1), QueueSubmissionStatus::Accepted);
  service();
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
  cp->set_queue_debug_suspended(1, 0, false);
  service();
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 1u);
}

TEST_F(Pm4ComputeQueueTest, EngineNotificationRejectsDisabledInvalidAndFaultedQueues) {
  const auto id = attach({});
  ASSERT_NE(id, 0u);
  ASSERT_EQ(cp->update_pm4_queue_registration(id, {}), QueueReconfigureStatus::Disabled);
  EXPECT_EQ(cp->notify_pm4_queue_doorbell(id, 1), QueueSubmissionStatus::Faulted);
  ASSERT_EQ(cp->update_pm4_queue_registration(id, {.ring_base_address = kRing,
                                                   .ring_size_bytes = kRingBytes,
                                                   .scheduling_percentage = 100}),
            QueueReconfigureStatus::Applied);
  EXPECT_EQ(cp->notify_pm4_queue_doorbell(id, kRingBytes / 4 + 1), QueueSubmissionStatus::Faulted);
  EXPECT_EQ(cp->notify_pm4_queue_doorbell(id, 1), QueueSubmissionStatus::Faulted);
  EXPECT_EQ(cp->notify_pm4_queue_doorbell(id + 100, 1), QueueSubmissionStatus::Faulted);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
}

TEST_F(Pm4ComputeQueueTest, NativeRegisterWritesRequireSupportedCallbacks) {
  for (const uint32_t opcode : {0x79u, 0xbeu}) {
    SCOPED_TRACE(opcode);
    memory->store<uint32_t>(kRing, 0xc0010000 | opcode << 8);
    memory->store<uint32_t>(kRing + 4, 0x40);
    memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
    uint32_t writes = 0;
    Pm4PacketCallbacks callbacks;
    if (opcode == 0xbe)
      callbacks.write_uconfig_register = [&](uint64_t, uint32_t) {
        ++writes;
        return Pm4RegisterWriteStatus::Complete;
      };
    const auto id = attach(std::move(callbacks));
    ASSERT_NE(id, 0u);
    ASSERT_EQ(cp->notify_pm4_queue_doorbell(id, 3), QueueSubmissionStatus::Accepted);
    service();
    EXPECT_EQ(cp->notify_pm4_queue_doorbell(id, 3), QueueSubmissionStatus::Faulted);
    EXPECT_EQ(writes, 0u);
    EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
    ASSERT_TRUE(cp->unregister_pm4_queue_registration(id));
  }
}

TEST(Pm4ComputeQueueStateTest, PendingWorkIncludesStreamsAndShaderDispatches) {
  ComputeQueueRecord queue;
  EXPECT_FALSE(queue.has_pending_commands());
  queue.commands.submissions.emplace_back();
  EXPECT_TRUE(queue.has_pending_commands());
  queue.commands.submissions.clear();
  queue.dispatches.entries.emplace_back();
  EXPECT_TRUE(queue.has_pending_commands());
  queue.dispatches.entries.clear();
  queue.entries.emplace_back();
  EXPECT_TRUE(queue.has_pending_commands());
  queue.entries.clear();
  EXPECT_FALSE(queue.has_pending_commands());
}

TEST(Pm4QueueBindingTest, RejectsUnsupportedVmPollingThroughTheComputeQueueBinding) {
  constexpr uint64_t kRing = 0x100;
  constexpr uint64_t kReadPointer = 0x200;
  constexpr uint64_t kWritePointer = 0x208;
  alignas(8) std::array<std::byte, 8> doorbell{};
  auto memory = std::make_shared<Pm4QueueMemory>();
  GpuVm gpu_vm;
  const AddressSpaceHandle address_space = gpu_vm.register_translated(1, memory, memory);
  ASSERT_TRUE(address_space);
  CommandProcessor command_processor("cp");
  command_processor.set_gpu_vm(&gpu_vm);
  GpuQueueRegistry registry(gpu_vm);
  const std::shared_ptr<QueueBindingFactory> binding_factory =
      make_compute_queue_binding_factory(command_processor, {});

  for (const QueueDoorbellMode mode :
       {QueueDoorbellMode::HostPolled, QueueDoorbellMode::VmPolled}) {
    const QueueHandle queue = registry.register_queue({
        .identity = {.address_space = address_space, .process_id = 1, .queue_id = 7},
        .ring = {.base_address = kRing,
                 .size_bytes = 64,
                 .consumer_pointer_address = kReadPointer,
                 .producer_pointer_address = kWritePointer},
        .doorbell = {.mode = mode, .host_base = doorbell.data(), .address = 0x300},
        .binding_factory = binding_factory,
        .type = QueueType::Compute,
        .packet_format = QueuePacketFormat::Pm4,
    });
    if (mode == QueueDoorbellMode::VmPolled) {
      EXPECT_FALSE(queue);
      EXPECT_EQ(command_processor.registered_pm4_queue_count_for_test(), 0u);
      continue;
    }
    ASSERT_TRUE(queue);
    EXPECT_EQ(registry.active_queues(), 1u);
    EXPECT_EQ(command_processor.registered_pm4_queue_count_for_test(), 1u);
    EXPECT_TRUE(registry.unregister_queue(queue));
    EXPECT_EQ(command_processor.registered_queue_count_for_test(), 0u);
  }

  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

TEST(Pm4QueueBindingTest, GracefulRegistryRemovalKeepsHandleUntilPublicationCompletes) {
  constexpr uint64_t kRing = 0x100;
  constexpr uint64_t kReadPointer = 0x200;
  auto memory = std::make_shared<Pm4QueueMemory>();
  GpuVm gpu_vm;
  const AddressSpaceHandle address_space = gpu_vm.register_translated(1, memory, memory);
  ASSERT_TRUE(address_space);
  CommandProcessor command_processor("cp");
  command_processor.set_gpu_vm(&gpu_vm);
  GpuQueueRegistry registry(gpu_vm);
  uint32_t writes = 0;
  const std::shared_ptr<QueueBindingFactory> binding_factory = make_compute_queue_binding_factory(
      command_processor, {.write_uconfig_register = [&](uint64_t, uint32_t) {
        ++writes;
        return Pm4RegisterWriteStatus::Complete;
      }});
  const QueueHandle queue = registry.register_queue({
      .identity = {.address_space = address_space, .process_id = 1, .queue_id = 7},
      .ring = {.base_address = kRing, .size_bytes = 64, .consumer_pointer_address = kReadPointer},
      .doorbell = {},
      .binding_factory = binding_factory,
      .type = QueueType::Compute,
      .packet_format = QueuePacketFormat::Pm4,
  });
  ASSERT_TRUE(queue);
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  memory->make_store_unavailable(kReadPointer);

  const QueueSubmissionResult blocked = registry.submit_producer(queue, 3);
  ASSERT_TRUE(blocked.found);
  EXPECT_EQ(blocked.status, QueueSubmissionStatus::Retry);
  EXPECT_EQ(writes, 1u);
  EXPECT_FALSE(registry.unregister_queue(queue));
  EXPECT_TRUE(registry.contains(queue));

  memory->make_available();
  const QueueSubmissionResult completed = registry.submit_producer(queue, 3);
  ASSERT_TRUE(completed.found);
  EXPECT_EQ(completed.status, QueueSubmissionStatus::Accepted);
  EXPECT_EQ(writes, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_TRUE(registry.unregister_queue(queue));
  EXPECT_FALSE(registry.contains(queue));
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

} // namespace
TEST_F(Pm4ComputeQueueTest, ComputeQueueRetainsCursorPublicationWithoutReplayingEffects) {
  CommandProcessor cp("compute");
  cp.set_gpu_vm(&gpu_vm, address_space);
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t effects = 0;
  const auto id = cp.register_pm4_queue(
      {.address_space = address_space,
       .ring_base = kRing,
       .ring_size_bytes = kRingBytes,
       .consumer_pointer_address = kReadPointer,
       .initial_consumer_cursor = std::nullopt,
       .packet_callbacks = {.write_uconfig_register = [&](uint64_t, uint32_t) {
         ++effects;
         return Pm4RegisterWriteStatus::Complete;
       }}});
  ASSERT_NE(id, 0u);
  memory->make_store_unavailable(kReadPointer);
  EXPECT_EQ(cp.notify_pm4_queue_doorbell(id, 3), QueueSubmissionStatus::Retry);
  EXPECT_EQ(effects, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 0u);
  EXPECT_EQ(cp.prepare_unregister_queue_registration(id), QueuePrepareCloseStatus::Busy);
  memory->make_available();
  EXPECT_EQ(cp.notify_pm4_queue_doorbell(id, 3), QueueSubmissionStatus::Accepted);
  EXPECT_EQ(effects, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_EQ(cp.prepare_unregister_queue_registration(id), QueuePrepareCloseStatus::Ready);
}

TEST_F(Pm4ComputeQueueTest, ComputeQueueUsesDwordCursorsAcrossRingWrap) {
  CommandProcessor cp("compute");
  cp.set_gpu_vm(&gpu_vm, address_space);
  memory->store<uint32_t>(kReadPointer, 15);
  memory->store<uint32_t>(kRing + 60, 0xc0017900);
  memory->store<uint32_t>(kRing, 0x40);
  memory->store<uint32_t>(kRing + 4, 0x1234);
  uint32_t effects = 0;
  const auto id = cp.register_pm4_queue(
      {.address_space = address_space,
       .ring_base = kRing,
       .ring_size_bytes = kRingBytes,
       .consumer_pointer_address = kReadPointer,
       .initial_consumer_cursor = std::nullopt,
       .packet_callbacks = {.write_uconfig_register = [&](uint64_t reg, uint32_t value) {
         EXPECT_EQ(reg, 0xc040u);
         EXPECT_EQ(value, 0x1234u);
         ++effects;
         return Pm4RegisterWriteStatus::Complete;
       }}});
  ASSERT_NE(id, 0u);
  EXPECT_EQ(cp.notify_pm4_queue_doorbell(id, 2), QueueSubmissionStatus::Accepted);
  EXPECT_EQ(effects, 1u);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 2u);
  EXPECT_TRUE(cp.unregister_queue_registration(id));
}

TEST_F(Pm4ComputeQueueTest, ComputeQueueRetainsItsSnapshotWhilePacketFetchIsBlocked) {
  CommandProcessor cp("compute");
  cp.set_gpu_vm(&gpu_vm, address_space);
  memory->store<uint32_t>(kRing, 0xc0017900);
  memory->store<uint32_t>(kRing + 4, 0x40);
  memory->store<uint32_t>(kRing + 8, 0xdeadbeef);
  uint32_t written = 0;
  const auto id = cp.register_pm4_queue(
      {.address_space = address_space,
       .ring_base = kRing,
       .ring_size_bytes = kRingBytes,
       .consumer_pointer_address = kReadPointer,
       .initial_consumer_cursor = std::nullopt,
       .packet_callbacks = {.write_uconfig_register = [&](uint64_t, uint32_t value) {
         written = value;
         return Pm4RegisterWriteStatus::Complete;
       }}});
  ASSERT_NE(id, 0u);
  memory->make_read_unavailable(kRing);
  EXPECT_EQ(cp.notify_pm4_queue_doorbell(id, 3), QueueSubmissionStatus::Retry);
  auto replacement = std::make_shared<Pm4QueueMemory>();
  replacement->store<uint32_t>(kRing, 0xc0017900);
  replacement->store<uint32_t>(kRing + 4, 0x40);
  replacement->store<uint32_t>(kRing + 8, 0x1234);
  replacement->store<uint32_t>(kReadPointer, 99);
  ASSERT_TRUE(gpu_vm.replace_translated(address_space, replacement, replacement));
  memory->make_available();
  EXPECT_EQ(cp.notify_pm4_queue_doorbell(id, 3), QueueSubmissionStatus::Accepted);
  EXPECT_EQ(written, 0xdeadbeefu);
  EXPECT_EQ(memory->load<uint32_t>(kReadPointer), 3u);
  EXPECT_EQ(replacement->load<uint32_t>(kReadPointer), 99u);
  EXPECT_TRUE(cp.unregister_queue_registration(id));
}

TEST_F(Pm4ComputeQueueTest, ComputeQueueCancellationDropsItsNestedStreamAndVmLease) {
  CommandProcessor cp("compute");
  cp.set_gpu_vm(&gpu_vm, address_space);
  constexpr uint64_t child = 0x400, gate = 0x700;
  const std::array<uint32_t, 4> root{0xc0023f00, uint32_t(child), 0, (1u << 23) | 7};
  const std::array<uint32_t, 7> wait{0xc0053c00, 0x13, uint32_t(gate), 0, 1, 0xffffffff, 4};
  for (uint32_t i = 0; i < root.size(); ++i)
    memory->store<uint32_t>(kRing + i * 4, root[i]);
  for (uint32_t i = 0; i < wait.size(); ++i)
    memory->store<uint32_t>(child + i * 4, wait[i]);
  const auto id = cp.register_pm4_queue({.address_space = address_space,
                                         .ring_base = kRing,
                                         .ring_size_bytes = kRingBytes,
                                         .consumer_pointer_address = kReadPointer,
                                         .initial_consumer_cursor = std::nullopt,
                                         .packet_callbacks = {}});
  ASSERT_NE(id, 0u);
  EXPECT_EQ(cp.notify_pm4_queue_doorbell(id, root.size()), QueueSubmissionStatus::Retry);
  EXPECT_FALSE(gpu_vm.unregister_address_space(address_space));
  EXPECT_EQ(cp.prepare_unregister_queue_registration(id), QueuePrepareCloseStatus::Busy);
  EXPECT_TRUE(cp.unregister_queue_registration(id));
  EXPECT_EQ(cp.registered_queue_count_for_test(), 0u);
  EXPECT_TRUE(gpu_vm.unregister_address_space(address_space));
}

} // namespace rocjitsu::amdgpu
