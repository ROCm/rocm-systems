// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_test.cpp
/// @brief Execution plugin lifecycle, concurrency, and reporting tests.

#include "execution_plugin_test_util.h"
#include "halt_snapshot_plugin.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/kmd/linux/legacy_gpu_vm.h"
#include "rocjitsu/vm/amdgpu/dispatch_entry.h"
#include "rocjitsu/vm/amdgpu/hwreg.h"
#include "rocjitsu/vm/plugins/instruction_mix/plugin.h"
#include "rocjitsu/vm/plugins/logging/plugin.h"
#include "rocjitsu/vm/plugins/plugin_config_resolver.h"
#include "rocjitsu/vm/plugins/race_detector/plugin.h"
#include "rocjitsu/vm/plugins/throughput/plugin.h"
#include "scoped_temp.h"

#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"

#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace rocjitsu::test {

class ExecutionPluginGroupTestAccess {
public:
  static uint64_t callback_lock_acquisitions(const ExecutionPluginGroup &group) {
    return group.callback_lock_acquisitions_;
  }
};

} // namespace rocjitsu::test

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;
using namespace rocjitsu::plugins::race_detector;
using namespace rocjitsu::test::execution_plugin;

/// Exercises sink lifetime through plugin destruction.
class DestructionTrackingSink : public PluginSink {
public:
  explicit DestructionTrackingSink(std::vector<std::string> &events) : events_(events) {}
  ~DestructionTrackingSink() override { events_.push_back("sink"); }
  void write(std::string_view msg) override { events_.push_back("write:" + std::string(msg)); }

private:
  std::vector<std::string> &events_;
};

class DestructorWritingPlugin : public ExecutionPlugin {
public:
  explicit DestructorWritingPlugin(std::vector<std::string> &events)
      : ExecutionPlugin("destructor_writer"), events_(events) {}
  ~DestructorWritingPlugin() override {
    sink().write("destroyed\n");
    events_.push_back("plugin");
  }

private:
  std::vector<std::string> &events_;
};

class ParallelSafePlugin final : public ExecutionPlugin {
public:
  ParallelSafePlugin() : ExecutionPlugin("parallel_safe") {}
  bool requires_serial_hot_hooks() const override { return false; }
};

class ParallelColdHookRecorder final : public ExecutionPlugin {
public:
  ParallelColdHookRecorder() : ExecutionPlugin("parallel_cold_hook_recorder"), startup_(2) {}

  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &,
                                        amdgpu::Wavefront &) override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      hot_hook_threads_.insert(std::this_thread::get_id());
    }
    if (startup_arrivals_.fetch_add(1, std::memory_order_relaxed) < 2)
      startup_.arrive_and_wait();
  }

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id, uint32_t, uint32_t,
                                   std::span<amdgpu::Wavefront *>) override {
    record(HookEvent::WORKGROUP_DISPATCHED, dispatch_id, wg_id);
  }

  void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) override {
    record(HookEvent::WORKGROUP_COMPLETED, dispatch_id, wg_id);
  }

  std::vector<HookEvent> events() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_;
  }

  std::set<std::thread::id> hot_hook_threads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return hot_hook_threads_;
  }

private:
  void record(HookEvent::Kind kind, uint32_t dispatch_id, uint32_t wg_id) {
    HookEvent event{kind};
    event.dispatch_id = dispatch_id;
    event.wg_id = wg_id;
    event.callback_thread = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(mutex_);
    events_.push_back(std::move(event));
  }

  std::barrier<> startup_;
  std::atomic<uint32_t> startup_arrivals_{0};
  mutable std::mutex mutex_;
  std::vector<HookEvent> events_;
  std::set<std::thread::id> hot_hook_threads_;
};

class SerialHotHookPlugin final : public ExecutionPlugin {
public:
  SerialHotHookPlugin() : ExecutionPlugin("serial_hot_hook") {}
  bool requires_serial_hot_hooks() const override { return true; }
};

class UnsubscribedHotHookPlugin final : public ExecutionPlugin {
public:
  UnsubscribedHotHookPlugin() : ExecutionPlugin("unsubscribed_hot_hook") {}
  bool requires_serial_hot_hooks() const override { return true; }
  bool observes_hot_hooks_for_wavefront(const amdgpu::Wavefront *) const override { return false; }
  void onAmdgpuReadSgpr(const amdgpu::Wavefront *, uint32_t) override { ++hot_callbacks; }
  void onAmdgpuWorkgroupCompleted(uint32_t, uint32_t) override { ++cold_callbacks; }

  uint32_t hot_callbacks = 0;
  uint32_t cold_callbacks = 0;
};

class HotHookSubscriptionProbe final : public ExecutionPlugin {
public:
  HotHookSubscriptionProbe(std::string name, bool subscribe_on_dispatch,
                           bool serial_hot_hooks = false)
      : ExecutionPlugin(std::move(name)), subscribe_on_dispatch_(subscribe_on_dispatch),
        serial_hot_hooks_(serial_hot_hooks) {}

  bool requires_serial_hot_hooks() const override { return serial_hot_hooks_; }

  bool observes_hot_hooks_for_wavefront(const amdgpu::Wavefront *) const override {
    ++predicate_queries;
    return subscribed;
  }

  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override {
    ++dispatch_callbacks;
    subscribed = subscribe_on_dispatch_;
    if (reenter_on_dispatch && group)
      group->onAmdgpuReadSgpr(&wf, 0);
  }

  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override {
    ++halt_callbacks;
    if (unsubscribe_on_halt)
      subscribed = false;
    if (reenter_on_halt && group)
      group->onAmdgpuReadSgpr(&wf, 0);
  }

  void onAmdgpuReadSgpr(const amdgpu::Wavefront *, uint32_t) override { ++hot_callbacks; }

  void set_subscribe_on_dispatch(bool value) { subscribe_on_dispatch_ = value; }

  ExecutionPluginGroup *group = nullptr;
  bool reenter_on_dispatch = false;
  bool reenter_on_halt = false;
  bool unsubscribe_on_halt = false;
  bool subscribed = false;
  mutable uint32_t predicate_queries = 0;
  uint32_t dispatch_callbacks = 0;
  uint32_t halt_callbacks = 0;
  uint32_t hot_callbacks = 0;

private:
  bool subscribe_on_dispatch_ = false;
  bool serial_hot_hooks_ = false;
};

class NoSgprReadPlugin final : public ExecutionPlugin {
public:
  NoSgprReadPlugin() : ExecutionPlugin("no_sgpr_read") {}
  bool observes_sgpr_reads() const override { return false; }
  void onAmdgpuReadSgpr(const amdgpu::Wavefront *, uint32_t) override { ++callbacks; }
  void onAmdgpuReadScalarRegister(const amdgpu::Wavefront *, RegisterRef) override { ++callbacks; }

  uint32_t callbacks = 0;
};

class AsyncIssueInterestPlugin final : public ExecutionPlugin {
public:
  AsyncIssueInterestPlugin(std::string name, bool observes_issue, bool throws_on_issue = false)
      : ExecutionPlugin(std::move(name)), observes_issue_(observes_issue),
        throws_on_issue_(throws_on_issue) {}

  bool supports_async_instructions() const override { return true; }
  bool observes_async_instruction_issued() const override { return observes_issue_; }

  void onAmdgpuAsyncInstructionIssued(uint64_t, const Instruction &, Wavefront &) override {
    ++callbacks;
    if (throws_on_issue_)
      throw std::runtime_error("injected async issue failure");
  }

  uint32_t callbacks = 0;

private:
  bool observes_issue_ = false;
  bool throws_on_issue_ = false;
};

class OverlapProbe {
public:
  void observe() {
    const int current = active_.fetch_add(1, std::memory_order_relaxed) + 1;
    int observed = max_active_.load(std::memory_order_relaxed);
    while (current > observed &&
           !max_active_.compare_exchange_weak(observed, current, std::memory_order_relaxed)) {
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (current == 1) {
      first_entered_ = true;
      cv_.notify_all();
      cv_.wait(lock, [&]() { return release_first_; });
    } else {
      overlap_observed_ = true;
      cv_.notify_all();
    }
    active_.fetch_sub(1, std::memory_order_relaxed);
  }

  bool wait_for_first(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [&]() { return first_entered_; });
  }

  bool wait_for_overlap(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, timeout, [&]() { return overlap_observed_; });
  }

  void release_first() {
    std::lock_guard<std::mutex> lock(mutex_);
    release_first_ = true;
    cv_.notify_all();
  }

  int max_active() const { return max_active_.load(std::memory_order_relaxed); }

private:
  std::atomic<int> active_{0};
  std::atomic<int> max_active_{0};
  std::mutex mutex_;
  std::condition_variable cv_;
  bool first_entered_ = false;
  bool overlap_observed_ = false;
  bool release_first_ = false;
};

class ConcurrencyProbePlugin final : public ExecutionPlugin {
public:
  explicit ConcurrencyProbePlugin(bool serialize_hot_hooks)
      : ExecutionPlugin("concurrency_probe"), serialize_hot_hooks_(serialize_hot_hooks) {}

  bool requires_serial_hot_hooks() const override { return serialize_hot_hooks_; }

  void onAmdgpuReadSgpr(const amdgpu::Wavefront *, uint32_t) override { hot_probe_.observe(); }

  void onAmdgpuWorkgroupCompleted(uint32_t, uint32_t) override { cold_probe_.observe(); }

  OverlapProbe &hot_probe() { return hot_probe_; }
  OverlapProbe &cold_probe() { return cold_probe_; }

private:
  bool serialize_hot_hooks_;
  OverlapProbe hot_probe_;
  OverlapProbe cold_probe_;
};

class CrossHookConcurrencyProbePlugin final : public ExecutionPlugin {
public:
  explicit CrossHookConcurrencyProbePlugin(bool serialize_hot_hooks)
      : ExecutionPlugin("cross_hook_concurrency_probe"), serialize_hot_hooks_(serialize_hot_hooks) {
  }

  bool requires_serial_hot_hooks() const override { return serialize_hot_hooks_; }

  void onAmdgpuReadSgpr(const amdgpu::Wavefront *, uint32_t) override { probe_.observe(); }

  void onAmdgpuWorkgroupCompleted(uint32_t, uint32_t) override { probe_.observe(); }

  OverlapProbe &probe() { return probe_; }

private:
  bool serialize_hot_hooks_;
  OverlapProbe probe_;
};

struct OverlapResult {
  bool first_entered;
  bool overlap_observed;
};

template <typename FirstCallback, typename SecondCallback>
OverlapResult run_staged_callbacks(OverlapProbe &probe, std::chrono::milliseconds overlap_timeout,
                                   FirstCallback first_callback, SecondCallback second_callback) {
  std::thread first(first_callback);
  const bool first_entered = probe.wait_for_first(std::chrono::seconds(5));
  if (!first_entered) {
    probe.release_first();
    first.join();
    return {false, false};
  }

  std::thread second(second_callback);
  const bool overlap_observed = probe.wait_for_overlap(overlap_timeout);
  probe.release_first();
  first.join();
  second.join();
  return {true, overlap_observed};
}

template <typename Callback>
OverlapResult run_staged_threads(OverlapProbe &probe, std::chrono::milliseconds overlap_timeout,
                                 Callback callback) {
  return run_staged_callbacks(probe, overlap_timeout, callback, callback);
}

template <typename Callback> void run_two_threads(Callback callback) {
  std::barrier start(3);
  std::thread first([&]() {
    start.arrive_and_wait();
    callback();
  });
  std::thread second([&]() {
    start.arrive_and_wait();
    callback();
  });
  start.arrive_and_wait();
  first.join();
  second.join();
}

inline const char *kindName(HookEvent::Kind k) {
  static const char *names[] = {
      "DISPATCH_PACKET_PROCESSED",
      "DISPATCH_EXECUTION_BEGIN",
      "DISPATCH_EXECUTION_END",
      "WORKGROUP_DISPATCHED",
      "WORKGROUP_COMPLETED",
      "WAVEFRONT_DISPATCHED",
      "WAVEFRONT_HALTED",
      "BEFORE_INSTRUCTION",
      "AFTER_INSTRUCTION",
      "ROUTE_MEMORY",
      "READ_VGPR",
      "WRITE_VGPR",
      "READ_SGPR",
      "WRITE_SGPR",
      "BARRIER_RESOLVED",
      "INIT",
      "SHUTDOWN",
  };
  return k < HookEvent::KIND_COUNT ? names[k] : "UNKNOWN";
}

/// Helper for asserting ordering invariants on a HookEvent log.
class EventLog {
public:
  using Kind = HookEvent::Kind;

  explicit EventLog(const std::vector<HookEvent> &events) : events_(events) {}

  /// Print the full lifecycle event timeline to stderr.
  void dump() const {
    std::cerr << "\n=== Event timeline (" << events_.size() << " events) ===\n";
    for (size_t i = 0; i < events_.size(); ++i) {
      const auto &e = events_[i];
      if (e.kind > Kind::WAVEFRONT_HALTED && e.kind != Kind::INIT && e.kind != Kind::SHUTDOWN)
        continue;
      std::cerr << std::setw(4) << i << "  " << std::setw(30) << std::left << kindName(e.kind)
                << std::right;
      switch (e.kind) {
      case Kind::DISPATCH_PACKET_PROCESSED:
      case Kind::DISPATCH_EXECUTION_BEGIN:
      case Kind::DISPATCH_EXECUTION_END:
        std::cerr << " d=" << e.dispatch_id;
        break;
      case Kind::WORKGROUP_DISPATCHED:
      case Kind::WORKGROUP_COMPLETED:
        std::cerr << " d=" << e.dispatch_id << " wg=" << e.wg_id;
        break;
      case Kind::WAVEFRONT_DISPATCHED:
      case Kind::WAVEFRONT_HALTED:
        std::cerr << " d=" << e.dispatch_id << " wg=" << e.wg_id << " wf=" << e.wf_id;
        break;
      default:
        break;
      }
      std::cerr << "\n";
    }
    std::cerr << "=== end ===\n\n";
  }

  /// Count events of a given kind, optionally filtered by dispatch_id.
  size_t count(Kind kind, uint32_t dispatch_id = UINT32_MAX) const {
    size_t n = 0;
    for (const auto &e : events_)
      if (e.kind == kind && (dispatch_id == UINT32_MAX || e.dispatch_id == dispatch_id))
        ++n;
    return n;
  }

  /// Return dispatch_ids in the order they first appear as DISPATCH_PACKET_PROCESSED.
  std::vector<uint32_t> dispatchIds() const {
    std::vector<uint32_t> ids;
    for (const auto &e : events_) {
      if (e.kind == Kind::DISPATCH_PACKET_PROCESSED &&
          std::ranges::find(ids, e.dispatch_id) == ids.end())
        ids.push_back(e.dispatch_id);
    }
    return ids;
  }

  /// Assert that the last event of kind 'a' precedes the first event of kind 'b'.
  void assertAllBefore(Kind a, Kind b) const {
    size_t last_a = 0;
    size_t first_b = events_.size();
    bool found_a = false;
    for (size_t i = 0; i < events_.size(); ++i) {
      if (events_[i].kind == a) {
        last_a = i;
        found_a = true;
      }
      if (events_[i].kind == b && i < first_b)
        first_b = i;
    }
    std::cerr << "  edge: last " << kindName(a) << " [" << last_a << "] -> first " << kindName(b)
              << " [" << first_b << "]\n";
    ASSERT_TRUE(found_a) << "No events of first kind found";
    EXPECT_LT(last_a, first_b)
        << "All events of first kind should precede all events of second kind";
  }

  /// Assert that the last (a, da) event precedes the first (b, db) event.
  void assertLastBeforeFirst(Kind a, uint32_t da, Kind b, uint32_t db) const {
    size_t last_a = 0;
    size_t first_b = events_.size();
    bool found_a = false;
    for (size_t i = 0; i < events_.size(); ++i) {
      if (events_[i].kind == a && events_[i].dispatch_id == da) {
        last_a = i;
        found_a = true;
      }
      if (events_[i].kind == b && events_[i].dispatch_id == db && i < first_b)
        first_b = i;
    }
    std::cerr << "  edge: last " << kindName(a) << "(d=" << da << ") [" << last_a << "] -> first "
              << kindName(b) << "(d=" << db << ") [" << first_b << "]\n";
    ASSERT_TRUE(found_a) << "No matching events for first kind";
    EXPECT_LT(last_a, first_b);
  }

  /// Return all unique dispatch_ids seen across all lifecycle events.
  std::set<uint32_t> allDispatchIds() const {
    std::set<uint32_t> ids;
    for (const auto &e : events_) {
      switch (e.kind) {
      case Kind::DISPATCH_PACKET_PROCESSED:
      case Kind::DISPATCH_EXECUTION_BEGIN:
      case Kind::DISPATCH_EXECUTION_END:
      case Kind::WORKGROUP_DISPATCHED:
      case Kind::WORKGROUP_COMPLETED:
      case Kind::WAVEFRONT_DISPATCHED:
      case Kind::WAVEFRONT_HALTED:
        ids.insert(e.dispatch_id);
        break;
      default:
        break;
      }
    }
    return ids;
  }

  /// Assert that begin/end events are matched by wf_id within a dispatch:
  /// each begin has a corresponding end, begin precedes end, none left open.
  void assertPaired(Kind begin_kind, Kind end_kind, uint32_t dispatch_id) const {
    assertPairedByKey(
        begin_kind, end_kind, dispatch_id, [](const HookEvent &e) { return e.wf_id; }, "wf");
  }

  /// Assert that begin/end events are matched by wg_id within a dispatch.
  void assertPairedByWg(Kind begin_kind, Kind end_kind, uint32_t dispatch_id) const {
    assertPairedByKey(
        begin_kind, end_kind, dispatch_id, [](const HookEvent &e) { return e.wg_id; }, "wg");
  }

private:
  template <typename KeyFn>
  void assertPairedByKey(Kind begin_kind, Kind end_kind, uint32_t dispatch_id, KeyFn key_fn,
                         const char *key_name) const {
    std::map<uint32_t, size_t> opens;
    for (size_t i = 0; i < events_.size(); ++i) {
      if (events_[i].dispatch_id != dispatch_id)
        continue;
      uint32_t key = key_fn(events_[i]);
      if (events_[i].kind == begin_kind) {
        opens[key] = i;
      } else if (events_[i].kind == end_kind) {
        auto it = opens.find(key);
        ASSERT_NE(it, opens.end()) << "End without matching begin for " << key_name << "=" << key;
        EXPECT_LT(it->second, i);
        opens.erase(it);
      }
    }
    EXPECT_TRUE(opens.empty()) << "Unmatched begin events remain";
  }

private:
  const std::vector<HookEvent> &events_;
};

TEST(ExecutionPluginTest, NoPluginNoCrash) {
  PluginFixture f;
  const uint32_t code[] = {S_NOP, S_ENDPGM};
  f.run_kernel(code, 2);
}

TEST(ExecutionPluginTest, DispatchPacketCarriesExecutionShapeAndTargetMetadata) {
  PluginFixture fixture(/*num_wf_slots=*/16, /*arch=*/"cdna5", /*wavefront_size=*/32,
                        /*sgprs_per_wf=*/128);
  auto *plugin = fixture.attach_ordering_plugin();

  constexpr uint32_t kCdna5Endpgm = 0xBFB00000u;
  constexpr uint32_t kStaticLdsBytes = 1537;
  constexpr uint32_t kDynamicLdsBytes = 2049;
  const uint64_t kernel_object =
      fixture.write_kernel(0x1000, &kCdna5Endpgm, 1, /*granulated_sgpr_count=*/15, kStaticLdsBytes);

  amdgpu::AmdExtKernelDispatchPacket packet{};
  packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
  packet.amd_format = amdgpu::kHsaAmdPacketTypeExtKernelDispatch;
  packet.setup = 3;
  packet.workgroup_size_x = 32;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.cluster_count_x = 1;
  packet.cluster_count_y = 1;
  packet.cluster_count_z = 1;
  packet.cluster_size_x = 2;
  packet.cluster_size_y = 2;
  packet.cluster_size_z = 2;
  packet.group_segment_size = kDynamicLdsBytes;
  packet.kernel_object = kernel_object;
  test::AqlQueue queue(fixture.mem, fixture.cp());
  queue.submit(packet);
  ASSERT_NO_THROW(fixture.run_until_idle());

  auto dispatch = std::ranges::find_if(plugin->events, [](const auto &event) {
    return event.kind == HookEvent::DISPATCH_PACKET_PROCESSED;
  });
  ASSERT_NE(dispatch, plugin->events.end());
  EXPECT_EQ(dispatch->lds_size_bytes, kDynamicLdsBytes);
  EXPECT_EQ(dispatch->wave_size, 32u);
  EXPECT_EQ(dispatch->code_target, ROCJITSU_CODE_TARGET_GFX1250);
  EXPECT_EQ(dispatch->cluster_size_x, 2u);
  EXPECT_EQ(dispatch->cluster_size_y, 2u);
  EXPECT_EQ(dispatch->cluster_size_z, 2u);
}

class ThroughputTestInstruction final : public Instruction {
public:
  explicit ThroughputTestInstruction(std::string_view mnemonic, uint64_t flags = 0,
                                     std::unique_ptr<DynamicInstState> state = nullptr)
      : Instruction(mnemonic, nullptr) {
    flags_ = flags;
    set_data(std::move(state));
  }
};

/// An instruction carrying only decoded memory-issue metadata: no MEMORY_OP,
/// no pipeline state, and a mnemonic that matches no memory prefix. The CDNA5
/// `cluster_load_*` set decodes exactly this way.
class IssueMetadataTestInstruction final : public Instruction {
public:
  IssueMetadataTestInstruction(std::string_view mnemonic,
                               amdgpu::MemoryCompletionClass completion_class)
      : Instruction(mnemonic, nullptr) {
    set_memory_issue_info(
        {amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::VMCNT, completion_class}});
  }
};

struct ParsedThroughputRecord {
  std::string record;
  uint64_t dispatch_id = 0;
  std::string kernel_name;
  std::string kernel_symbol;
  uint64_t workgroups = 0;
  uint64_t waves_per_workgroup = 0;
  double wall_seconds = 0.0;
  uint64_t wave_instructions = 0;
  double mips = 0.0;
  uint64_t dispatches = 0;
  double dispatch_seconds_sum = 0.0;
  plugins::throughput::InstructionCounts family_instructions{};
  plugins::throughput::InstructionCounts untimed_instructions{};
  std::array<bool, plugins::throughput::kInstructionFamilyCount> execution_timing_valid{};
  std::array<double, plugins::throughput::kInstructionFamilyCount> execution_seconds{};
  std::array<double, plugins::throughput::kInstructionFamilyCount> execution_mips{};
  std::array<double, plugins::throughput::kInstructionFamilyCount> dispatch_mips{};
};

std::vector<ParsedThroughputRecord> parse_throughput_jsonl(std::string_view jsonl) {
  std::vector<ParsedThroughputRecord> records;
  std::istringstream lines{std::string(jsonl)};
  std::string line;
  while (std::getline(lines, line)) {
    if (line.empty())
      continue;

    flexbuffers::Builder builder;
    if (!plugin_detail::flexbuffer_from_json(line, builder)) {
      ADD_FAILURE() << "invalid JSONL record: " << line;
      continue;
    }
    auto root = flexbuffers::GetRoot(builder.GetBuffer());
    if (!root.IsMap()) {
      ADD_FAILURE() << "throughput record is not a JSON object";
      continue;
    }
    auto object = root.AsMap();
    const auto schema = object["schema"];
    const auto record_type = object["record"];
    EXPECT_TRUE(schema.IsString()) << "schema must be a string";
    EXPECT_TRUE(record_type.IsString()) << "record must be a string";
    if (!schema.IsString() || !record_type.IsString())
      continue;
    EXPECT_EQ(schema.AsString().str(), "rocjitsu.throughput.v2");

    ParsedThroughputRecord record;
    record.record = record_type.AsString().str();

    const auto wall_seconds = object["wall_seconds"];
    const auto wave_instructions = object["wave_instructions"];
    const auto throughput_mips = object["mips"];
    const auto families_value = object["families"];
    EXPECT_TRUE(wall_seconds.IsNumeric()) << "wall_seconds must be numeric";
    EXPECT_TRUE(wave_instructions.IsIntOrUint()) << "wave_instructions must be an integer";
    EXPECT_TRUE(throughput_mips.IsNumeric()) << "mips must be numeric";
    EXPECT_TRUE(families_value.IsMap()) << "families must be an object";
    if (!wall_seconds.IsNumeric() || !wave_instructions.IsIntOrUint() ||
        !throughput_mips.IsNumeric() || !families_value.IsMap())
      continue;
    record.wall_seconds = wall_seconds.AsDouble();
    record.wave_instructions = wave_instructions.AsUInt64();
    record.mips = throughput_mips.AsDouble();

    if (record.record == "dispatch") {
      const auto dispatch_id = object["dispatch_id"];
      const auto kernel_name = object["kernel_name"];
      const auto kernel_symbol = object["kernel_symbol"];
      const auto grid = object["grid"];
      const auto workgroup = object["workgroup"];
      const auto workgroups = object["workgroups"];
      const auto waves_per_workgroup = object["waves_per_workgroup"];
      EXPECT_TRUE(dispatch_id.IsIntOrUint()) << "dispatch_id must be an integer";
      EXPECT_TRUE(kernel_name.IsString()) << "kernel_name must be a string";
      EXPECT_TRUE(kernel_symbol.IsString()) << "kernel_symbol must be a string";
      EXPECT_TRUE(grid.IsAnyVector()) << "grid must be an array";
      EXPECT_TRUE(workgroup.IsAnyVector()) << "workgroup must be an array";
      EXPECT_TRUE(workgroups.IsIntOrUint()) << "workgroups must be an integer";
      EXPECT_TRUE(waves_per_workgroup.IsIntOrUint()) << "waves_per_workgroup must be an integer";
      if (grid.IsAnyVector()) {
        const auto values = grid.AsVector();
        EXPECT_EQ(values.size(), 3u);
        for (size_t i = 0; i < values.size(); ++i)
          EXPECT_TRUE(values[i].IsIntOrUint()) << "grid[" << i << "] must be an integer";
      }
      if (workgroup.IsAnyVector()) {
        const auto values = workgroup.AsVector();
        EXPECT_EQ(values.size(), 3u);
        for (size_t i = 0; i < values.size(); ++i)
          EXPECT_TRUE(values[i].IsIntOrUint()) << "workgroup[" << i << "] must be an integer";
      }
      if (dispatch_id.IsIntOrUint())
        record.dispatch_id = dispatch_id.AsUInt64();
      if (kernel_name.IsString())
        record.kernel_name = kernel_name.AsString().str();
      if (kernel_symbol.IsString())
        record.kernel_symbol = kernel_symbol.AsString().str();
      if (workgroups.IsIntOrUint())
        record.workgroups = workgroups.AsUInt64();
      if (waves_per_workgroup.IsIntOrUint())
        record.waves_per_workgroup = waves_per_workgroup.AsUInt64();
    } else if (record.record == "summary") {
      const auto dispatches = object["dispatches"];
      const auto dispatch_seconds_sum = object["dispatch_seconds_sum"];
      EXPECT_TRUE(dispatches.IsIntOrUint()) << "dispatches must be an integer";
      EXPECT_TRUE(dispatch_seconds_sum.IsNumeric()) << "dispatch_seconds_sum must be numeric";
      if (dispatches.IsIntOrUint())
        record.dispatches = dispatches.AsUInt64();
      if (dispatch_seconds_sum.IsNumeric())
        record.dispatch_seconds_sum = dispatch_seconds_sum.AsDouble();
    } else {
      ADD_FAILURE() << "unknown throughput record type: " << record.record;
    }

    auto families = families_value.AsMap();
    for (size_t i = 0; i < plugins::throughput::kInstructionFamilyCount; ++i) {
      const auto family = static_cast<plugins::throughput::InstructionFamily>(i);
      const auto family_name = plugins::throughput::ThroughputPlugin::family_name(family);
      const auto family_value = families[family_name.data()];
      EXPECT_TRUE(family_value.IsMap()) << "families." << family_name << " must be an object";
      if (!family_value.IsMap())
        continue;
      const auto values = family_value.AsMap();
      const auto instructions = values["instructions"];
      const auto execution_seconds = values["execution_seconds"];
      const auto execution_mips = values["execution_mips"];
      const auto dispatch_mips = values["dispatch_mips"];
      EXPECT_TRUE(instructions.IsIntOrUint())
          << "families." << family_name << ".instructions must be an integer";
      const auto valid = values["execution_timing_valid"];
      const auto untimed = values["untimed_instructions"];
      EXPECT_TRUE(valid.IsBool());
      EXPECT_TRUE(untimed.IsIntOrUint());
      record.execution_timing_valid[i] = valid.AsBool();
      record.untimed_instructions[i] = untimed.AsUInt64();
      EXPECT_EQ(valid.AsBool(), untimed.AsUInt64() == 0);
      if (valid.AsBool()) {
        EXPECT_TRUE(execution_seconds.IsNumeric());
        EXPECT_TRUE(execution_mips.IsNumeric());
      } else {
        EXPECT_TRUE(execution_seconds.IsNull());
        EXPECT_TRUE(execution_mips.IsNull());
      }
      EXPECT_TRUE(dispatch_mips.IsNumeric())
          << "families." << family_name << ".dispatch_mips must be numeric";
      if (instructions.IsIntOrUint())
        record.family_instructions[i] = instructions.AsUInt64();
      if (execution_seconds.IsNumeric())
        record.execution_seconds[i] = execution_seconds.AsDouble();
      if (execution_mips.IsNumeric())
        record.execution_mips[i] = execution_mips.AsDouble();
      if (dispatch_mips.IsNumeric())
        record.dispatch_mips[i] = dispatch_mips.AsDouble();
    }
    records.push_back(std::move(record));
  }
  return records;
}

uint64_t family_total(const ParsedThroughputRecord &record) {
  uint64_t total = 0;
  for (const uint64_t count : record.family_instructions)
    total += count;
  return total;
}

TEST(ThroughputPluginTest, ClassifiesExclusiveInstructionFamilies) {
  using plugins::throughput::InstructionFamily;
  using plugins::throughput::ThroughputPlugin;

  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction("s_add_u32")),
            InstructionFamily::Scalar);
  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction("v_add_f32")),
            InstructionFamily::Vector);
  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction("v_wmma_f32_16x16x16_f16")),
            InstructionFamily::Matrix);
  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction("v_swmmac_f32_16x16x32_f8")),
            InstructionFamily::Matrix);
  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction(
                "ds_read_b32", MEMORY_OP, std::make_unique<VectorMemState>(LOCAL_MEM))),
            InstructionFamily::Lds);
  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction("ds_read_b32", MEMORY_OP)),
            InstructionFamily::Lds);
  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction(
                "s_load_b32", MEMORY_OP, std::make_unique<ScalarMemState>())),
            InstructionFamily::Global);
  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction("global_load_b32", MEMORY_OP)),
            InstructionFamily::Global);
  EXPECT_EQ(
      ThroughputPlugin::classify(ThroughputTestInstruction("s_branch", BRANCH | IGNORES_EXEC)),
      InstructionFamily::Control);
  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction("s_endpgm", PROGRAM_TERMINATOR)),
            InstructionFamily::Control);
  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction("s_nop")),
            InstructionFamily::Control);
  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction("s_sleep")),
            InstructionFamily::Control);
  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction("s_delay_alu")),
            InstructionFamily::Control);
  EXPECT_EQ(ThroughputPlugin::classify(ThroughputTestInstruction("exp")), InstructionFamily::Other);
}

TEST(ThroughputPluginTest, ReportsExactWaveInstructionCountsAsJsonl) {
  using plugins::throughput::InstructionFamily;

  PluginFixture f(/*num_wf_slots=*/1);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<plugins::throughput::ThroughputPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  const uint32_t code[] = {vop1_encode(/*v_mov_b32 opcode=*/1, /*vdst=*/0, /*constant 0=*/128),
                           S_NOP, S_ENDPGM};
  f.run_kernel(code, 3);
  f.shutdown();

  const auto records = parse_throughput_jsonl(sink.str());
  ASSERT_EQ(records.size(), 2u);
  const auto &dispatch = records[0];
  EXPECT_EQ(dispatch.record, "dispatch");
  EXPECT_GT(dispatch.dispatch_id, 0u);
  EXPECT_FALSE(dispatch.kernel_name.empty());
  EXPECT_FALSE(dispatch.kernel_symbol.empty());
  EXPECT_GT(dispatch.workgroups, 0u);
  EXPECT_GT(dispatch.waves_per_workgroup, 0u);
  EXPECT_GT(dispatch.wall_seconds, 0.0);
  EXPECT_EQ(dispatch.wave_instructions, 3u);
  EXPECT_GT(dispatch.mips, 0.0);
  EXPECT_EQ(family_total(dispatch), dispatch.wave_instructions);
  EXPECT_EQ(dispatch.family_instructions[static_cast<size_t>(InstructionFamily::Vector)], 1u);
  EXPECT_EQ(dispatch.family_instructions[static_cast<size_t>(InstructionFamily::Control)], 2u);
  EXPECT_GT(dispatch.execution_seconds[static_cast<size_t>(InstructionFamily::Vector)], 0.0);
  EXPECT_GT(dispatch.execution_seconds[static_cast<size_t>(InstructionFamily::Control)], 0.0);
  EXPECT_GT(dispatch.execution_mips[static_cast<size_t>(InstructionFamily::Vector)], 0.0);
  EXPECT_GT(dispatch.execution_mips[static_cast<size_t>(InstructionFamily::Control)], 0.0);
  EXPECT_GT(dispatch.dispatch_mips[static_cast<size_t>(InstructionFamily::Vector)], 0.0);
  EXPECT_GT(dispatch.dispatch_mips[static_cast<size_t>(InstructionFamily::Control)], 0.0);

  const auto &summary = records[1];
  EXPECT_EQ(summary.record, "summary");
  EXPECT_EQ(summary.dispatches, 1u);
  EXPECT_GT(summary.dispatch_seconds_sum, 0.0);
  EXPECT_GT(summary.wall_seconds, 0.0);
  EXPECT_GT(summary.mips, 0.0);
  EXPECT_EQ(summary.wave_instructions, dispatch.wave_instructions);
  EXPECT_EQ(summary.family_instructions, dispatch.family_instructions);
  EXPECT_EQ(summary.execution_seconds, dispatch.execution_seconds);
  EXPECT_EQ(summary.execution_mips, dispatch.execution_mips);
  EXPECT_EQ(family_total(summary), summary.wave_instructions);
}

TEST(ThroughputPluginTest, TimesTerminatorWithoutAfterExecuteCallback) {
  using plugins::throughput::InstructionFamily;

  PluginFixture f(/*num_wf_slots=*/1);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<plugins::throughput::ThroughputPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  const uint32_t code[] = {S_ENDPGM};
  f.run_kernel(code, 1);
  f.shutdown();

  const auto records = parse_throughput_jsonl(sink.str());
  ASSERT_EQ(records.size(), 2u);
  const auto &dispatch = records[0];
  const size_t control = static_cast<size_t>(InstructionFamily::Control);
  EXPECT_EQ(dispatch.wave_instructions, 1u);
  EXPECT_EQ(dispatch.family_instructions[control], 1u);
  EXPECT_GT(dispatch.execution_seconds[control], 0.0);
}

class InstructionMixTestInstruction final : public Instruction {
public:
  explicit InstructionMixTestInstruction(std::string_view mnemonic, uint64_t flags = 0,
                                         std::unique_ptr<DynamicInstState> state = nullptr)
      : Instruction(mnemonic, nullptr) {
    flags_ = flags;
    set_data(std::move(state));
  }
};

class EncodedTestInstruction final : public Instruction {
public:
  EncodedTestInstruction(std::string_view mnemonic, uint16_t encoding_id, uint16_t opcode = 0,
                         int size = 4)
      : Instruction(mnemonic, nullptr) {
    encoding_id_ = encoding_id;
    opcode_ = opcode;
    size_ = size;
  }
};

struct ParsedMixRecord {
  std::string record;
  uint64_t dispatch_id = 0;
  std::string kernel_name;
  uint64_t dispatches = 0;
  uint64_t incomplete_dispatches = 0;
  bool complete = false;
  uint64_t wave_instructions = 0;
  uint64_t unique_mnemonics = 0;
  std::map<std::string, uint64_t> mnemonic_executions;
  std::map<std::string, std::string> mnemonic_family;
  std::map<std::string, uint64_t> mnemonic_encoding;
  std::map<std::string, uint64_t> mnemonic_encoding_bytes;
  std::map<std::string, uint64_t> family_mnemonics;
  std::map<std::string, uint64_t> family_executions;
};

std::vector<ParsedMixRecord> parse_instruction_mix_jsonl(std::string_view jsonl) {
  std::vector<ParsedMixRecord> records;
  std::istringstream lines{std::string(jsonl)};
  std::string line;
  while (std::getline(lines, line)) {
    if (line.empty())
      continue;

    flexbuffers::Builder builder;
    if (!plugin_detail::flexbuffer_from_json(line, builder)) {
      ADD_FAILURE() << "invalid JSONL record: " << line;
      continue;
    }
    auto root = flexbuffers::GetRoot(builder.GetBuffer());
    if (!root.IsMap()) {
      ADD_FAILURE() << "instruction-mix record is not a JSON object";
      continue;
    }
    auto object = root.AsMap();
    const auto schema = object["schema"];
    const auto record_type = object["record"];
    EXPECT_TRUE(schema.IsString()) << "schema must be a string";
    EXPECT_TRUE(record_type.IsString()) << "record must be a string";
    if (!schema.IsString() || !record_type.IsString())
      continue;
    EXPECT_EQ(schema.AsString().str(), "rocjitsu.instruction_mix.v1");

    ParsedMixRecord record;
    record.record = record_type.AsString().str();
    record.wave_instructions = object["wave_instructions"].AsUInt64();
    record.unique_mnemonics = object["unique_mnemonics"].AsUInt64();
    if (record.record == "dispatch") {
      record.dispatch_id = object["dispatch_id"].AsUInt64();
      record.kernel_name = object["kernel_name"].AsString().str();
    } else if (record.record == "summary") {
      record.dispatches = object["dispatches"].AsUInt64();
      record.incomplete_dispatches = object["incomplete_dispatches"].AsUInt64();
      record.complete = object["complete"].AsBool();
    } else {
      ADD_FAILURE() << "unknown instruction-mix record type: " << record.record;
      continue;
    }

    const auto families = object["families"];
    EXPECT_TRUE(families.IsMap()) << "families must be an object";
    if (families.IsMap()) {
      auto family_map = families.AsMap();
      auto keys = family_map.Keys();
      for (size_t i = 0; i < keys.size(); ++i) {
        const std::string name = keys[i].AsString().str();
        auto entry = family_map[name.c_str()].AsMap();
        record.family_mnemonics[name] = entry["mnemonics"].AsUInt64();
        record.family_executions[name] = entry["executions"].AsUInt64();
      }
    }

    const auto mnemonics = object["mnemonics"];
    EXPECT_TRUE(mnemonics.IsMap()) << "mnemonics must be an object";
    if (mnemonics.IsMap()) {
      auto mnemonic_map = mnemonics.AsMap();
      auto keys = mnemonic_map.Keys();
      for (size_t i = 0; i < keys.size(); ++i) {
        const std::string name = keys[i].AsString().str();
        auto entry = mnemonic_map[name.c_str()].AsMap();
        record.mnemonic_executions[name] = entry["executions"].AsUInt64();
        record.mnemonic_family[name] = entry["family"].AsString().str();
        record.mnemonic_encoding[name] = entry["encoding_id"].AsUInt64();
        record.mnemonic_encoding_bytes[name] = entry["encoding_bytes"].AsUInt64();
      }
    }
    records.push_back(std::move(record));
  }
  return records;
}

TEST(InstructionMixPluginTest, ClassifiesExclusiveInstructionFamilies) {
  using plugins::instruction_mix::InstructionFamily;
  using plugins::instruction_mix::InstructionMixPlugin;

  EXPECT_EQ(InstructionMixPlugin::classify(InstructionMixTestInstruction("s_add_u32")),
            InstructionFamily::Scalar);
  EXPECT_EQ(InstructionMixPlugin::classify(InstructionMixTestInstruction("v_add_f32")),
            InstructionFamily::Vector);
  EXPECT_EQ(InstructionMixPlugin::classify(
                InstructionMixTestInstruction("v_mfma_f32_16x16x16_f16", MFMA)),
            InstructionFamily::Matrix);
  EXPECT_EQ(
      InstructionMixPlugin::classify(InstructionMixTestInstruction("v_wmma_f32_16x16x16_f16")),
      InstructionFamily::Matrix);
  EXPECT_EQ(InstructionMixPlugin::classify(InstructionMixTestInstruction("ds_read_b32", MEMORY_OP)),
            InstructionFamily::Lds);
  EXPECT_EQ(
      InstructionMixPlugin::classify(InstructionMixTestInstruction("global_load_b32", MEMORY_OP)),
      InstructionFamily::Global);
  EXPECT_EQ(InstructionMixPlugin::classify(
                InstructionMixTestInstruction("s_branch", BRANCH | IGNORES_EXEC)),
            InstructionFamily::Control);
  EXPECT_EQ(
      InstructionMixPlugin::classify(InstructionMixTestInstruction("s_endpgm", PROGRAM_TERMINATOR)),
      InstructionFamily::Control);
  EXPECT_EQ(InstructionMixPlugin::classify(InstructionMixTestInstruction("exp")),
            InstructionFamily::Other);
}

TEST(InstructionMixPluginTest, ReportsExecutedMnemonicsAsJsonl) {
  PluginFixture f(/*num_wf_slots=*/1);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(
      f.plugin_group_->add(std::make_unique<plugins::instruction_mix::InstructionMixPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  const uint32_t code[] = {vop1_encode(/*v_mov_b32 opcode=*/1, /*vdst=*/0, /*constant 0=*/128),
                           S_NOP, S_NOP, S_ENDPGM};
  f.run_kernel(code, 4);
  f.shutdown();

  const auto records = parse_instruction_mix_jsonl(sink.str());
  ASSERT_EQ(records.size(), 2u);

  const auto &dispatch = records[0];
  EXPECT_EQ(dispatch.record, "dispatch");
  EXPECT_GT(dispatch.dispatch_id, 0u);
  EXPECT_FALSE(dispatch.kernel_name.empty());
  EXPECT_EQ(dispatch.wave_instructions, 4u);
  // v_mov_b32, s_nop and s_endpgm: three distinct mnemonics over four executions.
  EXPECT_EQ(dispatch.unique_mnemonics, 3u);
  EXPECT_EQ(dispatch.mnemonic_executions.size(), 3u);
  EXPECT_EQ(dispatch.mnemonic_executions.at("s_nop"), 2u);
  EXPECT_EQ(dispatch.mnemonic_executions.at("s_endpgm"), 1u);
  EXPECT_EQ(dispatch.mnemonic_family.at("s_nop"), "control");
  EXPECT_EQ(dispatch.mnemonic_family.at("s_endpgm"), "control");
  EXPECT_EQ(dispatch.family_mnemonics.at("control"), 2u);
  EXPECT_EQ(dispatch.family_executions.at("control"), 3u);
  EXPECT_EQ(dispatch.family_mnemonics.at("vector"), 1u);
  EXPECT_EQ(dispatch.family_executions.at("vector"), 1u);
  EXPECT_EQ(dispatch.family_mnemonics.at("matrix"), 0u);

  uint64_t family_execution_total = 0;
  uint64_t family_mnemonic_total = 0;
  for (const auto &[name, count] : dispatch.family_executions)
    family_execution_total += count;
  for (const auto &[name, count] : dispatch.family_mnemonics)
    family_mnemonic_total += count;
  EXPECT_EQ(family_execution_total, dispatch.wave_instructions);
  EXPECT_EQ(family_mnemonic_total, dispatch.unique_mnemonics);

  const auto &summary = records[1];
  EXPECT_EQ(summary.record, "summary");
  EXPECT_EQ(summary.dispatches, 1u);
  // The run drained, so nothing was folded in unfinished and absence from this
  // summary really does mean the mnemonic never executed.
  EXPECT_EQ(summary.incomplete_dispatches, 0u);
  EXPECT_TRUE(summary.complete);
  EXPECT_EQ(summary.wave_instructions, dispatch.wave_instructions);
  EXPECT_EQ(summary.unique_mnemonics, dispatch.unique_mnemonics);
  EXPECT_EQ(summary.mnemonic_executions, dispatch.mnemonic_executions);
}

// The instruction-mix report is only joinable with the throughput report if
// the two plugins agree on what family an instruction is in. They now classify
// through one shared header (plugins/instruction_family.h) rather than through
// per-plugin copies, and this pins the agreement so a future divergence -- a
// plugin growing its own override, or the family list being reordered on one
// side -- fails here instead of silently desynchronising the two schemas.
TEST(InstructionMixPluginTest, AgreesWithThroughputOnInstructionFamilies) {
  using plugins::instruction_mix::InstructionFamily;
  struct Case {
    const char *mnemonic;
    uint64_t flags;
    InstructionFamily expected;
  };
  // One representative per family, plus the cases where the two classifiers
  // could plausibly drift: flag-driven vs prefix-driven, and the mnemonics
  // that are control flow despite a scalar prefix.
  //
  // The expectations matter as much as the agreement: two classifiers can agree
  // and both be wrong. The image, tensor, LDS-direct, typed-buffer,
  // scalar-buffer and split-barrier entries below are exactly that case --
  // their generated constructors set neither MEMORY_OP nor BARRIER, so before
  // the mnemonic fallbacks they agreed on `other` and `scalar` respectively.
  constexpr Case kCases[] = {
      {"s_add_u32", 0, InstructionFamily::Scalar},
      {"v_add_f32", 0, InstructionFamily::Vector},
      {"v_mfma_f32_16x16x16_f16", MFMA, InstructionFamily::Matrix},
      {"v_smfmac_f32_16x16x32_f16", 0, InstructionFamily::Matrix},
      {"v_wmma_f32_16x16x16_f16", 0, InstructionFamily::Matrix},
      {"v_swmmac_f32_16x16x32_f8", 0, InstructionFamily::Matrix},
      {"ds_read_b32", MEMORY_OP, InstructionFamily::Lds},
      {"global_load_b32", MEMORY_OP, InstructionFamily::Global},
      {"buffer_load_dword", MEMORY_OP, InstructionFamily::Global},
      {"s_load_dword", MEMORY_OP, InstructionFamily::Global},
      // Image and tensor encodings carry no MEMORY_OP flag (see
      // generated/rdna4/vimage.cpp and generated/cdna5/vimage.cpp), so these
      // reach the classifier exactly as the decoder produces them.
      {"image_load", 0, InstructionFamily::Global},
      {"image_sample", 0, InstructionFamily::Global},
      {"image_atomic_add", 0, InstructionFamily::Global},
      {"tensor_load_to_lds", 0, InstructionFamily::Global},
      {"tensor_store_from_lds", 0, InstructionFamily::Global},
      // The same gap, in the LDS-direct and typed-buffer encodings: RDNA4
      // `ds_direct_load`/`ds_param_load` (generated/rdna4/vdsdir.cpp), RDNA3.5
      // `lds_direct_load`/`lds_param_load` (generated/rdna3_5/ldsdir.cpp) and
      // the RDNA4 `tbuffer_*` set (generated/rdna4/vbuffer.cpp, where the
      // neighbouring `buffer_*` constructors do set MEMORY_OP) all arrive
      // unflagged, so a flag-only classifier reports them as `other`.
      {"ds_direct_load", 0, InstructionFamily::Lds},
      {"ds_param_load", 0, InstructionFamily::Lds},
      {"lds_direct_load", 0, InstructionFamily::Lds},
      {"lds_param_load", 0, InstructionFamily::Lds},
      {"tbuffer_load_format_x", 0, InstructionFamily::Global},
      {"tbuffer_store_format_xyzw", 0, InstructionFamily::Global},
      // Unflagged on at least one architecture while the same mnemonic is
      // flagged on another; the prefix fallback keeps the family stable.
      {"buffer_atomic_cmpswap_b32", 0, InstructionFamily::Global},
      {"global_atomic_cmpswap_b32", 0, InstructionFamily::Global},
      {"flat_prefetch_b8", 0, InstructionFamily::Global},
      {"ds_bpermute_b32", 0, InstructionFamily::Lds},
      // Scalar memory. generated/*/smem.cpp stopped setting MEMORY_OP
      // altogether when the decoded memory-issue metadata landed, so these
      // arrive unflagged and the bare `s_` prefix would bucket them with ALU
      // work. The flagged `s_buffer_load_dword` case is kept to pin the other
      // path: a synthetic or model-only instruction may still carry the flag.
      {"s_buffer_atomic_add", 0, InstructionFamily::Global},
      {"s_buffer_load_dword", MEMORY_OP, InstructionFamily::Global},
      {"s_load_b32", 0, InstructionFamily::Global},
      {"s_load_dwordx4", 0, InstructionFamily::Global},
      {"s_store_dword", 0, InstructionFamily::Global},
      {"s_atomic_add", 0, InstructionFamily::Global},
      {"s_scratch_load_dword", 0, InstructionFamily::Global},
      // Scalar cache maintenance and prefetch: MEMORY_WAIT_PRODUCER only.
      {"s_dcache_inv", 0, InstructionFamily::Global},
      {"s_dcache_wb", 0, InstructionFamily::Global},
      {"s_prefetch_data", 0, InstructionFamily::Global},
      {"s_prefetch_inst", 0, InstructionFamily::Global},
      {"s_atc_probe", 0, InstructionFamily::Global},
      {"s_gl1_inv", 0, InstructionFamily::Global},
      // Not memory: these carry MEMORY_WAIT_PRODUCER too, so the flag is not a
      // usable signal on its own and these must stay where they are.
      {"s_memtime", 0, InstructionFamily::Scalar},
      {"s_sendmsg", 0, InstructionFamily::Scalar},
      {"s_get_waveid_in_workgroup", 0, InstructionFamily::Scalar},
      {"s_branch", BRANCH | IGNORES_EXEC, InstructionFamily::Control},
      {"s_cbranch_execz", COND_BRANCH, InstructionFamily::Control},
      {"s_endpgm", PROGRAM_TERMINATOR, InstructionFamily::Control},
      {"s_waitcnt", WAITCNT, InstructionFamily::Control},
      {"s_barrier", BARRIER, InstructionFamily::Control},
      // On CDNA5 only s_barrier_wait carries BARRIER; the rest of the split
      // barrier family arrives unflagged.
      {"s_barrier_leave", 0, InstructionFamily::Control},
      {"s_barrier_init", 0, InstructionFamily::Control},
      {"s_barrier_join", 0, InstructionFamily::Control},
      {"s_barrier_signal", 0, InstructionFamily::Control},
      {"s_barrier_signal_isfirst", 0, InstructionFamily::Control},
      {"s_nop", 0, InstructionFamily::Control},
      {"s_sleep", 0, InstructionFamily::Control},
      {"s_delay_alu", 0, InstructionFamily::Control},
      // LDS-pipe synchronisation and no-ops: named like LDS traffic, but they
      // move no data, so the control carve-out is checked before the memory
      // fallback. GWS rows cover both pre-execute (no flags) and post-execute
      // MEMORY_OP flags used for wait-counter accounting.
      {"ds_nop", 0, InstructionFamily::Control},
      {"ds_gws_barrier", 0, InstructionFamily::Control},
      {"ds_gws_init", 0, InstructionFamily::Control},
      {"ds_gws_sema_v", 0, InstructionFamily::Control},
      {"ds_gws_sema_p", 0, InstructionFamily::Control},
      {"ds_gws_sema_br", 0, InstructionFamily::Control},
      {"ds_gws_sema_release_all", 0, InstructionFamily::Control},
      {"ds_gws_init", MEMORY_OP, InstructionFamily::Control},
      {"ds_gws_barrier", MEMORY_OP, InstructionFamily::Control},
      {"ds_gws_sema_v", MEMORY_OP, InstructionFamily::Control},
      {"ds_gws_sema_p", MEMORY_OP, InstructionFamily::Control},
      {"ds_gws_sema_br", MEMORY_OP, InstructionFamily::Control},
      {"ds_gws_sema_release_all", MEMORY_OP, InstructionFamily::Control},

      {"exp", 0, InstructionFamily::Other},
  };

  for (const auto &c : kCases) {
    const auto mix_family = plugins::instruction_mix::InstructionMixPlugin::classify(
        InstructionMixTestInstruction(c.mnemonic, c.flags));
    const auto throughput_family = plugins::throughput::ThroughputPlugin::classify(
        ThroughputTestInstruction(c.mnemonic, c.flags));
    EXPECT_EQ(mix_family, c.expected) << "wrong family for " << c.mnemonic;
    EXPECT_EQ(plugins::instruction_mix::InstructionMixPlugin::family_name(mix_family),
              plugins::throughput::ThroughputPlugin::family_name(throughput_family))
        << "instruction-mix and throughput disagree on " << c.mnemonic;
  }

  // The family name lists must also line up index for index, since the JSONL
  // schemas present them in enum order.
  ASSERT_EQ(plugins::instruction_mix::kInstructionFamilyCount,
            plugins::throughput::kInstructionFamilyCount);
  for (size_t i = 0; i < plugins::instruction_mix::kInstructionFamilyCount; ++i) {
    EXPECT_EQ(plugins::instruction_mix::InstructionMixPlugin::family_name(
                  static_cast<plugins::instruction_mix::InstructionFamily>(i)),
              plugins::throughput::ThroughputPlugin::family_name(
                  static_cast<plugins::throughput::InstructionFamily>(i)))
        << "family order differs at index " << i;
  }
}

// A prefix table can only recognise families it has been told about. The CDNA5
// `cluster_load_*` set follows no memory naming convention, carries no
// MEMORY_OP, and would be reported as `other` -- but it does declare its
// completion domain, so the classifier can still place it.
TEST(InstructionMixPluginTest, ClassifiesMemoryByIssueMetadataWhenThePrefixIsUnknown) {
  using plugins::instruction_mix::InstructionFamily;
  struct Case {
    const char *mnemonic;
    amdgpu::MemoryCompletionClass completion_class;
    InstructionFamily expected;
  };
  const Case kCases[] = {
      {"cluster_load_b32", amdgpu::MemoryCompletionClass::VMEM, InstructionFamily::Global},
      {"cluster_load_async_to_lds_b32", amdgpu::MemoryCompletionClass::ASYNC_LOAD,
       InstructionFamily::Global},
      {"some_future_lds_op", amdgpu::MemoryCompletionClass::LDS, InstructionFamily::Lds},
  };

  for (const auto &c : kCases) {
    const IssueMetadataTestInstruction inst(c.mnemonic, c.completion_class);
    const auto mix = plugins::instruction_mix::InstructionMixPlugin::classify(inst);
    const auto thr = plugins::throughput::ThroughputPlugin::classify(inst);
    EXPECT_EQ(mix, c.expected) << "wrong family for " << c.mnemonic;
    EXPECT_EQ(plugins::instruction_mix::InstructionMixPlugin::family_name(mix),
              plugins::throughput::ThroughputPlugin::family_name(thr))
        << "classifiers disagree on " << c.mnemonic;
  }
}

// Merging walks unordered maps, so a mnemonic seen through two encodings must
// not have its reported encoding decided by hash order -- the whole point of
// the report is that two runs of the same workload diff cleanly.
TEST(InstructionMixPluginTest, PicksTheSameEncodingRegardlessOfMergeOrder) {
  PluginFixture f(/*num_wf_slots=*/1);
  // Wavefront slots are created on demand by the CU's dispatch path, so take
  // the wave from there; reaching into an unpopulated slot would dereference
  // null. The same slot is reused on purpose -- recycling is what makes the
  // merge order vary.
  amdgpu::Wavefront *slot = f.cu()->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(slot, nullptr);

  auto run = [&](bool reversed) {
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    ExecutionPluginGroup group(std::move(sink_config));
    EXPECT_TRUE(group.add(std::make_unique<plugins::instruction_mix::InstructionMixPlugin>()));

    // Two waves in the same dispatch reach the same mnemonic through different
    // encodings. Feeding them in either order must produce one answer.
    const uint16_t first = reversed ? 421 : 417;
    const uint16_t second = reversed ? 417 : 421;
    for (const uint16_t encoding : {first, second}) {
      amdgpu::Wavefront &wf = *slot;
      wf.set_dispatch_id(1);
      group.onAmdgpuWavefrontDispatched(wf);
      EncodedTestInstruction inst("v_lshlrev_b64", encoding);
      group.onAmdgpuBeforeExecuteInstruction(0, inst, wf);
      group.onAmdgpuWavefrontHalted(wf);
    }
    group.onAmdgpuDispatchExecutionEnd(1);
    group.onShutdown();
    return sink.str();
  };

  const auto forward = parse_instruction_mix_jsonl(run(false));
  const auto backward = parse_instruction_mix_jsonl(run(true));
  ASSERT_FALSE(forward.empty());
  ASSERT_FALSE(backward.empty());
  EXPECT_EQ(forward.front().mnemonic_encoding, backward.front().mnemonic_encoding);
  // Both sightings are still counted, whichever encoding was reported.
  EXPECT_EQ(forward.front().mnemonic_executions.at("v_lshlrev_b64"), 2u);
  EXPECT_EQ(backward.front().mnemonic_executions.at("v_lshlrev_b64"), 2u);
}

// A mnemonic can be sighted through encodings that tie on every other ordered
// field and still print a different size: the generated VOP1 constructors set
// encoding_id_ and opcode_ before widening size_ for the DPP, SDWA and literal
// forms, so v_mov_b32_e32 is reachable at both four and eight bytes with the
// same encoding id and opcode. If the tie-break does not look at the size, the
// reported encoding_bytes is decided by whichever wave merged first and the
// report stops being diffable.
TEST(InstructionMixPluginTest, PicksTheSameEncodingSizeRegardlessOfMergeOrder) {
  PluginFixture f(/*num_wf_slots=*/1);
  // Wavefront slots are created on demand by the CU's dispatch path, so take
  // the wave from there; reaching into an unpopulated slot would dereference
  // null. The same slot is reused on purpose -- recycling is what makes the
  // merge order vary.
  amdgpu::Wavefront *slot = f.cu()->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(slot, nullptr);

  auto run = [&](bool reversed) {
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    ExecutionPluginGroup group(std::move(sink_config));
    EXPECT_TRUE(group.add(std::make_unique<plugins::instruction_mix::InstructionMixPlugin>()));

    // Same mnemonic, same encoding id, same opcode, same dispatch: the size is
    // the only field that separates the two sightings.
    const int first = reversed ? 8 : 4;
    const int second = reversed ? 4 : 8;
    for (const int size : {first, second}) {
      amdgpu::Wavefront &wf = *slot;
      wf.set_dispatch_id(1);
      group.onAmdgpuWavefrontDispatched(wf);
      EncodedTestInstruction inst("v_mov_b32_e32", /*encoding_id=*/63, /*opcode=*/1, size);
      group.onAmdgpuBeforeExecuteInstruction(0, inst, wf);
      group.onAmdgpuWavefrontHalted(wf);
    }
    group.onAmdgpuDispatchExecutionEnd(1);
    group.onShutdown();
    return sink.str();
  };

  const auto forward = parse_instruction_mix_jsonl(run(false));
  const auto backward = parse_instruction_mix_jsonl(run(true));
  ASSERT_FALSE(forward.empty());
  ASSERT_FALSE(backward.empty());
  EXPECT_EQ(forward.front().mnemonic_encoding_bytes, backward.front().mnemonic_encoding_bytes);
  // The smaller sighting is the one the total order keeps.
  EXPECT_EQ(forward.front().mnemonic_encoding_bytes.at("v_mov_b32_e32"), 4u);
  EXPECT_EQ(forward.front().mnemonic_executions.at("v_mov_b32_e32"), 2u);
  EXPECT_EQ(backward.front().mnemonic_executions.at("v_mov_b32_e32"), 2u);
}

// A bounded run (rj_vm_request_exit) stops with work in flight: workers are
// joined and plugin shutdown follows immediately, so a dispatch can have halted
// waves but never reach execution-end. Those instructions were still executed,
// and the report's contract is that absence means non-execution -- so the
// summary has to carry them, and has to say that it is a partial view.
TEST(InstructionMixPluginTest, ShutdownKeepsCoverageFromUnfinishedDispatches) {
  PluginFixture f(/*num_wf_slots=*/1);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  ExecutionPluginGroup group(std::move(sink_config));
  ASSERT_TRUE(group.add(std::make_unique<plugins::instruction_mix::InstructionMixPlugin>()));

  // Wavefront slots are created on demand by the CU's dispatch path; an
  // unpopulated slot is null.
  amdgpu::Wavefront *slot = f.cu()->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(slot, nullptr);

  // Dispatch 1 completes; dispatch 2 halts its wave but never ends.
  for (const uint32_t dispatch_id : {1u, 2u}) {
    amdgpu::Wavefront &wf = *slot;
    wf.set_dispatch_id(dispatch_id);
    group.onAmdgpuWavefrontDispatched(wf);
    InstructionMixTestInstruction inst(dispatch_id == 1 ? "s_nop" : "v_add_f32");
    group.onAmdgpuBeforeExecuteInstruction(0, inst, wf);
    group.onAmdgpuWavefrontHalted(wf);
  }
  group.onAmdgpuDispatchExecutionEnd(1);
  group.onShutdown();

  const auto records = parse_instruction_mix_jsonl(sink.str());
  ASSERT_EQ(records.size(), 2u);

  // Only the finished dispatch gets a per-dispatch record: dispatch 2's totals
  // are not final, so reporting them as a dispatch line would be a lie.
  const auto &dispatch = records[0];
  EXPECT_EQ(dispatch.record, "dispatch");
  EXPECT_EQ(dispatch.dispatch_id, 1u);
  EXPECT_EQ(dispatch.mnemonic_executions.size(), 1u);
  EXPECT_EQ(dispatch.mnemonic_executions.count("v_add_f32"), 0u);

  const auto &summary = records[1];
  EXPECT_EQ(summary.record, "summary");
  EXPECT_EQ(summary.dispatches, 1u);
  EXPECT_EQ(summary.incomplete_dispatches, 1u);
  EXPECT_FALSE(summary.complete);
  // The abandoned dispatch's mnemonic is in the union, so a reader cannot take
  // its absence for non-execution.
  EXPECT_EQ(summary.wave_instructions, 2u);
  EXPECT_EQ(summary.unique_mnemonics, 2u);
  EXPECT_EQ(summary.mnemonic_executions.at("s_nop"), 1u);
  EXPECT_EQ(summary.mnemonic_executions.at("v_add_f32"), 1u);
  EXPECT_EQ(summary.family_executions.at("vector"), 1u);
  EXPECT_EQ(summary.family_executions.at("control"), 1u);
}

// The mnemonic a plugin observes does not always point to static storage:
// the generated FLAT encoding on every architecture and VOPD on gfx11/gfx12
// and CDNA5 synthesise it into a per-instruction std::string member
// (generated/cdna4/encodings.cpp Flat::Flat, generated/cdna5/vopd.cpp), so a
// stored std::string_view outlives its characters. The plugin must copy on
// first sight; this test fails with a view-keyed map.
TEST(InstructionMixPluginTest, OwnsMnemonicStorageWhenTheSourceIsNotStatic) {
  PluginFixture f(/*num_wf_slots=*/1);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  ExecutionPluginGroup group(std::move(sink_config));
  ASSERT_TRUE(group.add(std::make_unique<plugins::instruction_mix::InstructionMixPlugin>()));

  // Wavefront slots are created on demand by the CU's dispatch path; an
  // unpopulated slot is null.
  amdgpu::Wavefront *slot = f.cu()->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(slot, nullptr);

  amdgpu::Wavefront &wf = *slot;
  wf.set_dispatch_id(1);
  group.onAmdgpuWavefrontDispatched(wf);
  {
    std::string transient_mnemonic = "global_load_dwordx4_transient_storage";
    InstructionMixTestInstruction inst(transient_mnemonic, MEMORY_OP);
    group.onAmdgpuBeforeExecuteInstruction(0, inst, wf);
    // Scribble over the characters the view pointed at, then release them.
    transient_mnemonic.assign(transient_mnemonic.size(), 'X');
    transient_mnemonic = std::string();
  }
  group.onAmdgpuWavefrontHalted(wf);
  group.onAmdgpuDispatchExecutionEnd(1);
  group.onShutdown();

  const auto records = parse_instruction_mix_jsonl(sink.str());
  ASSERT_FALSE(records.empty());
  const auto &dispatch = records.front();
  ASSERT_EQ(dispatch.mnemonic_executions.size(), 1u);
  EXPECT_EQ(dispatch.mnemonic_executions.begin()->first, "global_load_dwordx4_transient_storage");
  EXPECT_EQ(dispatch.mnemonic_executions.begin()->second, 1u);
}

class AsyncEventPlugin final : public ExecutionPlugin {
public:
  explicit AsyncEventPlugin(std::string name = "async-events") : ExecutionPlugin(std::move(name)) {}
  bool supports_async_instructions() const override { return true; }
  bool observes_sgpr_reads() const override { return false; }
  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &inst, Wavefront &wf) override {
    if (inst.mnemonic() == "s_set_vgpr_msb")
      hazard_before_set_vgpr = wf.setreg_vgpr_msb_hazard();
  }
  void onAmdgpuAsyncInstructionIssued(uint64_t, const Instruction &inst, Wavefront &wf) override {
    EXPECT_EQ(std::this_thread::get_id(), issuer_thread);
    EXPECT_EQ(inst.mnemonic(), "v_wmma_f32_16x16x64_fp8_fp8");
    hazard_at_async_issue = wf.setreg_vgpr_msb_hazard();
    ++issued;
  }
  std::thread::id issuer_thread = std::this_thread::get_id();
  unsigned issued = 0;
  std::optional<bool> hazard_at_async_issue;
  std::optional<bool> hazard_before_set_vgpr;
};

TEST(ExecutionPluginTest, AsyncSupportComesFromCapabilities) {
  ExecutionPluginGroup group(PluginSinkConfig{});
  EXPECT_TRUE(group.supports_async_instructions());
  ASSERT_TRUE(group.add(std::make_unique<AsyncEventPlugin>()));
  EXPECT_TRUE(group.supports_async_instructions());
  ASSERT_TRUE(group.add(std::make_unique<KernelLoggingPlugin>()));
  EXPECT_TRUE(group.supports_async_instructions());
  // A familiar name must not bypass the observation contract.
  ASSERT_TRUE(group.add(std::make_unique<ExecutionPlugin>("throughput")));
  EXPECT_FALSE(group.supports_async_instructions());

  ExecutionPluginGroup consan(PluginSinkConfig{});
  ASSERT_TRUE(consan.add(std::make_unique<RaceDetectorPlugin>()));
  EXPECT_FALSE(consan.supports_async_instructions());
}

std::vector<uint32_t> independent_wmma_kernel() {
  std::vector<uint32_t> code;
  // Zero operands are sufficient: this test checks real offload and callback
  // lifetimes; the matrix suites separately compare nonzero numerical results.
  for (uint8_t dst : {uint8_t{64}, uint8_t{96}}) {
    const auto words = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x64Fp8Fp8Vop3p,
                                          {.vdst = dst,
                                           .src0 = 256,
                                           .src1 = 288,
                                           .src2 = static_cast<uint16_t>(256 + dst),
                                           .opsel_hi = 3});
    code.insert(code.end(), words.begin(), words.end());
  }
  code.push_back(cdna5::build_sopp(cdna5::kSEndpgmSopp, {})[0]);
  return code;
}

TEST(ExecutionPluginTest, AsyncMmaEndsSetregVgprMsbAdjacencyHazard) {
  PluginFixture f(1, "cdna5", 32, 128, 256, 1, /*async_helpers=*/4);
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto observer = std::make_unique<AsyncEventPlugin>();
  auto *events = observer.get();
  ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  constexpr auto setreg =
      cdna5::build_sopk(cdna5::kSSetregImm32B32Sopk, {.simm16 = amdgpu::MODE_HWREG});
  constexpr auto set_vgpr_msb = cdna5::build_sopp(cdna5::kSSetVgprMsbSopp, {.simm16 = 0xC3});
  std::vector<uint32_t> code{setreg[0], 0u};
  auto mma_code = independent_wmma_kernel();
  code.insert(code.end(), mma_code.begin(), mma_code.end() - 1);
  code.push_back(set_vgpr_msb[0]);
  code.push_back(mma_code.back());

  f.run_kernel(code.data(), code.size(), 32, 32);
  f.shutdown();
  ASSERT_EQ(events->issued, 1u);
  ASSERT_TRUE(events->hazard_at_async_issue.has_value());
  EXPECT_FALSE(*events->hazard_at_async_issue);
  ASSERT_TRUE(events->hazard_before_set_vgpr.has_value());
  EXPECT_FALSE(*events->hazard_before_set_vgpr);
}

TEST(ExecutionPluginTest, SynchronousObserverDisablesActualMmaOffload) {
  PluginFixture f(1, "cdna5", 32, 128, 256, 1, /*async_helpers=*/4);
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto async_observer = std::make_unique<AsyncEventPlugin>();
  auto *async_events = async_observer.get();
  ASSERT_TRUE(f.plugin_group_->add(std::move(async_observer)));
  auto sync_observer = std::make_unique<OrderingPlugin>();
  auto *sync_events = sync_observer.get();
  ASSERT_TRUE(f.plugin_group_->add(std::move(sync_observer)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();
  const auto code = independent_wmma_kernel();
  f.run_kernel(code.data(), code.size(), 32, 32);
  f.shutdown();
  EXPECT_EQ(async_events->issued, 0u);
  unsigned before = 0, after = 0;
  for (const auto &event : sync_events->events) {
    if (!event.mnemonic.starts_with("v_wmma_"))
      continue;
    before += event.kind == HookEvent::BEFORE_INSTRUCTION;
    after += event.kind == HookEvent::AFTER_INSTRUCTION;
  }
  EXPECT_EQ(before, 2u);
  EXPECT_EQ(after, before);
}

enum class AsyncFailurePoint { HelperRegisterRead, Issue };

struct AsyncFailureObservation {
  static constexpr unsigned all_plugins = 0b111;
  struct Record {
    unsigned issued = 0;
    bool destroyed = false;
  };
  std::thread::id issuer = std::this_thread::get_id();
  std::atomic<bool> injected{false};
  std::map<uint64_t, Record> records;
};

// These WMMA handlers have no dynamic instruction state. Attach a test-only
// lifetime probe to verify issuer-thread destruction after a callback failure.
class AsyncDestructionProbe final : public DynamicInstState {
public:
  AsyncDestructionProbe(AsyncFailureObservation &observations, uint64_t pc)
      : observations_(observations), pc_(pc) {}
  ~AsyncDestructionProbe() override {
    EXPECT_EQ(std::this_thread::get_id(), observations_.issuer);
    auto &record = observations_.records.at(pc_);
    EXPECT_EQ(record.issued, AsyncFailureObservation::all_plugins);
    EXPECT_FALSE(record.destroyed);
    record.destroyed = true;
  }

private:
  AsyncFailureObservation &observations_;
  uint64_t pc_;
};

class AsyncFailurePlugin final : public ExecutionPlugin {
public:
  AsyncFailurePlugin(AsyncFailureObservation &observations, unsigned index, AsyncFailurePoint point)
      : ExecutionPlugin(std::format("async-failure-{}", index)), observations_(observations),
        index_(index), point_(point) {}
  bool supports_async_instructions() const override { return true; }
  bool observes_sgpr_reads() const override { return false; }
  void onAmdgpuAsyncInstructionIssued(uint64_t pc, const Instruction &inst, Wavefront &) override {
    EXPECT_EQ(std::this_thread::get_id(), observations_.issuer);
    auto &record = observations_.records[pc];
    EXPECT_FALSE(record.destroyed);
    EXPECT_EQ(record.issued & (1u << index_), 0u);
    record.issued |= 1u << index_;
    if (index_ == 0) {
      EXPECT_EQ(inst.data(), nullptr);
      const_cast<Instruction &>(inst).set_data(
          std::make_unique<AsyncDestructionProbe>(observations_, pc));
    }
    inject(AsyncFailurePoint::Issue);
  }
  void onAmdgpuReadVgprLanes(const Wavefront *, uint32_t, uint64_t, uint8_t) override {
    if (std::this_thread::get_id() != observations_.issuer)
      inject(AsyncFailurePoint::HelperRegisterRead);
  }

private:
  void inject(AsyncFailurePoint point) {
    // The first and last plugins never throw. The middle plugin must not keep
    // later observers from receiving the issue notification.
    if (index_ == 1 && point_ == point && !observations_.injected.exchange(true))
      throw std::runtime_error("injected async plugin failure");
  }
  AsyncFailureObservation &observations_;
  unsigned index_;
  AsyncFailurePoint point_;
};

class AsyncPluginFailureTest : public ::testing::TestWithParam<AsyncFailurePoint> {};

TEST_P(AsyncPluginFailureTest, DestroysEveryAcceptedInstructionOnIssuerAfterFailure) {
  AsyncFailureObservation observations;
  PluginFixture f(1, "cdna5", 32, 128, 256, 1, /*async_helpers=*/4);
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  for (unsigned index = 0; index != 3; ++index)
    ASSERT_TRUE(f.plugin_group_->add(
        std::make_unique<AsyncFailurePlugin>(observations, index, GetParam())));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();
  const auto code = independent_wmma_kernel();
  EXPECT_THROW(f.run_kernel(code.data(), code.size(), 32, 32), std::runtime_error);
  EXPECT_TRUE(observations.injected.load());
  ASSERT_FALSE(observations.records.empty());
  for (const auto &[pc, record] : observations.records) {
    SCOPED_TRACE(pc);
    EXPECT_EQ(record.issued, AsyncFailureObservation::all_plugins);
    EXPECT_TRUE(record.destroyed);
  }
  f.shutdown();
}

INSTANTIATE_TEST_SUITE_P(CallbackFailures, AsyncPluginFailureTest,
                         ::testing::Values(AsyncFailurePoint::HelperRegisterRead,
                                           AsyncFailurePoint::Issue),
                         [](const ::testing::TestParamInfo<AsyncFailurePoint> &info) {
                           switch (info.param) {
                           case AsyncFailurePoint::HelperRegisterRead:
                             return "HelperRegisterRead";
                           case AsyncFailurePoint::Issue:
                             return "Issue";
                           }
                           return "Unknown";
                         });

TEST(ThroughputPluginTest, AsyncMmaCountsHaveExplicitlyUnavailableHandlerTiming) {
  PluginFixture f(1, "cdna5", 32, 128, 256, 1, /*async_helpers=*/4);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<plugins::throughput::ThroughputPlugin>()));
  auto observer = std::make_unique<AsyncEventPlugin>();
  auto *events = observer.get();
  ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();
  const auto code = independent_wmma_kernel();
  f.run_kernel(code.data(), code.size(), 32, 32);
  f.shutdown();
  ASSERT_GT(events->issued, 0u);
  const auto records = parse_throughput_jsonl(sink.str());
  ASSERT_EQ(records.size(), 2u);
  const auto matrix = static_cast<size_t>(plugins::throughput::InstructionFamily::Matrix);
  const auto control = static_cast<size_t>(plugins::throughput::InstructionFamily::Control);
  for (const auto &record : records) {
    EXPECT_EQ(record.wave_instructions, 3u);
    EXPECT_EQ(record.family_instructions[matrix], 2u);
    EXPECT_EQ(record.untimed_instructions[matrix], events->issued);
    EXPECT_FALSE(record.execution_timing_valid[matrix]);
    EXPECT_TRUE(record.execution_timing_valid[control]);
    EXPECT_GT(record.dispatch_mips[matrix], 0.0);
    EXPECT_GT(record.wall_seconds, 0.0);
  }
}

// The group ANDs supports_async_instructions() across its members, so a plugin
// that inherits the false default silently disables MMA offload for everything
// loaded beside it. An instruction-mix run must not change the execution mode
// it is there to describe -- least of all when paired with throughput, whose
// own report then measures something other than what it would have measured
// alone.
TEST(InstructionMixPluginTest, CombinedGroupKeepsOffloadAndDropsSgprCallbacks) {
  PluginSinkConfig sink_config;
  sink_config.emplace<StringSink>();
  ExecutionPluginGroup group(std::move(sink_config));
  ASSERT_TRUE(group.add(std::make_unique<plugins::throughput::ThroughputPlugin>()));
  EXPECT_TRUE(group.supports_async_instructions());
  EXPECT_FALSE(group.observes_sgpr_reads());

  // Adding instruction-mix must not take either capability away.
  ASSERT_TRUE(group.add(std::make_unique<plugins::instruction_mix::InstructionMixPlugin>()));
  EXPECT_TRUE(group.supports_async_instructions());
  EXPECT_FALSE(group.observes_sgpr_reads());

  // Throughput times instructions at after-execute, so the combined group must
  // keep that hook even though instruction-mix declines it. These are ORed.
  EXPECT_TRUE(group.observes_after_execute_instruction());
}

// Alone, instruction-mix should ask for nothing it does not consume. The
// register hooks matter most: any member requesting them keeps
// ComputeUnit::observes_register_access_ on for the whole run.
TEST(InstructionMixPluginTest, AloneItSubscribesToOnlyTheHooksItConsumes) {
  PluginSinkConfig sink_config;
  sink_config.emplace<StringSink>();
  ExecutionPluginGroup group(std::move(sink_config));
  ASSERT_TRUE(group.add(std::make_unique<plugins::instruction_mix::InstructionMixPlugin>()));

  // Counted here.
  EXPECT_TRUE(group.observes_before_execute_instruction());
  EXPECT_TRUE(group.supports_async_instructions());

  // Never consumed.
  EXPECT_FALSE(group.observes_after_execute_instruction());
  EXPECT_FALSE(group.observes_memory_instruction_routing());
  EXPECT_FALSE(group.observes_vgpr_reads());
  EXPECT_FALSE(group.observes_vgpr_writes());
  EXPECT_FALSE(group.observes_sgpr_reads());
  EXPECT_FALSE(group.observes_scalar_register_writes());
}

// With helpers configured the WMMA instructions are actually offloaded, so they
// arrive as issue notifications instead of before/after pairs. They still
// executed, so they still have to appear in the mix -- opting in to async
// without handling the issue hook would have dropped them from the counts
// instead.
TEST(InstructionMixPluginTest, CountsActuallyOffloadedInstructions) {
  PluginFixture f(1, "cdna5", 32, 128, 256, 1, /*async_helpers=*/4);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(
      f.plugin_group_->add(std::make_unique<plugins::instruction_mix::InstructionMixPlugin>()));
  auto observer = std::make_unique<AsyncEventPlugin>();
  auto *events = observer.get();
  ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();
  const auto code = independent_wmma_kernel();
  f.run_kernel(code.data(), code.size(), 32, 32);
  f.shutdown();

  // Offload really happened: without the capability opt-in this is zero.
  ASSERT_GT(events->issued, 0u);

  const auto records = parse_instruction_mix_jsonl(sink.str());
  ASSERT_EQ(records.size(), 2u);
  for (const auto &record : records) {
    // Same three wave instructions the synchronous path would have counted.
    EXPECT_EQ(record.wave_instructions, 3u);
    EXPECT_EQ(record.family_executions.at("matrix"), 2u);
    EXPECT_EQ(record.family_mnemonics.at("matrix"), 1u);
  }
}

TEST(ExecutionPluginTest, KernelLoggingSupportsActualAsyncMma) {
  PluginFixture f(1, "cdna5", 32, 128, 256, 1, /*async_helpers=*/4);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<KernelLoggingPlugin>()));
  auto observer = std::make_unique<AsyncEventPlugin>();
  auto *events = observer.get();
  ASSERT_TRUE(f.plugin_group_->add(std::move(observer)));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();
  const auto code = independent_wmma_kernel();
  f.run_kernel(code.data(), code.size(), 32, 32);
  f.shutdown();
  EXPECT_GT(events->issued, 0u);
  EXPECT_NE(sink.str().find("mfma detected"), std::string::npos);
}

TEST(ExecutionPluginTest, HotHookPolicyComesFromContainedPlugins) {
  ExecutionPluginGroup group(PluginSinkConfig{});
  EXPECT_FALSE(group.requires_serial_hot_hooks());

  ASSERT_TRUE(group.add(std::make_unique<ParallelSafePlugin>()));
  EXPECT_FALSE(group.requires_serial_hot_hooks());

  ASSERT_TRUE(group.add(std::make_unique<OrderingPlugin>()));
  EXPECT_FALSE(group.requires_serial_hot_hooks());

  ASSERT_TRUE(group.add(std::make_unique<SerialHotHookPlugin>()));
  EXPECT_TRUE(group.requires_serial_hot_hooks());
}

TEST(ExecutionPluginTest, SgprReadPolicyComesFromContainedPlugins) {
  ExecutionPluginGroup group(PluginSinkConfig{});
  EXPECT_FALSE(group.observes_sgpr_reads());

  ASSERT_TRUE(group.add(std::make_unique<NoSgprReadPlugin>()));
  EXPECT_FALSE(group.observes_sgpr_reads());

  ASSERT_TRUE(group.add(std::make_unique<ParallelSafePlugin>()));
  EXPECT_TRUE(group.observes_sgpr_reads());
}

TEST(ExecutionPluginTest, SgprReadOptOutSkipsComputeUnitCallback) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto plugin = std::make_unique<NoSgprReadPlugin>();
  auto *plugin_ptr = plugin.get();
  ASSERT_TRUE(group->add(std::move(plugin)));
  f.soc->set_plugin_group(group);

  auto *wf = f.cu()->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*sgprs=*/32, /*vgprs=*/32);
  ASSERT_NE(wf, nullptr);
  f.cu()->write_sgpr(wf->sgpr_alloc().base, 0x12345678u);
  EXPECT_EQ(f.cu()->read_sgpr(wf->sgpr_alloc().base), 0x12345678u);
  EXPECT_EQ(RegisterAccess(*wf).read_sgpr(wf->sgpr_alloc().base), 0x12345678u);
  static_cast<void>(RegisterAccess(*wf).read_ttmp(0));
  EXPECT_EQ(plugin_ptr->callbacks, 0u);
}

TEST(ExecutionPluginTest, AsyncIssueFanoutHonorsPerPluginInterest) {
  PluginFixture fixture(/*num_wf_slots=*/1);
  auto *wave = fixture.cu()->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*sgprs=*/32, /*vgprs=*/32);
  ASSERT_NE(wave, nullptr);
  Instruction instruction("v_wmma_test", nullptr);

  ExecutionPluginGroup normal(PluginSinkConfig{});
  auto first = std::make_unique<AsyncIssueInterestPlugin>("async_first", true);
  auto *first_ptr = first.get();
  auto skipped = std::make_unique<AsyncIssueInterestPlugin>("async_skipped", false);
  auto *skipped_ptr = skipped.get();
  auto last = std::make_unique<AsyncIssueInterestPlugin>("async_last", true);
  auto *last_ptr = last.get();
  ASSERT_TRUE(normal.add(std::move(first)));
  ASSERT_TRUE(normal.add(std::move(skipped)));
  ASSERT_TRUE(normal.add(std::move(last)));

  normal.onAmdgpuAsyncInstructionIssued(0x1000, instruction, *wave);
  EXPECT_EQ(first_ptr->callbacks, 1u);
  EXPECT_EQ(skipped_ptr->callbacks, 0u);
  EXPECT_EQ(last_ptr->callbacks, 1u);

  ExecutionPluginGroup exceptional(PluginSinkConfig{});
  auto throwing = std::make_unique<AsyncIssueInterestPlugin>("async_throwing", true, true);
  auto *throwing_ptr = throwing.get();
  auto skipped_after_throw =
      std::make_unique<AsyncIssueInterestPlugin>("async_skipped_after_throw", false);
  auto *skipped_after_throw_ptr = skipped_after_throw.get();
  auto continued = std::make_unique<AsyncIssueInterestPlugin>("async_continued", true);
  auto *continued_ptr = continued.get();
  ASSERT_TRUE(exceptional.add(std::move(throwing)));
  ASSERT_TRUE(exceptional.add(std::move(skipped_after_throw)));
  ASSERT_TRUE(exceptional.add(std::move(continued)));

  EXPECT_THROW(exceptional.onAmdgpuAsyncInstructionIssued(0x1000, instruction, *wave),
               std::runtime_error);
  EXPECT_EQ(throwing_ptr->callbacks, 1u);
  EXPECT_EQ(skipped_after_throw_ptr->callbacks, 0u);
  EXPECT_EQ(continued_ptr->callbacks, 1u);
}

TEST(ExecutionPluginTest, InfrequentHooksSerializeAtGroupBoundary) {
  ExecutionPluginGroup group(PluginSinkConfig{});
  auto probe = std::make_unique<ConcurrencyProbePlugin>(false);
  auto *probe_ptr = probe.get();
  ASSERT_TRUE(group.add(std::move(probe)));

  const auto result = run_staged_threads(probe_ptr->cold_probe(), std::chrono::milliseconds(20),
                                         [&]() { group.onAmdgpuWorkgroupCompleted(0, 0); });

  EXPECT_TRUE(result.first_entered);
  EXPECT_FALSE(result.overlap_observed);
  EXPECT_EQ(probe_ptr->cold_probe().max_active(), 1);
}

TEST(ExecutionPluginTest, MultiXcdBeginClaimCannotOutrunPublication) {
  ExecutionPluginGroup group(PluginSinkConfig{});
  auto recorder = std::make_unique<OrderingPlugin>();
  auto *recorder_ptr = recorder.get();
  ASSERT_TRUE(group.add(std::move(recorder)));

  constexpr uint32_t kDispatchId = 97;
  amdgpu::GridCompletion grid;
  OverlapProbe claim_probe;
  const auto publish_begin = [&]() {
    group.onAmdgpuDispatchExecutionBeginOnce(kDispatchId, [&]() {
      // Hold the winning XCD between entering the claim path and publishing
      // begin. A peer must stop at the group lock rather than run this claim or
      // its first workgroup callback in the gap.
      claim_probe.observe();
      return grid.claim_execution_begin();
    });
  };

  const auto result =
      run_staged_callbacks(claim_probe, std::chrono::milliseconds(20), publish_begin, [&]() {
        publish_begin();
        group.onAmdgpuWorkgroupDispatched(kDispatchId, /*wg_id=*/1,
                                          /*physical_vgpr_count=*/0,
                                          /*physical_sgpr_count=*/0,
                                          std::span<amdgpu::Wavefront *>{});
      });

  EXPECT_TRUE(result.first_entered);
  EXPECT_FALSE(result.overlap_observed);
  EXPECT_EQ(claim_probe.max_active(), 1);
  ASSERT_EQ(recorder_ptr->events.size(), 2u);
  EXPECT_EQ(recorder_ptr->events[0].kind, HookEvent::DISPATCH_EXECUTION_BEGIN);
  EXPECT_EQ(recorder_ptr->events[0].dispatch_id, kDispatchId);
  EXPECT_EQ(recorder_ptr->events[1].kind, HookEvent::WORKGROUP_DISPATCHED);
  EXPECT_EQ(recorder_ptr->events[1].dispatch_id, kDispatchId);
}

TEST(ExecutionPluginTest, HighFrequencyHooksRunConcurrentlyByDefault) {
  ExecutionPluginGroup group(PluginSinkConfig{});
  auto probe = std::make_unique<ConcurrencyProbePlugin>(false);
  auto *probe_ptr = probe.get();
  ASSERT_TRUE(group.add(std::move(probe)));
  ASSERT_FALSE(group.requires_serial_hot_hooks());

  const auto result = run_staged_threads(probe_ptr->hot_probe(), std::chrono::seconds(5),
                                         [&]() { group.onAmdgpuReadSgpr(nullptr, 0); });

  EXPECT_TRUE(result.first_entered);
  EXPECT_TRUE(result.overlap_observed);
  EXPECT_EQ(probe_ptr->hot_probe().max_active(), 2);
}

TEST(ExecutionPluginTest, HighFrequencyHooksHonorSerialOptIn) {
  ExecutionPluginGroup group(PluginSinkConfig{});
  auto probe = std::make_unique<ConcurrencyProbePlugin>(true);
  auto *probe_ptr = probe.get();
  ASSERT_TRUE(group.add(std::move(probe)));
  ASSERT_TRUE(group.requires_serial_hot_hooks());

  const auto result = run_staged_threads(probe_ptr->hot_probe(), std::chrono::milliseconds(20),
                                         [&]() { group.onAmdgpuReadSgpr(nullptr, 0); });

  EXPECT_TRUE(result.first_entered);
  EXPECT_FALSE(result.overlap_observed);
  EXPECT_EQ(probe_ptr->hot_probe().max_active(), 1);
}

TEST(ExecutionPluginTest, UnsubscribedWaveBypassesHotHookAndCallbackLock) {
  ExecutionPluginGroup group(PluginSinkConfig{});
  auto plugin = std::make_unique<UnsubscribedHotHookPlugin>();
  auto *plugin_ptr = plugin.get();
  ASSERT_TRUE(group.add(std::move(plugin)));
  ASSERT_TRUE(group.requires_serial_hot_hooks());

  const uint64_t before = test::ExecutionPluginGroupTestAccess::callback_lock_acquisitions(group);
  group.onAmdgpuReadSgpr(nullptr, 0);

  EXPECT_EQ(plugin_ptr->hot_callbacks, 0u);
  EXPECT_EQ(test::ExecutionPluginGroupTestAccess::callback_lock_acquisitions(group), before);

  group.onAmdgpuWorkgroupCompleted(1, 2);
  EXPECT_EQ(plugin_ptr->cold_callbacks, 1u);
  EXPECT_EQ(test::ExecutionPluginGroupTestAccess::callback_lock_acquisitions(group), before + 1);
}

TEST(ExecutionPluginTest, WaveDispatchCachesEachPluginsHotHookSubscription) {
  PluginFixture fixture(/*num_wf_slots=*/1);
  ExecutionPluginGroup group(PluginSinkConfig{});
  auto subscribed = std::make_unique<HotHookSubscriptionProbe>("subscribed_probe", true);
  auto *subscribed_ptr = subscribed.get();
  auto unsubscribed = std::make_unique<HotHookSubscriptionProbe>("unsubscribed_probe", false);
  auto *unsubscribed_ptr = unsubscribed.get();
  ASSERT_TRUE(group.add(std::move(subscribed)));
  ASSERT_TRUE(group.add(std::move(unsubscribed)));

  auto *wave = fixture.cu()->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(wave, nullptr);
  group.onAmdgpuWavefrontDispatched(*wave);

  EXPECT_EQ(subscribed_ptr->predicate_queries, 1u);
  EXPECT_EQ(unsubscribed_ptr->predicate_queries, 1u);
  for (uint32_t i = 0; i != 8; ++i)
    group.onAmdgpuReadSgpr(wave, i);

  EXPECT_EQ(subscribed_ptr->hot_callbacks, 8u);
  EXPECT_EQ(unsubscribed_ptr->hot_callbacks, 0u);
  EXPECT_EQ(subscribed_ptr->predicate_queries, 1u);
  EXPECT_EQ(unsubscribed_ptr->predicate_queries, 1u);
}

TEST(ExecutionPluginTest, CachedUnsubscribedWaveBypassesCallbackLock) {
  PluginFixture fixture(/*num_wf_slots=*/1);
  ExecutionPluginGroup group(PluginSinkConfig{});
  auto plugin =
      std::make_unique<HotHookSubscriptionProbe>("serial_unsubscribed_probe", false, true);
  auto *plugin_ptr = plugin.get();
  ASSERT_TRUE(group.add(std::move(plugin)));

  auto *wave = fixture.cu()->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(wave, nullptr);
  group.onAmdgpuWavefrontDispatched(*wave);
  const uint64_t before = test::ExecutionPluginGroupTestAccess::callback_lock_acquisitions(group);

  group.onAmdgpuReadSgpr(wave, 0);

  EXPECT_EQ(plugin_ptr->predicate_queries, 1u);
  EXPECT_EQ(plugin_ptr->hot_callbacks, 0u);
  EXPECT_EQ(test::ExecutionPluginGroupTestAccess::callback_lock_acquisitions(group), before);
}

TEST(ExecutionPluginTest, ReentrantDispatchHotHookFallsBackUntilCachePublication) {
  PluginFixture fixture(/*num_wf_slots=*/1);
  ExecutionPluginGroup group(PluginSinkConfig{});
  auto plugin = std::make_unique<HotHookSubscriptionProbe>("reentrant_dispatch_probe", true, true);
  auto *plugin_ptr = plugin.get();
  plugin_ptr->group = &group;
  plugin_ptr->reenter_on_dispatch = true;
  ASSERT_TRUE(group.add(std::move(plugin)));

  auto *wave = fixture.cu()->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(wave, nullptr);
  group.onAmdgpuWavefrontDispatched(*wave);

  // The reentrant hook uses the original live-query path (aggregate + fanout),
  // then dispatch publishes one cached decision for all subsequent hooks.
  EXPECT_EQ(plugin_ptr->predicate_queries, 3u);
  EXPECT_EQ(plugin_ptr->hot_callbacks, 1u);
  group.onAmdgpuReadSgpr(wave, 1);
  EXPECT_EQ(plugin_ptr->predicate_queries, 3u);
  EXPECT_EQ(plugin_ptr->hot_callbacks, 2u);
}

TEST(ExecutionPluginTest, HaltAndRedispatchCannotReuseAStaleHotHookSubscription) {
  PluginFixture fixture(/*num_wf_slots=*/1);
  ExecutionPluginGroup group(PluginSinkConfig{});
  auto plugin = std::make_unique<HotHookSubscriptionProbe>("halt_reuse_probe", true, true);
  auto *plugin_ptr = plugin.get();
  plugin_ptr->group = &group;
  plugin_ptr->unsubscribe_on_halt = true;
  plugin_ptr->reenter_on_halt = true;
  ASSERT_TRUE(group.add(std::move(plugin)));

  auto *wave = fixture.cu()->dispatch_wf(/*wg_id=*/1, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(wave, nullptr);
  group.onAmdgpuWavefrontDispatched(*wave);
  group.onAmdgpuReadSgpr(wave, 0);
  ASSERT_EQ(plugin_ptr->hot_callbacks, 1u);
  ASSERT_EQ(plugin_ptr->predicate_queries, 1u);

  group.onAmdgpuWavefrontHalted(*wave);
  EXPECT_EQ(plugin_ptr->halt_callbacks, 1u);
  EXPECT_EQ(plugin_ptr->hot_callbacks, 1u)
      << "the halt callback observed the stale subscribed decision";
  EXPECT_EQ(plugin_ptr->predicate_queries, 2u);

  plugin_ptr->set_subscribe_on_dispatch(false);
  plugin_ptr->reenter_on_dispatch = false;
  group.onAmdgpuWavefrontDispatched(*wave);
  EXPECT_EQ(plugin_ptr->predicate_queries, 3u);
  group.onAmdgpuReadSgpr(wave, 1);
  EXPECT_EQ(plugin_ptr->predicate_queries, 3u);
  EXPECT_EQ(plugin_ptr->hot_callbacks, 1u);

  // reset() independently invalidates the decision while retaining its vector
  // allocation for the next incarnation.
  plugin_ptr->subscribed = true;
  wave->reset();
  group.onAmdgpuReadSgpr(wave, 2);
  EXPECT_EQ(plugin_ptr->predicate_queries, 5u);
  EXPECT_EQ(plugin_ptr->hot_callbacks, 2u);
}

TEST(ExecutionPluginTest, InfrequentAndHighFrequencyHooksMayOverlapByDefault) {
  ExecutionPluginGroup group(PluginSinkConfig{});
  auto probe = std::make_unique<CrossHookConcurrencyProbePlugin>(false);
  auto *probe_ptr = probe.get();
  ASSERT_TRUE(group.add(std::move(probe)));
  ASSERT_FALSE(group.requires_serial_hot_hooks());

  const auto result = run_staged_callbacks(
      probe_ptr->probe(), std::chrono::seconds(5),
      [&]() { group.onAmdgpuWorkgroupCompleted(0, 0); },
      [&]() { group.onAmdgpuReadSgpr(nullptr, 0); });

  EXPECT_TRUE(result.first_entered);
  EXPECT_TRUE(result.overlap_observed);
  EXPECT_EQ(probe_ptr->probe().max_active(), 2);
}

TEST(ExecutionPluginTest, SerialHotHookOptInPreventsInfrequentAndHighFrequencyOverlap) {
  ExecutionPluginGroup group(PluginSinkConfig{});
  auto probe = std::make_unique<CrossHookConcurrencyProbePlugin>(true);
  auto *probe_ptr = probe.get();
  ASSERT_TRUE(group.add(std::move(probe)));
  ASSERT_TRUE(group.requires_serial_hot_hooks());

  const auto result = run_staged_callbacks(
      probe_ptr->probe(), std::chrono::milliseconds(20),
      [&]() { group.onAmdgpuWorkgroupCompleted(0, 0); },
      [&]() { group.onAmdgpuReadSgpr(nullptr, 0); });

  EXPECT_TRUE(result.first_entered);
  EXPECT_FALSE(result.overlap_observed);
  EXPECT_EQ(probe_ptr->probe().max_active(), 1);
}

TEST(ExecutionPluginTest, EmptyGroupDispatchBypassesCallbackLock) {
  auto group = ExecutionPluginGroup::empty_group();
  ASSERT_TRUE(group->empty());
  const uint64_t before = test::ExecutionPluginGroupTestAccess::callback_lock_acquisitions(*group);

  run_two_threads([&]() {
    for (int i = 0; i < 10000; ++i) {
      group->onInit();
      group->onAmdgpuReadSgpr(nullptr, 0);
      group->onAmdgpuWorkgroupCompleted(0, 0);
    }
  });

  EXPECT_EQ(test::ExecutionPluginGroupTestAccess::callback_lock_acquisitions(*group), before);

  ExecutionPluginGroup non_empty_group(PluginSinkConfig{});
  ASSERT_TRUE(non_empty_group.add(std::make_unique<ExecutionPlugin>("no-op")));
  const uint64_t non_empty_before =
      test::ExecutionPluginGroupTestAccess::callback_lock_acquisitions(non_empty_group);
  non_empty_group.onInit();
  EXPECT_EQ(test::ExecutionPluginGroupTestAccess::callback_lock_acquisitions(non_empty_group),
            non_empty_before + 1);
}

int run_serial_hot_hook_halt_snapshot() {
  PluginFixture f;
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  if (!f.plugin_group_->add(std::make_unique<SerialHotHookPlugin>()))
    return 1;

  auto snapshot = std::make_unique<test::HaltSnapshotPlugin>();
  auto *snapshot_ptr = snapshot.get();
  if (!f.plugin_group_->add(std::move(snapshot)))
    return 2;
  f.soc->set_plugin_group(f.plugin_group_);

  const uint32_t code[] = {S_ENDPGM};
  f.run_kernel(code, 1);

  if (snapshot_ptr->snapshots().size() != 1u)
    return 3;
  if (snapshot_ptr->snapshots().front().sgprs.empty())
    return 4;
  if (snapshot_ptr->snapshots().front().vgprs.empty())
    return 5;
  return 0;
}

#if GTEST_HAS_DEATH_TEST && defined(__linux__)
TEST(ExecutionPluginDeathTest, SerialHotHooksAllowRegisterReadsFromHaltHook) {
  ASSERT_EXIT(
      {
        alarm(5);
        _exit(run_serial_hot_hook_halt_snapshot());
      },
      ::testing::ExitedWithCode(0), "");
}
#else
TEST(ExecutionPluginTest, SerialHotHooksAllowRegisterReadsFromHaltHook) {
  EXPECT_EQ(run_serial_hot_hook_halt_snapshot(), 0);
}
#endif

TEST(ExecutionPluginTest, DispatchPacketNameResolvesForVmidMappedCodeObject) {
  PluginFixture f;
  auto *plugin = f.attach_ordering_plugin();

  constexpr uint32_t process_id = 123;
  constexpr uint64_t code_object_va = 0x5400200000;
  constexpr uint64_t kernel_descriptor_offset = 0x800;
  auto image = make_loaded_kernel_symbol_elf(kernel_descriptor_offset, "vmid_dispatch_kernel.kd");
  std::vector<uint8_t> image_backing(image.size() + amdgpu::GpuMemory::PAGE_SIZE, 0);
  auto image_host = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<uintptr_t>(image_backing.data()) + amdgpu::GpuMemory::PAGE_MASK) &
      ~static_cast<uintptr_t>(amdgpu::GpuMemory::PAGE_MASK));
  std::memcpy(image_host, image.data(), image.size());

  KfdProcess process(process_id);
  amdgpu::LegacyGpuVmAdapter legacy_vm(f.soc->gpu_vm(), f.soc->memory());
  const amdgpu::AddressSpaceHandle address_space =
      legacy_vm.register_address_space(process_id, &process.page_table_, &process.page_table_mutex_,
                                       process.page_table_generation());
  ASSERT_TRUE(address_space);
  process.map_pages(code_object_va, image_host, image.size());

  std::vector<uint8_t> ring(4096, 0);
  std::array<uint8_t, 4096> queue_state{};
  *reinterpret_cast<uint64_t *>(queue_state.data()) = 0;
  *reinterpret_cast<uint64_t *>(queue_state.data() + 8) = 1;
  uint64_t doorbell = 0;
  constexpr uint64_t ring_va = 0x6100000000;
  constexpr uint64_t read_ptr_va = 0x6100010000;
  constexpr uint64_t write_ptr_va = 0x6100010008;
  process.map_pages(ring_va, ring.data(), ring.size());
  process.map_pages(read_ptr_va, queue_state.data(), queue_state.size());

  hsa_kernel_dispatch_packet_t packet{};
  packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  packet.setup = 1;
  packet.workgroup_size_x = 64;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;
  packet.grid_size_x = 64;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;
  packet.kernel_object = code_object_va + kernel_descriptor_offset;
  std::memcpy(ring.data(), &packet, sizeof(packet));

  constexpr uint32_t queue_id = 7;
  amdgpu::ComputeQueueConfig queue{};
  queue.address_space = address_space;
  queue.queue_id = queue_id;
  queue.process_id = process_id;
  queue.ring_base_va = ring_va;
  queue.ring_size = static_cast<uint32_t>(ring.size());
  queue.read_ptr_va = read_ptr_va;
  queue.write_ptr_va = write_ptr_va;
  queue.doorbell_base = &doorbell;
  queue.doorbell_mode = amdgpu::QueueDoorbellMode::HostPolled;
  queue.uses_kfd_queue_abi = true;
  f.cp()->register_queue(std::move(queue));
  f.cp()->engine()->schedule_event_now(f.cp()->doorbell_event());
  f.run_until_idle();

  auto it = std::ranges::find_if(plugin->events, [](const HookEvent &event) {
    return event.kind == HookEvent::DISPATCH_PACKET_PROCESSED;
  });
  bool found_dispatch = it != plugin->events.end();
  std::string kernel_name = found_dispatch ? it->kernel_name : "";
  std::string kernel_symbol = found_dispatch ? it->kernel_symbol : "";

  f.cp()->unregister_queue(queue_id, process_id);
  f.shutdown();
  EXPECT_TRUE(legacy_vm.unregister_address_space(address_space));

  ASSERT_TRUE(found_dispatch);
  EXPECT_EQ(kernel_name, "vmid_dispatch_kernel");
  EXPECT_EQ(kernel_symbol, "vmid_dispatch_kernel");
}

TEST(ExecutionPluginTest, HotHookSerializationDoesNotSetCpuDispatchPolicy) {
  {
    PluginFixture f(/*num_wf_slots=*/10, /*arch=*/"cdna4", /*wavefront_size=*/64,
                    /*sgprs_per_wf=*/104, /*vgprs_per_wf=*/256, /*num_cus=*/8);
    f.soc->set_dispatch_threads(8);
    auto pg = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    pg->add(std::make_unique<SerialHotHookPlugin>());
    f.soc->set_plugin_group(pg);
    EXPECT_EQ(f.cp()->dispatch_threads(), 8u);
    f.cp()->set_dispatch_threads(8);
    EXPECT_EQ(f.cp()->dispatch_threads(), 8u);
  }
  {
    PluginFixture f;
    auto pg = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    pg->add(std::make_unique<ParallelSafePlugin>());
    f.soc->set_plugin_group(pg);
    f.cp()->set_dispatch_threads(8);
    EXPECT_EQ(f.cp()->dispatch_threads(), 8u);
  }
}

TEST(HookOrderingTest, BarrierTwoWaves) {
  PluginFixture f;
  auto *p = f.attach_ordering_plugin();
  const uint32_t code[] = {S_BARRIER, S_ENDPGM};
  f.run_kernel(code, 2, /*grid=*/128, /*workgroup=*/128);
  f.shutdown();

  EventLog log(p->events);
  EXPECT_EQ(log.count(HookEvent::INIT), 1u);
  EXPECT_EQ(log.count(HookEvent::SHUTDOWN), 1u);
  EXPECT_EQ(log.count(HookEvent::BARRIER_RESOLVED), 1u);
  EXPECT_EQ(log.count(HookEvent::WAVEFRONT_DISPATCHED), 2u);
  EXPECT_EQ(log.count(HookEvent::WAVEFRONT_HALTED), 2u);

  ASSERT_EQ(p->events.front().kind, HookEvent::INIT);
  ASSERT_EQ(p->events.back().kind, HookEvent::SHUTDOWN);
}

TEST(HookOrderingTest, WorkgroupDispatchedReportsPhysicalRegisterBlockSizes) {
  PluginFixture f;
  auto *p = f.attach_ordering_plugin();
  const uint32_t code[] = {S_ENDPGM};
  constexpr uint32_t kGranulatedSgprCount = 3; // Requests 32 SGPRs on CDNA4.
  f.run_kernel(code, 1, /*grid=*/64, /*workgroup=*/64, kGranulatedSgprCount);
  f.shutdown();

  auto it = std::ranges::find_if(
      p->events, [](const HookEvent &e) { return e.kind == HookEvent::WORKGROUP_DISPATCHED; });
  ASSERT_NE(it, p->events.end());
  EXPECT_EQ(it->physical_vgpr_count, f.cu()->vgpr_allocation_block_size());
  EXPECT_GT(it->physical_vgpr_count, f.cu()->config().vgprs_per_wf);
  EXPECT_EQ(it->physical_sgpr_count, f.cu()->sgpr_allocation_block_size());
  EXPECT_GT(it->physical_sgpr_count, 32u);
}

TEST(HookOrderingTest, BeforeInstructionExposesMemoryIssueBeforeOperandReadsAndRouting) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *p = f.attach_ordering_plugin();
  const auto load =
      cdna4::build_smem(cdna4::kSLoadDwordSmem, {.sbase = 0, .sdata = 4, .imm = 1, .offset = 0});
  const std::array<uint32_t, 3> code = {load[0], load[1], S_ENDPGM};
  f.run_kernel(code.data(), code.size());
  f.shutdown();

  const auto before_instruction = std::ranges::find_if(p->events, [](const HookEvent &e) {
    return e.kind == HookEvent::BEFORE_INSTRUCTION && e.mnemonic == "s_load_dword";
  });
  ASSERT_NE(before_instruction, p->events.end());
  ASSERT_EQ(before_instruction->counter_obligations.size(), 1u);
  EXPECT_EQ(before_instruction->counter_obligations[0].wait_counter_type(),
            WaitCounterType::LGKMCNT);
  EXPECT_EQ(before_instruction->counter_obligations[0].completion_class(),
            MemoryCompletionClass::UNORDERED);

  const auto first_operand_read =
      std::ranges::find_if(std::next(before_instruction), p->events.end(),
                           [](const HookEvent &e) { return e.kind == HookEvent::READ_SGPR; });
  const auto route =
      std::ranges::find_if(std::next(before_instruction), p->events.end(), [](const HookEvent &e) {
        return e.kind == HookEvent::ROUTE_MEMORY && e.mnemonic == "s_load_dword";
      });
  ASSERT_NE(first_operand_read, p->events.end());
  ASSERT_NE(route, p->events.end());
  EXPECT_LT(before_instruction, first_operand_read);
  EXPECT_LT(first_operand_read, route);
}

TEST(HookOrderingTest, UniformLdsFlatPreservesPreAndPostRoutingViews) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_memory_observation_plugin();
  auto *cu = f.cu();

  constexpr uint64_t kSharedBase = 0x0001'0000'0000'0000ULL;
  cu->set_apertures(kSharedBase, kSharedBase + 0xFFFF, /*private_base=*/0,
                    /*private_limit=*/0);
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  cu->write_vgpr(wf->vgpr_alloc().base, /*lane=*/0, static_cast<uint32_t>(kSharedBase));
  cu->write_vgpr(wf->vgpr_alloc().base + 1, /*lane=*/0, static_cast<uint32_t>(kSharedBase >> 32));

  const auto words =
      cdna4::build_flat(cdna4::kFlatLoadDwordFlat, {.seg = 0, .addr = 0, .saddr = 0x7F, .vdst = 2});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> flat(decode_valid(*decoder, words.data()));
  ASSERT_NE(flat, nullptr);

  EXPECT_TRUE(cu->execute_instruction(flat.get(), *wf).succeeded());
  test::ComputeUnitTestAccess::route_memory_inst(*cu, flat.release(), *wf);

  ASSERT_EQ(plugin->before_tag.size(), 1u);
  EXPECT_EQ(plugin->before_tag[0], GLOBAL_MEM);
  ASSERT_EQ(plugin->accesses.size(), 1u);
  EXPECT_EQ(plugin->accesses[0].route, MemoryRoute::LOCAL);
  EXPECT_TRUE(plugin->accesses[0].normalized_to_local);
}

TEST(HookOrderingTest, TerminatorEmitsBeforeButNotAfterInstruction) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *p = f.attach_ordering_plugin();
  // Two non-terminators then the terminator, so the sequence exercises matched
  // BEFORE/AFTER pairs and the terminator's asymmetry in one run.
  const uint32_t code[] = {S_NOP, S_NOP, S_ENDPGM};
  f.run_kernel(code, 3);
  f.shutdown();

  // Collect the BEFORE/AFTER instruction hooks in order.
  size_t before_endpgm = 0, after_endpgm = 0;
  size_t before_nop = 0, after_nop = 0;
  for (const auto &e : p->events) {
    if (e.kind == HookEvent::BEFORE_INSTRUCTION) {
      if (e.mnemonic == "s_endpgm")
        ++before_endpgm;
      else if (e.mnemonic == "s_nop")
        ++before_nop;
    } else if (e.kind == HookEvent::AFTER_INSTRUCTION) {
      if (e.mnemonic == "s_endpgm")
        ++after_endpgm;
      else if (e.mnemonic == "s_nop")
        ++after_nop;
    }
  }

  // The terminator is observed before execution but frees the wave on execution,
  // so it must not emit an AFTER hook.
  EXPECT_EQ(before_endpgm, 1u) << "s_endpgm must fire BEFORE_INSTRUCTION";
  EXPECT_EQ(after_endpgm, 0u) << "s_endpgm must NOT fire AFTER_INSTRUCTION (slot freed at halt)";

  // Non-terminators keep matched BEFORE/AFTER pairs.
  EXPECT_EQ(before_nop, 2u);
  EXPECT_EQ(after_nop, 2u);

  // Exactly one wave, and it halted.
  EventLog log(p->events);
  EXPECT_EQ(log.count(HookEvent::WAVEFRONT_HALTED), 1u);
}

TEST(HookOrderingTest, FiveDispatchLifecycle) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *p = f.attach_ordering_plugin();

  // 3 distinct kernels.
  const uint32_t kernel_a[] = {S_NOP, S_ENDPGM};
  const uint32_t kernel_b[] = {S_NOP, S_NOP, S_ENDPGM};
  const uint32_t kernel_c[] = {S_NOP, S_NOP, S_NOP, S_ENDPGM};
  uint64_t ko_a = f.write_kernel(0x1000, kernel_a, 2);
  uint64_t ko_b = f.write_kernel(0x2000, kernel_b, 3);
  uint64_t ko_c = f.write_kernel(0x3000, kernel_c, 4);

  // 5 dispatches with varying workgroup counts (1 wave per WG, wave_size=64).
  struct DispatchSpec {
    uint64_t kernel;
    uint32_t grid;
    uint32_t wg_size;
    uint32_t expected_wgs;
  };
  DispatchSpec specs[] = {
      {ko_a, 192, 64, 3}, // dispatch 0: kernel A, 3 WGs
      {ko_b, 128, 64, 2}, // dispatch 1: kernel B, 2 WGs
      {ko_a, 256, 64, 4}, // dispatch 2: kernel A, 4 WGs
      {ko_c, 64, 64, 1},  // dispatch 3: kernel C, 1 WG
      {ko_b, 320, 64, 5}, // dispatch 4: kernel B, 5 WGs
  };
  constexpr size_t N = std::size(specs);
  constexpr uint32_t total_wgs = 3 + 2 + 4 + 1 + 5;

  test::AqlQueue queue(f.mem, f.cp());
  for (const auto &s : specs)
    queue.dispatch(s.kernel, s.grid, static_cast<uint16_t>(s.wg_size));
  f.run_until_idle();
  f.shutdown();

  EventLog log(p->events);
  log.dump();

  // -- Init/Shutdown lifecycle ------------------------------------------------

  EXPECT_EQ(log.count(HookEvent::INIT), 1u);
  EXPECT_EQ(log.count(HookEvent::SHUTDOWN), 1u);
  ASSERT_EQ(p->events.front().kind, HookEvent::INIT);
  ASSERT_EQ(p->events.back().kind, HookEvent::SHUTDOWN);

  // -- Dispatch ID integrity --------------------------------------------------

  auto dispatches = log.dispatchIds();
  ASSERT_EQ(dispatches.size(), N);

  // All dispatch_ids must be distinct.
  std::set<uint32_t> unique_ids(dispatches.begin(), dispatches.end());
  EXPECT_EQ(unique_ids.size(), N) << "All dispatch_ids must be distinct";

  // Every lifecycle event must carry a known dispatch_id.
  auto all_ids = log.allDispatchIds();
  EXPECT_EQ(all_ids, unique_ids) << "No lifecycle event should carry an unexpected dispatch_id";

  // -- Counts -----------------------------------------------------------------

  EXPECT_EQ(log.count(HookEvent::DISPATCH_PACKET_PROCESSED), N);
  EXPECT_EQ(log.count(HookEvent::DISPATCH_EXECUTION_BEGIN), N);
  EXPECT_EQ(log.count(HookEvent::DISPATCH_EXECUTION_END), N);
  EXPECT_EQ(log.count(HookEvent::WORKGROUP_DISPATCHED), total_wgs);
  EXPECT_EQ(log.count(HookEvent::WORKGROUP_COMPLETED), total_wgs);
  EXPECT_EQ(log.count(HookEvent::WAVEFRONT_DISPATCHED), log.count(HookEvent::WAVEFRONT_HALTED));

  for (size_t i = 0; i < N; ++i) {
    uint32_t d = dispatches[i];
    EXPECT_EQ(log.count(HookEvent::WORKGROUP_DISPATCHED, d), specs[i].expected_wgs)
        << "Workgroup count mismatch for dispatch index " << i;
  }

  // -- DAG edges --------------------------------------------------------------
  std::cerr << "--- DAG edge assertions ---\n";

  log.assertAllBefore(HookEvent::DISPATCH_PACKET_PROCESSED, HookEvent::DISPATCH_EXECUTION_BEGIN);

  // -- Per-dispatch lifecycle brackets ----------------------------------------

  for (uint32_t d : dispatches) {
    // Exactly one execution-begin and one execution-end per dispatch.
    EXPECT_EQ(log.count(HookEvent::DISPATCH_EXECUTION_BEGIN, d), 1u);
    EXPECT_EQ(log.count(HookEvent::DISPATCH_EXECUTION_END, d), 1u);
    EXPECT_EQ(log.count(HookEvent::DISPATCH_PACKET_PROCESSED, d), 1u);

    // Execution-begin precedes first workgroup dispatch.
    log.assertLastBeforeFirst(HookEvent::DISPATCH_EXECUTION_BEGIN, d,
                              HookEvent::WORKGROUP_DISPATCHED, d);
    // All wavefronts halt before execution-end.
    log.assertLastBeforeFirst(HookEvent::WAVEFRONT_HALTED, d, HookEvent::DISPATCH_EXECUTION_END, d);
    // Wavefront dispatched/halted are properly paired.
    log.assertPaired(HookEvent::WAVEFRONT_DISPATCHED, HookEvent::WAVEFRONT_HALTED, d);
    // Workgroup dispatched/completed: counts match and properly paired.
    EXPECT_EQ(log.count(HookEvent::WORKGROUP_DISPATCHED, d),
              log.count(HookEvent::WORKGROUP_COMPLETED, d));
    log.assertPairedByWg(HookEvent::WORKGROUP_DISPATCHED, HookEvent::WORKGROUP_COMPLETED, d);
    // Wavefront dispatched/halted: counts match and properly paired.
    EXPECT_EQ(log.count(HookEvent::WAVEFRONT_DISPATCHED, d),
              log.count(HookEvent::WAVEFRONT_HALTED, d));
  }

  // -- Sequential execution (functional mode, quantum=0) ----------------------
  // In functional mode, the CP drains each dispatch to completion before
  // starting the next on the same queue. This would not hold with quantum > 0
  // or with dispatches on separate queues.

  for (size_t i = 0; i + 1 < N; ++i) {
    log.assertLastBeforeFirst(HookEvent::DISPATCH_EXECUTION_END, dispatches[i],
                              HookEvent::DISPATCH_EXECUTION_BEGIN, dispatches[i + 1]);
  }
}

TEST(HookOrderingTest, NonKernelQueueEntriesDoNotEmitDispatchLifecycleHooks) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *p = f.attach_ordering_plugin();

  const uint32_t code[] = {S_ENDPGM};
  const uint64_t kernel_object = f.write_kernel(0x1000, code, std::size(code));

  test::AqlQueue queue(f.mem, f.cp());
  queue.barrier_and();
  queue.pm4_ib();
  queue.dispatch(kernel_object, /*grid_size_x=*/64);
  f.run_until_idle();
  f.shutdown();

  EventLog log(p->events);
  const auto dispatches = log.dispatchIds();
  ASSERT_EQ(dispatches.size(), 1u);
  const uint32_t kernel_dispatch_id = dispatches.front();

  EXPECT_EQ(log.count(HookEvent::DISPATCH_EXECUTION_BEGIN), 1u);
  EXPECT_EQ(log.count(HookEvent::DISPATCH_EXECUTION_END), 1u);
  EXPECT_EQ(log.count(HookEvent::DISPATCH_EXECUTION_BEGIN, kernel_dispatch_id), 1u);
  EXPECT_EQ(log.count(HookEvent::DISPATCH_EXECUTION_END, kernel_dispatch_id), 1u);

  const std::set<uint32_t> expected_ids{kernel_dispatch_id};
  EXPECT_EQ(log.allDispatchIds(), expected_ids);
}

TEST(HookOrderingTest, ParallelWorkgroupLifecycleRunsOnCommandProcessorAfterWorkersRejoin) {
  PluginFixture f(/*num_wf_slots=*/2, "cdna4", /*wavefront_size=*/64,
                  /*sgprs_per_wf=*/104, /*vgprs_per_wf=*/256, /*num_cus=*/2);
  auto group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto plugin = std::make_unique<ParallelColdHookRecorder>();
  auto *recorder = plugin.get();
  ASSERT_TRUE(group->add(std::move(plugin)));
  f.soc->set_plugin_group(group);
  f.cp()->set_dispatch_threads(2);
  ASSERT_EQ(f.cp()->dispatch_threads(), 2u);

  const uint32_t kernel_a[] = {S_NOP, S_ENDPGM};
  const uint32_t kernel_b[] = {S_NOP, S_NOP, S_ENDPGM};
  const uint64_t ko_a = f.write_kernel(0x1000, kernel_a, std::size(kernel_a));
  const uint64_t ko_b = f.write_kernel(0x2000, kernel_b, std::size(kernel_b));
  test::AqlQueue queue(f.mem, f.cp());
  queue.dispatch(ko_a, /*grid_size=*/128, /*workgroup_size=*/64);
  queue.dispatch(ko_b, /*grid_size=*/128, /*workgroup_size=*/64);

  const auto cp_thread = std::this_thread::get_id();
  f.run_until_idle();

  const auto events = recorder->events();
  const auto hot_hook_threads = recorder->hot_hook_threads();
  ASSERT_EQ(hot_hook_threads.size(), 2u);
  EXPECT_TRUE(hot_hook_threads.contains(cp_thread));

  EventLog log(events);
  std::set<uint32_t> dispatch_ids;
  for (const auto &event : events) {
    if (event.kind == HookEvent::WORKGROUP_DISPATCHED)
      dispatch_ids.insert(event.dispatch_id);
    EXPECT_EQ(event.callback_thread, cp_thread);
  }

  ASSERT_EQ(dispatch_ids.size(), 2u);
  EXPECT_EQ(log.count(HookEvent::WORKGROUP_DISPATCHED), 4u);
  EXPECT_EQ(log.count(HookEvent::WORKGROUP_COMPLETED), 4u);
  for (uint32_t dispatch_id : dispatch_ids) {
    EXPECT_EQ(log.count(HookEvent::WORKGROUP_DISPATCHED, dispatch_id), 2u);
    EXPECT_EQ(log.count(HookEvent::WORKGROUP_COMPLETED, dispatch_id), 2u);
    log.assertPairedByWg(HookEvent::WORKGROUP_DISPATCHED, HookEvent::WORKGROUP_COMPLETED,
                         dispatch_id);
  }
}

TEST(ExecutionPluginGroupTest, OwnsConfiguredSinkForRetainedGroupLifetime) {
  std::vector<std::string> events;
  std::shared_ptr<ExecutionPluginGroup> plugin_group;
  {
    PluginSinkConfig sink_config;
    sink_config.emplace<DestructionTrackingSink>(events);
    plugin_group = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    ASSERT_TRUE(plugin_group->add(std::make_unique<DestructorWritingPlugin>(events)));
  }

  EXPECT_TRUE(events.empty());
  plugin_group.reset();
  EXPECT_EQ(events, (std::vector<std::string>{"write:destroyed\n", "plugin", "sink"}));
}

TEST(ExecutionPluginGroupTest, FansOutToEveryConfiguredSink) {
  std::vector<std::string> events;
  {
    PluginSinkConfig sink_config;
    sink_config.emplace<DestructionTrackingSink>(events);
    sink_config.emplace<DestructionTrackingSink>(events);
    ExecutionPluginGroup plugin_group(std::move(sink_config));
    ASSERT_TRUE(plugin_group.add(std::make_unique<DestructorWritingPlugin>(events)));
  }

  ASSERT_EQ(events.size(), 5u);
  EXPECT_EQ(events[0], "write:destroyed\n");
  EXPECT_EQ(events[1], "write:destroyed\n");
  EXPECT_EQ(events[2], "plugin");
  EXPECT_EQ(std::ranges::count(events.begin() + 3, events.end(), "sink"), 2);
}

TEST(ExecutionPluginGroupTest, OwnsFileSinkThroughPluginDestruction) {
  test::ScopedTempDirectory sink_directory("rocjitsu-plugin-sink-lifetime-");
  const std::string log_path = sink_directory.path() + "/destructor_writer.log";
  std::vector<std::string> events;
  {
    PluginSinkConfig sink_config;
    sink_config.set_file_directory(sink_directory.path());
    ExecutionPluginGroup plugin_group(std::move(sink_config));
    ASSERT_TRUE(plugin_group.add(std::make_unique<DestructorWritingPlugin>(events)));
  }

  EXPECT_EQ(events, (std::vector<std::string>{"plugin"}));
  std::ifstream log(log_path);
  ASSERT_TRUE(log);
  const std::string contents{std::istreambuf_iterator<char>(log), std::istreambuf_iterator<char>()};
  EXPECT_EQ(contents, "destroyed\n");
}

} // namespace
