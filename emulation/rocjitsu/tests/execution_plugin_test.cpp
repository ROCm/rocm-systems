// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_test.cpp
/// @brief Tests for the ExecutionPlugin infrastructure.

#include "aql_queue.h"

#include "embedded_schema.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/soc.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

// SOPP encoding: bits[31:23]=0x17F, bits[22:16]=op, bits[15:0]=simm16.
constexpr uint32_t sopp(uint32_t op, uint16_t simm16 = 0) {
  return 0xBF800000u | (op << 16) | simm16;
}
constexpr uint32_t S_NOP = sopp(0);
constexpr uint32_t S_ENDPGM = sopp(1);
constexpr uint32_t S_BARRIER = sopp(10);

struct HookEvent {
  enum Kind {
    DISPATCH_PACKET_PROCESSED,
    DISPATCH_EXECUTION_BEGIN,
    DISPATCH_EXECUTION_END,
    WORKGROUP_DISPATCHED,
    WORKGROUP_COMPLETED,
    WAVEFRONT_DISPATCHED,
    WAVEFRONT_HALTED,
    BEFORE_INSTRUCTION,
    AFTER_INSTRUCTION,
    ROUTE_MEMORY,
    READ_VGPR,
    READ_SGPR,
    BARRIER_RESOLVED,
    INIT,
    SHUTDOWN,
    KIND_COUNT,
  };

  explicit HookEvent(Kind k) : kind(k) {}

  Kind kind;
  uint32_t dispatch_id = 0;
  uint32_t wg_id = 0;
  uint32_t wf_id = 0;
  uint64_t pc = 0;
  std::string mnemonic;
};

/// A plugin that records an ordered event log for ordering assertions.
class OrderingPlugin : public ExecutionPlugin {
public:
  OrderingPlugin() : ExecutionPlugin("ordering") {}
  std::vector<HookEvent> events;

  void onInit() override { events.push_back(HookEvent(HookEvent::INIT)); }

  void onShutdown() override { events.push_back(HookEvent(HookEvent::SHUTDOWN)); }

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override {
    HookEvent e{HookEvent::DISPATCH_PACKET_PROCESSED};
    e.dispatch_id = info.dispatch_id;
    events.push_back(e);
  }

  void onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) override {
    HookEvent e{HookEvent::DISPATCH_EXECUTION_BEGIN};
    e.dispatch_id = dispatch_id;
    events.push_back(e);
  }

  void onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) override {
    HookEvent e{HookEvent::DISPATCH_EXECUTION_END};
    e.dispatch_id = dispatch_id;
    events.push_back(e);
  }

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id, uint32_t, uint32_t,
                                   std::span<amdgpu::Wavefront *>) override {
    HookEvent e{HookEvent::WORKGROUP_DISPATCHED};
    e.dispatch_id = dispatch_id;
    e.wg_id = wg_id;
    events.push_back(e);
  }

  void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) override {
    HookEvent e{HookEvent::WORKGROUP_COMPLETED};
    e.dispatch_id = dispatch_id;
    e.wg_id = wg_id;
    events.push_back(e);
  }

  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) override {
    HookEvent e{HookEvent::WAVEFRONT_DISPATCHED};
    e.dispatch_id = wf.dispatch_id();
    e.wg_id = wf.wg_id();
    e.wf_id = wf.wf_id();
    events.push_back(e);
  }

  void onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) override {
    HookEvent e{HookEvent::WAVEFRONT_HALTED};
    e.dispatch_id = wf.dispatch_id();
    e.wg_id = wf.wg_id();
    e.wf_id = wf.wf_id();
    events.push_back(e);
  }

  void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                        amdgpu::Wavefront &wf) override {
    HookEvent e{HookEvent::BEFORE_INSTRUCTION};
    e.dispatch_id = wf.dispatch_id();
    e.wg_id = wf.wg_id();
    e.wf_id = wf.wf_id();
    e.pc = pc;
    e.mnemonic = inst.mnemonic();
    events.push_back(e);
  }

  void onAmdgpuAfterExecuteInstruction(uint64_t pc, const Instruction &inst,
                                       amdgpu::Wavefront &wf) override {
    HookEvent e{HookEvent::AFTER_INSTRUCTION};
    e.dispatch_id = wf.dispatch_id();
    e.wg_id = wf.wg_id();
    e.wf_id = wf.wf_id();
    e.pc = pc;
    e.mnemonic = inst.mnemonic();
    events.push_back(e);
  }

  void onAmdgpuRouteMemoryInstruction(const Instruction &inst, amdgpu::Wavefront &wf) override {
    HookEvent e{HookEvent::ROUTE_MEMORY};
    e.dispatch_id = wf.dispatch_id();
    e.wg_id = wf.wg_id();
    e.wf_id = wf.wf_id();
    e.mnemonic = inst.mnemonic();
    events.push_back(e);
  }

  void onAmdgpuReadVgprs(const amdgpu::Wavefront *wf, uint32_t, uint32_t, uint32_t,
                         uint8_t) override {
    HookEvent e{HookEvent::READ_VGPR};
    if (wf) {
      e.dispatch_id = wf->dispatch_id();
      e.wg_id = wf->wg_id();
      e.wf_id = wf->wf_id();
    }
    events.push_back(e);
  }

  void onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t) override {
    HookEvent e{HookEvent::READ_SGPR};
    if (wf) {
      e.dispatch_id = wf->dispatch_id();
      e.wg_id = wf->wg_id();
      e.wf_id = wf->wf_id();
    }
    events.push_back(e);
  }

  void onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wfs) override {
    HookEvent e{HookEvent::BARRIER_RESOLVED};
    if (!wfs.empty()) {
      e.dispatch_id = wfs[0]->dispatch_id();
      e.wg_id = wfs[0]->wg_id();
    }
    events.push_back(e);
  }
};

const char *kindName(HookEvent::Kind k) {
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
      "READ_SGPR",
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
          std::find(ids.begin(), ids.end(), e.dispatch_id) == ids.end())
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

/// Minimal SoC fixture: 1 XCD, 1 SE, 1 CU.
struct PluginFixture {
  std::unique_ptr<simdojo::SimulationEngine> engine;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *mem = nullptr;

  explicit PluginFixture(uint32_t num_wf_slots = 10) {
    std::string json = std::format(R"({{
      "max_ticks":10000,"num_threads":1,"exec_mode":"functional",
      "vm":{{"arch":"cdna4","gpu":{{"device":{{"wave_front_size":64}}}}}},
      "topology":{{"root":{{"name":"soc","type":"soc","children":[
        {{"name":"vram","type":"gpu_memory"}},
        {{"name":"xcd0","type":"xcd","children":[
          {{"name":"l2","type":"l2_cache"}},
          {{"name":"cp","type":"command_processor"}},
          {{"name":"se0","type":"shader_engine","children":[
            {{"name":"cu[0:1]","type":"compute_unit","config":[
              {{"key":"num_wf_slots","value":"{}"}},
              {{"key":"sgprs_per_wf","value":"104"}},
              {{"key":"vgprs_per_wf","value":"256"}},
              {{"key":"lds_size_kb","value":"64"}}
            ]}}
          ]}}
        ]}}
      ]}},"links":[
        {{"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2}},
        {{"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10}}
      ]}}}}
    )",
                                   num_wf_slots);
    auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
    soc = loaded.soc();
    mem = loaded.memory();
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    engine->build();
  }

  amdgpu::ComputeUnitCore *cu() { return soc->xcd(0)->shader_engine(0)->compute_unit(0); }
  amdgpu::CommandProcessor *cp() { return soc->xcd(0)->command_processor(); }

  uint64_t write_kernel(uint64_t addr, const uint32_t *code, size_t num_words) {
    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t kd{};
    kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 31);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
    mem->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), addr);
    mem->load_image(reinterpret_cast<const uint8_t *>(code), num_words * 4,
                    addr + sizeof(kernel_descriptor_t));
    return addr;
  }

  /// Attach an OrderingPlugin, fire onInit, and return a raw pointer to it.
  OrderingPlugin *attach_ordering_plugin() {
    plugin_group_ = std::make_shared<ExecutionPluginGroup>();
    auto plugin = std::make_unique<OrderingPlugin>();
    auto *p = plugin.get();
    plugin_group_->add(std::move(plugin));
    soc->set_plugin_group(plugin_group_);
    plugin_group_->onInit();
    return p;
  }

  void shutdown() {
    if (plugin_group_)
      plugin_group_->onShutdown();
  }

  std::shared_ptr<ExecutionPluginGroup> plugin_group_;

  void run_until_idle() {
    for (uint32_t i = 0; i < 100000 && engine->step(); ++i) {
    }
  }

  void run_kernel(const uint32_t *code, size_t num_words, uint32_t grid = 64,
                  uint32_t workgroup = 64) {
    uint64_t ko = write_kernel(0x1000, code, num_words);
    test::AqlQueue queue(mem, cp());
    queue.dispatch(ko, grid, workgroup);
    run_until_idle();
  }
};

TEST(ExecutionPluginTest, NoPluginNoCrash) {
  PluginFixture f;
  const uint32_t code[] = {S_NOP, S_ENDPGM};
  f.run_kernel(code, 2);
}

// -- Ordering tests ----------------------------------------------------------
//
// These tests use functional mode (the PluginFixture default). Tests that
// assert strictly sequential dispatch execution use num_wf_slots=1 so that
// only one wavefront can be active at a time, forcing the CP to complete each
// dispatch before starting the next.

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

} // namespace
