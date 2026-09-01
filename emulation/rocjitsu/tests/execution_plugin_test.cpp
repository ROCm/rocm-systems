// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_test.cpp
/// @brief Tests for the ExecutionPlugin infrastructure.
///
/// @details Register-observation tests in this file execute decoded/generated
/// instructions and assert that each callback names exactly the physical
/// registers, lanes, and bytes that the instruction architecturally accesses.
/// Machine-level preservation below RegisterAccess must not add callbacks.

#include "aql_queue.h"
#include "decode_test_util.h"
#include "mma_test_util.h"

#include "embedded_schema.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop1.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/kmd/linux/kfd_process.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/soc.h"
#include "scoped_temp.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include "halt_snapshot_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/plugins/plugin_config_resolver.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"
#include "rocjitsu/vm/plugins/race_detector/plugin.h"
#include "rocjitsu/vm/plugins/throughput/plugin.h"

#include <flatbuffers/flexbuffers.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

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

class ComputeUnitTestAccess {
public:
  static amdgpu::Wavefront *sgpr_owner(const amdgpu::ComputeUnitCore &cu, uint32_t reg_idx) {
    return cu.sgpr_owner(reg_idx);
  }
};

} // namespace rocjitsu::test

namespace {

using namespace rocjitsu;
using namespace rocjitsu::amdgpu;

static_assert(std::is_final_v<ExecutionPluginGroup>);
static_assert(!std::is_polymorphic_v<ExecutionPluginGroup>);
using namespace rocjitsu::plugins::race_detector;

static_assert(!std::is_default_constructible_v<ExecutionPluginGroup>);

// SOPP encoding: bits[31:23]=0x17F, bits[22:16]=op, bits[15:0]=simm16.
constexpr uint32_t sopp(uint32_t op, uint16_t simm16 = 0) {
  return 0xBF800000u | (op << 16) | simm16;
}
constexpr uint32_t S_NOP = sopp(0);
constexpr uint32_t S_ENDPGM = sopp(1);
constexpr uint32_t S_BARRIER = sopp(10);

constexpr uint32_t vgpr_src(uint32_t reg) { return 256u + reg; }

using mma_test::make_cdna4_mfma_scale_words;

// CDNA4 VOP2: opcode[30:25], vdst[24:17], vsrc1[16:9], src0[8:0]. Bit 31 = 0.
constexpr uint32_t vop2_encode(uint32_t opcode, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return ((opcode & 0x3F) << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF);
}

// CDNA4 VOP1: encoding[31:25]=0x3F, vdst[24:17], op[16:9], src0[8:0].
constexpr uint32_t vop1_encode(uint32_t opcode, uint32_t vdst, uint32_t src0) {
  return (0x3Fu << 25) | ((vdst & 0xFF) << 17) | ((opcode & 0xFF) << 9) | (src0 & 0x1FF);
}

constexpr uint32_t vop1_dpp_word(uint32_t vsrc0, uint32_t dpp_ctrl, uint32_t row_mask,
                                 uint32_t bank_mask, bool bound_ctrl = false) {
  return (vsrc0 & 0xFF) | ((dpp_ctrl & 0x1FF) << 8) | (static_cast<uint32_t>(bound_ctrl) << 19) |
         ((bank_mask & 0xF) << 24) | ((row_mask & 0xF) << 28);
}

constexpr uint32_t vop1_sdwa_word(uint32_t vsrc0, uint32_t dst_sel, uint32_t dst_unused,
                                  uint32_t src0_sel, bool clamp = false) {
  return (vsrc0 & 0xFF) | ((dst_sel & 0x7) << 8) | ((dst_unused & 0x3) << 11) |
         (static_cast<uint32_t>(clamp) << 13) | ((src0_sel & 0x7) << 16);
}

constexpr void vop3_encode(uint32_t opcode, uint32_t vdst, uint32_t src0, uint32_t src1,
                           uint32_t words[2]) {
  words[0] = (vdst & 0xFF) | ((opcode & 0x3FF) << 16) | (0x34u << 26);
  words[1] = (src0 & 0x1FF) | ((src1 & 0x1FF) << 9);
}

constexpr uint64_t kPartialExecMask = 0xA5A5'F0F0'1234'8001ULL;

class TestMemoryInstruction : public Instruction {
public:
  explicit TestMemoryInstruction(std::unique_ptr<DynamicInstState> state)
      : Instruction("test_mem", nullptr) {
    flags_ |= MEMORY_OP;
    set_data(std::move(state));
  }
};

class TestGlobalMemPipeline : public GlobalMemPipeline {
public:
  using GlobalMemPipeline::GlobalMemPipeline;
  using GlobalMemPipeline::initiate_access;
};

class TestLocalMemPipeline : public LocalMemPipeline {
public:
  using LocalMemPipeline::initiate_access;
};

class TestScalarMemPipeline : public ScalarMemPipeline {
public:
  using ScalarMemPipeline::ScalarMemPipeline;
};

class TestWaitcntInstruction : public Instruction {
public:
  explicit TestWaitcntInstruction(std::string_view mnemonic = "s_waitcnt")
      : Instruction(mnemonic, nullptr) {}
};

struct ForceScalarOverride {
  explicit ForceScalarOverride(bool value) : old(util::force_scalar()) {
    util::set_force_scalar_for_testing(value);
  }
  ~ForceScalarOverride() { util::set_force_scalar_for_testing(old); }

  bool old;
};

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
    WRITE_VGPR,
    READ_SGPR,
    WRITE_SGPR,
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
  uint32_t physical_vgpr_count = 0;
  uint32_t physical_sgpr_count = 0;
  uint32_t physical_reg = 0;
  uint64_t lane_mask = 0;
  uint8_t byte_mask = 0;
  uint64_t pc = 0;
  std::string mnemonic;
  std::string kernel_name;
  std::string kernel_symbol;
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
    e.kernel_name = info.kernel_name;
    e.kernel_symbol = info.kernel_symbol;
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

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                   uint32_t physical_vgpr_count, uint32_t physical_sgpr_count,
                                   std::span<amdgpu::Wavefront *>) override {
    HookEvent e{HookEvent::WORKGROUP_DISPATCHED};
    e.dispatch_id = dispatch_id;
    e.wg_id = wg_id;
    e.physical_vgpr_count = physical_vgpr_count;
    e.physical_sgpr_count = physical_sgpr_count;
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

  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask) override {
    HookEvent e{HookEvent::READ_VGPR};
    if (wf) {
      e.dispatch_id = wf->dispatch_id();
      e.wg_id = wf->wg_id();
      e.wf_id = wf->wf_id();
    }
    e.physical_reg = physical_reg;
    e.lane_mask = lane_mask;
    e.byte_mask = byte_mask;
    events.push_back(e);
  }

  void onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                              uint64_t lane_mask, uint8_t byte_mask) override {
    HookEvent e{HookEvent::WRITE_VGPR};
    if (wf) {
      e.dispatch_id = wf->dispatch_id();
      e.wg_id = wf->wg_id();
      e.wf_id = wf->wf_id();
    }
    e.physical_reg = physical_reg;
    e.lane_mask = lane_mask;
    e.byte_mask = byte_mask;
    events.push_back(e);
  }

  void onAmdgpuReadSgpr(const amdgpu::Wavefront *wf, uint32_t physical_reg) override {
    HookEvent e{HookEvent::READ_SGPR};
    if (wf) {
      e.dispatch_id = wf->dispatch_id();
      e.wg_id = wf->wg_id();
      e.wf_id = wf->wf_id();
    }
    e.physical_reg = physical_reg;
    events.push_back(e);
  }

  void onAmdgpuWriteScalarRegister(const amdgpu::Wavefront *wf, RegisterRef reg) override {
    if (!wf || reg.cls != RegClass::SGPR)
      return;
    for (uint32_t offset = 0; offset < reg.width; ++offset) {
      HookEvent e{HookEvent::WRITE_SGPR};
      e.dispatch_id = wf->dispatch_id();
      e.wg_id = wf->wg_id();
      e.wf_id = wf->wf_id();
      e.physical_reg = wf->sgpr_alloc().base + reg.index + offset;
      events.push_back(e);
    }
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

/// Exercises the group contract that sinks remain alive through plugin
/// destruction, including plugins that emit final output from their destructor.
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

class SerialHotHookPlugin final : public ExecutionPlugin {
public:
  SerialHotHookPlugin() : ExecutionPlugin("serial_hot_hook") {}
  bool requires_serial_hot_hooks() const override { return true; }
};

class NoSgprReadPlugin final : public ExecutionPlugin {
public:
  NoSgprReadPlugin() : ExecutionPlugin("no_sgpr_read") {}
  bool observes_sgpr_reads() const override { return false; }
  void onAmdgpuReadSgpr(const amdgpu::Wavefront *, uint32_t) override { ++callbacks; }
  void onAmdgpuReadScalarRegister(const amdgpu::Wavefront *, RegisterRef) override { ++callbacks; }

  uint32_t callbacks = 0;
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

class MfmaRacePlugin : public ExecutionPlugin {
public:
  MfmaRacePlugin() : ExecutionPlugin("mfma_race_probe") {}

  void onAmdgpuWorkgroupDispatched(uint32_t, uint32_t wg_id, uint32_t physical_vgpr_count,
                                   uint32_t physical_sgpr_count,
                                   std::span<amdgpu::Wavefront *> wavefronts) override {
    detector_ = std::make_unique<RaceDetector>(
        static_cast<int>(wavefronts.size()), static_cast<int>(physical_vgpr_count),
        static_cast<int>(physical_sgpr_count), Dim3d(static_cast<int>(wg_id)),
        [this](RaceViolation v) { violations.push_back(v); });
    wf_ = wavefronts.front();
    state_ = &detector_->getWaveRaceState(0);
  }

  void onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg, uint64_t lane_mask,
                             uint8_t byte_mask) override {
    if (wf != wf_ || !state_)
      return;
    uint32_t logical_reg = physical_reg - wf->vgpr_alloc().base;
    state_->checkVgprReadLanes(static_cast<int>(logical_reg), lane_mask, byte_mask);
  }

  void registerOutstandingLoad(uint32_t logical_reg, uint64_t exec_mask, uint8_t byte_mask = 0xF) {
    ASSERT_NE(state_, nullptr);
    state_->registerEvent(/*pc=*/0x1000, MemoryEventType::GLOBAL_TO_VGPR, {logical_reg}, exec_mask,
                          byte_mask);
  }

  std::vector<RaceViolation> violations;

private:
  std::unique_ptr<RaceDetector> detector_;
  amdgpu::Wavefront *wf_ = nullptr;
  WaveRaceState *state_ = nullptr;
};

std::vector<HookEvent> vgpr_read_events(const OrderingPlugin &plugin) {
  std::vector<HookEvent> reads;
  for (const HookEvent &e : plugin.events)
    if (e.kind == HookEvent::READ_VGPR)
      reads.push_back(e);
  return reads;
}

std::vector<HookEvent> vgpr_write_events(const OrderingPlugin &plugin) {
  std::vector<HookEvent> writes;
  for (const HookEvent &e : plugin.events)
    if (e.kind == HookEvent::WRITE_VGPR)
      writes.push_back(e);
  return writes;
}

void expect_vgpr_read_set(const std::vector<HookEvent> &events, uint32_t physical_base,
                          std::vector<uint32_t> expected_logical_regs, uint64_t expected_lane_mask,
                          uint8_t expected_byte_mask = ExecutionPlugin::kFullByteMask) {
  ASSERT_EQ(events.size(), expected_logical_regs.size());
  std::vector<uint32_t> actual_logical_regs;
  actual_logical_regs.reserve(events.size());
  for (const HookEvent &e : events) {
    EXPECT_EQ(e.lane_mask, expected_lane_mask);
    EXPECT_EQ(e.byte_mask, expected_byte_mask);
    ASSERT_GE(e.physical_reg, physical_base);
    actual_logical_regs.push_back(e.physical_reg - physical_base);
  }
  std::sort(actual_logical_regs.begin(), actual_logical_regs.end());
  std::sort(expected_logical_regs.begin(), expected_logical_regs.end());
  EXPECT_EQ(actual_logical_regs, expected_logical_regs);
}

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

  explicit PluginFixture(uint32_t num_wf_slots = 10, std::string_view arch = "cdna4",
                         uint32_t wavefront_size = 64, uint32_t sgprs_per_wf = 104,
                         uint32_t vgprs_per_wf = 256) {
    std::string json = std::format(R"({{
      "max_ticks":10000,"num_threads":1,"exec_mode":"functional",
      "vm":{{"arch":"{}","gpu":{{"device":{{"wave_front_size":{},
        "num_sdma_engines":0}}}}}},
      "topology":{{"root":{{"name":"soc","type":"soc","children":[
        {{"name":"vram","type":"gpu_memory"}},
        {{"name":"xcd0","type":"xcd","children":[
          {{"name":"l2","type":"l2_cache"}},
          {{"name":"cp","type":"command_processor"}},
          {{"name":"se0","type":"shader_engine","children":[
            {{"name":"cu[0:1]","type":"compute_unit","config":[
              {{"key":"num_wf_slots","value":"{}"}},
              {{"key":"sgprs_per_wf","value":"{}"}},
              {{"key":"vgprs_per_wf","value":"{}"}},
              {{"key":"lds_size_kb","value":"64"}}
            ]}}
          ]}}
        ]}}
      ]}},"links":[
        {{"src":"xcd0.cp.req_0","dst":"xcd0.se0.cu0.cpl","latency":1,"weight":2}},
        {{"src":"xcd0.se0.cu0.req","dst":"xcd0.l2.cpl_0","latency":1,"weight":10}}
      ]}}}}
    )",
                                   arch, wavefront_size, num_wf_slots, sgprs_per_wf, vgprs_per_wf);
    auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
    soc = loaded.soc();
    mem = loaded.memory();
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    engine->create();
  }

  amdgpu::ComputeUnitCore *cu() { return soc->xcd(0)->shader_engine(0)->compute_unit(0); }
  amdgpu::CommandProcessor *cp() { return soc->xcd(0)->command_processor(); }

  uint64_t write_kernel(uint64_t addr, const uint32_t *code, size_t num_words,
                        uint32_t granulated_sgpr_count = 12) {
    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t kd{};
    kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 31);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    granulated_sgpr_count);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
    mem->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), addr);
    mem->load_image(reinterpret_cast<const uint8_t *>(code), num_words * 4,
                    addr + sizeof(kernel_descriptor_t));
    return addr;
  }

  /// Attach an OrderingPlugin, fire onInit, and return a raw pointer to it.
  OrderingPlugin *attach_ordering_plugin() {
    plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
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
                  uint32_t workgroup = 64, uint32_t granulated_sgpr_count = 12) {
    uint64_t ko = write_kernel(0x1000, code, num_words, granulated_sgpr_count);
    test::AqlQueue queue(mem, cp());
    queue.dispatch(ko, grid, workgroup);
    run_until_idle();
  }
};

TEST(ExecutionPluginTest, SgprBlockOwnersTrackPhysicalAllocationAndSlotReuseAcrossTargets) {
  struct TargetCase {
    std::string_view arch;
    uint32_t wave_size;
    uint32_t sgprs_per_wf;
  };

  for (const TargetCase target : {TargetCase{"cdna4", 64, 104}, TargetCase{"cdna5", 32, 128}}) {
    SCOPED_TRACE(target.arch);
    PluginFixture f(/*num_wf_slots=*/3, target.arch, target.wave_size, target.sgprs_per_wf);
    auto *plugin = f.attach_ordering_plugin();
    const uint32_t first_sgpr_count = target.sgprs_per_wf / 2;
    auto *first = f.cu()->dispatch_wf(/*wg_id=*/10, /*pc=*/0, first_sgpr_count,
                                      /*vgprs=*/32);
    auto *second = f.cu()->dispatch_wf(/*wg_id=*/20, /*pc=*/0, target.sgprs_per_wf,
                                       /*vgprs=*/32);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    ASSERT_EQ(first->sgpr_alloc().base, 0u);
    ASSERT_EQ(second->sgpr_alloc().base, target.sgprs_per_wf);

    auto expect_owner = [&](const Wavefront &owner, uint32_t physical_reg) {
      plugin->events.clear();
      EXPECT_EQ(f.cu()->read_sgpr(physical_reg), 0u);
      ASSERT_EQ(plugin->events.size(), 1u);
      EXPECT_EQ(plugin->events[0].kind, HookEvent::READ_SGPR);
      EXPECT_EQ(plugin->events[0].wf_id, owner.wf_id());
      EXPECT_EQ(plugin->events[0].wg_id, owner.wg_id());
      EXPECT_EQ(plugin->events[0].physical_reg, physical_reg);
    };

    expect_owner(*first, first->sgpr_alloc().base + first_sgpr_count - 1);
    expect_owner(*second, second->sgpr_alloc().base);
    expect_owner(*first, first->sgpr_alloc().base + first_sgpr_count);

    const uint32_t reused_base = first->sgpr_alloc().base;
    const uint32_t reused_slot = first->wf_id();
    first->halt(Wavefront::CpCompletionNotice::Suppress);
    EXPECT_EQ(test::ComputeUnitTestAccess::sgpr_owner(*f.cu(), reused_base), nullptr);
    auto *reused = f.cu()->dispatch_wf(/*wg_id=*/30, /*pc=*/0, target.sgprs_per_wf,
                                       /*vgprs=*/32);
    ASSERT_NE(reused, nullptr);
    EXPECT_EQ(reused->wf_id(), reused_slot);
    EXPECT_EQ(reused->sgpr_alloc().base, reused_base);
    expect_owner(*reused, reused_base + target.sgprs_per_wf - 1);
    expect_owner(*second, second->sgpr_alloc().base + target.sgprs_per_wf - 1);
  }
}

struct Wave32PluginFixture {
  std::unique_ptr<amdgpu::GpuMemory> gpu_mem;
  std::unique_ptr<amdgpu::L2Cache> l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::shared_ptr<ExecutionPluginGroup> plugin_group;

  explicit Wave32PluginFixture(rj_code_arch_t arch = ROCJITSU_CODE_ARCH_CDNA5)
      : gpu_mem(std::make_unique<amdgpu::GpuMemory>("wave32_plugin_mem")),
        l2(std::make_unique<amdgpu::L2Cache>("wave32_plugin_l2")) {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 104;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("wave32_plugin_cu", cfg, gpu_mem.get(), l2.get());
  }

  OrderingPlugin *attach_ordering_plugin() {
    plugin_group = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto plugin = std::make_unique<OrderingPlugin>();
    auto *p = plugin.get();
    plugin_group->add(std::move(plugin));
    cu->set_plugin_group(plugin_group);
    plugin_group->onInit();
    return p;
  }
};

TEST(ExecutionPluginTest, Vop3CompareObservesOnlyArchitecturalDestinationReads) {
  Wave32PluginFixture f;
  ASSERT_NE(f.cu, nullptr);
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xFFFFFFFFULL);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  constexpr uint32_t kScalarDst = 12;
  uint32_t vbase = wf->vgpr_alloc().base;
  uint32_t sbase = wf->sgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    f.cu->write_vgpr(vbase + kSrc0, lane, lane);
    f.cu->write_vgpr(vbase + kSrc1, lane, lane);
  }
  f.cu->write_sgpr(sbase + kScalarDst, 0xFFFFFFFFu);
  f.cu->write_sgpr(sbase + kScalarDst + 1, 0u);

  auto destination_reads = [&] {
    std::vector<uint32_t> reads;
    for (const auto &event : plugin->events)
      if (event.kind == HookEvent::READ_SGPR)
        reads.push_back(event.physical_reg);
    return reads;
  };

  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  cdna5::Vop3MachineInst plain_raw{};
  plain_raw.vdst = kScalarDst;
  plain_raw.src0 = 256 + kSrc0;
  plain_raw.src1 = 256 + kSrc1;
  cdna5::VCmpEqU32Vop3 plain(reinterpret_cast<const cdna5::MachineInst *>(&plain_raw));
  plugin->events.clear();
  plain.execute_impl(*wf);
  EXPECT_TRUE(destination_reads().empty());

  cdna5::Vop3VopDpp16MachineInst dpp_raw{};
  dpp_raw.vdst = kScalarDst;
  dpp_raw.src0 = amdgpu::SRC_DPP;
  dpp_raw.src1 = 256 + kSrc1;
  dpp_raw.vsrc0 = kSrc0;
  dpp_raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  dpp_raw.fi = 1;
  dpp_raw.bound_ctrl = 0;
  dpp_raw.bank_mask = 0xF;
  dpp_raw.row_mask = 0xF;
  cdna5::VCmpEqU32Vop3 dpp(reinterpret_cast<const cdna5::MachineInst *>(&dpp_raw));
  plugin->events.clear();
  dpp.execute_impl(*wf);
  // BOUND_CTRL=0 forces invalid-source compare bits to zero; it does not
  // preserve or read the old lane-mask destination.
  EXPECT_TRUE(destination_reads().empty());

  // BOUND_CTRL=1 supplies zero for the OOB source and likewise requires no
  // old-destination read.
  dpp_raw.bound_ctrl = 1;
  cdna5::VCmpEqU32Vop3 zero_fill(reinterpret_cast<const cdna5::MachineInst *>(&dpp_raw));
  plugin->events.clear();
  zero_fill.execute_impl(*wf);
  EXPECT_TRUE(destination_reads().empty());
}

TEST(ExecutionPluginTest, Vop3DppSecondaryMaskObservesOnlyRequiredOldDestinationRead) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }

  Wave32PluginFixture f;
  ASSERT_NE(f.cu, nullptr);
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xFFFFFFFFULL);

  constexpr uint32_t kSrc0 = 4;
  constexpr uint32_t kSrc1 = 8;
  constexpr uint32_t kDst = 10;
  constexpr uint32_t kScalarDst = 12;
  constexpr uint32_t kCarryIn = 20;
  uint32_t vbase = wf->vgpr_alloc().base;
  uint32_t sbase = wf->sgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    f.cu->write_vgpr(vbase + kSrc0, lane, lane);
    f.cu->write_vgpr(vbase + kSrc1, lane, 1u);
  }
  f.cu->write_sgpr(sbase + kCarryIn, 0u);

  cdna5::Vop3SdstEncVopDpp16MachineInst raw{};
  raw.vdst = kDst;
  raw.sdst = kScalarDst;
  raw.src0 = amdgpu::SRC_DPP;
  raw.src1 = 256 + kSrc1;
  raw.src2 = kCarryIn;
  raw.vsrc0 = kSrc0;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
  raw.fi = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;

  auto destination_read_count = [&] {
    return std::ranges::count_if(plugin->events, [&](const HookEvent &event) {
      return event.kind == HookEvent::READ_SGPR && event.physical_reg == sbase + kScalarDst;
    });
  };

  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  for (bool force_scalar : {false, true}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "SIMD");
    ForceScalarOverride force_scalar_scope(force_scalar);

    // BOUND_CTRL=0 preserves the invalid lane's old carry bit, so exactly one
    // architectural read of the old scalar destination is required.
    raw.bound_ctrl = 0;
    f.cu->write_sgpr(sbase + kScalarDst, 0xFFFFFFFFu);
    cdna5::VAddCoCiU32Vop3SdstEnc preserve(reinterpret_cast<const cdna5::MachineInst *>(&raw));
    plugin->events.clear();
    preserve.execute_impl(*wf);
    EXPECT_EQ(destination_read_count(), 1);

    // BOUND_CTRL=1 supplies zero for the invalid source. No old-destination
    // merge or read is needed; the raw result is committed directly.
    raw.bound_ctrl = 1;
    f.cu->write_sgpr(sbase + kScalarDst, 0xFFFFFFFFu);
    cdna5::VAddCoCiU32Vop3SdstEnc zero_fill(reinterpret_cast<const cdna5::MachineInst *>(&raw));
    plugin->events.clear();
    zero_fill.execute_impl(*wf);
    EXPECT_EQ(destination_read_count(), 0);
  }
}

std::vector<uint8_t> make_loaded_kernel_symbol_elf(uint64_t kernel_descriptor_offset,
                                                   std::string_view symbol_name);

TEST(ExecutionPluginTest, NoPluginNoCrash) {
  PluginFixture f;
  const uint32_t code[] = {S_NOP, S_ENDPGM};
  f.run_kernel(code, 2);
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
      EXPECT_TRUE(execution_seconds.IsNumeric())
          << "families." << family_name << ".execution_seconds must be numeric";
      EXPECT_TRUE(execution_mips.IsNumeric())
          << "families." << family_name << ".execution_mips must be numeric";
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

TEST(ExecutionPluginTest, ValuSimdReadObservationUsesActiveExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    ForceScalarOverride force_simd(false);
    PluginFixture f(/*num_wf_slots=*/1);
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kPartialExecMask);

    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vb + 0, lane, lane);
      cu->write_vgpr(vb + 1, lane, lane * 3);
      cu->write_vgpr(vb + 2, lane, 0);
    }

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    uint32_t words[4] = {vop2_encode(/*opcode=*/52, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256), 0u, 0u,
                         0u};
    Instruction *inst = decode_valid(*decoder, words);
    ASSERT_NE(inst, nullptr);
    cu->execute_instruction(inst, *wf);
    delete inst;

    expect_vgpr_read_set(vgpr_read_events(*plugin), vb, {0, 1}, kPartialExecMask);
  }
}

TEST(ExecutionPluginTest, ValuSimdWriteObservationUsesActiveExecMask) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    ForceScalarOverride force_simd(false);
    PluginFixture f(/*num_wf_slots=*/1);
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kPartialExecMask);

    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vb + 0, lane, lane);
      cu->write_vgpr(vb + 1, lane, lane * 3);
      cu->write_vgpr(vb + 2, lane, 0);
    }

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    uint32_t words[4] = {vop2_encode(/*opcode=*/52, /*vdst=*/2, /*vsrc1=*/1, /*src0=*/256), 0u, 0u,
                         0u};
    Instruction *inst = decode_valid(*decoder, words);
    ASSERT_NE(inst, nullptr);
    cu->execute_instruction(inst, *wf);
    delete inst;

    expect_vgpr_read_set(vgpr_write_events(*plugin), vb, {2}, kPartialExecMask);
  }
}

// DPP observation tests exercise the interaction among EXEC, row/bank
// destination masks, source-lane permutation, BOUND_CTRL, fetch-inactive, and
// source/destination aliasing. Helper-level permutation coverage lives in
// shared_infra_test.cpp; these tests prove that decoded instructions carry the
// resulting source and destination lane sets through to plugin callbacks.
TEST(ExecutionPluginTest, DppObservationReportsExactSourceAndDestinationLanes) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    PluginFixture f(/*num_wf_slots=*/1);
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kPartialExecMask);

    constexpr uint32_t kSrc = 2;
    constexpr uint32_t kDst = 5;
    constexpr uint32_t kSrcValue = 0x11223344u;
    constexpr uint32_t kOldDst = 0xAABBCCDDu;
    constexpr uint64_t kDppWriteMask = 0x0000'0000'0000'0F0FULL;
    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vb + kSrc, lane, kSrcValue);
      cu->write_vgpr(vb + kDst, lane, kOldDst);
    }

    // V_MOV_B32 with a partial row/bank mask. Only lane 0 survives EXEC and the
    // destination mask. Source observation is determined from every active
    // destination before row/bank commit filtering; dpp_ctrl=0 selects the
    // first source lane of each active quad.
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    uint32_t words[2] = {
        vop1_encode(/*opcode=*/1, kDst, amdgpu::SRC_DPP),
        vop1_dpp_word(kSrc, /*dpp_ctrl=*/0, /*row_mask=*/0x1, /*bank_mask=*/0x5),
    };
    for (bool force_scalar : {false, true}) {
      SCOPED_TRACE(force_scalar ? "scalar" : "simd");
      ForceScalarOverride force_scalar_override(force_scalar);
      for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
        cu->write_vgpr(vb + kDst, lane, kOldDst);

      Instruction *inst = decode_valid(*decoder, words);
      ASSERT_NE(inst, nullptr);
      plugin->events.clear();
      cu->execute_instruction(inst, *wf);
      delete inst;

      for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
        const uint64_t lane_bit = uint64_t{1} << lane;
        const uint32_t expected =
            (kPartialExecMask & kDppWriteMask & lane_bit) ? kSrcValue : kOldDst;
        EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, lane), expected) << "lane " << lane;
      }

      const auto reads = vgpr_read_events(*plugin);
      ASSERT_EQ(reads.size(), 1u);
      EXPECT_EQ(reads[0].physical_reg, vb + kSrc);
      EXPECT_EQ(reads[0].lane_mask, 0x1111'1010'1111'1001ULL);
      EXPECT_EQ(reads[0].byte_mask, ExecutionPlugin::kFullByteMask);

      const auto writes = vgpr_write_events(*plugin);
      ASSERT_EQ(writes.size(), 1u);
      EXPECT_EQ(writes[0].physical_reg, vb + kDst);
      EXPECT_EQ(writes[0].lane_mask, 1u);
      EXPECT_EQ(writes[0].byte_mask, ExecutionPlugin::kFullByteMask);
    }
  }
}

TEST(ExecutionPluginTest, DppOutOfBoundsObservationHonorsBoundCtrl) {
  ForceScalarOverride force_simd(false);
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  constexpr uint32_t kOldDst = 0xAABBCCDDu;
  const uint32_t vb = wf->vgpr_alloc().base;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);

  auto run = [&](bool bound_ctrl) {
    cu->write_vgpr(vb + kSrc, 0, 0x11223344u);
    cu->write_vgpr(vb + kDst, 0, kOldDst);
    uint32_t words[2] = {
        vop1_encode(/*v_mov_b32 opcode=*/1, kDst, amdgpu::SRC_DPP),
        vop1_dpp_word(kSrc, amdgpu::dpp::ROW_SHR1, /*row_mask=*/0xF,
                      /*bank_mask=*/0xF, bound_ctrl),
    };
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
    ASSERT_NE(inst, nullptr);
    plugin->events.clear();
    cu->execute_instruction(inst.get(), *wf);
  };

  // Lane 0 has no source for row_shr:1. BOUND_CTRL=0 suppresses the
  // destination write entirely, so neither a source read nor destination write
  // is architectural.
  run(false);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), kOldDst);
  EXPECT_TRUE(vgpr_read_events(*plugin).empty());
  EXPECT_TRUE(vgpr_write_events(*plugin).empty());

  // BOUND_CTRL=1 turns the same missing source into zero. No VGPR source is
  // read, but the destination lane is now architecturally written.
  run(true);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0u);
  EXPECT_TRUE(vgpr_read_events(*plugin).empty());
  const auto writes = vgpr_write_events(*plugin);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].physical_reg, vb + kDst);
  EXPECT_EQ(writes[0].lane_mask, 1u);
  EXPECT_EQ(writes[0].byte_mask, ExecutionPlugin::kFullByteMask);
}

TEST(ExecutionPluginTest, DppSourceDestinationAliasStagesBeforeWriting) {
  ForceScalarOverride force_simd(false);
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xFu);

  constexpr uint32_t kReg = 5;
  constexpr std::array<uint32_t, 4> kValues{10u, 20u, 30u, 40u};
  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < kValues.size(); ++lane)
    cu->write_vgpr(vb + kReg, lane, kValues[lane]);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  uint32_t words[2] = {
      vop1_encode(/*v_mov_b32 opcode=*/1, kReg, amdgpu::SRC_DPP),
      vop1_dpp_word(kReg, /*quad_perm:[1,0,3,2]=*/0xB1, /*row_mask=*/0x1,
                    /*bank_mask=*/0x1, /*bound_ctrl=*/true),
  };
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  plugin->events.clear();
  cu->execute_instruction(inst.get(), *wf);

  EXPECT_EQ(cu->read_vgpr_storage(vb + kReg, 0), kValues[1]);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kReg, 1), kValues[0]);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kReg, 2), kValues[3]);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kReg, 3), kValues[2]);

  const auto reads = vgpr_read_events(*plugin);
  ASSERT_EQ(reads.size(), 1u);
  EXPECT_EQ(reads[0].physical_reg, vb + kReg);
  EXPECT_EQ(reads[0].lane_mask, 0xFu);
  const auto writes = vgpr_write_events(*plugin);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].physical_reg, vb + kReg);
  EXPECT_EQ(writes[0].lane_mask, 0xFu);
}

TEST(ExecutionPluginTest, Dpp8FetchInactiveControlsSourceObservation) {
  ForceScalarOverride force_simd(false);
  Wave32PluginFixture f;
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);
  wf->set_exec(1u << 1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  constexpr uint32_t kValue = 0x11223344u;
  const uint32_t vb = wf->vgpr_alloc().base;
  const uint32_t lane_sel = (0u << 0u) | (0u << 3u) | (2u << 6u) | (3u << 9u) | (4u << 12u) |
                            (5u << 15u) | (6u << 18u) | (7u << 21u);
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  auto run = [&](uint32_t src_marker) {
    f.cu->write_vgpr(vb + kSrc, 0, kValue);
    f.cu->write_vgpr(vb + kDst, 1, 0xAABBCCDDu);
    const uint32_t words[2] = {
        vop1_encode(/*v_mov_b32 opcode=*/1, kDst, src_marker),
        kSrc | (lane_sel << 8u),
    };
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
    ASSERT_NE(inst, nullptr);
    plugin->events.clear();
    f.cu->execute_instruction(inst.get(), *wf);
  };

  run(amdgpu::SRC_DPP8_FI_0);
  EXPECT_EQ(f.cu->read_vgpr_storage(vb + kDst, 1), 0u);
  EXPECT_TRUE(vgpr_read_events(*plugin).empty());
  auto writes = vgpr_write_events(*plugin);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].lane_mask, 1u << 1);

  run(amdgpu::SRC_DPP8_FI_1);
  EXPECT_EQ(f.cu->read_vgpr_storage(vb + kDst, 1), kValue);
  const auto reads = vgpr_read_events(*plugin);
  ASSERT_EQ(reads.size(), 1u);
  EXPECT_EQ(reads[0].physical_reg, vb + kSrc);
  EXPECT_EQ(reads[0].lane_mask, 1u);
  writes = vgpr_write_events(*plugin);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].physical_reg, vb + kDst);
  EXPECT_EQ(writes[0].lane_mask, 1u << 1);
}

// gfx1250 sub-dword and split-backend tests. The true16 cases cover both
// low-to-high and high-to-low half selection while checking that preservation
// of the other half is invisible to plugins. The 64-bit case separately pins
// write forwarding for both physical destination dwords.
TEST(ExecutionPluginTest, True16InstructionsReportSelectedSourceAndDestinationHalves) {
  Wave32PluginFixture f;
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  struct Case {
    uint32_t word;
    uint32_t source_value;
    uint32_t expected_destination;
    uint8_t source_byte_mask;
    uint8_t destination_byte_mask;
  };
  constexpr std::array cases{
      Case{0x7F02D300u, 0xAABB00FFu, 0xFF005555u, ExecutionPlugin::kLowHalfByteMask,
           ExecutionPlugin::kHighHalfByteMask}, // v_not_b16_e32 v1.h, v0.l
      Case{0x7E02D380u, 0x00FFAABBu, 0xAAAAFF00u, ExecutionPlugin::kHighHalfByteMask,
           ExecutionPlugin::kLowHalfByteMask}, // v_not_b16_e32 v1.l, v0.h
  };

  const uint32_t vb = wf->vgpr_alloc().base;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  for (const Case &test_case : cases) {
    SCOPED_TRACE(std::format("instruction word 0x{:08x}", test_case.word));
    f.cu->write_vgpr(vb + 0, 0, test_case.source_value);
    f.cu->write_vgpr(vb + 1, 0, 0xAAAA5555u);
    f.cu->write_vgpr(vb + 129, 0, 0xDEADBEEFu);

    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &test_case.word));
    ASSERT_NE(inst, nullptr);
    plugin->events.clear();
    f.cu->execute_instruction(inst.get(), *wf);

    EXPECT_EQ(f.cu->read_vgpr_storage(vb + 1, 0), test_case.expected_destination);
    EXPECT_EQ(f.cu->read_vgpr_storage(vb + 129, 0), 0xDEADBEEFu);

    const auto reads = vgpr_read_events(*plugin);
    ASSERT_EQ(reads.size(), 1u);
    EXPECT_EQ(reads[0].physical_reg, vb + 0);
    EXPECT_EQ(reads[0].lane_mask, 1u);
    EXPECT_EQ(reads[0].byte_mask, test_case.source_byte_mask);
    const auto writes = vgpr_write_events(*plugin);
    ASSERT_EQ(writes.size(), 1u);
    EXPECT_EQ(writes[0].physical_reg, vb + 1);
    EXPECT_EQ(writes[0].lane_mask, 1u);
    EXPECT_EQ(writes[0].byte_mask, test_case.destination_byte_mask);
  }
}

TEST(ExecutionPluginTest, Gfx1250Simd64BitWriteReportsBothDestinationRegisters) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    ForceScalarOverride force_simd(false);
    Wave32PluginFixture f;
    auto *plugin = f.attach_ordering_plugin();
    auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1);

    constexpr uint32_t kSrc = 2;
    constexpr uint32_t kDst = 5;
    const uint32_t vb = wf->vgpr_alloc().base;
    f.cu->write_vgpr(vb + kSrc, 0, 0x11223344u);
    f.cu->write_vgpr(vb + kSrc + 1, 0, 0x55667788u);
    f.cu->write_vgpr(vb + kDst, 0, 0xAABBCCDDu);
    f.cu->write_vgpr(vb + kDst + 1, 0, 0xEEFF0011u);

    const uint32_t word =
        vop1_encode(/*v_mov_b64 opcode=*/29, kDst, /*generic VGPR source=*/256 + kSrc);
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &word));
    ASSERT_NE(inst, nullptr);
    plugin->events.clear();
    f.cu->execute_instruction(inst.get(), *wf);

    EXPECT_EQ(f.cu->read_vgpr_storage(vb + kDst, 0), 0x11223344u);
    EXPECT_EQ(f.cu->read_vgpr_storage(vb + kDst + 1, 0), 0x55667788u);
    expect_vgpr_read_set(vgpr_read_events(*plugin), vb, {kSrc, kSrc + 1}, 1u);
    expect_vgpr_read_set(vgpr_write_events(*plugin), vb, {kDst, kDst + 1}, 1u);
  }
}

TEST(ExecutionPluginTest, DppTrue16SourceReportsSelectedHalf) {
  ForceScalarOverride force_scalar(true);
  ScopedIsaExecutionBackend execution_backend_scope{&cdna5::execution_backend()};
  Wave32PluginFixture f;
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 0;
  constexpr uint32_t kDst = 1;
  const uint32_t vb = wf->vgpr_alloc().base;
  f.cu->write_vgpr(vb + kSrc, 0, 0xAABB00FFu);
  f.cu->write_vgpr(vb + kDst, 0, 0xAAAA5555u);

  cdna5::Vop1VopDpp16MachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = 0xE4; // identity quad permutation
  raw.fi = 1;
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;
  cdna5::VNotB16Vop1 inst(reinterpret_cast<const cdna5::MachineInst *>(&raw));

  plugin->events.clear();
  inst.execute_impl(*wf);

  EXPECT_EQ(f.cu->read_vgpr_storage(vb + kDst, 0), 0xAAAAFF00u);
  const auto reads = vgpr_read_events(*plugin);
  ASSERT_EQ(reads.size(), 1u);
  EXPECT_EQ(reads[0].physical_reg, vb + kSrc);
  EXPECT_EQ(reads[0].lane_mask, 1u);
  EXPECT_EQ(reads[0].byte_mask, ExecutionPlugin::kLowHalfByteMask);
}

TEST(ExecutionPluginTest, Rdna4DppTrue16SourceReportsOpSelHalf) {
  ForceScalarOverride force_scalar(true);
  ScopedIsaExecutionBackend execution_backend_scope{&rdna4::execution_backend()};
  Wave32PluginFixture f(ROCJITSU_CODE_ARCH_RDNA4);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu.get();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x7);

  constexpr uint32_t kSrc0 = 0;
  constexpr uint32_t kSrc1 = 1;
  constexpr uint32_t kDst = 2;
  const uint32_t vb = wf->vgpr_alloc().base;

  for (bool source_high : {false, true}) {
    SCOPED_TRACE(source_high ? "high source half" : "low source half");
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(
          vb + kSrc0, lane,
          (static_cast<uint32_t>(util::f32_to_f16(static_cast<float>(lane + 20))) << 16) |
              util::f32_to_f16(static_cast<float>(lane + 10)));
      const uint32_t half = util::f32_to_f16(0.5f);
      cu->write_vgpr(vb + kSrc1, lane, half | (half << 16));
      cu->write_vgpr(vb + kDst, lane, (0x7000u + lane) << 16 | (0x4000u + lane));
    }

    rdna4::Vop3VopDpp16MachineInst raw{};
    raw.vdst = kDst;
    raw.opsel = source_high ? 0xBu : 0u;
    raw.op = 0x132u;
    raw.encoding = 0x35u;
    raw.src0 = amdgpu::SRC_DPP;
    raw.src1 = 256u + kSrc1;
    raw.vsrc0 = kSrc0;
    raw.dpp_ctrl = amdgpu::dpp::ROW_SHR1;
    raw.fi = 1;
    raw.bound_ctrl = 0;
    raw.bank_mask = 0xFu;
    raw.row_mask = 0x1u;

    rdna4::VAddF16Vop3 inst(reinterpret_cast<const rdna4::MachineInst *>(&raw));
    plugin->events.clear();
    inst.execute_impl(*wf);

    const float source_base = source_high ? 20.0f : 10.0f;
    const uint32_t lane1_result = util::f32_to_f16(source_base + 0.5f);
    const uint32_t lane2_result = util::f32_to_f16(source_base + 1.5f);
    EXPECT_EQ(cu->read_vgpr(vb + kDst, 1),
              source_high ? (lane1_result << 16) | 0x4001u : (0x7001u << 16) | lane1_result);
    EXPECT_EQ(cu->read_vgpr(vb + kDst, 2),
              source_high ? (lane2_result << 16) | 0x4002u : (0x7002u << 16) | lane2_result);

    std::vector<HookEvent> src0_reads;
    for (const auto &event : vgpr_read_events(*plugin))
      if (event.physical_reg == vb + kSrc0)
        src0_reads.push_back(event);
    ASSERT_EQ(src0_reads.size(), 1u);
    EXPECT_EQ(src0_reads[0].lane_mask, 0x3u);
    EXPECT_EQ(src0_reads[0].byte_mask,
              source_high ? ExecutionPlugin::kHighHalfByteMask : ExecutionPlugin::kLowHalfByteMask);
  }
}

TEST(ExecutionPluginTest, Dpp64BitSourceSimdStagesBothPhysicalDwords) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  }
  ForceScalarOverride force_simd(false);
  ScopedIsaExecutionBackend execution_backend_scope{&cdna4::execution_backend()};
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kSrc, 1, 0x11223344u);
  cu->write_vgpr(vb + kSrc + 1, 1, 0x55667788u);

  cdna4::Vop1VopDppMachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = amdgpu::dpp::ROW_SHARE_BASE + 1; // row_newbcast:1
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;
  cdna4::VMovB64Vop1 inst(reinterpret_cast<const cdna4::MachineInst *>(&raw));

  plugin->events.clear();
  inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0x11223344u);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst + 1, 0), 0x55667788u);
  expect_vgpr_read_set(vgpr_read_events(*plugin), vb, {kSrc, kSrc + 1}, 1u << 1);
}

TEST(ExecutionPluginTest, DppInstructionReuseRestagesOriginalSource) {
  ForceScalarOverride force_scalar(true);
  ScopedIsaExecutionBackend execution_backend_scope{&cdna4::execution_backend()};
  PluginFixture f(/*num_wf_slots=*/1);
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;

  cdna4::Vop1VopDppMachineInst raw{};
  raw.src0 = amdgpu::SRC_DPP;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.dpp_ctrl = 0xE4; // identity quad permutation
  raw.bound_ctrl = 1;
  raw.bank_mask = 0xF;
  raw.row_mask = 0xF;
  cdna4::VMovB32Vop1 inst(reinterpret_cast<const cdna4::MachineInst *>(&raw));

  cu->write_vgpr(vb + kSrc, 0, 0x11111111u);
  inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0x11111111u);

  cu->write_vgpr(vb + kSrc, 0, 0x22222222u);
  inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0x22222222u);
}

TEST(ExecutionPluginTest, Sdwa64BitDestinationWritesLegalConversionResult) {
  ForceScalarOverride force_scalar(true);
  ScopedIsaExecutionBackend execution_backend_scope{&cdna3::execution_backend()};
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"cdna3");
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint8_t kFp8Lo = 0x34u;
  constexpr uint8_t kFp8Hi = 0x12u;
  cu->write_vgpr(vb + kSrc, 0,
                 (static_cast<uint32_t>(kFp8Hi) << 24) | (static_cast<uint32_t>(kFp8Lo) << 16) |
                     0xABCDu);

  cdna3::Vop1VopSdwaMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.src0_sel = amdgpu::sdwa::WORD_1;
  raw.dst_sel = amdgpu::sdwa::DWORD;
  raw.dst_unused = amdgpu::sdwa::UNUSED_PAD;
  cdna3::VCvtPkF32Fp8Vop1 inst(reinterpret_cast<const cdna3::MachineInst *>(&raw));

  plugin->events.clear();
  inst.execute_impl(*wf);

  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0),
            std::bit_cast<uint32_t>(util::fp8_e4m3_fnuz_to_f32(kFp8Lo)));
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst + 1, 0),
            std::bit_cast<uint32_t>(util::fp8_e4m3_fnuz_to_f32(kFp8Hi)));

  const auto reads = vgpr_read_events(*plugin);
  ASSERT_EQ(reads.size(), 1u);
  EXPECT_EQ(reads[0].physical_reg, vb + kSrc);
  EXPECT_EQ(reads[0].lane_mask, 1u);
  EXPECT_EQ(reads[0].byte_mask, ExecutionPlugin::kHighHalfByteMask);
  expect_vgpr_read_set(vgpr_write_events(*plugin), vb, {kDst, kDst + 1}, 1u);
}

TEST(ExecutionPluginTest, SdwaInstructionReuseRestagesOriginalSource) {
  ForceScalarOverride force_scalar(true);
  ScopedIsaExecutionBackend execution_backend_scope{&cdna4::execution_backend()};
  PluginFixture f(/*num_wf_slots=*/1);
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;

  cdna4::Vop1VopSdwaMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.vsrc0 = kSrc;
  raw.vdst = kDst;
  raw.src0_sel = amdgpu::sdwa::BYTE_1;
  raw.dst_sel = amdgpu::sdwa::DWORD;
  raw.dst_unused = amdgpu::sdwa::UNUSED_PAD;
  cdna4::VMovB32Vop1 inst(reinterpret_cast<const cdna4::MachineInst *>(&raw));

  cu->write_vgpr(vb + kSrc, 0, 0x11223344u);
  inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0x33u);

  cu->write_vgpr(vb + kSrc, 0, 0xAABBCCDDu);
  inst.execute_impl(*wf);
  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0xCCu);
}

TEST(ExecutionPluginTest, SdwaFloatingModifiersUseSemanticSourceWidth) {
  for (bool force_scalar : {true, false}) {
    SCOPED_TRACE(force_scalar ? "scalar" : "simd");
    ForceScalarOverride execution_mode(force_scalar);
    PluginFixture f(/*num_wf_slots=*/1);
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(1);

    constexpr uint32_t kSrc0 = 2;
    constexpr uint32_t kSrc1 = 3;
    constexpr uint32_t kDst = 5;
    const uint32_t vb = wf->vgpr_alloc().base;

    cdna4::Vop2VopSdwaMachineInst add_f32{};
    add_f32.src0 = amdgpu::SRC_SDWA;
    add_f32.vsrc0 = kSrc0;
    add_f32.vsrc1 = kSrc1;
    add_f32.vdst = kDst;
    add_f32.op = cdna4::kVAddF32Vop2;
    add_f32.src0_sel = amdgpu::sdwa::DWORD;
    add_f32.src0_abs = 1;
    add_f32.src1_sel = amdgpu::sdwa::DWORD;
    add_f32.dst_sel = amdgpu::sdwa::DWORD;
    add_f32.dst_unused = amdgpu::sdwa::UNUSED_PAD;

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    std::unique_ptr<Instruction> f32_inst(
        decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&add_f32)));
    ASSERT_NE(f32_inst, nullptr);
    cu->write_vgpr(vb + kSrc0, 0, std::bit_cast<uint32_t>(-2.0f));
    cu->write_vgpr(vb + kSrc1, 0, std::bit_cast<uint32_t>(0.5f));
    cu->execute_instruction(f32_inst.get(), *wf);
    EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), std::bit_cast<uint32_t>(2.5f));

    cdna4::Vop1VopSdwaMachineInst cvt_bf16{};
    cvt_bf16.src0 = amdgpu::SRC_SDWA;
    cvt_bf16.vsrc0 = kSrc0;
    cvt_bf16.vdst = kDst;
    cvt_bf16.op = cdna4::kVCvtF32Bf16Vop1;
    cvt_bf16.encoding = cdna4::encoding::kVop1 >> 2;
    cvt_bf16.src0_sel = amdgpu::sdwa::WORD_0;
    cvt_bf16.src0_abs = 1;
    cvt_bf16.dst_sel = amdgpu::sdwa::DWORD;
    cvt_bf16.dst_unused = amdgpu::sdwa::UNUSED_PAD;

    std::unique_ptr<Instruction> bf16_inst(
        decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&cvt_bf16)));
    ASSERT_NE(bf16_inst, nullptr);
    ASSERT_EQ(std::string_view(bf16_inst->mnemonic()), "v_cvt_f32_bf16_sdwa");
    cu->write_vgpr(vb + kSrc0, 0, 0xCAFE'C000u);
    cu->execute_instruction(bf16_inst.get(), *wf);
    EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), std::bit_cast<uint32_t>(2.0f));

    for (uint32_t selection : {amdgpu::sdwa::WORD_0, amdgpu::sdwa::WORD_1}) {
      SCOPED_TRACE(selection);
      cdna4::Vop2VopSdwaMachineInst add_f16{};
      add_f16.src0 = amdgpu::SRC_SDWA;
      add_f16.vsrc0 = kSrc0;
      add_f16.vsrc1 = kSrc1;
      add_f16.vdst = kDst;
      add_f16.op = cdna4::kVAddF16Vop2;
      add_f16.src0_sel = selection;
      add_f16.src0_abs = 1;
      add_f16.src1_sel = selection;
      add_f16.dst_sel = amdgpu::sdwa::DWORD;
      add_f16.dst_unused = amdgpu::sdwa::UNUSED_PAD;

      std::unique_ptr<Instruction> f16_inst(
          decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&add_f16)));
      ASSERT_NE(f16_inst, nullptr);
      const uint32_t shift = selection == amdgpu::sdwa::WORD_1 ? 16u : 0u;
      const uint32_t selected_word_mask = uint32_t{0xFFFF} << shift;
      cu->write_vgpr(vb + kSrc0, 0,
                     (0xCAFE'BEEFu & ~selected_word_mask) | (uint32_t{0xC000} << shift));
      cu->write_vgpr(vb + kSrc1, 0,
                     (0x1234'5678u & ~selected_word_mask) | (uint32_t{0x3800} << shift));
      cu->execute_instruction(f16_inst.get(), *wf);
      EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), util::f32_to_f16(2.5f));
    }
  }
}

TEST(ExecutionPluginTest, SdwaVop2Src1SelectorReportsExactBytes) {
  ForceScalarOverride force_scalar(true);
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc0 = 2;
  constexpr uint32_t kSrc1 = 3;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kSrc0, 0, 10u);
  cu->write_vgpr(vb + kSrc1, 0, 0x11807F22u);

  cdna4::Vop2VopSdwaMachineInst raw{};
  raw.src0 = amdgpu::SRC_SDWA;
  raw.vsrc0 = kSrc0;
  raw.vsrc1 = kSrc1;
  raw.vdst = kDst;
  raw.op = cdna4::kVAddU32Vop2;
  raw.src0_sel = amdgpu::sdwa::DWORD;
  raw.src1_sel = amdgpu::sdwa::BYTE_2;
  raw.src1_sext = 1;
  raw.dst_sel = amdgpu::sdwa::DWORD;
  raw.dst_unused = amdgpu::sdwa::UNUSED_PAD;

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  std::unique_ptr<Instruction> inst(
      decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&raw)));
  ASSERT_NE(inst, nullptr);
  plugin->events.clear();
  cu->execute_instruction(inst.get(), *wf);

  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), 0xFFFFFF8Au);

  std::vector<HookEvent> src1_reads;
  for (const auto &event : vgpr_read_events(*plugin))
    if (event.physical_reg == vb + kSrc1)
      src1_reads.push_back(event);
  ASSERT_EQ(src1_reads.size(), 1u);
  EXPECT_EQ(src1_reads[0].lane_mask, 1u);
  EXPECT_EQ(src1_reads[0].byte_mask, 0b0100);
}

TEST(ExecutionPluginTest, SdwaVop2ScalarSelectorsUseSgprs) {
  ForceScalarOverride force_scalar(true);
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  constexpr uint32_t kVectorSrc = 2;
  constexpr uint32_t kScalarSrc = 4;
  constexpr uint32_t kDst = 5;
  constexpr uint32_t kActiveLane = 3;
  constexpr uint64_t kActiveMask = uint64_t{1} << kActiveLane;
  const uint32_t vb = wf->vgpr_alloc().base;
  const uint32_t sb = wf->sgpr_alloc().base;
  wf->set_exec(kActiveMask);

  auto run = [&](bool scalar_src1) {
    SCOPED_TRACE(scalar_src1 ? "scalar src1" : "scalar src0");
    cu->write_vgpr(vb + kVectorSrc, kActiveLane, 10u);
    cu->write_sgpr(sb + kScalarSrc, 0x11807F22u);

    cdna4::Vop2VopSdwaMachineInst raw{};
    raw.src0 = amdgpu::SRC_SDWA;
    raw.vsrc0 = scalar_src1 ? kVectorSrc : kScalarSrc;
    raw.vsrc1 = scalar_src1 ? kScalarSrc : kVectorSrc;
    raw.vdst = kDst;
    raw.op = cdna4::kVAddU32Vop2;
    raw.src0_sel = scalar_src1 ? amdgpu::sdwa::DWORD : amdgpu::sdwa::BYTE_2;
    raw.src0_sext = scalar_src1 ? 0 : 1;
    raw.s0 = scalar_src1 ? 0 : 1;
    raw.src1_sel = scalar_src1 ? amdgpu::sdwa::BYTE_2 : amdgpu::sdwa::DWORD;
    raw.src1_sext = scalar_src1 ? 1 : 0;
    raw.s1 = scalar_src1 ? 1 : 0;
    raw.dst_sel = amdgpu::sdwa::DWORD;
    raw.dst_unused = amdgpu::sdwa::UNUSED_PAD;

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    std::unique_ptr<Instruction> inst(
        decode_valid(*decoder, reinterpret_cast<const uint32_t *>(&raw)));
    ASSERT_NE(inst, nullptr);
    plugin->events.clear();
    cu->execute_instruction(inst.get(), *wf);

    EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, kActiveLane), 0xFFFFFF8Au);
    expect_vgpr_read_set(vgpr_read_events(*plugin), vb, {kVectorSrc}, kActiveMask);

    uint32_t sgpr_reads = 0;
    for (const HookEvent &event : plugin->events)
      sgpr_reads += event.kind == HookEvent::READ_SGPR;
    EXPECT_EQ(sgpr_reads, 1u);
  };

  run(false);
  run(true);
}

TEST(ExecutionPluginTest, SgprWriteObservationUsesExplicitWavePhysicalBlock) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  constexpr uint32_t kRequestedSgprs = 40;
  constexpr uint32_t kCompilerTemporary = 40;
  auto *wf = cu->dispatch_wf(0, 0, kRequestedSgprs, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  // s_mov_b32 s40, 1. The descriptor requests s0..s39, but the physical
  // 104-register wave block includes compiler/ABI temporary s40.
  constexpr uint32_t kMovTemporary = 0xBEA80081u;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  auto decoded = decoder->decode(&kMovTemporary);
  ASSERT_TRUE(decoded.succeeded());
  std::unique_ptr<Instruction> inst = std::move(decoded).value();
  ASSERT_NE(inst, nullptr);

  plugin->events.clear();
  cu->execute_instruction(inst.get(), *wf);

  std::vector<HookEvent> writes;
  for (const HookEvent &event : plugin->events)
    if (event.kind == HookEvent::WRITE_SGPR)
      writes.push_back(event);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].physical_reg, wf->sgpr_alloc().base + kCompilerTemporary);
  EXPECT_EQ(writes[0].wf_id, wf->wf_id());
  EXPECT_EQ(cu->read_sgpr_storage(writes[0].physical_reg), 1u);
}

// SDWA observation tests cover byte, word, and dword source selectors plus
// preserve, pad, sign-extension, and clamp behavior.
// Selected source bytes must be the only bytes read. Preserved destination
// bytes are storage bookkeeping, while pad/sign-extension and clamp can widen
// the architectural write mask.
TEST(ExecutionPluginTest, SdwaObservationReportsExactSourceAndDestinationBytes) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    ForceScalarOverride force_simd(false);
    PluginFixture f(/*num_wf_slots=*/1);
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kPartialExecMask);

    constexpr uint32_t kSrc = 2;
    constexpr uint32_t kDst = 5;
    constexpr uint32_t kSrcValue = 0x11223344u;
    constexpr uint32_t kOldDst = 0xAABBCCDDu;
    const uint32_t vb = wf->vgpr_alloc().base;

    struct Case {
      uint32_t dst_sel;
      uint32_t dst_unused;
      uint32_t src0_sel;
      uint32_t source_value;
      uint32_t expected_active;
      uint8_t architectural_byte_mask;
    };
    constexpr std::array cases{
        Case{amdgpu::sdwa::BYTE_1, amdgpu::sdwa::UNUSED_PRESERVE, amdgpu::sdwa::BYTE_2, kSrcValue,
             0xAABB22DDu, 0b0010},
        Case{amdgpu::sdwa::BYTE_3, amdgpu::sdwa::UNUSED_PRESERVE, amdgpu::sdwa::BYTE_3, kSrcValue,
             0x11BBCCDDu, 0b1000},
        Case{amdgpu::sdwa::WORD_0, amdgpu::sdwa::UNUSED_PRESERVE, amdgpu::sdwa::DWORD, 0x12345678u,
             0xAABB5678u, 0b0011},
        Case{amdgpu::sdwa::WORD_1, amdgpu::sdwa::UNUSED_PRESERVE, amdgpu::sdwa::DWORD, kSrcValue,
             0x3344CCDDu, 0b1100},
        Case{amdgpu::sdwa::BYTE_1, amdgpu::sdwa::UNUSED_PAD, amdgpu::sdwa::DWORD, kSrcValue,
             0x00004400u, ExecutionPlugin::kFullByteMask},
        Case{amdgpu::sdwa::BYTE_1, amdgpu::sdwa::UNUSED_SEXT, amdgpu::sdwa::BYTE_0, 0x00000080u,
             0xFFFF8000u, ExecutionPlugin::kFullByteMask},
        Case{amdgpu::sdwa::DWORD, amdgpu::sdwa::UNUSED_PAD, amdgpu::sdwa::DWORD, kSrcValue,
             kSrcValue, ExecutionPlugin::kFullByteMask},
    };

    // Selected sources and destinations carry their precise lane and byte
    // masks. Partial destinations currently use the scalar semantic path, so
    // they may report one exact write event per active lane.
    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    for (const Case &test_case : cases) {
      SCOPED_TRACE(test_case.dst_sel);
      for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
        cu->write_vgpr(vb + kSrc, lane, test_case.source_value);
        cu->write_vgpr(vb + kDst, lane, kOldDst);
      }

      uint32_t words[2] = {
          vop1_encode(/*opcode=*/1, kDst, amdgpu::SRC_SDWA),
          vop1_sdwa_word(kSrc, test_case.dst_sel, test_case.dst_unused, test_case.src0_sel),
      };
      Instruction *inst = decode_valid(*decoder, words);
      ASSERT_NE(inst, nullptr);
      plugin->events.clear();
      cu->execute_instruction(inst, *wf);
      delete inst;

      for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
        const uint32_t expected =
            (kPartialExecMask & (uint64_t{1} << lane)) ? test_case.expected_active : kOldDst;
        EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, lane), expected) << "lane " << lane;
      }

      const auto writes = vgpr_write_events(*plugin);
      ASSERT_FALSE(writes.empty());
      uint64_t observed_write_lanes = 0;
      for (const auto &write : writes) {
        EXPECT_EQ(write.physical_reg, vb + kDst);
        EXPECT_EQ(write.byte_mask, test_case.architectural_byte_mask);
        observed_write_lanes |= write.lane_mask;
      }
      EXPECT_EQ(observed_write_lanes, kPartialExecMask);

      const auto reads = vgpr_read_events(*plugin);
      ASSERT_FALSE(reads.empty());
      uint64_t observed_read_lanes = 0;
      for (const auto &read : reads) {
        EXPECT_EQ(read.physical_reg, vb + kSrc);
        EXPECT_EQ(read.byte_mask, amdgpu::sdwa::sdwa_src_byte_mask(test_case.src0_sel));
        observed_read_lanes |= read.lane_mask;
      }
      EXPECT_EQ(observed_read_lanes, kPartialExecMask);
    }
  }
}

TEST(ExecutionPluginTest, SdwaClampIsAppliedInsideArchitecturalDestinationWrite) {
  ForceScalarOverride force_simd(false);
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kSrc, 0, std::bit_cast<uint32_t>(0.5f));
  cu->write_vgpr(vb + kDst, 0, 0xDEADBEEFu);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  uint32_t words[2] = {
      vop1_encode(/*v_rcp_f32 opcode=*/34, kDst, amdgpu::SRC_SDWA),
      vop1_sdwa_word(kSrc, amdgpu::sdwa::DWORD, amdgpu::sdwa::UNUSED_PAD, amdgpu::sdwa::DWORD,
                     /*clamp=*/true),
  };
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  plugin->events.clear();
  cu->execute_instruction(inst.get(), *wf);

  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), std::bit_cast<uint32_t>(1.0f));
  const auto reads = vgpr_read_events(*plugin);
  ASSERT_EQ(reads.size(), 1u);
  EXPECT_EQ(reads[0].physical_reg, vb + kSrc);
  EXPECT_EQ(reads[0].lane_mask, 1u);
  EXPECT_EQ(reads[0].byte_mask, ExecutionPlugin::kFullByteMask);
  const auto writes = vgpr_write_events(*plugin);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].physical_reg, vb + kDst);
  EXPECT_EQ(writes[0].lane_mask, 1u);
  EXPECT_EQ(writes[0].byte_mask, ExecutionPlugin::kFullByteMask);
}

TEST(ExecutionPluginTest, SdwaClampHonorsDx10ClampMode) {
  ForceScalarOverride force_simd(false);
  PluginFixture f(/*num_wf_slots=*/1);
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  constexpr uint32_t kQuietNan = 0x7FC12345u;
  const uint32_t vb = wf->vgpr_alloc().base;

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  uint32_t words[2] = {
      vop1_encode(/*v_rcp_f32 opcode=*/34, kDst, amdgpu::SRC_SDWA),
      vop1_sdwa_word(kSrc, amdgpu::sdwa::DWORD, amdgpu::sdwa::UNUSED_PAD, amdgpu::sdwa::DWORD,
                     /*clamp=*/true),
  };
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);

  struct Case {
    uint32_t mode;
    uint32_t expected;
  };
  constexpr std::array cases{
      Case{0, kQuietNan},
      Case{amdgpu::Wavefront::DX10_CLAMP_BIT, std::bit_cast<uint32_t>(0.0f)},
  };
  for (const Case &test_case : cases) {
    SCOPED_TRACE(test_case.mode);
    wf->set_mode_raw(test_case.mode);
    cu->write_vgpr(vb + kSrc, 0, kQuietNan);
    cu->write_vgpr(vb + kDst, 0, 0xDEADBEEFu);
    cu->execute_instruction(inst.get(), *wf);
    EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), test_case.expected);
  }
}

TEST(ExecutionPluginTest, SdwaPartialPreserveClampReportsFullDwordWrite) {
  ForceScalarOverride force_simd(false);
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1);

  constexpr uint32_t kSrc = 2;
  constexpr uint32_t kDst = 5;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_vgpr(vb + kSrc, 0, std::bit_cast<uint32_t>(0.5f));
  // v_rcp produces 2.0f (0x40000000). BYTE_1 replaces 0xCC with 0x00,
  // yielding 2.0f before clamp; clamp then changes the complete dword to 1.0f.
  cu->write_vgpr(vb + kDst, 0, 0x4000CC00u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  uint32_t words[2] = {
      vop1_encode(/*v_rcp_f32 opcode=*/34, kDst, amdgpu::SRC_SDWA),
      vop1_sdwa_word(kSrc, amdgpu::sdwa::BYTE_1, amdgpu::sdwa::UNUSED_PRESERVE, amdgpu::sdwa::DWORD,
                     /*clamp=*/true),
  };
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  plugin->events.clear();
  cu->execute_instruction(inst.get(), *wf);

  EXPECT_EQ(cu->read_vgpr_storage(vb + kDst, 0), std::bit_cast<uint32_t>(1.0f));
  const auto reads = vgpr_read_events(*plugin);
  ASSERT_EQ(reads.size(), 1u);
  EXPECT_EQ(reads[0].physical_reg, vb + kSrc);
  EXPECT_EQ(reads[0].byte_mask, ExecutionPlugin::kFullByteMask);
  const auto writes = vgpr_write_events(*plugin);
  ASSERT_EQ(writes.size(), 1u);
  EXPECT_EQ(writes[0].physical_reg, vb + kDst);
  EXPECT_EQ(writes[0].lane_mask, 1u);
  EXPECT_EQ(writes[0].byte_mask, ExecutionPlugin::kFullByteMask);
}

TEST(ExecutionPluginTest, MemoryPipelineCompletionDoesNotObserveInstructionWrite) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kAddress = 0x8000;
  constexpr uint32_t kLoadedValue = 0x12345678u;
  constexpr uint32_t kDst = 3;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(&kLoadedValue), sizeof(kLoadedValue),
                    kAddress);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->elem_size = sizeof(kLoadedValue);
  state->num_elems = 1;
  state->is_load = true;
  state->wf_size = wf->wf_size();
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->dst_reg_base = wf->vgpr_alloc().base + kDst;
  state->per_lane_addr[0] = kAddress;

  plugin->events.clear();
  GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_TRUE(vgpr_write_events(*plugin).empty());
  EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + kDst, 0), kLoadedValue);
}

TEST(ExecutionPluginTest, MemoryPipelineCompletionDoesNotCrossWaveVgprBlock) {
  constexpr uint32_t kVgprsPerWave = 16;
  PluginFixture f(/*num_wf_slots=*/2, /*arch=*/"rdna4", /*wavefront_size=*/32,
                  /*sgprs_per_wf=*/128, kVgprsPerWave);
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/128, kVgprsPerWave, 32);
  auto *adjacent = cu->dispatch_wf(1, 0, /*sgprs=*/128, kVgprsPerWave, 32);
  ASSERT_NE(wf, nullptr);
  ASSERT_NE(adjacent, nullptr);
  ASSERT_EQ(adjacent->vgpr_alloc().base, wf->vgpr_alloc().base + kVgprsPerWave);

  constexpr uint64_t kAddress = 0x8100;
  constexpr std::array<uint32_t, 2> kLoaded = {0x11112222u, 0x33334444u};
  constexpr uint32_t kLastSentinel = 0xA5A5A5A5u;
  constexpr uint32_t kAdjacentSentinel = 0x5A5A5A5Au;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(kLoaded.data()), sizeof(kLoaded), kAddress);
  cu->write_vgpr(wf->vgpr_alloc().base + kVgprsPerWave - 1, 0, kLastSentinel);
  cu->write_vgpr(adjacent->vgpr_alloc().base, 0, kAdjacentSentinel);

  auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  state->elem_size = sizeof(uint32_t);
  state->num_elems = 2;
  state->is_load = true;
  state->wf_size = wf->wf_size();
  state->exec_mask = 1;
  state->lane_mask = 1;
  state->dst_reg_base = wf->vgpr_alloc().base + kVgprsPerWave - 1;
  state->per_lane_addr[0] = kAddress;

  GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_EQ(cu->read_vgpr_storage(wf->vgpr_alloc().base + kVgprsPerWave - 1, 0), kLastSentinel);
  EXPECT_EQ(cu->read_vgpr_storage(adjacent->vgpr_alloc().base, 0), kAdjacentSentinel);
}

TEST(ExecutionPluginTest, ScalarMemoryCompletionDoesNotObserveInstructionWrite) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  constexpr uint64_t kAddress = 0x8200;
  constexpr uint32_t kLoadedValue = 0x12345678u;
  constexpr uint32_t kDst = 3;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(&kLoadedValue), sizeof(kLoadedValue),
                    kAddress);

  auto state = std::make_unique<ScalarMemState>();
  state->addr = kAddress;
  state->dst_register = {ScalarRegisterStorage::SGPR, kDst, 1};
  state->num_dwords = 1;
  state->is_load = true;

  plugin->events.clear();
  TestScalarMemPipeline pipeline(&cu->l1_scalar());
  pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

  EXPECT_EQ(cu->read_sgpr_storage(wf->sgpr_alloc().base + kDst), kLoadedValue);
  EXPECT_TRUE(
      std::none_of(plugin->events.begin(), plugin->events.end(),
                   [](const HookEvent &event) { return event.kind == HookEvent::WRITE_SGPR; }));

  constexpr uint64_t kTtmpAddress = 0x8300;
  constexpr uint32_t kTtmpValue = 0xAABBCCDDu;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(&kTtmpValue), sizeof(kTtmpValue),
                    kTtmpAddress);
  auto ttmp_state = std::make_unique<ScalarMemState>();
  ttmp_state->addr = kTtmpAddress;
  ttmp_state->dst_register = {ScalarRegisterStorage::TTMP, 0, 1};
  ttmp_state->num_dwords = 1;
  ttmp_state->is_load = true;
  pipeline.issue(new TestMemoryInstruction(std::move(ttmp_state)), *wf);

  EXPECT_EQ(wf->ttmp(0), kTtmpValue);

  constexpr uint64_t kVccAddress = 0x8400;
  constexpr uint32_t kVccValue = 0x55667788u;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(&kVccValue), sizeof(kVccValue), kVccAddress);
  wf->set_vcc_raw(0xAABBCCDD00000000ull);
  auto vcc_state = std::make_unique<ScalarMemState>();
  vcc_state->addr = kVccAddress;
  vcc_state->dst_register = {ScalarRegisterStorage::VCC, 0, 1};
  vcc_state->num_dwords = 1;
  vcc_state->is_load = true;
  pipeline.issue(new TestMemoryInstruction(std::move(vcc_state)), *wf);

  EXPECT_EQ(wf->vcc(), 0xAABBCCDD55667788ull);
  EXPECT_TRUE(
      std::none_of(plugin->events.begin(), plugin->events.end(),
                   [](const HookEvent &event) { return event.kind == HookEvent::WRITE_SGPR; }));
}

TEST(ExecutionPluginTest, ScalarMemoryRejectsDestinationThatCrossesTtmpFile) {
  PluginFixture f(/*num_wf_slots=*/2);
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  auto *adjacent = cu->dispatch_wf(1, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_NE(adjacent, nullptr);

  constexpr uint32_t kAdjacentOffset = 20;
  constexpr uint32_t kSentinel = 0xA5A5A5A5u;
  cu->write_sgpr(adjacent->sgpr_alloc().base + kAdjacentOffset, kSentinel);
  cu->write_sgpr(wf->sgpr_alloc().base, 0x8000u);
  cu->write_sgpr(wf->sgpr_alloc().base + 1, 0u);

  const auto words = cdna4::build_smem(cdna4::kSLoadDwordx2Smem,
                                       {.sbase = 0, .sdata = amdgpu::kTtmpSelectorLast, .imm = 1});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> load(decode_valid(*decoder, words.data()));
  ASSERT_NE(load, nullptr);

  cu->execute_instruction(load.get(), *wf);

  EXPECT_EQ(load->data(), nullptr);
  EXPECT_EQ(cu->read_sgpr_storage(adjacent->sgpr_alloc().base + kAdjacentOffset), kSentinel);
}

TEST(ExecutionPluginTest, MemoryPipelinesSnapshotDispatchIdentityAtIssue) {
  PluginFixture f(/*num_wf_slots=*/4);
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf_at(/*wf_id=*/3, /*wg_id=*/17, /*pc=*/0, /*num_sgprs=*/104,
                                /*num_vgprs=*/256);
  ASSERT_NE(wf, nullptr);

  auto global_state = std::make_unique<VectorMemState>(GLOBAL_MEM);
  global_state->wg_id = 101;
  global_state->wf_id = 202;
  TestMemoryInstruction global_inst(std::move(global_state));
  TestGlobalMemPipeline global_pipeline(&cu->l1_vector(), cu->l2());
  global_pipeline.initiate_access(global_inst, *wf);
  EXPECT_EQ(global_inst.data_as<VectorMemState>()->wg_id, 17u);
  EXPECT_EQ(global_inst.data_as<VectorMemState>()->wf_id, 3u);

  auto local_state = std::make_unique<VectorMemState>(LOCAL_MEM);
  local_state->wg_id = 101;
  local_state->wf_id = 202;
  TestMemoryInstruction local_inst(std::move(local_state));
  TestLocalMemPipeline local_pipeline;
  local_pipeline.initiate_access(local_inst, *wf);
  EXPECT_EQ(local_inst.data_as<VectorMemState>()->wg_id, 17u);
  EXPECT_EQ(local_inst.data_as<VectorMemState>()->wf_id, 3u);
}

TEST(ExecutionPluginTest, D16MemoryCompletionPreservesHalfWithoutObservation) {
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_FALSE(cu->sram_ecc());

  constexpr uint64_t kAddress = 0x8100;
  constexpr uint16_t kLoadedValue = 0x1122u;
  constexpr uint32_t kOldDst = 0xAABBCCDDu;
  constexpr uint32_t kDst = 4;
  const uint32_t physical_dst = wf->vgpr_alloc().base + kDst;
  f.mem->load_image(reinterpret_cast<const uint8_t *>(&kLoadedValue), sizeof(kLoadedValue),
                    kAddress);

  GlobalMemPipeline pipeline(&cu->l1_vector(), cu->l2());
  for (bool high_half : {false, true}) {
    SCOPED_TRACE(high_half ? "high half" : "low half");
    cu->write_vgpr(physical_dst, 0, kOldDst);

    auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
    state->elem_size = sizeof(kLoadedValue);
    state->num_elems = 1;
    state->is_load = true;
    state->wf_size = wf->wf_size();
    state->exec_mask = 1;
    state->lane_mask = 1;
    state->dst_reg_base = physical_dst;
    state->per_lane_addr[0] = kAddress;
    state->d16_lo = !high_half;
    state->d16_hi = high_half;

    plugin->events.clear();
    pipeline.issue(new TestMemoryInstruction(std::move(state)), *wf);

    EXPECT_TRUE(vgpr_read_events(*plugin).empty());
    EXPECT_TRUE(vgpr_write_events(*plugin).empty());
    const uint32_t expected = high_half ? 0x1122CCDDu : 0xAABB1122u;
    EXPECT_EQ(cu->read_vgpr_storage(physical_dst, 0), expected);
  }
}

TEST(RaceDetectorPluginTest, D16LoadTracksFullDwordWhenSramEccEnabled) {
  auto opposite_half_read_reports_race = [](std::string_view arch, uint32_t wavefront_size) {
    PluginFixture f(/*num_wf_slots=*/1, arch, wavefront_size);
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    if (!f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>())) {
      ADD_FAILURE() << "failed to attach race detector";
      return false;
    }
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
    EXPECT_NE(wf, nullptr);
    if (wf == nullptr)
      return false;
    wf->set_exec(1u);
    std::array<amdgpu::Wavefront *, 1> waves{wf};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(
        /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256, /*sgpr_count=*/104, waves);

    constexpr uint32_t kDst = 4;
    auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
    state->elem_size = 1;
    state->num_elems = 1;
    state->is_load = true;
    state->wf_size = wf->wf_size();
    state->exec_mask = 1;
    state->lane_mask = 1;
    state->dst_reg_base = wf->vgpr_alloc().base + kDst;
    state->d16_lo = true;
    TestMemoryInstruction load(std::move(state));
    f.plugin_group_->onAmdgpuRouteMemoryInstruction(load, *wf);

    f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kDst, /*lane_mask=*/1,
                                           ExecutionPlugin::kHighHalfByteMask);
    return sink.str().find("RACE ") != std::string::npos;
  };

  EXPECT_TRUE(opposite_half_read_reports_race("cdna4", /*wavefront_size=*/64));
  EXPECT_TRUE(opposite_half_read_reports_race("cdna5", /*wavefront_size=*/32));
  EXPECT_FALSE(opposite_half_read_reports_race("rdna4", /*wavefront_size=*/32));
}

TEST(RaceDetectorPluginTest, ScalarLoadToTtmpReportsReadBeforeWait) {
  PluginFixture f(/*num_wf_slots=*/1);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/104, waves);

  constexpr uint32_t kTtmp0Selector = amdgpu::kTtmpSelectorFirst;
  const auto words =
      cdna4::build_smem(cdna4::kSLoadDwordSmem, {.sbase = 0, .sdata = kTtmp0Selector, .imm = 1});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> load(decode_valid(*decoder, words.data()));
  ASSERT_NE(load, nullptr);
  ASSERT_NE(load->dst_operand(0), nullptr);
  EXPECT_EQ(load->dst_operand(0)->to_register_ref(), (RegisterRef{RegClass::TTMP, 0, 1}));

  cu->write_sgpr(wf->sgpr_alloc().base, 0x1000u);
  cu->write_sgpr(wf->sgpr_alloc().base + 1, 0u);
  cu->execute_instruction(load.get(), *wf);
  ASSERT_NE(load->data(), nullptr);
  f.plugin_group_->onAmdgpuRouteMemoryInstruction(*load, *wf);

  static_cast<void>(RegisterAccess(*wf).read_scalar(*load->dst_operand(0)));
  EXPECT_NE(sink.str().find("type=TTMP"), std::string::npos);
}

TEST(RaceDetectorPluginTest, ScalarLoadToTtmpHonorsSplitKmcntWait) {
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"rdna4", /*wavefront_size=*/32,
                  /*sgprs_per_wf=*/128);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/128, /*vgprs=*/256, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/128, waves);

  auto state = std::make_unique<ScalarMemState>();
  state->dst_register = {ScalarRegisterStorage::TTMP, 0, 1};
  state->is_load = true;
  state->wait_counter_type = WaitCounterType::KMCNT;
  TestMemoryInstruction load(std::move(state));
  f.plugin_group_->onAmdgpuRouteMemoryInstruction(load, *wf);

  wf->set_wait_target_kmcnt(0);
  TestWaitcntInstruction wait("s_wait_kmcnt");
  f.plugin_group_->onAmdgpuAfterExecuteInstruction(/*pc=*/4, wait, *wf);
  static_cast<void>(RegisterAccess(*wf).read_ttmp(0));
  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

TEST(RaceDetectorPluginTest, NamedVmcntWaitRetiresMonolithicAndSplitLoadEvents) {
  auto run = [](WaitCounterType event_counter) {
    PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"rdna3", /*wavefront_size=*/32,
                    /*sgprs_per_wf=*/128);
    PluginSinkConfig sink_config;
    StringSink &sink = sink_config.emplace<StringSink>();
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
    EXPECT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/128, /*vgprs=*/256, 32);
    EXPECT_NE(wf, nullptr);
    if (!wf)
      return false;
    wf->set_exec(1u);
    std::array<amdgpu::Wavefront *, 1> waves{wf};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(
        /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
        /*physical_sgpr_count=*/128, waves);

    auto state = std::make_unique<VectorMemState>(GLOBAL_MEM);
    state->is_load = true;
    state->num_elems = 1;
    state->dst_reg_base = wf->vgpr_alloc().base;
    state->exec_mask = 1;
    state->wait_counter_type = event_counter;
    TestMemoryInstruction load(std::move(state));
    f.plugin_group_->onAmdgpuRouteMemoryInstruction(load, *wf);

    wf->set_wait_target_loadcnt(0);
    TestWaitcntInstruction wait("s_waitcnt_vmcnt");
    f.plugin_group_->onAmdgpuAfterExecuteInstruction(/*pc=*/4, wait, *wf);
    f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base, /*lane_mask=*/1);
    return sink.str().find("RACE ") == std::string::npos;
  };

  EXPECT_TRUE(run(WaitCounterType::VMCNT));
  EXPECT_TRUE(run(WaitCounterType::LOADCNT));
}

TEST(RaceDetectorPluginTest, NamedLgkmcntWaitRetiresSplitScalarEvent) {
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"rdna3", /*wavefront_size=*/32,
                  /*sgprs_per_wf=*/128);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/128, /*vgprs=*/256, 32);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(
      /*dispatch_id=*/1, /*wg_id=*/0, /*physical_vgpr_count=*/256,
      /*physical_sgpr_count=*/128, waves);

  auto state = std::make_unique<ScalarMemState>();
  state->dst_register = {ScalarRegisterStorage::TTMP, 0, 1};
  state->is_load = true;
  state->wait_counter_type = WaitCounterType::KMCNT;
  TestMemoryInstruction load(std::move(state));
  f.plugin_group_->onAmdgpuRouteMemoryInstruction(load, *wf);

  const auto current_wait = wf->wait_target();
  wf->set_wait_target(current_wait.vmcnt, 0, current_wait.expcnt);
  TestWaitcntInstruction wait("s_waitcnt_lgkmcnt");
  f.plugin_group_->onAmdgpuAfterExecuteInstruction(/*pc=*/4, wait, *wf);
  static_cast<void>(RegisterAccess(*wf).read_ttmp(0));
  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

TEST(ExecutionPluginTest, F64SourceReadObservationReportsBothHalves) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    ForceScalarOverride force_simd(false);
    PluginFixture f(/*num_wf_slots=*/1);
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kPartialExecMask);

    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vb + 0, lane, 0x0000'0000u);
      cu->write_vgpr(vb + 1, lane, 0x3ff0'0000u);
      cu->write_vgpr(vb + 2, lane, 0x0000'0000u);
      cu->write_vgpr(vb + 3, lane, 0x4000'0000u);
      cu->write_vgpr(vb + 4, lane, 0x0000'0000u);
      cu->write_vgpr(vb + 5, lane, 0x0000'0000u);
    }

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    uint32_t words[4] = {vop2_encode(/*opcode=*/4, /*vdst=*/4, /*vsrc1=*/2, /*src0=*/256), 0u, 0u,
                         0u};
    Instruction *inst = decode_valid(*decoder, words);
    ASSERT_NE(inst, nullptr);
    cu->execute_instruction(inst, *wf);
    delete inst;

    expect_vgpr_read_set(vgpr_read_events(*plugin), vb, {0, 1, 2, 3, 4, 5}, kPartialExecMask);
  }
}

TEST(ExecutionPluginTest, Vop3FmacSimdReadObservationReportsAccumulator) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "<experimental/simd> unavailable";
    return;
  } else {
    ForceScalarOverride force_simd(false);
    PluginFixture f(/*num_wf_slots=*/1);
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(kPartialExecMask);

    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      cu->write_vgpr(vb + 0, lane, 0x3f80'0000u);
      cu->write_vgpr(vb + 1, lane, 0x4000'0000u);
      cu->write_vgpr(vb + 4, lane, 0x0000'0000u);
    }

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    uint32_t words[4] = {0u, 0u, 0u, 0u};
    vop3_encode(/*opcode=*/315, /*vdst=*/4, /*src0=*/256, /*src1=*/257, words);
    Instruction *inst = decode_valid(*decoder, words);
    ASSERT_NE(inst, nullptr);
    cu->execute_instruction(inst, *wf);
    delete inst;

    expect_vgpr_read_set(vgpr_read_events(*plugin), vb, {0, 1, 4}, kPartialExecMask);
  }
}

TEST(ExecutionPluginTest, MfmaReadObservationUsesLaneMasks) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);

  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint32_t S0 = 0, S1 = 16, ACC = 32;

  amdgpu::observe_mfma_fast_path_reads(*cu, vb + S0, vb + S1, vb + ACC, /*M=*/16, /*N=*/16,
                                       /*K=*/32, /*B=*/1, /*data_bits=*/16, amdgpu::ACC_FROM_VGPR,
                                       wf->wf_size());

  expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                       {S0 + 0, S0 + 1, S0 + 2, S0 + 3, S1 + 0, S1 + 1, S1 + 2, S1 + 3, ACC + 0,
                        ACC + 1, ACC + 2, ACC + 3},
                       ~uint64_t{0});
}

TEST(ExecutionPluginTest, MfmaReadObservationSkipsConstantAccumulator) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);

  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint32_t S0 = 0, S1 = 16, ACC = 32;

  amdgpu::observe_mfma_fast_path_reads(*cu, vb + S0, vb + S1, vb + ACC, /*M=*/16, /*N=*/16,
                                       /*K=*/32, /*B=*/1, /*data_bits=*/16,
                                       /*const_acc=*/0, wf->wf_size());

  expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                       {S0 + 0, S0 + 1, S0 + 2, S0 + 3, S1 + 0, S1 + 1, S1 + 2, S1 + 3},
                       ~uint64_t{0});
}

TEST(ExecutionPluginTest, Cdna4BlockScaleMfmaAbidZeroPrefixRejectsBeforeExecuteCallbacks) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  const auto scale = make_cdna4_mfma_scale_words(45, /*abid=*/0, vgpr_src(96), vgpr_src(97));
  const std::array<uint32_t, 5> code = {scale[0], scale[1], scale[2], scale[3], S_ENDPGM};

  f.run_kernel(code.data(), code.size());

  size_t before_count = 0;
  for (const auto &event : plugin->events)
    if (event.kind == HookEvent::BEFORE_INSTRUCTION)
      ++before_count;
  EXPECT_EQ(before_count, 0u);
}

TEST(ExecutionPluginTest, Cdna4BlockScaleMfmaCompoundAdvancesCallbacksByFullPrefixSize) {
  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  const auto scale = make_cdna4_mfma_scale_words(45, /*abid=*/1, vgpr_src(96), vgpr_src(97));
  const std::array<uint32_t, 5> code = {scale[0], scale[1], scale[2], scale[3], S_ENDPGM};

  f.run_kernel(code.data(), code.size());

  std::vector<HookEvent> before_events;
  for (const auto &event : plugin->events)
    if (event.kind == HookEvent::BEFORE_INSTRUCTION)
      before_events.push_back(event);
  ASSERT_EQ(before_events.size(), 2u);
  EXPECT_EQ(before_events[0].mnemonic, "v_mfma_scale_f32_16x16x128_f8f6f4");
  EXPECT_EQ(before_events[1].mnemonic, "s_endpgm");
  ASSERT_GE(before_events[1].pc, before_events[0].pc);
  EXPECT_EQ(before_events[1].pc - before_events[0].pc, 16u);
}

TEST(ExecutionPluginTest, Cdna4BlockScaleMfmaScaleReadsUseSelectedByteMask) {
  constexpr uint32_t kSrc0 = 0;
  constexpr uint32_t kSrc1 = 16;
  constexpr uint32_t kAccumulator = 32;
  constexpr uint32_t kScaleA = 96;
  constexpr uint32_t kScaleB = 97;

  for (uint32_t byte = 0; byte < 4; ++byte) {
    PluginFixture f(/*num_wf_slots=*/1);
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    wf->set_exec(~uint64_t{0});

    const uint32_t vb = wf->vgpr_alloc().base;
    for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
      for (uint32_t reg = 0; reg < 8; ++reg) {
        cu->write_vgpr(vb + kSrc0 + reg, lane, 0x2222'2222u);
        cu->write_vgpr(vb + kSrc1 + reg, lane, 0x2222'2222u);
      }
      for (uint32_t reg = 0; reg < 4; ++reg)
        cu->write_vgpr(vb + kAccumulator + reg, lane, 0);
      cu->write_vgpr(vb + kScaleA, lane, 0x7F7F'7F7Fu);
      cu->write_vgpr(vb + kScaleB, lane, 0x7F7F'7F7Fu);
    }

    auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    const auto words = make_cdna4_mfma_scale_words(45, /*abid=*/1, vgpr_src(kScaleA),
                                                   vgpr_src(kScaleB), byte, byte);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    cu->execute_instruction(inst.get(), *wf);

    const uint8_t expected_mask = static_cast<uint8_t>(1u << byte);
    bool saw_scale_a = false;
    bool saw_scale_b = false;
    std::map<uint32_t, std::pair<uint64_t, uint8_t>> read_unions;
    const auto reads = vgpr_read_events(*plugin);
    for (const auto &event : reads) {
      auto &[lane_mask, byte_mask] = read_unions[event.physical_reg];
      lane_mask |= event.lane_mask;
      byte_mask |= event.byte_mask;
      if (event.physical_reg == vb + kScaleA) {
        EXPECT_EQ(event.byte_mask, expected_mask) << "A byte=" << byte;
        saw_scale_a = true;
      } else if (event.physical_reg == vb + kScaleB) {
        EXPECT_EQ(event.byte_mask, expected_mask) << "B byte=" << byte;
        saw_scale_b = true;
      }
    }
    EXPECT_TRUE(saw_scale_a) << "byte=" << byte;
    EXPECT_TRUE(saw_scale_b) << "byte=" << byte;

    std::map<uint32_t, std::pair<uint64_t, uint8_t>> expected_unions;
    for (uint32_t reg :
         {kSrc0, kSrc0 + 1, kSrc0 + 2, kSrc0 + 3, kSrc1, kSrc1 + 1, kSrc1 + 2, kSrc1 + 3,
          kAccumulator, kAccumulator + 1, kAccumulator + 2, kAccumulator + 3})
      expected_unions.emplace(vb + reg, std::pair{~uint64_t{0}, ExecutionPlugin::kFullByteMask});
    expected_unions.emplace(vb + kScaleA, std::pair{~uint64_t{0}, expected_mask});
    expected_unions.emplace(vb + kScaleB, std::pair{~uint64_t{0}, expected_mask});
    EXPECT_EQ(read_unions, expected_unions) << "byte=" << byte;
  }
}

TEST(ExecutionPluginTest, Cdna4BlockScaleMfmaInlineScalesDoNotReadVgprs) {
  constexpr uint32_t kSrc0 = 0;
  constexpr uint32_t kSrc1 = 16;
  constexpr uint32_t kAccumulator = 32;

  PluginFixture f(/*num_wf_slots=*/1);
  auto *plugin = f.attach_ordering_plugin();
  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(~uint64_t{0});

  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    for (uint32_t reg = 0; reg < 4; ++reg) {
      cu->write_vgpr(vb + kSrc0 + reg, lane, 0x2222'2222u);
      cu->write_vgpr(vb + kSrc1 + reg, lane, 0x2222'2222u);
      cu->write_vgpr(vb + kAccumulator + reg, lane, 0);
    }
  }

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  const auto words = make_cdna4_mfma_scale_words(45, /*abid=*/1, /*+1.0f=*/242,
                                                 /*1/(2*pi)=*/248);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  cu->execute_instruction(inst.get(), *wf);

  std::set<uint32_t> actual_regs;
  for (const auto &event : vgpr_read_events(*plugin))
    actual_regs.insert(event.physical_reg - vb);
  std::set<uint32_t> expected_regs;
  for (uint32_t reg :
       {kSrc0, kSrc0 + 1, kSrc0 + 2, kSrc0 + 3, kSrc1, kSrc1 + 1, kSrc1 + 2, kSrc1 + 3,
        kAccumulator, kAccumulator + 1, kAccumulator + 2, kAccumulator + 3})
    expected_regs.insert(reg);
  EXPECT_EQ(actual_regs, expected_regs);
}

TEST(ExecutionPluginTest, WmmaReadObservationUsesWave32RegisterSet) {
  Wave32PluginFixture f;
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint32_t S0 = 0, S1 = 16, ACC = 32;

  amdgpu::observe_wmma_fast_path_reads(*f.cu, vb + S0, vb + S1, vb + ACC, /*M=*/16, /*N=*/16,
                                       /*K=*/32, /*data_bits=*/16, /*acc_bits=*/32,
                                       amdgpu::ACC_FROM_VGPR, wf->wf_size());

  expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                       {S0 + 0,  S0 + 1,  S0 + 2,  S0 + 3,  S0 + 4,  S0 + 5,  S0 + 6,  S0 + 7,
                        S1 + 0,  S1 + 1,  S1 + 2,  S1 + 3,  S1 + 4,  S1 + 5,  S1 + 6,  S1 + 7,
                        ACC + 0, ACC + 1, ACC + 2, ACC + 3, ACC + 4, ACC + 5, ACC + 6, ACC + 7},
                       0xFFFF'FFFFu);
}

TEST(ExecutionPluginTest, WmmaReadObservationSkipsConstantAccumulator) {
  Wave32PluginFixture f;
  auto *plugin = f.attach_ordering_plugin();
  auto *wf = f.cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 32u);

  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint32_t S0 = 0, S1 = 16, ACC = 32;

  amdgpu::observe_wmma_fast_path_reads(*f.cu, vb + S0, vb + S1, vb + ACC, /*M=*/16, /*N=*/16,
                                       /*K=*/32, /*data_bits=*/16, /*acc_bits=*/32,
                                       /*const_acc=*/0, wf->wf_size());

  expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                       {S0 + 0, S0 + 1, S0 + 2, S0 + 3, S0 + 4, S0 + 5, S0 + 6, S0 + 7, S1 + 0,
                        S1 + 1, S1 + 2, S1 + 3, S1 + 4, S1 + 5, S1 + 6, S1 + 7},
                       0xFFFF'FFFFu);
}

static_assert(!amdgpu::wmma_f32_f32_native_width_supported(16, 1));
static_assert(amdgpu::wmma_f32_f32_native_width_supported(16, 4));
static_assert(amdgpu::wmma_f32_f32_native_width_supported(16, 8));
static_assert(amdgpu::wmma_f32_f32_native_width_supported(16, 16));
static_assert(!amdgpu::wmma_f32_f32_native_width_supported(16, 32));

TEST(ExecutionPluginTest, WmmaF32NativeWidthFastPathUsesRegionReads) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "stdx SIMD is unavailable";
  } else {
    constexpr uint32_t M = 16, N = 16, K = 4;
    constexpr uint32_t width = static_cast<uint32_t>(util::native<float>::size());
    if (!amdgpu::wmma_f32_f32_native_width_supported(N, width))
      GTEST_SKIP() << "f32 WMMA shape is not divisible by the native SIMD width";

    ForceScalarOverride force_simd(false);
    Wave32PluginFixture f;
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu.get();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    ASSERT_EQ(wf->wf_size(), 32u);

    const uint32_t vb = wf->vgpr_alloc().base;
    constexpr uint32_t S0 = 0, S1 = 16, ACC = 32, DST = 48;
    for (uint32_t reg = 0; reg < 64; ++reg)
      for (uint32_t lane = 0; lane < 32; ++lane)
        cu->write_vgpr(vb + reg, lane, 0x3f80'0000u);

    amdgpu::exec_wmma_f32_f32_spec<M, N, K>(*cu, vb + DST, vb + S0, vb + S1, vb + ACC,
                                            amdgpu::ACC_FROM_VGPR);

    expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                         {S0 + 0, S0 + 1, S1 + 0, S1 + 1, ACC + 0, ACC + 1, ACC + 2, ACC + 3,
                          ACC + 4, ACC + 5, ACC + 6, ACC + 7},
                         0xFFFF'FFFFu);
  }
}

TEST(ExecutionPluginTest, MfmaReadObservationReportsRace) {
  PluginFixture f(/*num_wf_slots=*/1);
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  auto plugin = std::make_unique<MfmaRacePlugin>();
  auto *race_plugin = plugin.get();
  f.plugin_group_->add(std::move(plugin));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  ASSERT_EQ(wf->wf_size(), 64u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0,
                                               /*physical_vgpr_count=*/256,
                                               /*physical_sgpr_count=*/104, waves);

  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint32_t S0 = 0;

  race_plugin->registerOutstandingLoad(S0, /*exec_mask=*/1u);
  amdgpu::observe_mfma_input_reads(*cu, vb + S0, /*dim=*/16, /*K=*/32, /*B=*/1,
                                   /*data_bits=*/16, wf->wf_size());

  ASSERT_FALSE(race_plugin->violations.empty());
  const auto &violation = race_plugin->violations.front();
  EXPECT_EQ(violation.space, RaceViolation::Space::VGPR);
  EXPECT_EQ(violation.index, static_cast<int>(S0));
  EXPECT_EQ(violation.lane, 0);
}

TEST(ExecutionPluginTest, MfmaF32NativeWidthFastPathUsesRegionReads) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "stdx SIMD is unavailable";
  } else {
    constexpr uint32_t M = 16, N = 16, K = 4, B = 1;
    constexpr uint32_t width = static_cast<uint32_t>(util::native<float>::size());
    if (width <= 1 || N % width != 0)
      GTEST_SKIP() << "f32 MFMA shape is not divisible by the native SIMD width";

    ForceScalarOverride force_simd(false);
    PluginFixture f(/*num_wf_slots=*/1);
    auto *plugin = f.attach_ordering_plugin();
    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    ASSERT_EQ(wf->wf_size(), 64u);

    const uint32_t vb = wf->vgpr_alloc().base;
    constexpr uint32_t S0 = 0, S1 = 16, ACC = 32, DST = 48;
    for (uint32_t reg = 0; reg < 64; ++reg)
      for (uint32_t lane = 0; lane < 64; ++lane)
        cu->write_vgpr(vb + reg, lane, 0x3f80'0000u);

    amdgpu::exec_f32_mfma_f32_spec<M, N, K, B>(*cu, vb + DST, vb + S0, vb + S1, vb + ACC,
                                               amdgpu::ACC_FROM_VGPR,
                                               /*cbsz=*/0, /*abid=*/0, /*blgp=*/0);

    expect_vgpr_read_set(vgpr_read_events(*plugin), vb,
                         {S0, S1, ACC + 0, ACC + 1, ACC + 2, ACC + 3}, ~uint64_t{0});
  }
}

TEST(ExecutionPluginTest, MfmaFastPathReadHookReportsRace) {
  if constexpr (!util::has_stdx_simd) {
    GTEST_SKIP() << "stdx SIMD is unavailable";
  } else {
    if (util::native<float>::size() != 16)
      GTEST_SKIP() << "MFMA fast path requires 16-lane native<float>";

    struct ForceScalarGuard {
      bool old = util::force_scalar();
      ~ForceScalarGuard() { util::set_force_scalar_for_testing(old); }
    } force_scalar_guard;
    util::set_force_scalar_for_testing(false);

    PluginFixture f(/*num_wf_slots=*/1);
    f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto plugin = std::make_unique<MfmaRacePlugin>();
    auto *race_plugin = plugin.get();
    f.plugin_group_->add(std::move(plugin));
    f.soc->set_plugin_group(f.plugin_group_);
    f.plugin_group_->onInit();

    auto *cu = f.cu();
    auto *wf = cu->dispatch_wf(0, 0, /*sgprs=*/104, /*vgprs=*/256);
    ASSERT_NE(wf, nullptr);
    ASSERT_EQ(wf->wf_size(), 64u);
    std::array<amdgpu::Wavefront *, 1> waves{wf};
    f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0,
                                                 /*physical_vgpr_count=*/256,
                                                 /*physical_sgpr_count=*/104, waves);

    const uint32_t vb = wf->vgpr_alloc().base;
    constexpr uint32_t S0 = 0, S1 = 16, ACC = 32, DST = 48;
    for (uint32_t reg = 0; reg < 64; ++reg)
      for (uint32_t lane = 0; lane < 64; ++lane)
        cu->write_vgpr(vb + reg, lane, 0x3c003c00u);

    race_plugin->registerOutstandingLoad(S0, /*exec_mask=*/1u);
    amdgpu::exec_f32_mfma_f16_spec<16, 16, 32>(*cu, vb + DST, vb + S0, vb + S1, vb + ACC,
                                               amdgpu::ACC_FROM_VGPR, /*cbsz=*/0, /*abid=*/0,
                                               /*blgp=*/0);

    ASSERT_FALSE(race_plugin->violations.empty());
    const auto &violation = race_plugin->violations.front();
    EXPECT_EQ(violation.space, RaceViolation::Space::VGPR);
    EXPECT_EQ(violation.index, static_cast<int>(S0));
    EXPECT_EQ(violation.lane, 0);
  }
}

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
  f.mem->register_process(process_id, &process.page_table_, &process.page_table_mutex_);
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
  amdgpu::HwQueue queue{};
  queue.queue_id = queue_id;
  queue.process_id = process_id;
  queue.ring_base_va = ring_va;
  queue.ring_size = static_cast<uint32_t>(ring.size());
  queue.read_ptr_va = read_ptr_va;
  queue.write_ptr_va = write_ptr_va;
  queue.doorbell_base = &doorbell;
  queue.host_accessible = true;
  f.cp()->register_queue(std::move(queue));
  f.cp()->engine()->schedule_event_now(f.cp()->doorbell_event());
  f.run_until_idle();

  auto it = std::find_if(plugin->events.begin(), plugin->events.end(), [](const HookEvent &event) {
    return event.kind == HookEvent::DISPATCH_PACKET_PROCESSED;
  });
  bool found_dispatch = it != plugin->events.end();
  std::string kernel_name = found_dispatch ? it->kernel_name : "";
  std::string kernel_symbol = found_dispatch ? it->kernel_symbol : "";

  f.cp()->unregister_queue(queue_id, process_id);
  f.shutdown();
  f.mem->unregister_process(process_id);

  ASSERT_TRUE(found_dispatch);
  EXPECT_EQ(kernel_name, "vmid_dispatch_kernel");
  EXPECT_EQ(kernel_symbol, "vmid_dispatch_kernel");
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

TEST(HookOrderingTest, WorkgroupDispatchedReportsPhysicalRegisterBlockSizes) {
  PluginFixture f;
  auto *p = f.attach_ordering_plugin();
  const uint32_t code[] = {S_ENDPGM};
  constexpr uint32_t kGranulatedSgprCount = 3; // Requests 32 SGPRs on CDNA4.
  f.run_kernel(code, 1, /*grid=*/64, /*workgroup=*/64, kGranulatedSgprCount);
  f.shutdown();

  auto it = std::find_if(p->events.begin(), p->events.end(), [](const HookEvent &e) {
    return e.kind == HookEvent::WORKGROUP_DISPATCHED;
  });
  ASSERT_NE(it, p->events.end());
  EXPECT_EQ(it->physical_vgpr_count, f.cu()->vgpr_allocation_block_size());
  EXPECT_GT(it->physical_vgpr_count, f.cu()->config().vgprs_per_wf);
  EXPECT_EQ(it->physical_sgpr_count, f.cu()->sgpr_allocation_block_size());
  EXPECT_GT(it->physical_sgpr_count, 32u);
}

// The immediate-halt branch frees a wave's registers the instant s_endpgm
// executes, so instruction hooks must not read the slot afterward. Concretely:
// the terminator must fire BEFORE_INSTRUCTION (it is fetched and decoded) but NOT
// AFTER_INSTRUCTION (there is no live slot to observe once it halts+frees), while
// every non-terminator retains a matched BEFORE/AFTER pair. This pins the guard
// that prevents hooks/logging from touching a freed register slot.
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

// -- Race trace tests --------------------------------------------------------

TEST(FindConflictTest, UsesRecordedConflictingEvent) {
  RaceDetector detector(/*nWaves=*/1, /*vgprCount=*/4, /*sgprCount=*/4, Dim3d(0),
                        [](RaceViolation) {});
  EventId first = detector.allocateEventId(WaveId{0}, /*pc=*/0x100, MemoryEventType::GLOBAL_TO_VGPR,
                                           {2}, /*execMask=*/1);
  EventId second = detector.allocateEventId(WaveId{0}, /*pc=*/0x200,
                                            MemoryEventType::GLOBAL_TO_VGPR, {2}, /*execMask=*/1);
  ASSERT_NE(first, second);

  RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, true, Dim3d(0), second};
  MarkedPc conflict = findConflict(violation, detector);

  EXPECT_EQ(conflict.pc, 0x200u);
}

TEST(FindConflictTest, RejectsUnavailableConflictingEvent) {
  RaceDetector detector(/*nWaves=*/1, /*vgprCount=*/4, /*sgprCount=*/4, Dim3d(0),
                        [](RaceViolation) {});
  RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, true, Dim3d(0), EventId{}};

  EXPECT_THROW(findConflict(violation, detector), std::out_of_range);
}

TEST(DecorateExceptionTest, UsesRecordedConflictingEvent) {
  RaceDetector detector(/*nWaves=*/1, /*vgprCount=*/4, /*sgprCount=*/4, Dim3d(0),
                        [](RaceViolation) {});
  EventId first = detector.allocateEventId(WaveId{0}, /*pc=*/10, MemoryEventType::GLOBAL_TO_VGPR,
                                           {2}, /*execMask=*/1);
  EventId second = detector.allocateEventId(WaveId{0}, /*pc=*/20, MemoryEventType::GLOBAL_TO_VGPR,
                                            {2}, /*execMask=*/1);
  ASSERT_NE(first, second);

  RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, false, Dim3d(0), second};
  std::vector<std::string> source_lines(64, "instruction");
  std::string report = detector.decorateException(
      violation, /*wavePc=*/30, static_cast<int>(source_lines.size()),
      [&](int line) -> std::string_view { return source_lines.at(static_cast<size_t>(line)); });

  EXPECT_NE(report.find("20 --> |"), std::string::npos);
  EXPECT_NE(report.find("30 --> |"), std::string::npos);
  EXPECT_EQ(report.find("10 --> |"), std::string::npos);
}

auto make_trace(std::initializer_list<uint64_t> pcs) {
  plugins::race_detector::RingBuffer<uint64_t, 256> rb;
  for (auto pc : pcs)
    rb.push(pc);
  return rb;
}

std::vector<uint8_t> make_loaded_kernel_symbol_elf(uint64_t kernel_descriptor_offset,
                                                   std::string_view symbol_name) {
  constexpr uint64_t dyn_offset = 0x100;
  constexpr uint64_t symtab_offset = 0x200;
  constexpr uint64_t strtab_offset = 0x300;
  constexpr uint64_t hash_offset = 0x380;
  constexpr uint64_t text_offset = 0x900;

  std::vector<uint8_t> image(4096, 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = 1;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  Elf64_Phdr phdr{};
  phdr.p_type = PT_DYNAMIC;
  phdr.p_vaddr = dyn_offset;
  phdr.p_memsz = 5 * sizeof(Elf64_Dyn);
  std::memcpy(image.data() + ehdr.e_phoff, &phdr, sizeof(phdr));

  auto *dyn = reinterpret_cast<Elf64_Dyn *>(image.data() + dyn_offset);
  dyn[0].d_tag = DT_SYMTAB;
  dyn[0].d_un.d_val = symtab_offset;
  dyn[1].d_tag = DT_STRTAB;
  dyn[1].d_un.d_val = strtab_offset;
  dyn[2].d_tag = DT_STRSZ;
  dyn[2].d_un.d_val = symbol_name.size() + 2;
  dyn[3].d_tag = DT_HASH;
  dyn[3].d_un.d_val = hash_offset;
  dyn[4].d_tag = DT_NULL;

  image[strtab_offset] = '\0';
  std::memcpy(image.data() + strtab_offset + 1, symbol_name.data(), symbol_name.size());

  auto *sym = reinterpret_cast<Elf64_Sym *>(image.data() + symtab_offset);
  sym[1].st_name = 1;
  sym[1].st_value = kernel_descriptor_offset;

  auto *hash = reinterpret_cast<uint32_t *>(image.data() + hash_offset);
  hash[1] = 2; // nchain: null symbol + kernel descriptor symbol.

  using namespace rocr::llvm::amdhsa;
  kernel_descriptor_t kd{};
  kd.kernel_code_entry_byte_offset = text_offset - kernel_descriptor_offset;
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 31);
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 12);
  AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
  std::memcpy(image.data() + kernel_descriptor_offset, &kd, sizeof(kd));

  const std::array<uint32_t, 2> code = {S_NOP, S_ENDPGM};
  std::memcpy(image.data() + text_offset, code.data(), code.size() * sizeof(code[0]));

  return image;
}

TEST(FormatTraceTest, WaveLaneAnnotations) {
  auto trace = make_trace({0x100, 0x108, 0x10c});
  std::unordered_map<uint64_t, std::string> disasm = {
      {0x100, "ds_write_b32 v9, v12"},
      {0x108, "s_nop 0"},
      {0x10c, "ds_read_b32 v8, v9"},
  };
  plugins::race_detector::MarkedPc conflict{0x100, 3, -1};
  plugins::race_detector::MarkedPc read{0x10c, 0, 5};
  auto result = formatTrace(trace, disasm, conflict, read);
  EXPECT_NE(result.find("; <-- wave 3"), std::string::npos);
  EXPECT_NE(result.find("; <-- wave 0 lane 5"), std::string::npos);
}

TEST(FormatTraceTest, NoLineBeforeFirstMarker) {
  auto trace = make_trace({0x100, 0x104, 0x108, 0x10c, 0x110});
  std::unordered_map<uint64_t, std::string> disasm = {
      {0x100, "s_nop 0"},
      {0x104, "s_nop 0"},
      {0x108, "ds_write_b32 v9, v12"},
      {0x10c, "s_nop 0"},
      {0x110, "ds_read_b32 v8, v9"},
  };
  plugins::race_detector::MarkedPc conflict{0x108, 2, -1};
  plugins::race_detector::MarkedPc read{0x110, 1, 3};
  auto result = formatTrace(trace, disasm, conflict, read);
  EXPECT_EQ(result.substr(0, 5), "  ==>");
  EXPECT_EQ(result.find("0x100"), std::string::npos);
  EXPECT_EQ(result.find("0x104"), std::string::npos);
}

TEST(FormatTraceTest, ConflictBeforeTraceWindow) {
  auto trace = make_trace({0x200, 0x204, 0x208});
  std::unordered_map<uint64_t, std::string> disasm = {
      {0x100, "buffer_load_dwordx4 v[148:151], v0, s[8:11], 0"},
      {0x200, "s_nop 0"},
      {0x204, "s_nop 0"},
      {0x208, "ds_read_b32 v8, v9"},
  };
  plugins::race_detector::MarkedPc conflict{0x100, 3, -1};
  plugins::race_detector::MarkedPc read{0x208, 0, 5};
  auto result = formatTrace(trace, disasm, conflict, read);
  EXPECT_NE(result.find("before trace window"), std::string::npos);
  EXPECT_NE(result.find("buffer_load_dwordx4"), std::string::npos);
  EXPECT_NE(result.find("; <-- wave 3"), std::string::npos);
  EXPECT_NE(result.find("not recorded"), std::string::npos);
  EXPECT_NE(result.find("; <-- wave 0 lane 5"), std::string::npos);
}

TEST(DisasmCacheTest, HandlesNonMonotonicPcOrder) {
  plugins::race_detector::DisasmCache cache;
  Instruction high_instruction("s_nop 0", nullptr);
  Instruction low_instruction("s_endpgm", nullptr);
  cache.record(0x540024b100, high_instruction);
  cache.record(0x100002a100, low_instruction);

  auto disasm = cache.to_map();
  EXPECT_EQ(disasm.at(0x540024b100), "s_nop 0");
  EXPECT_EQ(disasm.at(0x100002a100), "s_endpgm");
}

TEST(DisasmCacheTest, DisassemblesOnlyFirstInstructionAtPc) {
  class ObservableInstruction final : public Instruction {
  public:
    ObservableInstruction() : Instruction("s_count", nullptr) {}
    bool was_disassembled() const { return !disassembly_.empty(); }
  };

  plugins::race_detector::DisasmCache cache;
  ObservableInstruction first;
  ObservableInstruction duplicate;

  // DisasmCache is keyed by the absolute instruction PC. The value itself is
  // arbitrary here; using the same synthetic PC proves that a newly decoded
  // instruction at an already-cached address is not disassembled again.
  constexpr uint64_t synthetic_pc = 0x100;
  cache.record(synthetic_pc, first);
  cache.record(synthetic_pc, duplicate);

  EXPECT_TRUE(first.was_disassembled());
  EXPECT_FALSE(duplicate.was_disassembled());
  EXPECT_EQ(cache.to_map().at(synthetic_pc), "s_count");
}

TEST(RaceDetectorPluginOutputTest, DispatchLineUsesQuestionMarksForUnresolvedKernel) {
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  ExecutionPluginGroup plugin_group(std::move(sink_config));
  ASSERT_TRUE(plugin_group.add(std::make_unique<plugins::race_detector::RaceDetectorPlugin>()));

  KernelDispatchInfo info{};
  info.dispatch_id = 17;
  plugin_group.onAmdgpuDispatchPacketProcessed(info);

  EXPECT_NE(sink.str().find("[rocjitsu] Kernel dispatch: \"?\" symbol=\"?\"\n"), std::string::npos);
}

TEST(RaceDetectorPluginOutputTest, DispatchLineUsesReadableNameAndExactSymbol) {
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  ExecutionPluginGroup plugin_group(std::move(sink_config));
  ASSERT_TRUE(plugin_group.add(std::make_unique<plugins::race_detector::RaceDetectorPlugin>()));

  KernelDispatchInfo info{};
  info.dispatch_id = 18;
  info.kernel_name = "racy_kernel";
  info.kernel_symbol = "_Z11racy_kernelPKfPf";
  plugin_group.onAmdgpuDispatchPacketProcessed(info);

  EXPECT_NE(sink.str().find("[rocjitsu] Kernel dispatch: \"racy_kernel\" "
                            "symbol=\"_Z11racy_kernelPKfPf\"\n"),
            std::string::npos);
}

TEST(RaceDetectorPluginTest, DroppedAsyncLdsLaneDoesNotCreateLowAddressRace) {
  PluginFixture f(/*num_wf_slots=*/2, /*arch=*/"cdna5", /*wavefront_size=*/32);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *cu = f.cu();
  auto *writer = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  auto *reader = cu->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(writer, nullptr);
  ASSERT_NE(reader, nullptr);
  writer->set_exec(0x1u);
  reader->set_exec(0x1u);
  std::array<amdgpu::Wavefront *, 2> waves{writer, reader};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0,
                                               /*physical_vgpr_count=*/512, /*sgpr_count=*/208,
                                               waves);

  auto dropped = std::make_unique<VectorMemState>(GLOBAL_MEM);
  dropped->elem_size = 4;
  dropped->num_elems = 1;
  dropped->is_load = true;
  dropped->lds_dst = true;
  dropped->lds_per_lane_addr = true;
  dropped->wf_size = writer->wf_size();
  dropped->exec_mask = 0x1u;
  dropped->lane_mask = 0x1u;
  dropped->per_lane_lds_addr[0] = amdgpu::kInvalidLdsAddress;
  TestMemoryInstruction dropped_inst(std::move(dropped));
  f.plugin_group_->onAmdgpuRouteMemoryInstruction(dropped_inst, *writer);

  auto read = std::make_unique<VectorMemState>(LOCAL_MEM);
  read->elem_size = 4;
  read->num_elems = 1;
  read->is_load = true;
  read->wf_size = reader->wf_size();
  read->lane_mask = 0x1u;
  read->per_lane_addr[0] = 0;
  read->dst_reg_base = reader->vgpr_alloc().base;
  TestMemoryInstruction read_inst(std::move(read));
  f.plugin_group_->onAmdgpuRouteMemoryInstruction(read_inst, *reader);

  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

class IgnoredGlobalMemoryRaceTest : public ::testing::TestWithParam<AtomicOp> {};

TEST_P(IgnoredGlobalMemoryRaceTest, DoesNotCreateDestinationRace) {
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"cdna5", /*wavefront_size=*/32);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *wf = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0,
                                               /*physical_vgpr_count=*/256, /*sgpr_count=*/104,
                                               waves);

  constexpr uint32_t kDestinationVgpr = 7;
  auto ignored = std::make_unique<VectorMemState>(GLOBAL_MEM);
  ignored->elem_size = 4;
  ignored->num_elems = 1;
  ignored->is_load = true;
  ignored->atomic_op = GetParam();
  ignored->exec_mask = 0;
  ignored->lane_mask = 0;
  ignored->wf_size = wf->wf_size();
  ignored->dst_reg_base = wf->vgpr_alloc().base + kDestinationVgpr;
  TestMemoryInstruction ignored_inst(std::move(ignored));
  f.plugin_group_->onAmdgpuRouteMemoryInstruction(ignored_inst, *wf);

  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kDestinationVgpr,
                                         /*lane_mask=*/0x1u, /*byte_mask=*/0xFu);
  EXPECT_EQ(sink.str().find("RACE "), std::string::npos);
}

TEST_P(IgnoredGlobalMemoryRaceTest, DoesNotConsumeVmcntOrderingSlot) {
  PluginFixture f(/*num_wf_slots=*/1, /*arch=*/"cdna5", /*wavefront_size=*/32);
  PluginSinkConfig sink_config;
  StringSink &sink = sink_config.emplace<StringSink>();
  f.plugin_group_ = std::make_shared<ExecutionPluginGroup>(std::move(sink_config));
  ASSERT_TRUE(f.plugin_group_->add(std::make_unique<RaceDetectorPlugin>()));
  f.soc->set_plugin_group(f.plugin_group_);
  f.plugin_group_->onInit();

  auto *wf = f.cu()->dispatch_wf(/*wg_id=*/0, /*pc=*/0, /*sgprs=*/104, /*vgprs=*/256);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0x1u);
  std::array<amdgpu::Wavefront *, 1> waves{wf};
  f.plugin_group_->onAmdgpuWorkgroupDispatched(/*dispatch_id=*/1, /*wg_id=*/0,
                                               /*physical_vgpr_count=*/256, /*sgpr_count=*/104,
                                               waves);

  constexpr uint32_t kDestinationVgpr = 7;
  auto load = std::make_unique<VectorMemState>(GLOBAL_MEM);
  load->elem_size = 4;
  load->num_elems = 1;
  load->is_load = true;
  load->exec_mask = 0x1u;
  load->lane_mask = 0x1u;
  load->wf_size = wf->wf_size();
  load->dst_reg_base = wf->vgpr_alloc().base + kDestinationVgpr;
  TestMemoryInstruction load_inst(std::move(load));
  f.plugin_group_->onAmdgpuRouteMemoryInstruction(load_inst, *wf);

  auto ignored = std::make_unique<VectorMemState>(GLOBAL_MEM);
  ignored->elem_size = 4;
  ignored->num_elems = 1;
  ignored->is_load = false;
  ignored->atomic_op = GetParam();
  ignored->exec_mask = 0;
  ignored->lane_mask = 0;
  ignored->wf_size = wf->wf_size();
  TestMemoryInstruction ignored_inst(std::move(ignored));
  f.plugin_group_->onAmdgpuRouteMemoryInstruction(ignored_inst, *wf);

  wf->set_wait_target(/*vmcnt=*/1, /*lgkmcnt=*/0, /*expcnt=*/0);
  TestWaitcntInstruction waitcnt;
  f.plugin_group_->onAmdgpuAfterExecuteInstruction(wf->pc, waitcnt, *wf);

  f.plugin_group_->onAmdgpuReadVgprLanes(wf, wf->vgpr_alloc().base + kDestinationVgpr,
                                         /*lane_mask=*/0x1u, /*byte_mask=*/0xFu);
  EXPECT_NE(sink.str().find("RACE "), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(LoadAndReturningAtomic, IgnoredGlobalMemoryRaceTest,
                         ::testing::Values(AtomicOp::NONE, AtomicOp::SWAP));

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
  EXPECT_EQ(std::count(events.begin() + 3, events.end(), "sink"), 2);
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
