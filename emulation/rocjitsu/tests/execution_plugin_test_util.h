// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file execution_plugin_test_util.h
/// @brief Shared fixtures for execution plugin tests.
///
/// @details Register observation follows the physical register, lane, and byte contract
/// documented in execution_plugin_register_test.cpp.

#pragma once

#include "aql_queue.h"
#include "decode_test_util.h"

#include "embedded_schema.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_pipeline.h"
#include "rocjitsu/vm/soc.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/plugins/memory_access_observation.h"
#include "rocjitsu/vm/plugins/plugin_sink.h"

#include <flatbuffers/flexbuffers.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace rocjitsu::test {

class ComputeUnitTestAccess {
public:
  static amdgpu::Wavefront *sgpr_owner(const amdgpu::ComputeUnitCore &cu, uint32_t reg_idx) {
    return cu.sgpr_owner(reg_idx);
  }

  static void route_memory_inst(amdgpu::ComputeUnitCore &cu, Instruction *inst,
                                amdgpu::Wavefront &wf) {
    cu.route_memory_inst(inst, wf);
  }
};

} // namespace rocjitsu::test

namespace rocjitsu::test::execution_plugin {

static_assert(std::is_final_v<ExecutionPluginGroup>);
static_assert(!std::is_polymorphic_v<ExecutionPluginGroup>);

static_assert(!std::is_default_constructible_v<ExecutionPluginGroup>);

// SOPP encoding: bits[31:23]=0x17F, bits[22:16]=op, bits[15:0]=simm16.
constexpr uint32_t sopp(uint32_t op, uint16_t simm16 = 0) {
  return 0xBF800000u | (op << 16) | simm16;
}
inline constexpr uint32_t S_NOP = sopp(0);
inline constexpr uint32_t S_ENDPGM = sopp(1);
inline constexpr uint32_t S_BARRIER = sopp(10);

constexpr uint32_t vgpr_src(uint32_t reg) { return 256u + reg; }

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

inline constexpr uint64_t kPartialExecMask = 0xA5A5'F0F0'1234'8001ULL;

class TestMemoryInstruction : public Instruction {
public:
  TestMemoryInstruction(std::unique_ptr<DynamicInstState> state,
                        std::initializer_list<amdgpu::MemoryCounterObligation> counter_obligations,
                        bool exec_masked = true)
      : Instruction("test_mem", nullptr) {
    set_memory_issue_info(counter_obligations, exec_masked);
    set_data(std::move(state));
  }

  explicit TestMemoryInstruction(
      std::unique_ptr<DynamicInstState> state, std::string_view mnemonic = "test_mem",
      std::optional<amdgpu::WaitCounterType> additional_wait_counter_type = std::nullopt,
      bool memory_wait_producer = false)
      : Instruction(mnemonic, nullptr) {
    if (memory_wait_producer)
      flags_ |= MEMORY_WAIT_PRODUCER;
    if (state->tag() == amdgpu::SCALAR_MEM) {
      const auto &memory = *static_cast<const amdgpu::ScalarMemState *>(state.get());
      if (additional_wait_counter_type) {
        set_memory_issue_info(
            {{memory.wait_counter_type, amdgpu::MemoryCompletionClass::UNORDERED},
             {*additional_wait_counter_type, amdgpu::MemoryCompletionClass::UNORDERED}},
            false);
      } else {
        set_memory_issue_info(
            {{memory.wait_counter_type, amdgpu::MemoryCompletionClass::UNORDERED}}, false);
      }
    } else if (state->tag() == amdgpu::GLOBAL_MEM || state->tag() == amdgpu::LOCAL_MEM) {
      const auto &memory = *static_cast<const amdgpu::VectorMemState *>(state.get());
      const auto completion_class = state->tag() == amdgpu::LOCAL_MEM
                                        ? amdgpu::MemoryCompletionClass::LDS
                                        : amdgpu::MemoryCompletionClass::VMEM;
      if (additional_wait_counter_type) {
        set_memory_issue_info({{memory.wait_counter_type, completion_class},
                               {*additional_wait_counter_type, completion_class}});
      } else {
        set_memory_issue_info({{memory.wait_counter_type, completion_class}});
      }
    }
    set_data(std::move(state));
  }
};

// Routing tests supply a synthetic execution result, but wait tracking needs
// decoded operands and the pre-execution address registers, as in the issuer.
inline std::unique_ptr<Instruction>
prepare_cdna4_flat_load_for_routing(amdgpu::ComputeUnitCore &cu, amdgpu::Wavefront &wave,
                                    std::unique_ptr<amdgpu::VectorMemState> state) {
  for (unsigned lane = 0; lane < wave.wf_size(); ++lane) {
    cu.write_vgpr(wave.vgpr_alloc().base, lane, state->per_lane_addr[lane]);
    cu.write_vgpr(wave.vgpr_alloc().base + 1, lane, state->per_lane_addr[lane] >> 32);
  }
  const auto words = cdna4::build_flat(
      cdna4::kFlatLoadDwordFlat,
      {.seg = 0,
       .addr = 0,
       .saddr = 0x7F,
       .vdst = static_cast<uint8_t>(state->dst_reg_base - wave.vgpr_alloc().base)});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  std::unique_ptr<Instruction> load(decode_valid(*decoder, words.data()));
  if (!load)
    return nullptr;
  cu.track_memory_wait(*load, wave);
  load->set_data(std::move(state));
  return load;
}

class CounterObservingPipeline : public amdgpu::MemoryPipeline {
public:
  CounterObservingPipeline() : amdgpu::MemoryPipeline(amdgpu::WaitCounterType::VMCNT) {}
  using amdgpu::MemoryPipeline::issue_counters;

  amdgpu::WaitCounters counters_at_access{};

protected:
  amdgpu::VmAccessOutcome initiate_access(Instruction &, amdgpu::Wavefront &wf) override {
    counters_at_access = wf.wait_counters();
    return amdgpu::VmAccessOutcome::Complete;
  }

  amdgpu::MemoryAccessCompletion complete_access(Instruction &, amdgpu::Wavefront &,
                                                 amdgpu::MemoryAccessDeferredCompletion) override {
    return amdgpu::MemoryAccessCompletion::Complete;
  }
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
  uint8_t memory_tag = 0;
  uint64_t pc = 0;
  std::thread::id callback_thread;
  std::vector<amdgpu::MemoryCounterObligation> counter_obligations;
  std::string mnemonic;
  std::string kernel_name;
  std::string kernel_symbol;
  uint32_t lds_size_bytes = 0;
  uint32_t wave_size = 0;
  rj_code_target_id_t code_target = ROCJITSU_CODE_TARGET_INVALID;
  uint32_t cluster_size_x = 0;
  uint32_t cluster_size_y = 0;
  uint32_t cluster_size_z = 0;
};

/// @brief One routed access, copied out of the callback's borrowed spans.
struct CapturedAccess {
  std::string mnemonic;
  uint64_t pc = 0;
  uint32_t compute_unit_id = 0;
  uint32_t dispatch_id = 0;
  uint32_t workgroup_id = 0;
  uint32_t wavefront_id = 0;
  uint32_t process_id = 0;
  uint32_t queue_id = 0;
  amdgpu::MemoryRoute route = amdgpu::MemoryRoute::UNKNOWN;
  amdgpu::DecodedMemorySpace decoded_space = amdgpu::DecodedMemorySpace::UNKNOWN;
  bool normalized_to_local = false;
  bool is_load = true;
  amdgpu::AtomicOp atomic_op = amdgpu::AtomicOp::NONE;
  amdgpu::Mtype mtype = amdgpu::Mtype::RW;
  amdgpu::WaitCounterType wait_counter = amdgpu::WaitCounterType::VMCNT;
  bool force_l1_bypass = false;
  bool lds_destination = false;
  uint32_t wavefront_size = 0;
  uint32_t element_size_bytes = 0;
  uint32_t elements_per_lane = 0;
  uint64_t bytes_per_lane = 0;
  uint64_t active_lane_mask = 0;
  uint64_t architectural_exec_lane_mask = 0;
  uint64_t valid_lane_mask = 0;
  uint64_t request_lane_mask = 0;
  uint64_t inactive_lane_mask = 0;
  uint64_t unknown_lane_mask = 0;
  uint64_t scratch_lane_mask = 0;
  uint64_t flat_local_lane_mask = 0;
  uint64_t flat_dds_lane_mask = 0;
  uint32_t scratch_element_stride_bytes = 0;
  bool non_temporal = false;
  std::vector<uint64_t> element_lane_masks;
  std::vector<uint64_t> addresses;
  std::vector<uint64_t> pre_routing_addresses;
  std::vector<uint64_t> secondary_addresses;
  std::vector<amdgpu::WaitCounterType> issued_counters;
};

/// @brief Records both memory hooks, so a test can compare what routing was
///        told with what routing decided.
class MemoryObservationPlugin final : public ExecutionPlugin {
public:
  /// @param wants_hook What observes_memory_routing() answers. False models a
  ///        plugin that implements the hook but never opts in, which must
  ///        receive nothing.
  explicit MemoryObservationPlugin(bool wants_hook = true, std::string name = "memory_observation")
      : ExecutionPlugin(std::move(name)), wants_hook_(wants_hook) {}

  bool observes_memory_routing() const override { return wants_hook_; }

  void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access,
                                  const Instruction &inst, amdgpu::Wavefront &) override {
    onAmdgpuMemoryAccessRouted(access);
    const auto counters = CounterObservingPipeline{}.issue_counters(inst);
    accesses.back().issued_counters.assign(counters.types.begin(),
                                           counters.types.begin() + counters.size);
  }

  void onAmdgpuRouteMemoryInstruction(const Instruction &inst, amdgpu::Wavefront &wf) override {
    before_tag.push_back(inst.data()->tag());
    // Only the two vector tags carry a amdgpu::VectorMemState. An instruction no
    // pipeline will take has neither, and downcasting it walks off the object.
    if (inst.data()->tag() == amdgpu::GLOBAL_MEM || inst.data()->tag() == amdgpu::LOCAL_MEM) {
      const auto &state = *inst.data_as<amdgpu::VectorMemState>();
      before_first_address.push_back(state.per_lane_addr[0]);
      before_wait_counter.push_back(state.wait_counter_type);
    }
    static_cast<void>(wf);
  }

  void onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access) override {
    CapturedAccess captured;
    captured.mnemonic = access.mnemonic;
    captured.pc = access.pc;
    captured.compute_unit_id = access.compute_unit_id;
    captured.dispatch_id = access.dispatch_id;
    captured.workgroup_id = access.workgroup_id;
    captured.wavefront_id = access.wavefront_id;
    captured.process_id = access.process_id;
    captured.queue_id = access.queue_id;
    captured.route = access.route;
    captured.decoded_space = access.decoded_space;
    captured.normalized_to_local = access.normalized_to_local;
    captured.is_load = access.is_load;
    captured.atomic_op = access.atomic_op;
    captured.mtype = access.mtype;
    captured.wait_counter = access.wait_counter;
    captured.force_l1_bypass = access.force_l1_bypass;
    captured.lds_destination = access.lds_destination;
    captured.wavefront_size = access.wavefront_size;
    captured.element_size_bytes = access.element_size_bytes;
    captured.elements_per_lane = access.elements_per_lane;
    captured.bytes_per_lane = access.bytes_per_lane();
    captured.active_lane_mask = access.active_lane_mask;
    captured.architectural_exec_lane_mask = access.architectural_exec_lane_mask;
    captured.valid_lane_mask = access.valid_lane_mask;
    captured.request_lane_mask = access.request_lane_mask;
    captured.inactive_lane_mask = access.inactive_lane_mask();
    captured.unknown_lane_mask = access.unknown_lane_mask();
    captured.scratch_lane_mask = access.scratch_lane_mask;
    captured.flat_local_lane_mask = access.flat_local_lane_mask;
    captured.flat_dds_lane_mask = access.flat_dds_lane_mask;
    captured.scratch_element_stride_bytes = access.scratch_element_stride_bytes;
    captured.non_temporal = access.non_temporal;
    captured.element_lane_masks.assign(access.element_lane_masks.begin(),
                                       access.element_lane_masks.end());
    captured.addresses.assign(access.addresses.begin(), access.addresses.end());
    captured.pre_routing_addresses.assign(access.pre_routing_addresses.begin(),
                                          access.pre_routing_addresses.end());
    captured.secondary_addresses.assign(access.secondary_addresses.begin(),
                                        access.secondary_addresses.end());
    accesses.push_back(std::move(captured));
  }

  std::vector<CapturedAccess> accesses;
  std::vector<uint8_t> before_tag;
  const bool wants_hook_ = true;
  std::vector<uint64_t> before_first_address;
  std::vector<amdgpu::WaitCounterType> before_wait_counter;
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
    e.lds_size_bytes = info.lds_size_bytes;
    e.wave_size = info.wave_size;
    e.code_target = info.code_target;
    e.cluster_size_x = info.cluster_size_x;
    e.cluster_size_y = info.cluster_size_y;
    e.cluster_size_z = info.cluster_size_z;
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
    if (const auto *info = inst.amdgpu_memory_issue_info()) {
      const auto obligations = info->counter_obligations();
      e.counter_obligations.assign(obligations.begin(), obligations.end());
    }
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
    e.memory_tag = inst.data()->tag();
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

inline std::vector<HookEvent> vgpr_read_events(const OrderingPlugin &plugin) {
  std::vector<HookEvent> reads;
  for (const HookEvent &e : plugin.events)
    if (e.kind == HookEvent::READ_VGPR)
      reads.push_back(e);
  return reads;
}

inline std::vector<HookEvent> vgpr_write_events(const OrderingPlugin &plugin) {
  std::vector<HookEvent> writes;
  for (const HookEvent &e : plugin.events)
    if (e.kind == HookEvent::WRITE_VGPR)
      writes.push_back(e);
  return writes;
}

inline void expect_vgpr_read_set(const std::vector<HookEvent> &events, uint32_t physical_base,
                                 std::vector<uint32_t> expected_logical_regs,
                                 uint64_t expected_lane_mask,
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
  std::ranges::sort(actual_logical_regs);
  std::ranges::sort(expected_logical_regs);
  EXPECT_EQ(actual_logical_regs, expected_logical_regs);
}

inline std::vector<uint8_t> make_loaded_kernel_symbol_elf(uint64_t kernel_descriptor_offset,
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

/// Minimal SoC fixture: 1 XCD, 1 SE, and a configurable CU count.
struct PluginFixture {
  std::unique_ptr<simdojo::SimulationEngine> engine;
  SoC *soc = nullptr;
  amdgpu::GpuMemory *mem = nullptr;

  explicit PluginFixture(uint32_t num_wf_slots = 10, std::string_view arch = "cdna4",
                         uint32_t wavefront_size = 64, uint32_t sgprs_per_wf = 104,
                         uint32_t vgprs_per_wf = 256, uint32_t num_cus = 1,
                         uint32_t async_helpers = 0) {
    std::string cu_range = "cu[0:" + std::to_string(num_cus) + "]";
    std::string links;
    for (uint32_t i = 0; i < num_cus; ++i) {
      if (!links.empty())
        links += ',';
      links += R"({"src":"xcd0.cp.req_)" + std::to_string(i) + R"(","dst":"xcd0.se0.cu)" +
               std::to_string(i) + R"(.cpl","latency":1,"weight":2})";
      links += R"(,{"src":"xcd0.se0.cu)" + std::to_string(i) + R"(.req","dst":"xcd0.l2.cpl_)" +
               std::to_string(i) + R"(","latency":1,"weight":10})";
    }
    std::string json = std::format(R"({{
      "max_ticks":10000,"num_threads":1,"exec_mode":"functional",
      "async_helper_threads":{},
      "vm":{{"arch":"{}","gpu":{{"device":{{"wave_front_size":{},
        "num_sdma_engines":0}}}}}},
      "topology":{{"root":{{"name":"soc","type":"soc","children":[
        {{"name":"vram","type":"gpu_memory"}},
        {{"name":"xcd0","type":"xcd","children":[
          {{"name":"l2","type":"l2_cache"}},
          {{"name":"cp","type":"command_processor"}},
          {{"name":"se0","type":"shader_engine","children":[
            {{"name":"{}","type":"compute_unit","config":[
              {{"key":"num_wf_slots","value":"{}"}},
              {{"key":"sgprs_per_wf","value":"{}"}},
              {{"key":"vgprs_per_wf","value":"{}"}},
              {{"key":"memory_wait_diagnostics","value":"warn"}},
              {{"key":"lds_size_kb","value":"64"}}
            ]}}
          ]}}
        ]}}
      ]}},"links":[{}]}}}}
    )",
                                   async_helpers, arch, wavefront_size, cu_range, num_wf_slots,
                                   sgprs_per_wf, vgprs_per_wf, links);
    auto loaded = config::load_config_from_string(json, rocjitsu::kEmbeddedSchema);
    soc = loaded.soc();
    mem = loaded.memory();
    engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
    engine->topology().set_root(loaded.take_root());
    loaded.wire_links(engine->topology());
    engine->create();
  }

  amdgpu::ComputeUnitCore *cu(uint32_t idx = 0) {
    return soc->xcd(0)->shader_engine(0)->compute_unit(idx);
  }
  amdgpu::CommandProcessor *cp() { return soc->xcd(0)->command_processor(); }

  uint64_t write_kernel(uint64_t addr, const uint32_t *code, size_t num_words,
                        uint32_t granulated_sgpr_count = 12,
                        uint32_t group_segment_fixed_size = 0) {
    using namespace rocr::llvm::amdhsa;
    kernel_descriptor_t kd{};
    kd.kernel_code_entry_byte_offset = sizeof(kernel_descriptor_t);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 31);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                    granulated_sgpr_count);
    AMDHSA_BITS_SET(kd.compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 2);
    kd.group_segment_fixed_size = group_segment_fixed_size;
    mem->load_image(reinterpret_cast<const uint8_t *>(&kd), sizeof(kd), addr);
    mem->load_image(reinterpret_cast<const uint8_t *>(code), num_words * 4,
                    addr + sizeof(kernel_descriptor_t));
    return addr;
  }

  /// The attached plugin group.
  ExecutionPluginGroup &plugin_group() { return *plugin_group_; }

  /// Attach a MemoryObservationPlugin, fire onInit, and return a raw pointer.
  MemoryObservationPlugin *attach_memory_observation_plugin(bool wants_hook = true) {
    plugin_group_ = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
    auto plugin = std::make_unique<MemoryObservationPlugin>(wants_hook);
    auto *p = plugin.get();
    plugin_group_->add(std::move(plugin));
    soc->set_plugin_group(plugin_group_);
    plugin_group_->onInit();
    return p;
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

} // namespace rocjitsu::test::execution_plugin
