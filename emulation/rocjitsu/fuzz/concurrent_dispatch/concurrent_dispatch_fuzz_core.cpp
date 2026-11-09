// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "concurrent_dispatch_fuzz_core.h"

#include "embedded_schema.h"
#include "execution_fuzz_core.h"
#include "rocjitsu/code/builders/vector_builders.h"
#include "rocjitsu/code/kernel_descriptor_abi.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand_types.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_traits.h"
#include "rocjitsu/vm/amdgpu/aql/aql_packet_types.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/partitioning.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/amdgpu/xcd.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "rocjitsu/vm/plugins/execution_plugin_group.h"
#include "rocjitsu/vm/soc.h"
#include "simdojo/sim/simulation.h"
#include "simdojo/sim/topology.h"
#include "util/diagnostic.h"
#include "util/result.h"

#include "rocjitsu/base/rj_compiler.h"

#ifndef HSA_LARGE_MODEL
#define HSA_LARGE_MODEL 1
#endif

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/hsa.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <bitset>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::concurrent_dispatch_fuzz {
namespace {

constexpr uint64_t kKernelAddress = 0x10000;
constexpr uint64_t kDataBase = 0x200000;
constexpr std::size_t kDataSize = 64 * 1024;
constexpr uint64_t kSignalAddress = 0x300000;
constexpr uint64_t kSignalValueOffset = 8;
constexpr uint64_t kSignalStride = 64;
constexpr uint64_t kRingBase = 0xf0000000ULL;
constexpr uint32_t kRingSize = 64 * amdgpu::kAqlPacketBytes;
constexpr uint64_t kCompletedSignalValue = 0x100;
constexpr uint64_t kPendingSignalValue = kCompletedSignalValue + 1;
constexpr uint32_t kMaxPackets = 2;
constexpr uint32_t kMinWorkgroups = 3;
constexpr uint32_t kMaxWorkgroups = 5;
constexpr uint32_t kComputeUnitsPerXcd = 3;
constexpr uint8_t kGfx12DeviceScope = 2;
constexpr uint16_t kInlineOne = cdna1::OPR_SRC_POS_INT_MIN + 1;

constexpr uint64_t completion_signal_address(std::size_t packet) {
  return kSignalAddress + packet * kSignalStride;
}

uint64_t next_random(uint64_t &state) {
  state += 0x9e3779b97f4a7c15ULL;
  uint64_t mixed = state;
  mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
  mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
  return mixed ^ (mixed >> 31);
}

uint8_t input_byte(std::span<const uint8_t> input, std::size_t index) {
  return input[index % input.size()];
}

template <typename Words> void append_words(std::vector<uint8_t> &bytes, const Words &words) {
  for (uint32_t word : words)
    for (unsigned byte = 0; byte < sizeof(word); ++byte)
      bytes.push_back(static_cast<uint8_t>(word >> (8 * byte)));
}

std::vector<uint8_t> cache_program(execution_fuzz::FuzzTarget target) {
  std::vector<uint8_t> code;
  const rj_code_arch_t arch = execution_fuzz::target_arch(target);
  const util::FailureOr<uint32_t> address_lo = build_v_mov_b32_src(0, 0, arch);
  const util::FailureOr<uint32_t> address_hi = build_v_mov_b32_src(1, 1, arch);
  const util::FailureOr<uint32_t> increment = build_v_mov_b32_src(2, kInlineOne, arch);
  if (address_lo.failed() || address_hi.failed() || increment.failed())
    throw std::invalid_argument("invalid concurrent dispatch fuzz architecture");
  append_words(code, std::array{address_lo.value(), address_hi.value(), increment.value()});
  switch (target) {
  case execution_fuzz::FuzzTarget::Gfx908:
    append_words(code,
                 cdna1::build_flat(cdna1::kFlatAtomicAddFlat, {.addr = 0, .data = 2, .saddr = 0}));
    append_words(code, cdna1::build_sopp(cdna1::kSWaitcntSopp));
    break;
  case execution_fuzz::FuzzTarget::Gfx90a:
    append_words(code,
                 cdna2::build_flat(cdna2::kFlatAtomicAddFlat, {.addr = 0, .data = 2, .saddr = 0}));
    append_words(code, cdna2::build_sopp(cdna2::kSWaitcntSopp));
    break;
  case execution_fuzz::FuzzTarget::Gfx942:
    append_words(code,
                 cdna3::build_flat(cdna3::kFlatAtomicAddFlat, {.addr = 0, .data = 2, .saddr = 0}));
    append_words(code, cdna3::build_sopp(cdna3::kSWaitcntSopp));
    break;
  case execution_fuzz::FuzzTarget::Gfx950:
    append_words(code,
                 cdna4::build_flat(cdna4::kFlatAtomicAddFlat, {.addr = 0, .data = 2, .saddr = 0}));
    append_words(code, cdna4::build_sopp(cdna4::kSWaitcntSopp));
    break;
  case execution_fuzz::FuzzTarget::Gfx1010:
    append_words(code, rdna1::build_flat(rdna1::kFlatAtomicAddFlat,
                                         {.addr = 0, .data = 2, .saddr = rdna1::OPR_SREG_NULL}));
    append_words(code, rdna1::build_sopp(rdna1::kSWaitcntSopp));
    append_words(code,
                 rdna1::build_sopk(rdna1::kSWaitcntVscntSopk, {.sdst = rdna1::OPR_SREG_NULL}));
    break;
  case execution_fuzz::FuzzTarget::Gfx1030:
    append_words(code, rdna2::build_flat(rdna2::kFlatAtomicAddFlat,
                                         {.addr = 0, .data = 2, .saddr = rdna2::OPR_SREG_NULL}));
    append_words(code, rdna2::build_sopp(rdna2::kSWaitcntSopp));
    append_words(code,
                 rdna2::build_sopk(rdna2::kSWaitcntVscntSopk, {.sdst = rdna2::OPR_SREG_NULL}));
    break;
  case execution_fuzz::FuzzTarget::Gfx1100:
    append_words(code, rdna3::build_flat(rdna3::kFlatAtomicAddU32Flat,
                                         {.addr = 0, .data = 2, .saddr = rdna3::OPR_SREG_NULL}));
    append_words(code, rdna3::build_sopp(rdna3::kSWaitcntSopp));
    append_words(code,
                 rdna3::build_sopk(rdna3::kSWaitcntVscntSopk, {.sdst = rdna3::OPR_SREG_NULL}));
    break;
  case execution_fuzz::FuzzTarget::Gfx1151:
    append_words(code,
                 rdna3_5::build_flat(rdna3_5::kFlatAtomicAddU32Flat,
                                     {.addr = 0, .data = 2, .saddr = rdna3_5::OPR_SREG_NULL}));
    append_words(code, rdna3_5::build_sopp(rdna3_5::kSWaitcntSopp));
    append_words(
        code, rdna3_5::build_sopk(rdna3_5::kSWaitcntVscntSopk, {.sdst = rdna3_5::OPR_SREG_NULL}));
    break;
  case execution_fuzz::FuzzTarget::Gfx1200:
  case execution_fuzz::FuzzTarget::Gfx1201:
    append_words(
        code,
        rdna4::build_vflat(
            rdna4::kFlatAtomicAddU32Vflat,
            {.saddr = rdna4::OPR_SREG_NULL, .scope = kGfx12DeviceScope, .vsrc = 2, .vaddr = 0}));
    append_words(code, rdna4::build_sopp(rdna4::kSWaitStorecntDscntSopp));
    break;
  case execution_fuzz::FuzzTarget::Gfx1250:
  case execution_fuzz::FuzzTarget::Gfx1251:
    append_words(
        code,
        cdna5::build_vflat(
            cdna5::kFlatAtomicAddU32Vflat,
            {.saddr = cdna5::OPR_SREG_NULL, .scope = kGfx12DeviceScope, .vsrc = 2, .vaddr = 0}));
    append_words(code, cdna5::build_sopp(cdna5::kSWaitStorecntDscntSopp));
    break;
  case execution_fuzz::FuzzTarget::Count:
    throw std::invalid_argument("invalid concurrent dispatch fuzz target");
  }
  return code;
}

uint32_t representative_xcd_count(execution_fuzz::FuzzTarget target) {
  switch (target) {
  case execution_fuzz::FuzzTarget::Gfx942:
  case execution_fuzz::FuzzTarget::Gfx950:
  case execution_fuzz::FuzzTarget::Gfx1250:
  case execution_fuzz::FuzzTarget::Gfx1251:
    return 2;
  case execution_fuzz::FuzzTarget::Gfx908:
  case execution_fuzz::FuzzTarget::Gfx90a:
  case execution_fuzz::FuzzTarget::Gfx1010:
  case execution_fuzz::FuzzTarget::Gfx1030:
  case execution_fuzz::FuzzTarget::Gfx1100:
  case execution_fuzz::FuzzTarget::Gfx1151:
  case execution_fuzz::FuzzTarget::Gfx1200:
  case execution_fuzz::FuzzTarget::Gfx1201:
    return 1;
  case execution_fuzz::FuzzTarget::Count:
    throw std::invalid_argument("invalid concurrent dispatch fuzz target");
  }
  throw std::invalid_argument("invalid concurrent dispatch fuzz target");
}

std::string fuzz_config(execution_fuzz::FuzzTarget target, uint32_t xcd_count) {
  const rj_code_arch_t arch = execution_fuzz::target_arch(target);
  std::string json =
      R"({"max_ticks":100000,"num_threads":1,"async_helper_threads":0,"exec_mode":"functional","vm":{"arch":")";
  json += config::arch_to_string(arch);
  json += '"';
  if (execution_fuzz::target_id(target) != ROCJITSU_CODE_TARGET_INVALID) {
    json += R"(,"target":")";
    json += execution_fuzz::target_name(target);
    json += '"';
  }
  json +=
      R"(},"wait_checking":"off","topology":{"root":{"name":"soc","type":"soc","children":[{"name":"vram","type":"gpu_memory"},{"name":"iod0","type":"iod","config":[{"key":"num_hbm_stacks","value":"1"}]},{"name":"xcd[0:)";
  json += std::to_string(xcd_count);
  json +=
      R"(]","type":"xcd","children":[{"name":"l2","type":"l2_cache"},{"name":"cp","type":"command_processor"},{"name":"se0","type":"shader_engine","children":[{"name":"cu[0:)";
  json += std::to_string(kComputeUnitsPerXcd);
  json +=
      R"(]","type":"compute_unit","config":[{"key":"num_wf_slots","value":"4"},{"key":"sgprs_per_wf","value":"128"},{"key":"vgprs_per_wf","value":")";
  json += std::to_string(isa_properties(arch).max_addressable_vgprs_per_wf);
  json += R"("},{"key":"lds_size_kb","value":")";
  json += std::to_string(arch_lds_bytes(arch) / 1024);
  json += R"("}]}]}]}]},"links":[)";
  bool first = true;
  const auto append_link = [&](std::string_view source, std::string_view destination,
                               uint32_t latency, uint32_t weight) {
    if (!first)
      json += ',';
    first = false;
    json += R"({"src":")";
    json += source;
    json += R"(","dst":")";
    json += destination;
    json += R"(","latency":)";
    json += std::to_string(latency);
    json += R"(,"weight":)";
    json += std::to_string(weight);
    json += '}';
  };
  for (uint32_t xcd = 0; xcd < xcd_count; ++xcd) {
    const std::string prefix = "xcd" + std::to_string(xcd);
    for (uint32_t cu = 0; cu < kComputeUnitsPerXcd; ++cu) {
      const std::string index = std::to_string(cu);
      append_link(prefix + ".cp.req_" + index, prefix + ".se0.cu" + index + ".cpl", 1, 2);
      append_link(prefix + ".se0.cu" + index + ".req", prefix + ".l2.cpl_" + index, 1, 10);
    }
    append_link(prefix + ".l2.req", "iod0.msc.cpl_" + std::to_string(xcd), 1, 3);
  }
  json += R"(]}})";
  return json;
}

struct Scenario {
  execution_fuzz::FuzzTarget target;
  uint32_t xcd_count;
  uint32_t workgroups;
  uint16_t workgroup_size;
  uint32_t packets;
  uint32_t queue_ordinal;
  bool prime_caches;
  bool xcd_fanout;
  bool barrier;
  bool agent_acquire;
  bool agent_release;
  uint32_t prime_lines;
  std::span<const uint8_t> program;
  std::span<const uint8_t> input;
};

std::optional<Scenario> decode_scenario(std::span<const uint8_t> input) {
  const std::optional<execution_fuzz::FuzzTarget> target =
      execution_fuzz::indexed_target(input.front());
  if (!target)
    return std::nullopt;
  const uint32_t xcd_count = representative_xcd_count(*target);
  const uint8_t control = input_byte(input, 1);
  return Scenario{
      .target = *target,
      .xcd_count = xcd_count,
      .workgroups = kMinWorkgroups + input_byte(input, 2) % (kMaxWorkgroups - kMinWorkgroups + 1),
      .workgroup_size = static_cast<uint16_t>(1u + input_byte(input, 3) % 32u),
      .packets = 1u + input_byte(input, 4) % kMaxPackets,
      .queue_ordinal = input_byte(input, 5) % xcd_count,
      .prime_caches = (control & kPrimeCachesBit) != 0,
      .xcd_fanout = xcd_count > 1 && (control & kXcdFanoutBit) != 0,
      .barrier = (control & kBarrierBit) != 0,
      .agent_acquire = (control & kAgentAcquireBit) != 0,
      .agent_release = (control & kAgentReleaseBit) != 0,
      .prime_lines = 1u + input_byte(input, 7) % 8u,
      .program = input.subspan(std::min(kHeaderSize, input.size())),
      .input = input,
  };
}

class BoundedFuzzAddressSpace final : public amdgpu::AddressSpaceTranslator {
public:
  amdgpu::VmTranslationResult translate(uint64_t address, std::size_t size,
                                        amdgpu::VmAccessKind access) const override {
    const auto in_range = [&](uint64_t begin, uint64_t end) {
      return size != 0 && address >= begin && address < end && size <= end - address;
    };
    uint64_t end = 0;
    amdgpu::VmPermissions permissions;
    if (in_range(kKernelAddress, kKernelAddress + kMaxInputSize + 4096)) {
      end = kKernelAddress + kMaxInputSize + 4096;
      permissions = {.readable = true, .writable = false, .executable = true};
    } else if (in_range(kDataBase, kDataBase + kDataSize)) {
      end = kDataBase + kDataSize;
      permissions = {.readable = true, .writable = true, .executable = false};
    } else {
      return {.outcome = amdgpu::VmAccessOutcome::Faulted, .translation = {}};
    }
    if (!permissions.allows(access))
      return {.outcome = amdgpu::VmAccessOutcome::Faulted, .translation = {}};

    return {.outcome = amdgpu::VmAccessOutcome::Complete,
            .translation = {.domain = amdgpu::VmMemoryDomain::Local,
                            .address = address,
                            .contiguous_bytes = end - address,
                            .mtype = amdgpu::Mtype::RW,
                            .permissions = permissions}};
  }
};

class QueueDriver {
public:
  QueueDriver(amdgpu::GpuMemory *memory, amdgpu::CommandProcessor *cp, uint32_t queue_id,
              bool xcd_fanout)
      : memory_(memory), cp_(cp), queue_id_(queue_id) {
    uint64_t zero = 0;
    memory_->load_image(reinterpret_cast<const uint8_t *>(&zero), sizeof(zero), read_ptr_address());
    memory_->load_image(reinterpret_cast<const uint8_t *>(&zero), sizeof(zero),
                        write_ptr_address());
    memory_->load_image(reinterpret_cast<const uint8_t *>(&zero), sizeof(zero), doorbell_address());

    amdgpu::ComputeQueueConfig config{};
    config.process_id = kProcessId;
    config.queue_id = queue_id;
    config.ring_base_va = kRingBase;
    config.ring_size = kRingSize;
    config.read_ptr_va = read_ptr_address();
    config.write_ptr_va = write_ptr_address();
    config.doorbell_va = doorbell_address();
    config.doorbell_mode = amdgpu::QueueDoorbellMode::VmPolled;
    config.xcd_fanout = xcd_fanout;
    if (cp_->register_queue(std::move(config)) == 0)
      throw std::runtime_error("concurrent dispatch fuzz could not register its queue");
  }

  void submit(const hsa_kernel_dispatch_packet_t &packet) {
    const uint64_t slot = write_index_ % (kRingSize / amdgpu::kAqlPacketBytes);
    const uint64_t packet_address = kRingBase + slot * amdgpu::kAqlPacketBytes;
    memory_->load_image(reinterpret_cast<const uint8_t *>(&packet), amdgpu::kAqlPacketBytes,
                        packet_address);
    ++write_index_;
    memory_->load_image(reinterpret_cast<const uint8_t *>(&write_index_), sizeof(write_index_),
                        write_ptr_address());
    memory_->load_image(reinterpret_cast<const uint8_t *>(&write_index_), sizeof(write_index_),
                        doorbell_address());
    cp_->engine()->schedule_event_now(cp_->doorbell_event());
  }

  bool terminal_faulted() const { return cp_->queue_faulted_for_test(queue_id_, kProcessId); }

private:
  // The standalone harness registers queues outside a guest process context.
  static constexpr uint32_t kProcessId = 0;
  static constexpr uint64_t read_ptr_address() { return kRingBase + 0x10000; }
  static constexpr uint64_t write_ptr_address() { return kRingBase + 0x10008; }
  static constexpr uint64_t doorbell_address() { return kRingBase + 0x10010; }

  amdgpu::GpuMemory *memory_;
  amdgpu::CommandProcessor *cp_;
  uint32_t queue_id_;
  uint64_t write_index_ = 0;
};

class FuzzObserver final : public ExecutionPlugin {
public:
  FuzzObserver(amdgpu::AddressSpaceHandle address_space, const amdgpu::GpuMemory &memory,
               const Scenario &scenario)
      : ExecutionPlugin("concurrent-dispatch-fuzz-observer"), address_space_(address_space),
        memory_(memory), workgroups_(scenario.workgroups), barrier_(scenario.barrier) {}

  bool observes_after_execute_instruction() const override { return false; }
  bool observes_async_instruction_issued() const override { return false; }
  bool observes_memory_instruction_routing() const override { return false; }
  bool observes_vgpr_reads() const override { return false; }
  bool observes_vgpr_writes() const override { return false; }
  bool observes_scalar_register_writes() const override { return false; }
  bool observes_sgpr_reads() const override { return false; }
  bool observes_memory_routing() const override { return false; }
  bool observes_tensor_dma_memory_access() const override { return false; }

  void onAmdgpuBeforeExecuteInstruction(uint64_t, const Instruction &,
                                        amdgpu::Wavefront &) override {
    instructions_.fetch_add(1, std::memory_order_relaxed);
  }

  void onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wave) override {
    wave.set_address_space(address_space_);
    waves_.fetch_add(1, std::memory_order_relaxed);
    for (uint32_t reg = 2; reg < 16; reg += 2) {
      const uint32_t address = static_cast<uint32_t>(kDataBase + (reg / 2) * 512);
      wave.debug_write_sgpr(reg, address);
      wave.debug_write_sgpr(reg + 1, 0);
    }
    for (uint32_t reg = 0; reg < 16; reg += 2) {
      for (uint32_t lane = 0; lane < wave.wf_size(); ++lane) {
        const uint32_t address =
            static_cast<uint32_t>(kDataBase + (reg / 2) * 512 + lane * sizeof(uint32_t));
        wave.debug_write_vgpr(reg, lane, address);
        wave.debug_write_vgpr(reg + 1, lane, 0);
      }
    }
  }

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override {
    std::lock_guard lock(progress_mutex_);
    if (observed_packets_ == dispatch_ids_.size() || info.dispatch_id == 0 ||
        std::ranges::find(dispatch_ids_, info.dispatch_id) != dispatch_ids_.end()) {
      invalid_progress_ = true;
      return;
    }
    // The only queue's owner admits packets in ring order before fanning out
    // workgroups. Peers report completions using the same dispatch ID.
    dispatch_ids_[observed_packets_++] = info.dispatch_id;
  }

  void onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t, uint32_t, uint32_t,
                                   std::span<amdgpu::Wavefront *>) override {
    if (!barrier_)
      return;
    std::lock_guard lock(progress_mutex_);
    const std::optional<std::size_t> packet = packet_index(dispatch_id);
    if (!packet) {
      invalid_progress_ = true;
      return;
    }
    for (std::size_t predecessor = 0; predecessor < *packet; ++predecessor)
      if (completed_[predecessor].count() != workgroups_)
        barrier_violation_ = true;
  }

  void onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) override {
    std::lock_guard lock(progress_mutex_);
    const std::optional<std::size_t> packet = packet_index(dispatch_id);
    if (!packet || wg_id >= workgroups_ || completed_[*packet].test(wg_id)) {
      invalid_progress_ = true;
      return;
    }
    // The CP notifies the observer before crediting this workgroup toward
    // grid retirement. The signal must still be pending at this point.
    if (memory_.read64(completion_signal_address(*packet) + kSignalValueOffset) !=
        kPendingSignalValue)
      published_early_ = true;
    completed_[*packet].set(wg_id);
  }

  // Progress is sampled after the engine and dispatch workers have joined.
  bool valid_progress(uint32_t packets) const {
    return !invalid_progress_ && observed_packets_ <= packets;
  }
  bool published_early() const { return published_early_; }
  bool barrier_violation() const { return barrier_violation_; }
  uint64_t completed(uint32_t packet) const { return completed_[packet].count(); }
  uint64_t waves() const { return waves_.load(std::memory_order_relaxed); }
  uint64_t instructions() const { return instructions_.load(std::memory_order_relaxed); }

private:
  std::optional<std::size_t> packet_index(uint32_t dispatch_id) const {
    const std::array<uint32_t, kMaxPackets>::const_iterator packet =
        std::ranges::find(dispatch_ids_, dispatch_id);
    if (dispatch_id == 0 || packet == dispatch_ids_.end())
      return std::nullopt;
    return static_cast<std::size_t>(packet - dispatch_ids_.begin());
  }

  amdgpu::AddressSpaceHandle address_space_;
  const amdgpu::GpuMemory &memory_;
  uint32_t workgroups_;
  bool barrier_;
  std::mutex progress_mutex_;
  std::array<uint32_t, kMaxPackets> dispatch_ids_{};
  std::array<std::bitset<kMaxWorkgroups>, kMaxPackets> completed_{};
  std::size_t observed_packets_ = 0;
  bool invalid_progress_ = false;
  bool published_early_ = false;
  bool barrier_violation_ = false;
  std::atomic<uint64_t> waves_{0};
  std::atomic<uint64_t> instructions_{0};
};

void install_input_kernel(amdgpu::GpuMemory &memory, const Scenario &scenario) {
  using namespace rocjitsu::amdhsa;
  const rj_code_arch_t arch = execution_fuzz::target_arch(scenario.target);
  const IsaProperties properties = isa_properties(arch);
  const uint32_t wave_size = properties.wave_size;
  const std::optional<uint32_t> granule =
      descriptor_vgpr_count_granule_for_wavefront(arch, wave_size);
  if (!granule)
    throw std::logic_error("concurrent dispatch fuzz target has no VGPR descriptor granule");

  kernel_descriptor_t descriptor{};
  descriptor.kernel_code_entry_byte_offset = sizeof(descriptor);
  RJ_AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                     KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, wave_size == 32);
  RJ_AMDHSA_BITS_SET(descriptor.kernel_code_properties,
                     KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 1);
  RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                     properties.max_addressable_vgprs_per_wf / *granule - 1);
  if (properties.descriptor_sgpr_count_encoded)
    RJ_AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc1,
                       COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, (128 / 8) - 1);
  set_kernel_descriptor_user_sgpr_count(arch, descriptor, 2);
  memory.load_image(reinterpret_cast<const uint8_t *>(&descriptor), sizeof(descriptor),
                    kKernelAddress);

  const std::vector<uint8_t> code =
      execution_fuzz::terminated_program(scenario.program, scenario.target);
  memory.load_image(code.data(), code.size(), kKernelAddress + sizeof(descriptor));
}

std::array<uint8_t, kDataSize> initial_data(std::span<const uint8_t> input) {
  uint64_t seed = 0x2d358dccaa6c78a5ULL;
  for (uint8_t byte : input)
    seed = std::rotl(seed ^ byte, 9) * 0x9e3779b97f4a7c15ULL;
  std::array<uint8_t, kDataSize> data{};
  for (std::size_t offset = 0; offset < data.size(); offset += sizeof(uint64_t)) {
    const uint64_t value = next_random(seed);
    for (unsigned byte = 0; byte < sizeof(value); ++byte)
      data[offset + byte] = static_cast<uint8_t>(value >> (8 * byte));
  }
  return data;
}

uint64_t hash_data(std::span<const uint8_t> data) {
  uint64_t hash = 1469598103934665603ULL;
  for (uint8_t byte : data) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

uint64_t hash_data(const amdgpu::GpuMemory &memory) {
  std::array<uint8_t, kDataSize> data{};
  memory.read_block(kDataBase, data);
  return hash_data(data);
}

uint64_t expected_cache_hash(const Scenario &scenario) {
  std::array<uint8_t, kDataSize> data = initial_data(scenario.input);
  uint32_t value = 0;
  for (unsigned byte = 0; byte < sizeof(value); ++byte)
    value |= static_cast<uint32_t>(data[byte]) << (8 * byte);
  value += scenario.packets * scenario.workgroups * scenario.workgroup_size;
  for (unsigned byte = 0; byte < sizeof(value); ++byte)
    data[byte] = static_cast<uint8_t>(value >> (8 * byte));
  return hash_data(data);
}

struct ModeConfig {
  std::string_view name;
  bool topology_threads;
  uint32_t dispatch_threads;
};

constexpr std::array<ModeConfig, 4> kModes{{
    {"serial", false, 1},
    {"topology", true, 1},
    {"single-engine-pooled", false, 3},
    {"topology-pooled", true, 3},
}};

struct ModeResult {
  std::string_view name;
  uint64_t pending_packets;
  uint64_t completed_workgroups;
  uint64_t dispatched_workgroups;
  std::vector<uint64_t> dispatched_per_xcd;
  uint64_t waves;
  uint64_t instructions;
  uint64_t data_hash;
  uint32_t engine_threads;
  uint32_t dispatch_threads;
  simdojo::ExitReason exit_reason;
  simdojo::ExitCause exit_cause;
  int exit_code;
  bool terminal_faulted = false;
};

bool primary_stalled(const ModeResult &result) {
  return result.exit_cause == simdojo::ExitCause::ACTIVE_PRIMARY_STALL;
}

bool budget_limited(const ModeResult &result) {
  return !result.terminal_faulted && result.pending_packets != 0 &&
         result.exit_cause == simdojo::ExitCause::MAX_TICKS;
}

bool mode_complete(const ModeResult &result) {
  return result.exit_reason == simdojo::ExitReason::COMPLETED && result.exit_code == 0 &&
         !budget_limited(result) && !primary_stalled(result) && !result.terminal_faulted &&
         result.pending_packets == 0;
}

ModeResult run_mode(const Scenario &scenario, const ModeConfig &mode) {
  util::StringDiagnostic diagnostic;
  util::FailureOr<config::LoadedConfig> loaded_result = config::load_config_from_string(
      fuzz_config(scenario.target, scenario.xcd_count), kEmbeddedSchema, diagnostic.emitter());
  if (loaded_result.failed())
    throw std::runtime_error("concurrent dispatch fuzz config failed: " + diagnostic.message());
  config::LoadedConfig loaded = std::move(loaded_result).value();
  SoC *soc = loaded.soc();
  amdgpu::GpuMemory *memory = loaded.memory();
  if (!soc || !memory || soc->num_xcds() != scenario.xcd_count)
    throw std::runtime_error("concurrent dispatch fuzz topology is incomplete");

  soc->set_dispatch_threads(mode.dispatch_threads);
  const uint32_t effective_dispatch_threads = soc->dispatch_threads();
  if (effective_dispatch_threads != mode.dispatch_threads)
    throw std::runtime_error("concurrent dispatch fuzz topology capped the CPU dispatch pool");
  const uint32_t engine_threads =
      mode.topology_threads
          ? amdgpu::clamp_xcd_partition_count(soc, static_cast<uint32_t>(soc->num_xcds()))
          : 1u;
  loaded.engine_config.num_threads = engine_threads;
  auto engine = std::make_unique<simdojo::SimulationEngine>(loaded.engine_config);
  engine->topology().set_root(loaded.take_root());
  if (loaded.wire_links(engine->topology(), diagnostic.emitter()).failed())
    throw std::runtime_error("concurrent dispatch fuzz link wiring failed: " +
                             diagnostic.message());
  if (engine_threads > 1 &&
      !amdgpu::partition_topology_by_xcds(engine->topology(), soc, engine_threads))
    throw std::runtime_error("concurrent dispatch fuzz could not partition the XCD topology");
  if (engine->create(diagnostic.emitter()).failed())
    throw std::runtime_error("concurrent dispatch fuzz engine creation failed: " +
                             diagnostic.message());

  install_input_kernel(*memory, scenario);
  const std::array<uint8_t, kDataSize> data = initial_data(scenario.input);
  memory->load_image(data.data(), data.size(), kDataBase);
  if (scenario.prime_caches) {
    std::array<uint8_t, sizeof(uint32_t)> sink{};
    for (uint32_t xcd = 0; xcd < soc->num_xcds(); ++xcd) {
      amdgpu::L2Cache *l2 = soc->xcd(xcd)->l2_cache();
      if (!l2)
        throw std::runtime_error("concurrent dispatch fuzz target has no L2 cache");
      for (uint32_t line = 0; line < scenario.prime_lines; ++line) {
        const uint64_t address = kDataBase + line * amdgpu::L2Cache::LINE_SIZE;
        if (l2->read(address, sink.data(), sink.size()) != amdgpu::VmAccessOutcome::Complete)
          throw std::runtime_error("concurrent dispatch fuzz could not prime an XCD cache");
      }
    }
  }

  auto translator = std::make_shared<BoundedFuzzAddressSpace>();
  auto physical_memory = std::make_shared<amdgpu::GpuMemoryPhysicalAccess>(*memory);
  const amdgpu::AddressSpaceHandle fuzz_address_space =
      soc->gpu_vm().register_unrouted_address_space(1, translator, physical_memory);
  if (!fuzz_address_space)
    throw std::runtime_error("concurrent dispatch fuzz address-space registration failed");

  auto observer_plugin = std::make_unique<FuzzObserver>(fuzz_address_space, *memory, scenario);
  FuzzObserver *observer = observer_plugin.get();
  auto plugins = std::make_shared<ExecutionPluginGroup>(PluginSinkConfig{});
  if (!plugins->add(std::move(observer_plugin)))
    throw std::runtime_error("concurrent dispatch fuzz could not attach its observer");
  soc->set_plugin_group(plugins);

  amdgpu::CommandProcessor *owner = soc->assign_queue_owner_cp(scenario.queue_ordinal);
  if (!owner)
    throw std::runtime_error("concurrent dispatch fuzz has no queue owner");
  QueueDriver queue(memory, owner, scenario.queue_ordinal + 1, scenario.xcd_fanout);
  for (uint32_t packet_index = 0; packet_index < scenario.packets; ++packet_index) {
    hsa_kernel_dispatch_packet_t packet{};
    packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    if (scenario.barrier && packet_index != 0)
      packet.header |= 1u << HSA_PACKET_HEADER_BARRIER;
    if (scenario.agent_acquire)
      packet.header |= HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    if (scenario.agent_release)
      packet.header |= HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    packet.setup = 1;
    packet.workgroup_size_x = scenario.workgroup_size;
    packet.workgroup_size_y = 1;
    packet.workgroup_size_z = 1;
    packet.grid_size_x = scenario.workgroups * scenario.workgroup_size;
    packet.grid_size_y = 1;
    packet.grid_size_z = 1;
    packet.kernel_object = kKernelAddress;
    packet.kernarg_address = reinterpret_cast<void *>(kDataBase);
    packet.completion_signal.handle = completion_signal_address(packet_index);
    memory->write64(packet.completion_signal.handle + kSignalValueOffset, kPendingSignalValue);
    queue.submit(packet);
  }

  const simdojo::ExitStatus exit = engine->run();
  soc->flush_all();
  std::vector<uint64_t> histogram = soc->dispatched_workgroups_per_xcd();
  if (histogram.size() != scenario.xcd_count)
    throw std::runtime_error("concurrent dispatch fuzz produced an invalid XCD histogram");
  uint64_t dispatched = 0;
  for (uint64_t count : histogram)
    dispatched += count;
  if (!observer->valid_progress(scenario.packets))
    throw std::runtime_error(
        "concurrent dispatch fuzz observed an invalid dispatch or workgroup ID");
  if (observer->barrier_violation())
    throw std::runtime_error(
        "concurrent dispatch fuzz placed work before its barrier predecessor retired");
  if (observer->published_early())
    throw std::runtime_error(
        "concurrent dispatch fuzz published completion before workgroup retirement");
  uint64_t pending_packets = 0;
  uint64_t completed_workgroups = 0;
  for (uint32_t packet_index = 0; packet_index < scenario.packets; ++packet_index) {
    const uint64_t address = completion_signal_address(packet_index) + kSignalValueOffset;
    const uint64_t signal = memory->read64(address);
    const uint64_t retired = observer->completed(packet_index);
    if (retired > scenario.workgroups ||
        (signal != kPendingSignalValue &&
         (signal != kCompletedSignalValue || retired != scenario.workgroups))) {
      throw std::runtime_error(
          std::format("concurrent dispatch fuzz packet {} publication invariant failed "
                      "(signal={:#x}, retired={}/{})",
                      packet_index, signal, retired, scenario.workgroups));
    }
    pending_packets += signal == kPendingSignalValue;
    completed_workgroups += retired;
  }
  ModeResult result{
      .name = mode.name,
      .pending_packets = pending_packets,
      .completed_workgroups = completed_workgroups,
      .dispatched_workgroups = dispatched,
      .dispatched_per_xcd = std::move(histogram),
      .waves = observer->waves(),
      .instructions = observer->instructions(),
      .data_hash = hash_data(*memory),
      .engine_threads = engine_threads,
      .dispatch_threads = effective_dispatch_threads,
      .exit_reason = exit.reason,
      .exit_cause = exit.cause,
      .exit_code = exit.code,
  };
  // A fatal exit can leave peer-XCD faults in the owner CP's inbox. Shutdown
  // drains them after the simulator workers have stopped.
  engine->shutdown();
  result.terminal_faulted = queue.terminal_faulted();
  return result;
}

void validate_mode(const Scenario &scenario, const ModeResult &result) {
  const uint64_t workgroups = uint64_t{scenario.packets} * scenario.workgroups;
  if (result.completed_workgroups > workgroups || result.dispatched_workgroups > workgroups ||
      result.completed_workgroups > result.dispatched_workgroups ||
      (mode_complete(result) &&
       (result.completed_workgroups != workgroups || result.dispatched_workgroups != workgroups))) {
    throw std::runtime_error("concurrent dispatch fuzz bookkeeping invariant failed");
  }
  if (result.terminal_faulted && result.pending_packets == 0)
    throw std::runtime_error("concurrent dispatch fuzz faulted queue published every completion");
  if (mode_complete(result)) {
    for (uint32_t xcd = 0; xcd < scenario.xcd_count; ++xcd) {
      const uint32_t per_packet = scenario.xcd_fanout
                                      ? scenario.workgroups / scenario.xcd_count +
                                            (xcd < scenario.workgroups % scenario.xcd_count)
                                      : (xcd == scenario.queue_ordinal ? scenario.workgroups : 0);
      const uint64_t expected = uint64_t{per_packet} * scenario.packets;
      if (result.dispatched_per_xcd[xcd] != expected)
        throw std::runtime_error(
            std::format("concurrent dispatch fuzz {} placed {} workgroups on XCD {}, expected {}",
                        result.name, result.dispatched_per_xcd[xcd], xcd, expected));
    }
  }
  if (!detail::primary_lifecycle_is_valid(result.pending_packets == 0, result.terminal_faulted,
                                          result.exit_cause))
    throw std::runtime_error(std::format(
        "concurrent dispatch fuzz {} primary lifecycle disagrees with {} pending packets "
        "(cause={}, faulted={})",
        result.name, result.pending_packets, static_cast<unsigned>(result.exit_cause),
        result.terminal_faulted));
}

bool completed_results_match(const ModeResult &left, const ModeResult &right) {
  return left.pending_packets == right.pending_packets &&
         left.completed_workgroups == right.completed_workgroups &&
         left.dispatched_workgroups == right.dispatched_workgroups && left.waves == right.waves &&
         left.instructions == right.instructions && left.data_hash == right.data_hash &&
         left.exit_reason == right.exit_reason && left.exit_cause == right.exit_cause &&
         left.exit_code == right.exit_code && left.terminal_faulted == right.terminal_faulted;
}

bool raw_outcomes_match(const ModeResult &left, const ModeResult &right) {
  if (left.terminal_faulted != right.terminal_faulted)
    return false;
  // Per-mode validation already checked fault teardown and publication.
  // In-flight work may still reach a fatal instruction after the queue faults.
  if (left.terminal_faulted)
    return true;
  if (left.exit_reason != right.exit_reason || left.exit_code != right.exit_code)
    return false;
  // Work performed after a fatal request depends on how much of a pooled quantum
  // was already in flight. The fatal classification is the portable outcome.
  if (left.exit_reason != simdojo::ExitReason::COMPLETED || left.exit_code != 0)
    return true;
  // Raw kernels can branch on racy data. If either mode is incomplete, its
  // signal and progress can legitimately depend on host scheduling. Completed
  // modes still have to agree on the GPU-visible retirement bookkeeping.
  if (!mode_complete(left) || !mode_complete(right))
    return true;
  return left.completed_workgroups == right.completed_workgroups &&
         left.dispatched_workgroups == right.dispatched_workgroups;
}

bool has_differential_oracle(const Scenario &scenario) {
  const std::vector<uint8_t> expected = cache_program(scenario.target);
  return scenario.program.size() == expected.size() &&
         std::equal(scenario.program.begin(), scenario.program.end(), expected.begin());
}

std::string describe_mode(const ModeResult &result) {
  return std::format(
      "engine_threads={} dispatch_threads={} pending_packets={} completed={} dispatched={} "
      "waves={} "
      "instructions={} hash={:#x} exit={}:{} cause={} faulted={} budget={} stalled={}",
      result.engine_threads, result.dispatch_threads, result.pending_packets,
      result.completed_workgroups, result.dispatched_workgroups, result.waves, result.instructions,
      result.data_hash, static_cast<unsigned>(result.exit_reason), result.exit_code,
      static_cast<unsigned>(result.exit_cause), result.terminal_faulted, budget_limited(result),
      primary_stalled(result));
}

} // namespace

namespace detail {

bool primary_lifecycle_is_valid(bool published_all_completions, bool terminal_faulted,
                                simdojo::ExitCause cause) {
  switch (cause) {
  case simdojo::ExitCause::ACTIVE_PRIMARY_STALL:
    return !published_all_completions && !terminal_faulted;
  case simdojo::ExitCause::ALL_PRIMARIES_COMPLETED:
    return published_all_completions || terminal_faulted;
  case simdojo::ExitCause::MAX_TICKS:
  case simdojo::ExitCause::QUIESCENCE:
  case simdojo::ExitCause::NONE:
    return true;
  }
  return false;
}

} // namespace detail

RunStats run_input(std::span<const uint8_t> input) {
  if (input.empty() || input.size() > kMaxInputSize)
    throw std::invalid_argument("concurrent dispatch fuzz input must contain 1..1024 bytes");
  const std::optional<Scenario> decoded = decode_scenario(input);
  if (!decoded)
    return {.inputs = 1};
  const Scenario &scenario = *decoded;

  std::vector<ModeResult> results;
  for (const ModeConfig &mode : kModes)
    if (!mode.topology_threads || scenario.xcd_count > 1)
      results.push_back(run_mode(scenario, mode));
  for (const ModeResult &result : results)
    validate_mode(scenario, result);

  const ModeResult &serial = results.front();
  const ModeResult &pooled = results.back();
  const bool any_budget_limited = std::ranges::any_of(results, budget_limited);
  const bool completed = std::ranges::all_of(results, mode_complete);
  const bool differential = has_differential_oracle(scenario);
  const auto every_result_matches = [&](auto predicate) {
    return std::ranges::all_of(results, [&](const ModeResult &result) {
      return &result == &serial || predicate(serial, result);
    });
  };
  const uint64_t expected_hash = differential ? expected_cache_hash(scenario) : 0;
  const bool exact_result =
      !differential || std::ranges::all_of(results, [&](const ModeResult &result) {
        return result.data_hash == expected_hash;
      });

  std::string result_description;
  for (const ModeResult &result : results) {
    if (!result_description.empty())
      result_description += ' ';
    result_description += std::format("{}=[{}]", result.name, describe_mode(result));
  }
  if (differential &&
      (!completed || !exact_result || !every_result_matches(completed_results_match))) {
    throw std::runtime_error(std::format(
        "concurrent dispatch fuzz differential {} cache kernel failed (expected_hash={:#x}): {}",
        execution_fuzz::target_name(scenario.target), expected_hash, result_description));
  }
  if (!differential && !every_result_matches(raw_outcomes_match)) {
    throw std::runtime_error(std::format(
        "concurrent dispatch fuzz raw-kernel outcomes disagree: {}", result_description));
  }

  uint64_t completed_modes = 0;
  uint64_t budget_limited_modes = 0;
  uint64_t primary_stalled_modes = 0;
  uint64_t fatal_exit_modes = 0;
  uint64_t terminal_faulted_modes = 0;
  uint64_t published_completions = 0;
  uint64_t instructions = 0;
  for (const ModeResult &result : results) {
    completed_modes += mode_complete(result);
    budget_limited_modes += budget_limited(result);
    primary_stalled_modes += primary_stalled(result);
    fatal_exit_modes +=
        result.exit_reason == simdojo::ExitReason::EXIT_REQUEST && result.exit_code != 0;
    terminal_faulted_modes += result.terminal_faulted;
    published_completions += scenario.packets - result.pending_packets;
    instructions += result.instructions;
  }

  const uint64_t workgroups = uint64_t{scenario.packets} * scenario.workgroups;
  return {
      .inputs = 1,
      .targets = 1,
      .modes = results.size(),
      .completed_modes = completed_modes,
      .budget_limited_modes = budget_limited_modes,
      .primary_stalled_modes = primary_stalled_modes,
      .fatal_exit_modes = fatal_exit_modes,
      .terminal_faulted_modes = terminal_faulted_modes,
      .published_completions = published_completions,
      .instructions = instructions,
      .packets = scenario.packets,
      .workgroups = workgroups,
      .workitems = workgroups * scenario.workgroup_size,
      .kernel_bytes = scenario.program.size(),
      .completed_inputs = completed ? 1u : 0u,
      .incomplete_inputs = completed ? 0u : 1u,
      .budget_limited_inputs = any_budget_limited ? 1u : 0u,
      .multi_xcd_inputs = scenario.xcd_count > 1 ? 1u : 0u,
      .cache_primed_inputs = scenario.prime_caches ? 1u : 0u,
      .fanout_inputs = scenario.xcd_fanout ? 1u : 0u,
      .differential_inputs = differential ? 1u : 0u,
      .pooled_dispatch_threads = pooled.dispatch_threads,
  };
}

std::vector<uint8_t> random_input(uint64_t seed) {
  const uint64_t original_seed = seed;
  std::vector<uint8_t> input(kGeneratedInputSize);
  for (std::size_t offset = 0; offset < kHeaderSize; offset += sizeof(uint64_t)) {
    const uint64_t value = next_random(seed);
    for (unsigned byte = 0; byte < sizeof(value); ++byte)
      input[offset + byte] = static_cast<uint8_t>(value >> (8 * byte));
  }
  const auto target = static_cast<execution_fuzz::FuzzTarget>(
      original_seed % static_cast<uint8_t>(execution_fuzz::FuzzTarget::Count));
  input[0] = static_cast<uint8_t>(target);
  const std::vector<uint8_t> prefix = cache_program(target);
  if (prefix.size() > input.size() - kHeaderSize)
    throw std::logic_error(
        "concurrent dispatch fuzz canonical kernel exceeds the generated input size");
  std::copy(prefix.begin(), prefix.end(), input.begin() + kHeaderSize);
  // Keep a regular stream under the differential exact-result oracle. The
  // remaining inputs carry unconstrained target-specific instruction bytes.
  if ((input[6] & 3) == 0) {
    input.resize(kHeaderSize + prefix.size());
    return input;
  }
  const std::size_t tail_size = input.size() - kHeaderSize - prefix.size();
  const util::FailureOr<std::vector<uint8_t>> maybe_program = execution_fuzz::random_input(
      next_random(seed), tail_size + execution_fuzz::kHeaderSize, target);
  assert(maybe_program.succeeded());
  const std::vector<uint8_t> &program = maybe_program.value();
  std::copy(program.begin() + execution_fuzz::kHeaderSize, program.end(),
            input.begin() + kHeaderSize + prefix.size());
  return input;
}

std::vector<std::vector<uint8_t>> seed_corpus() {
  std::vector<std::vector<uint8_t>> seeds;
  for (uint8_t index = 0; index < static_cast<uint8_t>(execution_fuzz::FuzzTarget::Count);
       ++index) {
    const auto target = static_cast<execution_fuzz::FuzzTarget>(index);
    std::vector<uint8_t> seed(kHeaderSize + sizeof(uint32_t));
    seed[0] = index;
    seed[1] = static_cast<uint8_t>(index & 0xf);
    if (representative_xcd_count(target) > 1 && (index & 1) != 0)
      seed[1] |= kXcdFanoutBit;
    seed[2] = index;
    seed[3] = index * 3;
    seed[4] = (seed[1] & (kBarrierBit | kXcdFanoutBit)) != 0 ? 1 : 0;
    // Exercise owner-only routing on XCD 1 as well as on XCD 0.
    seed[5] = target == execution_fuzz::FuzzTarget::Gfx942 ? 1 : index;
    seed[7] = index;
    const std::vector<uint8_t> program = cache_program(target);
    seed.resize(kHeaderSize + program.size());
    std::copy(program.begin(), program.end(), seed.begin() + kHeaderSize);
    seeds.push_back(std::move(seed));
  }
  return seeds;
}

} // namespace rocjitsu::concurrent_dispatch_fuzz
