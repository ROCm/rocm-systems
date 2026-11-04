// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/plugin.h"

#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"

#include "rocjitsu/vm/plugins/memory_access_observation.h"
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"

#include <algorithm>
#include <cassert>
#include <format>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace rocjitsu::plugins::race_detector {

namespace {

InstructionSnapshot capture_instruction(uint64_t pc, const Instruction &inst) {
  static_assert(InstructionSnapshot::kMaxWords * sizeof(uint32_t) ==
                amdgpu::InstructionCache::kFetchBytes);
  if (!inst.raw_encoding())
    throw std::logic_error(
        std::format("race detector instruction at PC {:#x} has no encoding", pc));
  if (inst.size() <= 0 || inst.size() % sizeof(uint32_t) != 0 ||
      static_cast<size_t>(inst.size()) > InstructionSnapshot::kMaxWords * sizeof(uint32_t))
    throw std::logic_error(std::format(
        "race detector instruction at PC {:#x} has invalid encoding size {}", pc, inst.size()));
  return {pc, std::span(inst.raw_encoding(), inst.size() / sizeof(uint32_t))};
}

// onAmdgpuBeforeExecuteInstruction() records each instruction before it can
// access registers or memory, so the newest trace entry is the instruction that
// is executing. Use it instead of wf.pc, which some instructions change before
// they finish accessing registers (for example, s_swappc_b64).
const InstructionSnapshot &current_instruction(const RaceWavefrontState &state,
                                               const amdgpu::Wavefront &wf) {
  if (state.trace.size() == 0)
    throw std::logic_error(
        std::format("race detector has no current instruction at PC {:#x}", wf.pc));
  return state.trace[state.trace.size() - 1];
}

void warn_cluster_peer_writes_ignored_once() {
  static std::once_flag warned;
  std::call_once(warned, [] {
    util::Logger::warn(
        "race detector does not model cluster LDS multicast peer writes; peer writes are ignored");
  });
}

uint8_t vector_memory_byte_mask(const amdgpu::VectorMemState &state, const amdgpu::Wavefront &wave,
                                uint32_t register_offset = 0) {
  if (state.buffer_components && state.buffer_d16) {
    if (state.is_load && wave.cu().sram_ecc())
      return ExecutionPlugin::kFullByteMask;
    if (state.d16_hi)
      return ExecutionPlugin::kHighHalfByteMask;
    return register_offset * 2 + 1 < state.buffer_components ? ExecutionPlugin::kFullByteMask
                                                             : ExecutionPlugin::kLowHalfByteMask;
  }
  if (state.is_load && wave.cu().sram_ecc() && (state.d16_lo || state.d16_hi))
    return ExecutionPlugin::kFullByteMask;
  if (state.d16_lo)
    return ExecutionPlugin::kLowHalfByteMask;
  if (state.d16_hi)
    return ExecutionPlugin::kHighHalfByteMask;
  return ExecutionPlugin::kFullByteMask;
}

bool supports_counter_capacity(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
    return true;
  default:
    return false;
  }
}

std::optional<std::vector<uint32_t>>
validated_load_destinations(const amdgpu::VectorMemState &state, const amdgpu::Wavefront &wave) {
  const uint32_t vgpr_count = state.destination_vgpr_count();
  const uint32_t ds2_vgpr_count = state.ds2_active ? state.ds2_destination_vgpr_count() : 0;
  const amdgpu::RegisterAccess registers_for(wave);
  std::vector<uint32_t> registers;
  registers.reserve(vgpr_count + ds2_vgpr_count);
  auto append_range = [&](uint32_t physical_base, uint32_t count) {
    if (!state.is_block_transfer && !registers_for.owns_vgpr_range(physical_base, count))
      return false;
    const uint32_t logical_base = physical_base - wave.vgpr_alloc().base;
    // Blocks validate each enabled word independently; other loads retain their
    // full-span contract. An empty result still represents a counter event.
    for (uint32_t i = 0; i < count; ++i)
      if (state.block_dword_enabled(i) &&
          (!state.is_block_transfer || registers_for.owns_vgpr_range(physical_base + i, 1)))
        registers.push_back(logical_base + i);
    return true;
  };

  if (!append_range(state.dst_reg_base, vgpr_count) ||
      (state.ds2_active && !append_range(state.ds2_dst_reg_base, ds2_vgpr_count)))
    return std::nullopt;
  return registers;
}

std::optional<amdgpu::MemoryIssueInfo>
routed_flat_issue_info(const amdgpu::MemoryAccessObservation &access,
                       const amdgpu::MemoryIssueInfo &decoded) {
  if (access.decoded_space != amdgpu::DecodedMemorySpace::FLAT ||
      (access.route != amdgpu::MemoryRoute::GLOBAL && access.route != amdgpu::MemoryRoute::LOCAL))
    return std::nullopt;

  if (access.request_lane_mask == 0)
    return decoded.for_flat_memory_domains(false, false);

  const uint64_t shared_lanes =
      (access.flat_local_lane_mask | access.flat_dds_lane_mask) & access.request_lane_mask;
  // Functional execution splits mixed requests by address space. The plugin's
  // single mixed event retains both obligations until both domains are ready.
  if (shared_lanes != 0 && shared_lanes != access.request_lane_mask)
    return decoded;
  if (access.route != (shared_lanes ? amdgpu::MemoryRoute::LOCAL : amdgpu::MemoryRoute::GLOBAL))
    return std::nullopt;

  return decoded.for_flat_memory_domains(shared_lanes == 0, shared_lanes != 0);
}

} // namespace

MarkedInstruction findConflict(const RaceViolation &v, RaceDetector &detector) {
  EventId event_id = v.conflictingEvent;
  if (!detector.events().contains(event_id))
    throw std::out_of_range("race violation references an unavailable conflicting event");
  return {detector.events().instruction(event_id), detector.events().waveId(event_id).value, -1};
}

std::string formatTrace(const RingBuffer<InstructionSnapshot, 256> &trace,
                        const std::function<std::string(const InstructionSnapshot &)> &disassemble,
                        std::optional<MarkedInstruction> conflict, MarkedInstruction read) {
  auto isMarked = [&](const InstructionSnapshot &instruction) {
    return instruction == read.instruction || (conflict && instruction == conflict->instruction);
  };

  size_t n = trace.size();
  size_t first = n;
  bool conflict_found = false;
  for (size_t i = 0; i < n; ++i) {
    if (isMarked(trace[i])) {
      first = i;
      break;
    }
  }
  if (conflict) {
    for (size_t i = 0; i < n; ++i)
      if (trace[i] == conflict->instruction) {
        conflict_found = true;
        break;
      }
  }

  std::ostringstream oss;

  if (conflict && !conflict_found) {
    oss << "  ==>  0x" << std::hex << conflict->instruction.pc() << std::dec << "  ";
    const auto d = disassemble(conflict->instruction);
    if (!d.empty())
      oss << d << "  ";
    oss << "(before trace window)";
    oss << "  ; <-- wave " << conflict->wave;
    if (conflict->lane >= 0)
      oss << " lane " << conflict->lane;
    oss << "\n       ... " << first << " instructions not recorded ...\n";
  }

  size_t last = n;
  for (size_t i = first; i < n; ++i) {
    if (trace[i] == read.instruction) {
      last = i + 1;
      break;
    }
  }

  constexpr size_t MAX_PRINT_SIZE = 32;
  size_t span = (last > first) ? last - first : 0;

  auto emit = [&](size_t i) {
    const auto &instruction = trace[i];
    bool is_conflict = conflict && instruction == conflict->instruction;
    bool is_read = instruction == read.instruction;
    oss << ((is_conflict || is_read) ? "  ==>  " : "       ");
    oss << "0x" << std::hex << instruction.pc() << std::dec << "  " << disassemble(instruction);
    if (is_conflict) {
      oss << "  ; <-- wave " << conflict->wave;
      if (conflict->lane >= 0)
        oss << " lane " << conflict->lane;
    }
    if (is_read) {
      oss << "  ; <-- wave " << read.wave;
      if (read.lane >= 0)
        oss << " lane " << read.lane;
    }
    oss << "\n";
  };

  if (span <= MAX_PRINT_SIZE) {
    for (size_t i = first; i < last; ++i)
      emit(i);
  } else {
    size_t half = MAX_PRINT_SIZE / 2;
    for (size_t i = first; i < first + half; ++i)
      emit(i);
    oss << "       ... " << (span - MAX_PRINT_SIZE) << " instructions elided ...\n";
    for (size_t i = last - half; i < last; ++i)
      emit(i);
  }
  return oss.str();
}

RaceDetectorPlugin::RaceDetectorPlugin(const char * /*config_json*/) : ExecutionPlugin("race") {}

RaceDetectorPlugin::~RaceDetectorPlugin() { sink().write(getSummary()); }

std::string RaceDetectorPlugin::getSummary() const {
  const char *banner = "\n========================================\n"
                       " ROCJITSU RACE DETECTION SUMMARY\n"
                       "========================================\n";
  if (!observed_races_.empty()) {
    return std::string(banner) + "  " + std::to_string(observed_races_.size()) +
           " race(s) detected\n"
           "========================================\n";
  }
  return std::string(banner) + "  No races detected.\n"
                               "========================================\n";
}

void RaceDetectorPlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  std::lock_guard<std::mutex> lock(report_mutex_);
  KernelNames kernel_names{info.kernelNameOrUnknown(), info.kernelSymbolOrUnknown()};
  dispatch_kernel_names_[info.dispatch_id] = kernel_names;
  sink().write(std::format("[rocjitsu] Kernel dispatch: \"{}\" symbol=\"{}\"\n", kernel_names.name,
                           kernel_names.symbol));
}

void RaceDetectorPlugin::onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                                     uint32_t physical_vgpr_count,
                                                     uint32_t physical_sgpr_count,
                                                     std::span<amdgpu::Wavefront *> wavefronts) {
  uint32_t num_waves = static_cast<uint32_t>(wavefronts.size());
  WorkgroupKey key{dispatch_id, wg_id};

  std::vector<amdgpu::Wavefront *> wf_ptrs(wavefronts.begin(), wavefronts.end());
  auto handler = [this, wf_ptrs, dispatch_id](RaceViolation v) {
    assert(v.wave >= 0 && static_cast<size_t>(v.wave) < wf_ptrs.size() &&
           "wave index out of range");
    amdgpu::Wavefront *wf = wf_ptrs[v.wave];
    auto *ws = get_state(wf);
    assert(ws && ws->race_state && "no wavefront state for race");
    const InstructionSnapshot current = current_instruction(*ws, *wf);
    uint64_t pc = current.pc();

    {
      std::lock_guard<std::mutex> lock(report_mutex_);
      if (observed_races_.count({dispatch_id, pc}))
        return;
    }

    auto *detector = ws->race_state->getDetector();
    MarkedInstruction conflict = findConflict(v, *detector);

    std::ostringstream oss;
    if (v.space == RaceViolation::Space::VGPR)
      oss << "Race on VGPR v" << v.index;
    else if (v.space == RaceViolation::Space::SGPR)
      oss << "Race on SGPR s" << v.index;
    else if (v.space == RaceViolation::Space::TTMP)
      oss << "Race on TTMP ttmp" << v.index;
    else
      oss << "Race on LDS byte " << v.index;
    oss << " [workgroup (" << v.workgroupId.x << ", " << v.workgroupId.y << ", " << v.workgroupId.z
        << "), wave " << v.wave;
    if (v.space != RaceViolation::Space::SGPR && v.space != RaceViolation::Space::TTMP)
      oss << ", lane " << v.lane;
    oss << "]\n";

    MarkedInstruction access_mark{current, v.wave, v.lane};
    oss << formatTrace(
        ws->trace,
        [wf](const InstructionSnapshot &instruction) {
          return wf->disassemble_instruction(instruction.encoding(), instruction.pc());
        },
        conflict, access_mark);

    {
      std::lock_guard<std::mutex> lock(report_mutex_);
      bool is_new = !observed_races_.count({dispatch_id, pc}) &&
                    !observed_races_.count({dispatch_id, conflict.instruction.pc()});
      observed_races_.emplace(dispatch_id, pc);
      if (is_new) {
        observed_races_.emplace(dispatch_id, conflict.instruction.pc());
        const char *space = v.space == RaceViolation::Space::VGPR   ? "VGPR"
                            : v.space == RaceViolation::Space::SGPR ? "SGPR"
                            : v.space == RaceViolation::Space::TTMP ? "TTMP"
                                                                    : "LDS";
        auto kernel_name_iter = dispatch_kernel_names_.find(dispatch_id);
        const KernelNames kernel_names =
            kernel_name_iter == dispatch_kernel_names_.end()
                ? KernelNames{kUnknownKernelIdentity, kUnknownKernelIdentity}
                : kernel_name_iter->second;
        const char *access = v.isWrite ? "write" : "read";
        sink().write(std::format(
            "RACE kernel={} symbol={} dispatch={} type={} access={} reg={} wave={} "
            "lane={} "
            "wg={},{},{} "
            "conflict=unknown\n{}END_RACE\n",
            kernel_names.name, kernel_names.symbol, dispatch_id, space, access, v.index, v.wave,
            v.lane, v.workgroupId.x, v.workgroupId.y, v.workgroupId.z, oss.str()));
      }
    }
  };

  // Multi-XCD: workgroups from the same dispatch can arrive on different
  // partition threads, so the detector map needs protection.
  std::lock_guard<std::mutex> lock(dispatch_mutex_);
  detectors_[key] = std::make_unique<RaceDetector>(
      static_cast<int>(num_waves), static_cast<int>(physical_vgpr_count),
      static_cast<int>(physical_sgpr_count), Dim3d(static_cast<int>(wg_id)), std::move(handler),
      counterCapacitiesForArch(wavefronts.front()->cu().arch()),
      wavefronts.front()->lds().size_bytes());

  auto &det = *detectors_[key];
  for (uint32_t w = 0; w < num_waves; ++w) {
    auto state = std::make_unique<RaceWavefrontState>();
    state->race_state = &det.getWaveRaceState(static_cast<int>(w));
    wavefronts[w]->set_plugin_state(slot_index(), std::move(state));
  }
}

void RaceDetectorPlugin::onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access,
                                                    const Instruction &inst,
                                                    amdgpu::Wavefront &wf) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  auto *rs = s->race_state;
  auto *detector = rs->getDetector();
  auto waveId = rs->getWaveId();
  const InstructionSnapshot &instruction = current_instruction(*s, wf);
  const auto *decoded = inst.amdgpu_memory_issue_info();
  assert(decoded && "memory instruction reached the race detector without issue metadata");
  if (!decoded)
    return;
  const auto routed = routed_flat_issue_info(access, *decoded);
  const auto &issue = routed ? *routed : *decoded;
  const auto obligations = issue.counter_obligations();
  const MemoryOrderClass memoryOrder = memoryOrderForObligations(obligations);
  if (routed && access.request_lane_mask != 0 && supports_counter_capacity(wf.cu().arch()))
    rs->prepareForMemoryIssue(issue);

  const uint64_t sharedLanes = access.flat_local_lane_mask | access.flat_dds_lane_mask;
  const bool mixedFlat = inst.data()->tag() == amdgpu::GLOBAL_MEM && sharedLanes != 0;
  if (inst.data()->tag() == amdgpu::LOCAL_MEM || mixedFlat) {
    auto &d = *inst.data_as<amdgpu::VectorMemState>();
    const uint64_t execMask = d.exec_mask;
    // DDS and allocation-OOB lanes participate in the instruction's
    // dependencies, but do not access bytes in the ordinary LDS allocation.
    const uint64_t ldsMask = (mixedFlat ? execMask & access.flat_local_lane_mask : execMask) &
                             ~(access.flat_dds_lane_mask | access.flat_local_oob_lane_mask);
    if (execMask == 0)
      return;
    auto type = d.is_load ? MemoryEventType::LDS_TO_VGPR : MemoryEventType::VGPR_TO_LDS;
    const uint32_t perLaneBytes = d.num_elems * d.elem_size;

    std::vector<uint32_t> registers;
    if (d.is_load) {
      auto destinations = validated_load_destinations(d, wf);
      if (!destinations)
        return;
      registers = std::move(*destinations);
    }

    uint32_t laneAddrs[64]{};
    uint32_t secondLaneAddrs[64]{};
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(ldsMask & (1ULL << lane)))
        continue;
      auto validate = [&](uint64_t address) {
        auto bytes = detector->ldsBytesInRange(address, perLaneBytes, d.elem_size);
        if (access.flat_local_allocation_end && d.elem_size != 0)
          bytes = amdgpu::whole_element_bytes_in_range(address, bytes, d.elem_size,
                                                       *access.flat_local_allocation_end);
        if (bytes == 0)
          return amdgpu::kInvalidLdsAddress;
        const auto addr = static_cast<uint32_t>(address);
        if (d.is_load)
          detector->validateRead(static_cast<int>(addr), waveId, static_cast<int>(lane), bytes,
                                 memoryOrder);
        else
          detector->validateWrite(static_cast<int>(addr), waveId, static_cast<int>(lane), bytes,
                                  memoryOrder);
        return addr;
      };
      laneAddrs[lane] = validate(d.per_lane_addr[lane]);
      if (d.ds2_active)
        secondLaneAddrs[lane] = validate(d.ds2_per_lane_addr[lane]);
    }
    uint8_t byte_mask = vector_memory_byte_mask(d, wf);
    if (d.is_load) {
      for (uint32_t reg : registers)
        rs->checkVgprWrite(static_cast<int>(reg), execMask, byte_mask, memoryOrder);
    }
    if (obligations.empty())
      return;
    if (d.ds2_active) {
      rs->registerLdsEvent(instruction, type, std::move(registers), execMask, wf.wf_size(),
                           std::span<const uint32_t>(laneAddrs, wf.wf_size()),
                           std::span<const uint32_t>(secondLaneAddrs, wf.wf_size()), perLaneBytes,
                           byte_mask, obligations, memoryOrder, d.elem_size);
    } else {
      rs->registerLdsEvent(instruction, type, std::move(registers), execMask, wf.wf_size(),
                           std::span<const uint32_t>(laneAddrs, wf.wf_size()), perLaneBytes,
                           byte_mask, obligations, memoryOrder, ldsMask, d.elem_size,
                           access.flat_local_allocation_end);
    }
    // A mixed instruction has one counter entry, with all register lanes but
    // only the ordinary LDS lanes' intervals. Do not also register a global event.
    return;
  }

  if (inst.data()->tag() == amdgpu::GLOBAL_MEM) {
    auto &d = *inst.data_as<amdgpu::VectorMemState>();
    if (d.exec_mask == 0)
      return;
    if (d.lds_dst) {
      if (obligations.empty())
        return;
      uint32_t perLaneBytes = d.buffer_components
                                  ? 4u * d.destination_vgpr_count()
                                  : d.num_elems * (d.lds_dword_extend ? 4u : d.elem_size);
      if (d.cluster_multicast && d.cluster_mcast_mask != 0) {
        uint32_t selfMask = amdgpu::cluster_multicast_rank_mask(wf.cluster_rank());
        uint32_t peerMask = d.cluster_mcast_mask & ~selfMask;
        if (peerMask != 0)
          warn_cluster_peer_writes_ignored_once();
        if ((d.cluster_mcast_mask & selfMask) == 0)
          return;
      }
      uint32_t ldsAddrs[64];
      const uint64_t validLaneMask = amdgpu::lds_destination_write_mask(d, wf.cu().arch());
      for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
        ldsAddrs[lane] =
            d.lds_per_lane_addr ? d.per_lane_lds_addr[lane] : d.lds_base + lane * perLaneBytes;
      }
      rs->registerLdsEvent(instruction, MemoryEventType::GLOBAL_TO_LDS, {}, validLaneMask,
                           wf.wf_size(), std::span<const uint32_t>(ldsAddrs, wf.wf_size()),
                           perLaneBytes, 0xF, obligations, memoryOrder);
    } else if (d.is_load) {
      auto destinations = validated_load_destinations(d, wf);
      if (!destinations)
        return;
      std::vector<uint32_t> registers = std::move(*destinations);
      const uint8_t byte_mask = vector_memory_byte_mask(d, wf);
      const uint8_t last_byte_mask =
          registers.empty() ? byte_mask : vector_memory_byte_mask(d, wf, registers.size() - 1);
      for (uint32_t i = 0; i < registers.size(); ++i)
        rs->checkVgprWrite(static_cast<int>(registers[i]), d.exec_mask,
                           vector_memory_byte_mask(d, wf, i), memoryOrder);
      if (obligations.empty())
        return;
      rs->registerEvent(instruction, MemoryEventType::GLOBAL_TO_VGPR, std::move(registers),
                        d.exec_mask, byte_mask, obligations, memoryOrder, last_byte_mask);
    } else if (!obligations.empty()) {
      rs->registerEvent(instruction, MemoryEventType::VGPR_TO_GLOBAL, {}, d.exec_mask, 0xF,
                        obligations, memoryOrder);
    }
  }

  if (inst.data()->tag() == amdgpu::SCALAR_MEM) {
    auto &d = *inst.data_as<amdgpu::ScalarMemState>();
    if (d.is_load) {
      if (const auto reg = d.dst_register.register_ref()) {
        rs->registerScalarLoad(instruction, *reg, wf.exec(), obligations, memoryOrder);
      } else {
        rs->registerEvent(instruction, MemoryEventType::GLOBAL_TO_SGPR, {}, wf.exec(), 0xF,
                          obligations, memoryOrder);
      }
    } else {
      rs->registerEvent(instruction, MemoryEventType::SCALAR_TO_GLOBAL, {}, wf.exec(), 0xF,
                        obligations, memoryOrder);
    }
  }
}

void RaceDetectorPlugin::onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                                               uint64_t lane_mask, uint8_t byte_mask) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  uint32_t logical_reg = physical_reg - wf->vgpr_alloc().base;
  s->race_state->checkVgprReadLanes(static_cast<int>(logical_reg), lane_mask, byte_mask);
}

void RaceDetectorPlugin::onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                                                uint64_t lane_mask, uint8_t byte_mask) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  uint32_t logical_reg = physical_reg - wf->vgpr_alloc().base;
  s->race_state->checkVgprWriteLanes(static_cast<int>(logical_reg), lane_mask, byte_mask);
}

void RaceDetectorPlugin::onAmdgpuReadScalarRegister(const amdgpu::Wavefront *wf, RegisterRef reg) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  s->race_state->checkScalarRead(reg);
}

void RaceDetectorPlugin::onAmdgpuWriteScalarRegister(const amdgpu::Wavefront *wf, RegisterRef reg) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  s->race_state->checkScalarWrite(reg);
}

void RaceDetectorPlugin::onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                                          amdgpu::Wavefront &wf) {
  const InstructionSnapshot instruction = capture_instruction(pc, inst);
  auto *s = get_state(wf);
  assert(s && s->race_state);
  if (supports_counter_capacity(wf.cu().arch())) {
    // FLAT's address-dependent admission is applied after routing. Before its
    // operands are read, neither memory domain is guaranteed to be used.
    if (inst.is_memory_op() && !inst.mnemonic().starts_with("flat_")) {
      const auto *info = inst.amdgpu_memory_issue_info();
      assert(info && "memory instruction reached the race detector without issue metadata");
      if (info && (!info->exec_masked || wf.exec() != 0))
        s->race_state->prepareForMemoryIssue(*info);
    }

    // Messages share LGKMCNT on the capacity-modeled targets but are not memory
    // instructions, so they intentionally remain outside memory issue metadata.
    if (std::string_view(inst.mnemonic()).starts_with("s_sendmsg"))
      s->race_state->prepareForCounterIncrement(amdgpu::WaitCounterType::LGKMCNT);
  }
  s->trace.push(instruction);
}

void RaceDetectorPlugin::onAmdgpuAfterExecuteInstruction(uint64_t /*pc*/, const Instruction &inst,
                                                         amdgpu::Wavefront &wf) {
  // Only explicit waits are handled here. Embedded waits (e.g. VINTERP wait_exp)
  // remain unsupported by the plugin; they must retire events before the
  // instruction's register accesses, as in MemoryWaitScoreboard::before.
  if (!inst.is_waitcnt())
    return;
  const auto fields = amdgpu::WaitCounterPolicy::explicit_wait_fields(inst, wf.cu().arch());
  if (fields.failed() || !fields.value())
    return;

  auto *s = get_state(wf);
  assert(s && s->race_state);
  const auto model = amdgpu::waitcnt_model(wf.cu().arch()).value();
  PendingWaitCount wait;
  for (size_t i = 0; i < fields.value()->size(); ++i) {
    const auto threshold = (*fields.value())[i];
    if (!threshold)
      continue;
    if (const auto counter =
            amdgpu::memory_wait_counter_type(static_cast<amdgpu::WaitCounterKind>(i), model))
      wait.add(*counter, static_cast<int>(*threshold));
  }
  if (!wait.empty())
    s->race_state->dispatch(wait);
}

void RaceDetectorPlugin::onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts) {
  for (auto *wf : wavefronts) {
    auto *s = get_state(wf);
    assert(s && s->race_state);
    s->race_state->flushBarrierPendingEvents();
  }
}

} // namespace rocjitsu::plugins::race_detector
