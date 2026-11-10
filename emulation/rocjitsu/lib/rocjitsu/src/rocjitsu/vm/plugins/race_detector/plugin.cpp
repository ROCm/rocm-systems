// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/plugin.h"

#include "rocjitsu/code/analysis/waitcheck/target.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/sgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/indexed_ds_bounds.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"

#include "rocjitsu/vm/plugins/memory_access_observation.h"
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include "rocjitsu/vm/plugins/tensor_dma_issue_observation.h"

#include <algorithm>
#include <array>
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
  static_assert(InstructionSnapshot::kMaxWords == amdgpu::DecodedInstructionCache::kNumWords);
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

std::optional<std::vector<uint32_t>>
validated_load_destinations(const amdgpu::VectorMemState &state, const amdgpu::Wavefront &wave) {
  const uint32_t vgpr_count = state.destination_vgpr_count();
  if (state.has_feedback_result() &&
      !wave.vgpr_alloc().contains(state.dst_reg_base, vgpr_count + 1))
    return std::nullopt;
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

MemoryOrderClass vgpr_writeback_order(MemoryOrderClass completion, rj_code_arch_t arch) {
  if (completion == MemoryOrderClass::VMEM && !amdgpu::has_ordered_vmem_writeback(arch))
    return MemoryOrderClass::UNORDERED;
  return completion;
}

void apply_wait_fields(WaveRaceState &state, const amdgpu::WaitFields &fields,
                       rj_code_arch_t arch) {
  const auto model = amdgpu::waitcnt_model(arch).value();
  PendingWaitCount wait;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (fields[i]) {
      if (const auto counter =
              amdgpu::memory_wait_counter_type(static_cast<amdgpu::WaitCounterKind>(i), model))
        wait.add(*counter, static_cast<int>(*fields[i]));
    }
  }
  if (!wait.empty())
    state.dispatch(wait);
}

void drain_wave_counters(WaveRaceState &state, rj_code_arch_t arch) {
  amdgpu::WaitFields drained;
  drained.fill(0);
  apply_wait_fields(state, drained, arch);
}

void register_global_load(WaveRaceState &rs, const InstructionSnapshot &instruction,
                          const amdgpu::VectorMemState &d, const amdgpu::Wavefront &wf,
                          uint64_t result_lanes, const amdgpu::MemoryIssueInfo &issue) {
  const auto obligations = issue.counter_obligations();
  const auto order = vgpr_writeback_order(memoryOrderForObligations(obligations), wf.cu().arch());
  std::vector<uint32_t> registers;
  if (result_lanes) {
    auto destinations = validated_load_destinations(d, wf);
    if (!destinations)
      return;
    registers = std::move(*destinations);
    for (uint32_t i = 0; i < registers.size(); ++i)
      rs.checkVgprWrite(static_cast<int>(registers[i]), result_lanes,
                        vector_memory_byte_mask(d, wf, i), order);
  }
  if (!obligations.empty()) {
    const uint8_t bytes = vector_memory_byte_mask(d, wf);
    const uint8_t last =
        registers.empty() ? bytes : vector_memory_byte_mask(d, wf, registers.size() - 1);
    rs.registerEvent(instruction, MemoryEventType::GLOBAL_TO_VGPR, std::move(registers),
                     result_lanes, bytes, obligations, order, last);
  }
}

std::optional<amdgpu::MemoryIssueInfo>
routed_flat_issue_info(const amdgpu::MemoryAccessObservation &access,
                       const amdgpu::MemoryIssueInfo &decoded, rj_code_arch_t arch,
                       const Instruction &inst) {
  if (access.decoded_space != amdgpu::DecodedMemorySpace::FLAT ||
      (access.route != amdgpu::MemoryRoute::GLOBAL && access.route != amdgpu::MemoryRoute::LOCAL))
    return std::nullopt;

  // A GFX10 generic FLAT return needs both counter acknowledgments even
  // when every requesting lane took the same route (RDNA1/2 section 9.2).
  if (amdgpu::flat_requires_both_counters(arch) && access.active_lane_mask != 0 &&
      (access.is_load || access.request_lane_mask == 0))
    return decoded;
  // Routing also captures completions for rejected addresses without any
  // backing traffic. Synthetic observers may supply only address/mask metadata.
  if (const auto &routed = inst.data_as<amdgpu::VectorMemState>()->routed_issue_info)
    return routed;
  if (access.request_lane_mask == 0)
    return decoded.for_flat_memory_domains(false, false);
  if (amdgpu::flat_requires_both_counters(arch))
    return decoded;

  const uint64_t shared_lanes =
      (access.flat_local_lane_mask | access.flat_dds_lane_mask) & access.request_lane_mask;
  // A mixed route needs both issue domains.
  if (shared_lanes != 0 && shared_lanes != access.request_lane_mask)
    return decoded;
  if (access.route != (shared_lanes ? amdgpu::MemoryRoute::LOCAL : amdgpu::MemoryRoute::GLOBAL))
    return std::nullopt;

  return decoded.for_flat_memory_domains(shared_lanes == 0, shared_lanes != 0);
}

void register_global_vgpr_result(WaveRaceState &rs, const InstructionSnapshot &instruction,
                                 const amdgpu::VectorMemState &d, const amdgpu::Wavefront &wf,
                                 std::span<const amdgpu::MemoryCounterObligation> obligations,
                                 VgprWriteOrder writeOrder, ConditionalVgprWrite feedback = {}) {
  auto destinations = validated_load_destinations(d, wf);
  if (!destinations)
    return;
  std::vector<uint32_t> registers = std::move(*destinations);
  const uint8_t byte_mask = vector_memory_byte_mask(d, wf);
  uint8_t last_byte_mask =
      registers.empty() ? byte_mask : vector_memory_byte_mask(d, wf, registers.size() - 1);
  for (uint32_t i = 0; i < registers.size(); ++i)
    rs.checkVgprWrite(static_cast<int>(registers[i]), d.exec_mask,
                      vector_memory_byte_mask(d, wf, i), writeOrder);
  if (feedback.lanes) {
    rs.checkVgprWrite(static_cast<int>(feedback.reg), feedback.lanes,
                      ExecutionPlugin::kFullByteMask, writeOrder);
    // Keep the possibly half-width last texel in its existing position.
    registers.insert(registers.begin(), feedback.reg);
  }
  if (obligations.empty())
    return;
  rs.registerEvent(instruction, MemoryEventType::GLOBAL_TO_VGPR, std::move(registers), d.exec_mask,
                   byte_mask, obligations, writeOrder, last_byte_mask, feedback);
}

} // namespace

MarkedInstruction findConflict(const RaceViolation &v, RaceDetector &detector) {
  if (v.sourceInstruction)
    return {*v.sourceInstruction, v.wave, -1};
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

size_t RaceDetectorPlugin::retainedWorkgroupCount() const {
  std::lock_guard<std::mutex> lock(dispatch_mutex_);
  return detectors_.size();
}

bool RaceDetectorPlugin::observes_hot_hooks_for_wavefront(const amdgpu::Wavefront *wf) const {
  return wf != nullptr && wf->has_plugin_state(slot_index());
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
    else if (v.space == RaceViolation::Space::EXEC)
      oss << "Race on EXEC";
    else
      oss << "Race on LDS byte " << v.index;
    oss << " [workgroup (" << v.workgroupId.x << ", " << v.workgroupId.y << ", " << v.workgroupId.z
        << "), wave " << v.wave;
    if (v.space == RaceViolation::Space::VGPR || v.space == RaceViolation::Space::LDS)
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
                            : v.space == RaceViolation::Space::EXEC ? "EXEC"
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
      static_cast<int>(wavefronts.front()->cu().arch() == ROCJITSU_CODE_ARCH_CDNA3
                           ? std::max<uint32_t>(physical_sgpr_count, cdna3::sgpr_alias_end(102))
                           : physical_sgpr_count),
      Dim3d(static_cast<int>(wg_id)), std::move(handler),
      counterCapacitiesForArch(wavefronts.front()->cu().arch()),
      wavefronts.front()->lds().size_bytes());

  auto &det = *detectors_[key];
  for (uint32_t w = 0; w < num_waves; ++w) {
    auto state = std::make_unique<RaceWavefrontState>();
    state->race_state = &det.getWaveRaceState(static_cast<int>(w));
    wavefronts[w]->set_plugin_state(slot_index(), std::move(state));
  }
}

void RaceDetectorPlugin::onAmdgpuWorkgroupCompleted(uint32_t dispatch_id, uint32_t wg_id) {
  // Each wave released its pointer into this detector when it halted.
  std::lock_guard<std::mutex> lock(dispatch_mutex_);
  detectors_.erase(WorkgroupKey{dispatch_id, wg_id});
}

void RaceDetectorPlugin::onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
  if (wf.end_reason() != amdgpu::Wavefront::EndReason::None && wf.wait_counters().empty() &&
      wf.has_plugin_state(slot_index())) {
    drain_wave_counters(*get_state(wf)->race_state, wf.cu().arch());
  }
  release_wave_state(wf);
}

void RaceDetectorPlugin::onAmdgpuWavefrontCancelled(amdgpu::Wavefront &wf) {
  release_wave_state(wf);
}

void RaceDetectorPlugin::onAmdgpuImageFeedbackQualified(const Instruction &inst,
                                                        amdgpu::Wavefront &wf) {
  auto *s = get_state(wf);
  const auto *issue = inst.amdgpu_memory_issue_info();
  if (!s || !s->race_state || !issue)
    return;
  const auto &d = *inst.data_as<amdgpu::VectorMemState>();
  // Empty image issues are accounted for when routed, before feedback qualification.
  if (!d.exec_mask)
    return;
  const uint64_t lanes = d.feedback_write_lanes();
  const ConditionalVgprWrite feedback{
      d.dst_reg_base - wf.vgpr_alloc().base + d.destination_vgpr_count(), lanes};
  const auto obligations = issue->counter_obligations();
  const auto completion = memoryOrderForObligations(obligations);
  const auto arch = wf.cu().arch();
  const VgprWriteOrder writeOrder{vgpr_writeback_order(completion, arch),
                                  amdgpu::has_ordered_vmem_reads(arch) &&
                                          completion == MemoryOrderClass::VMEM
                                      ? amdgpu::vmem_read_type(inst.mnemonic())
                                      : amdgpu::VmemWriteback::Ordinary};
  register_global_vgpr_result(*s->race_state, current_instruction(*s, wf), d, wf, obligations,
                              writeOrder, feedback);
}

void RaceDetectorPlugin::onAmdgpuMemoryAccessCancelled(const Instruction &inst,
                                                       amdgpu::Wavefront &wf) {
  if (wf.has_plugin_state(slot_index()))
    get_state(wf)->race_state->cancelRoutedIssue(inst);
}

void RaceDetectorPlugin::onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access,
                                                    const Instruction &inst,
                                                    amdgpu::Wavefront &wf) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  auto *rs = s->race_state;
  rs->beginRoutedIssue(inst);
  auto *detector = rs->getDetector();
  auto waveId = rs->getWaveId();
  const InstructionSnapshot &instruction = current_instruction(*s, wf);
  // One DDS request faults the entire FLAT operation before any segment accesses memory.
  if (access.flat_dds_lane_mask)
    return;
  if (const auto *data = inst.data();
      data && (data->tag() == amdgpu::LOCAL_MEM || data->tag() == amdgpu::GLOBAL_MEM) &&
      inst.data_as<amdgpu::VectorMemState>()->rejects_routed_access())
    return;
  const auto *decoded = inst.amdgpu_memory_issue_info();
  assert(decoded && "memory instruction reached the race detector without issue metadata");
  if (!decoded)
    return;
  const auto arch = wf.cu().arch();
  auto routed = routed_flat_issue_info(access, *decoded, arch, inst);
  // Address-dependent FLAT acquires only its routed domains. Admit them before
  // registering either half of a mixed return.
  if (routed && !amdgpu::flat_issues_both_counter_domains(arch))
    rs->prepareForMemoryIssue(*routed);
  const uint64_t shared_lanes = access.flat_local_lane_mask | access.flat_dds_lane_mask;
  const bool split_results = routed && access.is_load &&
                             !amdgpu::flat_requires_both_counters(arch) &&
                             (amdgpu::flat_issues_both_counter_domains(arch) ||
                              (shared_lanes && (access.active_lane_mask & ~shared_lanes))) &&
                             !inst.data_as<amdgpu::VectorMemState>()->lds_dst;
  if (split_results) {
    const auto &state = *inst.data_as<amdgpu::VectorMemState>();
    // GFX12 allocates both counter positions, even when only one route has lanes.
    // Each destination lane completes with the pipeline that supplies its value.
    register_global_load(*rs, instruction, state, wf, state.exec_mask & ~shared_lanes,
                         decoded->for_flat_memory_domains(true, false));
    routed = decoded->for_flat_memory_domains(false, true);
    if (!(state.exec_mask & shared_lanes)) {
      const auto lds = routed->counter_obligations();
      rs->registerEvent(instruction, MemoryEventType::LDS_TO_VGPR, {}, 0, 0xf, lds,
                        memoryOrderForObligations(lds));
      return;
    }
  }
  const auto &issue = routed ? *routed : *decoded;
  const auto obligations = issue.counter_obligations();
  const MemoryOrderClass memoryOrder = memoryOrderForObligations(obligations);
  const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  const bool typed_vmem = amdgpu::has_ordered_vmem_reads(arch);
  VgprWriteOrder writeOrder{vgpr_writeback_order(memoryOrder, arch),
                            gfx10 && routed && access.is_load ? amdgpu::VmemWriteback::GenericFlat
                            : typed_vmem && memoryOrder == MemoryOrderClass::VMEM
                                ? amdgpu::vmem_read_type(inst.mnemonic())
                                : amdgpu::VmemWriteback::Ordinary};
  if (routed && access.is_load && amdgpu::flat_requires_both_counters(arch) &&
      amdgpu::has_ordered_flat_lds_writeback(arch))
    writeOrder.flat_lds_lanes = shared_lanes;
  if (memoryOrder == MemoryOrderClass::LDS && amdgpu::has_ordered_lds_direct_writeback(arch) &&
      std::ranges::any_of(obligations, [](auto o) {
        return o.wait_counter_type() == amdgpu::WaitCounterType::EXPCNT;
      }))
    writeOrder.direct_lds = true;
  if (inst.data()->tag() == amdgpu::LOCAL_MEM || inst.data()->tag() == amdgpu::GLOBAL_MEM) {
    const auto &d = *inst.data_as<amdgpu::VectorMemState>();
    // The pipeline plan also contains any slots with no data dependency,
    // including both domains of an empty GFX10-GFX12 generic FLAT instruction.
    const auto &issued = d.routed_issue_info ? *d.routed_issue_info : issue;
    if (d.exec_mask == 0 && !(gfx10 && inst.mnemonic().starts_with("ds_gws_")) &&
        !d.ordered_count && !d.gds_m0) {
      const bool local = access.route == amdgpu::MemoryRoute::LOCAL;
      if (local || access.route == amdgpu::MemoryRoute::GLOBAL) {
        // An instruction can occupy counter slots without reading or writing any lanes.
        const auto reg_load =
            local ? MemoryEventType::LDS_TO_VGPR : MemoryEventType::GLOBAL_TO_VGPR;
        const auto reg_store =
            local ? MemoryEventType::VGPR_TO_LDS : MemoryEventType::VGPR_TO_GLOBAL;
        const bool unconditional = gfx10 || issued.counts_with_empty_exec ||
                                   (routed && amdgpu::flat_issues_both_counter_domains(arch));
        for (const auto &obligation : issued.counter_obligations()) {
          if (!unconditional && !rs->hasPendingCounter(obligation.wait_counter_type()))
            continue;
          const bool returns =
              d.is_load && obligation.wait_counter_type() != amdgpu::WaitCounterType::EXPCNT;
          auto counter_order = writeOrder;
          counter_order.completion = obligation.completion_class();
          rs->registerEvent(instruction, returns ? reg_load : reg_store, {}, 0, 0xf,
                            std::span(&obligation, 1), counter_order);
        }
      }
      return;
    }
    if (routed && !split_results && d.routed_issue_info) {
      for (const auto &obligation : issued.counter_obligations()) {
        const auto counter = obligation.wait_counter_type();
        if (std::ranges::find(obligations, counter,
                              &amdgpu::MemoryCounterObligation::wait_counter_type) ==
            obligations.end()) {
          rs->registerEvent(instruction,
                            d.is_load ? MemoryEventType::GLOBAL_TO_VGPR
                                      : MemoryEventType::VGPR_TO_GLOBAL,
                            {}, 0, 0xf, std::span(&obligation, 1), obligation.completion_class());
        }
      }
    }
  }

  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA3 && access.scratch_lane_mask &&
      (access.decoded_space == amdgpu::DecodedMemorySpace::SCRATCH ||
       access.decoded_space == amdgpu::DecodedMemorySpace::FLAT))
    rs->checkScalarRead(wf.memory_wait_shadow().canonical({RegClass::FLAT_SCRATCH, 0, 2}));

  // RDNA2 issues vector memory instructions with empty effective EXEC,
  // including loads and returning atomics nullified by VGPR tuple bounds.
  // Keep each routed counter obligation in order without any register or LDS bytes.
  const auto count_empty_issue = [&](MemoryEventType type) {
    if (wf.cu().arch() == ROCJITSU_CODE_ARCH_RDNA2 && !obligations.empty())
      rs->registerEvent(instruction, type, {}, 0, 0xf, obligations, memoryOrder);
  };
  const bool mixedFlat = inst.data()->tag() == amdgpu::GLOBAL_MEM && shared_lanes != 0;
  if (inst.data()->tag() == amdgpu::LOCAL_MEM || mixedFlat) {
    auto &d = *inst.data_as<amdgpu::VectorMemState>();
    if ((d.gds_m0 ||
         std::ranges::any_of(
             obligations, [](auto o) { return o.completion_class() == MemoryOrderClass::GDS; })) &&
        amdgpu::has_ordered_gds_writeback(wf.cu().arch())) {
      // GFX10 GDS returns are ordered with one another on LGKMCNT. EXPCNT
      // protects the outgoing registers independently, including the one
      // contributing lane of DS_ORDERED_COUNT.
      for (const auto &obligation : obligations) {
        const bool outgoing = obligation.wait_counter_type() == amdgpu::WaitCounterType::EXPCNT;
        const bool returning = !outgoing && d.is_load;
        const uint64_t lanes = d.ordered_count
                                   ? (outgoing ? d.lane_mask : d.ordered_count->result_lanes)
                                   : d.exec_mask;
        const auto order = obligation.completion_class();
        const auto bytes = returning ? vector_memory_byte_mask(d, wf) : uint8_t{0xf};
        std::vector<uint32_t> registers;
        if (returning && lanes) {
          if (auto destinations = validated_load_destinations(d, wf)) {
            registers = std::move(*destinations);
            for (const auto reg : registers)
              rs->checkVgprWrite(static_cast<int>(reg), lanes, bytes, order);
          }
        } else if (outgoing && lanes &&
                   (d.ordered_count || wf.cu().arch() != ROCJITSU_CODE_ARCH_RDNA2)) {
          RegisterSet sources;
          for (int i = 0; i < inst.num_src_operands(); ++i)
            if (const auto src = inst.src_operand(i)->decoded_vgpr())
              sources.expand(*src);
          sources.for_each_ordinary([&](RegisterRef src) {
            if (src.index < wf.num_vgprs())
              registers.push_back(src.index);
          });
        }
        const auto type = returning                     ? MemoryEventType::LDS_TO_VGPR
                          : outgoing && d.ordered_count ? MemoryEventType::GDS_ORDERED_COUNT_SOURCE
                                                        : MemoryEventType::VGPR_TO_LDS;
        rs->registerEvent(instruction, type, std::move(registers), lanes, bytes,
                          std::span(&obligation, 1), order);
      }
      const amdgpu::RegisterAccess registers(wf);
      // Indexed RDNA2 GDS faults can enter the handler after the source grant.
      const bool enters_memory_trap =
          arch == ROCJITSU_CODE_ARCH_RDNA2 && d.gds_m0 && d.address_range_violation_lanes &&
          (wf.mode_raw() & amdgpu::Wavefront::kRdna2MemoryViolationEnable);
      if (gfx10 && !enters_memory_trap && registers.vgpr_results_fit_allocation(inst)) {
        rs->lockGdsExec(instruction);
        const auto lanes = d.ordered_count ? d.lane_mask : registers.gds_source_lanes(inst);
        if (lanes)
          registers.for_each_gds_source(inst, [&](RegisterRef source) {
            for (unsigned word = 0; word < source.width; ++word)
              rs->lockGdsVgpr(instruction, source.index + word, lanes);
          });
      }
      return;
    }
    const uint64_t execMask = split_results ? d.exec_mask & shared_lanes : d.exec_mask;
    const bool native_ds = (arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5) &&
                           !inst.mnemonic().starts_with("flat_") && !d.lds_stack_inputs;

    const uint32_t granule = native_ds && d.atomic_op == amdgpu::AtomicOp::NONE
                                 ? (d.is_load ? std::min(d.elem_size, 4u) : 1u)
                                 : d.elem_size;
    // DDS and allocation-OOB lanes participate in the instruction's
    // dependencies, but do not access bytes in the ordinary LDS allocation.
    uint64_t activeLdsLanes =
        (d.ds_src2 || arch == ROCJITSU_CODE_ARCH_CDNA3) ? d.lane_mask : execMask;
    // LDS byte checks follow requesting lanes. Register dependencies can cover more lanes.
    // A parameter load writes all four quad lanes while only three lanes read LDS.
    if (access.decoded_space == amdgpu::DecodedMemorySpace::LOCAL && access.wavefront_size)
      activeLdsLanes = access.request_lane_mask;
    const uint64_t ldsMask =
        d.gds || d.gds_m0 ? 0
                          : (mixedFlat || access.normalized_to_local
                                 ? activeLdsLanes & access.flat_local_lane_mask
                                 : activeLdsLanes) &
                                ~(access.flat_dds_lane_mask | access.flat_local_oob_lane_mask);
    LdsAccess ldsAccess{
        .uses_flat_lds = routed.has_value(),
        .read_modify_write =
            (routed.has_value() || access.decoded_space == amdgpu::DecodedMemorySpace::LOCAL) &&
            d.atomic_op != amdgpu::AtomicOp::NONE};
    ldsAccess.aligned_atomic_bytes = ldsAccess.read_modify_write ? d.elem_size : 0;
    const auto ldsOrder = ldsAccess.order(memoryOrder);
    auto type = d.is_load ? MemoryEventType::LDS_TO_VGPR : MemoryEventType::VGPR_TO_LDS;
    if (execMask == 0) {
      count_empty_issue(type);
      return;
    }
    const uint32_t perLaneBytes = d.num_elems * d.elem_size;
    const uint32_t elementBytes = d.atomic_op == amdgpu::AtomicOp::NONE ? granule : perLaneBytes;
    const auto allocation = access.lds_allocation;
    const auto end = allocation ? std::optional{allocation->end} : access.flat_local_allocation_end;

    std::vector<uint32_t> registers;
    if (d.is_load) {
      auto destinations = validated_load_destinations(d, wf);
      if (!destinations)
        return;
      registers = std::move(*destinations);
    }

    const auto bounds =
        amdgpu::indexed_ds_access_bounds(d, wf.cu().arch(), wf.lds_base(), wf.lds_size());
    const bool secondAccess = d.has_second_ds_address(wf.cu().arch(), ldsMask);
    uint32_t laneAddrs[64]{};
    uint32_t secondLaneAddrs[64]{};
    IntervalSet src2Reads;
    auto secondAddresses = access.secondary_addresses;
    // A caller with an unknown route may supply only the instruction state.
    if (access.route == amdgpu::MemoryRoute::UNKNOWN &&
        d.has_second_ds_address(wf.cu().arch(), d.lane_mask))
      secondAddresses = d.ds2_per_lane_addr;
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(ldsMask & (1ULL << lane)))
        continue;
      auto validate = [&](uint64_t address, bool source = false) {
        if (allocation && address < allocation->begin)
          return amdgpu::kInvalidLdsAddress;
        if (d.gds_m0 || (bounds && !bounds->contains(address)))
          return amdgpu::kInvalidLdsAddress;
        auto bytes = detector->ldsBytesInRange(address, perLaneBytes, elementBytes);
        if (allocation)
          bytes = allocation->retained_bytes(address, bytes, elementBytes);
        else if (end)
          bytes = amdgpu::whole_element_bytes_in_range(address, bytes, elementBytes, *end);
        if (bytes == 0)
          return amdgpu::kInvalidLdsAddress;
        const auto addr = static_cast<uint32_t>(address);
        // Equal-width, naturally aligned atomic intervals overlap only when
        // they refer to the same whole element. Partial or unaligned accesses
        // retain the conservative byte-overlap check.
        const uint8_t atomicBytes =
            ldsAccess.read_modify_write && bytes == d.elem_size && address % d.elem_size == 0
                ? d.elem_size
                : 0;
        if (!atomicBytes)
          ldsAccess.aligned_atomic_bytes = 0;
        if (source) {
          detector->validateRead(static_cast<int>(addr), waveId, lane, bytes, ldsOrder);
          src2Reads.append(addr, addr + bytes);
          return addr;
        }
        if (d.is_load || ldsAccess.read_modify_write)
          detector->validateRead(static_cast<int>(addr), waveId, static_cast<int>(lane), bytes,
                                 ldsOrder, atomicBytes);
        if (!d.is_load || ldsAccess.read_modify_write)
          detector->validateWrite(static_cast<int>(addr), waveId, static_cast<int>(lane), bytes,
                                  ldsOrder, atomicBytes);
        return addr;
      };
      const uint64_t bit = uint64_t{1} << lane;
      laneAddrs[lane] =
          (d.ds_oob_lane_mask & bit) ? amdgpu::kInvalidLdsAddress : validate(d.per_lane_addr[lane]);
      if (secondAccess)
        secondLaneAddrs[lane] = (d.ds2_oob_lane_mask & bit) ? amdgpu::kInvalidLdsAddress
                                                            : validate(secondAddresses[lane]);
      else if (d.ds_src2 && !(d.ds2_oob_lane_mask & bit))
        validate(d.ds2_per_lane_addr[lane], true);
    }
    uint8_t byte_mask = vector_memory_byte_mask(d, wf);
    if (d.is_load) {
      for (uint32_t reg : registers)
        rs->checkVgprWrite(static_cast<int>(reg), execMask, byte_mask, writeOrder);
    }
    if (obligations.empty())
      return;
    EventId eventId;
    if (secondAccess) {
      eventId = rs->registerLdsEvent(
          instruction, type, std::move(registers), execMask, wf.wf_size(),
          std::span<const uint32_t>(laneAddrs, wf.wf_size()),
          std::span<const uint32_t>(secondLaneAddrs, wf.wf_size()), perLaneBytes, byte_mask,
          obligations, memoryOrder, elementBytes, ldsAccess);
    } else {
      eventId = rs->registerLdsEvent(
          instruction, type, std::move(registers), execMask, wf.wf_size(),
          std::span<const uint32_t>(laneAddrs, wf.wf_size()), perLaneBytes, byte_mask, obligations,
          writeOrder, ldsMask, elementBytes, end, ldsAccess);
    }
    if (eventId.isValid() && d.ds_src2) {
      src2Reads.finalize();
      detector->attachSrc2Source(eventId, std::move(src2Reads));
    }
    // Only ordinary LDS lanes contribute intervals. When results split by route,
    // the vector-memory lanes have already been registered separately.
    return;
  }

  if (inst.data()->tag() == amdgpu::GLOBAL_MEM) {
    auto &d = *inst.data_as<amdgpu::VectorMemState>();
    if (d.lds_dst) {
      uint32_t perLaneBytes = d.buffer_components
                                  ? 4u * d.destination_vgpr_count()
                                  : d.num_elems * (d.lds_dword_extend ? 4u : d.elem_size);
      // A cluster load outside a cluster is an ordinary global load (CDNA5 ISA
      // 10.7), so the mask selects targets only when the wave is in a cluster,
      // as in the memory pipeline.
      if (d.cluster_multicast && d.cluster_mcast_mask != 0 && wf.cluster_size() > 1) {
        uint32_t selfMask = amdgpu::cluster_multicast_rank_mask(wf.cluster_rank());
        uint32_t peerMask = d.cluster_mcast_mask & ~selfMask;
        if (peerMask != 0)
          warn_cluster_peer_writes_ignored_once();
        if ((d.cluster_mcast_mask & selfMask) == 0) {
          // A peer-only multicast writes no local LDS, but the issuing wave still
          // gets its completion (CDNA5 ISA 10.7), in issue order with its other
          // async loads (10.8). Register an event without LDS lanes so it takes
          // its counter slot.
          if (!obligations.empty()) {
            const uint32_t noAddrs[64] = {};
            rs->registerLdsEvent(instruction, MemoryEventType::GLOBAL_TO_LDS, {}, /*execMask=*/0,
                                 wf.wf_size(), std::span<const uint32_t>(noAddrs, wf.wf_size()),
                                 perLaneBytes, 0xF, obligations, memoryOrder);
          }
          return;
        }
      }
      uint32_t ldsAddrs[64];
      const uint64_t validLaneMask = amdgpu::lds_destination_write_mask(d, wf.cu().arch());
      for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
        ldsAddrs[lane] =
            d.lds_per_lane_addr ? d.per_lane_lds_addr[lane] : d.lds_base + lane * perLaneBytes;
        // Check each written lane against pending LDS reads, such as a tensor
        // store's source or a DS read, as the DS write path does. That includes
        // the wave's own pending DS reads: the ISA does not establish that a
        // direct-to-LDS write waits for them, so as a conservative rule the
        // overlap is reported until a DS wait completes the read.
        if (!(validLaneMask & (1ULL << lane)))
          continue;
        const auto bytes = detector->ldsBytesInRange(ldsAddrs[lane], perLaneBytes, perLaneBytes);
        if (bytes != 0)
          detector->validateWrite(static_cast<int>(ldsAddrs[lane]), waveId, static_cast<int>(lane),
                                  static_cast<int>(bytes), memoryOrder);
      }
      if (obligations.empty())
        return;
      rs->registerLdsEvent(instruction, MemoryEventType::GLOBAL_TO_LDS, {}, validLaneMask,
                           wf.wf_size(), std::span<const uint32_t>(ldsAddrs, wf.wf_size()),
                           perLaneBytes, 0xF, obligations, memoryOrder);
    } else if (!access.lds_source_addresses.empty()) {
      // The global request owns both counter obligations. Its LDS reads
      // participate in races even though the payload never uses a VGPR.
      const uint32_t bytes = d.num_elems * d.elem_size;
      uint64_t sourceLanes = 0;
      for (uint64_t lanes = access.request_lane_mask; lanes; lanes &= lanes - 1) {
        const unsigned lane = std::countr_zero(lanes);
        const uint32_t address = access.lds_source_addresses[lane];
        if (detector->ldsBytesInRange(address, bytes, bytes) != bytes ||
            (access.lds_allocation &&
             access.lds_allocation->retained_bytes(address, bytes, bytes) != bytes))
          continue;
        sourceLanes |= uint64_t{1} << lane;
        detector->validateRead(static_cast<int>(address), waveId, static_cast<int>(lane),
                               static_cast<int>(bytes), MemoryOrderClass::UNORDERED);
      }
      if (!obligations.empty())
        rs->registerLdsEvent(instruction, MemoryEventType::LDS_TO_GLOBAL, {}, d.exec_mask,
                             wf.wf_size(), access.lds_source_addresses, bytes, 0xf, obligations,
                             memoryOrder, sourceLanes, bytes);
    } else if (d.is_load) {
      if (!d.has_feedback_result())
        register_global_vgpr_result(*rs, instruction, d, wf, obligations, writeOrder);
    } else if (!obligations.empty()) {
      rs->registerEvent(instruction, MemoryEventType::VGPR_TO_GLOBAL, {}, d.exec_mask, 0xF,
                        obligations, memoryOrder);
    }
  }

  if (inst.data()->tag() == amdgpu::SCALAR_MEM) {
    auto &d = *inst.data_as<amdgpu::ScalarMemState>();
    if (d.is_load) {
      if (const auto reg = d.dst_register.register_ref()) {
        if (reg->cls == RegClass::EXEC && reg->index < wf.wf_size() / 32)
          rs->checkGdsExecWrite();
        rs->registerScalarLoad(instruction, wf.memory_wait_shadow().canonical(*reg), wf.exec(),
                               obligations, memoryOrder);
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

void RaceDetectorPlugin::onAmdgpuLdsDirectRead(const amdgpu::Wavefront &wf, uint32_t address,
                                               uint32_t bytes, uint32_t lane) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  // A broadcast can materialize once per lane. Keep one observation per range
  // and retain distinct interpolation coefficients from the same instruction.
  if (std::ranges::none_of(s->lds_operand_reads, [&](const auto &read) {
        return read.address == address && read.bytes == bytes;
      }))
    s->lds_operand_reads.push_back({address, bytes, lane});
}

void RaceDetectorPlugin::onAmdgpuTensorDmaIssued(const amdgpu::TensorDmaIssueObservation &issue,
                                                 const amdgpu::Wavefront &wf) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  // The hook fires in the tensor instruction's own step, before the transfer's
  // global requests and LDS writes, as onAmdgpuMemoryAccessRouted() does for
  // other memory instructions.
  const InstructionSnapshot &instruction = current_instruction(*s, wf);
  auto *rs = s->race_state;
  rs->prepareForCounterIncrement(amdgpu::WaitCounterType::TENSORCNT);
  auto *detector = rs->getDetector();
  const auto waveId = rs->getWaveId();
  // Tensor transfers from one wave complete on TENSORCNT in issue order (CDNA5
  // ISA 10.11.1), which partial waits rely on, so the event takes the TENSOR
  // class. The ISA does not specify the relative timing of their LDS reads and
  // writes, so as a conservative rule, check the transfer's LDS bytes as
  // unordered with every pending access, including the wave's own transfers in
  // the other direction: a load against a pending store's source, and a store
  // against a pending load's destination.
  constexpr MemoryOrderClass memoryOrder = MemoryOrderClass::TENSOR;
  constexpr MemoryOrderClass ldsAccessOrder = MemoryOrderClass::UNORDERED;
  // An observed range can end mid-element: an element that straddles the end
  // of the wave's LDS allocation transfers only its leading bytes. Validate the
  // ranges at byte granularity, as registerLdsRangeEvent registers them;
  // rounding them to whole elements would drop those bytes.
  constexpr int kByteGranularity = 1;

  // A transfer has no lanes; its violations report lane -1. The core takes
  // its own range type, independent of the observation protocol.
  std::vector<LdsByteRange> ranges;
  ranges.reserve(issue.lds_ranges.size());
  for (const auto &range : issue.lds_ranges) {
    ranges.push_back({.address = range.address, .bytes = range.bytes});
    const auto bytes = detector->ldsBytesInRange(range.address, range.bytes, kByteGranularity);
    if (bytes == 0)
      continue;
    const int addr = static_cast<int>(range.address);
    if (issue.is_load)
      detector->validateWrite(addr, waveId, -1, static_cast<int>(bytes), ldsAccessOrder);
    else
      detector->validateRead(addr, waveId, -1, static_cast<int>(bytes), ldsAccessOrder);
  }

  // An instruction that moves no data, such as a NOP tile, a gather with no
  // valid indices or a store wholly outside the tensor, still occupies a
  // TENSORCNT slot, so its event is registered even without LDS ranges. A NULL
  // descriptor is registered the same way, matching the emulator's assumption
  // that it counts on TENSORCNT.
  const std::array obligations{
      amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::TENSORCNT, memoryOrder}};
  rs->registerLdsRangeEvent(
      instruction, issue.is_load ? MemoryEventType::GLOBAL_TO_LDS : MemoryEventType::LDS_TO_GLOBAL,
      ranges, obligations, memoryOrder);
}

void RaceDetectorPlugin::onAmdgpuReadVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                                               uint64_t lane_mask, uint8_t byte_mask) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  uint32_t logical_reg = physical_reg - wf->vgpr_alloc().base;
  if (s->race_state->getVgprMemoryEvents(static_cast<int>(logical_reg)).empty())
    return;
  s->race_state->checkVgprReadLanes(static_cast<int>(logical_reg), lane_mask, byte_mask);
}

void RaceDetectorPlugin::onAmdgpuWriteVgprLanes(const amdgpu::Wavefront *wf, uint32_t physical_reg,
                                                uint64_t lane_mask, uint8_t byte_mask) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  uint32_t logical_reg = physical_reg - wf->vgpr_alloc().base;
  if (auto &query = s->image_query; query && !wf->instruction_execution_failed()) {
    assert(byte_mask == kFullByteMask);
    query->lanes |= lane_mask;
    if (std::ranges::find(query->registers, logical_reg) == query->registers.end())
      query->registers.push_back(logical_reg);
    return;
  }
  if (!s->race_state->needsVgprWriteCheck(static_cast<int>(logical_reg)))
    return;
  s->race_state->checkVgprWriteLanes(static_cast<int>(logical_reg), lane_mask, byte_mask);
}

void RaceDetectorPlugin::onAmdgpuReadScalarRegister(const amdgpu::Wavefront *wf, RegisterRef reg) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  s->race_state->checkScalarRead(wf->memory_wait_shadow().canonical(reg));
}

void RaceDetectorPlugin::onAmdgpuWriteScalarRegister(const amdgpu::Wavefront *wf, RegisterRef reg) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  s->race_state->checkScalarWrite(wf->memory_wait_shadow().canonical(reg));
}

void RaceDetectorPlugin::onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                                          amdgpu::Wavefront &wf) {
  const InstructionSnapshot instruction = capture_instruction(pc, inst);
  auto *s = get_state(wf);
  assert(s && s->race_state);
  s->race_state->finishRoutedIssue();
  s->trace.push(instruction);
  if (inst.has_embedded_memory_wait() && !inst.is_waitcnt()) {
    const auto fields =
        waitcheck_detail::WaitcheckTarget::embedded_wait_fields(inst, wf.cu().arch());
    if (fields.succeeded() && fields.value())
      apply_wait_fields(*s->race_state, *fields.value(), wf.cu().arch());
  }
  // Failed instructions skip the after-execute callback. A retried query
  // starts a fresh capture and can publish only its own result.
  s->lds_operand_reads.clear();
  s->image_query.reset();
  const auto arch = wf.cu().arch();
  const auto name = inst.mnemonic();
  // Inline queries reserve their issue slot before accessing mip/coordinate
  // operands. A successful execution publishes the captured destination lanes.
  const bool legacy_lod = name == "image_get_lod" && amdgpu::has_ordered_vmem_reads(arch);
  if (inst.is_memory_wait_producer() && !inst.is_memory_op() &&
      (name == "image_get_resinfo" || legacy_lod)) {
    const auto counter = arch == ROCJITSU_CODE_ARCH_RDNA4 ? amdgpu::WaitCounterType::LOADCNT
                                                          : amdgpu::WaitCounterType::VMCNT;
    s->race_state->prepareForCounterIncrement(counter);
    s->image_query.emplace(amdgpu::vmem_read_type(name), counter);
  }
  const bool generic_flat = inst.mnemonic().starts_with("flat_");
  // Fixed-domain FLAT must secure both slots before reading its address,
  // including when EXEC is empty. Other FLAT requests reserve slots at routing.
  if (inst.is_memory_op() && (!generic_flat || amdgpu::flat_issues_both_counter_domains(arch))) {
    if (const auto *info = inst.amdgpu_memory_issue_info())
      s->race_state->prepareForMemoryIssue(*info);
  }
  // Messages also share the aggregate LGKMCNT on legacy targets.
  if (inst.mnemonic().starts_with("s_sendmsg") && counterCapacitiesForArch(arch).lgkmcnt != 0)
    s->race_state->prepareForCounterIncrement(amdgpu::WaitCounterType::LGKMCNT);
  // Implicit VCC operands use wave mask operations directly. Observe only
  // the architectural mask, excluding temporary VCC saves by SDWA execution.
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA3 && !s->race_state->getWaveMemoryEvents().empty()) {
    auto uses_vcc = [](const Operand *operand) {
      return operand && operand->to_special_reg_class() == RegClass::VCC;
    };
    const auto pair = wf.memory_wait_shadow().canonical({RegClass::VCC, 0, 2});
    for (int i = 0; i < inst.num_src_operands(); ++i) {
      if (!inst.tracks_explicit_source(i) || !uses_vcc(inst.src_operand(i)))
        continue;
      const uint64_t mask = inst.mnemonic().starts_with("v_") ? wf.exec() : ~uint64_t{0};
      if ((mask >> 32) && static_cast<uint32_t>(mask))
        s->race_state->checkScalarRead(pair);
      else if (mask)
        s->race_state->checkScalarRead(
            {RegClass::SGPR, static_cast<uint16_t>(pair.index + (mask >> 32 != 0)), 1});
    }
    for (int i = 0; i < inst.num_dst_operands(); ++i)
      if (uses_vcc(inst.dst_operand(i)))
        s->race_state->checkScalarWrite(pair);
  }
}

void RaceDetectorPlugin::onAmdgpuAfterExecuteInstruction(uint64_t /*pc*/, const Instruction &inst,
                                                         amdgpu::Wavefront &wf) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  // S_TRAP reaches after-execute only after the CU enters the configured handler.
  if (inst.mnemonic() == "s_trap" && wf.in_trap_handler()) {
    drain_wave_counters(*s->race_state, wf.cu().arch());
    return;
  }
  if (!s->lds_operand_reads.empty()) {
    const auto &instruction = current_instruction(*s, wf);
    for (const auto &read : s->lds_operand_reads)
      s->race_state->recordLdsOperandRead(instruction, read.address, read.bytes, read.lane);
    s->lds_operand_reads.clear();
  }
  if (auto &query = s->image_query) {
    const std::array obligations{
        amdgpu::MemoryCounterObligation{query->counter, MemoryOrderClass::VMEM}};
    // GFX12 decrements LOADCNT in order but can write the VGPRs out of order.
    const VgprWriteOrder order{query->counter == amdgpu::WaitCounterType::LOADCNT
                                   ? MemoryOrderClass::UNORDERED
                                   : MemoryOrderClass::VMEM,
                               query->type};
    for (const auto reg : query->registers)
      s->race_state->checkVgprWrite(static_cast<int>(reg), query->lanes, kFullByteMask, order);
    // Success consumes one counter position even when EXEC or the destination
    // tuple suppressed all output. The query never routes a texel request.
    s->race_state->registerEvent(current_instruction(*s, wf), MemoryEventType::GLOBAL_TO_VGPR,
                                 std::move(query->registers), query->lanes, kFullByteMask,
                                 obligations, order);
    query.reset();
    return;
  }
  if (const auto reg = wf.instruction_exec_write();
      reg && reg->index < wf.wf_size() / 32 && !wf.instruction_execution_failed())
    get_state(wf)->race_state->checkGdsExecWrite();
  // The CDNA1 CU delivers this hook for S_TRAP only after entering a configured
  // handler. Its implicit wait completes this wave's register obligations.
  // Cross-wave LDS intervals still need a barrier.
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA1 && inst.mnemonic() == "s_trap" &&
      wf.in_trap_handler()) {
    PendingWaitCount wait;
    for (auto counter : {amdgpu::WaitCounterType::VMCNT, amdgpu::WaitCounterType::EXPCNT,
                         amdgpu::WaitCounterType::LGKMCNT})
      wait.add(counter, 0);
    get_state(wf)->race_state->dispatch(wait);
    return;
  }

  // Embedded waits (e.g. VINTERP wait_exp) must retire before operand access.
  if (!inst.is_waitcnt())
    return;
  const auto arch = wf.cu().arch();
  const auto executed_limits = wf.wait_target().gfx10_wait_fields();
  const auto fields = amdgpu::WaitCounterPolicy::executed_wait_fields(inst, arch, executed_limits);
  if (fields.failed() || !fields.value())
    return;

  onAmdgpuImplicitMemoryWait(wf, *fields.value());
}

void RaceDetectorPlugin::onAmdgpuImplicitMemoryWait(amdgpu::Wavefront &wf,
                                                    const amdgpu::WaitFields &fields) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  apply_wait_fields(*s->race_state, fields, wf.cu().arch());
}

void RaceDetectorPlugin::onAmdgpuTrapEntered(amdgpu::Wavefront &wf) {
  if (wf.cu().arch() != ROCJITSU_CODE_ARCH_CDNA2)
    return;
  auto *s = get_state(wf);
  if (!s || !s->race_state)
    return;
  PendingWaitCount drain;
  for (const auto counter : {amdgpu::WaitCounterType::VMCNT, amdgpu::WaitCounterType::EXPCNT,
                             amdgpu::WaitCounterType::LGKMCNT})
    drain.add(counter, 0);
  s->race_state->dispatch(drain);
}

void RaceDetectorPlugin::onAmdgpuBarrierResolved(std::span<amdgpu::Wavefront *> wavefronts,
                                                 BarrierKind kind) {
  if (wavefronts.empty())
    return;
  if (kind == BarrierKind::Workgroup) {
    auto *s = get_state(wavefronts.front());
    assert(s && s->race_state);
    s->race_state->getDetector()->flushWorkgroupBarrierEvents();
    return;
  }
  for (auto *wf : wavefronts) {
    auto *s = get_state(wf);
    assert(s && s->race_state);
    s->race_state->flushBarrierPendingEvents();
  }
}

} // namespace rocjitsu::plugins::race_detector
