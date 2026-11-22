// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/plugin.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/isa/isa_traits.h"

#include "rocjitsu/code/analysis/waitcheck/target.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/sgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/indexed_ds_bounds.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"
#include "util/result.h"

#include "flatbuffers/flexbuffers.h"
#include "flatbuffers/idl.h"

#include "rocjitsu/vm/plugins/memory_access_observation.h"
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include "rocjitsu/vm/plugins/tensor_dma_issue_observation.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <ios>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::plugins::race_detector {

namespace {

void report_plugin_failure(const amdgpu::Wavefront &wf, const char *detail) {
  if (!wf.instruction_execution_failed())
    wf.report_instruction_execution_error(amdgpu::InstructionExecutionError::PluginFailure, detail);
}

util::FailureOr<InstructionSnapshot> capture_instruction(uint64_t pc, const Instruction &inst,
                                                         const amdgpu::Wavefront &wf) {
  static_assert(InstructionSnapshot::kMaxWords == amdgpu::DecodedInstructionCache::kNumWords);
  if (!inst.raw_encoding() || inst.size() <= 0 || inst.size() % sizeof(uint32_t) != 0 ||
      static_cast<size_t>(inst.size()) > InstructionSnapshot::kMaxWords * sizeof(uint32_t)) {
    report_plugin_failure(wf, "race detector instruction has no valid encoding");
    return util::Result::failure();
  }
  // Some synchronization operations lack the BARRIER flag.
  const bool preserve_in_trace = inst.is_waitcnt() || inst.has_embedded_memory_wait() ||
                                 inst.is_barrier() ||
                                 inst.mnemonic().find("barrier") != std::string_view::npos ||
                                 inst.mnemonic().starts_with("ds_gws_") || inst.writes_pc();
  return InstructionSnapshot{pc, std::span(inst.raw_encoding(), inst.size() / sizeof(uint32_t)),
                             preserve_in_trace, inst.mnemonic() == "s_nop"};
}

// Before-execute records the instruction before its register or memory hooks.
// Reports may also be requested after a rejected capture; fail closed in that case.
const InstructionSnapshot *current_instruction(const RaceWavefrontState &state,
                                               const amdgpu::Wavefront &wf) {
  if (state.trace.size() == 0) {
    report_plugin_failure(wf, "race detector has no current instruction");
    return nullptr;
  }
  return &state.trace[state.trace.size() - 1];
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

/// gfx1250 (CDNA5) issues an EXEC-zero memory instruction and counts it, unless
/// it is skipped: "VMEM can be skipped only if: EXEC == 0 and LOADcnt / STOREcnt
/// / ASYNCcnt / TENSORcnt == 0", FLAT also requires DScnt == 0, and "LDS can be
/// skipped only if: DScnt==0 and EXEC==0" (CDNA5 ISA 3.2.2.1). A skipped
/// instruction leaves no older operation on those counters to complete, so
/// either way its admission is sound.
///
/// On CDNA1 through CDNA4, a buffer, global or scratch instruction issued with EXEC
/// zero counts on VMCNT, in order with the wave's other non-FLAT VMEM
/// instructions (all four ISA manuals, sections 3.3, 4.4 and 10.4–10.5).
bool exec_zero_memory_issue_counts(rocjitsu::IsaFamily arch) {
  return arch == rocjitsu::IsaFamily::Cdna1 || arch == rocjitsu::IsaFamily::Cdna2 ||
         arch == rocjitsu::IsaFamily::Cdna3 || arch == rocjitsu::IsaFamily::Cdna4 ||
         arch == rocjitsu::IsaFamily::Cdna5;
}

/// Counter obligations retained when wave EXEC is zero.
/// On CDNA1 through CDNA4, buffer, global and scratch instructions keep VMCNT.
/// DS and generic FLAT are excluded pending validation of their counter,
/// routing and completion rules (see docs/race-detector.md).
///
/// A gfx1250 memory instruction is skipped only when the counters its encoding names are
/// all zero (CDNA5 ISA 3.2.2.1): "VMEM can be skipped only if: EXEC == 0 and
/// LOADcnt / STOREcnt / ASYNCcnt / TENSORcnt == 0", FLAT also requires DScnt ==
/// 0, and "LDS can be skipped only if: DScnt==0 and EXEC==0". On those counters
/// it either counts or was skipped because they were empty, so admitting it and
/// recording its slot are sound. On any other counter, such as the ASYNCcnt of
/// ds_atomic_async_barrier_arrive_b64, an LDS instruction, it may be skipped
/// while the counter is nonzero, so that obligation is left out.
amdgpu::MemoryIssueInfo exec_zero_counted_obligations(const amdgpu::MemoryIssueInfo &info,
                                                      std::string_view mnemonic) {
  const bool lds = mnemonic.starts_with("ds_");
  const bool flat = mnemonic.starts_with("flat_");
  amdgpu::MemoryIssueInfo counted;
  counted.exec_masked = info.exec_masked;
  for (const auto obligation : info.counter_obligations()) {
    bool counts = false;
    switch (obligation.wait_counter_type()) {
    case amdgpu::WaitCounterType::DSCNT:
      counts = lds || flat;
      break;
    case amdgpu::WaitCounterType::LOADCNT:
    case amdgpu::WaitCounterType::STORECNT:
    case amdgpu::WaitCounterType::ASYNCCNT:
    case amdgpu::WaitCounterType::TENSORCNT:
      counts = !lds;
      break;
    case amdgpu::WaitCounterType::VMCNT:
      counts = !lds && !flat;
      break;
    default:
      break;
    }
    if (counts)
      counted.counter_obligations_[counted.num_counter_obligations_++] = obligation;
  }
  return counted;
}

/// A gfx1250 VBUFFER instruction whose resource descriptor TYPE is not a buffer
/// is ignored (CDNA5 ISA 9.6), so it applies no backpressure and holds no
/// counter slot. If hardware still counted it, leaving it out only keeps older
/// operations pending longer.
bool ignored_memory_issue(const Instruction &inst, const amdgpu::Wavefront &wf) {
  // Every gfx1250 VBUFFER instruction has conditional memory lanes, so others
  // skip decoding their register modifiers.
  if (wf.cu().arch() != rocjitsu::IsaFamily::Cdna5 || !(inst.flags() & CONDITIONAL_MEMORY_LANES))
    return false;
  amdgpu::RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  return amdgpu::addr_calc::cdna5_ignores_buffer_resource(modifiers, wf);
}

/// Record a counter position without register or LDS dependencies.
void register_counter_slot(WaveRaceState &race, const InstructionSnapshot &instruction,
                           std::span<const amdgpu::MemoryCounterObligation> obligations,
                           MemoryOrderClass memoryOrder, bool is_load) {
  if (obligations.empty())
    return;
  race.registerEvent(instruction,
                     is_load ? MemoryEventType::GLOBAL_TO_VGPR : MemoryEventType::VGPR_TO_GLOBAL,
                     /*registers=*/{}, /*execMask=*/0, /*byteMask=*/0xF, obligations, memoryOrder);
}

/// The counter slot a gfx1250 wait-counter producer without memory issue
/// metadata takes when it issues, and whether it returns data:
/// - GLOBAL_INV counts on LOADcnt, and GLOBAL_WB and GLOBAL_WBINV on STOREcnt
///   (CDNA5 ISA 5.7.1.2), in order with the VMEM operations on that counter, as
///   LLVM orders GLOBAL_INV with VMEM reads. None is skipped for EXEC zero
///   (CDNA5 ISA 3.2.2.1).
/// - The LDS lane permutes and DS_SWIZZLE_B32 use the LDS hardware without LDS
///   memory (CDNA5 ISA 11.2.3), and LDS operations count on DScnt and are kept
///   in order (CDNA5 ISA 5.7.1.4). With EXEC zero one is skipped only when
///   DScnt is zero (CDNA5 ISA 3.2.2.1), so its admission is sound either way.
/// - DS_NOP holds an ordered DScnt slot too, as the core memory-wait checker
///   models it; the ISA describes it only as "Do nothing".
struct CounterOnlySlot {
  amdgpu::MemoryCounterObligation obligation;
  bool is_load;
};

std::optional<CounterOnlySlot> counter_only_slot(amdgpu::Operation operation) {
  using amdgpu::MemoryCompletionClass;
  using amdgpu::WaitCounterType;
  if (operation == amdgpu::Operation::GLOBAL_INV)
    return CounterOnlySlot{{WaitCounterType::LOADCNT, MemoryCompletionClass::VMEM}, true};
  if (operation == amdgpu::Operation::GLOBAL_WB || operation == amdgpu::Operation::GLOBAL_WBINV)
    return CounterOnlySlot{{WaitCounterType::STORECNT, MemoryCompletionClass::VMEM}, false};
  if (operation == amdgpu::Operation::DS_PERMUTE_B32 ||
      operation == amdgpu::Operation::DS_BPERMUTE_B32 ||
      operation == amdgpu::Operation::DS_BPERMUTE_FI_B32 ||
      operation == amdgpu::Operation::DS_SWIZZLE_B32 || operation == amdgpu::Operation::DS_NOP)
    return CounterOnlySlot{{WaitCounterType::DSCNT, MemoryCompletionClass::LDS}, true};
  return std::nullopt;
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

MemoryOrderClass vgpr_writeback_order(MemoryOrderClass completion, rocjitsu::IsaFamily arch) {
  if (completion == MemoryOrderClass::VMEM && !amdgpu::has_ordered_vmem_writeback(arch))
    return MemoryOrderClass::UNORDERED;
  return completion;
}

void apply_wait_fields(WaveRaceState &state, const amdgpu::WaitFields &fields,
                       rocjitsu::IsaFamily arch) {
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

void drain_wave_counters(WaveRaceState &state, rocjitsu::IsaFamily arch) {
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
    if (auto destinations = validated_load_destinations(d, wf))
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
                       const amdgpu::MemoryIssueInfo &decoded, rocjitsu::IsaFamily arch,
                       const Instruction &inst) {
  if (access.decoded_space != amdgpu::DecodedMemorySpace::FLAT ||
      (access.route != amdgpu::MemoryRoute::GLOBAL && access.route != amdgpu::MemoryRoute::LOCAL))
    return std::nullopt;

  // On RDNA1 through RDNA3.5 a generic FLAT result needs both counter
  // acknowledgments, even when all requesting lanes use one route.
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
  std::vector<uint32_t> registers;
  if (auto destinations = validated_load_destinations(d, wf))
    registers = std::move(*destinations);
  else
    feedback = {};
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

util::FailureOr<MarkedInstruction> findConflict(const RaceViolation &v, RaceDetector &detector) {
  if (v.sourceInstruction)
    return MarkedInstruction{*v.sourceInstruction, v.wave, -1};
  EventId event_id = v.conflictingEvent;
  if (!detector.events().contains(event_id))
    return util::Result::failure();
  return MarkedInstruction{detector.events().instruction(event_id),
                           detector.events().waveId(event_id).value, -1};
}

util::FailureOr<std::string> formatTrace(
    const RingBuffer<InstructionSnapshot, kTraceHistorySize> &trace,
    const std::function<util::FailureOr<std::string>(const InstructionSnapshot &)> &disassemble,
    std::optional<MarkedInstruction> conflict, MarkedInstruction access,
    size_t context_instructions) {
  assert(context_instructions >= 2 && context_instructions <= kTraceHistorySize);
  const size_t trace_size = trace.size();
  std::optional<size_t> access_index;
  for (size_t i = 0; i < trace_size; ++i)
    if (trace[i] == access.instruction)
      access_index = i;
  std::optional<size_t> conflict_index;
  if (conflict && conflict->wave == access.wave)
    for (size_t i = 0; i < trace_size; ++i)
      if (trace[i] == conflict->instruction) {
        conflict_index = i;
        break;
      }
  assert((!conflict_index || !access_index || *conflict_index <= *access_index) &&
         "race conflict follows the current access in the wave trace");
  const size_t last = access_index ? *access_index + 1 : trace_size;
  size_t first = access_index.value_or(trace_size);
  if (conflict && conflict->wave == access.wave) {
    first = conflict_index.value_or(0);
  } else if (conflict) {
    // Waves usually run the same code, so start where this wave last ran the
    // conflicting instruction; that shows its own path to the access.
    first = last - std::min({last, kOtherWaveTraceContext, context_instructions});
    for (size_t i = 0; i < last; ++i)
      if (trace[i] == conflict->instruction)
        first = i;
  }
  const size_t span = last > first ? last - first : 0;
  std::vector<bool> shown(trace_size);
  const size_t head = context_instructions / 2;
  const size_t tail = context_instructions - head;
  for (size_t i = first; i < last; ++i)
    shown[i] = span <= context_instructions || i - first < head || last - i <= tail ||
               trace[i].preserve_in_trace();
  // An elision notice takes a line, so hiding a single instruction saves nothing.
  for (size_t i = first; i < last; ++i)
    if (!shown[i] && (i == first || shown[i - 1]) && (i + 1 == last || shown[i + 1]))
      shown[i] = true;
  // Pad PCs to the widest printed one so the instruction column lines up.
  uint64_t widest_pc = 0;
  for (size_t i = first; i < last; ++i)
    if (shown[i])
      widest_pc = std::max(widest_pc, trace[i].pc());
  if (conflict && !conflict_index)
    widest_pc = std::max(widest_pc, conflict->instruction.pc());
  if (!access_index)
    widest_pc = std::max(widest_pc, access.instruction.pc());
  const size_t pc_width = std::format("{:#x}", widest_pc).size();
  auto format_pc = [&](uint64_t pc) {
    return std::format("{:<{}}", std::format("{:#x}", pc), pc_width);
  };
  std::ostringstream oss;
  oss << "       " << std::format("{:<{}}", "PC", pc_width) << "  Instruction\n";
  auto annotation = [&](const MarkedInstruction &mark) {
    oss << "  ; <-- wave " << mark.wave;
    if (mark.lane >= 0)
      oss << " lane " << mark.lane;
  };
  auto external = [&](const MarkedInstruction &mark, std::string_view note) -> util::Result {
    const InstructionSnapshot &instruction = mark.instruction;
    util::FailureOr<std::string> text = disassemble(instruction);
    if (text.failed())
      return util::Result::failure();
    oss << "  ==>  " << format_pc(instruction.pc()) << "  " << text.value() << note;
    annotation(mark);
    oss << '\n';
    return util::Result::success();
  };
  if (conflict && !conflict_index) {
    if (external(*conflict, conflict->wave == access.wave ? "  (before trace window)" : "")
            .failed())
      return util::Result::failure();
    if (conflict->wave == access.wave) {
      oss << "       ... earlier instructions not recorded ...\n";
    } else
      oss << "       Execution trace below is for wave " << access.wave << ".\n";
  }
  auto is_marked = [&](size_t i) {
    return (conflict_index && i == *conflict_index) || (access_index && i == *access_index);
  };
  auto can_compact = [&](size_t i) {
    return trace[i].is_nop() && !trace[i].preserve_in_trace() && !is_marked(i);
  };
  auto emit = [&](size_t i, size_t end) -> util::Result {
    const bool is_conflict = conflict_index && i == *conflict_index;
    const bool is_access = access_index && i == *access_index;
    const InstructionSnapshot &instruction = trace[i];
    util::FailureOr<std::string> text = disassemble(instruction);
    if (text.failed())
      return util::Result::failure();
    oss << ((is_conflict || is_access) ? "  ==>  " : "       ") << format_pc(instruction.pc())
        << "  " << text.value();
    if (is_conflict)
      annotation(*conflict);
    if (is_access)
      annotation(access);
    if (end > i + 1)
      oss << "  ; " << end - i << " executions, through PC "
          << std::format("{:#x}", trace[end - 1].pc());
    oss << '\n';
    return util::Result::success();
  };
  for (size_t i = first; i < last;) {
    if (shown[i]) {
      size_t end = i + 1;
      if (can_compact(i))
        while (end < last && shown[end] && can_compact(end) &&
               trace[end].pc() > trace[end - 1].pc() &&
               trace[end].pc() - trace[end - 1].pc() == trace[end - 1].encoding().size_bytes() &&
               std::ranges::equal(trace[i].encoding(), trace[end].encoding()))
          ++end;
      if (emit(i, end).failed())
        return util::Result::failure();
      i = end;
    } else {
      const size_t gap = i;
      while (i < last && !shown[i])
        ++i;
      oss << "       ... " << i - gap << " instructions elided ...\n";
    }
  }
  if (!access_index && external(access, "  (outside recorded trace)").failed())
    return util::Result::failure();
  return oss.str();
}

namespace {

// Load events do not retain a reliable source memory space: FLAT lanes can
// resolve differently, and scalar and direct-to-LDS loads can read scratch.
std::string_view event_description(const EventRegistry &events, EventId event,
                                   RaceViolation::Space space) {
  if (space == RaceViolation::Space::LDS && events.readsLds(event) && events.writesLds(event))
    return events.registers(event).empty() ? "atomic update of LDS"
                                           : "atomic update of LDS returning values to VGPRs";
  using enum MemoryEventType;
  switch (events.type(event)) {
  case GLOBAL_TO_VGPR:
    return "load into VGPRs";
  case VGPR_TO_GLOBAL:
    return "transfer from VGPRs into global memory";
  case LDS_TO_VGPR:
    return space == RaceViolation::Space::LDS ? "transfer from LDS into VGPRs" : "load into VGPRs";
  case VGPR_TO_LDS:
    return "transfer from VGPRs into LDS";
  case GLOBAL_TO_LDS:
    return "load into LDS";
  case GLOBAL_TO_SGPR:
    return "load into SGPRs";
  case GLOBAL_TO_TTMP:
    return "load into TTMPs";
  case SCALAR_TO_GLOBAL:
    return "transfer from scalar registers into global memory";
  case LDS_TO_GLOBAL:
    return "transfer from LDS into global memory";
  case GDS_ORDERED_COUNT_SOURCE:
    return "VGPR source read by DS_ORDERED_COUNT";
  case LDS_OPERAND_READ:
    return "read from LDS";
  }
  throw std::logic_error("race event has an unknown memory operation");
}

} // namespace

std::string formatRaceDescription(const RaceViolation &v, const EventRegistry &events,
                                  uint64_t access_pc, rocjitsu::IsaFamily arch) {
  if (!v.sourceInstruction && !events.contains(v.conflictingEvent))
    throw std::out_of_range("race violation references an unavailable conflicting event");
  const util::FailureOr<amdgpu::WaitcntModel> model = amdgpu::waitcnt_model(arch);
  if (model.failed())
    throw std::invalid_argument("race report has an unsupported architecture");
  const EventId event = v.conflictingEvent;
  const bool conflicting_read =
      v.sourceInstruction || v.conflictsWithSrc2Read ||
      (v.space == RaceViolation::Space::VGPR && isFromVgpr(events.type(event))) ||
      (v.space == RaceViolation::Space::LDS && events.readsLds(event));
  const char *hazard = !v.isWrite         ? "read-after-write (RAW)"
                       : conflicting_read ? "write-after-read (WAR)"
                                          : "write-after-write (WAW)";
  std::ostringstream oss;
  oss << "Race on ";
  switch (v.space) {
  case RaceViolation::Space::VGPR:
    oss << "VGPR v" << v.index;
    break;
  case RaceViolation::Space::SGPR:
    oss << "SGPR s" << v.index;
    break;
  case RaceViolation::Space::TTMP:
    oss << "TTMP ttmp" << v.index;
    break;
  case RaceViolation::Space::EXEC:
    oss << "EXEC";
    break;
  case RaceViolation::Space::LDS:
    oss << "LDS byte " << v.index;
    break;
  }
  oss << " [workgroup (" << v.workgroupId.x << ", " << v.workgroupId.y << ", " << v.workgroupId.z
      << "), wave " << v.wave;
  if (v.lane >= 0)
    oss << ", lane " << v.lane;
  oss << "]: " << hazard << ".\n";
  oss << "  Current access: " << (v.isWrite ? "write" : "read") << " at PC 0x" << std::hex
      << access_pc << std::dec << ".\n";
  if (v.sourceInstruction) {
    oss << "  Conflict: GDS source read at PC 0x" << std::hex << v.sourceInstruction->pc()
        << std::dec << ", wave " << v.wave << ".\n";
    oss << "  Completion for wave " << v.wave << " is not guaranteed; pending counters: expcnt.\n";
    return oss.str();
  }
  const int owner = events.waveId(event).value;
  oss << "  Conflict: "
      << (v.conflictsWithSrc2Read ? "read of LDS source operand"
                                  : event_description(events, event, v.space));
  oss << " at PC 0x" << std::hex << events.instruction(event).pc() << std::dec << ", wave " << owner
      << ".\n";
  const bool flat_lds = v.space == RaceViolation::Space::LDS && events.usesFlatLds(event);
  const auto status = events.status(event);
  if (status == EventStatus::ACTIVE && flat_lds && events.ldsAccessComplete(event)) {
    oss << "  Wait requirements satisfied for the LDS access in wave " << owner << ".\n";
  } else if (status == EventStatus::ACTIVE) {
    oss << (flat_lds ? "  Completion of the LDS access for wave " : "  Completion for wave ")
        << owner << " is not guaranteed; pending counters: ";
    std::vector<amdgpu::WaitCounterKind> displayed;
    for (const amdgpu::MemoryCounterObligation obligation : events.counterObligations(event)) {
      const amdgpu::WaitCounterType counter = obligation.wait_counter_type();
      if (!events.hasPendingWaitCounter(event, counter))
        continue;
      const amdgpu::WaitCounterKind kind = amdgpu::canonical_wait_counter(counter, model.value());
      if ((flat_lds && kind != amdgpu::WaitCounterKind::Ds) ||
          std::ranges::find(displayed, kind) != displayed.end())
        continue;
      if (!displayed.empty())
        oss << ", ";
      oss << amdgpu::wait_counter_name(kind, model.value());
      displayed.push_back(kind);
    }
    oss << ".\n";
  } else if (status == EventStatus::WAVE_COMPLETE) {
    oss << "  Wait requirements satisfied for this operation in wave " << owner << ".\n";
  } else {
    throw std::logic_error("race violation references a retired memory event");
  }
  if (owner != v.wave)
    oss << "  The detector observed no synchronization between waves " << owner << " and " << v.wave
        << " for this access.\n";
  return oss.str();
}

RaceDetectorPlugin::RaceDetectorPlugin(const char *config_json) : ExecutionPlugin("race") {
  if (!config_json || !*config_json)
    return;
  flexbuffers::Builder builder;
  flatbuffers::Parser parser;
  if (!parser.ParseFlexBuffer(config_json, nullptr, &builder))
    throw std::invalid_argument("Race plugin configuration is not valid JSON");
  const flexbuffers::Reference root = flexbuffers::GetRoot(builder.GetBuffer());
  if (!root.IsMap())
    throw std::invalid_argument("Race plugin configuration must be an object");
  const flexbuffers::Reference context = root.AsMap()["trace_context"];
  if (context.IsNull())
    return;
  if (!context.IsIntOrUint() || (context.IsInt() && context.AsInt64() < 2) ||
      context.AsUInt64() < 2 || context.AsUInt64() > kTraceHistorySize)
    throw std::invalid_argument(std::format(
        "Race plugin 'trace_context' must be an integer between 2 and {}", kTraceHistorySize));
  trace_context_ = static_cast<size_t>(context.AsUInt64());
}

RaceDetectorPlugin::~RaceDetectorPlugin() { sink().write(getSummary()); }

std::string RaceDetectorPlugin::getSummary() const {
  std::lock_guard<std::mutex> lock(report_mutex_);
  const char *banner = "\n========================================\n"
                       " ROCJITSU RACE DETECTION SUMMARY\n"
                       "========================================\n";
  if (reports_emitted_) {
    return std::string(banner) + "  " + std::to_string(reports_emitted_) +
           " race report(s) emitted; " + std::to_string(observations_suppressed_) +
           " observation(s) suppressed by PC deduplication\n"
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
    const InstructionSnapshot *current = current_instruction(*ws, *wf);
    if (!current)
      return;
    uint64_t pc = current->pc();

    {
      std::lock_guard<std::mutex> lock(report_mutex_);
      if (observed_races_.count({dispatch_id, pc})) {
        ++observations_suppressed_;
        return;
      }
    }

    auto *detector = ws->race_state->getDetector();
    util::FailureOr<MarkedInstruction> maybe_conflict = findConflict(v, *detector);
    if (maybe_conflict.failed()) {
      report_plugin_failure(*wf, "race violation references an unavailable conflicting event");
      return;
    }
    const MarkedInstruction &conflict = maybe_conflict.value();

    std::ostringstream oss;
    oss << formatRaceDescription(v, detector->events(), pc, wf->cu().arch());

    MarkedInstruction access_mark{*current, v.wave, v.lane};
    util::StringDiagnostic diagnostic;
    util::FailureOr<std::string> trace = formatTrace(
        ws->trace,
        [wf, &diagnostic](const InstructionSnapshot &instruction) {
          return wf->disassemble_instruction(instruction.encoding(), instruction.pc(),
                                             diagnostic.emitter());
        },
        conflict, access_mark, trace_context_);
    if (trace.failed()) {
      report_plugin_failure(*wf, "race detector could not disassemble the instruction trace");
      std::lock_guard<std::mutex> lock(report_mutex_);
      sink().write(diagnostic.message());
      return;
    }
    oss << trace.value();

    {
      std::lock_guard<std::mutex> lock(report_mutex_);
      bool is_new = !observed_races_.count({dispatch_id, pc}) &&
                    !observed_races_.count({dispatch_id, conflict.instruction.pc()});
      observed_races_.emplace(dispatch_id, pc);
      if (is_new) {
        ++reports_emitted_;
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
      } else {
        ++observations_suppressed_;
      }
    }
  };

  // Multi-XCD: workgroups from the same dispatch can arrive on different
  // partition threads, so the detector map needs protection.
  std::lock_guard<std::mutex> lock(dispatch_mutex_);
  detectors_[key] = std::make_unique<RaceDetector>(
      static_cast<int>(num_waves), static_cast<int>(physical_vgpr_count),
      static_cast<int>(wavefronts.front()->cu().arch() == rocjitsu::IsaFamily::Cdna3
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
  if (wf.has_plugin_state(slot_index())) {
    WaveRaceState &race = *get_state(wf)->race_state;
    if (wf.end_reason() != amdgpu::Wavefront::EndReason::None && wf.wait_counters().empty())
      drain_wave_counters(race, wf.cu().arch());
    // A workgroup round this wave has not signalled could not complete until
    // now, so the next such round releases what the wave completed.
    race.getDetector()->waveEnded(race.getWaveId());
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
                                      ? amdgpu::vmem_read_type(arch, inst.mnemonic())
                                      : amdgpu::VmemWriteback::Ordinary};
  const auto *instruction = current_instruction(*s, wf);
  if (!instruction)
    return;
  register_global_vgpr_result(*s->race_state, *instruction, d, wf, obligations, writeOrder,
                              feedback);
}

void RaceDetectorPlugin::onAmdgpuMemoryAccessCancelled(const Instruction &inst,
                                                       amdgpu::Wavefront &wf) {
  if (wf.has_plugin_state(slot_index())) {
    auto *state = get_state(wf);
    state->mixed_lds.reset();
    state->race_state->cancelRoutedIssue(inst);
  }
}

void RaceDetectorPlugin::onAmdgpuMixedFlatLdsAccess(const Instruction &, amdgpu::Wavefront &wf) {
  auto *s = get_state(wf);
  if (!s || !s->mixed_lds)
    return;
  auto &pending = *s->mixed_lds;
  auto &rs = *s->race_state;
  auto &detector = *rs.getDetector();
  const auto wave = rs.getWaveId();
  const bool reads =
      pending.reads && detector.mayConflict(pending.intervals, wave, false, pending.order);
  const bool writes =
      pending.writes && detector.mayConflict(pending.intervals, wave, true, pending.order);
  if (reads || writes) {
    for (const auto &access : pending.lanes) {
      if (reads)
        detector.validateRead(access.address, wave, access.lane, access.bytes, pending.order,
                              access.atomic_bytes);
      if (writes)
        detector.validateWrite(access.address, wave, access.lane, access.bytes, pending.order,
                               access.atomic_bytes);
    }
  }
  detector.attachLdsIntervals(pending.event, std::move(pending.intervals));
  s->mixed_lds.reset();
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
  const InstructionSnapshot *current = current_instruction(*s, wf);
  if (!current)
    return;
  const InstructionSnapshot &instruction = *current;
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
  if (routed && access.request_lane_mask && !amdgpu::flat_issues_both_counter_domains(arch) &&
      rs->prepareForMemoryIssue(*routed).failed()) {
    report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
    return;
  }
  const uint64_t shared_lanes = access.flat_local_lane_mask | access.flat_dds_lane_mask;
  if (routed && !access.is_load && amdgpu::flat_issues_both_counter_domains(arch) &&
      !amdgpu::flat_requires_both_counters(arch) && access.active_lane_mask) {
    routed = decoded->for_flat_memory_domains((access.active_lane_mask & ~shared_lanes) != 0,
                                              (access.active_lane_mask & shared_lanes) != 0);
  }
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
  const bool gfx10 = arch == rocjitsu::IsaFamily::Rdna1 || arch == rocjitsu::IsaFamily::Rdna2;
  const bool typed_vmem = amdgpu::has_ordered_vmem_reads(arch);
  VgprWriteOrder writeOrder{vgpr_writeback_order(memoryOrder, arch),
                            gfx10 && routed && access.is_load ? amdgpu::VmemWriteback::GenericFlat
                            : typed_vmem && memoryOrder == MemoryOrderClass::VMEM
                                ? amdgpu::vmem_read_type(arch, inst.mnemonic())
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
      if (ignored_memory_issue(inst, wf) ||
          (!arch_is_rdna(arch) && !exec_zero_memory_issue_counts(arch)) ||
          (arch_is_cdna_4_or_lower(arch) && wf.exec() != 0))
        return;
      auto counted = issued;
      if (exec_zero_memory_issue_counts(arch) && wf.exec() == 0)
        counted = exec_zero_counted_obligations(issued, inst.mnemonic());
      if (counted.empty())
        return;
      if (arch_is_cdna_4_or_lower(arch)) {
        register_counter_slot(*rs, instruction, counted.counter_obligations(),
                              memoryOrderForObligations(counted.counter_obligations()), d.is_load);
        return;
      }
      // An empty CDNA5 async copy still takes an ASYNCcnt slot. Keep its direction
      // for barrier arrivals and partial waits even though it touches no LDS bytes.
      if ((d.lds_src || d.lds_dst) && std::ranges::any_of(obligations, [](const auto &obligation) {
            return obligation.wait_counter_type() == amdgpu::WaitCounterType::ASYNCCNT;
          })) {
        rs->registerLdsRangeEvent(instruction,
                                  d.lds_src ? MemoryEventType::LDS_TO_GLOBAL
                                            : MemoryEventType::GLOBAL_TO_LDS,
                                  {}, obligations, memoryOrder);
        return;
      }
      if (arch == rocjitsu::IsaFamily::Cdna5 && wf.exec() != 0 && !routed)
        return;
      const bool local = access.route == amdgpu::MemoryRoute::LOCAL;
      if (local || access.route == amdgpu::MemoryRoute::GLOBAL) {
        // An instruction can occupy counter slots without reading or writing any lanes.
        const auto reg_load =
            local ? MemoryEventType::LDS_TO_VGPR : MemoryEventType::GLOBAL_TO_VGPR;
        const auto reg_store =
            local ? MemoryEventType::VGPR_TO_LDS : MemoryEventType::VGPR_TO_GLOBAL;
        const bool unconditional =
            gfx10 || issued.counts_with_empty_exec ||
            (routed && amdgpu::flat_issues_both_counter_domains(arch)) ||
            (arch == rocjitsu::IsaFamily::Cdna5 && !rs->getWaveMemoryEvents().empty());
        for (const auto &obligation : counted.counter_obligations()) {
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
    if (routed && !split_results && amdgpu::flat_issues_both_counter_domains(arch)) {
      for (const auto &obligation : decoded->counter_obligations()) {
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

  if (wf.cu().arch() == rocjitsu::IsaFamily::Cdna3 && access.scratch_lane_mask &&
      (access.decoded_space == amdgpu::DecodedMemorySpace::SCRATCH ||
       access.decoded_space == amdgpu::DecodedMemorySpace::FLAT))
    rs->checkScalarRead(wf.memory_wait_shadow().canonical({RegClass::FLAT_SCRATCH, 0, 2}));

  // RDNA2 issues vector memory instructions with empty effective EXEC,
  // including loads and returning atomics nullified by VGPR tuple bounds.
  // Keep each routed counter obligation in order without any register or LDS bytes.
  const auto count_empty_issue = [&](MemoryEventType type) {
    if (wf.cu().arch() == rocjitsu::IsaFamily::Rdna2 && !obligations.empty())
      rs->registerEvent(instruction, type, {}, 0, 0xf, obligations, memoryOrder);
  };
  const bool mixedFlat = inst.data()->tag() == amdgpu::GLOBAL_MEM && shared_lanes != 0;
  if (inst.data()->tag() == amdgpu::LOCAL_MEM || mixedFlat) {
    auto &d = *inst.data_as<amdgpu::VectorMemState>();
    const bool ordered_gds = amdgpu::has_ordered_gds_writeback(arch);
    const bool protect_gds_sources = amdgpu::checks_gds_source_lifetime(arch);
    if ((ordered_gds || protect_gds_sources) &&
        (d.gds_m0 || std::ranges::any_of(obligations, [](auto o) {
           return o.completion_class() == MemoryOrderClass::GDS;
         }))) {
      // GFX10 GDS returns are ordered with one another on LGKMCNT. EXPCNT
      // protects the outgoing registers independently, including the one
      // contributing lane of DS_ORDERED_COUNT.
      for (const auto &obligation : obligations) {
        const bool outgoing = obligation.wait_counter_type() == amdgpu::WaitCounterType::EXPCNT;
        const bool returning = !outgoing && d.is_load;
        const uint64_t lanes = d.ordered_count
                                   ? (outgoing ? d.lane_mask : d.ordered_count->result_lanes)
                                   : d.exec_mask;
        const auto order =
            returning && !ordered_gds ? MemoryOrderClass::UNORDERED : obligation.completion_class();
        const auto bytes = returning ? vector_memory_byte_mask(d, wf) : uint8_t{0xf};
        std::vector<uint32_t> registers;
        if (returning && lanes) {
          if (auto destinations = validated_load_destinations(d, wf)) {
            registers = std::move(*destinations);
            for (const auto reg : registers)
              rs->checkVgprWrite(static_cast<int>(reg), lanes, bytes, order);
          }
        } else if (outgoing && lanes && d.ordered_count) {
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
          arch == rocjitsu::IsaFamily::Rdna2 && d.gds_m0 && d.address_range_violation_lanes &&
          (wf.mode_raw() & amdgpu::Wavefront::kRdna2MemoryViolationEnable);
      if (protect_gds_sources && !enters_memory_trap &&
          registers.vgpr_results_fit_allocation(inst)) {
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
    const bool native_ds =
        (arch == rocjitsu::IsaFamily::Rdna4 || arch == rocjitsu::IsaFamily::Cdna5) &&
        !inst.mnemonic().starts_with("flat_") && !d.lds_stack_inputs;

    const uint32_t granule = native_ds && d.atomic_op == amdgpu::AtomicOp::NONE
                                 ? (d.is_load ? std::min(d.elem_size, 4u) : 1u)
                                 : d.elem_size;
    // DDS and allocation-OOB lanes participate in the instruction's
    // dependencies, but do not access bytes in the ordinary LDS allocation.
    uint64_t activeLdsLanes =
        (d.ds_src2 || arch == rocjitsu::IsaFamily::Cdna3) ? d.lane_mask : execMask;
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
    if (amdgpu::cdna1_ds_atomic_nullified(d, arch, wf.vgpr_alloc())) {
      if (!obligations.empty())
        rs->registerEvent(instruction, type, {}, 0, 0xf, obligations, memoryOrder);
      return;
    }
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
      if (auto destinations = validated_load_destinations(d, wf))
        registers = std::move(*destinations);
    }

    const auto bounds =
        amdgpu::indexed_ds_access_bounds(d, wf.cu().arch(), wf.lds_base(), wf.lds_size());
    const auto atomicWidth = [&](uint64_t address, uint64_t bytes) -> uint8_t {
      return ldsAccess.read_modify_write && bytes == d.elem_size && address % d.elem_size == 0
                 ? d.elem_size
                 : 0;
    };
    const bool secondAccess = d.has_second_ds_address(wf.cu().arch(), ldsMask);
    IntervalSet src2Reads;
    auto secondAddresses = access.secondary_addresses;
    // A caller with an unknown route may supply only the instruction state.
    if (d.ds_src2 || (access.route == amdgpu::MemoryRoute::UNKNOWN && secondAccess))
      secondAddresses = d.ds2_per_lane_addr;
    const std::array<std::span<const uint64_t>, 2> addressSets{d.per_lane_addr, secondAddresses};
    const size_t numAddressSets = secondAccess || d.ds_src2 ? 2 : 1;
    // Retain each address set's own in-bounds lanes and bytes within the LDS allocation.
    std::array<std::array<uint32_t, 64>, 2> laneBytes{};
    // Appending one address set at a time lets neighboring lanes merge into
    // the previous interval before the final sort.
    IntervalSet intervals;
    for (size_t set = 0; set < numAddressSets; ++set) {
      const uint64_t lanes = ldsMask & ~(set == 0 ? d.ds_oob_lane_mask : d.ds2_oob_lane_mask);
      for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
        if (!(lanes & (1ULL << lane)))
          continue;
        const uint64_t address = addressSets[set][lane];
        if ((allocation && address < allocation->begin) || (bounds && !bounds->contains(address)))
          continue;
        auto bytes = detector->ldsBytesInRange(address, perLaneBytes, elementBytes);
        if (allocation)
          bytes = allocation->retained_bytes(address, bytes, elementBytes);
        else if (end)
          bytes = amdgpu::whole_element_bytes_in_range(address, bytes, elementBytes, *end);
        if (bytes == 0)
          continue;
        // Interval endpoints use signed indices even when the configured LDS
        // backing is larger. Keep register and counter dependencies below.
        if (address + bytes > std::numeric_limits<int>::max())
          continue;
        if (set == 1 && d.ds_src2) {
          detector->validateRead(static_cast<int>(address), waveId, lane, bytes, ldsOrder);
          src2Reads.append(static_cast<int>(address), static_cast<int>(address + bytes));
          continue;
        }
        if (ldsAccess.aligned_atomic_bytes && !atomicWidth(address, bytes))
          ldsAccess.aligned_atomic_bytes = 0;
        laneBytes[set][lane] = static_cast<uint32_t>(bytes);
        intervals.append(static_cast<int>(address), static_cast<int>(address + bytes));
      }
    }
    intervals.finalize();
    // The global pipeline validates translations before touching mixed FLAT's LDS lanes.
    // Keep the registers and counter positions at issue, and attach the LDS bytes at access.
    const bool defer_lds =
        mixedFlat && d.routed_issue_info && !intervals.empty() && !obligations.empty();
    if (defer_lds) {
      auto &pending = s->mixed_lds.emplace();
      pending.order = ldsOrder;
      pending.reads = d.is_load || ldsAccess.read_modify_write;
      pending.writes = !d.is_load || ldsAccess.read_modify_write;
    }
    const bool checkReads = !defer_lds && (d.is_load || ldsAccess.read_modify_write) &&
                            detector->mayConflict(intervals, waveId, /*isWrite=*/false, ldsOrder);
    const bool checkWrites = !defer_lds && (!d.is_load || ldsAccess.read_modify_write) &&
                             detector->mayConflict(intervals, waveId, /*isWrite=*/true, ldsOrder);
    if (defer_lds || checkReads || checkWrites) {
      for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
        for (size_t set = 0; set < numAddressSets; ++set) {
          const int bytes = static_cast<int>(laneBytes[set][lane]);
          if (bytes == 0)
            continue;
          const int address = static_cast<int>(addressSets[set][lane]);
          const auto atomicBytes = atomicWidth(address, bytes);
          if (defer_lds) {
            s->mixed_lds->lanes.push_back({address, static_cast<int>(lane), bytes, atomicBytes});
            continue;
          }
          if (checkReads)
            detector->validateRead(address, waveId, static_cast<int>(lane), bytes, ldsOrder,
                                   atomicBytes);
          if (checkWrites)
            detector->validateWrite(address, waveId, static_cast<int>(lane), bytes, ldsOrder,
                                    atomicBytes);
        }
      }
    }
    uint8_t byte_mask = vector_memory_byte_mask(d, wf);
    if (d.is_load) {
      for (uint32_t reg : registers)
        rs->checkVgprWrite(static_cast<int>(reg), execMask, byte_mask, writeOrder);
    }
    if (obligations.empty())
      return;
    // Only ordinary LDS lanes contribute intervals. When results split by route,
    // the vector-memory lanes have already been registered separately.
    if (defer_lds)
      s->mixed_lds->intervals = std::exchange(intervals, IntervalSet{});
    const auto event = rs->registerEventWithIntervals(
        instruction, type, std::move(registers), execMask, byte_mask, std::move(intervals),
        obligations, secondAccess ? VgprWriteOrder{memoryOrder} : writeOrder,
        /*lastRegisterByteMask=*/0, ldsAccess,
        /*feedback=*/{}, defer_lds);
    if (event.isValid() && d.ds_src2) {
      src2Reads.finalize();
      detector->attachSrc2Source(event, std::move(src2Reads));
    }
    if (defer_lds) {
      if (event.isValid())
        s->mixed_lds->event = event;
      else
        s->mixed_lds.reset();
    }
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
    } else if (d.lds_src) {
      // An async store reads its LDS source, which stays an LDS read until a
      // wait retires the store. Async copies may access LDS out of order with
      // each other (CDNA5 ISA 10.8). The ISA does not establish their LDS access
      // order relative to DS instructions or tensor DMA, so as a conservative
      // rule the read is checked as unordered against every pending LDS write,
      // including the wave's own DS writes, async loads and tensor loads.
      // Each lane reads the leading bytes the execution recorded: the bytes of
      // its source inside the allocation. They are checked and registered at
      // byte granularity.
      std::vector<LdsByteRange> ranges;
      ranges.reserve(wf.wf_size());
      for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
        const uint32_t address = d.per_lane_lds_addr[lane];
        const uint32_t laneBytes = d.per_lane_lds_src_bytes[lane];
        if (!(d.exec_mask & (1ULL << lane)) || address == amdgpu::kInvalidLdsAddress ||
            laneBytes == 0)
          continue;
        ranges.push_back({.address = address, .bytes = laneBytes});
        const auto bytes = detector->ldsBytesInRange(address, laneBytes, /*elementBytes=*/1);
        if (bytes != 0)
          detector->validateRead(static_cast<int>(address), waveId, static_cast<int>(lane),
                                 static_cast<int>(bytes), MemoryOrderClass::UNORDERED);
      }
      if (obligations.empty())
        return;
      // One event per instruction, so the store takes one ASYNCcnt slot, also
      // when no lane reads LDS.
      rs->registerLdsRangeEvent(instruction, MemoryEventType::LDS_TO_GLOBAL, ranges, obligations,
                                memoryOrder);
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
  const InstructionSnapshot *current = current_instruction(*s, wf);
  if (!current)
    return;
  const InstructionSnapshot &instruction = *current;
  auto *rs = s->race_state;
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
  // Tensor instructions carry no memory issue metadata, so their TENSORcnt
  // backpressure is applied here, once per tensor instruction, before the
  // transfer's LDS bytes are checked. That way the issue can complete the
  // oldest transfer first, as the stall at a full TENSORcnt does.
  if (rs->prepareForCounterIncrement(amdgpu::WaitCounterType::TENSORCNT).failed()) {
    report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
    return;
  }
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

  // TENSORcnt counts each tensor instruction (CDNA5 ISA 10.11.1), so one that
  // moves no data, such as a gather with no valid indices or a store wholly
  // outside the tensor, still occupies a TENSORCNT slot, and its event is
  // registered even without LDS ranges. A NULL descriptor and a NOP tile (zero
  // tile_dim0) are registered the same way, matching the emulator's assumption
  // that they count on TENSORCNT; hardware behavior for them is not
  // established.
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
  util::FailureOr<InstructionSnapshot> instruction = capture_instruction(pc, inst, wf);
  if (instruction.failed())
    return;
  auto *s = get_state(wf);
  assert(s && s->race_state);
  s->race_state->finishRoutedIssue();
  s->trace.push(instruction.value());
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
  const auto operation = inst.amdgpu_operation();
  // Inline queries reserve their issue slot before accessing mip/coordinate
  // operands. A successful execution publishes the captured destination lanes.
  const bool legacy_lod =
      operation == amdgpu::Operation::IMAGE_GET_LOD && amdgpu::has_ordered_vmem_reads(arch);
  if (inst.is_memory_wait_producer() && !inst.is_memory_op() &&
      (operation == amdgpu::Operation::IMAGE_GET_RESINFO || legacy_lod)) {
    const auto counter = arch == rocjitsu::IsaFamily::Rdna4 ? amdgpu::WaitCounterType::LOADCNT
                                                            : amdgpu::WaitCounterType::VMCNT;
    if (s->race_state->prepareForCounterIncrement(counter).failed()) {
      report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
      return;
    }
    s->image_query.emplace(amdgpu::vmem_read_type(arch, inst.mnemonic()), counter);
  }
  // Fixed-domain FLAT reserves both slots before reading its address.
  // Other FLAT requests reserve their slots at routing.
  if (inst.is_memory_op() && (inst.amdgpu_memory_addressing() != amdgpu::MemoryAddressing::Flat ||
                              amdgpu::flat_issues_both_counter_domains(arch))) {
    const auto *info = inst.amdgpu_memory_issue_info();
    assert(info && "memory instruction reached the race detector without issue metadata");
    util::Result admitted = util::Result::success();
    if (info && !ignored_memory_issue(inst, wf)) {
      if (!info->exec_masked || wf.exec() != 0 || arch_is_rdna(arch))
        admitted = s->race_state->prepareForMemoryIssue(*info);
      else if (exec_zero_memory_issue_counts(arch))
        admitted = s->race_state->prepareForMemoryIssue(
            exec_zero_counted_obligations(*info, inst.mnemonic()));
    }
    if (admitted.failed()) {
      report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
      return;
    }
  }

  // Wait-counter producers without memory issue metadata, such as GLOBAL_INV
  // and the LDS lane permutes, apply their gfx1250 backpressure here and then
  // hold their slot (see counter_only_slot()).
  if (wf.cu().arch() == rocjitsu::IsaFamily::Cdna5 && inst.is_memory_wait_producer() &&
      !inst.is_memory_op()) {
    if (const auto slot = counter_only_slot(inst.amdgpu_operation())) {
      const std::span<const amdgpu::MemoryCounterObligation> obligations{&slot->obligation, 1};
      if (s->race_state->prepareForCounterIncrement(slot->obligation.wait_counter_type())
              .failed()) {
        report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
        return;
      }
      register_counter_slot(*s->race_state, instruction.value(), obligations,
                            memoryOrderForObligations(obligations), slot->is_load);
    }
  }

  // Messages are not memory instructions, so they remain outside memory
  // issue metadata. They count on LGKMCNT before GFX12 and on KMcnt on
  // gfx1250 (CDNA5 ISA 5.7.1.5), where S_SENDMSG_RTN_B* adds two tokens, one
  // for the message and one for its returned data (CDNA5 ISA 5.4).
  if ((operation == amdgpu::Operation::S_SENDMSG || operation == amdgpu::Operation::S_SENDMSGHALT ||
       operation == amdgpu::Operation::S_SENDMSG_RTN_B32 ||
       operation == amdgpu::Operation::S_SENDMSG_RTN_B64) &&
      (arch == rocjitsu::IsaFamily::Cdna5 || counterCapacitiesForArch(arch).lgkmcnt != 0)) {
    const bool cdna5 = wf.cu().arch() == rocjitsu::IsaFamily::Cdna5;
    const auto counter = cdna5 ? amdgpu::WaitCounterType::KMCNT : amdgpu::WaitCounterType::LGKMCNT;
    const uint8_t increment = cdna5 && (operation == amdgpu::Operation::S_SENDMSG_RTN_B32 ||
                                        operation == amdgpu::Operation::S_SENDMSG_RTN_B64)
                                  ? 2
                                  : 1;
    if (s->race_state->prepareForCounterIncrement(counter, increment).failed()) {
      report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
      return;
    }
  }
  // Implicit VCC operands use wave mask operations directly. Observe only
  // the architectural mask, excluding temporary VCC saves by SDWA execution.
  if (wf.cu().arch() == rocjitsu::IsaFamily::Cdna3 &&
      !s->race_state->getWaveMemoryEvents().empty()) {
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
    const auto *instruction = current_instruction(*s, wf);
    if (!instruction)
      return;
    for (const auto &read : s->lds_operand_reads)
      s->race_state->recordLdsOperandRead(*instruction, read.address, read.bytes, read.lane);
    s->lds_operand_reads.clear();
  }
  if (auto &query = s->image_query) {
    const auto *instruction = current_instruction(*s, wf);
    if (!instruction)
      return;
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
    s->race_state->registerEvent(*instruction, MemoryEventType::GLOBAL_TO_VGPR,
                                 std::move(query->registers), query->lanes, kFullByteMask,
                                 obligations, order);
    query.reset();
    return;
  }
  if (const auto reg = wf.instruction_exec_write();
      reg && reg->index < wf.wf_size() / 32 && !wf.instruction_execution_failed())
    s->race_state->checkGdsExecWrite();
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
  if (wf.cu().arch() != rocjitsu::IsaFamily::Cdna2)
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

namespace {

BarrierRoundId roundId(const BarrierRound &round) {
  return {
      .barrier = round.completion_bit, .namedId = round.named_id, .generation = round.generation};
}

} // namespace

void RaceDetectorPlugin::onAmdgpuBarrierSignaled(const amdgpu::Wavefront &wf,
                                                 const BarrierRound &round) {
  // The barrier hooks are not filtered by wave subscription, so a wave this
  // plugin does not track can reach them.
  WaveRaceState *race = tracked_race_state(wf);
  if (!race)
    return;
  race->getDetector()->barrierSignaled(race->getWaveId(), roundId(round));
}

void RaceDetectorPlugin::onAmdgpuBarrierRoundCompleted(std::span<amdgpu::Wavefront *> notified,
                                                       std::span<amdgpu::Wavefront *> armed,
                                                       const BarrierRound &round) {
  const bool wholeWorkgroup = round.kind == BarrierKind::Workgroup;
  // Each notified workgroup's detector hears of the completion, with its
  // armed waves, which it then awaits. A detector whose waves were all
  // notified with their bit already set has none to await and closes the
  // round.
  if (round.kind != BarrierKind::Cluster) {
    // Workgroup and named rounds notify the waves of one workgroup, and so of
    // one detector, which holds at most RaceDetector::kMaxWaves waves.
    RaceDetector *detector = nullptr;
    for (auto *wf : notified) {
      if (WaveRaceState *race = tracked_race_state(*wf)) {
        detector = race->getDetector();
        break;
      }
    }
    if (!detector)
      return;
    std::array<WaveId, RaceDetector::kMaxWaves> waves;
    size_t count = 0;
    for (auto *wf : armed) {
      WaveRaceState *race = tracked_race_state(*wf);
      if (!race)
        continue;
      assert(detector == race->getDetector());
      assert(count < waves.size());
      waves[count++] = race->getWaveId();
    }
    detector->barrierRoundCompleted(roundId(round), std::span(waves.data(), count), wholeWorkgroup);
    return;
  }
  // A cluster round's members span workgroups, and so detectors. Each detector
  // orders only its own waves; ordering between workgroups is not modeled.
  std::vector<std::pair<RaceDetector *, std::vector<WaveId>>> members;
  const auto entry = [&](RaceDetector *detector) -> std::vector<WaveId> & {
    auto it = std::ranges::find(members, detector, &decltype(members)::value_type::first);
    if (it == members.end())
      it = members.insert(members.end(), {detector, {}});
    return it->second;
  };
  for (auto *wf : notified)
    if (WaveRaceState *race = tracked_race_state(*wf))
      entry(race->getDetector());
  for (auto *wf : armed)
    if (WaveRaceState *race = tracked_race_state(*wf))
      entry(race->getDetector()).push_back(race->getWaveId());
  for (auto &[detector, waves] : members)
    detector->barrierRoundCompleted(roundId(round), waves, wholeWorkgroup);
}

void RaceDetectorPlugin::onAmdgpuBarrierWaitPassed(const amdgpu::Wavefront &wf,
                                                   const BarrierRound &round) {
  WaveRaceState *race = tracked_race_state(wf);
  if (!race)
    return;
  race->getDetector()->barrierWaitPassed(race->getWaveId(), roundId(round));
}

} // namespace rocjitsu::plugins::race_detector
