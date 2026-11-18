// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/plugin.h"
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
#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"
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

bool supports_counter_capacity(rj_code_arch_t arch) {
  switch (arch) {
  case ROCJITSU_CODE_ARCH_CDNA1:
  case ROCJITSU_CODE_ARCH_CDNA2:
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
  case ROCJITSU_CODE_ARCH_RDNA3:
  case ROCJITSU_CODE_ARCH_RDNA3_5:
  case ROCJITSU_CODE_ARCH_CDNA5:
    return true;
  default:
    return false;
  }
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
bool exec_zero_memory_issue_counts(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
         arch == ROCJITSU_CODE_ARCH_CDNA5;
}

bool pending_counter(const WaveRaceState &state, amdgpu::WaitCounterType counter) {
  return state.pendingOpaqueCounterCredits(counter) != 0 ||
         std::ranges::any_of(state.getWaveMemoryEvents(), [&](EventId event) {
           return state.getDetector()->events().pendingCounterIncrement(event, counter) != 0;
         });
}

/// The obligations on which a memory instruction issued with EXEC zero is sure
/// to count. On CDNA1 through CDNA4, buffer, global and scratch instructions
/// keep VMCNT; DS and generic FLAT require separate qualification. On CDNA5,
/// the instruction is skipped only when the counters its encoding names are
/// all zero (CDNA5 ISA 3.2.2.1): "VMEM can be skipped only if: EXEC == 0 and
/// LOADcnt / STOREcnt / ASYNCcnt / TENSORcnt == 0", FLAT also requires DScnt ==
/// 0, and "LDS can be skipped only if: DScnt==0 and EXEC==0". On those counters
/// it either counts or was skipped because they were empty, so admitting it and
/// recording its slot are sound. On any other counter, such as the ASYNCcnt of
/// ds_atomic_async_barrier_arrive_b64, an LDS instruction, it may be skipped
/// while the counter is nonzero unless physical DScnt is pending. On CDNA1
/// through CDNA4, buffer, global, and scratch instructions retain VMCNT.
amdgpu::MemoryIssueInfo exec_zero_counted_obligations(const amdgpu::MemoryIssueInfo &info,
                                                      std::string_view mnemonic,
                                                      bool physical_ds_pending,
                                                      rj_code_arch_t arch) {
  const bool lds = mnemonic.starts_with("ds_");
  const bool flat = mnemonic.starts_with("flat_");
  const bool legacy_cdna = arch_is_cdna_4_or_lower(arch);
  // The decoded counter obligation identifies native VMEM even when a test or
  // an inline instruction has no architectural mnemonic.
  const bool native_vmem = !lds && !flat;
  const bool async_arrive_issues =
      mnemonic == "ds_atomic_async_barrier_arrive_b64" && physical_ds_pending;
  amdgpu::MemoryIssueInfo counted;
  counted.exec_masked = info.exec_masked;
  for (const auto obligation : info.counter_obligations()) {
    bool counts = false;
    if (legacy_cdna) {
      counts = native_vmem && obligation.wait_counter_type() == amdgpu::WaitCounterType::VMCNT;
    } else {
      switch (obligation.wait_counter_type()) {
      case amdgpu::WaitCounterType::DSCNT:
        counts = lds || flat;
        break;
      case amdgpu::WaitCounterType::LOADCNT:
      case amdgpu::WaitCounterType::STORECNT:
      case amdgpu::WaitCounterType::ASYNCCNT:
        counts = !lds || async_arrive_issues;
        break;
      case amdgpu::WaitCounterType::TENSORCNT:
        counts = !lds;
        break;
      case amdgpu::WaitCounterType::VMCNT:
        counts = !lds && !flat;
        break;
      default:
        break;
      }
    }
    if (counts)
      counted.counter_obligations_[counted.num_counter_obligations_++] = obligation;
  }
  return counted;
}

/// A modern buffer instruction whose resource descriptor TYPE is not a buffer
/// has no memory effect. Its counter admission is unspecified, so it provides
/// no younger completion credit that could prove an older request finished.
bool ignored_memory_issue(const Instruction &inst, const amdgpu::Wavefront &wf) {
  // Modern buffer instructions have conditional memory lanes, so others
  // skip decoding their register modifiers.
  if (!(inst.flags() & CONDITIONAL_MEMORY_LANES))
    return false;
  amdgpu::RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  return amdgpu::addr_calc::modern_ignores_buffer_resource(modifiers, wf);
}

/// Record a counter slot that an instruction holds without registers or LDS
/// bytes, as an event, so that later capacity checks and partial waits count
/// it. A GFX11/GFX12 image issue without result lanes holds a slot on each
/// pending image counter. Other cases include:
/// - A gfx1250 or gfx950 memory instruction issued with EXEC zero, on the counters
///   exec_zero_counted_obligations() selects. Recording the slot is sound even
///   if the instruction was skipped: those counters were then zero, so every
///   earlier operation on them had completed, and oldest-first retirement
///   reaches the slot before any later operation. A FLAT instruction's slots
///   are recorded by register_unused_flat_domains(), so its routed obligations
///   here are empty.
/// - A cache operation or gfx1250 wait-counter producer without memory issue
///   metadata, on the counter counter_only_slot() selects.
void register_counter_slot(WaveRaceState &race, const InstructionSnapshot &instruction,
                           std::span<const amdgpu::MemoryCounterObligation> obligations,
                           MemoryOrderClass memoryOrder, bool is_load) {
  if (obligations.empty())
    return;
  race.registerEvent(instruction,
                     is_load ? MemoryEventType::GLOBAL_TO_VGPR : MemoryEventType::VGPR_TO_GLOBAL,
                     /*registers=*/{}, /*execMask=*/0, /*byteMask=*/0xF, obligations, memoryOrder);
}

/// The counter slot a wait-counter producer without memory issue metadata
/// takes when it issues, and whether it returns data:
/// - CDNA2-4 BUFFER_WBL2 and CDNA3-4 BUFFER_INV return a VMcnt acknowledgment
///   in issue order with vector memory.
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

std::optional<CounterOnlySlot> counter_only_slot(std::string_view mnemonic, rj_code_arch_t arch) {
  using amdgpu::MemoryCompletionClass;
  using amdgpu::WaitCounterType;
  if (arch == ROCJITSU_CODE_ARCH_CDNA2 || arch == ROCJITSU_CODE_ARCH_CDNA3 ||
      arch == ROCJITSU_CODE_ARCH_CDNA4) {
    if (mnemonic == "buffer_inv" || mnemonic == "buffer_wbl2")
      return CounterOnlySlot{{WaitCounterType::VMCNT, MemoryCompletionClass::VMEM},
                             mnemonic == "buffer_inv"};
    return std::nullopt;
  }
  if (arch != ROCJITSU_CODE_ARCH_CDNA5)
    return std::nullopt;
  if (mnemonic == "global_inv")
    return CounterOnlySlot{{WaitCounterType::LOADCNT, MemoryCompletionClass::VMEM}, true};
  if (mnemonic == "global_wb" || mnemonic == "global_wbinv")
    return CounterOnlySlot{{WaitCounterType::STORECNT, MemoryCompletionClass::VMEM}, false};
  if (mnemonic == "ds_permute_b32" || mnemonic == "ds_bpermute_b32" ||
      mnemonic == "ds_bpermute_fi_b32" || mnemonic == "ds_swizzle_b32" || mnemonic == "ds_nop")
    return CounterOnlySlot{{WaitCounterType::DSCNT, MemoryCompletionClass::LDS}, true};
  return std::nullopt;
}

/// On gfx1250 (CDNA5) "Every Flat instruction increments both DScnt and either
/// LOADcnt or STOREcnt when the instruction issues" (CDNA5 ISA 5.7.1.3),
/// whichever memory its addresses resolve to.
bool flat_counts_on_both_domains(rj_code_arch_t arch) { return arch == ROCJITSU_CODE_ARCH_CDNA5; }

/// Keep each unused-route counter slot of a dual-credit FLAT instruction.
/// GFX10 folds counter-only positions into the preceding ordered event.
/// GFX11 keeps ordered issue positions, RDNA4 retains bounded opaque credits,
/// and gfx1250 reserves both positions before routing.
/// `routed` holds the obligations of the domains the addresses used; its
/// event carries the instruction's data dependencies.
void register_unused_flat_domains(WaveRaceState &race, const InstructionSnapshot &instruction,
                                  const amdgpu::MemoryIssueInfo &decoded,
                                  const amdgpu::MemoryIssueInfo &routed, bool is_load,
                                  rj_code_arch_t arch) {
  for (const auto obligation : decoded.counter_obligations()) {
    const bool used = std::ranges::any_of(routed.counter_obligations(), [&](const auto routed_one) {
      return routed_one.wait_counter_type() == obligation.wait_counter_type();
    });
    if (used)
      continue;
    if (arch != ROCJITSU_CODE_ARCH_CDNA5 && arch != ROCJITSU_CODE_ARCH_RDNA1 &&
        arch != ROCJITSU_CODE_ARCH_RDNA2 && arch != ROCJITSU_CODE_ARCH_RDNA3 &&
        arch != ROCJITSU_CODE_ARCH_RDNA3_5) {
      race.recordOpaqueCounterCredit(obligation.wait_counter_type(), obligation.completion_class(),
                                     obligation.counter_increment());
      continue;
    }
    // GFX10 FLAT positions without routed data do not establish native LDS
    // or VMEM result ordering. Keep the physical counter increment but use
    // an unordered class for that conditional issue slot.
    const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
    const auto slot_obligation =
        gfx10 ? amdgpu::MemoryCounterObligation{obligation.wait_counter_type(),
                                                amdgpu::MemoryCompletionClass::UNORDERED,
                                                obligation.counter_increment()}
              : obligation;
    const std::array slot{slot_obligation};
    race.registerEvent(
        instruction, is_load ? MemoryEventType::GLOBAL_TO_VGPR : MemoryEventType::VGPR_TO_GLOBAL,
        /*registers=*/{}, /*execMask=*/0, /*byteMask=*/0xF, slot,
        slot_obligation.completion_class());
  }
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

// VMEM can retire its counter in order even when the VGPR writes can pass one another.
MemoryOrderClass vmem_vgpr_write_order(MemoryOrderClass completion_order, rj_code_arch_t arch) {
  return completion_order == MemoryOrderClass::VMEM && !amdgpu::has_ordered_vmem_writeback(arch)
             ? MemoryOrderClass::UNORDERED
             : completion_order;
}

std::optional<amdgpu::MemoryIssueInfo>
routed_flat_issue_info(const amdgpu::MemoryAccessObservation &access,
                       const amdgpu::MemoryIssueInfo &decoded, rj_code_arch_t arch,
                       const Instruction &inst, bool returned_without_request) {
  // A FLAT opcode with an explicit global or LDS segment keeps its FLAT
  // issue behavior even when routing reports that segment as its space.
  if ((access.decoded_space != amdgpu::DecodedMemorySpace::FLAT &&
       !inst.mnemonic().starts_with("flat_")) ||
      (access.route != amdgpu::MemoryRoute::GLOBAL && access.route != amdgpu::MemoryRoute::LOCAL))
    return std::nullopt;

  // On GFX10 and GFX11 a generic FLAT result remains pending on both
  // counters even after the actual address route is known.
  if (amdgpu::flat_requires_both_counters(arch) && access.active_lane_mask != 0 &&
      (access.is_load || access.request_lane_mask == 0))
    return decoded;
  // A private-OOB load returns zero through the VM result path without a
  // byte request. The pipeline's dual issue record must not attach that
  // result to its unused LDS position on split-result targets.
  if (access.request_lane_mask == 0 && returned_without_request)
    return decoded.for_flat_memory_domains(true, false);
  if (const auto &routed = inst.data_as<amdgpu::VectorMemState>()->routed_issue_info)
    return routed;

  if (access.request_lane_mask == 0)
    return decoded.for_flat_issue(arch, 0, 0, access.active_lane_mask == 0);
  if (amdgpu::flat_requires_both_counters(arch))
    return decoded;

  const uint64_t shared_lanes =
      (access.flat_local_lane_mask | access.flat_dds_lane_mask) & access.request_lane_mask;
  // Mixed requests include both address spaces. The consumer can attach each
  // obligation to its own lanes on targets with independent return paths.
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
                                  uint64_t access_pc, rj_code_arch_t arch) {
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
  if (ignored_memory_issue(inst, wf))
    return;
  const auto *decoded = inst.amdgpu_memory_issue_info();
  assert(decoded && "memory instruction reached the race detector without issue metadata");
  if (!decoded)
    return;
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA3 && access.scratch_lane_mask &&
      (access.decoded_space == amdgpu::DecodedMemorySpace::SCRATCH ||
       access.decoded_space == amdgpu::DecodedMemorySpace::FLAT))
    rs->checkScalarRead(wf.memory_wait_shadow().canonical({RegClass::FLAT_SCRATCH, 0, 2}));

  const auto arch = wf.cu().arch();
  const auto *memory = access.decoded_space == amdgpu::DecodedMemorySpace::FLAT
                           ? inst.data_as<amdgpu::VectorMemState>()
                           : nullptr;
  // The legacy private bounds check can clear every byte request while a FLAT
  // load still returns zero to active lanes. This is a VM result, not an LDS
  // result, and it must retain its wait obligation.
  const bool returned_without_request =
      (arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
       arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2) &&
      memory && memory->is_load && !memory->nullified_result &&
      (access.active_lane_mask & ~access.request_lane_mask);
  auto routed = routed_flat_issue_info(access, *decoded, arch, inst, returned_without_request);
  // Dual-credit FLAT reserves both counter positions, while its result uses
  // the counter for the lane's actual route on split-counter targets.
  const bool dual_route = routed &&
                          (amdgpu::flat_issues_both_routed_domains(arch) ||
                           amdgpu::legacy_flat_issues_both_counter_domains(arch));
  const bool split_route_results = dual_route && !amdgpu::flat_requires_both_counters(arch);
  const uint64_t sharedLanes = access.flat_local_lane_mask | access.flat_dds_lane_mask;
  const uint64_t globalRequests = access.request_lane_mask & ~sharedLanes;
  // Routing can normalize a FLAT load to LOCAL_MEM when its only requests
  // are LDS. An active private-OOB lane still returns a non-LDS zero result.
  // A legacy private-OOB lane can return zero without a byte request, even
  // when the only actual request routes to LDS and the instruction normalizes
  // to LOCAL_MEM. That zero result still belongs to the VM half.
  const bool mixedFlat = sharedLanes != 0 &&
                         (inst.data()->tag() == amdgpu::GLOBAL_MEM ||
                          (returned_without_request &&
                           (access.active_lane_mask & ~sharedLanes) != 0));
  if (split_route_results && mixedFlat)
    routed = decoded->for_flat_memory_domains(false, true);
  else if (split_route_results && access.request_lane_mask != 0)
    routed = decoded->for_flat_memory_domains(sharedLanes == 0, sharedLanes != 0);

  const auto &issue = routed ? *routed : *decoded;
  const auto obligations = issue.counter_obligations();
  const MemoryOrderClass memoryOrder = memoryOrderForObligations(obligations);
  const bool gfx10 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  VgprWriteOrder writeOrder{vgpr_writeback_order(memoryOrder, arch),
                            gfx10 && routed && access.is_load ? amdgpu::VmemWriteback::GenericFlat
                            : amdgpu::has_ordered_vmem_reads(arch) &&
                                      memoryOrder == MemoryOrderClass::VMEM
                                ? amdgpu::vmem_read_type(arch, inst.mnemonic())
                                : amdgpu::VmemWriteback::Ordinary};
  if (routed && access.is_load && amdgpu::flat_requires_both_counters(arch) &&
      amdgpu::has_ordered_flat_lds_writeback(arch))
    writeOrder.flat_lds_lanes = sharedLanes;
  if (memoryOrder == MemoryOrderClass::LDS && amdgpu::has_ordered_lds_direct_writeback(arch) &&
      std::ranges::any_of(obligations, [](auto o) {
        return o.wait_counter_type() == amdgpu::WaitCounterType::EXPCNT;
      }))
    writeOrder.direct_lds = true;
  if (!inst.mnemonic().starts_with("flat_") &&
      (inst.data()->tag() == amdgpu::LOCAL_MEM || inst.data()->tag() == amdgpu::GLOBAL_MEM)) {
    const auto &state = *inst.data_as<amdgpu::VectorMemState>();
    const bool gfx11 = arch == ROCJITSU_CODE_ARCH_RDNA3 ||
                       arch == ROCJITSU_CODE_ARCH_RDNA3_5;
    if (state.exec_mask == 0 && (gfx10 || gfx11) &&
        access.decoded_space != amdgpu::DecodedMemorySpace::GDS &&
        !inst.mnemonic().starts_with("ds_gws_") && !ignored_memory_issue(inst, wf)) {
      // An empty native issue still advances its ordered counter while an
      // older request is outstanding. GFX10 issues even without that older
      // request; GFX11 needs an older modeled counter position to justify
      // its conditional slot. Keep it without register or LDS bytes.
      const bool local = inst.data()->tag() == amdgpu::LOCAL_MEM;
      for (const auto &obligation : obligations) {
        if (gfx11 && !pending_counter(*rs, obligation.wait_counter_type()))
          continue;
        const bool returns = state.is_load &&
                             obligation.wait_counter_type() != amdgpu::WaitCounterType::EXPCNT;
        const auto type = local ? (returns ? MemoryEventType::LDS_TO_VGPR
                                           : MemoryEventType::VGPR_TO_LDS)
                                : (returns ? MemoryEventType::GLOBAL_TO_VGPR
                                           : MemoryEventType::VGPR_TO_GLOBAL);
        // EXEC-zero native memory skips the before-execute admission pass on
        // these targets. Retire a full counter's oldest slot before recording
        // the conditional issue, just as an active issue does.
        if (supports_counter_capacity(arch) &&
            rs->prepareForCounterIncrement(obligation.wait_counter_type(),
                                           obligation.counter_increment())
                .failed()) {
          report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
          return;
        }
        rs->registerEvent(instruction, type, {}, 0, 0xf, std::span(&obligation, 1),
                          obligation.completion_class());
      }
      return;
    }
  }
  if (routed) {
    const bool is_load = inst.data_as<amdgpu::VectorMemState>()->is_load;
    const bool empty_exec = access.request_lane_mask == 0 && access.active_lane_mask == 0;
    // CDNA5 reserves both positions before reading the operands. Elsewhere
    // only the requests observed here establish which positions can advance.
    const bool admitted_before_routing =
        flat_counts_on_both_domains(arch) ||
        (access.active_lane_mask != 0 && amdgpu::flat_issues_both_counter_domains(arch));
    bool issue_empty = false;
    if (dual_route && empty_exec) {
      // The plugin's events can remain pending after functional memory returns.
      using C = amdgpu::WaitCounterType;
      issue_empty = amdgpu::flat_issues_with_empty_exec(
          arch,
          pending_counter(*rs, C::VMCNT) || pending_counter(*rs, C::VSCNT) ||
              pending_counter(*rs, C::LOADCNT) || pending_counter(*rs, C::STORECNT),
          pending_counter(*rs, C::DSCNT), pending_counter(*rs, C::SAMPLECNT));
    }
    auto credits = dual_route ? decoded->for_flat_issue(arch, access.request_lane_mask, sharedLanes,
                                                        empty_exec, issue_empty,
                                                        returned_without_request)
                              : issue;
    // A GFX11 empty-EXEC FLAT can skip after the physical VM queues drain,
    // independently of an older DS request. A modeled pending VM operation
    // can justify only the VM position if the instruction may have skipped.
    if (dual_route && empty_exec &&
        (arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5))
      credits = decoded->for_flat_memory_domains(issue_empty, false);
    // Fixed-domain FLAT reserves both issue positions. On CDNA1-4 the
    // capacity proof is applied only after the requested route is known.
    const auto capacity_routes = amdgpu::flat_issues_both_counter_domains(arch)
                                     ? credits
                                     : decoded->for_flat_memory_domains(globalRequests != 0,
                                                                        sharedLanes != 0);
    if (!admitted_before_routing && !capacity_routes.empty() && supports_counter_capacity(arch) &&
        rs->prepareForMemoryIssue(capacity_routes).failed()) {
      report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
      return;
    }
    if (empty_exec) {
      if ((wf.exec() == 0 || inst.data_as<amdgpu::VectorMemState>()->nullified_result) &&
          admitted_before_routing) {
        // The pipeline's routed metadata records whether a zero-EXEC FLAT
        // actually occupied its issue positions. With all physical queues
        // idle the hardware may skip it and no counter event exists.
        if (memory && memory->routed_issue_info &&
            (!memory->routed_issue_info->empty() || issue_empty)) {
          const auto counted = exec_zero_counted_obligations(
              *decoded, inst.mnemonic(), wf.wait_counters().dscnt != 0, arch);
          register_unused_flat_domains(*rs, instruction, counted, {}, is_load, arch);
        }
      } else if (!admitted_before_routing) {
        register_unused_flat_domains(*rs, instruction, credits, {}, is_load, arch);
      }
      return;
    }
    if (dual_route && !mixedFlat)
      register_unused_flat_domains(*rs, instruction, credits, issue, is_load, arch);
  }

  // An instruction with effective EXEC zero can hold its counter slot,
  // including a load whose destination nullifies every request. An ignored
  // buffer instruction does not issue (see ignored_memory_issue()).
  const auto record_exec_zero_slot = [&](bool is_load, bool nullified_result) {
    if (!exec_zero_memory_issue_counts(arch) ||
        (arch_is_cdna_4_or_lower(arch) && wf.exec() != 0) ||
        (wf.exec() != 0 && !nullified_result) ||
        ignored_memory_issue(inst, wf))
      return;
    const auto counted =
        exec_zero_counted_obligations(issue, inst.mnemonic(), wf.wait_counters().dscnt != 0,
                                      arch);
    for (const auto obligation : counted.counter_obligations()) {
      const auto &physical = wf.wait_counters();
      bool issued = arch != ROCJITSU_CODE_ARCH_CDNA5 ||
                    pending_counter(*rs, obligation.wait_counter_type());
      switch (obligation.wait_counter_type()) {
      case amdgpu::WaitCounterType::LOADCNT:
        issued |= physical.vmcnt != 0;
        break;
      case amdgpu::WaitCounterType::STORECNT:
        issued |= physical.vscnt != 0;
        break;
      case amdgpu::WaitCounterType::DSCNT:
        issued |= physical.dscnt != 0;
        break;
      case amdgpu::WaitCounterType::ASYNCCNT:
        issued |= physical.asynccnt != 0;
        break;
      case amdgpu::WaitCounterType::TENSORCNT:
        issued |= physical.tensorcnt != 0;
        break;
      default:
        break;
      }
      if (issued)
        register_counter_slot(*rs, instruction, std::span(&obligation, 1),
                              obligation.completion_class(), is_load);
    }
  };

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
          arch == ROCJITSU_CODE_ARCH_RDNA2 && d.gds_m0 && d.address_range_violation_lanes &&
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
    const uint64_t execMask = d.exec_mask;
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
    if (amdgpu::cdna1_ds_atomic_nullified(d, arch, wf.vgpr_alloc())) {
      if (!obligations.empty())
        rs->registerEvent(instruction, type, {}, 0, 0xf, obligations, memoryOrder);
      return;
    }
    if (execMask == 0) {
      if (arch == ROCJITSU_CODE_ARCH_CDNA5)
        record_exec_zero_slot(d.is_load, d.nullified_result);
      else if (arch == ROCJITSU_CODE_ARCH_RDNA4 && (wf.exec() == 0 || d.nullified_result))
        for (const auto obligation : obligations)
          rs->recordOpaqueCounterCredit(obligation.wait_counter_type(),
                                        obligation.completion_class(),
                                        obligation.counter_increment());
      else if (arch == ROCJITSU_CODE_ARCH_RDNA2 && !obligations.empty())
        rs->registerEvent(instruction, type, {}, 0, 0xf, obligations, memoryOrder);
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
    const uint64_t localResult = split_route_results && mixedFlat ? execMask & sharedLanes : execMask;
    // Legacy FLAT's LDS bytes retain DS issue order, but its VGPR result can
    // pass a native DS result. Keep those two ordering claims separate.
    const VgprWriteOrder localWriteOrder = split_route_results
                                               ? VgprWriteOrder{
                                                     routed &&
                                                             !amdgpu::has_ordered_flat_lds_writeback(arch)
                                                         ? MemoryOrderClass::UNORDERED
                                                         : memoryOrder}
                                               : writeOrder;
    if (d.is_load)
      for (uint32_t reg : registers)
        rs->checkVgprWrite(static_cast<int>(reg), localResult, byte_mask, localWriteOrder);
    if (obligations.empty())
      return;
    if (split_route_results && mixedFlat) {
      // A mixed FLAT returns each lane on exactly one queue. An ordinary wait
      // can release those lanes while the other route still has work in flight.
      // An active private-OOB load lane can write zero without making a
      // request; its result still belongs to the non-LDS route. Store source
      // lifetimes follow requests because an OOB store reads no payload.
      const uint64_t globalEventLanes =
          d.is_load ? access.active_lane_mask & ~sharedLanes : globalRequests;
      const auto global = decoded->for_flat_memory_domains(true, false);
      const auto order =
          vmem_vgpr_write_order(memoryOrderForObligations(global.counter_obligations()), arch);
      if (d.is_load)
        for (uint32_t reg : registers)
          rs->checkVgprWrite(static_cast<int>(reg), globalEventLanes, byte_mask, order);
      rs->registerEvent(
          instruction,
          d.is_load ? MemoryEventType::GLOBAL_TO_VGPR : MemoryEventType::VGPR_TO_GLOBAL, registers,
          globalEventLanes, byte_mask, global.counter_obligations(), order);
    }
    if (defer_lds)
      s->mixed_lds->intervals = std::exchange(intervals, IntervalSet{});
    const auto event = rs->registerEventWithIntervals(
        instruction, type, std::move(registers), localResult, byte_mask, std::move(intervals),
        obligations, memoryOrder, /*lastRegisterByteMask=*/0, ldsAccess,
        /*feedback=*/{}, defer_lds, secondAccess ? VgprWriteOrder{memoryOrder} : localWriteOrder);
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
    if (d.exec_mask == 0) {
      // An async copy with no active lanes still issues and counts, even while
      // wave EXEC is nonzero, so it holds its slot on every counter it names,
      // and later partial waits complete the copies issued before it.
      const bool asyncCopy =
          (d.lds_src || d.lds_dst) && std::ranges::any_of(obligations, [](const auto &obligation) {
            return obligation.wait_counter_type() == amdgpu::WaitCounterType::ASYNCCNT;
          });
      if (asyncCopy && wf.exec() != 0)
        register_counter_slot(*rs, instruction, obligations, memoryOrder, d.is_load);
      else if (amdgpu::has_gfx11_plus_image_operands(wf.cu().arch()) &&
               inst.mnemonic().starts_with("image_")) {
        // The before-execute hook already admitted an issue that had active wave EXEC,
        // even if the image destination later suppressed every result lane.
        const bool needsAdmission =
            issue.exec_masked && wf.exec() == 0 && supports_counter_capacity(wf.cu().arch());
        for (const auto &obligation : obligations) {
          const auto counter = obligation.wait_counter_type();
          if (!rs->hasPendingCounter(counter)) {
            if (arch == ROCJITSU_CODE_ARCH_RDNA4 && (wf.exec() == 0 || d.nullified_result))
              rs->recordOpaqueCounterCredit(counter, obligation.completion_class(),
                                            obligation.counter_increment());
            continue;
          }
          if (needsAdmission &&
              rs->prepareForCounterIncrement(counter, obligation.counter_increment()).failed()) {
            report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
            return;
          }
          register_counter_slot(*rs, instruction, std::span(&obligation, 1),
                                obligation.completion_class(), d.is_load);
        }
      } else {
        if (exec_zero_memory_issue_counts(arch))
          record_exec_zero_slot(d.is_load, d.nullified_result);
        else if (arch == ROCJITSU_CODE_ARCH_RDNA4 && (wf.exec() == 0 || d.nullified_result))
          for (const auto obligation : obligations)
            rs->recordOpaqueCounterCredit(obligation.wait_counter_type(),
                                          obligation.completion_class(),
                                          obligation.counter_increment());
      }
      return;
    }
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
  if (std::ranges::any_of(s->partial_valu_destinations, [&](const auto &destination) {
        return destination.first == logical_reg && destination.second == byte_mask;
      }))
    s->race_state->checkVgprReadLanes(static_cast<int>(logical_reg), lane_mask,
                                      static_cast<uint8_t>(0xf & ~byte_mask));
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
  auto *s = get_state(wf);
  assert(s && s->race_state);
  s->partial_valu_destinations.clear();
  util::FailureOr<InstructionSnapshot> instruction = capture_instruction(pc, inst, wf);
  if (instruction.failed())
    return;
  s->race_state->finishRoutedIssue();
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
    if (s->race_state->prepareForCounterIncrement(counter).failed()) {
      report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
      return;
    }
    s->image_query.emplace(amdgpu::vmem_read_type(arch, name), counter);
  }
  if (supports_counter_capacity(arch) ||
      (name.starts_with("flat_") && amdgpu::flat_issues_both_counter_domains(arch))) {
    // An active generic FLAT occupies both issue positions before reading
    // its address. Zero-EXEC admission still follows each target's skip rule.
    if (inst.is_memory_op() &&
        (!name.starts_with("flat_") || amdgpu::flat_issues_both_counter_domains(arch))) {
      const auto *info = inst.amdgpu_memory_issue_info();
      assert(info && "memory instruction reached the race detector without issue metadata");
      util::Result admitted = util::Result::success();
      if (info && !ignored_memory_issue(inst, wf)) {
        if (!info->exec_masked || wf.exec() != 0)
          admitted = s->race_state->prepareForMemoryIssue(*info);
        else if (exec_zero_memory_issue_counts(wf.cu().arch()))
          admitted = s->race_state->prepareForMemoryIssue(
              exec_zero_counted_obligations(*info, inst.mnemonic(),
                                            wf.wait_counters().dscnt != 0, arch));
      }
      if (admitted.failed()) {
        report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
        return;
      }
    }

    // Wait-counter producers without memory issue metadata, such as cache
    // operations and the LDS lane permutes, apply backpressure here and hold
    // their ordered slot (see counter_only_slot()).
    if (inst.is_memory_wait_producer() && !inst.is_memory_op()) {
      if (const auto slot = counter_only_slot(inst.mnemonic(), wf.cu().arch())) {
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
  }

  // Messages are not memory instructions, so they remain outside memory
  // issue metadata. They count on LGKMCNT before GFX12 and on KMcnt on
  // gfx1250 (CDNA5 ISA 5.7.1.5), where S_SENDMSG_RTN_B* adds two tokens, one
  // for the message and one for its returned data (CDNA5 ISA 5.4).
  const std::string_view mnemonic = inst.mnemonic();
  if (mnemonic.starts_with("s_sendmsg") &&
      (arch == ROCJITSU_CODE_ARCH_CDNA5 || counterCapacitiesForArch(arch).lgkmcnt != 0)) {
    const bool cdna5 = wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5;
    const auto counter = cdna5 ? amdgpu::WaitCounterType::KMCNT : amdgpu::WaitCounterType::LGKMCNT;
    const uint8_t increment = cdna5 && mnemonic.starts_with("s_sendmsg_rtn") ? 2 : 1;
    if (s->race_state->prepareForCounterIncrement(counter, increment).failed()) {
      report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
      return;
    }
  }
  s->trace.push(instruction.value());
  if ((arch == ROCJITSU_CODE_ARCH_RDNA3 || arch == ROCJITSU_CODE_ARCH_RDNA3_5 ||
       arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_CDNA5) &&
      (inst.flags() & VALU)) {
    const amdgpu::RegisterAccess registers(wf);
    for (int i = 0; i < inst.num_dst_operands(); ++i) {
      const Operand *operand = inst.dst_operand(i);
      if (!operand || operand->size_bits() != 16)
        continue;
      const uint8_t bytes = operand->register_byte_mask();
      if (bytes != 0x3 && bytes != 0xc)
        continue;
      if (auto reg = registers.destination_register(*operand);
          reg && reg->cls == RegClass::VGPR && reg->width == 1)
        s->partial_valu_destinations.emplace_back(reg->index, bytes);
    }
  }
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
  const auto fields = amdgpu::RegisterAccess::explicit_wait_fields(inst, wf.cu().arch(), &wf);
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
