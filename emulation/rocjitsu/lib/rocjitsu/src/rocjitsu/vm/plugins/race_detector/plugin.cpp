// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/plugin.h"

#include "rocjitsu/isa/arch/amdgpu/cdna3/sgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/addr_calc_buffer.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/indexed_ds_bounds.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/lds_allocation.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/log.h"

#include "rocjitsu/vm/plugins/memory_access_observation.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include "rocjitsu/vm/plugins/tensor_dma_issue_observation.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <format>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::plugins::race_detector {

namespace {

void report_plugin_failure(const amdgpu::Wavefront &wf, const char *detail) {
  if (!wf.instruction_execution_failed())
    wf.report_instruction_execution_error(amdgpu::InstructionExecutionError::PluginFailure, detail);
}

// LDS accesses sharing all their completion classes are ordered as a whole.
// Each obligation still has its own FIFO class for partial waits.
MemoryOrderClass
memoryOrderForObligations(std::span<const amdgpu::MemoryCounterObligation> obligations) {
  if (obligations.empty())
    return MemoryOrderClass::UNORDERED;
  const auto order = obligations.front().completion_class();
  for (const auto obligation : obligations)
    if (obligation.completion_class() != order)
      return MemoryOrderClass::UNORDERED;
  return order;
}

std::string_view register_space(RegClass cls) {
  switch (cls) {
  case RegClass::VGPR:
    return "VGPR";
  case RegClass::SGPR:
    return "SGPR";
  case RegClass::TTMP:
    return "TTMP";
  case RegClass::ACC_VGPR:
    return "ACC_VGPR";
  case RegClass::EXEC:
    return "EXEC";
  case RegClass::VCC:
    return "VCC";
  case RegClass::SCC:
    return "SCC";
  case RegClass::M0:
    return "M0";
  case RegClass::FLAT_SCRATCH:
    return "FLAT_SCRATCH";
  case RegClass::PC:
    return "PC";
  }
  return "UNKNOWN";
}

util::FailureOr<InstructionSnapshot> capture_instruction(uint64_t pc, const Instruction &inst,
                                                         const amdgpu::Wavefront &wf) {
  static_assert(InstructionSnapshot::kMaxWords == amdgpu::DecodedInstructionCache::kNumWords);
  if (!inst.raw_encoding() || inst.size() <= 0 || inst.size() % sizeof(uint32_t) != 0 ||
      static_cast<size_t>(inst.size()) > InstructionSnapshot::kMaxWords * sizeof(uint32_t)) {
    report_plugin_failure(wf, "race detector instruction has no valid encoding");
    return util::Result::failure();
  }
  return InstructionSnapshot{pc, std::span(inst.raw_encoding(), inst.size() / sizeof(uint32_t))};
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
bool exec_zero_memory_issue_counts(rj_code_arch_t arch) { return arch == ROCJITSU_CODE_ARCH_CDNA5; }

/// A gfx1250 VBUFFER instruction whose resource descriptor TYPE is not a buffer
/// is ignored (CDNA5 ISA 9.6), so it applies no backpressure and holds no
/// counter slot. If hardware still counted it, leaving it out only keeps older
/// operations pending longer.
bool ignored_memory_issue(const Instruction &inst, const amdgpu::Wavefront &wf) {
  // Every gfx1250 VBUFFER instruction has conditional memory lanes, so others
  // skip decoding their register modifiers.
  if (wf.cu().arch() != ROCJITSU_CODE_ARCH_CDNA5 || !(inst.flags() & CONDITIONAL_MEMORY_LANES))
    return false;
  amdgpu::RegisterModifiers modifiers;
  inst.amdgpu_register_modifiers(modifiers);
  return amdgpu::addr_calc::cdna5_ignores_buffer_resource(modifiers, wf);
}

/// The counter slot a gfx1250 wait-counter producer without memory issue
/// metadata takes when it issues:
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
std::optional<amdgpu::MemoryCounterObligation> counter_only_slot(std::string_view mnemonic) {
  using amdgpu::MemoryCompletionClass;
  using amdgpu::WaitCounterType;
  if (mnemonic == "global_inv")
    return amdgpu::MemoryCounterObligation{WaitCounterType::LOADCNT, MemoryCompletionClass::VMEM};
  if (mnemonic == "global_wb" || mnemonic == "global_wbinv")
    return amdgpu::MemoryCounterObligation{WaitCounterType::STORECNT, MemoryCompletionClass::VMEM};
  if (mnemonic == "ds_permute_b32" || mnemonic == "ds_bpermute_b32" ||
      mnemonic == "ds_bpermute_fi_b32" || mnemonic == "ds_swizzle_b32" || mnemonic == "ds_nop")
    return amdgpu::MemoryCounterObligation{WaitCounterType::DSCNT, MemoryCompletionClass::LDS};
  return std::nullopt;
}

/// On gfx1250 (CDNA5) "Every Flat instruction increments both DScnt and either
/// LOADcnt or STOREcnt when the instruction issues" (CDNA5 ISA 5.7.1.3),
/// whichever memory its addresses resolve to.
bool flat_counts_on_both_domains(rj_code_arch_t arch) { return arch == ROCJITSU_CODE_ARCH_CDNA5; }

/// Record each counter slot that a gfx1250 FLAT instruction takes in a memory
/// domain its addresses did not use, as an event with no registers and no LDS
/// bytes. The slot completes in order with that domain's other operations: the
/// LDS portion of a FLAT stays in order with LDS instructions, and the VMEM
/// portion with other VMEM operations (CDNA5 ISA 5.7.1.3). Recording it lets
/// later capacity checks and partial waits count it, as hardware does.
/// `routed` holds the obligations of the domains the addresses used; its
/// event carries the instruction's data dependencies.
void register_unused_flat_domains(WaveRaceState &race, const InstructionSnapshot &instruction,
                                  const amdgpu::MemoryIssueInfo &decoded,
                                  const amdgpu::MemoryIssueInfo &routed) {
  for (const auto obligation : decoded.counter_obligations()) {
    const bool used = std::ranges::any_of(routed.counter_obligations(), [&](const auto routed_one) {
      return routed_one.wait_counter_type() == obligation.wait_counter_type();
    });
    if (used)
      continue;
    const std::array slot{obligation};
    race.registerCounterEvent(instruction, slot);
  }
}

} // namespace

util::FailureOr<MarkedInstruction> findConflict(const RaceViolation &v, RaceDetector &detector) {
  EventId event_id = v.conflictingEvent;
  if (!detector.events().contains(event_id))
    return util::Result::failure();
  return MarkedInstruction{detector.events().instruction(event_id),
                           detector.events().waveId(event_id).value, -1};
}

util::FailureOr<std::string> formatTrace(
    const RingBuffer<InstructionSnapshot, 256> &trace,
    const std::function<util::FailureOr<std::string>(const InstructionSnapshot &)> &disassemble,
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
    util::FailureOr<std::string> text = disassemble(conflict->instruction);
    if (text.failed())
      return util::Result::failure();
    if (!text.value().empty())
      oss << text.value() << "  ";
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

  auto emit = [&](size_t i) -> util::Result {
    const auto &instruction = trace[i];
    bool is_conflict = conflict && instruction == conflict->instruction;
    bool is_read = instruction == read.instruction;
    oss << ((is_conflict || is_read) ? "  ==>  " : "       ");
    util::FailureOr<std::string> text = disassemble(instruction);
    if (text.failed())
      return util::Result::failure();
    oss << "0x" << std::hex << instruction.pc() << std::dec << "  " << text.value();
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
    return util::Result::success();
  };

  if (span <= MAX_PRINT_SIZE) {
    for (size_t i = first; i < last; ++i)
      if (emit(i).failed())
        return util::Result::failure();
  } else {
    size_t half = MAX_PRINT_SIZE / 2;
    for (size_t i = first; i < first + half; ++i)
      if (emit(i).failed())
        return util::Result::failure();
    oss << "       ... " << (span - MAX_PRINT_SIZE) << " instructions elided ...\n";
    for (size_t i = last - half; i < last; ++i)
      if (emit(i).failed())
        return util::Result::failure();
  }
  return oss.str();
}

RaceDetectorPlugin::RaceDetectorPlugin(const char * /*config_json*/) : ExecutionPlugin("race") {}

RaceDetectorPlugin::~RaceDetectorPlugin() { sink().write(getSummary()); }

std::string RaceDetectorPlugin::getSummary() const {
  std::lock_guard lock(report_mutex_);
  const char *banner = "\n========================================\n"
                       " ROCJITSU RACE DETECTION SUMMARY\n"
                       "========================================\n";
  std::string summary = banner;
  summary += reported_races_ ? std::format("  {} race(s) detected\n", reported_races_)
                             : "  No races detected.\n";
  summary += std::format("  {} repeated register finding(s) suppressed (site-level reporting)\n",
                         repeated_register_races_);
  return summary + "========================================\n";
}

void RaceDetectorPlugin::emit_report(uint32_t dispatch, const ReportLocation &location,
                                     std::string_view details,
                                     const amdgpu::RegisterHazardObservation *hazard) {
  const auto iter = dispatch_kernel_names_.find(dispatch);
  const KernelNames names = iter == dispatch_kernel_names_.end()
                                ? KernelNames{kUnknownKernelIdentity, kUnknownKernelIdentity}
                                : iter->second;
  std::string fields =
      std::format(" producer_pc={:#x} consumer_pc={:#x} checker={}", location.producer_pc,
                  location.consumer_pc, hazard ? "core" : "plugin");
  if (hazard)
    fields +=
        std::format(" counter={} required_wait={} width={} lanes={:#x} bytes={:#x}",
                    amdgpu::wait_counter_name(location.arch, hazard->counter),
                    hazard->required_wait, hazard->reg.width, hazard->lane_mask, hazard->byte_mask);
  std::string target(location.space);
  if (location.space == "LDS")
    target += std::format(" byte {}", location.index);
  else if (location.space == "VGPR" || location.space == "SGPR" || location.space == "TTMP" ||
           location.space == "ACC_VGPR") {
    const char *prefix = location.space == "VGPR"   ? "v"
                         : location.space == "SGPR" ? "s"
                         : location.space == "TTMP" ? "ttmp"
                                                    : "acc";
    target += std::format(" {}{}", prefix, location.index);
  }
  const auto &wg = location.workgroup;
  sink().write(
      std::format("RACE kernel={} symbol={} dispatch={} type={} access={} reg={} wave={} lane={} "
                  "wg={},{},{} conflict=unknown{}\n"
                  "Race on {} [workgroup ({}, {}, {}), wave {}{}]\n{}END_RACE\n",
                  names.name, names.symbol, dispatch, location.space,
                  location.write ? "write" : "read", location.index, location.wave, location.lane,
                  wg.x, wg.y, wg.z, fields, target, wg.x, wg.y, wg.z, location.wave,
                  location.lane >= 0 ? std::format(", lane {}", location.lane) : "", details));
  ++reported_races_;
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

void RaceDetectorPlugin::onAmdgpuRegisterHazard(const amdgpu::RegisterHazardObservation &hazard,
                                                const amdgpu::Wavefront &wf) {
  auto *state = get_state(wf);
  assert(state && state->race_state);
  const uint32_t dispatch = wf.dispatch_id();
  const RegisterSite site{dispatch,         hazard.consumer_pc, hazard.producer_pc, hazard.reg.cls,
                          hazard.reg.index, hazard.write,       hazard.counter};
  {
    std::lock_guard lock(report_mutex_);
    if (observed_register_races_.contains(site)) {
      ++repeated_register_races_;
      return;
    }
  }

  const int wave = state->race_state->getWaveId().value;
  const bool vector = hazard.reg.cls == RegClass::VGPR || hazard.reg.cls == RegClass::ACC_VGPR;
  const int lane = vector && hazard.lane_mask ? std::countr_zero(hazard.lane_mask) : -1;
  util::StringDiagnostic diagnostic;
  const auto disassemble = [&wf, &diagnostic](const InstructionSnapshot &instruction) {
    return wf.disassemble_instruction(instruction.encoding(), instruction.pc(),
                                      diagnostic.emitter());
  };
  auto details = [&]() -> util::FailureOr<std::string> {
    if (hazard.producer_instruction && hazard.consumer_instruction) {
      // The consumer's instruction notification may not have run yet. Include
      // its owned encoding in a copy of the trace only when emitting a report.
      auto trace = state->trace;
      trace.push(*hazard.consumer_instruction);
      return formatTrace(trace, disassemble,
                         MarkedInstruction{*hazard.producer_instruction, wave, -1},
                         MarkedInstruction{*hazard.consumer_instruction, wave, lane});
    }
    // Do not recover historical bytes by reading the code mapping at a saved PC.
    const auto describe =
        [&](const std::optional<InstructionSnapshot> &instruction) -> util::FailureOr<std::string> {
      if (!instruction)
        return std::string("(encoding unavailable)");
      return disassemble(*instruction);
    };
    auto producer = describe(hazard.producer_instruction);
    if (producer.failed())
      return util::Result::failure();
    auto consumer = describe(hazard.consumer_instruction);
    if (consumer.failed())
      return util::Result::failure();
    return std::format("  ==>  {:#x}  {}  ; <-- producer, wave {}\n"
                       "  ==>  {:#x}  {}  ; <-- consumer, wave {}\n",
                       hazard.producer_pc, producer.value(), wave, hazard.consumer_pc,
                       consumer.value(), wave);
  }();
  if (details.failed()) {
    report_plugin_failure(wf, "race detector could not disassemble the instruction trace");
    std::lock_guard lock(report_mutex_);
    sink().write(diagnostic.message());
    return;
  }
  const auto &wg = wf.wg_coord();
  const ReportLocation location{register_space(hazard.reg.cls),
                                hazard.reg.index,
                                wave,
                                lane,
                                hazard.write,
                                Dim3d(wg[0], wg[1], wg[2]),
                                hazard.producer_pc,
                                hazard.consumer_pc,
                                wf.cu().arch()};
  std::lock_guard lock(report_mutex_);
  if (!observed_register_races_.insert(site).second) {
    ++repeated_register_races_;
    return;
  }
  emit_report(dispatch, location, details.value(), &hazard);
}

void RaceDetectorPlugin::onAmdgpuWorkgroupDispatched(uint32_t dispatch_id, uint32_t wg_id,
                                                     std::span<amdgpu::Wavefront *> wavefronts) {
  uint32_t num_waves = static_cast<uint32_t>(wavefronts.size());
  auto handler = [this,
                  wf_ptrs = std::vector<amdgpu::Wavefront *>(wavefronts.begin(), wavefronts.end()),
                  dispatch_id](RaceViolation v) {
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
      if (observed_races_.count({dispatch_id, pc}))
        return;
    }

    auto *detector = ws->race_state->getDetector();
    util::FailureOr<MarkedInstruction> maybe_conflict = findConflict(v, *detector);
    if (maybe_conflict.failed()) {
      report_plugin_failure(*wf, "race violation references an unavailable conflicting event");
      return;
    }
    const MarkedInstruction &conflict = maybe_conflict.value();

    MarkedInstruction access_mark{*current, v.wave, v.lane};
    util::StringDiagnostic diagnostic;
    util::FailureOr<std::string> trace = formatTrace(
        ws->trace,
        [wf, &diagnostic](const InstructionSnapshot &instruction) {
          return wf->disassemble_instruction(instruction.encoding(), instruction.pc(),
                                             diagnostic.emitter());
        },
        conflict, access_mark);
    if (trace.failed()) {
      report_plugin_failure(*wf, "race detector could not disassemble the instruction trace");
      std::lock_guard<std::mutex> lock(report_mutex_);
      sink().write(diagnostic.message());
      return;
    }

    {
      std::lock_guard<std::mutex> lock(report_mutex_);
      bool is_new = !observed_races_.count({dispatch_id, pc}) &&
                    !observed_races_.count({dispatch_id, conflict.instruction.pc()});
      observed_races_.emplace(dispatch_id, pc);
      if (is_new) {
        observed_races_.emplace(dispatch_id, conflict.instruction.pc());
        emit_report(dispatch_id,
                    {"LDS", v.address, v.wave, v.lane, v.isWrite, v.workgroupId,
                     conflict.instruction.pc(), pc, wf->cu().arch()},
                    trace.value());
      }
    }
  };

  auto detector = std::make_shared<RaceDetector>(
      static_cast<int>(num_waves), Dim3d(static_cast<int>(wg_id)), std::move(handler),
      counterCapacitiesForArch(wavefronts.front()->cu().arch()),
      wavefronts.front()->lds().size_bytes());

  for (uint32_t w = 0; w < num_waves; ++w) {
    auto state = std::make_unique<RaceWavefrontState>();
    state->race_state =
        std::shared_ptr<WaveRaceState>(detector, &detector->getWaveRaceState(static_cast<int>(w)));
    wavefronts[w]->set_plugin_state(slot_index(), std::move(state));
  }
}

void RaceDetectorPlugin::onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
  if (RaceWavefrontState *state = get_state(wf); state && state->race_state) {
    // Normal termination drains the wave's memory operations before this hook.
    // Model s_endpgm's implicit wait, retaining LDS history until a subsequent
    // workgroup barrier supplies ordering with the surviving waves.
    const amdgpu::WaitcntModel model = amdgpu::waitcnt_model(wf.cu().arch()).value();
    for (size_t index = 0; index < amdgpu::kWaitCounterCount; ++index) {
      const auto kind = static_cast<amdgpu::WaitCounterKind>(index);
      if (const std::optional<amdgpu::WaitCounterType> counter =
              amdgpu::memory_wait_counter_type(kind, model))
        state->race_state->applyWaitCounter(*counter, 0);
    }
  }
  release_wave_state(wf);
}

void RaceDetectorPlugin::onAmdgpuWavefrontCancelled(amdgpu::Wavefront &wf) {
  release_wave_state(wf);
}

void RaceDetectorPlugin::onAmdgpuMemoryAccessRouted(const amdgpu::MemoryAccessObservation &access,
                                                    const Instruction &inst,
                                                    amdgpu::Wavefront &wf) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  auto *rs = s->race_state.get();
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
  const auto *issue = amdgpu::resolved_memory_issue_info(inst);
  assert(issue && "memory instruction reached the race detector without issue metadata");
  if (!issue)
    return;
  const auto obligations = issue->counter_obligations();
  const MemoryOrderClass memoryOrder = memoryOrderForObligations(obligations);
  // gfx1250 FLAT counts on both domains and was admitted before execution.
  // Earlier targets apply capacity only after routing selects a domain.
  const bool flat =
      access.decoded_space == amdgpu::DecodedMemorySpace::FLAT &&
      (access.route == amdgpu::MemoryRoute::GLOBAL || access.route == amdgpu::MemoryRoute::LOCAL);
  const bool flatOnBothDomains = flat && flat_counts_on_both_domains(wf.cu().arch());
  if (flat && !flatOnBothDomains && access.request_lane_mask != 0 &&
      supports_counter_capacity(wf.cu().arch()) && rs->prepareForMemoryIssue(*issue).failed()) {
    report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
    return;
  }
  if (flatOnBothDomains)
    register_unused_flat_domains(*rs, instruction, *inst.amdgpu_memory_issue_info(), *issue);

  // A gfx1250 instruction issued with wave EXEC zero holds its counter slot. An
  // access whose lanes are all cleared while EXEC is active records none, and
  // neither does an ignored buffer instruction (see ignored_memory_issue()).
  const auto record_exec_zero_slot = [&]() {
    if (!exec_zero_memory_issue_counts(wf.cu().arch()) || wf.exec() != 0 ||
        ignored_memory_issue(inst, wf))
      return;
    const auto counted = issue->for_exec_zero(wf.cu().arch(), inst.mnemonic());
    if (!counted.empty())
      rs->registerCounterEvent(instruction, counted.counter_obligations());
  };

  const uint64_t sharedLanes = access.flat_local_lane_mask | access.flat_dds_lane_mask;
  const bool mixedFlat = inst.data()->tag() == amdgpu::GLOBAL_MEM && sharedLanes != 0;
  if (inst.data()->tag() == amdgpu::LOCAL_MEM || mixedFlat) {
    auto &d = *inst.data_as<amdgpu::VectorMemState>();
    const uint64_t execMask = d.exec_mask;
    // DDS and allocation-OOB lanes participate in the instruction's
    // dependencies, but do not access bytes in the ordinary LDS allocation.
    const uint64_t ldsMask = (mixedFlat ? execMask & access.flat_local_lane_mask : execMask) &
                             ~(access.flat_dds_lane_mask | access.flat_local_oob_lane_mask);
    if (execMask == 0) {
      record_exec_zero_slot();
      return;
    }
    const bool readModifyWrite = (access.decoded_space == amdgpu::DecodedMemorySpace::FLAT ||
                                  access.decoded_space == amdgpu::DecodedMemorySpace::LOCAL) &&
                                 d.atomic_op != amdgpu::AtomicOp::NONE;
    const auto ldsAccess = readModifyWrite ? LdsAccess::ReadWrite
                           : d.is_load     ? LdsAccess::Read
                                           : LdsAccess::Write;
    uint8_t alignedAtomicBytes = readModifyWrite ? d.elem_size : 0;
    const uint32_t perLaneBytes = d.num_elems * d.elem_size;

    const auto bounds =
        amdgpu::indexed_ds_access_bounds(d, wf.cu().arch(), wf.lds_base(), wf.lds_size());
    const auto atomicWidth = [&](uint64_t address, uint64_t bytes) -> uint8_t {
      return readModifyWrite && bytes == d.elem_size && address % d.elem_size == 0 ? d.elem_size
                                                                                   : 0;
    };
    // A DS read2/write2 accesses two addresses per lane; other instructions one.
    const std::array<const std::array<uint64_t, 64> *, 2> addressSets{&d.per_lane_addr,
                                                                      &d.ds2_per_lane_addr};
    const size_t numAddressSets = d.ds2_active ? 2 : 1;
    // Bytes each lane accesses within the LDS backing and, for FLAT, the
    // workgroup allocation. Zero for lanes that access no LDS bytes.
    std::array<std::array<uint32_t, 64>, 2> laneBytes{};
    // Appending one address set at a time lets neighboring lanes merge into
    // the previous interval before the final sort.
    IntervalSet intervals;
    for (size_t set = 0; set < numAddressSets; ++set) {
      for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
        if (!(ldsMask & (1ULL << lane)))
          continue;
        const uint64_t address = (*addressSets[set])[lane];
        if (bounds && !bounds->contains(address))
          continue;
        uint64_t bytes = detector->ldsBytesInRange(address, perLaneBytes, d.elem_size);
        if (access.flat_local_allocation_end && d.elem_size != 0)
          bytes = amdgpu::whole_element_bytes_in_range(address, bytes, d.elem_size,
                                                       *access.flat_local_allocation_end);
        if (bytes == 0)
          continue;
        // Interval endpoints use signed indices even when the configured LDS
        // backing is larger. Keep counter dependencies below.
        if (address + bytes > std::numeric_limits<int>::max())
          continue;
        if (alignedAtomicBytes && !atomicWidth(address, bytes))
          alignedAtomicBytes = 0;
        laneBytes[set][lane] = static_cast<uint32_t>(bytes);
        intervals.append(static_cast<int>(address), static_cast<int>(address + bytes));
      }
    }
    intervals.finalize();
    const bool checkReads =
        (d.is_load || readModifyWrite) &&
        detector->mayConflict(intervals, waveId, /*isWrite=*/false, memoryOrder);
    const bool checkWrites =
        (!d.is_load || readModifyWrite) &&
        detector->mayConflict(intervals, waveId, /*isWrite=*/true, memoryOrder);
    if (checkReads || checkWrites) {
      for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
        for (size_t set = 0; set < numAddressSets; ++set) {
          const int bytes = static_cast<int>(laneBytes[set][lane]);
          if (bytes == 0)
            continue;
          const int address = static_cast<int>((*addressSets[set])[lane]);
          const auto atomicBytes = atomicWidth(address, bytes);
          if (checkReads)
            detector->validateRead(address, waveId, static_cast<int>(lane), bytes, memoryOrder,
                                   atomicBytes);
          if (checkWrites)
            detector->validateWrite(address, waveId, static_cast<int>(lane), bytes, memoryOrder,
                                    atomicBytes);
        }
      }
    }
    if (obligations.empty())
      return;
    rs->registerEventWithIntervals(instruction, ldsAccess, std::move(intervals), obligations,
                                   memoryOrder, alignedAtomicBytes);
    // A mixed instruction has one counter entry and only the ordinary LDS
    // lanes' intervals. Do not also register a global event.
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
        rs->registerCounterEvent(instruction, obligations);
      else
        record_exec_zero_slot();
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
          if (!obligations.empty())
            rs->registerCounterEvent(instruction, obligations);
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
      rs->registerLdsEvent(instruction, LdsAccess::Write, validLaneMask, wf.wf_size(),
                           std::span<const uint32_t>(ldsAddrs, wf.wf_size()), perLaneBytes,
                           obligations, memoryOrder);
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
      std::vector<amdgpu::LdsByteRange> ranges;
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
      rs->registerLdsRangeEvent(instruction, LdsAccess::Read, ranges, obligations, memoryOrder);
    } else if (!obligations.empty()) {
      rs->registerCounterEvent(instruction, obligations);
    }
  }

  if (inst.data()->tag() == amdgpu::SCALAR_MEM && !obligations.empty())
    rs->registerCounterEvent(instruction, obligations);
}

void RaceDetectorPlugin::onAmdgpuLdsDirectRead(const amdgpu::Wavefront &wf, uint32_t address,
                                               uint32_t bytes, uint32_t lane) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  // The uniform VALU source is broadcast, even if the backend materializes it
  // once per lane. A successful instruction publishes one completed interval.
  if (!s->lds_operand_read)
    s->lds_operand_read.emplace(address, bytes, lane);
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
  auto *rs = s->race_state.get();
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
  if (supports_counter_capacity(wf.cu().arch()) &&
      rs->prepareForCounterIncrement(amdgpu::WaitCounterType::TENSORCNT).failed()) {
    report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
    return;
  }
  // An observed range can end mid-element: an element that straddles the end
  // of the wave's LDS allocation transfers only its leading bytes. Validate the
  // ranges at byte granularity, as registerLdsRangeEvent registers them;
  // rounding them to whole elements would drop those bytes.
  constexpr int kByteGranularity = 1;

  // A transfer has no lanes; its violations report lane -1.
  for (const auto &range : issue.lds_ranges) {
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
  rs->registerLdsRangeEvent(instruction, issue.is_load ? LdsAccess::Write : LdsAccess::Read,
                            issue.lds_ranges, obligations, memoryOrder);
}

void RaceDetectorPlugin::onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                                          amdgpu::Wavefront &wf,
                                                          std::span<const uint32_t>) {
  util::FailureOr<InstructionSnapshot> instruction = capture_instruction(pc, inst, wf);
  if (instruction.failed())
    return;
  auto *s = get_state(wf);
  assert(s && s->race_state);
  if (supports_counter_capacity(wf.cu().arch())) {
    // On the pre-GFX12 capacity targets, FLAT's address-dependent admission is
    // applied after routing: before its operands are read, neither memory
    // domain is guaranteed to be used. On gfx1250 every FLAT counts on both
    // domains, so it is admitted here like any other memory instruction.
    if (inst.is_memory_op() &&
        (!inst.mnemonic().starts_with("flat_") || flat_counts_on_both_domains(wf.cu().arch()))) {
      const auto *info = inst.amdgpu_memory_issue_info();
      assert(info && "memory instruction reached the race detector without issue metadata");
      util::Result admitted = util::Result::success();
      if (info && !ignored_memory_issue(inst, wf)) {
        if (!info->exec_masked || wf.exec() != 0)
          admitted = s->race_state->prepareForMemoryIssue(*info);
        else if (exec_zero_memory_issue_counts(wf.cu().arch()))
          admitted = s->race_state->prepareForMemoryIssue(
              info->for_exec_zero(wf.cu().arch(), inst.mnemonic()));
      }
      if (admitted.failed()) {
        report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
        return;
      }
    }

    // Wait-counter producers without memory issue metadata, such as GLOBAL_INV
    // and the LDS lane permutes, apply their gfx1250 backpressure here and then
    // hold their slot (see counter_only_slot()).
    if (wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5 && inst.is_memory_wait_producer() &&
        !inst.is_memory_op()) {
      if (const auto slot = counter_only_slot(inst.mnemonic())) {
        const std::span<const amdgpu::MemoryCounterObligation> obligations{&*slot, 1};
        if (s->race_state->prepareForCounterIncrement(slot->wait_counter_type()).failed()) {
          report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
          return;
        }
        s->race_state->registerCounterEvent(instruction.value(), obligations);
      }
    }

    // Messages are not memory instructions, so they remain outside memory
    // issue metadata. They count on LGKMCNT before GFX12 and on KMcnt on
    // gfx1250 (CDNA5 ISA 5.7.1.5), where S_SENDMSG_RTN_B* adds two tokens, one
    // for the message and one for its returned data (CDNA5 ISA 5.4).
    const std::string_view mnemonic = inst.mnemonic();
    if (mnemonic.starts_with("s_sendmsg")) {
      const bool cdna5 = wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA5;
      const auto counter =
          cdna5 ? amdgpu::WaitCounterType::KMCNT : amdgpu::WaitCounterType::LGKMCNT;
      const uint8_t increment = cdna5 && mnemonic.starts_with("s_sendmsg_rtn") ? 2 : 1;
      if (s->race_state->prepareForCounterIncrement(counter, increment).failed()) {
        report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
        return;
      }
    }
  }
  s->lds_operand_read.reset();
  s->trace.push(instruction.value());
}

void RaceDetectorPlugin::onAmdgpuAsyncInstructionIssued(uint64_t pc, const Instruction &inst,
                                                        amdgpu::Wavefront &wf) {
  util::FailureOr<InstructionSnapshot> instruction = capture_instruction(pc, inst, wf);
  if (instruction.failed())
    return;
  auto *s = get_state(wf);
  assert(s && s->race_state);
  s->trace.push(instruction.value());
}

void RaceDetectorPlugin::onAmdgpuAfterExecuteInstruction(uint64_t /*pc*/, const Instruction &inst,
                                                         amdgpu::Wavefront &wf) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  if (auto &read = s->lds_operand_read) {
    const InstructionSnapshot *current = current_instruction(*s, wf);
    if (!current)
      return;
    s->race_state->recordLdsOperandRead(*current, read->address, read->bytes, read->lane);
    read.reset();
    return;
  }
  // The CDNA1 CU delivers this hook for S_TRAP only after entering a configured
  // handler. Its implicit wait completes this wave's memory obligations.
  // Cross-wave LDS intervals still need a barrier.
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA1 && inst.mnemonic() == "s_trap" &&
      wf.in_trap_handler()) {
    for (auto counter : {amdgpu::WaitCounterType::VMCNT, amdgpu::WaitCounterType::EXPCNT,
                         amdgpu::WaitCounterType::LGKMCNT})
      s->race_state->applyWaitCounter(counter, 0);
    return;
  }

  // Embedded EXP waits affect register source lifetimes, handled by the core.
  // LDS progress observes the explicit memory-counter waits.
  if (!inst.is_waitcnt())
    return;
  const auto fields = amdgpu::WaitCounterPolicy::explicit_wait_fields(inst, wf.cu().arch());
  if (fields.failed() || !fields.value())
    return;

  onAmdgpuImplicitMemoryWait(wf, *fields.value());
}

void RaceDetectorPlugin::onAmdgpuImplicitMemoryWait(amdgpu::Wavefront &wf,
                                                    const amdgpu::WaitFields &fields) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  const auto model = amdgpu::waitcnt_model(wf.cu().arch()).value();
  for (size_t i = 0; i < fields.size(); ++i) {
    const auto threshold = fields[i];
    if (!threshold)
      continue;
    if (const auto counter =
            amdgpu::memory_wait_counter_type(static_cast<amdgpu::WaitCounterKind>(i), model))
      s->race_state->applyWaitCounter(*counter, static_cast<int>(*threshold));
  }
}

void RaceDetectorPlugin::onAmdgpuTrapEntered(amdgpu::Wavefront &wf) {
  if (wf.cu().arch() != ROCJITSU_CODE_ARCH_CDNA2)
    return;
  auto *s = get_state(wf);
  if (!s || !s->race_state)
    return;
  for (const auto counter : {amdgpu::WaitCounterType::VMCNT, amdgpu::WaitCounterType::EXPCNT,
                             amdgpu::WaitCounterType::LGKMCNT})
    s->race_state->applyWaitCounter(counter, 0);
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
