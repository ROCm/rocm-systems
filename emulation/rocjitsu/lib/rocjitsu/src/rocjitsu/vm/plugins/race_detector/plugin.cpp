// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/plugin.h"

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
#include "util/result.h"

#include "rocjitsu/vm/plugins/memory_access_observation.h"
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/interval_set.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"
#include "rocjitsu/vm/plugins/tensor_dma_issue_observation.h"

#include <algorithm>
#include <array>
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

util::FailureOr<InstructionSnapshot> capture_instruction(uint64_t pc, const Instruction &inst,
                                                         const amdgpu::Wavefront &wf) {
  static_assert(InstructionSnapshot::kMaxWords == amdgpu::DecodedInstructionCache::kNumWords);
  if (!inst.raw_encoding() || inst.size() <= 0 || inst.size() % sizeof(uint32_t) != 0 ||
      static_cast<size_t>(inst.size()) > InstructionSnapshot::kMaxWords * sizeof(uint32_t)) {
    report_plugin_failure(wf, "race detector instruction has no valid encoding");
    return util::Result::failure();
  }
  const bool sync_or_branch =
      inst.is_waitcnt() || inst.has_embedded_memory_wait() || inst.is_barrier() ||
      (inst.flags() & (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL));
  return InstructionSnapshot{pc, std::span(inst.raw_encoding(), inst.size() / sizeof(uint32_t)),
                             sync_or_branch};
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

util::FailureOr<MarkedInstruction> findConflict(const RaceViolation &v, RaceDetector &detector) {
  EventId event_id = v.conflictingEvent;
  if (!detector.events().contains(event_id))
    return util::Result::failure();
  return MarkedInstruction{detector.events().instruction(event_id),
                           detector.events().waveId(event_id).value, -1};
}

util::FailureOr<std::string> formatTrace(const RingBuffer<InstructionSnapshot, 256> &trace,
                        const std::function<util::FailureOr<std::string>(const InstructionSnapshot &)> &disassemble,
                        std::optional<MarkedInstruction> conflict, MarkedInstruction read) {
  const size_t n = trace.size();
  std::optional<size_t> read_index;
  for (size_t i = 0; i < n; ++i)
    if (trace[i] == read.instruction)
      read_index = i;
  std::optional<size_t> conflict_index;
  if (conflict && conflict->wave == read.wave)
    for (size_t i = 0; i < n; ++i)
      if (trace[i] == conflict->instruction) {
        conflict_index = i;
        break;
      }
  if (conflict_index && read_index && *conflict_index > *read_index)
    throw std::invalid_argument("race conflict follows the current access in the wave trace");
  const size_t last = read_index ? *read_index + 1 : n;
  size_t first = read_index.value_or(n);
  if (conflict && conflict->wave == read.wave) {
    first = conflict_index.value_or(0);
  } else if (conflict) {
    // Waves usually run the same code, so start where this wave last ran the
    // conflicting instruction; that shows its own path to the access.
    first = last - std::min(last, kOtherWaveContextRows);
    for (size_t i = 0; i < last; ++i)
      if (trace[i] == conflict->instruction)
        first = i;
  }
  const size_t span = last > first ? last - first : 0;
  auto nonsequential = [&](size_t i) {
    if (i <= first || i >= last)
      return false;
    const InstructionSnapshot &previous = trace[i - 1];
    return (trace[i].pc() < previous.pc() ||
            trace[i].pc() - previous.pc() != previous.encoding().size_bytes());
  };
  std::vector<bool> shown(n);
  constexpr size_t head = kTraceContextRows / 2;
  constexpr size_t tail = kTraceContextRows - head;
  for (size_t i = first; i < last; ++i) {
    shown[i] = span <= kTraceContextRows || i - first < head || last - i <= tail ||
               trace[i].is_sync_or_branch();
    if (nonsequential(i))
      shown[i - 1] = shown[i] = true;
  }
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
  if (!read_index)
    widest_pc = std::max(widest_pc, read.instruction.pc());
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
    const auto &instruction = mark.instruction;
    util::FailureOr<std::string> text = disassemble(instruction);
    if (text.failed())
      return util::Result::failure();
    oss << "  ==>  " << format_pc(instruction.pc()) << "  " << text.value() << note;
    annotation(mark);
    oss << '\n';
    return util::Result::success();
  };
  if (conflict && !conflict_index) {
    if (external(*conflict, conflict->wave == read.wave ? "  (before trace window)" : "").failed())
      return util::Result::failure();
    if (conflict->wave == read.wave) {
      oss << "       ... earlier instructions not recorded ...\n";
    } else
      oss << "       Execution trace below is for wave " << read.wave << ".\n";
  }
  auto emit = [&](size_t i) -> util::Result {
    const bool is_conflict = conflict_index && i == *conflict_index;
    const bool is_read = read_index && i == *read_index;
    const InstructionSnapshot &instruction = trace[i];
    if (nonsequential(i)) {
      const auto previous_pc = trace[i - 1].pc();
      oss << "       --- "
          << (instruction.pc() < previous_pc   ? "jump backward"
              : instruction.pc() > previous_pc ? "jump forward"
                                               : "same PC executed again")
          << " ---\n";
    }
    util::FailureOr<std::string> text = disassemble(instruction);
    if (text.failed())
      return util::Result::failure();
    oss << ((is_conflict || is_read) ? "  ==>  " : "       ") << format_pc(instruction.pc()) << "  "
        << text.value();
    if (is_conflict)
      annotation(*conflict);
    if (is_read)
      annotation(read);
    oss << '\n';
    return util::Result::success();
  };
  for (size_t i = first; i < last;) {
    if (shown[i]) {
      if (emit(i++).failed())
        return util::Result::failure();
    } else {
      const size_t gap = i;
      while (i < last && !shown[i])
        ++i;
      oss << "       ... " << i - gap << " instructions elided ...\n";
    }
  }
  if (!read_index && external(read, "  (outside recorded trace)").failed())
    return util::Result::failure();
  return oss.str();
}

namespace {

struct Transfer {
  std::string_view source;
  std::string_view destination;
};

// Name only where the data moves; one event type covers several operations,
// such as loads and returning atomics.
Transfer event_transfer(MemoryEventType type) {
  using enum MemoryEventType;
  switch (type) {
  case GLOBAL_TO_VGPR:
    return {"global memory", "VGPRs"};
  case VGPR_TO_GLOBAL:
    return {"VGPRs", "global memory"};
  case LDS_TO_VGPR:
    return {"LDS", "VGPRs"};
  case VGPR_TO_LDS:
    return {"VGPRs", "LDS"};
  case GLOBAL_TO_LDS:
    return {"global memory", "LDS"};
  case GLOBAL_TO_SGPR:
    return {"global memory", "SGPRs"};
  case GLOBAL_TO_TTMP:
    return {"global memory", "TTMPs"};
  case SCALAR_TO_GLOBAL:
    return {"scalar registers", "global memory"};
  case LDS_TO_GLOBAL:
    return {"LDS", "global memory"};
  }
  throw std::logic_error("race event has an unknown memory operation");
}

} // namespace

std::string formatRaceDescription(const RaceViolation &v, const EventRegistry &events,
                                  uint64_t access_pc, rj_code_arch_t arch) {
  if (!events.contains(v.conflictingEvent))
    throw std::out_of_range("race violation references an unavailable conflicting event");
  const util::FailureOr<amdgpu::WaitcntModel> model = amdgpu::waitcnt_model(arch);
  if (model.failed())
    throw std::invalid_argument("race report has an unsupported architecture");
  const EventId event = v.conflictingEvent;
  const int owner = events.waveId(event).value;
  const MemoryEventType type = events.type(event);
  const bool conflicting_read = v.space == RaceViolation::Space::LDS && isFromLds(type);
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
  case RaceViolation::Space::LDS:
    oss << "LDS byte " << v.index;
    break;
  }
  oss << " [workgroup (" << v.workgroupId.x << ", " << v.workgroupId.y << ", " << v.workgroupId.z
      << "), wave " << v.wave;
  if (v.space != RaceViolation::Space::SGPR && v.space != RaceViolation::Space::TTMP)
    oss << ", lane " << v.lane;
  oss << "]: " << hazard << ".\n";
  oss << "  Current access: " << (v.isWrite ? "write" : "read") << " at PC 0x" << std::hex
      << access_pc << std::dec << ".\n";
  const Transfer transfer = event_transfer(type);
  oss << "  Conflict: transfer from " << transfer.source << " into " << transfer.destination
      << " at PC 0x" << std::hex << events.instruction(event).pc() << std::dec << ", wave " << owner
      << ".\n";
  if (events.status(event) == EventStatus::ACTIVE) {
    oss << "  Completion for wave " << owner << " is not guaranteed; pending counters: ";
    std::vector<amdgpu::WaitCounterKind> displayed;
    for (const auto obligation : events.counterObligations(event)) {
      const amdgpu::WaitCounterType counter = obligation.wait_counter_type();
      if (!events.hasPendingWaitCounter(event, counter))
        continue;
      const amdgpu::WaitCounterKind kind = amdgpu::canonical_wait_counter(counter, model.value());
      if (std::ranges::find(displayed, kind) != displayed.end())
        continue;
      if (!displayed.empty())
        oss << ", ";
      oss << amdgpu::wait_counter_name(kind, model.value());
      displayed.push_back(kind);
    }
    oss << ".\n";
  } else if (events.status(event) == EventStatus::WAVE_COMPLETE) {
    oss << "  Completed for wave " << owner << ".\n";
  } else {
    throw std::logic_error("race violation references a retired memory event");
  }
  if (owner != v.wave)
    oss << "  Waves " << owner << " and " << v.wave << " have not synchronized this access.\n";
  return oss.str();
}

RaceDetectorPlugin::RaceDetectorPlugin(const char * /*config_json*/) : ExecutionPlugin("race") {}

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
        conflict, access_mark);
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
  if (RaceWavefrontState *state = get_state(wf); state && state->race_state) {
    // Normal termination drains the wave's memory operations before this hook.
    // Model s_endpgm's implicit wait, retaining LDS history until a subsequent
    // workgroup barrier supplies ordering with the surviving waves.
    const amdgpu::WaitcntModel model = amdgpu::waitcnt_model(wf.cu().arch()).value();
    PendingWaitCount wait;
    for (size_t index = 0; index < amdgpu::kWaitCounterCount; ++index) {
      const auto kind = static_cast<amdgpu::WaitCounterKind>(index);
      if (const std::optional<amdgpu::WaitCounterType> counter =
              amdgpu::memory_wait_counter_type(kind, model))
        wait.add(*counter, 0);
    }
    state->race_state->dispatch(wait);
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
  auto *rs = s->race_state;
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
  if (wf.cu().arch() == ROCJITSU_CODE_ARCH_CDNA3 && access.scratch_lane_mask &&
      (access.decoded_space == amdgpu::DecodedMemorySpace::SCRATCH ||
       access.decoded_space == amdgpu::DecodedMemorySpace::FLAT))
    rs->checkScalarRead(wf.memory_wait_shadow().canonical({RegClass::FLAT_SCRATCH, 0, 2}));

  const auto routed = routed_flat_issue_info(access, *decoded);
  const auto &issue = routed ? *routed : *decoded;
  const auto obligations = issue.counter_obligations();
  const MemoryOrderClass memoryOrder = memoryOrderForObligations(obligations);
  if (routed && access.request_lane_mask != 0 && supports_counter_capacity(wf.cu().arch()) &&
      rs->prepareForMemoryIssue(issue).failed()) {
    report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
    return;
  }

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
    LdsAccess ldsAccess{
        .read_modify_write =
            (routed.has_value() || access.decoded_space == amdgpu::DecodedMemorySpace::LOCAL) &&
            d.atomic_op != amdgpu::AtomicOp::NONE};
    ldsAccess.aligned_atomic_bytes = ldsAccess.read_modify_write ? d.elem_size : 0;
    auto type = d.is_load ? MemoryEventType::LDS_TO_VGPR : MemoryEventType::VGPR_TO_LDS;
    const uint32_t perLaneBytes = d.num_elems * d.elem_size;

    std::vector<uint32_t> registers;
    if (d.is_load) {
      auto destinations = validated_load_destinations(d, wf);
      if (!destinations)
        return;
      registers = std::move(*destinations);
    }

    const auto bounds =
        amdgpu::indexed_ds_access_bounds(d, wf.cu().arch(), wf.lds_base(), wf.lds_size());
    const auto atomicWidth = [&](uint64_t address, uint64_t bytes) -> uint8_t {
      return ldsAccess.read_modify_write && bytes == d.elem_size && address % d.elem_size == 0
                 ? d.elem_size
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
        // backing is larger. Keep register and counter dependencies below.
        if (address + bytes > std::numeric_limits<int>::max())
          continue;
        if (ldsAccess.aligned_atomic_bytes && !atomicWidth(address, bytes))
          ldsAccess.aligned_atomic_bytes = 0;
        laneBytes[set][lane] = static_cast<uint32_t>(bytes);
        intervals.append(static_cast<int>(address), static_cast<int>(address + bytes));
      }
    }
    intervals.finalize();
    const bool checkReads =
        (d.is_load || ldsAccess.read_modify_write) &&
        detector->mayConflict(intervals, waveId, /*isWrite=*/false, memoryOrder);
    const bool checkWrites =
        (!d.is_load || ldsAccess.read_modify_write) &&
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
    uint8_t byte_mask = vector_memory_byte_mask(d, wf);
    if (d.is_load) {
      for (uint32_t reg : registers)
        rs->checkVgprWrite(static_cast<int>(reg), execMask, byte_mask, memoryOrder);
    }
    if (obligations.empty())
      return;
    rs->registerEventWithIntervals(instruction, type, std::move(registers), execMask, byte_mask,
                                   std::move(intervals), obligations, memoryOrder,
                                   /*lastRegisterByteMask=*/0, ldsAccess);
    // A mixed instruction has one counter entry, with all register lanes but
    // only the ordinary LDS lanes' intervals. Do not also register a global event.
    return;
  }

  if (inst.data()->tag() == amdgpu::GLOBAL_MEM) {
    auto &d = *inst.data_as<amdgpu::VectorMemState>();
    if (d.exec_mask == 0) {
      // A VMEM instruction with EXEC zero issues and counts like any other; the
      // hardware may skip it only when LOADcnt, STOREcnt, ASYNCcnt and TENSORcnt
      // are all zero (CDNA5 ISA 3.2.2.1). An async copy is therefore registered
      // as an event without LDS bytes, so later partial waits complete the copies
      // issued before it. Registering it is safe even if it was skipped: every
      // earlier async copy had then completed, and partial waits retire the
      // oldest copies first, so counting it never retires a later copy.
      const bool asyncCopy =
          (d.lds_src || d.lds_dst) && std::ranges::any_of(obligations, [](const auto &obligation) {
            return obligation.wait_counter_type() == amdgpu::WaitCounterType::ASYNCCNT;
          });
      if (asyncCopy)
        rs->registerLdsRangeEvent(instruction,
                                  d.lds_src ? MemoryEventType::LDS_TO_GLOBAL
                                            : MemoryEventType::GLOBAL_TO_LDS,
                                  {}, obligations, memoryOrder);
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
  if (s->race_state->getVgprMemoryEvents(static_cast<int>(logical_reg)).empty())
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
  if (supports_counter_capacity(wf.cu().arch())) {
    // FLAT's address-dependent admission is applied after routing. Before its
    // operands are read, neither memory domain is guaranteed to be used.
    if (inst.is_memory_op() && !inst.mnemonic().starts_with("flat_")) {
      const auto *info = inst.amdgpu_memory_issue_info();
      assert(info && "memory instruction reached the race detector without issue metadata");
      if (info && (!info->exec_masked || wf.exec() != 0) &&
          s->race_state->prepareForMemoryIssue(*info).failed()) {
        report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
        return;
      }
    }

    // Messages share LGKMCNT on the capacity-modeled targets but are not memory
    // instructions, so they intentionally remain outside memory issue metadata.
    if (std::string_view(inst.mnemonic()).starts_with("s_sendmsg") &&
        s->race_state->prepareForCounterIncrement(amdgpu::WaitCounterType::LGKMCNT).failed()) {
      report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
      return;
    }
  }
  s->lds_operand_read.reset();
  s->trace.push(instruction.value());
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
  if (auto &read = s->lds_operand_read) {
    const InstructionSnapshot *current = current_instruction(*s, wf);
    if (!current)
      return;
    s->race_state->recordLdsOperandRead(*current, read->address, read->bytes, read->lane);
    read.reset();
    return;
  }
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

  // Only explicit waits are handled here. Embedded waits (e.g. VINTERP wait_exp)
  // remain unsupported by the plugin; they must retire events before the
  // instruction's register accesses, as in MemoryWaitScoreboard::before.
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
  PendingWaitCount wait;
  for (size_t i = 0; i < fields.size(); ++i) {
    const auto threshold = fields[i];
    if (!threshold)
      continue;
    if (const auto counter =
            amdgpu::memory_wait_counter_type(static_cast<amdgpu::WaitCounterKind>(i), model))
      wait.add(*counter, static_cast<int>(*threshold));
  }
  if (!wait.empty())
    s->race_state->dispatch(wait);
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
