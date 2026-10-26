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

#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"

#include <flatbuffers/flexbuffers.h>
#include <flatbuffers/idl.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <format>
#include <iomanip>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace rocjitsu::plugins::race_detector {

namespace {

InstructionSnapshot capture_instruction(uint64_t pc, const Instruction &inst) {
  InstructionSnapshot snapshot{.pc = pc};
  static_assert(InstructionSnapshot::kMaxWords * sizeof(uint32_t) ==
                amdgpu::InstructionCache::kFetchBytes);
  if (!inst.raw_encoding())
    throw std::logic_error(
        std::format("race detector instruction at PC {:#x} has no encoding", pc));
  if (inst.size() <= 0 || inst.size() % sizeof(uint32_t) != 0 ||
      static_cast<size_t>(inst.size()) > sizeof(snapshot.words))
    throw std::logic_error(std::format(
        "race detector instruction at PC {:#x} has invalid encoding size {}", pc, inst.size()));
  snapshot.keep_in_trace =
      inst.is_waitcnt() || inst.has_embedded_memory_wait() || inst.is_barrier() ||
      (inst.flags() & (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL));
  snapshot.word_count = static_cast<uint32_t>(inst.size()) / sizeof(uint32_t);
  std::memcpy(snapshot.words.data(), inst.raw_encoding(), inst.size());
  return snapshot;
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
                        std::optional<MarkedInstruction> conflict, MarkedInstruction read,
                        size_t trace_context_instructions) {
  if (trace_context_instructions < 2 || trace_context_instructions > 256)
    throw std::invalid_argument("race trace_context_instructions must be an integer from 2 to 256");
  const size_t n = trace.size();
  auto trace_index = [&](const MarkedInstruction &mark) -> std::optional<size_t> {
    if (mark.wave != read.wave || mark.instruction.sequence == UINT64_MAX)
      return std::nullopt;
    for (size_t i = 0; i < n; ++i)
      if (trace.sequence(i) == mark.instruction.sequence)
        return i;
    return std::nullopt;
  };
  const auto conflict_index = conflict ? trace_index(*conflict) : std::nullopt;
  const auto read_index = trace_index(read);
  if (conflict_index && read_index && *conflict_index > *read_index)
    throw std::invalid_argument("race conflict follows the current access in the wave trace");
  const size_t first = conflict ? conflict_index.value_or(0) : read_index.value_or(n);
  const size_t last = read_index ? *read_index + 1 : n;
  const size_t span = last > first ? last - first : 0;
  auto nonsequential = [&](size_t i) {
    if (i <= first || i >= last)
      return false;
    const auto previous = trace[i - 1];
    return previous.word_count &&
           (trace[i].pc < previous.pc ||
            trace[i].pc - previous.pc != previous.word_count * sizeof(uint32_t));
  };
  std::array<bool, 256> shown{};
  const size_t head = (trace_context_instructions + 1) / 2;
  const size_t tail = trace_context_instructions / 2;
  size_t nonsequential_count = 0;
  size_t backward_count = 0;
  for (size_t i = first; i < last; ++i) {
    shown[i] = span <= trace_context_instructions || i - first < head || last - i <= tail ||
               trace[i].keep_in_trace;
    if (nonsequential(i)) {
      shown[i - 1] = shown[i] = true;
      ++nonsequential_count;
      backward_count += trace[i].pc < trace[i - 1].pc;
    }
  }
  std::ostringstream oss;
  oss << "       PC diff  PC  Instruction\n";
  auto annotation = [&](const MarkedInstruction &mark) {
    oss << "  ; <-- wave " << mark.wave;
    if (mark.lane >= 0)
      oss << " lane " << mark.lane;
  };
  auto external = [&](const MarkedInstruction &mark, std::string_view note) {
    const auto &instruction = mark.instruction.snapshot;
    oss << "  ==>  " << std::setw(7) << "-" << "  0x" << std::hex << instruction.pc << std::dec
        << "  " << disassemble(instruction) << note;
    annotation(mark);
    oss << '\n';
  };
  if (conflict && !conflict_index) {
    external(*conflict, conflict->wave == read.wave ? "  (before trace window)"
                                                    : "  (outside this wave's trace)");
    if (conflict->wave == read.wave) {
      if (span && conflict->instruction.sequence != UINT64_MAX &&
          trace.sequence(first) > conflict->instruction.sequence)
        oss << "       ... " << trace.sequence(first) - conflict->instruction.sequence - 1
            << " earlier instructions not recorded ...\n";
      else
        oss << "       ... earlier instructions not recorded ...\n";
    } else
      oss << "       Execution trace below is for wave " << read.wave << ".\n";
  }
  auto emit = [&](size_t i) {
    const bool is_conflict = conflict_index && i == *conflict_index;
    const bool is_read = read_index && i == *read_index;
    const auto instruction = is_conflict ? conflict->instruction.snapshot
                             : is_read   ? read.instruction.snapshot
                                         : trace[i];
    std::string delta = "-";
    if (i > first) {
      const uint64_t previous = trace[i - 1].pc;
      delta = instruction.pc >= previous ? "+" + std::to_string(instruction.pc - previous)
                                         : "-" + std::to_string(previous - instruction.pc);
    }
    oss << ((is_conflict || is_read) ? "  ==>  " : "       ") << std::setw(7) << delta << "  0x"
        << std::hex << instruction.pc << std::dec << "  " << disassemble(instruction);
    if (is_conflict)
      annotation(*conflict);
    if (is_read)
      annotation(read);
    if (nonsequential(i))
      oss << "  ; nonsequential PC";
    oss << '\n';
  };
  size_t shown_count = 0;
  for (size_t i = first; i < last;) {
    if (shown[i]) {
      emit(i++);
      ++shown_count;
    } else {
      const size_t gap = i;
      while (i < last && !shown[i])
        ++i;
      oss << "       ... " << i - gap << " instructions elided ...\n";
    }
  }
  if (!read_index)
    external(read, "  (outside recorded trace)");
  oss << "  Trace window: " << span << " recorded instructions, " << shown_count << " shown, "
      << span - shown_count << " elided; " << nonsequential_count
      << " nonsequential PC transitions (" << backward_count << " backward).\n";
  oss << "  PC diffs are signed decimal bytes from the previous executed instruction in wave "
      << read.wave << ", including elided instructions.\n";
  if (conflict && conflict->wave == read.wave && conflict->instruction.sequence != UINT64_MAX &&
      read.instruction.sequence != UINT64_MAX &&
      read.instruction.sequence >= conflict->instruction.sequence)
    oss << "  Same-wave instruction distance: "
        << read.instruction.sequence - conflict->instruction.sequence << " instruction steps.\n";
  return oss.str();
}

namespace {

std::string_view counter_name(amdgpu::WaitCounterType type) {
  using enum amdgpu::WaitCounterType;
  switch (type) {
  case VMCNT:
    return "vmcnt";
  case LGKMCNT:
    return "lgkmcnt";
  case EXPCNT:
    return "expcnt";
  case VSCNT:
    return "vscnt";
  case LOADCNT:
    return "loadcnt";
  case STORECNT:
    return "storecnt";
  case DSCNT:
    return "dscnt";
  case KMCNT:
    return "kmcnt";
  case TENSORCNT:
    return "tensorcnt";
  case ASYNCCNT:
    return "asynccnt";
  }
  throw std::logic_error("race event has an unknown completion counter");
}

std::string_view event_description(MemoryEventType type) {
  using enum MemoryEventType;
  switch (type) {
  case GLOBAL_TO_VGPR:
    return "load from global memory into VGPRs";
  case VGPR_TO_GLOBAL:
    return "store from VGPRs into global memory";
  case LDS_TO_VGPR:
    return "load from LDS into VGPRs";
  case VGPR_TO_LDS:
    return "store from VGPRs into LDS";
  case GLOBAL_TO_LDS:
    return "transfer from global memory into LDS";
  case GLOBAL_TO_SGPR:
    return "load from global memory into SGPRs";
  case GLOBAL_TO_TTMP:
    return "load from global memory into TTMPs";
  case SCALAR_TO_GLOBAL:
    return "store from scalar registers into global memory";
  case N:
    break;
  }
  throw std::logic_error("race event has an unknown memory operation");
}

size_t parse_trace_limit(const char *config_json) {
  if (!config_json)
    return 128;
  flexbuffers::Builder builder;
  flatbuffers::Parser parser;
  if (!parser.ParseFlexBuffer(config_json, nullptr, &builder))
    throw std::invalid_argument("race plugin configuration is not valid JSON");
  const auto root = flexbuffers::GetRoot(builder.GetBuffer());
  if (!root.IsMap())
    throw std::invalid_argument("race plugin configuration must be an object");
  const auto config = root.AsMap();
  if (config.size() == 0)
    return 128;
  for (size_t i = 0; i < config.size(); ++i)
    if (std::string_view(config.Keys()[i].AsKey()) != "trace_context_instructions")
      throw std::invalid_argument("race plugin configuration has an unknown option");
  const auto value = config["trace_context_instructions"];
  if (!value.IsIntOrUint() || (value.IsInt() && value.AsInt64() < 2) || value.AsUInt64() > 256)
    throw std::invalid_argument("race trace_context_instructions must be an integer from 2 to 256");
  return static_cast<size_t>(value.AsUInt64());
}

} // namespace

std::string formatRaceDescription(const RaceViolation &v, const EventRegistry &events,
                                  uint64_t access_pc, rj_code_arch_t arch) {
  if (!events.contains(v.conflictingEvent))
    throw std::out_of_range("race violation references an unavailable conflicting event");
  const auto model = amdgpu::waitcnt_model(arch);
  if (model.failed())
    throw std::invalid_argument("race report has an unsupported architecture");
  const auto event = v.conflictingEvent;
  const int owner = events.waveId(event).value;
  const auto type = events.type(event);
  const bool conflicting_read =
      v.space == RaceViolation::Space::LDS && type == MemoryEventType::LDS_TO_VGPR;
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
  oss << "  Conflict: " << event_description(type) << " at PC 0x" << std::hex << events.pc(event)
      << std::dec << ", wave " << owner << ".\n";
  if (events.status(event) == EventStatus::ACTIVE) {
    oss << "  Completion for wave " << owner << " is not guaranteed; pending counters: ";
    std::set<amdgpu::WaitCounterType> displayed;
    for (const auto obligation : events.counterObligations(event)) {
      const auto counter = obligation.wait_counter_type();
      if (!events.hasPendingWaitCounter(event, counter))
        continue;
      const auto architectural_counter =
          amdgpu::memory_wait_counter_type(amdgpu::canonical_wait_counter(counter, model.value()),
                                           model.value())
              .value_or(counter);
      if (displayed.contains(architectural_counter))
        continue;
      if (!displayed.empty())
        oss << ", ";
      oss << counter_name(architectural_counter);
      displayed.insert(architectural_counter);
    }
    oss << ".\n";
  } else if (events.status(event) == EventStatus::WAVE_COMPLETE) {
    oss << "  Completed for wave " << owner << ", but not synchronized with wave " << v.wave
        << ".\n";
  } else {
    throw std::logic_error("race violation references a retired memory event");
  }
  if (owner != v.wave && events.status(event) == EventStatus::ACTIVE)
    oss << "  Waves " << owner << " and " << v.wave << " have not synchronized this access.\n";
  return oss.str();
}

RaceDetectorPlugin::RaceDetectorPlugin(const char *config_json)
    : ExecutionPlugin("race"), trace_context_instructions_(parse_trace_limit(config_json)) {}

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
    uint64_t pc = wf->pc;

    {
      std::lock_guard<std::mutex> lock(report_mutex_);
      if (observed_races_.count({dispatch_id, pc})) {
        ++observations_suppressed_;
        return;
      }
    }

    auto *ws = get_state(wf);
    assert(ws && ws->race_state && "no wavefront state for race");
    auto *detector = ws->race_state->getDetector();
    MarkedInstruction conflict = findConflict(v, *detector);

    std::ostringstream oss;
    oss << formatRaceDescription(v, detector->events(), pc, wf->cu().arch());

    if (!ws->trace.size() || ws->trace[ws->trace.size() - 1].pc != pc)
      throw std::logic_error(
          std::format("race detector has no current instruction at PC {:#x}", pc));
    const RecordedInstruction current{ws->trace[ws->trace.size() - 1],
                                      ws->trace.sequence(ws->trace.size() - 1)};
    MarkedInstruction access_mark{current, v.wave, v.lane};
    oss << formatTrace(
        ws->trace,
        [wf](const InstructionSnapshot &instruction) {
          return wf->disassemble_instruction(instruction.encoding(), instruction.pc);
        },
        conflict, access_mark, trace_context_instructions_);

    {
      std::lock_guard<std::mutex> lock(report_mutex_);
      bool is_new = !observed_races_.count({dispatch_id, pc}) &&
                    !observed_races_.count({dispatch_id, conflict.instruction.snapshot.pc});
      observed_races_.emplace(dispatch_id, pc);
      if (is_new) {
        ++reports_emitted_;
        observed_races_.emplace(dispatch_id, conflict.instruction.snapshot.pc);
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
  const RecordedInstruction instruction{capture_instruction(wf.pc, inst),
                                        s->trace.size() && s->trace[s->trace.size() - 1].pc == wf.pc
                                            ? s->trace.sequence(s->trace.size() - 1)
                                            : UINT64_MAX};
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
    // DDS participates in the instruction's dependencies, but its addresses
    // do not describe bytes in the ordinary LDS allocation.
    const uint64_t ldsMask =
        (mixedFlat ? execMask & access.flat_local_lane_mask : execMask) & ~d.flat_dds_lane_mask;
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
        const auto bytes = detector->ldsBytesInRange(address, perLaneBytes, d.elem_size);
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
      rs->registerLdsEvent(wf.pc, type, std::move(registers), execMask, wf.wf_size(),
                           std::span<const uint32_t>(laneAddrs, wf.wf_size()),
                           std::span<const uint32_t>(secondLaneAddrs, wf.wf_size()), perLaneBytes,
                           byte_mask, obligations, memoryOrder, d.elem_size, instruction);
    } else {
      rs->registerLdsEvent(wf.pc, type, std::move(registers), execMask, wf.wf_size(),
                           std::span<const uint32_t>(laneAddrs, wf.wf_size()), perLaneBytes,
                           byte_mask, obligations, memoryOrder, ldsMask, d.elem_size, instruction);
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
      rs->registerLdsEvent(wf.pc, MemoryEventType::GLOBAL_TO_LDS, {}, validLaneMask, wf.wf_size(),
                           std::span<const uint32_t>(ldsAddrs, wf.wf_size()), perLaneBytes, 0xF,
                           obligations, memoryOrder, ~uint64_t{0}, 0, instruction);
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
      rs->registerEvent(wf.pc, MemoryEventType::GLOBAL_TO_VGPR, std::move(registers), d.exec_mask,
                        byte_mask, obligations, memoryOrder, last_byte_mask, instruction);
    } else if (!obligations.empty()) {
      rs->registerEvent(wf.pc, MemoryEventType::VGPR_TO_GLOBAL, {}, d.exec_mask, 0xF, obligations,
                        memoryOrder, 0, instruction);
    }
  }

  if (inst.data()->tag() == amdgpu::SCALAR_MEM) {
    auto &d = *inst.data_as<amdgpu::ScalarMemState>();
    if (d.is_load) {
      if (const auto reg = d.dst_register.register_ref()) {
        rs->registerScalarLoad(wf.pc, *reg, wf.exec(), obligations, memoryOrder, instruction);
      } else {
        rs->registerEvent(wf.pc, MemoryEventType::GLOBAL_TO_SGPR, {}, wf.exec(), 0xF, obligations,
                          memoryOrder, 0, instruction);
      }
    } else {
      rs->registerEvent(wf.pc, MemoryEventType::SCALAR_TO_GLOBAL, {}, wf.exec(), 0xF, obligations,
                        memoryOrder, 0, instruction);
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
  const auto instruction = capture_instruction(pc, inst);
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
