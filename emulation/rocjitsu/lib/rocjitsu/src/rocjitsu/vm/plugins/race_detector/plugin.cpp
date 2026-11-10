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
/// every counter it could add to is zero (CDNA5 ISA 3.2.2.1), in which case no
/// older operation remains to complete. Either way, its admission is sound.
bool exec_zero_memory_issue_counts(rj_code_arch_t arch) { return arch == ROCJITSU_CODE_ARCH_CDNA5; }

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
                                  const amdgpu::MemoryIssueInfo &routed, bool is_load) {
  for (const auto obligation : decoded.counter_obligations()) {
    const bool used = std::ranges::any_of(routed.counter_obligations(), [&](const auto routed_one) {
      return routed_one.wait_counter_type() == obligation.wait_counter_type();
    });
    if (used)
      continue;
    const std::array slot{obligation};
    race.registerEvent(
        instruction, is_load ? MemoryEventType::GLOBAL_TO_VGPR : MemoryEventType::VGPR_TO_GLOBAL,
        /*registers=*/{}, /*execMask=*/0, /*byteMask=*/0xF, slot, obligation.completion_class());
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
  // Before gfx1250, FLAT is admitted here, on the domains its addresses used.
  // On gfx1250 it counts on both domains and was admitted at issue.
  const bool flatOnBothDomains = routed && flat_counts_on_both_domains(wf.cu().arch());
  if (routed && !flatOnBothDomains && access.request_lane_mask != 0 &&
      supports_counter_capacity(wf.cu().arch()) && rs->prepareForMemoryIssue(issue).failed()) {
    report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
    return;
  }
  if (flatOnBothDomains)
    register_unused_flat_domains(*rs, instruction, *decoded, *routed,
                                 inst.data_as<amdgpu::VectorMemState>()->is_load);

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
  // Tensor instructions carry no memory issue metadata, so their TENSORcnt
  // backpressure is applied here, once per counted transfer, before the
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
    // Before gfx1250, FLAT's address-dependent admission is applied after
    // routing: before its operands are read, neither memory domain is
    // guaranteed to be used. On gfx1250 every FLAT counts on both domains, so
    // it is admitted here like any other memory instruction.
    const bool flat = inst.mnemonic().starts_with("flat_");
    if (inst.is_memory_op() && (!flat || flat_counts_on_both_domains(wf.cu().arch()))) {
      const auto *info = inst.amdgpu_memory_issue_info();
      assert(info && "memory instruction reached the race detector without issue metadata");
      if (info &&
          (!info->exec_masked || wf.exec() != 0 || exec_zero_memory_issue_counts(wf.cu().arch())) &&
          s->race_state->prepareForMemoryIssue(*info).failed()) {
        report_plugin_failure(wf, "race detector counter capacity or increment is invalid");
        return;
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
