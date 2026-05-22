// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "plugins/race_detection_plugin.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <race-detector/CommonRegister.h>
#include <race-detector/WaveRaceState.h>

#include <cassert>
#include <sstream>

namespace rocjitsu {
namespace amdgpu {

std::optional<MarkedPc> findConflict(const raceemulator::RaceViolation &v,
                                     raceemulator::RaceDetector &detector) {
  auto make = [&](auto eid) -> MarkedPc {
    return {detector.getEventPc(eid), detector.getEventWaveId(eid).value, -1};
  };
  if (v.space == raceemulator::RaceViolation::Space::VGPR) {
    auto &wrs = detector.getWaveRaceState(v.wave);
    for (auto eid : wrs.getVgprMemoryEvents(v.index))
      if (raceemulator::isToVgpr(detector.getEventType(eid)))
        return make(eid);
  } else if (v.space == raceemulator::RaceViolation::Space::SGPR) {
    auto &wrs = detector.getWaveRaceState(v.wave);
    for (auto eid : wrs.getWaveMemoryEvents()) {
      if (!raceemulator::isToSgpr(detector.getEventType(eid)))
        continue;
      for (uint32_t r : detector.getEventRegisters(eid))
        if (static_cast<int>(r) == v.index)
          return make(eid);
    }
  } else {
    assert(v.space == raceemulator::RaceViolation::Space::LDS &&
           "unexpected RaceViolation space (expected LDS)");
    const auto &events = v.isWrite ? detector.getLdsReadEvents() : detector.getLdsWriteEvents();
    for (auto eid : events)
      if (detector.getEventIntervals(eid).contains(v.index))
        return make(eid);
  }
  return std::nullopt;
}

// Format a race trace showing the instruction stream between the memory
// operation that wrote a register (conflict) and the instruction that read
// it before the write completed (read). The trace is a rolling window of
// recent PCs; disasm maps every PC seen in the kernel to its disassembly.
//
// Output uses ==> markers for the two involved instructions and annotates
// each with wave/lane. Instructions before the first marker are trimmed.
// When the conflict fell outside the trace window, its disassembly is still
// shown (from the disasm map) with a "(before trace window)" note.
std::string formatTrace(const RingBuffer<uint64_t, 256> &trace,
                        const std::unordered_map<uint64_t, std::string> &disasm,
                        std::optional<MarkedPc> conflict, MarkedPc read) {
  auto isMarked = [&](uint64_t pc) { return pc == read.pc || (conflict && pc == conflict->pc); };

  auto lookup = [&](uint64_t pc) -> const std::string & {
    static const std::string empty;
    auto it = disasm.find(pc);
    return it != disasm.end() ? it->second : empty;
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
      if (trace[i] == conflict->pc) {
        conflict_found = true;
        break;
      }
  }

  std::ostringstream oss;

  if (conflict && !conflict_found) {
    oss << "  ==>  0x" << std::hex << conflict->pc << std::dec << "  ";
    auto &d = lookup(conflict->pc);
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
    if (trace[i] == read.pc) {
      last = i + 1;
      break;
    }
  }

  constexpr size_t MAX_PRINT_SIZE = 32;
  size_t span = (last > first) ? last - first : 0;

  auto emit = [&](size_t i) {
    uint64_t pc = trace[i];
    bool is_conflict = conflict && pc == conflict->pc;
    bool is_read = pc == read.pc;
    oss << ((is_conflict || is_read) ? "  ==>  " : "       ");
    oss << "0x" << std::hex << pc << std::dec << "  " << lookup(pc);
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

RaceDetectionPlugin::RaceDetectionPlugin() : ExecutionPlugin("race") {
  const char *path = std::getenv("RJ_RACE_LOG");
  if (path) {
    log_file_ = fopen(path, "w");
  }
}

RaceDetectionPlugin::~RaceDetectionPlugin() {
  fprintf(stderr, "%s", getSummary().c_str());
  if (log_file_)
    fclose(log_file_);
}

std::string RaceDetectionPlugin::getSummary() const {
  const char *banner = "\n========================================\n"
                       " ROCJITSU RACE DETECTION SUMMARY\n"
                       "========================================\n";
  if (!observed_races_.empty()) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "%s  %zu race(s) detected\n"
             "========================================\n",
             banner, observed_races_.size());
    return buf;
  }
  return std::string(banner) + "  No races detected.\n"
                               "========================================\n";
}

void RaceDetectionPlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  fprintf(stderr, "[rocjitsu] Kernel dispatch: \"%s\"\n",
          info.kernel_name.empty() ? "?" : info.kernel_name.c_str());
}

void RaceDetectionPlugin::onAmdgpuWorkgroupDispatched(uint32_t wg_id, uint32_t dispatch_id,
                                                      uint32_t vgpr_count, uint32_t sgpr_count,
                                                      std::span<Wavefront *> wavefronts) {
  uint32_t num_waves = static_cast<uint32_t>(wavefronts.size());
  WorkgroupKey key{dispatch_id, wg_id};

  std::vector<Wavefront *> wf_ptrs(wavefronts.begin(), wavefronts.end());
  auto handler = [this, wf_ptrs, dispatch_id](raceemulator::RaceViolation v) {
    assert(v.wave >= 0 && static_cast<size_t>(v.wave) < wf_ptrs.size() &&
           "wave index out of range");
    Wavefront *wf = wf_ptrs[v.wave];
    uint64_t pc = wf->pc;

    std::ostringstream oss;
    if (v.space == raceemulator::RaceViolation::Space::VGPR)
      oss << "Race on VGPR v" << v.index;
    else if (v.space == raceemulator::RaceViolation::Space::SGPR)
      oss << "Race on SGPR s" << v.index;
    else
      oss << "Race on LDS byte " << v.index;
    oss << " [workgroup (" << v.workgroupId.x << ", " << v.workgroupId.y << ", " << v.workgroupId.z
        << "), wave " << v.wave;
    if (v.space != raceemulator::RaceViolation::Space::SGPR)
      oss << ", lane " << v.lane;
    oss << "]\n";

    auto *ws = get_state(wf);
    assert(ws && ws->race_state && "no wavefront state for race");
    auto *detector = ws->race_state->getDetector();
    auto conflict = findConflict(v, *detector);
    assert(conflict.has_value() && "conflict not found for race violation");

    MarkedPc read_mark{pc, v.wave, v.lane};
    oss << formatTrace(ws->trace, ws->disasm->to_map(), conflict, read_mark);

    auto contained0 = observed_races_.count({dispatch_id, pc});
    auto contained1 = observed_races_.count({dispatch_id, conflict->pc});

    if (!contained0 && !contained1) {
      observed_races_.emplace(dispatch_id, pc);
      observed_races_.emplace(dispatch_id, conflict->pc);
      fprintf(stderr, "[rocjitsu] RACE #%zu:\n%s", observed_races_.size(), oss.str().c_str());

      if (log_file_) {
        std::string conflict_mnemonic = "unknown";
        const char *space = v.space == raceemulator::RaceViolation::Space::VGPR   ? "VGPR"
                            : v.space == raceemulator::RaceViolation::Space::SGPR ? "SGPR"
                                                                                  : "LDS";
        fprintf(log_file_,
                "RACE type=%s reg=%d wave=%d lane=%d wg=%d,%d,%d "
                "conflict=%s\n%sEND_RACE\n",
                space, v.index, v.wave, v.lane, v.workgroupId.x, v.workgroupId.y, v.workgroupId.z,
                conflict_mnemonic.c_str(), oss.str().c_str());
        fflush(log_file_);
      }
    }
  };

  detectors_[key] = std::make_unique<raceemulator::RaceDetector>(
      static_cast<int>(num_waves), static_cast<int>(vgpr_count), static_cast<int>(sgpr_count),
      raceemulator::Dim3d(static_cast<int>(wg_id)), std::move(handler));

  auto &det = *detectors_[key];
  auto &dc = dispatch_disasm_[dispatch_id];
  if (!dc)
    dc = std::make_shared<DisasmCache>();
  for (uint32_t w = 0; w < num_waves; ++w) {
    auto state = std::make_unique<RaceWavefrontState>();
    state->race_state = &det.getWaveRaceState(static_cast<int>(w));
    state->disasm = dc;
    wavefronts[w]->set_plugin_state(slot_index(), std::move(state));
  }
}

void RaceDetectionPlugin::onAmdgpuRouteMemoryInstruction(const Instruction &inst, Wavefront &wf) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  auto *rs = s->race_state;
  auto *detector = rs->getDetector();
  auto waveId = rs->getWaveId();

  if (inst.data()->tag() == LOCAL_MEM) {
    auto &d = *inst.data_as<VectorMemState>();
    auto type = d.is_load ? raceemulator::MemoryEventType::LDS_TO_VGPR
                          : raceemulator::MemoryEventType::VGPR_TO_LDS;

    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(wf.exec() & (1ULL << lane)))
        continue;
      int addr = static_cast<int>(d.per_lane_addr[lane]);
      int nBytes = static_cast<int>(d.elem_size);
      if (d.is_load)
        detector->validateRead(addr, waveId, static_cast<int>(lane), nBytes);
      else
        detector->validateWrite(addr, waveId, static_cast<int>(lane), nBytes);
    }
    uint32_t laneAddrs[64];
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
      laneAddrs[lane] = static_cast<uint32_t>(d.per_lane_addr[lane]);
    std::vector<uint32_t> registers;
    if (d.is_load) {
      uint32_t logicalBase = d.dst_reg_base - wf.vgpr_alloc().base;
      registers.resize(d.num_elems);
      for (uint32_t i = 0; i < d.num_elems; ++i)
        registers[i] = logicalBase + i;
    }
    uint8_t byteMask = d.d16_lo ? 0x3 : d.d16_hi ? 0xC : 0xF;
    rs->registerLdsEvent(wf.pc, type, std::move(registers), wf.exec(), wf.wf_size(),
                         std::span<const uint32_t>(laneAddrs, wf.wf_size()), d.elem_size, byteMask);
  }

  if (inst.data()->tag() == GLOBAL_MEM) {
    auto &d = *inst.data_as<VectorMemState>();
    if (d.lds_dst) {
      uint32_t perLaneBytes = d.num_elems * d.elem_size;
      uint32_t ldsAddrs[64];
      for (uint32_t lane = 0; lane < wf.wf_size(); ++lane)
        ldsAddrs[lane] = d.lds_base + lane * perLaneBytes;
      rs->registerLdsEvent(wf.pc, raceemulator::MemoryEventType::GLOBAL_TO_LDS, {}, wf.exec(),
                           wf.wf_size(), std::span<const uint32_t>(ldsAddrs, wf.wf_size()),
                           perLaneBytes);
    } else if (d.is_load && d.dst_reg_base >= wf.vgpr_alloc().base) {
      uint32_t logicalBase = d.dst_reg_base - wf.vgpr_alloc().base;
      std::vector<uint32_t> registers(d.num_elems);
      for (uint32_t i = 0; i < d.num_elems; ++i)
        registers[i] = logicalBase + i;
      uint8_t byteMask = d.d16_lo ? 0x3 : d.d16_hi ? 0xC : 0xF;
      rs->registerEvent(wf.pc, raceemulator::MemoryEventType::GLOBAL_TO_VGPR, std::move(registers),
                        wf.exec(), byteMask);
    } else if (!d.is_load) {
      rs->registerEvent(wf.pc, raceemulator::MemoryEventType::VGPR_TO_GLOBAL, {}, wf.exec());
    }
  }

  if (inst.data()->tag() == SCALAR_MEM) {
    auto &d = *inst.data_as<ScalarMemState>();
    if (d.is_load) {
      uint32_t logicalBase = d.dst_reg_base - wf.sgpr_alloc().base;
      std::vector<uint32_t> registers(d.num_dwords);
      for (uint32_t i = 0; i < d.num_dwords; ++i)
        registers[i] = logicalBase + i;
      rs->registerEvent(wf.pc, raceemulator::MemoryEventType::GLOBAL_TO_SGPR, std::move(registers),
                        wf.exec());
    }
  }
}

void RaceDetectionPlugin::onAmdgpuReadVgpr(Wavefront *wf, uint32_t physical_reg, uint32_t lane,
                                           uint8_t byteMask) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  uint32_t logical_reg = physical_reg - wf->vgpr_alloc().base;
  s->race_state->checkVgprRead(static_cast<int>(logical_reg), static_cast<int>(lane), byteMask);
}

void RaceDetectionPlugin::onAmdgpuReadSgpr(Wavefront *wf, uint32_t physical_reg) {
  auto *s = get_state(wf);
  assert(s && s->race_state);
  uint32_t logical_reg = physical_reg - wf->sgpr_alloc().base;
  s->race_state->checkSgprRead(static_cast<int>(logical_reg));
}

void RaceDetectionPlugin::beforeAmdgpuExecuteInstruction(uint64_t pc, const Instruction &inst,
                                                         Wavefront &wf) {
  auto *s = get_state(wf);
  assert(s);
  s->trace.push(pc);
  s->disasm->record(pc, inst);
}

void RaceDetectionPlugin::afterAmdgpuExecuteInstruction(uint64_t /*pc*/, const Instruction &inst,
                                                        Wavefront &wf) {
  auto *s = get_state(wf);
  assert(s && s->race_state);

  if (inst.mnemonic().starts_with("s_waitcnt")) {
    auto &tgt = wf.wait_target();
    s->race_state->dispatch(
        raceemulator::PendingWaitCount{static_cast<int>(tgt.vmcnt), static_cast<int>(tgt.lgkmcnt)});
  }
}

void RaceDetectionPlugin::onAmdgpuBarrierResolved(std::span<Wavefront *> wavefronts) {
  for (auto *wf : wavefronts) {
    auto *s = get_state(wf);
    assert(s && s->race_state);
    s->race_state->flushWaveCompleteMemoryEvents();
  }
}

} // namespace amdgpu
} // namespace rocjitsu
