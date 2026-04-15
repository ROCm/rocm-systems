// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/race_detection_plugin.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <race-emulator/CommonRegister.h>
#include <race-emulator/WaveRaceState.h>

#include <sstream>

namespace rocjitsu {
namespace amdgpu {

void RaceDetectionPlugin::onWorkgroupDispatched(
    uint32_t wg_id, uint32_t lds_size, uint32_t vgpr_count,
    uint32_t sgpr_count, std::span<Wavefront *> wavefronts) {
  uint32_t num_waves = static_cast<uint32_t>(wavefronts.size());

  auto handler = [this, wg_id](raceemulator::RaceViolation v) {
    // Find the violating wavefront to get its PC.
    Wavefront *violating_wf = nullptr;
    for (auto &[wf, idx] : wavefront_to_index_) {
      if (static_cast<int>(idx) == v.wave && wf->wg_id() == wg_id) {
        violating_wf = wf;
        break;
      }
    }

    std::ostringstream oss;
    auto *detector = detectors_[wg_id].get();

    if (v.space == raceemulator::RaceViolation::Space::VGPR) {
      oss << "VGPR race on v" << v.index << ": wave " << v.wave
          << (v.isWrite ? " write" : " read") << " lane " << v.lane;
      if (violating_wf) {
        oss << " (pc=0x" << std::hex << violating_wf->pc << std::dec << ")";
      }

      auto &waveRaceState = detector->getWaveRaceState(v.wave);
      for (auto eventId : waveRaceState.getVgprMemoryEvents(v.index)) {
        if (raceemulator::isToVgpr(detector->getEventType(eventId))) {
          oss << " conflicts with pending load (pc=0x" << std::hex
              << detector->getEventPc(eventId) << std::dec << ")";
          break;
        }
      }
    } else if (v.space == raceemulator::RaceViolation::Space::SGPR) {
      oss << "SGPR race on s" << v.index << ": wave " << v.wave
          << (v.isWrite ? " write" : " read");
      if (violating_wf) {
        oss << " (pc=0x" << std::hex << violating_wf->pc << std::dec << ")";
      }

      auto &waveRaceState = detector->getWaveRaceState(v.wave);
      for (auto eventId : waveRaceState.getWaveMemoryEvents()) {
        if (!raceemulator::isToSgpr(detector->getEventType(eventId)))
          continue;
        auto regs = detector->getEventRegisters(eventId);
        bool found = false;
        for (uint32_t r : regs) {
          if (static_cast<int>(r) == v.index) {
            found = true;
            break;
          }
        }
        if (found) {
          oss << " conflicts with pending load (pc=0x" << std::hex
              << detector->getEventPc(eventId) << std::dec << ")";
          break;
        }
      }
    } else {
      oss << "LDS race at byte " << v.index << " (0x" << std::hex << v.index
          << std::dec << "): wave " << v.wave
          << (v.isWrite ? " write" : " read") << " lane " << v.lane;
      if (violating_wf) {
        oss << " (pc=0x" << std::hex << violating_wf->pc << std::dec << ")";
      }

      const auto &events = v.isWrite ? detector->getLdsReadEvents()
                                     : detector->getLdsWriteEvents();
      for (auto eventId : events) {
        if (detector->getEventIntervals(eventId).contains(v.index)) {
          auto conflictingWave = detector->getEventWaveId(eventId);
          auto conflictingPc = detector->getEventPc(eventId);
          bool conflictingIsWrite = !v.isWrite;
          oss << " conflicts with wave " << conflictingWave.value
              << (conflictingIsWrite ? " write" : " read") << " (pc=0x"
              << std::hex << conflictingPc << std::dec << ")";
          break;
        }
      }
    }

    oss << ", workgroup (" << v.workgroupId.x << "," << v.workgroupId.y << ","
        << v.workgroupId.z << ")";
    diagnostics_[wg_id].push_back(oss.str());
  };

  detectors_[wg_id] = std::make_unique<raceemulator::RaceDetector>(
      static_cast<int>(lds_size), static_cast<int>(num_waves),
      static_cast<int>(vgpr_count), static_cast<int>(sgpr_count),
      raceemulator::Dim3d(static_cast<int>(wg_id)), std::move(handler));

  auto &det = *detectors_[wg_id];
  for (uint32_t w = 0; w < num_waves; ++w) {
    auto *wf = wavefronts[w];
    wavefront_to_index_[wf] = w;
    wavefront_to_race_state_[wf] =
        &det.getWaveRaceState(static_cast<int>(w));
  }
}

void RaceDetectionPlugin::onMemoryInstruction(Instruction *inst,
                                              Wavefront &wf) {
  auto it = wavefront_to_race_state_.find(&wf);
  auto *rs = (it != wavefront_to_race_state_.end()) ? it->second : nullptr;
  if (!rs)
    return;

  auto *detector = rs->getDetector();
  auto waveId = rs->getWaveId();

  if (inst->data()->tag() == LOCAL_MEM) {
    auto &d = *inst->data_as<VectorMemState>();
    auto type = d.is_load ? raceemulator::MemoryEventType::LDS_TO_VGPR
                          : raceemulator::MemoryEventType::VGPR_TO_LDS;

    // Validate each active lane's access against outstanding events.
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      if (!(wf.exec() & (1ULL << lane)))
        continue;
      int addr = static_cast<int>(d.per_lane_addr[lane]);
      int nBytes = static_cast<int>(d.elem_size);
      if (d.is_load) {
        detector->validateRead(addr, waveId, static_cast<int>(lane), nBytes);
      } else {
        detector->validateWrite(addr, waveId, static_cast<int>(lane), nBytes);
      }
    }

    // Register the event for future conflict detection.
    std::vector<uint32_t> laneAddrs(wf.wf_size());
    for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
      laneAddrs[lane] = static_cast<uint32_t>(d.per_lane_addr[lane]);
    }
    std::vector<uint32_t> registers;
    if (d.is_load) {
      uint32_t logicalBase = d.dst_reg_base - wf.vgpr_alloc().base;
      registers.resize(d.num_elems);
      for (uint32_t i = 0; i < d.num_elems; ++i) {
        registers[i] = logicalBase + i;
      }
    }
    rs->registerLdsEvent(static_cast<int>(wf.pc), type, std::move(registers),
                         wf.exec(), wf.wf_size(), laneAddrs, d.elem_size);
  }

  if (inst->data()->tag() == GLOBAL_MEM) {
    auto &d = *inst->data_as<VectorMemState>();
    if (d.lds_dst) {
      uint32_t perLaneBytes = d.num_elems * d.elem_size;
      std::vector<uint32_t> ldsAddrs(wf.wf_size());
      for (uint32_t lane = 0; lane < wf.wf_size(); ++lane) {
        ldsAddrs[lane] = d.lds_base + lane * perLaneBytes;
      }
      rs->registerLdsEvent(static_cast<int>(wf.pc),
                           raceemulator::MemoryEventType::GLOBAL_TO_LDS,
                           /*registers=*/{}, wf.exec(), wf.wf_size(), ldsAddrs,
                           perLaneBytes);
    } else if (d.is_load && d.dst_reg_base >= wf.vgpr_alloc().base) {
      uint32_t logicalBase = d.dst_reg_base - wf.vgpr_alloc().base;
      std::vector<uint32_t> registers(d.num_elems);
      for (uint32_t i = 0; i < d.num_elems; ++i) {
        registers[i] = logicalBase + i;
      }
      rs->registerEvent(static_cast<int>(wf.pc),
                        raceemulator::MemoryEventType::GLOBAL_TO_VGPR,
                        std::move(registers), wf.exec());
    }
  }

  if (inst->data()->tag() == SCALAR_MEM) {
    auto &d = *inst->data_as<ScalarMemState>();
    if (d.is_load) {
      uint32_t logicalBase = d.dst_reg_base - wf.sgpr_alloc().base;
      std::vector<uint32_t> registers(d.num_dwords);
      for (uint32_t i = 0; i < d.num_dwords; ++i) {
        registers[i] = logicalBase + i;
      }
      rs->registerEvent(static_cast<int>(wf.pc),
                        raceemulator::MemoryEventType::GLOBAL_TO_SGPR,
                        std::move(registers), wf.exec());
    }
  }
}

void RaceDetectionPlugin::onVgprRead(Wavefront *wf, uint32_t logical_reg,
                                     uint32_t lane) {
  auto it = wavefront_to_race_state_.find(wf);
  if (it != wavefront_to_race_state_.end()) {
    it->second->checkVgprRead(static_cast<int>(logical_reg),
                              static_cast<int>(lane), 0xF);
  }
}

void RaceDetectionPlugin::onSgprRead(Wavefront *wf, uint32_t logical_reg) {
  auto it = wavefront_to_race_state_.find(wf);
  if (it != wavefront_to_race_state_.end()) {
    it->second->checkSgprRead(static_cast<int>(logical_reg));
  }
}

void RaceDetectionPlugin::onWaitcnt(Wavefront *wf, int vmcnt, int lgkmcnt) {
  auto it = wavefront_to_race_state_.find(wf);
  if (it != wavefront_to_race_state_.end()) {
    it->second->dispatch(raceemulator::PendingWaitCount{vmcnt, lgkmcnt});
  }
}

} // namespace amdgpu
} // namespace rocjitsu
