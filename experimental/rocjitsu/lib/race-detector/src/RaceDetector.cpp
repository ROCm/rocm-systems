// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-emulator/RaceDetector.h"
#include <algorithm>
#include <cassert>
#include <sstream>

namespace raceemulator {

RaceDetector::RaceDetector(int ldsSize, int nWaves, int vgprCount,
                           int sgprCount, Dim3d workgroupId,
                           std::function<void(RaceViolation)> raceHandler)
    : byteWriteCounts(ldsSize, 0), byteReadCounts(ldsSize, 0),
      workgroupId(workgroupId), raceHandler(std::move(raceHandler)) {
  waveRaceStates.reserve(nWaves);
  for (int i = 0; i < nWaves; ++i) {
    waveRaceStates.emplace_back(vgprCount, sgprCount, WaveId{i}, this);
  }
}

WaveRaceState &RaceDetector::getWaveRaceState(int waveIndex) {
  assert(waveIndex >= 0 && waveIndex < static_cast<int>(waveRaceStates.size()));
  return waveRaceStates[waveIndex];
}

void RaceDetector::setProfiler(ProfilerInterface &p) {
  for (auto &wrs : waveRaceStates) {
    wrs.setProfiler(p);
  }
}

EventId RaceDetector::allocateEventId(WaveId waveId, int pc,
                                      MemoryEventType type,
                                      std::vector<uint32_t> registers,
                                      uint64_t execMask, uint8_t byteMask,
                                      IntervalSet ldsIntervals) {
  int id = static_cast<int>(eventRegistry.size());
  bool hasLds = !ldsIntervals.empty();
  eventRegistry.push_back({waveId, pc, type, EventStatus::ACTIVE, byteMask,
                           execMask, std::move(registers),
                           std::move(ldsIntervals)});
  EventId eid{id};
  if (hasLds) {
    const auto &ivs = eventRegistry[id].ldsIntervals;
    if (isToLds(type)) {
      ldsWriteEvents.push_back(eid);
      adjustByteCounts(ivs, byteWriteCounts, +1);
    } else if (type == MemoryEventType::LDS_TO_VGPR) {
      ldsReadEvents.push_back(eid);
      adjustByteCounts(ivs, byteReadCounts, +1);
    }
  }
  return eid;
}

void RaceDetector::markEventWaveComplete(EventId eventId) {
  assert(eventId.value >= 0 &&
         eventId.value < static_cast<int64_t>(eventRegistry.size()));
  assert(eventRegistry[eventId.value].status == EventStatus::ACTIVE);
  eventRegistry[eventId.value].status = EventStatus::WAVE_COMPLETE;
}

void RaceDetector::retireEvent(EventId eventId) {
  const auto &info = eventRegistry[eventId.value];
  if (isToLds(info.type)) {
    removeFromUnorderedList(ldsWriteEvents, eventId);
    adjustByteCounts(info.ldsIntervals, byteWriteCounts, -1);
  } else if (info.type == MemoryEventType::LDS_TO_VGPR) {
    removeFromUnorderedList(ldsReadEvents, eventId);
    adjustByteCounts(info.ldsIntervals, byteReadCounts, -1);
  }
}

void RaceDetector::validateRead(int addr, WaveId wave, int lane,
                                int nBytes) const {
  if (addr < 0 || addr + nBytes > static_cast<int>(byteWriteCounts.size())) {
    return;
  }
  bool anyWrites = false;
  for (int i = 0; i < nBytes; ++i) {
    if (byteWriteCounts[addr + i] > 0) {
      anyWrites = true;
      break;
    }
  }
  if (!anyWrites) {
    return;
  }

  for (EventId eventId : ldsWriteEvents) {
    const auto &info = eventRegistry[eventId.value];
    if (wave == info.waveId && info.status == EventStatus::WAVE_COMPLETE) {
      continue;
    }
    if (info.ldsIntervals.overlapsRange(addr, addr + nBytes)) {
      raceHandler({RaceViolation::Space::LDS, addr, wave.value, lane, false,
                   workgroupId});
    }
  }
}

void RaceDetector::validateWrite(int addr, WaveId wave, int lane,
                                 int nBytes) const {
  if (addr < 0 || addr + nBytes > static_cast<int>(byteReadCounts.size())) {
    return;
  }
  bool anyReads = false;
  for (int i = 0; i < nBytes; ++i) {
    if (byteReadCounts[addr + i] > 0) {
      anyReads = true;
      break;
    }
  }
  if (!anyReads) {
    return;
  }

  for (EventId eventId : ldsReadEvents) {
    const auto &info = eventRegistry[eventId.value];
    if (wave == info.waveId && info.status == EventStatus::WAVE_COMPLETE) {
      continue;
    }
    if (info.ldsIntervals.overlapsRange(addr, addr + nBytes)) {
      raceHandler({RaceViolation::Space::LDS, addr, wave.value, lane, true,
                   workgroupId});
    }
  }
}

void RaceDetector::adjustByteCounts(const IntervalSet &ivs,
                                    std::vector<int> &counts, int delta) {
  int size = static_cast<int>(counts.size());
  for (const auto &iv : ivs) {
    for (int b = iv.start; b < iv.end && b < size; ++b) {
      if (b >= 0)
        counts[b] += delta;
    }
  }
}

std::string RaceDetector::decorateException(
    const RaceViolation &e, int wavePc, WaveRaceState *waveRaceState,
    int numSourceLines,
    std::function<std::string_view(int)> getSourceLine) const {

  auto printCodeBlock = [&](std::ostringstream &oss, int startLine, int endLine,
                            std::span<const int> arrowLines) {
    for (int i = startLine; i <= endLine; ++i) {
      if (i < 0 || i >= numSourceLines) {
        continue;
      }
      bool isArrow = std::find(arrowLines.begin(), arrowLines.end(), i) !=
                     arrowLines.end();
      if (isArrow) {
        oss << i << " --> | " << getSourceLine(i) << "\n";
      } else {
        oss << i << "     | " << getSourceLine(i) << "\n";
      }
    }
  };

  constexpr int nBefore = 1;
  constexpr int nAfter = 1;

  auto printCodeBlocks = [&](std::ostringstream &oss,
                             std::vector<int> eventPcs) {
    std::sort(eventPcs.begin(), eventPcs.end());
    if (eventPcs.empty()) {
      return;
    }
    int blockStart = eventPcs[0] - nBefore;
    int blockEnd = eventPcs[0] + nAfter;
    std::vector<int> arrows = {eventPcs[0]};

    for (size_t i = 1; i < eventPcs.size(); ++i) {
      if (eventPcs[i] - nBefore <= blockEnd + 1) {
        blockEnd = std::max(blockEnd, eventPcs[i] + nAfter);
        arrows.push_back(eventPcs[i]);
      } else {
        printCodeBlock(oss, blockStart, blockEnd, arrows);
        oss << "\n";
        blockStart = eventPcs[i] - nBefore;
        blockEnd = eventPcs[i] + nAfter;
        arrows = {eventPcs[i]};
      }
    }
    printCodeBlock(oss, blockStart, blockEnd, arrows);
    oss << "\n";
  };

  if (e.space == RaceViolation::Space::VGPR) {
    std::ostringstream oss;
    oss << "\nVGPR race detected on line " << wavePc << " (wave " << e.wave
        << ", lane " << e.lane << ") in workgroup (" << workgroupId.x << ","
        << workgroupId.y << "," << workgroupId.z
        << "). Conflicting events:\n\n";

    std::vector<int> eventPcs{wavePc};
    if (waveRaceState) {
      for (EventId evtId : waveRaceState->getVgprMemoryEvents(e.index)) {
        eventPcs.push_back(getEventPc(evtId));
      }
    }
    printCodeBlocks(oss, std::move(eventPcs));
    return oss.str();
  }

  if (e.space == RaceViolation::Space::SGPR) {
    std::ostringstream oss;
    oss << "\nSGPR race detected on line " << wavePc << " (wave " << e.wave
        << ") in workgroup (" << workgroupId.x << "," << workgroupId.y << ","
        << workgroupId.z << "). Conflicting events:\n\n";

    std::vector<int> eventPcs{wavePc};
    if (waveRaceState) {
      for (EventId evtId : waveRaceState->getWaveMemoryEvents()) {
        if (isToSgpr(getEventType(evtId))) {
          for (uint32_t reg : getEventRegisters(evtId)) {
            if (reg == static_cast<uint32_t>(e.index)) {
              eventPcs.push_back(getEventPc(evtId));
              break;
            }
          }
        }
      }
    }
    printCodeBlocks(oss, std::move(eventPcs));
    return oss.str();
  }

  if (e.space == RaceViolation::Space::LDS) {
    std::ostringstream oss;
    oss << "\nLDS race in byte " << e.index << " detected in workgroup ("
        << workgroupId.x << "," << workgroupId.y << "," << workgroupId.z
        << "). Race between a pair in:\n\n";

    struct PcWaveLane {
      int pc;
      int wave;
      int lane;
    };
    std::vector<PcWaveLane> entries{{wavePc, e.wave, e.lane}};

    auto scanEvents = [&](const std::vector<EventId> &events) {
      for (EventId eventId : events) {
        if (getEventIntervals(eventId).contains(e.index)) {
          entries.push_back(
              {getEventPc(eventId), getEventWaveId(eventId).value, -1});
        }
      }
    };
    scanEvents(ldsWriteEvents);
    scanEvents(ldsReadEvents);
    std::sort(entries.begin(), entries.end(),
              [](const PcWaveLane &a, const PcWaveLane &b) {
                return std::tie(a.pc, a.wave) < std::tie(b.pc, b.wave);
              });

    for (const auto &entry : entries) {
      oss << "Wave " << entry.wave;
      if (entry.lane >= 0) {
        oss << " Lane " << entry.lane;
      }
      oss << ":\n";
      std::array<int, 1> arrowLine = {entry.pc};
      printCodeBlock(oss, entry.pc - nBefore, entry.pc + nAfter, arrowLine);
      oss << "\n";
    }
    return oss.str();
  }

  return "\nUnknown race space\n";
}

} // namespace raceemulator
