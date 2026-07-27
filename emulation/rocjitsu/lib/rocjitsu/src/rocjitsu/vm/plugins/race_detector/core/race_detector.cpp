// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include <algorithm>
#include <cassert>
#include <sstream>

namespace rocjitsu::plugins::race_detector {

RaceDetector::RaceDetector(int nWaves, int vgprCount, int sgprCount, Dim3d workgroupId,
                           std::function<void(RaceViolation)> raceHandler)
    : workgroupId(workgroupId), raceHandler(std::move(raceHandler)) {
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

EventId RaceDetector::allocateEventId(WaveId waveId, uint64_t pc, MemoryEventType type,
                                      std::vector<uint32_t> registers, uint64_t execMask,
                                      uint8_t byteMask, IntervalSet ldsIntervals) {
  bool hasLds = !ldsIntervals.empty();
  EventId eid = events_.add(waveId, pc, type, std::move(registers), execMask, byteMask,
                            std::move(ldsIntervals));
  if (hasLds) {
    const auto &ivs = events_.ldsIntervals(eid);
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

void RaceDetector::markEventWaveComplete(EventId eventId) { events_.markComplete(eventId); }

void RaceDetector::retireEvent(EventId eventId) {
  auto type = events_.type(eventId);
  if (isToLds(type)) {
    removeFromUnorderedList(ldsWriteEvents, eventId);
    adjustByteCounts(events_.ldsIntervals(eventId), byteWriteCounts, -1);
  } else if (type == MemoryEventType::LDS_TO_VGPR) {
    removeFromUnorderedList(ldsReadEvents, eventId);
    adjustByteCounts(events_.ldsIntervals(eventId), byteReadCounts, -1);
  }
  events_.markRetired(eventId);
}

void RaceDetector::validateRead(int addr, WaveId wave, int lane, int nBytes) const {
  bool anyWrites = false;
  int limit = static_cast<int>(byteWriteCounts.size());
  int cStart = addr / kCountGranularity;
  int cEnd = (addr + nBytes + kCountGranularity - 1) / kCountGranularity;
  for (int c = cStart; c < cEnd; ++c) {
    if (c < limit && byteWriteCounts[c] > 0) {
      anyWrites = true;
      break;
    }
  }
  if (!anyWrites) {
    return;
  }

  for (EventId eventId : ldsWriteEvents) {
    if (wave == events_.waveId(eventId) && events_.status(eventId) == EventStatus::WAVE_COMPLETE) {
      continue;
    }
    if (events_.ldsIntervals(eventId).overlapsRange(addr, addr + nBytes)) {
      raceHandler({RaceViolation::Space::LDS, addr, wave.value, lane, false, workgroupId});
    }
  }
}

void RaceDetector::validateWrite(int addr, WaveId wave, int lane, int nBytes) const {
  bool anyReads = false;
  int limit = static_cast<int>(byteReadCounts.size());
  int cStart = addr / kCountGranularity;
  int cEnd = (addr + nBytes + kCountGranularity - 1) / kCountGranularity;
  for (int c = cStart; c < cEnd; ++c) {
    if (c < limit && byteReadCounts[c] > 0) {
      anyReads = true;
      break;
    }
  }
  if (!anyReads) {
    return;
  }

  for (EventId eventId : ldsReadEvents) {
    if (wave == events_.waveId(eventId) && events_.status(eventId) == EventStatus::WAVE_COMPLETE) {
      continue;
    }
    if (events_.ldsIntervals(eventId).overlapsRange(addr, addr + nBytes)) {
      raceHandler({RaceViolation::Space::LDS, addr, wave.value, lane, true, workgroupId});
    }
  }
}

void RaceDetector::adjustByteCounts(const IntervalSet &ivs, std::vector<int> &counts, int delta) {
  for (const auto &iv : ivs) {
    int cStart = iv.start / kCountGranularity;
    int cEnd = (iv.end + kCountGranularity - 1) / kCountGranularity;
    if (cEnd > static_cast<int>(counts.size()))
      counts.resize(cEnd, 0);
    for (int c = cStart; c < cEnd; ++c) {
      int byteStart = std::max(iv.start, c * kCountGranularity);
      int byteEnd = std::min(iv.end, (c + 1) * kCountGranularity);
      counts[c] += delta * (byteEnd - byteStart);
    }
  }
}

std::string
RaceDetector::decorateException(const RaceViolation &e, uint64_t wavePc,
                                WaveRaceState *waveRaceState, int numSourceLines,
                                std::function<std::string_view(int)> getSourceLine) const {

  auto printCodeBlock = [&](std::ostringstream &oss, int64_t startLine, int64_t endLine,
                            std::span<const uint64_t> arrowLines) {
    for (int64_t i = startLine; i <= endLine; ++i) {
      if (i < 0 || i >= numSourceLines) {
        continue;
      }
      bool isArrow = std::find(arrowLines.begin(), arrowLines.end(), i) != arrowLines.end();
      if (isArrow) {
        oss << i << " --> | " << getSourceLine(i) << "\n";
      } else {
        oss << i << "     | " << getSourceLine(i) << "\n";
      }
    }
  };

  constexpr int nBefore = 1;
  constexpr int nAfter = 1;

  auto printCodeBlocks = [&](std::ostringstream &oss, std::vector<uint64_t> eventPcs) {
    std::sort(eventPcs.begin(), eventPcs.end());
    if (eventPcs.empty()) {
      return;
    }
    int64_t blockStart = static_cast<int64_t>(eventPcs[0]) - nBefore;
    int64_t blockEnd = static_cast<int64_t>(eventPcs[0]) + nAfter;
    std::vector<uint64_t> arrows = {eventPcs[0]};

    for (size_t i = 1; i < eventPcs.size(); ++i) {
      int64_t pc = static_cast<int64_t>(eventPcs[i]);
      if (pc - nBefore <= blockEnd + 1) {
        blockEnd = std::max(blockEnd, pc + nAfter);
        arrows.push_back(eventPcs[i]);
      } else {
        printCodeBlock(oss, blockStart, blockEnd, arrows);
        oss << "\n";
        blockStart = pc - nBefore;
        blockEnd = pc + nAfter;
        arrows = {eventPcs[i]};
      }
    }
    printCodeBlock(oss, blockStart, blockEnd, arrows);
    oss << "\n";
  };

  if (e.space == RaceViolation::Space::VGPR) {
    std::ostringstream oss;
    oss << "\nVGPR race detected on line " << wavePc << " (wave " << e.wave << ", lane " << e.lane
        << ") in workgroup (" << workgroupId.x << "," << workgroupId.y << "," << workgroupId.z
        << "). Conflicting events:\n\n";

    std::vector<uint64_t> eventPcs{wavePc};
    if (waveRaceState) {
      for (EventId evtId : waveRaceState->getVgprMemoryEvents(e.index)) {
        eventPcs.push_back(events_.pc(evtId));
      }
    }
    printCodeBlocks(oss, std::move(eventPcs));
    return oss.str();
  }

  if (e.space == RaceViolation::Space::SGPR) {
    std::ostringstream oss;
    oss << "\nSGPR race detected on line " << wavePc << " (wave " << e.wave << ") in workgroup ("
        << workgroupId.x << "," << workgroupId.y << "," << workgroupId.z
        << "). Conflicting events:\n\n";

    std::vector<uint64_t> eventPcs{wavePc};
    if (waveRaceState) {
      for (EventId evtId : waveRaceState->getWaveMemoryEvents()) {
        if (isToSgpr(events_.type(evtId))) {
          for (uint32_t reg : events_.registers(evtId)) {
            if (reg == static_cast<uint32_t>(e.index)) {
              eventPcs.push_back(events_.pc(evtId));
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
    oss << "\nLDS race in byte " << e.index << " detected in workgroup (" << workgroupId.x << ","
        << workgroupId.y << "," << workgroupId.z << "). Race between a pair in:\n\n";

    struct PcWaveLane {
      uint64_t pc;
      int wave;
      int lane;
    };
    std::vector<PcWaveLane> entries{{wavePc, e.wave, e.lane}};

    auto scanEvents = [&](const std::vector<EventId> &events) {
      for (EventId eventId : events) {
        if (events_.ldsIntervals(eventId).contains(e.index)) {
          entries.push_back({events_.pc(eventId), events_.waveId(eventId).value, -1});
        }
      }
    };
    scanEvents(ldsWriteEvents);
    scanEvents(ldsReadEvents);
    std::sort(entries.begin(), entries.end(), [](const PcWaveLane &a, const PcWaveLane &b) {
      return std::tie(a.pc, a.wave) < std::tie(b.pc, b.wave);
    });

    for (const auto &entry : entries) {
      oss << "Wave " << entry.wave;
      if (entry.lane >= 0) {
        oss << " Lane " << entry.lane;
      }
      oss << ":\n";
      std::array<uint64_t, 1> arrowLine = {entry.pc};
      printCodeBlock(oss, entry.pc - nBefore, entry.pc + nAfter, arrowLine);
      oss << "\n";
    }
    return oss.str();
  }

  return "\nUnknown race space\n";
}

} // namespace rocjitsu::plugins::race_detector
