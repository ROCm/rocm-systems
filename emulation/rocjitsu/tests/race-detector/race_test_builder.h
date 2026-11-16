// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// RaceTestBuilder: test helper for race detection logic.
//
// Drives RaceDetector + WaveRaceState directly — no instruction parser,
// no assembly, no emulator.

#pragma once

#include "race_test_instruction.h"
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/core/race_detector.h"
#include "rocjitsu/vm/plugins/race_detector/core/wave_race_state.h"

#include <gtest/gtest.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::plugins::race_detector {

class RaceTestBuilder {
public:
  RaceTestBuilder(
      int numWaves, int vgprs, int sgprs, int waveSize = 64, Dim3d wgId = Dim3d(0),
      CounterCapacities counterCapacities = counterCapacitiesForArch(ROCJITSU_CODE_ARCH_CDNA4))
      : waveSize_(waveSize), defaultExec_(waveSize == 64 ? ~0ULL : (1ULL << waveSize) - 1) {
    detector_ = std::make_unique<RaceDetector>(
        numWaves, vgprs, sgprs, wgId, [this](RaceViolation v) { violations_.push_back(v); },
        counterCapacities, /*ldsSizeBytes=*/64 * 1024);
    for (int w = 0; w < numWaves; ++w) {
      waves_.push_back(&detector_->getWaveRaceState(w));
    }
  }

  // -- Memory events --

  /// Register a global load into VGPRs (tracked by vmcnt).
  void
  globalLoad(int wave, int vgprBase, int numRegs, uint64_t exec = 0, uint8_t byteMask = 0xF,
             MemoryOrderClass memoryOrder = defaultMemoryOrder(MemoryEventType::GLOBAL_TO_VGPR),
             std::optional<amdgpu::WaitCounterType> additionalWaitCounterType = std::nullopt) {
    if (!exec) {
      exec = defaultExec_;
    }
    ASSERT_TRUE(
        waves_[wave]->prepareForCounterIncrement(amdgpu::WaitCounterType::VMCNT).succeeded());
    if (additionalWaitCounterType) {
      ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(*additionalWaitCounterType).succeeded());
    }
    std::vector<uint32_t> regs(numRegs);
    for (int i = 0; i < numRegs; ++i) {
      regs[i] = vgprBase + i;
      waves_[wave]->checkVgprWrite(vgprBase + i, exec, byteMask, memoryOrder);
    }
    waves_[wave]->registerEvent(
        testInstruction(pc_++), MemoryEventType::GLOBAL_TO_VGPR, std::move(regs), exec, byteMask,
        CounterObligations{amdgpu::WaitCounterType::VMCNT, memoryOrder, additionalWaitCounterType},
        memoryOrder);
  }

  /// Register a generic FLAT load whose active lanes resolve uniformly to
  /// global memory. The caller supplies architecture-specific counter
  /// obligations because their domains and ordering vary by target.
  void flatGlobalLoad(int wave, int vgprBase, int numRegs,
                      std::span<const amdgpu::MemoryCounterObligation> counterObligations,
                      uint64_t exec = 0) {
    if (!exec)
      exec = defaultExec_;
    const MemoryOrderClass memoryOrder = memoryOrderForObligations(counterObligations);
    std::vector<uint32_t> registers(numRegs);
    for (int i = 0; i < numRegs; ++i) {
      registers[i] = vgprBase + i;
      waves_[wave]->checkVgprWrite(vgprBase + i, exec, /*byteMask=*/0xF, memoryOrder);
    }
    waves_[wave]->registerEvent(testInstruction(pc_++), MemoryEventType::GLOBAL_TO_VGPR,
                                std::move(registers), exec,
                                /*byteMask=*/0xF, counterObligations, memoryOrder);
  }

  /// Register a Direct-to-LDS global load (tracked by vmcnt).
  /// ldsAddrs is per-active-lane; padded to waveSize internally.
  void globalToLds(int wave, std::vector<uint32_t> ldsAddrs, int bytesPerLane, uint64_t exec = 0) {
    if (!exec) {
      exec = defaultExec_;
    }
    constexpr MemoryOrderClass memoryOrder = MemoryOrderClass::VMEM;
    ASSERT_TRUE(
        waves_[wave]->prepareForCounterIncrement(amdgpu::WaitCounterType::VMCNT).succeeded());
    ldsAddrs.resize(waveSize_, 0);
    // Validate each active lane's write against outstanding reads, as the plugin does.
    for (int lane = 0; lane < waveSize_; ++lane)
      if (exec & (1ULL << lane))
        detector_->validateWrite(static_cast<int>(ldsAddrs[lane]), WaveId{wave}, lane, bytesPerLane,
                                 memoryOrder);
    waves_[wave]->registerLdsEvent(testInstruction(pc_++), MemoryEventType::GLOBAL_TO_LDS,
                                   /*registers=*/{}, exec, waveSize_, ldsAddrs, bytesPerLane,
                                   /*byteMask=*/0xF,
                                   CounterObligations{amdgpu::WaitCounterType::VMCNT, memoryOrder},
                                   memoryOrder);
  }

  /// Register a tensor DMA load into LDS (tracked by tensorcnt). The load
  /// writes every range; it is validated against outstanding LDS reads.
  void tensorLoadToLds(int wave, std::vector<LdsByteRange> ranges) {
    tensorTransfer(wave, MemoryEventType::GLOBAL_TO_LDS, ranges);
  }

  /// Register a tensor DMA store from LDS (tracked by tensorcnt). The store
  /// reads every range; it is validated against outstanding LDS writes.
  void tensorStoreFromLds(int wave, std::vector<LdsByteRange> ranges) {
    tensorTransfer(wave, MemoryEventType::LDS_TO_GLOBAL, ranges);
  }

  /// Register a global store from VGPRs (tracked by vmcnt).
  /// Stores read VGPRs at issue time — no destination registers.
  void globalStore(int wave, uint64_t exec = 0) {
    if (!exec) {
      exec = defaultExec_;
    }
    constexpr MemoryOrderClass memoryOrder = MemoryOrderClass::VMEM;
    ASSERT_TRUE(
        waves_[wave]->prepareForCounterIncrement(amdgpu::WaitCounterType::VMCNT).succeeded());
    waves_[wave]->registerEvent(testInstruction(pc_++), MemoryEventType::VGPR_TO_GLOBAL,
                                /*registers=*/{}, exec, /*byteMask=*/0xF,
                                CounterObligations{amdgpu::WaitCounterType::VMCNT, memoryOrder},
                                memoryOrder);
  }

  /// Register a scalar load into SGPRs with its architecture-specific counter.
  void scalarLoad(int wave, int sgprBase, int numRegs,
                  amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT) {
    const uint8_t increment = numRegs > 1 ? 2 : 1;
    ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(waitCounterType, increment).succeeded());
    const std::array obligations{
        amdgpu::MemoryCounterObligation{waitCounterType, MemoryOrderClass::UNORDERED, increment}};
    waves_[wave]->registerScalarLoad(
        testInstruction(pc_++),
        RegisterRef{RegClass::SGPR, static_cast<uint16_t>(sgprBase), static_cast<uint8_t>(numRegs)},
        defaultExec_, obligations, MemoryOrderClass::UNORDERED);
  }

  /// Register a scalar load into TTMPs with its architecture-specific counter.
  void ttmpLoad(int wave, int ttmpBase, int numRegs,
                amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT) {
    const uint8_t increment = numRegs > 1 ? 2 : 1;
    ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(waitCounterType, increment).succeeded());
    const std::array obligations{
        amdgpu::MemoryCounterObligation{waitCounterType, MemoryOrderClass::UNORDERED, increment}};
    waves_[wave]->registerScalarLoad(
        testInstruction(pc_++),
        RegisterRef{RegClass::TTMP, static_cast<uint16_t>(ttmpBase), static_cast<uint8_t>(numRegs)},
        defaultExec_, obligations, MemoryOrderClass::UNORDERED);
  }

  /// Register a scalar store so partial waits retain counter ordering.
  void scalarStore(int wave,
                   amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT) {
    ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(waitCounterType).succeeded());
    waves_[wave]->registerEvent(testInstruction(pc_++), MemoryEventType::SCALAR_TO_GLOBAL, {},
                                defaultExec_, 0xF,
                                CounterObligations{waitCounterType, MemoryOrderClass::UNORDERED},
                                MemoryOrderClass::UNORDERED);
  }

  /// Apply issue-time backpressure before checking an operation's accesses.
  void prepareCounterIncrement(int wave, amdgpu::WaitCounterType counter) {
    ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(counter).succeeded());
  }

  void prepareMemoryIssue(int wave, const amdgpu::MemoryIssueInfo &info) {
    ASSERT_TRUE(waves_[wave]->prepareForMemoryIssue(info).succeeded());
  }

  // -- Operations on one explicit counter and completion class --
  //
  // Each applies issue-time backpressure for `increment` tokens on `counter`,
  // as the plugin does before an instruction's accesses, then registers one
  // event. They name the counter exactly, so they can exercise split counters
  // (LOADCNT, STORECNT, DSCNT, KMCNT, ASYNCCNT, TENSORCNT) directly.

  /// An operation with no destination register and no LDS bytes, such as a
  /// store to global memory.
  void counterOp(int wave, amdgpu::WaitCounterType counter, MemoryOrderClass order,
                 uint8_t increment = 1) {
    ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(counter, increment).succeeded());
    const std::array obligations{amdgpu::MemoryCounterObligation{counter, order, increment}};
    waves_[wave]->registerEvent(testInstruction(pc_++), MemoryEventType::VGPR_TO_GLOBAL,
                                /*registers=*/{}, defaultExec_, /*byteMask=*/0xF, obligations,
                                order);
  }

  /// A transfer that writes (GLOBAL_TO_LDS) or reads (LDS_TO_GLOBAL) LDS byte
  /// ranges, such as an async or tensor transfer. Its LDS accesses are checked
  /// as unordered, as the plugin checks async and tensor transfers.
  void counterLdsTransfer(int wave, amdgpu::WaitCounterType counter, MemoryOrderClass order,
                          MemoryEventType type, std::vector<LdsByteRange> ranges) {
    ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(counter).succeeded());
    for (const auto &range : ranges) {
      const auto bytes = static_cast<int>(
          detector_->ldsBytesInRange(range.address, range.bytes, /*elementBytes=*/1));
      if (bytes == 0)
        continue;
      const int addr = static_cast<int>(range.address);
      if (type == MemoryEventType::GLOBAL_TO_LDS)
        detector_->validateWrite(addr, WaveId{wave}, /*lane=*/-1, bytes,
                                 MemoryOrderClass::UNORDERED);
      else
        detector_->validateRead(addr, WaveId{wave}, /*lane=*/-1, bytes,
                                MemoryOrderClass::UNORDERED);
    }
    const std::array obligations{amdgpu::MemoryCounterObligation{counter, order}};
    waves_[wave]->registerLdsRangeEvent(testInstruction(pc_++), type, ranges, obligations, order);
  }

  /// Dispatch a wait on one counter, such as s_wait_loadcnt.
  void waitCounter(int wave, amdgpu::WaitCounterType counter, int threshold) {
    PendingWaitCount wait;
    wait.add(counter, threshold);
    waves_[wave]->dispatch(wait);
  }

  /// Register an LDS write and validate against outstanding reads.
  void ldsWrite(int wave, int lane, int addr, int bytes,
                MemoryOrderClass memoryOrder = MemoryOrderClass::LDS,
                amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT,
                std::optional<amdgpu::WaitCounterType> additionalWaitCounterType = std::nullopt) {
    ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(waitCounterType).succeeded());
    if (additionalWaitCounterType) {
      ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(*additionalWaitCounterType).succeeded());
    }
    detector_->validateWrite(addr, WaveId{wave}, lane, bytes, memoryOrder);
    std::vector<uint32_t> ldsAddrs(waveSize_, 0);
    ldsAddrs[lane] = addr;
    uint64_t laneMask = 1ULL << lane;
    waves_[wave]->registerLdsEvent(
        testInstruction(pc_++), MemoryEventType::VGPR_TO_LDS,
        /*registers=*/{}, laneMask, waveSize_, ldsAddrs, bytes,
        /*byteMask=*/0xF,
        CounterObligations{waitCounterType, memoryOrder, additionalWaitCounterType}, memoryOrder);
  }

  /// Register an LDS read and validate against outstanding writes.
  /// byteMask: which bytes of the destination VGPR are written by this load
  /// (0xF=full, 0x3=lo D16, 0xC=hi D16). Used for byte-level race tracking.
  void ldsRead(int wave, int lane, int addr, int bytes, int vgprDst, uint8_t byteMask = 0xF,
               amdgpu::WaitCounterType waitCounterType = amdgpu::WaitCounterType::LGKMCNT,
               MemoryOrderClass memoryOrder = MemoryOrderClass::LDS,
               std::optional<amdgpu::WaitCounterType> additionalWaitCounterType = std::nullopt) {
    ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(waitCounterType).succeeded());
    if (additionalWaitCounterType) {
      ASSERT_TRUE(waves_[wave]->prepareForCounterIncrement(*additionalWaitCounterType).succeeded());
    }
    detector_->validateRead(addr, WaveId{wave}, lane, bytes, memoryOrder);
    std::vector<uint32_t> ldsAddrs(waveSize_, 0);
    ldsAddrs[lane] = addr;
    uint64_t laneMask = 1ULL << lane;
    std::vector<uint32_t> regs = {static_cast<uint32_t>(vgprDst)};
    waves_[wave]->registerLdsEvent(
        testInstruction(pc_++), MemoryEventType::LDS_TO_VGPR, std::move(regs), laneMask, waveSize_,
        ldsAddrs, bytes, byteMask,
        CounterObligations{waitCounterType, memoryOrder, additionalWaitCounterType}, memoryOrder);
  }

  /// Register a CDNA1-4 generic FLAT load whose active lanes resolve uniformly
  /// to LDS. Its unordered VMCNT and LGKMCNT obligations are independent of
  /// that resolved memory route.
  void flatLdsLoad(int wave, int lane, int addr, int bytes, int vgprDst) {
    ldsRead(wave, lane, addr, bytes, vgprDst, /*byteMask=*/0xF, amdgpu::WaitCounterType::VMCNT,
            MemoryOrderClass::UNORDERED, amdgpu::WaitCounterType::LGKMCNT);
  }

  /// Register a CDNA1-4 generic FLAT store whose active lane resolves to LDS.
  void flatLdsStore(int wave, int lane, int addr, int bytes) {
    ldsWrite(wave, lane, addr, bytes, MemoryOrderClass::UNORDERED, amdgpu::WaitCounterType::VMCNT,
             amdgpu::WaitCounterType::LGKMCNT);
  }

  // -- Sync --

  /// Dispatch s_waitcnt. -1 means "don't change this counter".
  void waitcnt(int wave, int vmcnt = -1, int lgkmcnt = -1) {
    PendingWaitCount wait;
    wait.add(amdgpu::WaitCounterType::VMCNT, vmcnt);
    wait.add(amdgpu::WaitCounterType::LGKMCNT, lgkmcnt);
    waves_[wave]->dispatch(wait);
  }

  void waitKmcnt(int wave, int kmcnt) {
    PendingWaitCount wait;
    wait.add(amdgpu::WaitCounterType::KMCNT, kmcnt);
    waves_[wave]->dispatch(wait);
  }

  void waitTensorcnt(int wave, int tensorcnt) {
    PendingWaitCount wait;
    wait.add(amdgpu::WaitCounterType::TENSORCNT, tensorcnt);
    waves_[wave]->dispatch(wait);
  }

  /// Barrier: flush all waves' barrier-pending events (simulates s_barrier).
  void barrier() {
    for (auto *w : waves_) {
      w->flushBarrierPendingEvents();
    }
  }

  // -- Trigger race checks --

  /// Check a VGPR read for races against pending loads.
  /// byteMask selects which bytes of the 4-byte register are read:
  ///   0xF = all 4 bytes (default, full 32-bit read)
  ///   0x3 = lo 2 bytes (D16 lo half)
  ///   0xC = hi 2 bytes (D16 hi half)
  /// A race fires only when the pending load's bytes overlap the read's bytes.
  void checkVgprRead(int wave, int reg, int lane, uint8_t byteMask = 0xF) {
    waves_[wave]->checkVgprReadLanes(reg, uint64_t{1} << lane, byteMask);
  }

  void checkVgprReadLanes(int wave, int reg, uint64_t laneMask, uint8_t byteMask = 0xF) {
    waves_[wave]->checkVgprReadLanes(reg, laneMask, byteMask);
  }

  void checkVgprWrite(int wave, int reg, int lane, uint8_t byteMask = 0xF) {
    waves_[wave]->checkVgprWriteLanes(reg, uint64_t{1} << lane, byteMask);
  }

  void checkVgprWriteLanes(int wave, int reg, uint64_t laneMask, uint8_t byteMask = 0xF) {
    waves_[wave]->checkVgprWriteLanes(reg, laneMask, byteMask);
  }

  void checkSgprRead(int wave, int reg, int width = 1) {
    waves_[wave]->checkScalarRead(
        RegisterRef{RegClass::SGPR, static_cast<uint16_t>(reg), static_cast<uint8_t>(width)});
  }

  void checkSgprWrite(int wave, int reg, int width = 1) {
    waves_[wave]->checkScalarWrite(
        RegisterRef{RegClass::SGPR, static_cast<uint16_t>(reg), static_cast<uint8_t>(width)});
  }

  void checkTtmpRead(int wave, int reg) {
    waves_[wave]->checkScalarRead(RegisterRef{RegClass::TTMP, static_cast<uint16_t>(reg), 1});
  }

  void checkTtmpWrite(int wave, int reg, int width = 1) {
    waves_[wave]->checkScalarWrite(
        RegisterRef{RegClass::TTMP, static_cast<uint16_t>(reg), static_cast<uint8_t>(width)});
  }

  void checkLdsRead(int wave, int lane, int addr, int bytes,
                    MemoryOrderClass memoryOrder = MemoryOrderClass::LDS) {
    detector_->validateRead(addr, WaveId{wave}, lane, bytes, memoryOrder);
  }

  void checkLdsWrite(int wave, int lane, int addr, int bytes,
                     MemoryOrderClass memoryOrder = MemoryOrderClass::LDS) {
    detector_->validateWrite(addr, WaveId{wave}, lane, bytes, memoryOrder);
  }

  // -- Results --

  bool hasRace() const { return !violations_.empty(); }
  int raceCount() const { return static_cast<int>(violations_.size()); }
  const std::vector<RaceViolation> &violations() const { return violations_; }

  bool hasVgprRace(int reg) const {
    for (const auto &v : violations_) {
      if (v.space == RaceViolation::Space::VGPR && v.index == reg) {
        return true;
      }
    }
    return false;
  }

  bool hasSgprRace(int reg) const {
    for (const auto &v : violations_) {
      if (v.space == RaceViolation::Space::SGPR && v.index == reg) {
        return true;
      }
    }
    return false;
  }

  bool hasTtmpRace(int reg) const {
    for (const auto &v : violations_) {
      if (v.space == RaceViolation::Space::TTMP && v.index == reg)
        return true;
    }
    return false;
  }

  bool hasLdsRace(int addr) const {
    for (const auto &v : violations_) {
      if (v.space == RaceViolation::Space::LDS && v.index == addr) {
        return true;
      }
    }
    return false;
  }

  void clearViolations() { violations_.clear(); }

  const EventRegistry &events() const { return detector_->events(); }
  const RaceDetector &detector() const { return *detector_; }

private:
  /// A transfer has no lanes; its violations report lane -1. Validates and
  /// registers the same bytes as the plugin's tensor DMA hook.
  void tensorTransfer(int wave, MemoryEventType type, std::span<const LdsByteRange> ranges) {
    // Completion is in issue order (TENSOR); LDS accesses are checked as
    // unordered, a conservative rule, as in the plugin.
    constexpr MemoryOrderClass memoryOrder = MemoryOrderClass::TENSOR;
    constexpr MemoryOrderClass ldsAccessOrder = MemoryOrderClass::UNORDERED;
    for (const auto &range : ranges) {
      const auto bytes = static_cast<int>(
          detector_->ldsBytesInRange(range.address, range.bytes, /*elementBytes=*/1));
      if (bytes == 0)
        continue;
      const int addr = static_cast<int>(range.address);
      if (type == MemoryEventType::GLOBAL_TO_LDS)
        detector_->validateWrite(addr, WaveId{wave}, /*lane=*/-1, bytes, ldsAccessOrder);
      else
        detector_->validateRead(addr, WaveId{wave}, /*lane=*/-1, bytes, ldsAccessOrder);
    }
    const std::array obligations{
        amdgpu::MemoryCounterObligation{amdgpu::WaitCounterType::TENSORCNT, memoryOrder}};
    waves_[wave]->registerLdsRangeEvent(testInstruction(pc_++), type, ranges, obligations,
                                        memoryOrder);
  }

  std::unique_ptr<RaceDetector> detector_;
  std::vector<WaveRaceState *> waves_;
  std::vector<RaceViolation> violations_;
  int waveSize_;
  uint64_t defaultExec_;
  int pc_ = 0; // Auto-incrementing fake PC for event registration.
};

} // namespace rocjitsu::plugins::race_detector
