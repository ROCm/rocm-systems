//===-- TrampolineBridge.cpp - Trampoline Bridge Orchestrator ----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Thin coordinator for the trampoline pipeline. The heavy lifting lives in
/// the per-strategy files under src/transform/:
///   SharedBodyStrategy.cpp         — s_call_b64 shared-body
///   SwapPCSharedBodyStrategy.cpp   — s_swappc_b64 shared-body
///   AdaptiveStrategy.cpp           — per-site direct / relay picker
///
/// This file is responsible for:
///   - creating the ISAEncoder
///   - implementing the empty trampoline path
///   - resolving Plan + Scratch into strategy inputs
///   - static passthroughs to SiteAnalyzer (API compatibility shims)
///
//===----------------------------------------------------------------------===//

#include "aegisbit/TrampolineBridge.h"

#include "aegisbit/Disassembler.h"
#include "aegisbit/ISAEncoder.h"
#include "aegisbit/RegisterHelper.h"
#include "aegisbit/SiteAnalyzer.h"
#include "aegisbit/TrampolineEmitter.h"

#include "TrampolineStrategy.h"

using namespace llvm;

namespace aegisbit {

// Strategy factories implemented in their respective translation units.
std::unique_ptr<TrampolineStrategy> createSharedBodyStrategy();
std::unique_ptr<TrampolineStrategy> createSwapPCSharedBodyStrategy();
std::unique_ptr<TrampolineStrategy> createAdaptiveStrategy();

TrampolineStrategy::~TrampolineStrategy() = default;

std::unique_ptr<TrampolineStrategy>
createStrategyForPlan(const InstrumentationPlan &Plan) {
  switch (Plan.Jump) {
  case JumpStrategy::SharedBody:
    return createSharedBodyStrategy();
  case JumpStrategy::SwapPCSharedBody:
    return createSwapPCSharedBodyStrategy();
  default:
    return createAdaptiveStrategy();
  }
}

TrampolineBridge::~TrampolineBridge() = default;

//===----------------------------------------------------------------------===//
// Factory
//===----------------------------------------------------------------------===//

Expected<std::unique_ptr<TrampolineBridge>>
TrampolineBridge::create(StringRef GPUArch, Disassembler &Disasm) {
  auto Bridge = std::unique_ptr<TrampolineBridge>(new TrampolineBridge());
  Bridge->Arch = GPUArch.str();

  auto EncOrErr = ISAEncoder::create(GPUArch, Disasm);
  if (!EncOrErr)
    return EncOrErr.takeError();
  Bridge->Enc = std::move(*EncOrErr);

  return Bridge;
}

//===----------------------------------------------------------------------===//
// Static delegates to SiteAnalyzer (API compatibility)
//===----------------------------------------------------------------------===//

std::vector<InstrumentationSite>
TrampolineBridge::findMemorySites(const ControlFlowGraph &CFG,
                                   uint64_t BaseAddr, Disassembler &Disasm,
                                   const ScratchRegisters &Scratch,
                                   bool SupportsGPUAtomics) {
  return SiteAnalyzer::findMemorySites(CFG, BaseAddr, Disasm, Scratch,
                                        SupportsGPUAtomics);
}

void TrampolineBridge::computePreSpillDrainValues(
    const ControlFlowGraph &CFG, std::vector<InstrumentationSite> &Sites,
    const ScratchRegisters &Scratch, Disassembler &Disasm) {
  SiteAnalyzer::computePreSpillDrainValues(CFG, Sites, Scratch, Disasm);
}

//===----------------------------------------------------------------------===//
// buildEmpty
//===----------------------------------------------------------------------===//

Expected<BridgeResult>
TrampolineBridge::buildEmpty(ArrayRef<uint8_t> Code, uint64_t BaseAddr,
                              uint64_t TextSectionSize,
                              const std::vector<InstrumentationSite> &Sites,
                              const ScratchRegisters &Scratch,
                              uint64_t PreKernelSpace,
                              const OccupiedRanges &Occupied) {
  (void)PreKernelSpace;
  BridgeResult Result;
  if (Sites.empty())
    return Result;

  unsigned SGPRIndex = Scratch.ReturnAddrSGPR - RegisterHelper::SGPR_BASE;
  auto PairOrErr = Enc->resolveSGPRPair(SGPRIndex);
  if (!PairOrErr)
    return PairOrErr.takeError();
  unsigned RetAddrSGPRPair = *PairOrErr;

  uint64_t IslandStartAbsolute = (TextSectionSize + 255) & ~255ULL;
  for (const auto &R : Occupied) {
    if (IslandStartAbsolute >= R.first && IslandStartAbsolute < R.second)
      IslandStartAbsolute = (R.second + 255) & ~255ULL;
  }
  uint64_t IslandCursor = 0;

  for (const auto &Site : Sites) {
    TrampolineSlot Slot;
    Slot.OriginalPC = Site.Address;
    Slot.DisplacedSize = Site.OrigInstSize;
    Slot.TrampolineOffset = IslandCursor;

    uint64_t PatchSiteAbs = BaseAddr + Site.Offset;
    uint64_t TrampolineAbs = IslandStartAbsolute + IslandCursor;

    int64_t ForwardByteOffset = static_cast<int64_t>(TrampolineAbs) -
                                static_cast<int64_t>(PatchSiteAbs);
    int64_t BranchToDword = (ForwardByteOffset - 4) / 4;
    bool NeedLongJump = (BranchToDword < -32768 || BranchToDword > 32767);

    if (!NeedLongJump) {
      auto CallBytes = Enc->encodeSCall(RetAddrSGPRPair,
                                        static_cast<int16_t>(BranchToDword));
      if (!CallBytes)
        return CallBytes.takeError();
      Slot.PatchBytes = std::move(*CallBytes);
      if (Site.OrigInstSize > 4) {
        auto NopBytes = Enc->encodeNop();
        if (!NopBytes)
          return NopBytes.takeError();
        Slot.PatchBytes.insert(Slot.PatchBytes.end(), NopBytes->begin(),
                                NopBytes->end());
      }
    } else {
      Slot.UsedLongJump = true;
      Result.LongJumpCount++;
      auto LJBytes = Enc->encodeLongJump(RetAddrSGPRPair, ForwardByteOffset);
      if (!LJBytes)
        return LJBytes.takeError();
      Slot.PatchBytes = std::move(*LJBytes);
    }

    Slot.TrampolineBytes.insert(Slot.TrampolineBytes.end(),
                                 Code.data() + Site.Offset,
                                 Code.data() + Site.Offset + Site.OrigInstSize);

    auto RetBytes = Enc->encodeSetPC(RetAddrSGPRPair);
    if (!RetBytes)
      return RetBytes.takeError();
    Slot.TrampolineBytes.insert(Slot.TrampolineBytes.end(), RetBytes->begin(),
                                 RetBytes->end());

    IslandCursor += Slot.TrampolineBytes.size();
    Result.Slots.push_back(std::move(Slot));
    Result.PatchedCount++;
  }

  TrampolineIsland Isl;
  Isl.Offset = IslandStartAbsolute;
  for (const auto &Slot : Result.Slots)
    Isl.Bytes.insert(Isl.Bytes.end(), Slot.TrampolineBytes.begin(),
                      Slot.TrampolineBytes.end());
  if (!Isl.Bytes.empty())
    Result.Islands.push_back(std::move(Isl));

  return Result;
}

//===----------------------------------------------------------------------===//
// buildInstrumented (Plan-aware dispatch)
//===----------------------------------------------------------------------===//

Expected<BridgeResult> TrampolineBridge::buildInstrumented(
    ArrayRef<uint8_t> Code, uint64_t BaseAddr, uint64_t TextSectionSize,
    const std::vector<InstrumentationSite> &Sites,
    const InstrumentationPlan &Plan, const ScratchRegisters &Scratch,
    const TraceConfig &Trace, uint64_t PreKernelSpace,
    const OccupiedRanges &Occupied) {
  BridgeResult Result;
  if (Sites.empty())
    return Result;

  unsigned RetAddrSGPRPair = 0;
  if (Plan.Register != RegisterMode::ZeroSGPR) {
    unsigned SGPRIndex = Scratch.ReturnAddrSGPR - RegisterHelper::SGPR_BASE;
    auto PairOrErr = Enc->resolveSGPRPair(SGPRIndex);
    if (!PairOrErr)
      return PairOrErr.takeError();
    RetAddrSGPRPair = *PairOrErr;
  }

  auto EmitterOrErr = TrampolineEmitter::create(*Enc, Arch, Trace);
  if (!EmitterOrErr)
    return EmitterOrErr.takeError();
  auto &Emitter = **EmitterOrErr;

  BridgeInputs In;
  In.Enc = Enc.get();
  In.Emitter = &Emitter;
  In.Code = Code;
  In.BaseAddr = BaseAddr;
  In.TextSectionSize = TextSectionSize;
  In.Sites = &Sites;
  In.Plan = &Plan;
  In.Scratch = &Scratch;
  In.Trace = &Trace;
  In.RetAddrSGPRPair = RetAddrSGPRPair;
  In.PreKernelSpace = PreKernelSpace;
  In.Occupied = &Occupied;

  auto Strategy = createStrategyForPlan(Plan);
  return Strategy->build(In);
}

//===----------------------------------------------------------------------===//
// buildInstrumented (compatibility overload — derives Plan from Scratch/Trace)
//===----------------------------------------------------------------------===//

Expected<BridgeResult> TrampolineBridge::buildInstrumented(
    ArrayRef<uint8_t> Code, uint64_t BaseAddr, uint64_t TextSectionSize,
    const std::vector<InstrumentationSite> &Sites,
    const ScratchRegisters &Scratch, const TraceConfig &Trace,
    uint64_t PreKernelSpace, const OccupiedRanges &Occupied) {
  InstrumentationPlan Plan;
  Plan.Instrumented = true;
  Plan.Register = Scratch.ZeroSGPR ? RegisterMode::ZeroSGPR
                                    : RegisterMode::StandardScratch;
  Plan.Payload = Trace.Strategy;
  Plan.SupportsGPUAtomics = Trace.SupportsGPUAtomics;
  if (Scratch.UseSwapPC) {
    Plan.Jump = JumpStrategy::SwapPCSharedBody;
  } else if (Plan.Register == RegisterMode::StandardScratch &&
             Plan.Payload == PayloadStrategy::OnGpuReduce) {
    Plan.Jump = JumpStrategy::SharedBody;
  } else if (Plan.Register == RegisterMode::ZeroSGPR) {
    Plan.Jump = JumpStrategy::Adaptive;
  } else {
    Plan.Jump = JumpStrategy::Direct;
  }

  auto ValidatedPlanOrErr = validateInstrumentationPlan(Plan);
  if (!ValidatedPlanOrErr)
    return ValidatedPlanOrErr.takeError();

  return buildInstrumented(Code, BaseAddr, TextSectionSize, Sites,
                            *ValidatedPlanOrErr, Scratch, Trace,
                            PreKernelSpace, Occupied);
}

} // namespace aegisbit
