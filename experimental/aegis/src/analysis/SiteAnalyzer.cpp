//===-- SiteAnalyzer.cpp - Instrumentation Site Analysis ---------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/SiteAnalyzer.h"
#include "aegisbit/CoalescingAnalyzer.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/RegisterHelper.h"
#include "aegisbit/RuntimeConfig.h"

#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/raw_ostream.h"

#include <climits>
#include <cstdlib>
#include <string>

using namespace llvm;

namespace aegisbit {

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

static bool destOverlapsVictims(const DecodedInstruction &DI,
                                unsigned V0, unsigned V1, unsigned V2,
                                const Disassembler &Disasm) {
  if (DI.Inst.getNumOperands() == 0) return false;
  const auto &Op = DI.Inst.getOperand(0);
  if (!Op.isReg()) return false;
  unsigned Reg = Op.getReg();
  const auto &MRI = Disasm.getMRI();
  auto check = [&](unsigned R) -> bool {
    if (!RegisterHelper::isVGPR(R)) return false;
    unsigned Idx = RegisterHelper::getVGPRIndex(R);
    return Idx == V0 || Idx == V1 || Idx == V2;
  };
  if (check(Reg)) return true;
  for (llvm::MCRegister Sub : MRI.subregs(Reg))
    if (check(Sub)) return true;
  return false;
}

static std::pair<unsigned, unsigned> parseWaitCntImm(uint16_t Imm) {
  unsigned Vm   = (Imm & 0xF) | (((Imm >> 14) & 0x3) << 4);
  unsigned Lgkm = (Imm >> 8) & 0xF;
  return {Vm, Lgkm};
}

//===----------------------------------------------------------------------===//
// findMemorySites
//===----------------------------------------------------------------------===//

std::vector<InstrumentationSite>
SiteAnalyzer::findMemorySites(const ControlFlowGraph &CFG,
                               uint64_t BaseAddr,
                               Disassembler &Disasm,
                               const ScratchRegisters &Scratch,
                               bool SupportsGPUAtomics) {
  std::vector<InstrumentationSite> Sites;

  const auto &Cfg = RuntimeConfig::getInstance();
  const int MaxLDSSites =
      Cfg.Debug.MaxLDSSites > 0 ? static_cast<int>(Cfg.Debug.MaxLDSSites)
                                 : INT_MAX;
  const int SkipLDSFirst = Cfg.Debug.SkipLDSFirst;
  const int MaxTotalSites =
      Cfg.Debug.MaxSites > 0 ? static_cast<int>(Cfg.Debug.MaxSites) : INT_MAX;
  int LDSSiteCount = 0;

  for (const auto &BB : CFG.BasicBlocks) {
    for (const auto &DI : BB.Instructions) {
      if (DI.Category != InstructionCategory::VMEM &&
          DI.Category != InstructionCategory::LDS)
        continue;
      if (static_cast<int>(Sites.size()) >= MaxTotalSites)
        break;
      if (DI.Category == InstructionCategory::LDS &&
          (LDSSiteCount < SkipLDSFirst || LDSSiteCount >= SkipLDSFirst + MaxLDSSites)) {
        LDSSiteCount++;
        continue;
      }

      std::string Name = Disasm.getInstructionName(DI.Inst);

      // ---- LDS (DS) instruction handling ----
      if (DI.Category == InstructionCategory::LDS) {

        if (Name.find("DS_PERMUTE") != std::string::npos ||
            Name.find("DS_BPERMUTE") != std::string::npos ||
            Name.find("DS_SWIZZLE") != std::string::npos ||
            Name.find("DS_GWS") != std::string::npos ||
            Name.find("DS_APPEND") != std::string::npos ||
            Name.find("DS_CONSUME") != std::string::npos)
          continue;

        bool IsLoad = (Name.find("DS_READ") != std::string::npos);
        bool IsStore = (Name.find("DS_WRITE") != std::string::npos);
        bool IsRTN = (Name.find("_RTN_") != std::string::npos ||
                      (Name.size() >= 4 &&
                       Name.rfind("_RTN") == Name.size() - 4));
        bool IsDual = (Name.find("DS_READ2") != std::string::npos ||
                       Name.find("DS_WRITE2") != std::string::npos);

        unsigned AddrOpIdx;
        if (IsLoad || IsRTN)
          AddrOpIdx = 1;
        else
          AddrOpIdx = 0;

        if (AddrOpIdx >= DI.Inst.getNumOperands() ||
            !DI.Inst.getOperand(AddrOpIdx).isReg())
          continue;

        unsigned AddrReg = DI.Inst.getOperand(AddrOpIdx).getReg();
        if (!RegisterHelper::isVGPR(AddrReg))
          continue;

        InstrumentationSite Site;
        Site.Address = DI.Address;
        Site.Offset = DI.Address - BaseAddr;
        Site.OrigInst = DI.Inst;
        Site.OrigInstSize = DI.Size;
        Site.IsLoad = IsLoad || IsRTN;
        Site.IsGlobal = false;
        Site.AddrVGPRIndex = RegisterHelper::getVGPRIndex(AddrReg);
        Site.Addr64 = false;

        Site.IsDualDS = IsDual;
        unsigned NumOps = DI.Inst.getNumOperands();
        if (IsDual && NumOps >= 2) {
          for (unsigned i = NumOps; i > 0; --i) {
            if (DI.Inst.getOperand(i - 1).isImm()) {
              if (Site.DSOffset1 == 0 && Site.DSOffset0 != 0)
                Site.DSOffset1 = static_cast<uint16_t>(DI.Inst.getOperand(i - 1).getImm());
              else if (Site.DSOffset0 == 0)
                Site.DSOffset0 = static_cast<uint16_t>(DI.Inst.getOperand(i - 1).getImm());
              if (Site.DSOffset0 != 0 && Site.DSOffset1 != 0) break;
            }
          }
          uint32_t ES = CoalescingAnalyzer::inferElemSize(Name);
          Site.DSOffset0 *= ES;
          Site.DSOffset1 *= ES;
        } else {
          for (unsigned i = NumOps; i > 0; --i) {
            if (DI.Inst.getOperand(i - 1).isImm()) {
              Site.DSOffset0 = static_cast<uint16_t>(DI.Inst.getOperand(i - 1).getImm());
              break;
            }
          }
        }

        if (Scratch.HasAccumVGPRs) {
          unsigned S0 = RegisterHelper::getVGPRIndex(Scratch.ScratchVGPR);
          unsigned S1 = RegisterHelper::getVGPRIndex(Scratch.LaneVGPR);
          unsigned S2 = RegisterHelper::getVGPRIndex(Scratch.TempVGPR);
          bool Conflicts = false;
          for (unsigned i = 0; i < DI.Inst.getNumOperands() && !Conflicts; ++i) {
            const auto &Op = DI.Inst.getOperand(i);
            if (!Op.isReg()) continue;
            unsigned Reg = Op.getReg();
            const auto &MRI = Disasm.getMRI();
            auto checkReg = [&](unsigned R) {
              if (RegisterHelper::isVGPR(R)) {
                unsigned Idx = RegisterHelper::getVGPRIndex(R);
                if (Idx == S0 || Idx == S1 || Idx == S2)
                  Conflicts = true;
              }
            };
            checkReg(Reg);
            for (llvm::MCRegister Sub : MRI.subregs(Reg))
              checkReg(Sub);
          }
          if (Conflicts) {
            if (Cfg.Debug.LogLevel >= 1) {
              llvm::errs() << "[aegisbit] Skipping LDS site @ 0x"
                           << llvm::Twine::utohexstr(Site.Offset).str()
                           << " (" << Name
                           << "): operands overlap scratch v"
                           << S0 << "/v" << S1 << "/v" << S2 << "\n";
            }
            LDSSiteCount++;
            continue;
          }
        }

        if (Cfg.Debug.LogLevel >= 1) {
          const auto &MRI2 = Disasm.getMRI();
          llvm::errs() << "[aegisbit] LDS site " << LDSSiteCount
                       << " [" << Name << " @ 0x"
                       << llvm::Twine::utohexstr(Site.Offset).str()
                       << "] vaddr=v" << Site.AddrVGPRIndex
                       << " size=" << Site.OrigInstSize
                       << " ops:";
          for (unsigned i2 = 0; i2 < DI.Inst.getNumOperands(); ++i2) {
            const auto &Op2 = DI.Inst.getOperand(i2);
            if (Op2.isReg())
              llvm::errs() << " " << MRI2.getName(Op2.getReg());
          }
          llvm::errs() << "\n";
        }
        Sites.push_back(std::move(Site));
        LDSSiteCount++;
        continue;
      }

      // ---- VMEM (GLOBAL/BUFFER) instruction handling ----
      bool IsGlobalLoad  = Name.find("GLOBAL_LOAD") != std::string::npos;
      bool IsGlobalStore = Name.find("GLOBAL_STORE") != std::string::npos;
      bool IsBufferLoad  = Name.find("BUFFER_LOAD") != std::string::npos;
      bool IsBufferStore = Name.find("BUFFER_STORE") != std::string::npos;
      if (!IsGlobalLoad && !IsGlobalStore && !IsBufferLoad && !IsBufferStore)
        continue;

      bool IsLoad = IsGlobalLoad || IsBufferLoad;
      bool IsBuffer = IsBufferLoad || IsBufferStore;

      InstrumentationSite Site;
      Site.Address = DI.Address;
      Site.Offset = DI.Address - BaseAddr;
      Site.OrigInst = DI.Inst;
      Site.OrigInstSize = DI.Size;
      Site.IsLoad = IsLoad;
      Site.IsGlobal = true;

      unsigned AddrOpIdx;
      if (IsBuffer)
        AddrOpIdx = 1;
      else
        AddrOpIdx = IsLoad ? 1 : 0;

      if (AddrOpIdx < DI.Inst.getNumOperands() &&
          DI.Inst.getOperand(AddrOpIdx).isReg()) {
        unsigned AddrReg = DI.Inst.getOperand(AddrOpIdx).getReg();
        if (RegisterHelper::isVGPR(AddrReg)) {
          Site.AddrVGPRIndex = RegisterHelper::getVGPRIndex(AddrReg);
          Site.Addr64 = false;
        } else if (!IsBuffer) {
          const auto &MRI = Disasm.getMRI();
          for (llvm::MCRegister Sub : MRI.subregs(AddrReg)) {
            if (RegisterHelper::isVGPR(Sub)) {
              Site.AddrVGPRIndex = RegisterHelper::getVGPRIndex(Sub);
              Site.Addr64 = true;
              break;
            }
          }
        } else {
          continue;
        }
      } else {
        continue;
      }

      if (Scratch.HasAccumVGPRs) {
        unsigned S0 = RegisterHelper::getVGPRIndex(Scratch.ScratchVGPR);
        unsigned S1 = RegisterHelper::getVGPRIndex(Scratch.LaneVGPR);
        unsigned S2 = RegisterHelper::getVGPRIndex(Scratch.TempVGPR);
        bool Conflicts = false;
        for (unsigned i = 0; i < DI.Inst.getNumOperands() && !Conflicts; ++i) {
          const auto &Op = DI.Inst.getOperand(i);
          if (!Op.isReg()) continue;
          unsigned Reg = Op.getReg();
          const auto &MRI = Disasm.getMRI();
          auto checkReg = [&](unsigned R) {
            if (RegisterHelper::isVGPR(R)) {
              unsigned Idx = RegisterHelper::getVGPRIndex(R);
              if (Idx == S0 || Idx == S1 || Idx == S2)
                Conflicts = true;
            }
          };
          checkReg(Reg);
          for (llvm::MCRegister Sub : MRI.subregs(Reg))
            checkReg(Sub);
        }
        if (Conflicts) {
          if (Cfg.Debug.LogLevel >= 1) {
            llvm::errs() << "[aegisbit] Skipping VMEM site @ 0x"
                         << llvm::Twine::utohexstr(Site.Offset).str()
                         << " (" << Name
                         << "): operands overlap scratch v"
                         << S0 << "/v" << S1 << "/v" << S2 << "\n";
          }
          continue;
        }
      }

      if (Cfg.Debug.LogLevel >= 2) {
        const auto &MRI = Disasm.getMRI();
        llvm::errs() << "[aegisbit] VMEM site " << Sites.size()
                     << " [" << Name << " @ 0x"
                     << llvm::Twine::utohexstr(Site.Offset).str()
                     << "] addr=v" << Site.AddrVGPRIndex
                     << (Site.Addr64 ? "(64b)" : "(32b)")
                     << " ops:";
        for (unsigned i = 0; i < DI.Inst.getNumOperands(); ++i) {
          const auto &Op = DI.Inst.getOperand(i);
          if (Op.isReg()) {
            llvm::errs() << " " << MRI.getName(Op.getReg());
          }
        }
        llvm::errs() << "\n";
      }
      Sites.push_back(std::move(Site));
    }
  }

  return Sites;
}

//===----------------------------------------------------------------------===//
// computePreSpillDrainValues
//===----------------------------------------------------------------------===//

void SiteAnalyzer::computePreSpillDrainValues(
    const ControlFlowGraph &CFG,
    std::vector<InstrumentationSite> &Sites,
    const ScratchRegisters &Scratch,
    Disassembler &Disasm) {

  if (!Scratch.NeedsAccVGPRSpill) return;

  unsigned V0 = RegisterHelper::getVGPRIndex(Scratch.ScratchVGPR);
  unsigned V1 = RegisterHelper::getVGPRIndex(Scratch.LaneVGPR);
  unsigned V2 = RegisterHelper::getVGPRIndex(Scratch.TempVGPR);

  constexpr unsigned NoWaitVm   = 63;
  constexpr unsigned NoWaitLgkm = 15;

  for (auto &Site : Sites) {
    const BasicBlock *BB = nullptr;
    for (const auto &B : CFG.BasicBlocks) {
      if (B.Instructions.empty()) continue;
      if (Site.Address >= B.StartAddress && Site.Address < B.EndAddress) {
        BB = &B;
        break;
      }
    }
    if (!BB) {
      Site.PreSpillVmWait   = 0;
      Site.PreSpillLgkmWait = 0;
      continue;
    }

    unsigned VmDist   = 0;
    unsigned LgkmDist = 0;
    int VmQuota   = INT_MAX;
    int LgkmQuota = INT_MAX;
    unsigned ClosestVmVictimDist   = 0;
    unsigned ClosestLgkmVictimDist = 0;

    for (int i = static_cast<int>(BB->Instructions.size()) - 1; i >= 0; --i) {
      const auto &DI = BB->Instructions[i];
      if (DI.Address >= Site.Address) continue;

      std::string Name = Disasm.getInstructionName(DI.Inst);

      if (Name == "S_WAITCNT") {
        if (DI.Inst.getNumOperands() > 0 && DI.Inst.getOperand(0).isImm()) {
          auto [WVm, WLgkm] = parseWaitCntImm(
              static_cast<uint16_t>(DI.Inst.getOperand(0).getImm()));
          VmQuota   = std::min(VmQuota,   static_cast<int>(WVm));
          LgkmQuota = std::min(LgkmQuota, static_cast<int>(WLgkm));
          if (VmQuota <= 0 && LgkmQuota <= 0) break;
        }
        continue;
      }

      if (DI.Category == InstructionCategory::VMEM) {
        if (VmQuota <= 0) continue;
        VmDist++;
        VmQuota--;
        bool IsLoad = (Name.find("LOAD") != std::string::npos) ||
                      (Name.find("ATOMIC") != std::string::npos &&
                       Name.find("RTN") != std::string::npos);
        if (IsLoad && !ClosestVmVictimDist &&
            destOverlapsVictims(DI, V0, V1, V2, Disasm))
          ClosestVmVictimDist = VmDist;
      }

      if (DI.Category == InstructionCategory::LDS) {
        if (LgkmQuota <= 0) continue;
        LgkmDist++;
        LgkmQuota--;
        bool IsRead = (Name.find("DS_READ") != std::string::npos);
        if (IsRead && !ClosestLgkmVictimDist &&
            destOverlapsVictims(DI, V0, V1, V2, Disasm))
          ClosestLgkmVictimDist = LgkmDist;
      }

      if (DI.Category == InstructionCategory::SMEM) {
        if (LgkmQuota <= 0) continue;
        LgkmDist++;
        LgkmQuota--;
      }
    }

    constexpr unsigned MaxUsableVm   = 62;
    constexpr unsigned MaxUsableLgkm = 14;

    if (ClosestVmVictimDist)
      Site.PreSpillVmWait = std::min(ClosestVmVictimDist - 1, MaxUsableVm);
    else if (VmQuota > 0 && VmDist > 0)
      Site.PreSpillVmWait = 0;
    else
      Site.PreSpillVmWait = NoWaitVm;

    if (ClosestLgkmVictimDist)
      Site.PreSpillLgkmWait = std::min(ClosestLgkmVictimDist - 1, MaxUsableLgkm);
    else if (LgkmQuota > 0 && LgkmDist > 0)
      Site.PreSpillLgkmWait = 0;
    else
      Site.PreSpillLgkmWait = NoWaitLgkm;

    if (RuntimeConfig::getInstance().Debug.LogLevel >= 1) {
      llvm::errs() << "[aegisbit] Pre-spill drain @ 0x"
                   << llvm::Twine::utohexstr(Site.Address).str()
                   << ": vmcnt(" << Site.PreSpillVmWait
                   << ") lgkmcnt(" << Site.PreSpillLgkmWait
                   << ") [vmDist=" << VmDist
                   << " lgkmDist=" << LgkmDist
                   << " victimVm=" << ClosestVmVictimDist
                   << " victimLgkm=" << ClosestLgkmVictimDist << "]\n";
    }
  }
}

} // namespace aegisbit
