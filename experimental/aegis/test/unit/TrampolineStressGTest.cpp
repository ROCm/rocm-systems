//===-- TrampolineStressGTest.cpp - Parameterized Stress Tests ----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Parameterized stress tests that sweep across (KernelSize, NumSites,
/// BaseAddr, PreKernelSpace, ZeroSGPR) and assert universal invariants
/// for every combination.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/TrampolineBridge.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/RegisterHelper.h"

#include "gtest/gtest.h"
#include <cstring>
#include <tuple>

using namespace aegisbit;
using namespace llvm;

namespace {

struct StressParams {
  uint64_t KernelSize;
  uint32_t NumSites;
  uint64_t BaseAddr;
  uint64_t PreKernelSpace;
  bool ZeroSGPR;
  const char *Label;
};

class TrampolineStressTest
    : public ::testing::TestWithParam<StressParams> {
protected:
  void SetUp() override {
    auto DisasmOrErr =
        Disassembler::create("amdgcn-amd-amdhsa", "gfx942", "+wavefrontsize64");
    if (!DisasmOrErr) {
      GTEST_SKIP() << "Cannot create AMDGPU disassembler";
    }
    Disasm = std::move(*DisasmOrErr);
  }

  std::unique_ptr<Disassembler> Disasm;
};

static int16_t extractSBranchDword(const uint8_t *Bytes) {
  uint32_t Word = Bytes[0] | (Bytes[1] << 8)
                | (Bytes[2] << 16) | (Bytes[3] << 24);
  return static_cast<int16_t>(Word & 0xFFFF);
}

TEST_P(TrampolineStressTest, AllInvariantsHold) {
  auto P = GetParam();
  SCOPED_TRACE(P.Label);

  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr) << toString(BridgeOrErr.takeError());

  std::vector<uint8_t> Code(P.KernelSize, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  std::vector<InstrumentationSite> Sites;
  uint64_t Step = ((P.KernelSize - 4) / P.NumSites / 4) * 4;
  if (Step < 4) Step = 4;
  for (uint32_t i = 0; i < P.NumSites; ++i) {
    InstrumentationSite S;
    S.Offset = i * Step;
    if (S.Offset + 4 > P.KernelSize) break;
    S.Address = P.BaseAddr + S.Offset;
    S.OrigInstSize = 4;
    S.IsLoad = true;
    S.IsGlobal = true;
    S.AddrVGPRIndex = 0;
    Sites.push_back(S);
  }

  uint64_t TextEnd = P.BaseAddr + P.KernelSize;
  if (P.ZeroSGPR && !P.PreKernelSpace)
    TextEnd = P.KernelSize; // simple case

  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;

  ScratchRegisters Scratch;
  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;

  if (P.ZeroSGPR) {
    KD.SGPRCount = 104;
    Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);
    Trace.Strategy = PayloadStrategy::OnGpuReduce;
    Trace.SupportsGPUAtomics = true;
  } else {
    KD.SGPRCount = 24;
    Scratch = ScratchRegisters::fromDescriptorInstrumented(KD);
    Trace.Strategy = PayloadStrategy::OnGpuReduce;
  }

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, P.BaseAddr, TextEnd, Sites, Scratch, Trace, P.PreKernelSpace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  auto &R = *ResultOrErr;
  uint32_t NumSitesActual = static_cast<uint32_t>(Sites.size());

  // Invariant 1: Zero drop — ForceAllRelay should ensure 100% coverage.
  EXPECT_EQ(R.PatchedCount, NumSitesActual)
      << "All " << NumSitesActual << " sites should be patched";

  // Invariant 2: No island overlaps (pairwise)
  for (size_t i = 0; i < R.Islands.size(); ++i) {
    uint64_t Ai = R.Islands[i].Offset;
    uint64_t Bi = Ai + R.Islands[i].Bytes.size();
    for (size_t j = i + 1; j < R.Islands.size(); ++j) {
      uint64_t Aj = R.Islands[j].Offset;
      uint64_t Bj = Aj + R.Islands[j].Bytes.size();
      bool Overlaps = (Ai < Bj && Aj < Bi);
      EXPECT_FALSE(Overlaps)
          << "Island " << i << " [0x" << std::hex << Ai << ",0x" << Bi
          << ") overlaps island " << j << " [0x" << Aj << ",0x" << Bj << ")";
    }
  }

  // Invariant 3: No island overlaps kernel
  for (size_t i = 0; i < R.Islands.size(); ++i) {
    uint64_t IslStart = R.Islands[i].Offset;
    uint64_t IslEnd = IslStart + R.Islands[i].Bytes.size();
    bool OverlapsKernel = (IslStart < P.BaseAddr + P.KernelSize &&
                           IslEnd > P.BaseAddr);
    EXPECT_FALSE(OverlapsKernel)
        << "Island " << i << " overlaps kernel code";
  }

  // Invariant 4: Alignment
  for (size_t i = 0; i < R.Islands.size(); ++i) {
    EXPECT_EQ(R.Islands[i].Offset % 256, 0u)
        << "Island " << i << " not 256-byte aligned";
  }

  // Invariant 5: Patch branch lands in an island
  for (size_t i = 0; i < R.Slots.size(); ++i) {
    const auto &Slot = R.Slots[i];
    if (Slot.PatchBytes.size() < 4) continue;

    uint64_t PatchSiteAbs = Slot.OriginalPC;
    uint32_t PatchWord = Slot.PatchBytes[0]
                       | (Slot.PatchBytes[1] << 8)
                       | (Slot.PatchBytes[2] << 16)
                       | (Slot.PatchBytes[3] << 24);
    uint16_t Opcode = PatchWord >> 16;

    uint64_t Target = 0;
    bool CanCheck = false;

    if (Opcode == 0xBF82u) {
      int16_t Dw = static_cast<int16_t>(PatchWord & 0xFFFF);
      Target = PatchSiteAbs + 4 + static_cast<int64_t>(Dw) * 4;
      CanCheck = true;
    } else if ((Opcode & 0xFF80u) == 0xBA80u) {
      int16_t Dw = static_cast<int16_t>(PatchWord & 0xFFFF);
      Target = PatchSiteAbs + 4 + static_cast<int64_t>(Dw) * 4;
      CanCheck = true;
    }

    if (CanCheck) {
      bool LandsInIsland = false;
      for (const auto &Isl : R.Islands) {
        if (Target >= Isl.Offset &&
            Target < Isl.Offset + Isl.Bytes.size()) {
          LandsInIsland = true;
          break;
        }
      }
      EXPECT_TRUE(LandsInIsland)
          << "Slot " << i << " patch branch target 0x" << std::hex
          << Target << " does not land in any island";
    }
  }

  // Invariant 6: Return branch correct (ZeroSGPR direct slots only)
  // Use the forward branch target to identify which island contains each slot.
  if (P.ZeroSGPR) {
    for (size_t i = 0; i < R.Slots.size(); ++i) {
      const auto &Slot = R.Slots[i];
      const auto &TB = Slot.TrampolineBytes;
      if (TB.size() < 4) continue;

      uint32_t LastWord = TB[TB.size() - 4]
                        | (TB[TB.size() - 3] << 8)
                        | (TB[TB.size() - 2] << 16)
                        | (TB[TB.size() - 1] << 24);

      if ((LastWord >> 16) != 0xBF82u) continue;

      // Compute forward branch target to find the correct island
      if (Slot.PatchBytes.size() < 4) continue;
      uint32_t PatchWord = Slot.PatchBytes[0]
                         | (Slot.PatchBytes[1] << 8)
                         | (Slot.PatchBytes[2] << 16)
                         | (Slot.PatchBytes[3] << 24);
      if ((PatchWord >> 16) != 0xBF82u) continue; // skip non-s_branch patches

      int16_t FwdDw = static_cast<int16_t>(PatchWord & 0xFFFF);
      uint64_t FwdTarget = Slot.OriginalPC + 4 +
                           static_cast<int64_t>(FwdDw) * 4;

      for (const auto &Isl : R.Islands) {
        if (FwdTarget >= Isl.Offset &&
            FwdTarget < Isl.Offset + Isl.Bytes.size()) {
          uint64_t SlotAbs = FwdTarget; // Forward branch lands at start of slot
          uint64_t RetBranchPC = SlotAbs + TB.size() - 4;
          int16_t RetDw = extractSBranchDword(TB.data() + TB.size() - 4);
          uint64_t RetTarget = RetBranchPC + 4 +
                               static_cast<int64_t>(RetDw) * 4;
          EXPECT_EQ(RetTarget, Slot.OriginalPC + Slot.DisplacedSize)
              << "Slot " << i << " return branch mismatch";
          break;
        }
      }
    }
  }

  // Invariant 7: Accounting (single-island, non-ZeroSGPR)
  // With shared-body architecture, the island contains shared body + dispatch
  // table + return table. The slot TrampolineBytes only hold dispatch entries.
  if (!P.ZeroSGPR && R.Islands.size() == 1) {
    uint64_t ByteSum = 0;
    for (const auto &Slot : R.Slots)
      ByteSum += Slot.TrampolineBytes.size();
    // Island is >= slot byte sum (shared body + return table add overhead)
    EXPECT_GE(R.Islands[0].Bytes.size(), ByteSum)
        << "Single island byte count should be >= slot byte sum";
  }
}

INSTANTIATE_TEST_SUITE_P(
    Sweep, TrampolineStressTest,
    ::testing::Values(
        StressParams{4096,   10,  0,          0,          false,
                     "tiny_kernel_standard"},
        StressParams{4096,   10,  0,          0,          true,
                     "tiny_kernel_zerospr"},
        StressParams{32768,  200, 0,          0,          true,
                     "medium_kernel_multi_island"},
        StressParams{65536,  500, 200*1024,   200*1024,   true,
                     "large_kernel_bidirectional"},
        StressParams{131072, 100, 0,          0,          false,
                     "huge_kernel_standard_longjumps"},
        StressParams{20480,  300, 300*1024,   300*1024,   true,
                     "medium_kernel_high_offset_relays"},
        StressParams{8192,   50,  0,          0,          true,
                     "moderate_density_zerospr"},
        StressParams{65536,  800, 100*1024,   100*1024,   true,
                     "high_density_all_strategies"}
    ),
    [](const ::testing::TestParamInfo<StressParams> &Info) {
      return std::string(Info.param.Label);
    }
);

} // namespace
