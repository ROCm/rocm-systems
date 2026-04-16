//===-- TrampolineBridgeGTest.cpp - Trampoline Bridge Tests ------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tests for TrampolineBridge using the above-the-count register API.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/TrampolineBridge.h"
#include "aegisbit/CFGBuilder.h"
#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/RegisterHelper.h"
#include "fixtures/gemm_gfx950_elf.h"

#include "gtest/gtest.h"
#include <cstring>

using namespace aegisbit;
using namespace llvm;

namespace {

class TrampolineBridgeTest : public ::testing::Test {
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

TEST_F(TrampolineBridgeTest, CreateSuccess) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr) << toString(BridgeOrErr.takeError());
}

TEST_F(TrampolineBridgeTest, BuildEmptyNoSites) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  std::vector<uint8_t> Code = {
      0x00, 0x00, 0x80, 0xBF, // s_nop 0
      0x00, 0x00, 0x81, 0xBF, // s_endpgm
  };

  std::vector<InstrumentationSite> Sites;
  KernelDescriptor KD{};
  KD.VGPRCount = 4;
  KD.SGPRCount = 8;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptor(KD);

  auto ResultOrErr = (*BridgeOrErr)->buildEmpty(Code, 0, Code.size(), Sites, Scratch);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  EXPECT_EQ(ResultOrErr->PatchedCount, 0u);
  EXPECT_TRUE(ResultOrErr->Islands.empty());
}

TEST_F(TrampolineBridgeTest, BuildEmptyWithSite) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  std::vector<uint8_t> Code(32, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00;
    Code[i + 1] = 0x00;
    Code[i + 2] = 0x80;
    Code[i + 3] = 0xBF;
  }

  InstrumentationSite Site;
  Site.Address = 8;
  Site.Offset = 8;
  Site.OrigInstSize = 4;
  Site.IsLoad = true;
  Site.IsGlobal = true;
  Site.AddrVGPRIndex = 0;
  std::vector<InstrumentationSite> Sites = {Site};

  KernelDescriptor KD{};
  KD.VGPRCount = 4;
  KD.SGPRCount = 8;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptor(KD);

  auto ResultOrErr = (*BridgeOrErr)->buildEmpty(Code, 0, Code.size(), Sites, Scratch);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  EXPECT_EQ(ResultOrErr->PatchedCount, 1u);
  ASSERT_FALSE(ResultOrErr->Islands.empty());
  EXPECT_GT(ResultOrErr->Islands[0].Bytes.size(), 0u);
  EXPECT_EQ(ResultOrErr->Islands[0].Offset % 256, 0u);

  ASSERT_EQ(ResultOrErr->Slots.size(), 1u);
  const auto &Slot = ResultOrErr->Slots[0];
  EXPECT_EQ(Slot.OriginalPC, 8u);
  EXPECT_EQ(Slot.DisplacedSize, 4u);
  EXPECT_EQ(Slot.PatchBytes.size(), 4u);
}

TEST_F(TrampolineBridgeTest, ExecBufferSizeConstant) {
  EXPECT_EQ(TrampolineBridge::ExecBufferSize, 64ULL * 1024 * 1024);
}

//===----------------------------------------------------------------------===//
// buildEmpty with multiple sites
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, BuildEmptyMultipleSites) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  // 128 bytes of s_nop instructions (32 instructions × 4 bytes)
  std::vector<uint8_t> Code(128, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  std::vector<InstrumentationSite> Sites;
  for (uint64_t Off = 8; Off < 40; Off += 8) {
    InstrumentationSite S;
    S.Address = Off;
    S.Offset = Off;
    S.OrigInstSize = 4;
    S.IsLoad = true;
    S.IsGlobal = true;
    S.AddrVGPRIndex = 0;
    Sites.push_back(S);
  }
  ASSERT_EQ(Sites.size(), 4u);

  KernelDescriptor KD{};
  KD.VGPRCount = 4;
  KD.SGPRCount = 8;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptor(KD);

  auto ResultOrErr = (*BridgeOrErr)->buildEmpty(Code, 0, Code.size(), Sites, Scratch);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  EXPECT_EQ(ResultOrErr->PatchedCount, 4u);
  EXPECT_EQ(ResultOrErr->Slots.size(), 4u);
  ASSERT_FALSE(ResultOrErr->Islands.empty());
  EXPECT_GT(ResultOrErr->Islands[0].Bytes.size(), 0u);

  // Each slot should have unique trampoline offsets
  for (uint32_t i = 1; i < ResultOrErr->Slots.size(); ++i) {
    EXPECT_GT(ResultOrErr->Slots[i].TrampolineOffset,
              ResultOrErr->Slots[i - 1].TrampolineOffset)
        << "Trampoline offsets must be monotonically increasing";
  }
}

//===----------------------------------------------------------------------===//
// buildEmpty with 8-byte instruction: verify NOP padding
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, BuildEmptyEightByteInstGetsPadded) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  // buffer_load_dword v1, v0, s[0:3], 0 offen  (8 bytes)
  std::vector<uint8_t> BufLoad = {0x00, 0x10, 0x50, 0xE0,
                                   0x00, 0x01, 0x00, 0x80};
  // Pad with s_nop for context
  std::vector<uint8_t> Code(64, 0);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }
  // Place buffer_load at offset 8
  std::memcpy(Code.data() + 8, BufLoad.data(), 8);

  InstrumentationSite Site;
  Site.Address = 8;
  Site.Offset = 8;
  Site.OrigInstSize = 8;
  Site.IsLoad = true;
  Site.IsGlobal = true;
  Site.AddrVGPRIndex = 0;

  KernelDescriptor KD{};
  KD.VGPRCount = 4;
  KD.SGPRCount = 8;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptor(KD);

  auto ResultOrErr =
      (*BridgeOrErr)->buildEmpty(Code, 0, Code.size(), {Site}, Scratch);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  ASSERT_EQ(ResultOrErr->Slots.size(), 1u);
  const auto &Slot = ResultOrErr->Slots[0];

  // Patch should be s_call_b64 (4 bytes) + s_nop (4 bytes) = 8 bytes
  EXPECT_EQ(Slot.PatchBytes.size(), 8u)
      << "8-byte instruction patch should be s_call_b64 + s_nop = 8 bytes";

  // Trampoline should contain the displaced instruction (8 bytes) + s_setpc (4 bytes)
  EXPECT_GE(Slot.TrampolineBytes.size(), 12u);

  // Verify displaced instruction bytes are preserved in the trampoline
  std::vector<uint8_t> DisplacedInTrampo(
      Slot.TrampolineBytes.begin(),
      Slot.TrampolineBytes.begin() + 8);
  EXPECT_EQ(DisplacedInTrampo, BufLoad)
      << "Displaced instruction bytes must be preserved exactly";
}

//===----------------------------------------------------------------------===//
// Zero-SGPR mode: forward overflow stops instrumentation
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, ZeroSGPRForwardOverflowStopsInstrumentation) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  // Small kernel (256 bytes)
  std::vector<uint8_t> Code(256, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  InstrumentationSite Site;
  Site.Address = 0;
  Site.Offset = 0;
  Site.OrigInstSize = 4;
  Site.IsLoad = true;
  Site.IsGlobal = true;
  Site.AddrVGPRIndex = 0;

  // Set up zero-SGPR scratch registers
  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.SGPRCount = 104;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);
  ASSERT_TRUE(Scratch.ZeroSGPR);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;
  Trace.Strategy    = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  // TextSectionSize = 256 KB → island starts at ~256 KB.
  // s_branch range is ±128 KB (±32767 dwords), so from offset 0 the island
  // is unreachable and buildInstrumented should stop with PatchedCount = 0.
  uint64_t TextSize = 256 * 1024;
  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, TextSize, {Site}, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  EXPECT_EQ(ResultOrErr->PatchedCount, 0u)
      << "Zero-SGPR mode should produce 0 patches when island is beyond "
         "s_branch range";
  EXPECT_TRUE(ResultOrErr->Slots.empty());
  EXPECT_TRUE(ResultOrErr->Islands.empty());
}

//===----------------------------------------------------------------------===//
// Zero-SGPR mode: within range succeeds
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, ZeroSGPRWithinRangeSucceeds) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  std::vector<uint8_t> Code(256, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  InstrumentationSite Site;
  Site.Address = 0;
  Site.Offset = 0;
  Site.OrigInstSize = 4;
  Site.IsLoad = true;
  Site.IsGlobal = true;
  Site.AddrVGPRIndex = 0;

  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.SGPRCount = 104;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;
  Trace.Strategy    = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  // TextSize = Code.size() so island is placed at 256 bytes (close)
  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, Code.size(), {Site}, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  EXPECT_EQ(ResultOrErr->PatchedCount, 1u)
      << "Zero-SGPR mode should succeed when island is within s_branch range";
  ASSERT_EQ(ResultOrErr->Slots.size(), 1u);
  ASSERT_FALSE(ResultOrErr->Islands.empty());
  EXPECT_FALSE(ResultOrErr->Islands[0].Bytes.empty());

  // The patch site should use s_branch (4 bytes), not s_call_b64
  const auto &Slot = ResultOrErr->Slots[0];
  EXPECT_EQ(Slot.PatchBytes.size(), 4u)
      << "Zero-SGPR patch for 4-byte instruction should be exactly 4 bytes (s_branch)";
  EXPECT_FALSE(Slot.UsedLongJump)
      << "Zero-SGPR mode never uses long-jump (s_branch only)";

  // Verify the trampoline ends with s_branch (return jump)
  // s_branch encoding: 0xBF82xxxx (top 16 bits = 0xBF82)
  size_t TB_size = Slot.TrampolineBytes.size();
  ASSERT_GE(TB_size, 4u);
  uint32_t LastWord = Slot.TrampolineBytes[TB_size - 4]
                    | (Slot.TrampolineBytes[TB_size - 3] << 8)
                    | (Slot.TrampolineBytes[TB_size - 2] << 16)
                    | (Slot.TrampolineBytes[TB_size - 1] << 24);
  EXPECT_EQ(LastWord >> 16, 0xBF82u >> 0)
      << "Zero-SGPR trampoline should end with s_branch (return), got 0x"
      << std::hex << LastWord;
}

//===----------------------------------------------------------------------===//
// buildEmpty: island alignment is 256-byte
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, IslandAlignment) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  // Test with various code sizes (must be 4-byte aligned) for 256-byte alignment
  for (uint64_t CodeSize : {64u, 128u, 256u, 512u, 1024u, 4096u}) {
    std::vector<uint8_t> Code(CodeSize, 0x00);
    for (size_t i = 0; i + 3 < Code.size(); i += 4) {
      Code[i + 0] = 0x00; Code[i + 1] = 0x00;
      Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
    }

    InstrumentationSite Site;
    Site.Address = 0;
    Site.Offset = 0;
    Site.OrigInstSize = 4;
    Site.IsLoad = true;
    Site.IsGlobal = true;
    Site.AddrVGPRIndex = 0;

    KernelDescriptor KD{};
    KD.VGPRCount = 4;
    KD.SGPRCount = 8;
    ScratchRegisters Scratch = ScratchRegisters::fromDescriptor(KD);

    auto R = (*BridgeOrErr)->buildEmpty(Code, 0, Code.size(), {Site}, Scratch);
    ASSERT_TRUE(!!R) << toString(R.takeError());

    ASSERT_FALSE(R->Islands.empty());
    EXPECT_EQ(R->Islands[0].Offset % 256, 0u)
        << "Island offset must be 256-byte aligned for CodeSize=" << CodeSize
        << ", got offset=" << R->Islands[0].Offset;
  }
}

//===----------------------------------------------------------------------===//
// buildInstrumented: standard mode (with SGPRs) basic contract
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, InstrumentedStandardModePatchesSite) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  // buffer_load_dword v1, v0, s[0:3], 0 offen
  std::vector<uint8_t> BufLoad = {0x00, 0x10, 0x50, 0xE0,
                                   0x00, 0x01, 0x00, 0x80};
  std::vector<uint8_t> Code(128, 0);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }
  std::memcpy(Code.data() + 4, BufLoad.data(), 8);

  InstrumentationSite Site;
  Site.Address = 4;
  Site.Offset = 4;
  Site.OrigInstSize = 8;
  Site.IsLoad = true;
  Site.IsGlobal = true;
  Site.AddrVGPRIndex = 0;

  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorInstrumented(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, 4096, {Site}, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  EXPECT_EQ(ResultOrErr->PatchedCount, 1u);
  ASSERT_EQ(ResultOrErr->Slots.size(), 1u);
  ASSERT_FALSE(ResultOrErr->Islands.empty());
  EXPECT_FALSE(ResultOrErr->Islands[0].Bytes.empty());

  const auto &Slot = ResultOrErr->Slots[0];
  EXPECT_EQ(Slot.PatchBytes.size(), 8u)
      << "8-byte instruction should produce 8-byte patch (s_call + nop)";

  // In shared-body architecture, TrampolineBytes is the 12-byte dispatch entry
  // (8-byte s_mov_b32 literal + 4-byte s_branch)
  EXPECT_EQ(Slot.TrampolineBytes.size(), 12u)
      << "Standard mode dispatch entry should be exactly 12 bytes";

  // The island must be large (shared body + dispatch + return table)
  EXPECT_GT(ResultOrErr->Islands[0].Bytes.size(), 100u)
      << "Instrumented island should contain shared body + dispatch + return table";
}

//===----------------------------------------------------------------------===//
// Zero-SGPR mode: chained islands when single island exceeds s_branch range
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, ZeroSGPRChainedIslands) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  // 32 KB kernel filled with s_nop instructions.
  constexpr uint64_t KernelSize = 32 * 1024;
  std::vector<uint8_t> Code(KernelSize, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  // Create 300 instrumentation sites spread across the kernel.
  // Each site is a 4-byte s_nop at a unique offset.
  constexpr uint32_t NumSites = 300;
  std::vector<InstrumentationSite> Sites;
  Sites.reserve(NumSites);
  uint64_t Step = (KernelSize - 4) / NumSites;
  if (Step < 4) Step = 4;
  // Ensure step is 4-byte aligned.
  Step = (Step / 4) * 4;
  for (uint32_t i = 0; i < NumSites; ++i) {
    InstrumentationSite S;
    S.Offset = i * Step;
    S.Address = S.Offset;
    S.OrigInstSize = 4;
    S.IsLoad = true;
    S.IsGlobal = true;
    S.AddrVGPRIndex = 0;
    Sites.push_back(S);
  }

  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.SGPRCount = 104;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);
  ASSERT_TRUE(Scratch.ZeroSGPR);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;
  Trace.Strategy    = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, Code.size(), Sites, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  const auto &R = *ResultOrErr;

  // With chaining, we should instrument more sites than a single island allows.
  EXPECT_GT(R.PatchedCount, 0u)
      << "Chained islands should instrument at least some sites";

  // Verify multiple islands were created.
  EXPECT_GT(R.Islands.size(), 1u)
      << "300 sites in a 32 KB kernel should require multiple islands "
         "(each slot is ~500-700 bytes, s_branch range is ±128 KB)";

  // All island offsets must be 256-byte aligned.
  for (size_t i = 0; i < R.Islands.size(); ++i) {
    EXPECT_EQ(R.Islands[i].Offset % 256, 0u)
        << "Island " << i << " offset 0x" << std::hex << R.Islands[i].Offset
        << " is not 256-byte aligned";
    EXPECT_FALSE(R.Islands[i].Bytes.empty())
        << "Island " << i << " should not be empty";
  }

  // Islands must be non-overlapping and ordered by offset.
  for (size_t i = 1; i < R.Islands.size(); ++i) {
    EXPECT_GE(R.Islands[i].Offset,
              R.Islands[i - 1].Offset + R.Islands[i - 1].Bytes.size())
        << "Islands must be non-overlapping";
  }

  // Every slot's patch bytes should be 4 bytes (s_branch).
  for (size_t i = 0; i < R.Slots.size(); ++i) {
    EXPECT_EQ(R.Slots[i].PatchBytes.size(), 4u)
        << "Slot " << i << " patch should be exactly 4 bytes (s_branch)";
    EXPECT_FALSE(R.Slots[i].UsedLongJump)
        << "Zero-SGPR slots must not use long-jump";
  }

  // Every slot should have non-empty trampoline bytes and end with
  // s_branch (return): 0xBF82xxxx.  Both direct-path and relay-path
  // trampolines end with an s_branch back to the kernel.
  for (size_t i = 0; i < R.Slots.size(); ++i) {
    const auto &TB = R.Slots[i].TrampolineBytes;
    ASSERT_GE(TB.size(), 4u) << "Slot " << i << " trampoline too small";
    uint32_t LastWord = TB[TB.size() - 4]
                      | (TB[TB.size() - 3] << 8)
                      | (TB[TB.size() - 2] << 16)
                      | (TB[TB.size() - 1] << 24);
    EXPECT_EQ(LastWord >> 16, 0xBF82u)
        << "Slot " << i << " trampoline should end with s_branch (return), got 0x"
        << std::hex << LastWord;
  }
}

//===----------------------------------------------------------------------===//
// Zero-SGPR mode: bidirectional islands (forward + backward) for full coverage
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, ZeroSGPRBidirectionalIslands) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  // Simulate a rocBLAS-like scenario: a 20 KB kernel at offset 200 KB in a
  // large .text section. 300 sites span the kernel. With forward-only islands
  // (after the kernel), we can't cover all 300 sites because the cumulative
  // island size exceeds s_branch ±128 KB. With bidirectional placement, we
  // use the 200 KB of space BEFORE the kernel for backward islands.
  constexpr uint64_t KernelSize = 20 * 1024;
  constexpr uint64_t KernelOffset = 200 * 1024; // BaseAddr = KStart
  constexpr uint64_t TextEnd = KernelOffset + KernelSize; // EffectiveTextEnd

  std::vector<uint8_t> Code(KernelSize, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  constexpr uint32_t NumSites = 300;
  std::vector<InstrumentationSite> Sites;
  Sites.reserve(NumSites);
  uint64_t Step = (KernelSize - 4) / NumSites;
  if (Step < 4) Step = 4;
  Step = (Step / 4) * 4;
  for (uint32_t i = 0; i < NumSites; ++i) {
    InstrumentationSite S;
    S.Offset = i * Step;
    S.Address = KernelOffset + S.Offset;
    S.OrigInstSize = 4;
    S.IsLoad = true;
    S.IsGlobal = true;
    S.AddrVGPRIndex = 0;
    Sites.push_back(S);
  }

  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.SGPRCount = 104;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);
  ASSERT_TRUE(Scratch.ZeroSGPR);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;
  Trace.Strategy    = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  // Pass PreKernelSpace = KernelOffset to enable backward islands.
  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, KernelOffset, TextEnd, Sites, Scratch, Trace, KernelOffset);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  const auto &R = *ResultOrErr;

  // With bidirectional placement, we should achieve full coverage.
  EXPECT_EQ(R.PatchedCount, NumSites)
      << "Bidirectional islands should cover all " << NumSites << " sites, got "
      << R.PatchedCount;
  EXPECT_EQ(R.Slots.size(), static_cast<size_t>(NumSites));

  // With 300 sites in a 20 KB kernel, the all-relay pre-check triggers:
  // relay stubs go in a near-kernel island, bodies go in a separate body
  // island.  We must have at least 2 islands.
  ASSERT_GE(R.Islands.size(), 2u)
      << "Should have at least 2 islands (stub + body)";

  // All island offsets must be 256-byte aligned.
  for (size_t i = 0; i < R.Islands.size(); ++i) {
    EXPECT_EQ(R.Islands[i].Offset % 256, 0u)
        << "Island " << i << " offset 0x" << std::hex << R.Islands[i].Offset
        << " is not 256-byte aligned";
    EXPECT_FALSE(R.Islands[i].Bytes.empty())
        << "Island " << i << " should not be empty";
  }

  // No island should overlap with the kernel code [KernelOffset, TextEnd).
  for (size_t i = 0; i < R.Islands.size(); ++i) {
    uint64_t IslStart = R.Islands[i].Offset;
    uint64_t IslEnd = IslStart + R.Islands[i].Bytes.size();
    bool Overlaps = (IslStart < TextEnd && IslEnd > KernelOffset);
    EXPECT_FALSE(Overlaps)
        << "Island " << i << " [0x" << std::hex << IslStart << ", 0x" << IslEnd
        << ") overlaps kernel [0x" << KernelOffset << ", 0x" << TextEnd << ")";
  }

  // Every slot's patch bytes should be an s_branch (4 bytes).
  for (size_t i = 0; i < R.Slots.size(); ++i) {
    EXPECT_EQ(R.Slots[i].PatchBytes.size(), 4u)
        << "Slot " << i << " patch should be exactly 4 bytes (s_branch)";
    EXPECT_FALSE(R.Slots[i].UsedLongJump)
        << "Zero-SGPR slots must not use long-jump";
  }

  // Every trampoline should end with s_branch (return): 0xBF82xxxx
  for (size_t i = 0; i < R.Slots.size(); ++i) {
    const auto &TB = R.Slots[i].TrampolineBytes;
    ASSERT_GE(TB.size(), 4u) << "Slot " << i << " trampoline too small";
    uint32_t LastWord = TB[TB.size() - 4]
                      | (TB[TB.size() - 3] << 8)
                      | (TB[TB.size() - 2] << 16)
                      | (TB[TB.size() - 1] << 24);
    EXPECT_EQ(LastWord >> 16, 0xBF82u)
        << "Slot " << i << " trampoline should end with s_branch (return), got 0x"
        << std::hex << LastWord;
  }
}

//===----------------------------------------------------------------------===//
// Helper: extract s_branch dword offset from a 4-byte s_branch instruction
//===----------------------------------------------------------------------===//

static int16_t extractSBranchDword(const uint8_t *Bytes) {
  uint32_t Word = Bytes[0] | (Bytes[1] << 8)
                | (Bytes[2] << 16) | (Bytes[3] << 24);
  return static_cast<int16_t>(Word & 0xFFFF);
}

// Helper: extract s_call_b64 dword offset from a 4-byte s_call_b64 instruction
static int16_t extractSCallDword(const uint8_t *Bytes) {
  uint32_t Word = Bytes[0] | (Bytes[1] << 8)
                | (Bytes[2] << 16) | (Bytes[3] << 24);
  return static_cast<int16_t>(Word & 0xFFFF);
}

//===----------------------------------------------------------------------===//
// Branch-target validation: standard mode
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, StandardModePatchBranchTargetsAreCorrect) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  std::vector<uint8_t> Code(512, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  std::vector<InstrumentationSite> Sites;
  for (uint64_t Off = 8; Off < 48; Off += 8) {
    InstrumentationSite S;
    S.Address = Off;
    S.Offset = Off;
    S.OrigInstSize = 4;
    S.IsLoad = true;
    S.IsGlobal = true;
    S.AddrVGPRIndex = 0;
    Sites.push_back(S);
  }
  ASSERT_EQ(Sites.size(), 5u);

  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorInstrumented(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, 4096, Sites, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  auto &R = *ResultOrErr;
  ASSERT_EQ(R.PatchedCount, 5u);
  ASSERT_FALSE(R.Islands.empty());

  uint64_t IslandOffset = R.Islands[0].Offset;
  uint64_t IslandEnd = IslandOffset + R.Islands[0].Bytes.size();

  for (size_t i = 0; i < R.Slots.size(); ++i) {
    const auto &Slot = R.Slots[i];
    uint64_t PatchSiteAbs = Sites[i].Offset;

    // For standard mode with short branch: first 4 bytes are s_call_b64
    if (!Slot.UsedLongJump) {
      ASSERT_GE(Slot.PatchBytes.size(), 4u);
      int16_t Dword = extractSCallDword(Slot.PatchBytes.data());
      uint64_t Target = PatchSiteAbs + 4 + static_cast<int64_t>(Dword) * 4;

      EXPECT_GE(Target, IslandOffset)
          << "Slot " << i << " s_call target 0x" << std::hex << Target
          << " is before island start 0x" << IslandOffset;
      EXPECT_LT(Target, IslandEnd)
          << "Slot " << i << " s_call target 0x" << std::hex << Target
          << " is past island end 0x" << IslandEnd;

      EXPECT_EQ(Target, IslandOffset + Slot.TrampolineOffset)
          << "Slot " << i << " s_call target should match Island.Offset + TrampolineOffset";
    }
  }
}

//===----------------------------------------------------------------------===//
// Branch-target validation: ZeroSGPR mode (patch + return)
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, ZeroSGPRPatchAndReturnBranchTargets) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  constexpr uint64_t KernelSize = 1024;
  std::vector<uint8_t> Code(KernelSize, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  std::vector<InstrumentationSite> Sites;
  for (uint64_t Off = 0; Off < 40; Off += 8) {
    InstrumentationSite S;
    S.Address = Off;
    S.Offset = Off;
    S.OrigInstSize = 4;
    S.IsLoad = true;
    S.IsGlobal = true;
    S.AddrVGPRIndex = 0;
    Sites.push_back(S);
  }

  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.SGPRCount = 104;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;
  Trace.Strategy    = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, Code.size(), Sites, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  auto &R = *ResultOrErr;
  EXPECT_GT(R.PatchedCount, 0u);

  for (size_t i = 0; i < R.Slots.size(); ++i) {
    const auto &Slot = R.Slots[i];
    uint64_t PatchSiteAbs = Sites[i].Offset;

    // Patch should be s_branch (4 bytes)
    ASSERT_EQ(Slot.PatchBytes.size(), 4u) << "Slot " << i;
    int16_t FwdDword = extractSBranchDword(Slot.PatchBytes.data());
    uint64_t FwdTarget = PatchSiteAbs + 4 + static_cast<int64_t>(FwdDword) * 4;

    // Forward branch should land somewhere in an island
    bool LandsInIsland = false;
    for (const auto &Isl : R.Islands) {
      if (FwdTarget >= Isl.Offset &&
          FwdTarget < Isl.Offset + Isl.Bytes.size()) {
        LandsInIsland = true;
        break;
      }
    }
    EXPECT_TRUE(LandsInIsland)
        << "Slot " << i << " forward s_branch target 0x" << std::hex
        << FwdTarget << " does not land in any island";

    // Return branch: last 4 bytes of TrampolineBytes should be s_branch
    size_t TB_size = Slot.TrampolineBytes.size();
    ASSERT_GE(TB_size, 4u);
    uint32_t LastWord = Slot.TrampolineBytes[TB_size - 4]
                      | (Slot.TrampolineBytes[TB_size - 3] << 8)
                      | (Slot.TrampolineBytes[TB_size - 2] << 16)
                      | (Slot.TrampolineBytes[TB_size - 1] << 24);
    ASSERT_EQ(LastWord >> 16, 0xBF82u) << "Slot " << i;

    // Find the absolute address of this return s_branch instruction.
    // It's at: (island containing this slot).Offset + Slot.TrampolineOffset + TB_size - 4
    for (const auto &Isl : R.Islands) {
      uint64_t SlotAbs = Isl.Offset + Slot.TrampolineOffset;
      if (FwdTarget >= Isl.Offset &&
          FwdTarget < Isl.Offset + Isl.Bytes.size()) {
        uint64_t RetBranchPC = SlotAbs + TB_size - 4;
        int16_t RetDword = extractSBranchDword(
            Slot.TrampolineBytes.data() + TB_size - 4);
        uint64_t RetTarget = RetBranchPC + 4 +
                             static_cast<int64_t>(RetDword) * 4;
        EXPECT_EQ(RetTarget, PatchSiteAbs + Sites[i].OrigInstSize)
            << "Slot " << i << " return s_branch should target "
            << "PatchSite + OrigInstSize = 0x" << std::hex
            << (PatchSiteAbs + Sites[i].OrigInstSize)
            << " but got 0x" << RetTarget;
        break;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Zero-site-drop: standard mode 100 sites
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, StandardMode100SitesAllPatched) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  constexpr uint64_t KernelSize = 128 * 1024;
  std::vector<uint8_t> Code(KernelSize, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  constexpr uint32_t NumSites = 100;
  std::vector<InstrumentationSite> Sites;
  uint64_t Step = ((KernelSize - 4) / NumSites / 4) * 4;
  if (Step < 4) Step = 4;
  for (uint32_t i = 0; i < NumSites; ++i) {
    InstrumentationSite S;
    S.Offset = i * Step;
    S.Address = S.Offset;
    S.OrigInstSize = 4;
    S.IsLoad = true;
    S.IsGlobal = true;
    S.AddrVGPRIndex = 0;
    Sites.push_back(S);
  }

  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorInstrumented(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, KernelSize, Sites, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  EXPECT_EQ(ResultOrErr->PatchedCount, NumSites)
      << "Standard mode should patch all " << NumSites << " sites";
  EXPECT_EQ(ResultOrErr->Slots.size(), static_cast<size_t>(NumSites));
}

//===----------------------------------------------------------------------===//
// Zero-site-drop: ZeroSGPR 500 sites with bidirectional
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, ZeroSGPR500SitesAllPatched) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  constexpr uint64_t KernelSize = 32 * 1024;
  constexpr uint64_t KernelOffset = 200 * 1024;
  constexpr uint64_t TextEnd = KernelOffset + KernelSize;

  std::vector<uint8_t> Code(KernelSize, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  constexpr uint32_t NumSites = 500;
  std::vector<InstrumentationSite> Sites;
  uint64_t Step = ((KernelSize - 4) / NumSites / 4) * 4;
  if (Step < 4) Step = 4;
  for (uint32_t i = 0; i < NumSites; ++i) {
    InstrumentationSite S;
    S.Offset = i * Step;
    S.Address = KernelOffset + S.Offset;
    S.OrigInstSize = 4;
    S.IsLoad = true;
    S.IsGlobal = true;
    S.AddrVGPRIndex = 0;
    Sites.push_back(S);
  }

  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.SGPRCount = 104;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;
  Trace.Strategy    = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, KernelOffset, TextEnd, Sites, Scratch, Trace, KernelOffset);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  EXPECT_EQ(ResultOrErr->PatchedCount, NumSites)
      << "ZeroSGPR all-relay should patch all " << NumSites << " sites";
}

//===----------------------------------------------------------------------===//
// Zero-SGPR mode: all-relay overflow test (500 sites, zero drops)
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, ZeroSGPRAllRelayOverflow) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  constexpr uint64_t KernelSize = 10 * 1024;
  std::vector<uint8_t> Code(KernelSize, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  constexpr uint32_t NumSites = 500;
  std::vector<InstrumentationSite> Sites;
  uint64_t Step = ((KernelSize - 4) / NumSites / 4) * 4;
  if (Step < 4) Step = 4;
  for (uint32_t i = 0; i < NumSites; ++i) {
    InstrumentationSite S;
    S.Offset = i * Step;
    S.Address = S.Offset;
    S.OrigInstSize = 4;
    S.IsLoad = true;
    S.IsGlobal = true;
    S.AddrVGPRIndex = 0;
    Sites.push_back(S);
  }

  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.SGPRCount = 104;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);
  ASSERT_TRUE(Scratch.ZeroSGPR);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;
  Trace.Strategy    = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, Code.size(), Sites, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  const auto &R = *ResultOrErr;

  // All 500 sites must be patched — zero drops.
  EXPECT_EQ(R.PatchedCount, NumSites)
      << "All-relay mode should patch all " << NumSites
      << " sites with zero drops, got " << R.PatchedCount;
  EXPECT_EQ(R.Slots.size(), static_cast<size_t>(NumSites));

  // At least 2 islands: stub island + relay body island.
  ASSERT_GE(R.Islands.size(), 2u)
      << "All-relay should produce at least 2 islands (stubs + body)";

  // Every slot must have non-empty patch bytes and trampoline bytes,
  // and the trampoline must end with s_branch (return).
  for (size_t i = 0; i < R.Slots.size(); ++i) {
    EXPECT_FALSE(R.Slots[i].PatchBytes.empty())
        << "Slot " << i << " has empty patch bytes";
    const auto &TB = R.Slots[i].TrampolineBytes;
    ASSERT_GE(TB.size(), 4u) << "Slot " << i << " trampoline too small";
    uint32_t LastWord = TB[TB.size() - 4]
                      | (TB[TB.size() - 3] << 8)
                      | (TB[TB.size() - 2] << 16)
                      | (TB[TB.size() - 1] << 24);
    EXPECT_EQ(LastWord >> 16, 0xBF82u)
        << "Slot " << i << " trampoline should end with s_branch (return), got 0x"
        << std::hex << LastWord;
  }

  // Verify a relay body island exists (from fixupRelays): the last island
  // should be placed after all stub islands.
  bool HasBodyIsland = false;
  uint64_t MaxStubEnd = 0;
  for (size_t i = 0; i + 1 < R.Islands.size(); ++i) {
    uint64_t End = R.Islands[i].Offset + R.Islands[i].Bytes.size();
    if (End > MaxStubEnd) MaxStubEnd = End;
  }
  const auto &LastIsl = R.Islands.back();
  if (LastIsl.Offset >= MaxStubEnd)
    HasBodyIsland = true;
  EXPECT_TRUE(HasBodyIsland) << "Expected a relay body island after stub islands";
}

//===----------------------------------------------------------------------===//
// Standard mode: correct monotonic offsets
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, StandardModeMultipleSitesCorrectOffsets) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  std::vector<uint8_t> Code(256, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  std::vector<InstrumentationSite> Sites;
  for (uint64_t Off = 4; Off < 28; Off += 4) {
    InstrumentationSite S;
    S.Address = Off;
    S.Offset = Off;
    S.OrigInstSize = 4;
    S.IsLoad = true;
    S.IsGlobal = true;
    S.AddrVGPRIndex = 0;
    Sites.push_back(S);
  }

  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorInstrumented(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, 4096, Sites, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  auto &R = *ResultOrErr;
  ASSERT_EQ(R.Slots.size(), Sites.size());

  // Offsets must be monotonically increasing
  for (size_t i = 1; i < R.Slots.size(); ++i) {
    EXPECT_GT(R.Slots[i].TrampolineOffset, R.Slots[i - 1].TrampolineOffset)
        << "TrampolineOffset must be monotonically increasing at slot " << i;
  }

  // With shared-body architecture, dispatch entries have uniform 12-byte stride
  // (8-byte s_mov_b32 literal + 4-byte s_branch)
  for (size_t i = 1; i < R.Slots.size(); ++i) {
    EXPECT_EQ(R.Slots[i].TrampolineOffset - R.Slots[i - 1].TrampolineOffset, 12u)
        << "Dispatch entry stride should be uniform 12 bytes at slot " << i;
  }
}

//===----------------------------------------------------------------------===//
// Standard mode: island byte sum matches slot trampoline byte sum
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, StandardModeIslandByteSumMatchesSlots) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  std::vector<uint8_t> Code(256, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  std::vector<InstrumentationSite> Sites;
  for (uint64_t Off = 4; Off < 20; Off += 4) {
    InstrumentationSite S;
    S.Address = Off;
    S.Offset = Off;
    S.OrigInstSize = 4;
    S.IsLoad = true;
    S.IsGlobal = true;
    S.AddrVGPRIndex = 0;
    Sites.push_back(S);
  }

  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorInstrumented(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, 4096, Sites, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  auto &R = *ResultOrErr;
  ASSERT_EQ(R.Islands.size(), 1u);

  // With shared-body architecture, island contains:
  //   shared body(s) + dispatch table (N*12) + return table (N*24)
  // The island is larger than the sum of slot TrampolineBytes (dispatch entries only).
  uint64_t SlotByteSum = 0;
  for (const auto &Slot : R.Slots)
    SlotByteSum += Slot.TrampolineBytes.size();

  uint64_t NumSites = R.Slots.size();
  uint64_t ExpectedDispatchSize = NumSites * 12;
  uint64_t ExpectedReturnSize = NumSites * 16;

  EXPECT_EQ(SlotByteSum, ExpectedDispatchSize)
      << "Slot trampoline bytes should be dispatch entries (12 bytes each)";

  EXPECT_GT(R.Islands[0].Bytes.size(), SlotByteSum)
      << "Island must be larger than dispatch table (includes shared body + return table)";

  // Island size = shared_body + dispatch_table + return_table
  uint64_t SharedBodySize = R.Islands[0].Bytes.size() - ExpectedDispatchSize - ExpectedReturnSize;
  EXPECT_GT(SharedBodySize, 40u)
      << "Shared body should be substantial (>40 bytes)";
}

//===----------------------------------------------------------------------===//
// LDS site test
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, ZeroSGPRWithLDSSite) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  std::vector<uint8_t> Code(256, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  InstrumentationSite Site;
  Site.Address = 0;
  Site.Offset = 0;
  Site.OrigInstSize = 4;
  Site.IsLoad = true;
  Site.IsGlobal = false; // LDS site
  Site.AddrVGPRIndex = 0;
  Site.DSOffset0 = 64;

  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.SGPRCount = 104;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;
  Trace.Strategy    = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, Code.size(), {Site}, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  EXPECT_EQ(ResultOrErr->PatchedCount, 1u);
  ASSERT_EQ(ResultOrErr->Slots.size(), 1u);
  EXPECT_GT(ResultOrErr->Slots[0].TrampolineBytes.size(), 40u)
      << "LDS site trampoline should be substantial (has LDS counting payload)";
}

//===----------------------------------------------------------------------===//
// Mixed instruction sizes (4-byte and 8-byte)
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, MixedInstructionSizes) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  // buffer_load_dword v1, v0, s[0:3], 0 offen  (8 bytes)
  std::vector<uint8_t> BufLoad = {0x00, 0x10, 0x50, 0xE0,
                                   0x00, 0x01, 0x00, 0x80};
  std::vector<uint8_t> Code(128, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }
  // Place 8-byte instruction at offset 16
  std::memcpy(Code.data() + 16, BufLoad.data(), 8);

  // Site 0: 4-byte at offset 4
  InstrumentationSite S0;
  S0.Address = 4; S0.Offset = 4; S0.OrigInstSize = 4;
  S0.IsLoad = true; S0.IsGlobal = true; S0.AddrVGPRIndex = 0;

  // Site 1: 8-byte at offset 16
  InstrumentationSite S1;
  S1.Address = 16; S1.Offset = 16; S1.OrigInstSize = 8;
  S1.IsLoad = true; S1.IsGlobal = true; S1.AddrVGPRIndex = 0;

  // Site 2: 4-byte at offset 28
  InstrumentationSite S2;
  S2.Address = 28; S2.Offset = 28; S2.OrigInstSize = 4;
  S2.IsLoad = true; S2.IsGlobal = true; S2.AddrVGPRIndex = 0;

  KernelDescriptor KD{};
  KD.SGPRCount = 24;
  KD.VGPRCount = 8;
  KD.VGPRGranularity = 8;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorInstrumented(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, 4096, {S0, S1, S2}, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  auto &R = *ResultOrErr;
  ASSERT_EQ(R.PatchedCount, 3u);
  ASSERT_EQ(R.Slots.size(), 3u);

  // 4-byte site -> 4-byte patch (or 8 if long-jump)
  if (!R.Slots[0].UsedLongJump) {
    EXPECT_EQ(R.Slots[0].PatchBytes.size(), 4u) << "4-byte site should have 4-byte patch";
  }

  // 8-byte site -> 8-byte patch (s_call + s_nop, or long-jump)
  EXPECT_EQ(R.Slots[1].PatchBytes.size(), 8u) << "8-byte site should have 8-byte patch";

  // 4-byte site -> 4-byte patch
  if (!R.Slots[2].UsedLongJump) {
    EXPECT_EQ(R.Slots[2].PatchBytes.size(), 4u) << "4-byte site should have 4-byte patch";
  }
}

//===----------------------------------------------------------------------===//
// Scratch-spill path (NeedsScratchSpill=true)
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, ScratchSpillPathDirect) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  std::vector<uint8_t> Code(256, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  InstrumentationSite Site;
  Site.Address = 0;
  Site.Offset = 0;
  Site.OrigInstSize = 4;
  Site.IsLoad = true;
  Site.IsGlobal = true;
  Site.AddrVGPRIndex = 0;

  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.SGPRCount = 104;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);
  Scratch.NeedsScratchSpill = true;
  Scratch.ScratchSpillOffset = 128;
  Scratch.ExtraScratchBytes = 12;

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;
  Trace.Strategy    = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, Code.size(), {Site}, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  EXPECT_EQ(ResultOrErr->PatchedCount, 1u);
  ASSERT_FALSE(ResultOrErr->Slots.empty());

  // The trampoline should be larger than a non-spill trampoline
  // due to scratch store/load + waitcnt instructions
  EXPECT_GT(ResultOrErr->Slots[0].TrampolineBytes.size(), 60u)
      << "Scratch-spill trampoline should be substantial";
}

//===----------------------------------------------------------------------===//
// Error paths
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, ZeroSGPRPlusFullCaptureRejected) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  std::vector<uint8_t> Code(64, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  InstrumentationSite Site;
  Site.Address = 0; Site.Offset = 0; Site.OrigInstSize = 4;
  Site.IsLoad = true; Site.IsGlobal = true; Site.AddrVGPRIndex = 0;

  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.SGPRCount = 104;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;
  Trace.Strategy    = PayloadStrategy::FullCapture;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, Code.size(), {Site}, Scratch, Trace);
  EXPECT_FALSE(!!ResultOrErr)
      << "ZeroSGPR + FullCapture should be rejected";
  if (!ResultOrErr)
    consumeError(ResultOrErr.takeError());
}

TEST_F(TrampolineBridgeTest, ZeroSGPRWithoutGPUAtomicsRejected) {
  auto BridgeOrErr = TrampolineBridge::create("gfx942", *Disasm);
  ASSERT_TRUE(!!BridgeOrErr);

  std::vector<uint8_t> Code(64, 0x00);
  for (size_t i = 0; i < Code.size(); i += 4) {
    Code[i + 0] = 0x00; Code[i + 1] = 0x00;
    Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
  }

  InstrumentationSite Site;
  Site.Address = 0; Site.Offset = 0; Site.OrigInstSize = 4;
  Site.IsLoad = true; Site.IsGlobal = true; Site.AddrVGPRIndex = 0;

  KernelDescriptor KD{};
  KD.VGPRCount = 8;
  KD.SGPRCount = 104;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorZeroSGPR(KD);

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 1024 * 1024;
  Trace.Strategy    = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = false;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      Code, 0, Code.size(), {Site}, Scratch, Trace);
  EXPECT_FALSE(!!ResultOrErr)
      << "ZeroSGPR without SupportsGPUAtomics should be rejected";
  if (!ResultOrErr)
    consumeError(ResultOrErr.takeError());
}

//===----------------------------------------------------------------------===//
// Real-kernel regression: GEMM ELF (gfx950)
//===----------------------------------------------------------------------===//

TEST_F(TrampolineBridgeTest, GEMMKernelInstrumentedAllSitesPatched) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(
      ArrayRef<uint8_t>(gemm_gfx950_elf, gemm_gfx950_elf_len));
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());

  auto Names = HandlerOrErr->getKernelNames();
  ASSERT_FALSE(Names.empty());
  const KernelInfo *KI = HandlerOrErr->getKernel(Names[0]);
  ASSERT_NE(KI, nullptr);
  ASSERT_GT(KI->CodeSize, 0u);

  auto TextSection = HandlerOrErr->getTextSection();
  ASSERT_GE(TextSection.size(), KI->CodeOffset + KI->CodeSize);
  ArrayRef<uint8_t> KernelCode =
      TextSection.slice(KI->CodeOffset, KI->CodeSize);

  // Need gfx950 disassembler for this ELF
  auto Disasm950OrErr =
      Disassembler::create("amdgcn-amd-amdhsa", "gfx950", "+wavefrontsize64");
  if (!Disasm950OrErr) {
    GTEST_SKIP() << "Cannot create gfx950 disassembler";
  }
  auto &Disasm950 = *Disasm950OrErr;

  // Build CFG and find real memory sites
  CFGBuilder Builder(*Disasm950);
  auto CFGOrErr = Builder.build(KernelCode, 0);
  ASSERT_TRUE(!!CFGOrErr) << toString(CFGOrErr.takeError());

  KernelDescriptor KD = KI->Descriptor;
  ScratchRegisters Scratch = ScratchRegisters::fromDescriptorInstrumented(KD);

  auto Sites = TrampolineBridge::findMemorySites(
      *CFGOrErr, 0, *Disasm950, Scratch);
  if (Sites.empty()) {
    GTEST_SKIP() << "No memory sites found in GEMM kernel (fixture may not have memory ops)";
  }

  // Build instrumented trampoline in standard mode
  auto BridgeOrErr = TrampolineBridge::create("gfx950", *Disasm950);
  ASSERT_TRUE(!!BridgeOrErr) << toString(BridgeOrErr.takeError());

  TraceConfig Trace;
  Trace.BufferAddr  = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xBEEF000000000000ULL;
  Trace.BufferSize  = 4 * 1024 * 1024;

  auto ResultOrErr = (*BridgeOrErr)->buildInstrumented(
      KernelCode, 0, TextSection.size(), Sites, Scratch, Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());

  auto &R = *ResultOrErr;
  uint32_t NumSites = static_cast<uint32_t>(Sites.size());

  // Zero-drop: all sites must be patched
  EXPECT_EQ(R.PatchedCount, NumSites)
      << "GEMM kernel: all " << NumSites << " sites should be patched, got "
      << R.PatchedCount;

  // All islands must be 256-byte aligned and non-overlapping
  for (size_t i = 0; i < R.Islands.size(); ++i) {
    EXPECT_EQ(R.Islands[i].Offset % 256, 0u)
        << "GEMM island " << i << " not aligned";
    EXPECT_FALSE(R.Islands[i].Bytes.empty())
        << "GEMM island " << i << " is empty";

    for (size_t j = i + 1; j < R.Islands.size(); ++j) {
      uint64_t Ai = R.Islands[i].Offset;
      uint64_t Bi = Ai + R.Islands[i].Bytes.size();
      uint64_t Aj = R.Islands[j].Offset;
      uint64_t Bj = Aj + R.Islands[j].Bytes.size();
      EXPECT_FALSE(Ai < Bj && Aj < Bi)
          << "GEMM islands " << i << " and " << j << " overlap";
    }
  }

  // Every slot trampoline must be non-empty
  for (size_t i = 0; i < R.Slots.size(); ++i) {
    EXPECT_FALSE(R.Slots[i].TrampolineBytes.empty())
        << "GEMM slot " << i << " has empty trampoline";
    EXPECT_FALSE(R.Slots[i].PatchBytes.empty())
        << "GEMM slot " << i << " has empty patch";
  }
}

} // namespace
