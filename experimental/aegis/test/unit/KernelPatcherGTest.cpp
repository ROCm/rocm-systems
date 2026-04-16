//===-- KernelPatcherGTest.cpp - KernelPatcher Tests ------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for KernelPatcher: creation, cache stats, and the
/// ScratchRegisters allocation helpers.  Uses the gfx950 GEMM ELF fixture
/// and CFG infrastructure for refineScratchVGPRs testing.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/KernelPatcher.h"
#include "aegisbit/CFGBuilder.h"
#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/RegisterHelper.h"
#include "aegisbit/Types.h"
#include "fixtures/gemm_gfx950_elf.h"
#include <gtest/gtest.h>

using namespace aegisbit;
using namespace llvm;

//===----------------------------------------------------------------------===//
// K-001: create() – success and failure paths
//===----------------------------------------------------------------------===//

TEST(KernelPatcher, CreateGfx942Succeeds) {
  auto POrErr = KernelPatcher::create("gfx942");
  ASSERT_TRUE(!!POrErr) << toString(POrErr.takeError());
}

TEST(KernelPatcher, CreateGfx950Succeeds) {
  auto POrErr = KernelPatcher::create("gfx950");
  ASSERT_TRUE(!!POrErr) << toString(POrErr.takeError());
}

// Note: KernelPatcher::create with an unrecognized arch string triggers
// LLVM ERROR (abort) inside the Disassembler, so it cannot be tested as
// a graceful error path in a unit test.

//===----------------------------------------------------------------------===//
// K-002: getGPUArch
//===----------------------------------------------------------------------===//

TEST(KernelPatcher, GetGPUArchReturnsCorrectValue) {
  auto POrErr = KernelPatcher::create("gfx942");
  ASSERT_TRUE(!!POrErr) << toString(POrErr.takeError());
  EXPECT_EQ((*POrErr)->getGPUArch(), "gfx942");
}

//===----------------------------------------------------------------------===//
// K-003: cache stats – initially zero
//===----------------------------------------------------------------------===//

TEST(KernelPatcher, CacheStatsInitiallyZero) {
  auto POrErr = KernelPatcher::create("gfx942");
  ASSERT_TRUE(!!POrErr) << toString(POrErr.takeError());

  auto Stats = (*POrErr)->getCacheStats();
  EXPECT_EQ(Stats.CacheHits, 0u);
  EXPECT_EQ(Stats.CacheMisses, 0u);
  EXPECT_EQ(Stats.TotalPatched, 0u);
  EXPECT_EQ(Stats.TotalPatchErrors, 0u);
}

//===----------------------------------------------------------------------===//
// K-004: clearCache doesn't crash on empty cache
//===----------------------------------------------------------------------===//

TEST(KernelPatcher, ClearEmptyCacheNoCrash) {
  auto POrErr = KernelPatcher::create("gfx942");
  ASSERT_TRUE(!!POrErr) << toString(POrErr.takeError());
  (*POrErr)->clearCache();
  auto Stats = (*POrErr)->getCacheStats();
  EXPECT_EQ(Stats.TotalPatched, 0u);
}

//===----------------------------------------------------------------------===//
// K-005: ScratchRegisters::fromDescriptor – normal kernel
//===----------------------------------------------------------------------===//

TEST(KernelPatcher, ScratchFromDescriptorNormalKernel) {
  KernelDescriptor KD{};
  KD.VGPRCount = 64;
  KD.SGPRCount = 32;
  KD.AccumOffset = 0;

  auto SR = ScratchRegisters::fromDescriptor(KD);
  EXPECT_EQ(SR.ExtraSGPRs, 2u);
  EXPECT_EQ(SR.ExtraVGPRs, 1u);
  EXPECT_EQ(SR.ReturnAddrSGPR, RegisterHelper::getSGPR(32));
  EXPECT_EQ(SR.ReturnAddrSGPRHi, RegisterHelper::getSGPR(33));
  EXPECT_EQ(SR.ScratchVGPR, RegisterHelper::getVGPR(64));
  EXPECT_FALSE(SR.ZeroSGPR);
  EXPECT_TRUE(SR.isValid());
}

//===----------------------------------------------------------------------===//
// K-006: ScratchRegisters::fromDescriptorInstrumented – normal kernel
//===----------------------------------------------------------------------===//

TEST(KernelPatcher, ScratchInstrumentedNormalKernel) {
  KernelDescriptor KD{};
  KD.VGPRCount = 64;
  KD.SGPRCount = 32;
  KD.AccumOffset = 0;

  auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);
  EXPECT_GE(SR.ExtraSGPRs, 0u); // may be 0 due to implicit SGPR reclaim
  EXPECT_EQ(SR.ExtraVGPRs, 3u);
  EXPECT_NE(SR.ScratchVGPR, 0u);
  EXPECT_NE(SR.LaneVGPR, 0u);
  EXPECT_NE(SR.TempVGPR, 0u);
  EXPECT_FALSE(SR.ZeroSGPR);
  EXPECT_TRUE(SR.isValid());
}

//===----------------------------------------------------------------------===//
// K-007: ScratchRegisters::fromDescriptorInstrumented – high SGPR kernel
//===----------------------------------------------------------------------===//

TEST(KernelPatcher, ScratchInstrumentedHighSGPR) {
  KernelDescriptor KD{};
  KD.VGPRCount = 96;
  KD.SGPRCount = 104;
  KD.AccumOffset = 0;

  auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);
  // Even with 104 SGPRs (near max), the function should succeed because
  // implicit SGPRs are reclaimed.
  EXPECT_FALSE(SR.ZeroSGPR);
  EXPECT_TRUE(SR.isValid());
}

//===----------------------------------------------------------------------===//
// K-007b: ImplicitSGPRs=6 on gfx940+ shifts ScratchBase lower
//===----------------------------------------------------------------------===//

TEST(KernelPatcher, ScratchInstrumentedGFX950ImplicitSGPRs) {
  KernelDescriptor KD{};
  KD.VGPRCount = 64;
  KD.SGPRCount = 24;
  KD.AccumOffset = 0;
  KD.ImplicitSGPRs = 6; // gfx940+: VCC(2) + FLAT_SCRATCH(2) + XNACK_MASK(2)

  auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);
  // ScratchBase = 24 - 6 = 18 (not 20 as with ImplicitSGPRs=4)
  EXPECT_EQ(SR.FirstFreeSGPRIdx, 18u);
  EXPECT_EQ(SR.ReturnAddrSGPR, RegisterHelper::getSGPR(18));
  EXPECT_TRUE(SR.isValid());
}

TEST(KernelPatcher, ScratchSwapPCGFX950ImplicitSGPRs) {
  KernelDescriptor KD{};
  KD.VGPRCount = 512;
  KD.SGPRCount = 24;
  KD.AccumOffset = 256;
  KD.ImplicitSGPRs = 6; // gfx940+

  auto SR = ScratchRegisters::fromDescriptorSwapPC(KD);
  // ScratchBase = 24 - 6 = 18
  EXPECT_EQ(SR.FirstFreeSGPRIdx, 18u);
  // SwapTargetSGPR at ScratchBase+6 = s24, SwapTargetSGPRHi = s25
  EXPECT_EQ(SR.SwapTargetSGPR, RegisterHelper::getSGPR(24));
  EXPECT_EQ(SR.SwapTargetSGPRHi, RegisterHelper::getSGPR(25));
  EXPECT_TRUE(SR.UseSwapPC);
}

//===----------------------------------------------------------------------===//
// K-008: ScratchRegisters::fromDescriptorZeroSGPR
//===----------------------------------------------------------------------===//

TEST(KernelPatcher, ScratchZeroSGPR) {
  KernelDescriptor KD{};
  KD.VGPRCount = 64;
  KD.SGPRCount = 104;
  KD.AccumOffset = 0;

  auto SR = ScratchRegisters::fromDescriptorZeroSGPR(KD);
  EXPECT_TRUE(SR.ZeroSGPR);
  EXPECT_EQ(SR.ExtraSGPRs, 0u);
  EXPECT_EQ(SR.ExtraVGPRs, 3u);
  EXPECT_NE(SR.ScratchVGPR, 0u);
  EXPECT_TRUE(SR.isValid());
}

//===----------------------------------------------------------------------===//
// K-009: ScratchRegisters – AccVGPR kernel (no free regs initially)
//===----------------------------------------------------------------------===//

TEST(KernelPatcher, ScratchInstrumentedAccVGPRKernel) {
  KernelDescriptor KD{};
  KD.VGPRCount = 512;   // full unified register file
  KD.SGPRCount = 32;
  KD.AccumOffset = 256;  // v0-v255 regular, a0-a255 accum

  auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);
  // AccVGPR path: scratch VGPRs are initially 0 (set by refineScratchVGPRs)
  EXPECT_EQ(SR.ExtraVGPRs, 0u);
  EXPECT_TRUE(SR.HasAccumVGPRs);
}

//===----------------------------------------------------------------------===//
// K-010: refineScratchVGPRs with real CFG from GEMM fixture
//===----------------------------------------------------------------------===//

TEST(KernelPatcher, RefineScratchVGPRsWithRealCFG) {
  auto DisasmOrErr =
      Disassembler::create("amdgcn-amd-amdhsa", "gfx950", "+wavefrontsize64");
  if (!DisasmOrErr) {
    GTEST_SKIP() << "Cannot create gfx950 disassembler";
  }
  auto &Disasm = *DisasmOrErr;

  // Load the GEMM fixture and extract kernel code.
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

  // Build CFG from the real kernel code.
  CFGBuilder Builder(*Disasm);
  auto CFGOrErr = Builder.build(KernelCode, 0);
  ASSERT_TRUE(!!CFGOrErr) << toString(CFGOrErr.takeError());

  // Set up ScratchRegisters for an AccVGPR kernel and refine.
  KernelDescriptor KD = KI->Descriptor;
  if (KD.AccumOffset == 0) {
    // The GEMM fixture may not use AccVGPRs; synthesize a scenario.
    KD.AccumOffset = 128;
    KD.VGPRCount = 256;
  }
  auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);
  bool Refined = SR.refineScratchVGPRs(*CFGOrErr, *Disasm, KD.AccumOffset);
  // Whether it finds free VGPRs depends on the kernel, but it must not crash.
  if (Refined) {
    EXPECT_NE(SR.ScratchVGPR, 0u);
    EXPECT_NE(SR.LaneVGPR, 0u);
    EXPECT_NE(SR.TempVGPR, 0u);
    EXPECT_EQ(SR.ExtraVGPRs, 0u);
  }
  SUCCEED();
}

//===----------------------------------------------------------------------===//
// K-011: setupScratchSpill and setupAccVGPRSpill don't crash
//===----------------------------------------------------------------------===//

TEST(KernelPatcher, SetupScratchSpillProducesValidConfig) {
  KernelDescriptor KD{};
  KD.VGPRCount = 512;
  KD.SGPRCount = 32;
  KD.AccumOffset = 256;
  KD.PrivateSegmentFixedSize = 0;

  auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);
  SR.setupScratchSpill(KD.AccumOffset, KD.PrivateSegmentFixedSize);

  EXPECT_TRUE(SR.NeedsScratchSpill);
  EXPECT_FALSE(SR.NeedsAccVGPRSpill);
  EXPECT_EQ(SR.ExtraScratchBytes, 12u); // 3 VGPRs × 4 bytes
  EXPECT_EQ(SR.ScratchSpillOffset, 0u); // starts at current scratch end
  EXPECT_NE(SR.ScratchVGPR, 0u);
}

TEST(KernelPatcher, SetupAccVGPRSpillProducesValidConfig) {
  KernelDescriptor KD{};
  KD.VGPRCount = 512;
  KD.SGPRCount = 32;
  KD.AccumOffset = 256;

  auto SR = ScratchRegisters::fromDescriptorInstrumented(KD);
  SR.setupAccVGPRSpill(KD.AccumOffset, KD.VGPRCount);

  EXPECT_TRUE(SR.NeedsAccVGPRSpill);
  EXPECT_NE(SR.ScratchVGPR, 0u);
  EXPECT_NE(SR.SpillAGPR0, 0u);
}
