//===-- CodeObjectHandlerGTest.cpp - CodeObjectHandler Tests -----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for CodeObjectHandler: load, query, modify, and build code
/// objects. Uses the gfx950 GEMM ELF fixture for real-binary coverage.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/Types.h"
#include "fixtures/gemm_gfx950_elf.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace aegisbit;
using namespace llvm;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

ArrayRef<uint8_t> gemmELF() {
  return ArrayRef<uint8_t>(gemm_gfx950_elf, gemm_gfx950_elf_len);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// H-001: loadFromBytes – success path
//===----------------------------------------------------------------------===//

TEST(CodeObjectHandler, LoadFromValidELF) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());
}

//===----------------------------------------------------------------------===//
// H-002: getGPUArch
//===----------------------------------------------------------------------===//

TEST(CodeObjectHandler, GPUArchIsGfx950) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());
  EXPECT_EQ(HandlerOrErr->getGPUArch(), "gfx950");
}

//===----------------------------------------------------------------------===//
// H-003: getKernelCount
//===----------------------------------------------------------------------===//

TEST(CodeObjectHandler, KernelCountIsPositive) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());
  EXPECT_GE(HandlerOrErr->getKernelCount(), 1u);
}

//===----------------------------------------------------------------------===//
// H-004: getKernelNames contains sgemm_naive
//===----------------------------------------------------------------------===//

TEST(CodeObjectHandler, KernelNamesContainsSgemmNaive) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());

  auto Names = HandlerOrErr->getKernelNames();
  EXPECT_FALSE(Names.empty());
  bool Found = false;
  for (const auto &N : Names) {
    if (N.find("sgemm_naive") != std::string::npos)
      Found = true;
  }
  EXPECT_TRUE(Found) << "Expected 'sgemm_naive' in kernel names";
}

//===----------------------------------------------------------------------===//
// H-005: getKernel by name
//===----------------------------------------------------------------------===//

TEST(CodeObjectHandler, GetKernelByNameFound) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());

  auto Names = HandlerOrErr->getKernelNames();
  ASSERT_FALSE(Names.empty());
  const KernelInfo *KI = HandlerOrErr->getKernel(Names[0]);
  ASSERT_NE(KI, nullptr);
  EXPECT_EQ(KI->Name, Names[0]);
  EXPECT_GT(KI->CodeSize, 0u);
}

TEST(CodeObjectHandler, GetKernelByNameWithKdSuffix) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());

  auto Names = HandlerOrErr->getKernelNames();
  ASSERT_FALSE(Names.empty());
  std::string NameWithKd = Names[0] + ".kd";
  const KernelInfo *KI = HandlerOrErr->getKernel(NameWithKd);
  EXPECT_NE(KI, nullptr);
}

TEST(CodeObjectHandler, GetKernelNonexistentReturnsNull) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());
  EXPECT_EQ(HandlerOrErr->getKernel("nonexistent_kernel_xyz"), nullptr);
}

//===----------------------------------------------------------------------===//
// H-006: getTextSection / setTextSection
//===----------------------------------------------------------------------===//

TEST(CodeObjectHandler, TextSectionNonEmpty) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());
  EXPECT_GT(HandlerOrErr->getTextSection().size(), 0u);
}

TEST(CodeObjectHandler, SetTextSectionReflected) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());

  auto OrigText = HandlerOrErr->getTextSection();
  std::vector<uint8_t> NewText(OrigText.begin(), OrigText.end());
  NewText[0] ^= 0xFF; // flip a byte

  HandlerOrErr->setTextSection(NewText);
  auto Result = HandlerOrErr->getTextSection();
  ASSERT_EQ(Result.size(), NewText.size());
  EXPECT_EQ(Result[0], NewText[0]);
}

//===----------------------------------------------------------------------===//
// H-007: build produces valid ELF
//===----------------------------------------------------------------------===//

TEST(CodeObjectHandler, BuildProducesValidELF) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());

  auto BuiltOrErr = HandlerOrErr->build();
  ASSERT_TRUE(!!BuiltOrErr) << toString(BuiltOrErr.takeError());

  auto &Built = *BuiltOrErr;
  ASSERT_GE(Built.size(), 4u);
  EXPECT_EQ(Built[0], 0x7F);
  EXPECT_EQ(Built[1], 'E');
  EXPECT_EQ(Built[2], 'L');
  EXPECT_EQ(Built[3], 'F');
}

//===----------------------------------------------------------------------===//
// H-008: loadFromBytes – error paths
//===----------------------------------------------------------------------===//

TEST(CodeObjectHandler, LoadFromGarbageBytesReturnsError) {
  std::vector<uint8_t> Garbage(256, 0xCC);
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(Garbage);
  EXPECT_FALSE(!!HandlerOrErr);
  consumeError(HandlerOrErr.takeError());
}

TEST(CodeObjectHandler, LoadFromEmptyBytesReturnsError) {
  std::vector<uint8_t> Empty;
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(Empty);
  EXPECT_FALSE(!!HandlerOrErr);
  consumeError(HandlerOrErr.takeError());
}

//===----------------------------------------------------------------------===//
// H-009: getRodataSection / getNoteSection non-empty
//===----------------------------------------------------------------------===//

TEST(CodeObjectHandler, RodataSectionNonEmpty) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());
  EXPECT_GT(HandlerOrErr->getRodataSection().size(), 0u);
}

TEST(CodeObjectHandler, NoteSectionNonEmpty) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());
  EXPECT_GT(HandlerOrErr->getNoteSection().size(), 0u);
}

//===----------------------------------------------------------------------===//
// H-010: build round-trip preserves loadability
//===----------------------------------------------------------------------===//

TEST(CodeObjectHandler, BuildRoundTripReloadable) {
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(gemmELF());
  ASSERT_TRUE(!!HandlerOrErr) << toString(HandlerOrErr.takeError());

  auto BuiltOrErr = HandlerOrErr->build();
  ASSERT_TRUE(!!BuiltOrErr) << toString(BuiltOrErr.takeError());

  auto Handler2OrErr = CodeObjectHandler::loadFromBytes(*BuiltOrErr);
  ASSERT_TRUE(!!Handler2OrErr) << toString(Handler2OrErr.takeError());
  EXPECT_EQ(Handler2OrErr->getGPUArch(), "gfx950");
  EXPECT_EQ(Handler2OrErr->getKernelCount(), HandlerOrErr->getKernelCount());
}
