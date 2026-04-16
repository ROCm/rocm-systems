//===-- PayloadCompilerGTest.cpp - PayloadCompiler Tests ---------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for PayloadCompiler: IR building and AMDGPU code generation.
/// All tests are CPU-only (compilation is pure LLVM, no GPU needed).
///
//===----------------------------------------------------------------------===//

#include "aegisbit/PayloadCompiler.h"
#include "llvm/IR/Module.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace aegisbit;
using namespace llvm;

//===----------------------------------------------------------------------===//
// P-001: create() – success and failure paths
//===----------------------------------------------------------------------===//

TEST(PayloadCompiler, CreateGfx942Succeeds) {
  auto PCOrErr = PayloadCompiler::create("gfx942");
  ASSERT_TRUE(!!PCOrErr) << toString(PCOrErr.takeError());
}

TEST(PayloadCompiler, CreateGfx950Succeeds) {
  auto PCOrErr = PayloadCompiler::create("gfx950");
  ASSERT_TRUE(!!PCOrErr) << toString(PCOrErr.takeError());
}

// Note: PayloadCompiler::create with an unrecognized arch string does not fail
// because LLVM's TargetMachine silently ignores unknown processors.
// The "invalid arch" error path is not testable at this level.

//===----------------------------------------------------------------------===//
// P-002: getGPUArch() returns what was passed
//===----------------------------------------------------------------------===//

TEST(PayloadCompiler, GetGPUArchGfx942) {
  auto PCOrErr = PayloadCompiler::create("gfx942");
  ASSERT_TRUE(!!PCOrErr) << toString(PCOrErr.takeError());
  EXPECT_EQ((*PCOrErr)->getGPUArch(), "gfx942");
}

TEST(PayloadCompiler, GetGPUArchGfx950) {
  auto PCOrErr = PayloadCompiler::create("gfx950");
  ASSERT_TRUE(!!PCOrErr) << toString(PCOrErr.takeError());
  EXPECT_EQ((*PCOrErr)->getGPUArch(), "gfx950");
}

//===----------------------------------------------------------------------===//
// P-003: buildCountingLoop → compile produces non-empty bytes
//===----------------------------------------------------------------------===//

TEST(PayloadCompiler, CountingLoopCompilesNonEmpty) {
  auto PCOrErr = PayloadCompiler::create("gfx942");
  ASSERT_TRUE(!!PCOrErr) << toString(PCOrErr.takeError());

  auto M = PayloadCompiler::buildCountingLoop((*PCOrErr)->getContext());
  ASSERT_NE(M, nullptr);

  auto BlobOrErr = (*PCOrErr)->compile(std::move(M));
  ASSERT_TRUE(!!BlobOrErr) << toString(BlobOrErr.takeError());
  EXPECT_GT(BlobOrErr->size(), 0u);
}

//===----------------------------------------------------------------------===//
// P-004: buildMaxPopCountLoop → compile produces non-empty bytes
//===----------------------------------------------------------------------===//

TEST(PayloadCompiler, MaxPopCountLoopCompilesNonEmpty) {
  auto PCOrErr = PayloadCompiler::create("gfx942");
  ASSERT_TRUE(!!PCOrErr) << toString(PCOrErr.takeError());

  auto M = PayloadCompiler::buildMaxPopCountLoop((*PCOrErr)->getContext());
  ASSERT_NE(M, nullptr);

  auto BlobOrErr = (*PCOrErr)->compile(std::move(M));
  ASSERT_TRUE(!!BlobOrErr) << toString(BlobOrErr.takeError());
  EXPECT_GT(BlobOrErr->size(), 0u);
}

//===----------------------------------------------------------------------===//
// P-005: buildAtomicAccumulator → compile produces non-empty bytes
//===----------------------------------------------------------------------===//

TEST(PayloadCompiler, AtomicAccumulatorCompilesNonEmpty) {
  auto PCOrErr = PayloadCompiler::create("gfx942");
  ASSERT_TRUE(!!PCOrErr) << toString(PCOrErr.takeError());

  auto M = PayloadCompiler::buildAtomicAccumulator((*PCOrErr)->getContext(),
                                                    /*UseAtomics=*/true);
  ASSERT_NE(M, nullptr);

  auto BlobOrErr = (*PCOrErr)->compile(std::move(M));
  ASSERT_TRUE(!!BlobOrErr) << toString(BlobOrErr.takeError());
  EXPECT_GT(BlobOrErr->size(), 0u);
}

TEST(PayloadCompiler, NonAtomicAccumulatorCompilesNonEmpty) {
  auto PCOrErr = PayloadCompiler::create("gfx942");
  ASSERT_TRUE(!!PCOrErr) << toString(PCOrErr.takeError());

  auto M = PayloadCompiler::buildAtomicAccumulator((*PCOrErr)->getContext(),
                                                    /*UseAtomics=*/false);
  ASSERT_NE(M, nullptr);

  auto BlobOrErr = (*PCOrErr)->compile(std::move(M));
  ASSERT_TRUE(!!BlobOrErr) << toString(BlobOrErr.takeError());
  EXPECT_GT(BlobOrErr->size(), 0u);
}

//===----------------------------------------------------------------------===//
// P-006: compiled blob is not all zeros
//===----------------------------------------------------------------------===//

TEST(PayloadCompiler, CompiledBlobNotAllZeros) {
  auto PCOrErr = PayloadCompiler::create("gfx942");
  ASSERT_TRUE(!!PCOrErr) << toString(PCOrErr.takeError());

  auto M = PayloadCompiler::buildCountingLoop((*PCOrErr)->getContext());
  auto BlobOrErr = (*PCOrErr)->compile(std::move(M));
  ASSERT_TRUE(!!BlobOrErr) << toString(BlobOrErr.takeError());

  bool AllZero = true;
  for (uint8_t B : *BlobOrErr) {
    if (B != 0) {
      AllZero = false;
      break;
    }
  }
  EXPECT_FALSE(AllZero) << "Compiled blob should not be all zeros";
}

//===----------------------------------------------------------------------===//
// P-007: compiled blob does NOT start with s_waitcnt (prologue stripped)
//===----------------------------------------------------------------------===//

TEST(PayloadCompiler, CompiledBlobPrologueStripped) {
  auto PCOrErr = PayloadCompiler::create("gfx942");
  ASSERT_TRUE(!!PCOrErr) << toString(PCOrErr.takeError());

  auto M = PayloadCompiler::buildCountingLoop((*PCOrErr)->getContext());
  auto BlobOrErr = (*PCOrErr)->compile(std::move(M));
  ASSERT_TRUE(!!BlobOrErr) << toString(BlobOrErr.takeError());
  ASSERT_GE(BlobOrErr->size(), 4u);

  // s_waitcnt encoding: 0xBF8Cxxxx
  uint32_t FirstWord;
  std::memcpy(&FirstWord, BlobOrErr->data(), 4);
  EXPECT_NE(FirstWord & 0xFFFF0000u, 0xBF8C0000u)
      << "Compiled blob should not start with s_waitcnt (prologue should be stripped)";
}

//===----------------------------------------------------------------------===//
// P-008: compile on gfx950 also works
//===----------------------------------------------------------------------===//

TEST(PayloadCompiler, CountingLoopCompilesOnGfx950) {
  auto PCOrErr = PayloadCompiler::create("gfx950");
  ASSERT_TRUE(!!PCOrErr) << toString(PCOrErr.takeError());

  auto M = PayloadCompiler::buildCountingLoop((*PCOrErr)->getContext());
  auto BlobOrErr = (*PCOrErr)->compile(std::move(M));
  ASSERT_TRUE(!!BlobOrErr) << toString(BlobOrErr.takeError());
  EXPECT_GT(BlobOrErr->size(), 0u);
}

//===----------------------------------------------------------------------===//
// P-009: getContext returns a usable context
//===----------------------------------------------------------------------===//

TEST(PayloadCompiler, GetContextReturnsUsableContext) {
  auto PCOrErr = PayloadCompiler::create("gfx942");
  ASSERT_TRUE(!!PCOrErr) << toString(PCOrErr.takeError());

  LLVMContext &Ctx = (*PCOrErr)->getContext();
  auto M = std::make_unique<Module>("test", Ctx);
  EXPECT_NE(M, nullptr);
}
