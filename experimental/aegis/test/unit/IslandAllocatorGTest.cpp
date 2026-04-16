//===-- IslandAllocatorGTest.cpp - Island Allocator Tests ---------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/IslandAllocator.h"
#include "gtest/gtest.h"

using namespace aegisbit;

namespace {

class IslandAllocatorTest : public ::testing::Test {};

TEST_F(IslandAllocatorTest, InitialState) {
  // BaseAddr=0, TextSectionSize=1024, PreKernelSpace=0
  IslandAllocator Alloc(0, 1024, 0);

  EXPECT_EQ(Alloc.getCursor(), 0u);
  // Island start should be 1024 aligned up to 256 = 1024 (already aligned)
  EXPECT_EQ(Alloc.getIslandStart(), 1024u);
  EXPECT_EQ(Alloc.getCurrentAbsolute(), 1024u);
  EXPECT_FALSE(Alloc.isBackwardMode());
}

TEST_F(IslandAllocatorTest, InitialStateUnalignedTextSize) {
  IslandAllocator Alloc(0, 1000, 0);
  // (1000 + 255) & ~255 = 1024
  EXPECT_EQ(Alloc.getIslandStart(), 1024u);
  EXPECT_EQ(Alloc.getCursor(), 0u);
}

TEST_F(IslandAllocatorTest, CommitSlotAdvancesCursor) {
  IslandAllocator Alloc(0, 1024, 0);

  TrampolineSlot Slot;
  Slot.TrampolineBytes.resize(512, 0xCC);
  Alloc.commitSlot(std::move(Slot), 512);

  EXPECT_EQ(Alloc.getCursor(), 512u);
  EXPECT_EQ(Alloc.getCurrentAbsolute(), 1024u + 512u);
}

TEST_F(IslandAllocatorTest, FinalizeResetsIslandCursor) {
  IslandAllocator Alloc(0, 1024, 0);

  TrampolineSlot Slot;
  Slot.TrampolineBytes.resize(256, 0xCC);
  Alloc.commitSlot(std::move(Slot), 256);
  EXPECT_EQ(Alloc.getCursor(), 256u);

  Alloc.finalizeCurrentIsland();
  EXPECT_EQ(Alloc.getCursor(), 0u);

  ASSERT_EQ(Alloc.getIslands().size(), 1u);
  EXPECT_EQ(Alloc.getIslands()[0].Offset, 1024u);
  EXPECT_EQ(Alloc.getIslands()[0].Bytes.size(), 256u);
}

TEST_F(IslandAllocatorTest, FinalizeEmptyIsNoop) {
  IslandAllocator Alloc(0, 1024, 0);
  Alloc.finalizeCurrentIsland();
  EXPECT_TRUE(Alloc.getIslands().empty());
  EXPECT_EQ(Alloc.getCursor(), 0u);
}

TEST_F(IslandAllocatorTest, StartNewIslandForwardProgression) {
  IslandAllocator Alloc(0, 1024, 0);

  TrampolineSlot Slot;
  Slot.TrampolineBytes.resize(512, 0xCC);
  Alloc.commitSlot(std::move(Slot), 512);

  uint64_t OldStart = Alloc.getIslandStart();
  Alloc.startNewIsland();

  EXPECT_GT(Alloc.getIslandStart(), OldStart);
  EXPECT_EQ(Alloc.getIslandStart() % 256, 0u);
  EXPECT_EQ(Alloc.getCursor(), 0u);
  EXPECT_EQ(Alloc.getIslands().size(), 1u);
}

TEST_F(IslandAllocatorTest, StartNewIslandBackwardProgression) {
  // Kernel at offset 200KB, pre-kernel space available
  uint64_t BaseAddr = 200 * 1024;
  uint64_t TextEnd = BaseAddr + 4096;
  IslandAllocator Alloc(BaseAddr, TextEnd, BaseAddr);

  // Switch to backward mode
  Alloc.switchToBackward(BaseAddr);
  EXPECT_TRUE(Alloc.isBackwardMode());
  uint64_t BackStart = Alloc.getIslandStart();
  EXPECT_LT(BackStart, BaseAddr);

  TrampolineSlot Slot;
  Slot.TrampolineBytes.resize(256, 0xCC);
  Alloc.commitSlot(std::move(Slot), 256);

  Alloc.startNewIsland();
  EXPECT_LE(Alloc.getIslandStart(), BackStart);
  EXPECT_EQ(Alloc.getIslandStart() % 256, 0u);
  EXPECT_EQ(Alloc.getCursor(), 0u);
}

TEST_F(IslandAllocatorTest, BranchDwordCalculation) {
  IslandAllocator Alloc(0, 1024, 0);
  // Island starts at 1024, cursor = 0, so trampoline is at 1024
  // Patch site at address 100
  // byte offset = 1024 - 100 = 924
  // dword = (924 - 4) / 4 = 230
  EXPECT_EQ(Alloc.branchDword(100), 230);
  EXPECT_EQ(Alloc.byteOffset(100), 924);
}

TEST_F(IslandAllocatorTest, BranchRangeClassification) {
  EXPECT_TRUE(IslandAllocator::inBranchRange(0));
  EXPECT_TRUE(IslandAllocator::inBranchRange(32767));
  EXPECT_TRUE(IslandAllocator::inBranchRange(-32768));
  EXPECT_FALSE(IslandAllocator::inBranchRange(32768));
  EXPECT_FALSE(IslandAllocator::inBranchRange(-32769));
}

TEST_F(IslandAllocatorTest, TryResolveOverflowStartsNewIsland) {
  IslandAllocator Alloc(0, 1024, 256 * 1024);

  // Commit some bytes so cursor > 0
  TrampolineSlot Slot;
  Slot.TrampolineBytes.resize(512, 0xCC);
  Alloc.commitSlot(std::move(Slot), 512);

  uint32_t LastRetry = UINT32_MAX;
  uint32_t RetryCount = 0;

  // Should start new island and return true (retry)
  bool Retry = Alloc.tryResolveOverflow(500, 0, LastRetry, RetryCount, 4);
  EXPECT_TRUE(Retry);
  EXPECT_EQ(Alloc.getCursor(), 0u);
  EXPECT_EQ(Alloc.getIslands().size(), 1u);
}

TEST_F(IslandAllocatorTest, TryResolveOverflowSwitchesToBackward) {
  IslandAllocator Alloc(100 * 1024, 200 * 1024, 100 * 1024);

  uint32_t LastRetry = UINT32_MAX;
  uint32_t RetryCount = 0;

  // Cursor = 0, no backward mode yet, pre-kernel > 0 -> should switch to backward
  bool Retry = Alloc.tryResolveOverflow(100 * 1024, 0, LastRetry, RetryCount, 4);
  EXPECT_TRUE(Retry);
  EXPECT_TRUE(Alloc.isBackwardMode());
}

TEST_F(IslandAllocatorTest, TryResolveOverflowReturnsRelayFallback) {
  // No pre-kernel space, cursor=0, forward mode
  IslandAllocator Alloc(0, 1024, 0);

  uint32_t LastRetry = UINT32_MAX;
  uint32_t RetryCount = 0;

  // Cursor = 0, not backward, no pre-kernel => relay fallback
  bool Retry = Alloc.tryResolveOverflow(0, 0, LastRetry, RetryCount, 4);
  EXPECT_FALSE(Retry);
}

TEST_F(IslandAllocatorTest, WouldOverlapKernel) {
  uint64_t BaseAddr = 100 * 1024;
  IslandAllocator Alloc(BaseAddr, BaseAddr + 4096, BaseAddr);

  // Switch to backward mode close to kernel
  Alloc.switchToBackward(BaseAddr);

  // Place island just before kernel
  if (Alloc.getIslandStart() + 65536 > BaseAddr) {
    EXPECT_TRUE(Alloc.wouldOverlapKernel(65536));
  }

  EXPECT_FALSE(Alloc.wouldOverlapKernel(0));
}

TEST_F(IslandAllocatorTest, ComputeBodyIslandStart) {
  IslandAllocator Alloc(0, 1024, 0);

  TrampolineSlot Slot;
  Slot.TrampolineBytes.resize(300, 0xCC);
  Alloc.commitSlot(std::move(Slot), 300);
  Alloc.finalizeCurrentIsland();

  // Island at 1024, size 300 => end = 1324
  // Body start = (1324 + 255) & ~255 = 1536
  uint64_t BodyStart = Alloc.computeBodyIslandStart();
  EXPECT_EQ(BodyStart % 256, 0u);
  EXPECT_GE(BodyStart, 1024u + 300u);
}

} // namespace
