//===-- CFGBuilderGTest.cpp - CFG Builder Tests (GoogleTest) ----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for CFG construction using GoogleTest.
/// Tests use InstructionBuilder to create architecture-agnostic instruction
/// sequences, then verify CFG structure.
///
/// Test IDs: C-001 through C-008 (Part A: CFG construction tests)
///
//===----------------------------------------------------------------------===//

#include "fixtures/CFGFixture.h"
#include <gtest/gtest.h>

using namespace aegisbit;
using namespace aegisbit::test;

class CFGTest : public CFGFixture {};

//===----------------------------------------------------------------------===//
// C-001: Straight-line code (no branches)
//===----------------------------------------------------------------------===//

TEST_F(CFGTest, C001_StraightLineCode) {
  // Build: s_nop 0; s_nop 0; s_endpgm
  auto Code = makeStraightLine();
  ASSERT_EQ(Code.size(), 12u) << "Expected 3 x 4-byte instructions";

  auto CFGOrErr = buildCFG(Code);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr))
      << llvm::toString(CFGOrErr.takeError());

  const auto& CFG = *CFGOrErr;

  // Should have exactly 1 basic block
  ASSERT_EQ(CFG.BasicBlocks.size(), 1u);

  const auto& BB = CFG.BasicBlocks[0];

  // Block should have 3 instructions
  EXPECT_EQ(BB.Instructions.size(), 3u);

  // Block should be terminal (ends with s_endpgm)
  EXPECT_TRUE(BB.IsTerminal);

  // No successors
  EXPECT_TRUE(BB.Successors.empty());

  // Verify O(1) block lookup works
  EXPECT_NE(CFG.getBlock(0), nullptr);
  EXPECT_EQ(CFG.getBlock(1), nullptr);
}

//===----------------------------------------------------------------------===//
// C-002: Unconditional branch
//===----------------------------------------------------------------------===//

TEST_F(CFGTest, C002_UnconditionalBranch) {
  // Build: s_branch +1 (skip next); s_nop 0; s_endpgm
  auto Code = makeUnconditionalBranch();
  ASSERT_EQ(Code.size(), 12u);

  auto CFGOrErr = buildCFG(Code);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr))
      << llvm::toString(CFGOrErr.takeError());

  const auto& CFG = *CFGOrErr;

  // Should have 3 basic blocks:
  // BB0: s_branch (jumps to BB2)
  // BB1: s_nop (unreachable, but still a block)
  // BB2: s_endpgm
  ASSERT_EQ(CFG.BasicBlocks.size(), 3u);

  // BB0 should have BB2 as successor (target of branch)
  const auto& BB0 = CFG.BasicBlocks[0];
  ASSERT_EQ(BB0.Successors.size(), 1u);
  EXPECT_EQ(BB0.Successors[0], 2u) << "BB0 should jump to BB2";

  // BB2 should be terminal
  EXPECT_TRUE(CFG.BasicBlocks[2].IsTerminal);

  // Verify block lookup
  EXPECT_EQ(CFG.getBlock(2), &CFG.BasicBlocks[2]);
}

//===----------------------------------------------------------------------===//
// C-003: Conditional branch
//===----------------------------------------------------------------------===//

TEST_F(CFGTest, C003_ConditionalBranch) {
  // Build: s_cbranch_scc0 +1; s_nop 0; s_endpgm
  auto Code = makeConditionalBranch();
  ASSERT_FALSE(Code.empty());

  auto CFGOrErr = buildCFG(Code);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr))
      << llvm::toString(CFGOrErr.takeError());

  const auto& CFG = *CFGOrErr;

  // Should have 3 basic blocks
  ASSERT_EQ(CFG.BasicBlocks.size(), 3u);

  // BB0 should have 2 successors (fall-through + branch target)
  const auto& BB0 = CFG.BasicBlocks[0];
  ASSERT_EQ(BB0.Successors.size(), 2u);

  // Verify successors are BB1 (fall-through) and BB2 (target)
  bool hasBB1 = std::find(BB0.Successors.begin(), BB0.Successors.end(), 1u)
                != BB0.Successors.end();
  bool hasBB2 = std::find(BB0.Successors.begin(), BB0.Successors.end(), 2u)
                != BB0.Successors.end();

  EXPECT_TRUE(hasBB1) << "BB0 should have BB1 as successor (fall-through)";
  EXPECT_TRUE(hasBB2) << "BB0 should have BB2 as successor (branch target)";
}

//===----------------------------------------------------------------------===//
// C-004: Simple loop (back-edge)
//===----------------------------------------------------------------------===//

TEST_F(CFGTest, C004_SimpleLoop) {
  // Build: s_nop; s_cbranch_scc0 -2; s_endpgm
  // The branch goes back to the s_nop (offset -2 dwords from PC+4)
  auto Code = concat({
      encode("S_NOP", {IB::Operand::Imm(0)}),
      encode("S_CBRANCH_SCC0", {IB::Operand::Imm(-2)}),
      encode("S_ENDPGM", {})
  });
  ASSERT_EQ(Code.size(), 12u);

  auto CFGOrErr = buildCFG(Code);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr))
      << llvm::toString(CFGOrErr.takeError());

  const auto& CFG = *CFGOrErr;

  // Should have 2 basic blocks:
  // BB0: s_nop, s_cbranch (loop body) - branch target is BB0 itself
  // BB1: s_endpgm (exit)
  ASSERT_EQ(CFG.BasicBlocks.size(), 2u);

  const auto& BB0 = CFG.BasicBlocks[0];

  // BB0 should have 2 instructions
  EXPECT_EQ(BB0.Instructions.size(), 2u);

  // BB0 should have 2 successors: BB0 (back-edge) and BB1 (fall-through)
  ASSERT_EQ(BB0.Successors.size(), 2u);

  bool hasBackEdge = std::find(BB0.Successors.begin(), BB0.Successors.end(), 0u)
                     != BB0.Successors.end();
  bool hasFallThrough = std::find(BB0.Successors.begin(), BB0.Successors.end(), 1u)
                        != BB0.Successors.end();

  EXPECT_TRUE(hasBackEdge) << "Should have back-edge to BB0";
  EXPECT_TRUE(hasFallThrough) << "Should have fall-through to BB1";

  // BB1 should be terminal
  EXPECT_TRUE(CFG.BasicBlocks[1].IsTerminal);
}

//===----------------------------------------------------------------------===//
// C-005: Diamond pattern (if-then-else)
//===----------------------------------------------------------------------===//

TEST_F(CFGTest, C005_DiamondPattern) {
  // Diamond: cbranch -> then/else -> merge -> endpgm
  auto Code = makeDiamond();
  ASSERT_FALSE(Code.empty());

  auto CFGOrErr = buildCFG(Code);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr))
      << llvm::toString(CFGOrErr.takeError());

  const auto& CFG = *CFGOrErr;

  // Diamond has multiple blocks
  EXPECT_GE(CFG.BasicBlocks.size(), 3u)
      << "Diamond pattern should have at least 3 blocks";

  // Entry block should have 2 successors (then/else branches)
  const auto& Entry = CFG.BasicBlocks[0];
  EXPECT_EQ(Entry.Successors.size(), 2u)
      << "Entry block should have 2 successors for if-then-else";
}

//===----------------------------------------------------------------------===//
// C-006: Nested loops
//===----------------------------------------------------------------------===//

TEST_F(CFGTest, C006_NestedStructure) {
  // Build a slightly more complex structure with multiple branches
  auto Code = concat({
      encode("S_CBRANCH_SCC0", {IB::Operand::Imm(2)}),  // Skip 2 dwords
      encode("S_NOP", {IB::Operand::Imm(0)}),
      encode("S_CBRANCH_SCC1", {IB::Operand::Imm(1)}),  // Skip 1 dword
      encode("S_NOP", {IB::Operand::Imm(0)}),
      encode("S_ENDPGM", {})
  });
  ASSERT_FALSE(Code.empty());

  auto CFGOrErr = buildCFG(Code);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr))
      << llvm::toString(CFGOrErr.takeError());

  const auto& CFG = *CFGOrErr;

  // Should have multiple blocks due to branch targets
  EXPECT_GE(CFG.BasicBlocks.size(), 3u);

  // Last block should be terminal
  EXPECT_TRUE(CFG.BasicBlocks.back().IsTerminal);
}

//===----------------------------------------------------------------------===//
// C-007: Infinite loop detection
//===----------------------------------------------------------------------===//

TEST_F(CFGTest, C007_InfiniteLoop) {
  // Infinite loop: nop; branch -2 (back to start)
  auto Code = makeInfiniteLoop();
  ASSERT_FALSE(Code.empty());

  auto CFGOrErr = buildCFG(Code);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr))
      << llvm::toString(CFGOrErr.takeError());

  const auto& CFG = *CFGOrErr;

  // Should have 1 block with a self-loop
  ASSERT_EQ(CFG.BasicBlocks.size(), 1u);

  const auto& BB = CFG.BasicBlocks[0];

  // Should have itself as successor (back-edge)
  ASSERT_EQ(BB.Successors.size(), 1u);
  EXPECT_EQ(BB.Successors[0], 0u) << "Should loop back to itself";

  // Not terminal (no endpgm)
  EXPECT_FALSE(BB.IsTerminal);
}

//===----------------------------------------------------------------------===//
// C-008: s_endpgm terminates (no successors)
//===----------------------------------------------------------------------===//

TEST_F(CFGTest, C008_EndpgmTerminates) {
  // Just s_endpgm
  auto Code = encode("S_ENDPGM", {});
  ASSERT_EQ(Code.size(), 4u);

  auto CFGOrErr = buildCFG(Code);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr))
      << llvm::toString(CFGOrErr.takeError());

  const auto& CFG = *CFGOrErr;

  ASSERT_EQ(CFG.BasicBlocks.size(), 1u);

  const auto& BB = CFG.BasicBlocks[0];
  EXPECT_TRUE(BB.IsTerminal);
  EXPECT_TRUE(BB.Successors.empty());
  EXPECT_EQ(BB.Instructions.size(), 1u);
}

//===----------------------------------------------------------------------===//
// Additional CFG Tests
//===----------------------------------------------------------------------===//

TEST_F(CFGTest, EmptyCFG) {
  // Empty code should produce empty CFG
  std::vector<uint8_t> Empty;
  auto CFGOrErr = buildCFG(Empty);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr));

  EXPECT_TRUE(CFGOrErr->BasicBlocks.empty());
}

TEST_F(CFGTest, BlockAddresses) {
  // Verify block start/end addresses are correct
  auto Code = makeUnconditionalBranch();  // 3 blocks
  auto CFGOrErr = buildCFG(Code);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr));

  const auto& CFG = *CFGOrErr;
  ASSERT_EQ(CFG.BasicBlocks.size(), 3u);

  // BB0 starts at 0
  EXPECT_EQ(CFG.BasicBlocks[0].StartAddress, 0u);

  // BB1 starts at 4 (after 4-byte branch)
  EXPECT_EQ(CFG.BasicBlocks[1].StartAddress, 4u);

  // BB2 starts at 8 (after 4-byte nop)
  EXPECT_EQ(CFG.BasicBlocks[2].StartAddress, 8u);
}

TEST_F(CFGTest, PredecessorEdges) {
  // Verify predecessor edges are set correctly
  auto Code = makeConditionalBranch();  // cbranch +1; nop; endpgm
  auto CFGOrErr = buildCFG(Code);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr));

  const auto& CFG = *CFGOrErr;
  ASSERT_EQ(CFG.BasicBlocks.size(), 3u);

  // BB0 has no predecessors (entry)
  EXPECT_TRUE(CFG.BasicBlocks[0].Predecessors.empty());

  // BB1 has BB0 as predecessor (fall-through)
  EXPECT_EQ(CFG.BasicBlocks[1].Predecessors.size(), 1u);
  if (!CFG.BasicBlocks[1].Predecessors.empty()) {
    EXPECT_EQ(CFG.BasicBlocks[1].Predecessors[0], 0u);
  }

  // BB2 has predecessors (BB0 branch target, BB1 fall-through)
  EXPECT_GE(CFG.BasicBlocks[2].Predecessors.size(), 1u);
}

TEST_F(CFGTest, BlockLookupPerformance) {
  // Verify O(1) block lookup via index
  auto Code = concat({
      encode("S_CBRANCH_SCC0", {IB::Operand::Imm(3)}),
      encode("S_NOP", {IB::Operand::Imm(0)}),
      encode("S_CBRANCH_SCC1", {IB::Operand::Imm(1)}),
      encode("S_NOP", {IB::Operand::Imm(0)}),
      encode("S_NOP", {IB::Operand::Imm(0)}),
      encode("S_ENDPGM", {})
  });

  auto CFGOrErr = buildCFG(Code);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr));

  const auto& CFG = *CFGOrErr;

  // All valid block IDs should be findable
  for (const auto& BB : CFG.BasicBlocks) {
    EXPECT_EQ(CFG.getBlock(BB.ID), &BB)
        << "getBlock(" << BB.ID << ") should return correct block";
  }

  // Invalid IDs should return nullptr
  EXPECT_EQ(CFG.getBlock(999), nullptr);
}

TEST_F(CFGTest, ConditionalLoopStructure) {
  // While-style loop: condition check, body, back-edge
  auto Code = makeConditionalLoop();

  auto CFGOrErr = buildCFG(Code);
  ASSERT_TRUE(static_cast<bool>(CFGOrErr));

  const auto& CFG = *CFGOrErr;

  // Should have at least 2 blocks (loop body + exit)
  EXPECT_GE(CFG.BasicBlocks.size(), 2u);

  // Find block with back-edge
  bool hasBackEdge = false;
  for (const auto& BB : CFG.BasicBlocks) {
    for (auto Succ : BB.Successors) {
      if (Succ <= BB.ID) {
        hasBackEdge = true;
        break;
      }
    }
  }
  EXPECT_TRUE(hasBackEdge) << "Loop should have a back-edge";

  // Last block should be terminal
  EXPECT_TRUE(CFG.BasicBlocks.back().IsTerminal);
}
