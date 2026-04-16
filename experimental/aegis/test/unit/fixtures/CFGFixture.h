//===-- CFGFixture.h - CFG Builder Test Fixture ----------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Test fixture for CFG construction tests.
/// Provides common code patterns and CFG building helpers.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_TEST_CFG_FIXTURE_H
#define AEGISBIT_TEST_CFG_FIXTURE_H

#include "DisasmFixture.h"
#include "aegisbit/CFGBuilder.h"

namespace aegisbit {
namespace test {

/// Fixture for CFG construction tests.
/// Inherits disassembler setup and adds CFG-specific helpers.
class CFGFixture : public DisasmFixture {
protected:
  std::unique_ptr<CFGBuilder> Builder;

  void SetUp() override {
    DisasmFixture::SetUp();
    if (Disasm) {
      Builder = std::make_unique<CFGBuilder>(*Disasm);
    }
  }

  /// Build CFG from code bytes.
  llvm::Expected<ControlFlowGraph> buildCFG(llvm::ArrayRef<uint8_t> Code,
                                             uint64_t BaseAddr = 0) {
    EXPECT_TRUE(Builder) << "CFGBuilder not initialized";
    if (!Builder) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                      "CFGBuilder not initialized");
    }
    return Builder->build(Code, BaseAddr);
  }

  /// Build CFG from decoded instructions.
  llvm::Expected<ControlFlowGraph> buildCFG(
      const std::vector<DecodedInstruction>& Instructions) {
    EXPECT_TRUE(Builder) << "CFGBuilder not initialized";
    if (!Builder) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                      "CFGBuilder not initialized");
    }
    return Builder->build(Instructions);
  }

  //===--------------------------------------------------------------------===//
  // Common Code Patterns
  //===--------------------------------------------------------------------===//

  /// Straight-line code: nop; nop; endpgm
  std::vector<uint8_t> makeStraightLine() {
    return concat({
        encode("S_NOP", {IB::Operand::Imm(0)}),
        encode("S_NOP", {IB::Operand::Imm(0)}),
        encode("S_ENDPGM")
    });
  }

  /// Unconditional branch: branch +1; nop; endpgm
  /// Branch skips the nop, goes to endpgm
  std::vector<uint8_t> makeUnconditionalBranch() {
    return concat({
        encode("S_BRANCH", {IB::Operand::Imm(1)}),  // Skip 1 dword
        encode("S_NOP", {IB::Operand::Imm(0)}),
        encode("S_ENDPGM")
    });
  }

  /// Conditional branch: cbranch_scc0 +1; nop; endpgm
  /// If SCC==0, skip the nop
  std::vector<uint8_t> makeConditionalBranch() {
    return concat({
        encode("S_CBRANCH_SCC0", {IB::Operand::Imm(1)}),
        encode("S_NOP", {IB::Operand::Imm(0)}),
        encode("S_ENDPGM")
    });
  }

  /// Diamond pattern: if-then-else structure
  /// BB0: cbranch +2 (skip BB1, go to BB2)
  /// BB1: nop (then block)
  /// BB2: nop (else block)
  /// BB3: endpgm (merge)
  std::vector<uint8_t> makeDiamond() {
    return concat({
        encode("S_CBRANCH_SCC0", {IB::Operand::Imm(2)}),  // BB0: skip 2 dwords
        encode("S_NOP", {IB::Operand::Imm(0)}),            // BB1: then
        encode("S_BRANCH", {IB::Operand::Imm(1)}),         // BB1: skip else
        encode("S_NOP", {IB::Operand::Imm(0)}),            // BB2: else
        encode("S_ENDPGM")                                  // BB3: merge
    });
  }

  /// Simple loop: while(true) { nop; }
  /// BB0: nop; branch -2 (back to start)
  std::vector<uint8_t> makeInfiniteLoop() {
    return concat({
        encode("S_NOP", {IB::Operand::Imm(0)}),
        encode("S_BRANCH", {IB::Operand::Imm(-2)})  // Back 2 dwords
    });
  }

  /// Conditional loop: while(SCC) { nop; }
  /// BB0: cbranch_scc0 +2 (exit if SCC==0)
  /// BB0: nop
  /// BB0: branch -3 (back to condition)
  /// BB1: endpgm
  std::vector<uint8_t> makeConditionalLoop() {
    return concat({
        encode("S_CBRANCH_SCC0", {IB::Operand::Imm(2)}),  // Exit loop
        encode("S_NOP", {IB::Operand::Imm(0)}),           // Loop body
        encode("S_BRANCH", {IB::Operand::Imm(-3)}),       // Back to start
        encode("S_ENDPGM")
    });
  }
};

} // namespace test
} // namespace aegisbit

#endif // AEGISBIT_TEST_CFG_FIXTURE_H
