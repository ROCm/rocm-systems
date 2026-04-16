//===-- DisasmFixture.h - Base Disassembler Test Fixture --------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Base test fixture providing disassembler setup for GoogleTest tests.
/// All tests that need instruction encoding/decoding should inherit from this.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_TEST_DISASM_FIXTURE_H
#define AEGISBIT_TEST_DISASM_FIXTURE_H

#include "aegisbit/Disassembler.h"
#include "aegisbit/InstructionBuilder.h"
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace aegisbit {
namespace test {

/// Shorthand for InstructionBuilder
using IB = InstructionBuilder;

/// Base fixture providing disassembler and encoding helpers.
/// Inherit from this for any test that needs to work with instructions.
class DisasmFixture : public ::testing::Test {
protected:
  std::unique_ptr<Disassembler> Disasm;

  void SetUp() override {
    auto DisasmOrErr = Disassembler::create();
    ASSERT_TRUE(static_cast<bool>(DisasmOrErr))
        << "Failed to create disassembler: "
        << llvm::toString(DisasmOrErr.takeError());
    Disasm = std::move(*DisasmOrErr);
  }

  /// Encode an instruction from mnemonic and operands.
  /// Returns empty vector on failure (test will fail via EXPECT).
  ///
  /// Note: Some instructions like S_ENDPGM require an immediate operand
  /// for the real (non-pseudo) variant. If no operands are provided for
  /// these instructions, a default value (0) is automatically added.
  std::vector<uint8_t> encode(const std::string& Mnemonic,
                               std::initializer_list<IB::Operand> Ops = {}) {
    std::vector<IB::Operand> OpVec(Ops);

    // Auto-add default operand for instructions that require one
    // S_ENDPGM requires simm16 operand (typically 0)
    if (Mnemonic == "S_ENDPGM" && OpVec.empty()) {
      OpVec.push_back(IB::Operand::Imm(0));
    }

    auto InstOrErr = IB::build(*Disasm, Mnemonic, OpVec);
    EXPECT_TRUE(static_cast<bool>(InstOrErr))
        << "Failed to build " << Mnemonic;
    if (!InstOrErr) {
      llvm::consumeError(InstOrErr.takeError());
      return {};
    }

    auto BytesOrErr = Disasm->encode(*InstOrErr);
    EXPECT_TRUE(static_cast<bool>(BytesOrErr))
        << "Failed to encode " << Mnemonic;
    if (!BytesOrErr) {
      llvm::consumeError(BytesOrErr.takeError());
      return {};
    }
    return *BytesOrErr;
  }

  /// Concatenate multiple byte vectors into one.
  static std::vector<uint8_t> concat(
      std::initializer_list<std::vector<uint8_t>> Vecs) {
    std::vector<uint8_t> Result;
    for (const auto& V : Vecs) {
      Result.insert(Result.end(), V.begin(), V.end());
    }
    return Result;
  }

  /// Decode a single instruction.
  llvm::Expected<DecodedInstruction> decodeOne(
      llvm::ArrayRef<uint8_t> Code, uint64_t Addr = 0) {
    uint64_t Size = 0;
    return Disasm->disassemble(Code, Addr, Size);
  }

  /// Decode all instructions and verify success.
  llvm::Expected<std::vector<DecodedInstruction>> decodeAll(
      llvm::ArrayRef<uint8_t> Code, uint64_t Addr = 0) {
    return Disasm->disassembleAll(Code, Addr);
  }

  /// Get instruction name for pretty printing.
  std::string getName(const llvm::MCInst& Inst) {
    return Disasm->getInstructionName(Inst);
  }
};

} // namespace test
} // namespace aegisbit

#endif // AEGISBIT_TEST_DISASM_FIXTURE_H
