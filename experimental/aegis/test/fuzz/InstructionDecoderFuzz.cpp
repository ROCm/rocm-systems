//===-- InstructionDecoderFuzz.cpp - Fuzz Test for Disassembler --*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Fuzz test F-001: Random bytes to instruction decoder, must never crash.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/Disassembler.h"
#include <cstdint>
#include <cstddef>

using namespace aegisbit;

// Global disassembler - initialized once
static std::unique_ptr<Disassembler> G_Disasm;

extern "C" int LLVMFuzzerInitialize(int* /* argc */, char*** /* argv */) {
  // Initialize disassembler for gfx942 (MI300)
  // Triple is first argument, CPU is second
  auto DisasmOrErr = Disassembler::create("amdgcn-amd-amdhsa", "gfx942");
  if (!DisasmOrErr) {
    // Try gfx90a as fallback
    DisasmOrErr = Disassembler::create("amdgcn-amd-amdhsa", "gfx90a");
    if (!DisasmOrErr) {
      // Can't initialize - but don't crash
      llvm::consumeError(DisasmOrErr.takeError());
      return 0;
    }
  }
  G_Disasm = std::move(*DisasmOrErr);
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size) {
  if (!G_Disasm) {
    return 0;  // No disassembler available
  }

  if (Size == 0) {
    return 0;  // Empty input
  }

  // Try to decode instructions from the random bytes
  // The decoder should handle any input gracefully without crashing

  llvm::ArrayRef<uint8_t> Code(Data, Size);
  uint64_t Address = 0;

  while (!Code.empty() && Code.size() >= 4) {
    // Attempt to decode one instruction
    uint64_t InstSize = 0;
    auto InstOrErr = G_Disasm->disassemble(Code, Address, InstSize);

    if (!InstOrErr) {
      // Decode failed - this is expected for random bytes
      // Just consume the error and move to next 4-byte aligned position
      llvm::consumeError(InstOrErr.takeError());

      // Try next 4-byte aligned position (AMDGPU instructions are 4-byte aligned)
      if (Code.size() >= 4) {
        Code = Code.slice(4);
        Address += 4;
      } else {
        break;
      }
      continue;
    }

    // Decoded successfully - get instruction size and advance
    const DecodedInstruction& Inst = *InstOrErr;

    // Sanity check: instruction size should be 4 or 8 bytes for AMDGPU
    // But we don't want to crash if it's not
    if (InstSize == 0 || InstSize > Code.size()) {
      InstSize = 4;  // Fallback: advance by minimum instruction size
    }

    Code = Code.slice(InstSize);
    Address += InstSize;

    // Exercise additional decoder functionality
    // These should all handle gracefully without crashing
    (void)Inst.Category;
    (void)Inst.Size;
    (void)Inst.Address;

    // Check properties via Disassembler
    (void)G_Disasm->isBranch(Inst.Inst);
    (void)G_Disasm->isMemory(Inst.Inst);
    (void)G_Disasm->categorize(Inst.Inst);
    (void)G_Disasm->getInstructionName(Inst.Inst);

    if (G_Disasm->isBranch(Inst.Inst)) {
      auto TargetOrErr = G_Disasm->getBranchTarget(Inst.Inst, Inst.Address);
      if (!TargetOrErr) {
        llvm::consumeError(TargetOrErr.takeError());
      }
    }
  }

  return 0;
}
