//===-- ExploreVALUTest.cpp - Explore VALU Instruction Building -*- C++ -*-===//
//
// Exploratory test to understand what's needed to build VALU/VMEM instructions
//
//===----------------------------------------------------------------------===//

#include "aegisbit/Disassembler.h"
#include "aegisbit/InstructionBuilder.h"
#include <iostream>

using namespace aegisbit;

int main() {
  auto Disasm = Disassembler::create();
  if (!Disasm) {
    std::cerr << "Failed to create disassembler\n";
    return 1;
  }

  auto& MCII = (*Disasm)->getMCII();
  auto& MRI = (*Disasm)->getMRI();

  std::cout << "=== Exploring VALU Instruction Requirements ===\n\n";

  // 1. Find V_ADD_F32 instruction variants
  std::cout << "1. V_ADD_F32 variants:\n";
  for (unsigned i = 0; i < MCII.getNumOpcodes(); ++i) {
    const char* Name = MCII.getName(i).data();
    if (Name && std::string(Name).find("V_ADD_F32") == 0) {
      const auto& Desc = MCII.get(i);
      std::cout << "   " << Name << " (opcode " << i << ")"
                << " - " << Desc.getNumOperands() << " operands"
                << " - " << (Desc.isPseudo() ? "PSEUDO" : "REAL")
                << "\n";
    }
  }

  // 2. Find VGPR register numbers
  std::cout << "\n2. VGPR register numbering:\n";
  int vgprCount = 0;
  for (unsigned i = 0; i < 500 && vgprCount < 5; ++i) {
    const char* Name = MRI.getName(i);
    if (Name) {
      std::string RegName(Name);
      if (RegName.find("VGPR") == 0) {
        std::cout << "   " << RegName << " = register " << i << "\n";
        vgprCount++;
      }
    }
  }

  // 3. Find SGPR register numbers
  std::cout << "\n3. SGPR register numbering:\n";
  int sgprCount = 0;
  for (unsigned i = 0; i < 500 && sgprCount < 5; ++i) {
    const char* Name = MRI.getName(i);
    if (Name) {
      std::string RegName(Name);
      if (RegName.find("SGPR") == 0 && RegName.find("SGPR_") == std::string::npos) {
        std::cout << "   " << RegName << " = register " << i << "\n";
        sgprCount++;
      }
    }
  }

  // 4. Try to build V_MOV_B32 (simplest VALU)
  std::cout << "\n4. Attempting to build V_MOV_B32:\n";
  auto MovOrErr = InstructionBuilder::build(*(*Disasm), "V_MOV_B32", {});
  if (MovOrErr) {
    std::cout << "   ✓ Found instruction: " << (*Disasm)->getInstructionName(*MovOrErr) << "\n";
    const auto& Desc = MCII.get(MovOrErr->getOpcode());
    std::cout << "   ✓ Needs " << Desc.getNumOperands() << " operands\n";

    // Try to add operands
    std::cout << "   ✓ Operand types:\n";
    for (unsigned i = 0; i < Desc.getNumOperands(); ++i) {
      std::cout << "     - Operand " << i << "\n";
    }
  } else {
    std::cout << "   ✗ Failed to find V_MOV_B32\n";
    llvm::consumeError(MovOrErr.takeError());
  }

  // 5. Find memory instructions
  std::cout << "\n5. Sample memory instructions:\n";
  std::vector<std::string> MemInstrs = {"BUFFER_LOAD_DWORD", "FLAT_LOAD_DWORD",
                                         "GLOBAL_LOAD_DWORD", "DS_READ_B32"};
  for (const auto& SearchName : MemInstrs) {
    bool found = false;
    for (unsigned i = 0; i < MCII.getNumOpcodes() && !found; ++i) {
      const char* Name = MCII.getName(i).data();
      if (Name && std::string(Name).find(SearchName) == 0) {
        const auto& Desc = MCII.get(i);
        if (!Desc.isPseudo()) {
          std::cout << "   " << Name << " - " << Desc.getNumOperands() << " operands\n";
          found = true;
        }
      }
    }
    if (!found) {
      std::cout << "   " << SearchName << " - NOT FOUND\n";
    }
  }

  // 5b. Detailed GLOBAL_STORE_DWORDX2 analysis
  std::cout << "\n5b. GLOBAL_STORE_DWORDX2 variants:\n";
  for (unsigned i = 0; i < MCII.getNumOpcodes(); ++i) {
    const char* Name = MCII.getName(i).data();
    if (Name && std::string(Name).find("GLOBAL_STORE_DWORDX2") != std::string::npos) {
      const auto& Desc = MCII.get(i);
      std::cout << "   " << Name
                << " (opcode=" << i
                << ", numOps=" << Desc.getNumOperands()
                << ", isPseudo=" << Desc.isPseudo()
                << ", numDefs=" << Desc.getNumDefs()
                << ")\n";
    }
  }

  std::cout << "\n5c. GLOBAL_LOAD_DWORDX2 variants:\n";
  for (unsigned i = 0; i < MCII.getNumOpcodes(); ++i) {
    const char* Name = MCII.getName(i).data();
    if (Name && std::string(Name).find("GLOBAL_LOAD_DWORDX2") != std::string::npos) {
      const auto& Desc = MCII.get(i);
      std::cout << "   " << Name
                << " (opcode=" << i
                << ", numOps=" << Desc.getNumOperands()
                << ", isPseudo=" << Desc.isPseudo()
                << ", numDefs=" << Desc.getNumDefs()
                << ")\n";
    }
  }

  std::cout << "\n5d. S_ENDPGM variants:\n";
  for (unsigned i = 0; i < MCII.getNumOpcodes(); ++i) {
    const char* Name = MCII.getName(i).data();
    if (Name && std::string(Name).find("S_ENDPGM") == 0) {
      const auto& Desc = MCII.get(i);
      std::cout << "   " << Name
                << " (opcode=" << i
                << ", numOps=" << Desc.getNumOperands()
                << ", isPseudo=" << Desc.isPseudo()
                << ")\n";
    }
  }

  std::cout << "\n5e. S_LOAD_DWORD_IMM variants:\n";
  for (unsigned i = 0; i < MCII.getNumOpcodes(); ++i) {
    const char* Name = MCII.getName(i).data();
    if (Name && std::string(Name).find("S_LOAD_DWORD_IMM") != std::string::npos) {
      const auto& Desc = MCII.get(i);
      std::cout << "   " << Name
                << " (opcode=" << i
                << ", numOps=" << Desc.getNumOperands()
                << ", isPseudo=" << Desc.isPseudo()
                << ")\n";
    }
  }

  // 6. Summary
  std::cout << "\n=== Summary ===\n";
  std::cout << "To write VALU/VMEM tests, we need:\n";
  std::cout << "1. Register number mapping (VGPR0-255, SGPR0-103)\n";
  std::cout << "2. Understanding of multi-operand instructions\n";
  std::cout << "3. Operand type handling (reg, imm, modifier bits)\n";
  std::cout << "\nRecommendation: Can proceed, but needs register mapping helper\n";

  return 0;
}
