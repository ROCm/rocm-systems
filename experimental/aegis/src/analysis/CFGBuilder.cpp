//===-- CFGBuilder.cpp - Control Flow Graph Builder -----------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/CFGBuilder.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "aegisbit-cfg"

namespace aegisbit {

CFGBuilder::CFGBuilder(Disassembler& Disasm) : Disasm(Disasm) {}

//===----------------------------------------------------------------------===//
// Instruction Classification
//
// These methods use LLVM's semantic flags (isBranch, isCall, isReturn) where
// possible, falling back to name matching only for AMDGPU-specific patterns
// that LLVM doesn't expose via flags.
//
// Performance: Each method that needs the instruction name should receive it
// as a parameter to avoid repeated getInstructionName() allocations.
//===----------------------------------------------------------------------===//

bool CFGBuilder::isTerminator(const DecodedInstruction& DI) const {
  // Use LLVM semantic check first (covers most branches)
  if (Disasm.isBranch(DI.Inst)) {
    return true;
  }

  // For AMDGPU-specific terminators not marked as branches by LLVM,
  // we need to check by name. Get name once and reuse.
  const std::string& Name = Disasm.getInstructionName(DI.Inst);

  // End of program (LLVM marks this as isBranch, but check explicitly)
  if (Name.find("S_ENDPGM") != std::string::npos) {
    return true;
  }

  // Indirect jumps (s_setpc_b64, s_swappc_b64)
  if (Name.find("S_SETPC") != std::string::npos ||
      Name.find("S_SWAPPC") != std::string::npos) {
    return true;
  }

  return false;
}

bool CFGBuilder::isUnconditionalBranch(const DecodedInstruction& DI) const {
  // Must be a branch according to LLVM
  if (!Disasm.isBranch(DI.Inst)) {
    return false;
  }

  const std::string& Name = Disasm.getInstructionName(DI.Inst);

  // S_BRANCH is unconditional, S_CBRANCH_* is conditional
  // S_ENDPGM is marked as branch by LLVM but is not a jump
  return Name.find("S_BRANCH") != std::string::npos &&
         Name.find("S_CBRANCH") == std::string::npos;
}

bool CFGBuilder::isConditionalBranch(const DecodedInstruction& DI) const {
  // Must be a branch according to LLVM
  if (!Disasm.isBranch(DI.Inst)) {
    return false;
  }

  const std::string& Name = Disasm.getInstructionName(DI.Inst);

  // All conditional branches: S_CBRANCH_SCC0, S_CBRANCH_SCC1,
  // S_CBRANCH_VCCZ, S_CBRANCH_VCCNZ, S_CBRANCH_EXECZ, S_CBRANCH_EXECNZ
  return Name.find("S_CBRANCH") != std::string::npos;
}

bool CFGBuilder::isIndirectBranch(const DecodedInstruction& DI) const {
  const std::string& Name = Disasm.getInstructionName(DI.Inst);

  // Indirect jumps: s_setpc_b64, s_swappc_b64
  return Name.find("S_SETPC") != std::string::npos ||
         Name.find("S_SWAPPC") != std::string::npos;
}

bool CFGBuilder::isEndProgram(const DecodedInstruction& DI) const {
  const std::string& Name = Disasm.getInstructionName(DI.Inst);
  return Name.find("S_ENDPGM") != std::string::npos;
}

llvm::Expected<uint64_t>
CFGBuilder::getBranchTarget(const DecodedInstruction& DI) const {
  // Verify instruction has operands before calling getBranchTarget
  if (DI.Inst.getNumOperands() == 0) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Branch instruction has no operands (indirect or malformed)");
  }

  // Use Disassembler's getBranchTarget which handles offset calculation
  auto TargetOrErr = Disasm.getBranchTarget(DI.Inst, DI.Address);
  if (!TargetOrErr) {
    return TargetOrErr.takeError();
  }

  // getBranchTarget returns signed offset-based target, cast to uint64_t
  return static_cast<uint64_t>(*TargetOrErr);
}

//===----------------------------------------------------------------------===//
// CFG Construction Algorithm
//
// Standard leader-based basic block identification:
// 1. First instruction is a leader
// 2. Target of any branch is a leader
// 3. Instruction immediately after a branch/terminator is a leader
//===----------------------------------------------------------------------===//

std::unordered_set<uint64_t> CFGBuilder::findLeaders(
    const std::vector<DecodedInstruction>& Instructions) {
  std::unordered_set<uint64_t> Leaders;

  if (Instructions.empty()) {
    return Leaders;
  }

  // Rule 1: First instruction is always a leader
  Leaders.insert(Instructions[0].Address);

  for (size_t i = 0; i < Instructions.size(); ++i) {
    const auto& DI = Instructions[i];

    if (isTerminator(DI)) {
      // Rule 2: Instruction after a branch/terminator is a leader
      if (i + 1 < Instructions.size()) {
        Leaders.insert(Instructions[i + 1].Address);
      }

      // Rule 3: Branch target is a leader
      // Skip s_endpgm (it's a terminator but has no target)
      // Skip indirect branches (can't resolve statically)
      if (Disasm.isBranch(DI.Inst) && !isIndirectBranch(DI) && !isEndProgram(DI)) {
        auto TargetOrErr = getBranchTarget(DI);
        if (TargetOrErr) {
          Leaders.insert(*TargetOrErr);
        }
        // If we can't get target, it's likely indirect or out of range
        else {
          LLVM_DEBUG(llvm::dbgs() << "CFGBuilder: Cannot get branch target at 0x"
                                  << llvm::Twine::utohexstr(DI.Address)
                                  << " (likely indirect or out of range)\n");
          llvm::consumeError(TargetOrErr.takeError());
        }
      }
    }
  }

  return Leaders;
}

std::vector<BasicBlock> CFGBuilder::createBasicBlocks(
    const std::vector<DecodedInstruction>& Instructions,
    const std::unordered_set<uint64_t>& Leaders) {
  std::vector<BasicBlock> Blocks;

  if (Instructions.empty()) {
    return Blocks;
  }

  BasicBlock CurrentBB;
  CurrentBB.ID = 0;
  CurrentBB.StartAddress = Instructions[0].Address;
  CurrentBB.IsTerminal = false;

  for (const auto& DI : Instructions) {
    // Check if this instruction starts a new block
    if (!CurrentBB.Instructions.empty() &&
        Leaders.count(DI.Address) > 0) {
      // Finish current block
      CurrentBB.EndAddress = DI.Address;
      Blocks.push_back(std::move(CurrentBB));

      // Start new block
      CurrentBB = BasicBlock();
      CurrentBB.ID = static_cast<uint32_t>(Blocks.size());
      CurrentBB.StartAddress = DI.Address;
      CurrentBB.IsTerminal = false;
    }

    // Add instruction to current block
    CurrentBB.Instructions.push_back(DI);

    // Check if this instruction ends the block as terminal
    if (isEndProgram(DI)) {
      CurrentBB.IsTerminal = true;
    }
  }

  // Don't forget the last block
  if (!CurrentBB.Instructions.empty()) {
    CurrentBB.EndAddress = CurrentBB.Instructions.back().Address +
                           CurrentBB.Instructions.back().Size;
    Blocks.push_back(std::move(CurrentBB));
  }

  return Blocks;
}

void CFGBuilder::connectEdges(
    std::vector<BasicBlock>& Blocks,
    const std::unordered_map<uint64_t, uint32_t>& AddressToBlockID,
    std::vector<UnresolvedBranch>& Errors) {
  for (auto& BB : Blocks) {
    if (BB.Instructions.empty()) {
      continue;
    }

    const auto& LastInst = BB.Instructions.back();

    // Terminal blocks have no successors
    if (BB.IsTerminal) {
      continue;
    }

    // Check for indirect branch - mark as having unknown successors
    if (isIndirectBranch(LastInst)) {
      // We can't resolve indirect branches statically
      // The CFG will have no successors for this block
      // Future: could add a special "unknown" successor marker
      continue;
    }

    // Unconditional branch: only the target is a successor
    if (isUnconditionalBranch(LastInst)) {
      auto TargetOrErr = getBranchTarget(LastInst);
      if (TargetOrErr) {
        auto It = AddressToBlockID.find(*TargetOrErr);
        if (It != AddressToBlockID.end()) {
          BB.Successors.push_back(It->second);
          Blocks[It->second].Predecessors.push_back(BB.ID);
        }
        // Target not found: may be jumping outside this code section
      } else {
        // Collect error for post-hoc validation instead of silently consuming
        Errors.push_back({
            LastInst.Address,
            BB.ID,
            llvm::toString(TargetOrErr.takeError())});
      }
      continue;
    }

    // Conditional branch: two successors (fall-through + target)
    if (isConditionalBranch(LastInst)) {
      // Fall-through successor
      uint64_t FallThrough = LastInst.Address + LastInst.Size;
      auto FTIt = AddressToBlockID.find(FallThrough);
      if (FTIt != AddressToBlockID.end()) {
        BB.Successors.push_back(FTIt->second);
        Blocks[FTIt->second].Predecessors.push_back(BB.ID);
      }

      // Branch target successor
      auto TargetOrErr = getBranchTarget(LastInst);
      if (TargetOrErr) {
        uint64_t Target = *TargetOrErr;
        auto TgtIt = AddressToBlockID.find(Target);
        if (TgtIt != AddressToBlockID.end()) {
          BB.Successors.push_back(TgtIt->second);
          Blocks[TgtIt->second].Predecessors.push_back(BB.ID);
        }
        // Target not in any block: out of bounds or external
      } else {
        // Collect error for post-hoc validation instead of silently consuming
        Errors.push_back({
            LastInst.Address,
            BB.ID,
            llvm::toString(TargetOrErr.takeError())});
      }
      continue;
    }

    // Non-branch terminator or fall-through
    // The next block (if any) is the successor
    uint64_t FallThrough = LastInst.Address + LastInst.Size;
    auto It = AddressToBlockID.find(FallThrough);
    if (It != AddressToBlockID.end()) {
      BB.Successors.push_back(It->second);
      Blocks[It->second].Predecessors.push_back(BB.ID);
    }
  }
}

llvm::Expected<ControlFlowGraph> CFGBuilder::build(
    const std::vector<DecodedInstruction>& Instructions) {
  if (Instructions.empty()) {
    ControlFlowGraph CFG;
    CFG.EntryBlockID = 0;
    return CFG;
  }

  // Step 1: Find all leader addresses
  auto Leaders = findLeaders(Instructions);

  // Step 2: Create basic blocks
  auto Blocks = createBasicBlocks(Instructions, Leaders);

  // Step 3: Build address-to-block-ID map for O(1) edge connection
  std::unordered_map<uint64_t, uint32_t> AddressToBlockID;
  for (const auto& BB : Blocks) {
    AddressToBlockID[BB.StartAddress] = BB.ID;
  }

  // Step 4: Connect edges (collecting any unresolved branches)
  std::vector<UnresolvedBranch> Errors;
  connectEdges(Blocks, AddressToBlockID, Errors);

  // Build final CFG with block lookup index
  ControlFlowGraph CFG;
  CFG.BasicBlocks = std::move(Blocks);
  CFG.EntryBlockID = 0;  // First block is entry
  CFG.UnresolvedBranches = std::move(Errors);

  // Build ID-to-index map for O(1) block lookup
  CFG.buildBlockIndex();

  return CFG;
}

llvm::Expected<ControlFlowGraph> CFGBuilder::build(
    llvm::ArrayRef<uint8_t> Code, uint64_t BaseAddress) {
  // Disassemble first
  auto InstructionsOrErr = Disasm.disassembleAll(Code, BaseAddress);
  if (!InstructionsOrErr) {
    return InstructionsOrErr.takeError();
  }

  return build(*InstructionsOrErr);
}

} // namespace aegisbit
