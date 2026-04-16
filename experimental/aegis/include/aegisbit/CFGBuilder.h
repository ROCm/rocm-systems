//===-- aegisbit/CFGBuilder.h - Control Flow Graph Builder ------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Control flow graph construction from disassembled instruction stream.
///
/// Identifies basic blocks from branch instructions and constructs a CFG
/// with edges representing control flow. Handles:
/// - Unconditional branches (s_branch)
/// - Conditional branches (s_cbranch_*)
/// - Terminal instructions (s_endpgm)
/// - Indirect branches (s_setpc_b64) - flagged as unresolved
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_CFG_BUILDER_H
#define AEGISBIT_CFG_BUILDER_H

#include "aegisbit/Disassembler.h"
#include "aegisbit/Types.h"
#include "llvm/Support/Error.h"
#include <unordered_set>
#include <unordered_map>

namespace aegisbit {

/// Builder for control flow graphs from instruction streams
class CFGBuilder {
public:
  /// Construct CFG builder with a disassembler reference
  explicit CFGBuilder(Disassembler& Disasm);

  /// Build CFG from a vector of decoded instructions
  /// \param Instructions Pre-decoded instruction stream (in program order)
  /// \return Constructed CFG or error
  llvm::Expected<ControlFlowGraph> build(
      const std::vector<DecodedInstruction>& Instructions);

  /// Build CFG from raw bytes (convenience wrapper)
  /// \param Code Raw instruction bytes
  /// \param BaseAddress Starting PC address
  /// \return Constructed CFG or error
  llvm::Expected<ControlFlowGraph> build(
      llvm::ArrayRef<uint8_t> Code,
      uint64_t BaseAddress = 0);

  /// Check if an instruction is a terminator (ends a basic block)
  /// Includes: branches, s_endpgm, indirect jumps
  bool isTerminator(const DecodedInstruction& DI) const;

  /// Check if an instruction is an unconditional branch
  bool isUnconditionalBranch(const DecodedInstruction& DI) const;

  /// Check if an instruction is a conditional branch
  bool isConditionalBranch(const DecodedInstruction& DI) const;

  /// Check if an instruction is an indirect branch (unresolved target)
  bool isIndirectBranch(const DecodedInstruction& DI) const;

  /// Check if an instruction terminates the program (s_endpgm)
  bool isEndProgram(const DecodedInstruction& DI) const;

private:
  Disassembler& Disasm;

  /// Find all basic block leader addresses
  /// Leaders are: first instruction, branch targets, instructions after branches
  std::unordered_set<uint64_t> findLeaders(
      const std::vector<DecodedInstruction>& Instructions);

  /// Create basic blocks from instruction stream using leader set
  std::vector<BasicBlock> createBasicBlocks(
      const std::vector<DecodedInstruction>& Instructions,
      const std::unordered_set<uint64_t>& Leaders);

  /// Connect basic blocks with edges based on control flow
  /// Errors are collected in UnresolvedBranches rather than being discarded
  void connectEdges(std::vector<BasicBlock>& Blocks,
                    const std::unordered_map<uint64_t, uint32_t>& AddressToBlockID,
                    std::vector<UnresolvedBranch>& UnresolvedBranches);

  /// Get branch target address for a branch instruction
  llvm::Expected<uint64_t> getBranchTarget(const DecodedInstruction& DI) const;
};

} // namespace aegisbit

#endif // AEGISBIT_CFG_BUILDER_H
