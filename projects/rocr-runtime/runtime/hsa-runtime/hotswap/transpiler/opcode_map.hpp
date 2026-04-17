#ifndef HOTSWAP_TRANSPILER_OPCODE_MAP_HPP
#define HOTSWAP_TRANSPILER_OPCODE_MAP_HPP

#include "semop.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCInstrInfo.h"

namespace transpiler {

// Maps raw LLVM MC opcodes emitted by the AMDGPU disassembler to
// architecture-neutral SemOp tags that the raiser dispatches on.
//
// The map is built once at raiser-initialization time from MCInstrInfo and
// TableGen-generated AMDGPU instruction tables.  No string parsing is involved:
// encoding variants (e32/e64/DPP/SDWA/subtarget-specific real encodings) are
// folded onto their canonical pseudo via the AMDGPU::InstrMapping helpers
// (getMCOpcode, getVOPe64, getDPPOp32, getDPPOp64, getSDWAOp, getGlobalVaddrOp),
// and the resulting canonical pseudo is matched against a small compile-time
// table of AMDGPU::<opcode> enum constants.
struct OpcodeMap {
  llvm::DenseMap<unsigned, SemOp> map;

  SemOp lookup(unsigned opcode) const;
  void build(const llvm::MCInstrInfo &mcii);
};

} // namespace transpiler

#endif
