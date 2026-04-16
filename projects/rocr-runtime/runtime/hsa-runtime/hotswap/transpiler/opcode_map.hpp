#ifndef HOTSWAP_TRANSPILER_OPCODE_MAP_HPP
#define HOTSWAP_TRANSPILER_OPCODE_MAP_HPP

#include "semop.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringMap.h"

namespace transpiler {

struct OpcodeMap {
  llvm::DenseMap<unsigned, SemOp> map;

  SemOp lookup(unsigned opcode) const;
  void build(const llvm::MCInstrInfo &mcii);

private:
  static llvm::StringRef stripMCIISuffix(llvm::StringRef name);
};

} // namespace transpiler

#endif
