#ifndef HOTSWAP_TRANSPILER_DECODED_INST_HPP
#define HOTSWAP_TRANSPILER_DECODED_INST_HPP

#include "amdgpu_formats.hpp"
#include "semop.hpp"

#include "llvm/MC/MCInst.h"
#include <cstdint>
#include <string>

namespace transpiler {

struct DecodedInst {
  std::string mnemonic;
  std::string rawMnemonic;
  std::string fullText;
  llvm::MCInst inst;
  SemOp semOp = SemOp::Unknown;
  unsigned numDefs = 0;
  bool isBranch = false;
  bool isConditionalBranch = false;
  uint64_t offset = 0;
  uint64_t size = 0;

  uint64_t tsFlags = 0;
  FormatKind format = FormatKind::Unknown;
  bool defsSCC = false;
  bool defsVCC = false;
  bool defsEXEC = false;
  unsigned firstSrcIdx = 0;

  static constexpr unsigned kMaxSrcs = 16;
  unsigned srcMap[kMaxSrcs] = {};
  unsigned modMap[kMaxSrcs] = {};
  unsigned numSrcs = 0;

  unsigned numOps() const { return inst.getNumOperands(); }
  bool isReg(unsigned i) const {
    return i < numOps() && inst.getOperand(i).isReg();
  }
  bool isImm(unsigned i) const {
    return i < numOps() && inst.getOperand(i).isImm();
  }
  unsigned getReg(unsigned i) const { return inst.getOperand(i).getReg(); }
  int64_t getImm(unsigned i) const { return inst.getOperand(i).getImm(); }
};

} // namespace transpiler

#endif
