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

  // Upper bound on the logical-source count the raiser's walk can produce.
  // Actual value is conservatively sized so it never clips any AMDGPU opcode
  // LLVM ships today; the bound is checked at MCState init time against the
  // widest `NumOperands - NumDefs` in MCInstrInfo, so a future LLVM that adds
  // a wider encoding will fatal at startup rather than silently truncate. See
  // `initMCState` for the check. If you bump the bound here, keep it as a
  // safe upper limit, not a tight fit: the startup assertion already makes
  // drift visible.
  static constexpr unsigned kMaxSrcs = 24;
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
