#ifndef HOTSWAP_TRANSPILER_DECODED_INST_HPP
#define HOTSWAP_TRANSPILER_DECODED_INST_HPP

#include "amdgpu_formats.hpp"
#include "semop.hpp"

#include "llvm/MC/MCInst.h"
#include <cstdint>
#include <string>

namespace transpiler {
struct VCmpMeta;
}

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
  // Non-null iff `semOp == V_CMP || semOp == V_CMPX`. Points into the
  // OpcodeMap side-table; stable for the lifetime of the map.
  const VCmpMeta *vcmp = nullptr;
  bool defsSCC = false;
  bool defsVCC = false;
  bool defsEXEC = false;
  // `scale_offset` flag from the CPol operand (gfx12+ FLAT/GLOBAL). When
  // set on a global_load/store in SADDR form, the per-lane vaddr is
  // multiplied by the access element size before being added to the
  // SGPR base. Decoded from the MCInst's `cpol` operand bit
  // `AMDGPU::CPol::SCAL` at disassembly time so handlers can branch
  // on a decoded bit instead of scanning `fullText`.
  bool hasScaleOffset = false;

  // ── DPP modifier state (SPE_DESIGN.md §3 Class 2: DppCrossLane) ──
  //
  // DPP is a src0-pathway cross-lane shuffle modifier. The original
  // `inst.getOpcode()` retains the `_dpp` suffix and its MCInstrDesc
  // advertises the DPP bit via `TSFlags & SIInstrFlags::DPP`; the
  // opcode_map canonicalises the SemOp down to the base op, so handlers
  // dispatched by SemOp do not see the DPP variant directly.
  //
  // `hasDpp` is set in decode.cpp exactly when `tsFlags & SIInstrFlags::DPP`
  // is true; the modifier-operand fields below are populated by looking
  // up their named-operand indices in the original (non-canonicalised)
  // MCInstrDesc.
  //
  // Handler contract: `OpResolver::src(0)` / `srcF(0)` / `src64(0)`
  // transparently wrap their result through `emitUpdateDpp` when
  // `hasDpp` is true, using these fields as the intrinsic's
  // `dpp_ctrl` / `row_mask` / `bank_mask` / `bound_ctrl` immediates.
  // Handlers therefore need no per-op DPP awareness.
  //
  // `fi` (fetch-invalid) is an encoding-level flag on DPP8 (not DPP16);
  // `llvm.amdgcn.update.dpp` exposes only the DPP16 operand set, so we
  // do not surface `fi` here. A future DPP8 lift would extend this
  // block.
  bool hasDpp = false;
  uint16_t dppCtrl = 0;
  uint8_t dppRowMask = 0xF;
  uint8_t dppBankMask = 0xF;
  bool dppBoundCtrl = false;

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
