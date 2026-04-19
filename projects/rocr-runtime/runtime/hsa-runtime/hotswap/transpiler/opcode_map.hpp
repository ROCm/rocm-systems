#ifndef HOTSWAP_TRANSPILER_OPCODE_MAP_HPP
#define HOTSWAP_TRANSPILER_OPCODE_MAP_HPP

#include "semop.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/MC/MCInstrInfo.h"

#include <cstdint>

namespace transpiler {

// Side-table metadata for vector compare instructions (V_CMP_* / V_CMPX_*).
//
// The raiser collapses every V_CMP_* and V_CMPX_* MC opcode onto two SemOps
// (`V_CMP`, `V_CMPX`) to avoid enumerating ~100 near-identical cases in
// handlers. The actual predicate / element type / width carried in the
// pseudo name (`v_cmp_EQ_U32_e64` → EQ, unsigned 32-bit) is lifted here at
// map-build time and looked up by MC opcode at dispatch time.
struct VCmpMeta {
  // LLVM predicate to feed to `IRBuilder::CreateICmp` / `CreateFCmp`.
  // Ignored when `isClass == true` (no predicate; the comparison is the
  // floating-point classification mask in src1).
  llvm::CmpInst::Predicate pred;
  // Element width in bits: 16, 32, or 64.
  uint8_t bits;
  // False → integer compare (ICmp; `pred` is `ICMP_*`).
  // True  → float compare   (FCmp; `pred` is `FCMP_*`).
  bool isFloat;
  // True for the V_CMP_CLASS_F* / V_CMPX_CLASS_F* family. These are
  // NOT predicate compares — src0 is a float operand and src1 is an
  // i32 mask of FP classes; the result lane bit is set iff src0
  // matches any class enabled in the mask. Lifts to
  // `llvm.amdgcn.class.f<bits>(src0, src1)` rather than CreateFCmp;
  // `pred` is unused on class entries. See V_CMP_CLASS in the
  // gfx9+ AMDGPU ISA manual and the dispatch in handle_valu_vcmp.cpp.
  bool isClass = false;
};

// Maps raw LLVM MC opcodes emitted by the AMDGPU disassembler to
// architecture-neutral SemOp tags that the raiser dispatches on.
//
// The map is built once at raiser-initialization time from MCInstrInfo and
// TableGen-generated AMDGPU instruction tables.  Almost no string parsing is
// involved: encoding variants (e32/e64/DPP/SDWA/subtarget-specific real
// encodings) are folded onto their canonical pseudo via the
// AMDGPU::InstrMapping helpers (getMCOpcode, getVOPe64, getDPPOp32,
// getDPPOp64, getSDWAOp, getGlobalVaddrOp), and the resulting canonical
// pseudo is matched against a small compile-time table of AMDGPU::<opcode>
// enum constants.
//
// The one narrow exception is the V_CMP / V_CMPX family, where the pseudo
// name is parsed at build time to extract (predicate, element width, int/
// float) into a side table -- see `VCmpMeta` above. That keeps the SemOp
// enum small (two entries instead of ~100) without pushing string handling
// into the hot dispatch path.
class OpcodeMap {
public:
  // Lookup is hot-path: called once per decoded instruction.
  SemOp lookup(unsigned opcode) const;

  // Vector-compare metadata lookup. Returns nullptr if `opcode` is not a
  // V_CMP_* or V_CMPX_* instruction (or isn't known to this map at all).
  // Callers should only query this when `lookup(opcode)` returned
  // `SemOp::V_CMP` or `SemOp::V_CMPX`.
  const VCmpMeta *lookupVCmp(unsigned opcode) const;

  // Build is one-shot: called during raiser initialization.
  void build(const llvm::MCInstrInfo &MCII);

private:
  llvm::DenseMap<unsigned, SemOp> map_;
  llvm::DenseMap<unsigned, VCmpMeta> vcmp_;
};

} // namespace transpiler

#endif
