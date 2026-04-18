#include "sem_op_attrs.hpp"

#include "opcode_map.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::EXEC, EXEC_LO, EXEC_HI
#include "Utils/AMDGPUBaseInfo.h"            // AMDGPU::mc2PseudoReg

#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <iterator>

using namespace llvm;

namespace transpiler {

namespace {

// Materialise the attribute table once at first-use. Aggregates every
// per-handler registration declared in `sem_op_attrs.hpp`. The
// `Meyers singleton`-style static local sidesteps any cross-TU static
// initialisation ordering fuss — the table is built lazily, not at
// dynamic-init time.
class AttrTable {
public:
  AttrTable() {
    auto ingest = [&](ArrayRef<SemOpAttrSpec> specs) {
      for (const SemOpAttrSpec &s : specs) {
        auto idx = static_cast<std::size_t>(s.op);
        assert(idx < std::size(entries_) &&
               "SemOp out of range; bump entries_ to SemOp_COUNT");
        entries_[idx] = s.attrs;
      }
    };
    ingest(getHandlerSOP1Attrs());
    ingest(getHandlerSOP2Attrs());
    ingest(getHandlerVALU_VcmpAttrs());
  }

  const SemOpAttrs &operator[](SemOp op) const {
    auto idx = static_cast<std::size_t>(op);
    assert(idx < std::size(entries_) && "SemOp out of range");
    return entries_[idx];
  }

private:
  SemOpAttrs entries_[static_cast<std::size_t>(SemOp::SemOp_COUNT)] = {};
};

const AttrTable &theTable() {
  static const AttrTable t;
  return t;
}

static bool descImplicitlyDefinesEXEC(const MCInstrDesc &desc) {
  for (MCPhysReg r : desc.implicit_defs()) {
    MCRegister reg = AMDGPU::mc2PseudoReg(r);
    if (reg == AMDGPU::EXEC || reg == AMDGPU::EXEC_LO ||
        reg == AMDGPU::EXEC_HI)
      return true;
  }
  return false;
}

} // namespace

const SemOpAttrs &getSemOpAttrs(SemOp op) { return theTable()[op]; }

void verifyExecAttrCoverage(const MCInstrInfo &MCII, const OpcodeMap &opcMap) {
  for (unsigned mc = 0, end = MCII.getNumOpcodes(); mc < end; ++mc) {
    const MCInstrDesc &desc = MCII.get(mc);
    if (!descImplicitlyDefinesEXEC(desc))
      continue;
    SemOp sop = opcMap.lookup(mc);
    if (sop == SemOp::Unknown)
      continue; // covered by the generic unsupported-opcode path
    if (getSemOpAttrs(sop).routesExecThroughStoreExec)
      continue;
    report_fatal_error(Twine("transpiler: MC opcode ") + MCII.getName(mc) +
                       " (#" + Twine(mc) +
                       ") declares EXEC as an implicit def but its SemOp "
                       "(" + semOpName(sop) +
                       ") is not marked routesExecThroughStoreExec. Audit "
                       "the handler's EXEC write path against SPE before "
                       "declaring the SemOp in that handler's "
                       "get*Attrs() registration.");
  }
}

} // namespace transpiler
