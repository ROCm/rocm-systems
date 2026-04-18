#ifndef HOTSWAP_TRANSPILER_SEM_OP_ATTRS_HPP
#define HOTSWAP_TRANSPILER_SEM_OP_ATTRS_HPP

#include "semop.hpp"

#include "llvm/ADT/ArrayRef.h"

namespace llvm {
class MCInstrInfo;
} // namespace llvm

namespace transpiler {

class OpcodeMap;

// Per-SemOp metadata that the raiser cross-references at its pre-
// translation gates. Grows incrementally as new per-writer predicates
// are needed (today only `routesExecThroughStoreExec`; SPE_DESIGN §4's
// wave-size-obliviousness predicate will attach here too).
//
// Default-constructed attrs mean "no declared guarantees" — the gate
// aborts for any EXEC-writer encountered that does not have its
// `routesExecThroughStoreExec` bit set, so the default is fail-closed.
struct SemOpAttrs {
  // The SemOp's handler routes any write to ParsedReg::EXEC through
  // `regs.storeExec` — either directly or via `writeReg{32,64,ExecWidth}`
  // which dispatch EXEC→storeExec. Required for any SemOp whose
  // MCInstrDesc declares EXEC as a def (the startup invariant in
  // `verifyExecAttrCoverage`); also required to pass the per-kernel
  // Phase 1.5 SPE gate when an explicit `exec_lo`/`exec_hi` operand
  // makes the instruction a runtime EXEC-writer.
  bool routesExecThroughStoreExec = false;
};

// Pair of (SemOp, attrs) registered by a handler file. The
// per-handler-file registrations below aggregate into the central
// lookup table on first use.
struct SemOpAttrSpec {
  SemOp op;
  SemOpAttrs attrs;
};

// Per-handler-TU registrations. Each handler that claims SemOps with
// non-default attrs exposes a getter returning its static array of
// specs. Adding a new EXEC-writing SemOp therefore requires declaring
// the attrs in the handler file that implements it — the attribute
// registration moves with the handler.
//
// `theTable()` in `sem_op_attrs.cpp` calls every getter declared here
// at first-use to materialise the combined table; adding a new getter
// is a two-line change (function definition + one line in
// `sem_op_attrs.cpp` aggregating it).
llvm::ArrayRef<SemOpAttrSpec> getHandlerSOP1Attrs();
llvm::ArrayRef<SemOpAttrSpec> getHandlerSOP2Attrs();
llvm::ArrayRef<SemOpAttrSpec> getHandlerVALU_VcmpAttrs();

// O(1) lookup keyed on SemOp. Returns a default-constructed SemOpAttrs
// for SemOps that have no declared attrs (fail-closed).
const SemOpAttrs &getSemOpAttrs(SemOp op);

// Startup invariant: for every MC opcode whose MCInstrDesc declares
// EXEC (or EXEC_LO/EXEC_HI) as an implicit def, the SemOp it maps to
// in `opcMap` must have `routesExecThroughStoreExec = true`. Catches
// the case where a new EXEC-writing opcode lands in LLVM and gets
// mapped to a SemOp whose handler we have not yet audited for SPE.
// Explicit-operand EXEC writes (e.g. `s_mov_b32 exec_lo, s2`) remain
// the Phase 1.5 per-kernel gate's responsibility since they depend on
// runtime operand values.
//
// Called once at raiser init from the same slot that runs
// `verifyMFMACoverage`. Fails loudly via `report_fatal_error`.
void verifyExecAttrCoverage(const llvm::MCInstrInfo &MCII,
                             const OpcodeMap &opcMap);

} // namespace transpiler

#endif
