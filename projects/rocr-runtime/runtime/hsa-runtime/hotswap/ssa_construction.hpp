////////////////////////////////////////////////////////////////////////////////
//
// SSA Construction Pass: converts physical register references in the
// lifted waveasm IR into proper SSA def-use chains with virtual registers.
//
// Before: each instruction independently materializes its operands via
//         precolored.{vreg,sreg,areg} ops with no data flow connections.
// After:  instructions consume results of earlier instructions, forming
//         proper SSA def-use chains. Virtual register types replace physical.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef ROCR_HOTSWAP_SSA_CONSTRUCTION_HPP
#define ROCR_HOTSWAP_SSA_CONSTRUCTION_HPP

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace hotswap {

/// Run SSA construction on all waveasm.program ops in the module.
/// Replaces precolored register references with def-use chains.
mlir::LogicalResult constructSSA(mlir::ModuleOp module);

} // namespace hotswap

#endif // ROCR_HOTSWAP_SSA_CONSTRUCTION_HPP
