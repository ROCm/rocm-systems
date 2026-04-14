////////////////////////////////////////////////////////////////////////////////
//
// Assembly Text Emission from waveasm MLIR IR
//
// Converts the lifted/transformed waveasm IR back to AMDGPU assembly text
// that can be assembled by LLVM MC. Operates on the physical-register IR
// (before SSA construction).
//
////////////////////////////////////////////////////////////////////////////////

#ifndef ROCR_HOTSWAP_EMIT_ASSEMBLY_HPP
#define ROCR_HOTSWAP_EMIT_ASSEMBLY_HPP

#include <mlir/IR/BuiltinOps.h>
#include <string>

namespace hotswap {

/// Emit the waveasm IR in the module as AMDGPU assembly text.
/// Returns the assembly string suitable for LLVM MC's AsmParser.
/// The IR must use physical register types (pre-SSA).
std::string emitAssembly(mlir::ModuleOp module);

} // namespace hotswap

#endif // ROCR_HOTSWAP_EMIT_ASSEMBLY_HPP
