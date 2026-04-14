////////////////////////////////////////////////////////////////////////////////
//
// Cross-Target Instruction Mapping Pass
//
// Transforms waveasm ops from one AMDGPU ISA to another. Handles:
// - Mnemonic renames (e.g., global_load_b32 -> global_load_dword)
// - Wait counter translation (GFX12 split counters -> GFX9 unified s_waitcnt)
// - Memory instruction format changes
//
////////////////////////////////////////////////////////////////////////////////

#ifndef ROCR_HOTSWAP_CROSS_TARGET_HPP
#define ROCR_HOTSWAP_CROSS_TARGET_HPP

#include <mlir/IR/BuiltinOps.h>
#include <string>

namespace hotswap {

/// Retarget all waveasm.program ops in the module from sourceISA to targetISA.
mlir::LogicalResult retargetModule(mlir::ModuleOp module,
                                   llvm::StringRef sourceISA,
                                   llvm::StringRef targetISA);

/// Get the target mnemonic for a source mnemonic when translating
/// from sourceISA to targetISA. Returns empty string if no mapping exists.
std::string mapMnemonic(llvm::StringRef mnemonic, llvm::StringRef sourceISA,
                        llvm::StringRef targetISA);

} // namespace hotswap

#endif // ROCR_HOTSWAP_CROSS_TARGET_HPP
