////////////////////////////////////////////////////////////////////////////////
//
// Wave Width Translation: wave32 (RDNA/GFX12) -> wave64 (CDNA/GFX9)
//
// Handles exec register width differences when retargeting code from
// wave32 ISAs (e.g., gfx1250) to wave64 ISAs (e.g., gfx942/gfx950).
//
// Key transformations:
// - saveexec_b32 -> manual exec_lo save + ALU + exec_hi = 0
// - v_cmpx -> v_cmp + manual exec AND + VCC save/restore + exec_hi clear
// - VCC width: clear vcc_hi after v_cmp on wave64
// - Exec_hi = 0 after any operation that writes exec_lo
//
////////////////////////////////////////////////////////////////////////////////

#ifndef ROCR_HOTSWAP_WAVE_WIDTH_HPP
#define ROCR_HOTSWAP_WAVE_WIDTH_HPP

#include <mlir/IR/BuiltinOps.h>

namespace hotswap {

/// Widen exec mask operations from wave32 to wave64.
/// Must be called AFTER cross-target mnemonic mapping and BEFORE SSA
/// construction, since it inserts new physical register references.
mlir::LogicalResult widenExecMask(mlir::ModuleOp module);

} // namespace hotswap

#endif // ROCR_HOTSWAP_WAVE_WIDTH_HPP
