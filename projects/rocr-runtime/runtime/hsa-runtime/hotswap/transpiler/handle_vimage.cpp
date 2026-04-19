// VIMAGE TENSOR handler — gfx1250-only tensor-descriptor memory ops.
//
// Covers the four `VIMAGE_TENSOR_Pseudo` instructions defined in
// `MIMGInstructions.td:2049-2113`:
//
//   * `tensor_load_to_lds_d2`    / `tensor_load_to_lds_d4`
//   * `tensor_store_from_lds_d2` / `tensor_store_from_lds_d4`
//
// Both `_d2` (up-to-2D) and `_d4` (up-to-4D) variants share a SemOp
// (`TENSOR_LOAD_TO_LDS` / `TENSOR_STORE_FROM_LDS`); see the
// docstrings in `semop.hpp` and the canonicalization entries in
// `opcode_map.cpp`.
//
// === Cross-target contract ===
//
// On gfx942 (and every pre-gfx1250 target) there is no equivalent
// hardware unit:
//   * The TENSORcnt register and the `TENSOR_CNT` TSFlags bit live
//     under `let SubtargetPredicate = isGFX125xOnly` in TableGen,
//     and there is no MFMA/WMMA-style decomposition that emulates a
//     full Tensor Descriptor walk plus LDS gather/scatter on earlier
//     ISAs (the descriptor itself encodes per-dim strides, padding,
//     and addressing modes that no gfx9xx instruction can reproduce
//     atomically).
//   * The matching LLVM intrinsics
//     (`int_amdgcn_tensor_load_to_lds` /
//     `int_amdgcn_tensor_store_from_lds`,
//     IntrinsicsAMDGPU.td:4213) are themselves gated on
//     `SubtargetPredicate = isGFX125xOnly`, so a cross-target lift
//     to the intrinsic would also fail at codegen.
//
// The user-rules forbid silent fallbacks. We therefore refuse loudly
// via `RaiseFailure::unsupportedShape` with a precise diagnostic
// that names the offending mnemonic, the architectural mismatch,
// and the intrinsic that would be the same-target lift. The
// `formatName(VIMAGE)` bucket lets `kerneldex` / `corpus_test`
// summarise these without parsing the diagnostic text.
//
// === Same-target contract (gfx1250 -> gfx1250) ===
//
// When the compilation target is itself gfx1250 (TENSORcnt unit
// available), the principled lift is a direct call to the matching
// intrinsic with the four D# operands marshalled from the SReg_128 /
// SReg_256 source operands. That path is documented but not yet
// implemented because no kernel in the current corpus exercises it
// (every captured `tensor_load_to_lds` kernel is lifted with
// `--target-isa=gfx942`); see the FUTURE block below for the
// operand-marshalling sketch. When the same-target path lands, this
// handler will branch on `ctx.targetIsa.hasTensorOps`.

#include "handlers.hpp"

#include "decoded_inst.hpp"
#include "isa_profile.hpp"
#include "raise_context.hpp"
#include "raise_failure.hpp"
#include "semop.hpp"

#include "llvm/Support/raw_ostream.h"

namespace transpiler {

HandlerResult handleVIMAGE(RaiseContext &ctx, const DecodedInst &di,
                           OpResolver &op) {
  (void)op; // VIMAGE TENSOR currently has no source-register lift path.
  HandlerResult hr;
  SemOp sop = di.semOp;

  // Only the two TENSOR SemOps reach this handler; anything else in
  // the VIMAGE format family (e.g. future image_load/image_sample
  // ops if we ever lift them) would fall through with `handled =
  // false` so the main raiser loop reports the canonical
  // `UnsupportedOpcode [VIMAGE]` diagnostic. Intentionally do NOT
  // synthesise a generic refusal here — that would mask new VIMAGE
  // members the kerneldex sweep surfaces in the future.
  if (sop != SemOp::TENSOR_LOAD_TO_LDS &&
      sop != SemOp::TENSOR_STORE_FROM_LDS) {
    return hr;
  }

  // FUTURE — same-target gfx1250 -> gfx1250 intrinsic lift. Sketch:
  //
  //   if (ctx.targetIsa.hasTensorOps) {
  //     // Operand layout (MIMGInstructions.td:2073):
  //     //   _d2: vaddr0:SReg_128, vaddr1:SReg_256, r128:imm,  cpol:imm
  //     //   _d4: vaddr0:SReg_128, vaddr1:SReg_256, vaddr2:SReg_128,
  //     //        vaddr3:SReg_128, r128:imm, cpol:imm
  //     // Marshall each SReg_128 into <4 x i32> via four
  //     // `regs.loadSGPR32` calls, each SReg_256 into <8 x i32> via
  //     // eight; pass <8 x i32> zeroinitializer for the reserved
  //     // group-4 operand the MC encoding hardcodes to NULL
  //     // (MIMGInstructions.td:2103). For `_d2` pass a <4 x i32>
  //     // zero for groups 2/3. Then call:
  //     //   sop == TENSOR_LOAD_TO_LDS    ? Intrinsic::amdgcn_tensor_load_to_lds
  //     //                                : Intrinsic::amdgcn_tensor_store_from_lds
  //     // The intrinsic returns void so there is no SSA chain.
  //     hr.handled = true;
  //     return hr;
  //   }
  //
  // Implemented when the first same-target kernel surfaces in the
  // corpus; until then the cross-target refusal below is the only
  // exercised path.

  // Cross-target (gfx1250 -> gfx942 and earlier) loud refusal.
  llvm::errs()
      << "transpiler: VIMAGE: " << di.mnemonic
      << " has no equivalent on the compilation target "
      << "(gfx1250 TENSORcnt unit; LLVM intrinsic "
      << (sop == SemOp::TENSOR_LOAD_TO_LDS
              ? "amdgcn.tensor.load.to.lds"
              : "amdgcn.tensor.store.from.lds")
      << " is gated isGFX125xOnly). Refusing to emit a fallback "
         "lowering — Tensor Descriptor walks cannot be reconstructed "
         "from gfx9xx primitives without violating the descriptor's "
         "per-dim addressing semantics.\n";

  hr.failure = RaiseFailure::unsupportedShape(
      di, "VIMAGE",
      "gfx1250-only TENSOR cnt op; no equivalent on "
      "non-gfx1250 compilation target");
  return hr;
}

} // namespace transpiler
