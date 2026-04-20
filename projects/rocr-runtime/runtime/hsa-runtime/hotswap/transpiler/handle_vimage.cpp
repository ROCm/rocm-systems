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
// `opcode_map.cpp`. The form is recovered from `op.nSrcs()`: the
// pseudo's InOperandList has `vaddr0, vaddr1, r128, cpol` (4) for
// `_d2` and `vaddr0, vaddr1, vaddr2, vaddr3, r128, cpol` (6) for
// `_d4`; both `_d{2,4}_gfx1250` Reals inherit the same operand list
// (`MIMGInstructions.td:2087`), so this count is stable across the
// MC -> pseudo collapse `OpcodeMap::canonicalize` performs.
//
// === Same-target contract (gfx1250 -> gfx1250) ===
//
// When the compilation target is itself gfx1250 (TENSORcnt unit
// available, `ISAProfile::hasTensorOps`), the principled lift is a
// direct call to the matching LLVM intrinsic
// (`llvm.amdgcn.tensor.{load.to.lds,store.from.lds}`,
// IntrinsicsAMDGPU.td:4213). The intrinsic's signature mirrors the
// hardware operand bank exactly:
//
//   void int_amdgcn_tensor_load_to_lds(<4 x i32> grp0,
//                                       <8 x i32> grp1,
//                                       <4 x i32> grp2,
//                                       <4 x i32> grp3,
//                                       <8 x i32> grp4_reserved,
//                                       i32 cachepolicy)
//
// We marshal each SReg_128/SReg_256 source into the matching
// `<N x i32>` by reading consecutive dwords via `regs.loadSGPR32`
// and packing them into a vector with `CreateInsertElement`. Group 4
// is hardcoded to <8 x i32> zeroinitializer per the IntrinsicsAMDGPU
// docstring ("reserved for future targets, use zeroinitializer for
// now"); for the `_d2` form the unused groups 2 and 3 are also
// passed as <4 x i32> zeroinitializer (the MC `_d2` encoding pins
// `vaddr2`/`vaddr3` to the NULL SGPR sentinel via
// `MIMGInstructions.td:2099-2100`, which the disassembler does not
// surface as operands). The `r128` immediate is consumed by the
// hardware encoding and is not part of the intrinsic's argument
// vector — see IntrinsicsAMDGPU.td:4197-4211.
//
// === Cross-target contract (gfx1250 -> gfx942 and earlier) ===
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
//   * The matching LLVM intrinsics are themselves gated on
//     `SubtargetPredicate = isGFX125xOnly`, so a cross-target
//     intrinsic emit would also fail at codegen on a non-gfx1250
//     backend.
//
// The user-rules forbid silent fallbacks. We therefore refuse loudly
// via `RaiseFailure::unsupportedShape` with a precise diagnostic
// that names the offending mnemonic, the architectural mismatch,
// and the intrinsic that would be the same-target lift. The
// `formatName(VIMAGE)` bucket lets `kerneldex` / `corpus_test`
// summarise these without parsing the diagnostic text.

#include "handlers.hpp"

#include "decoded_inst.hpp"
#include "isa_profile.hpp"
#include "parsed_reg.hpp"
#include "raise_context.hpp"
#include "raise_failure.hpp"
#include "reg_file.hpp"
#include "semop.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace transpiler {

namespace {

// Marshal `n` consecutive SGPR dwords starting at `base.baseIdx` into
// an `<n x i32>` vector. `base.kind` MUST be `ParsedReg::SGPR` —
// callers that accept the SReg_128_XNULL `null` sentinel must
// short-circuit to a zero vector before reaching this helper.
// `n` is hardcoded by the caller from the operand class (4 for
// SReg_128 / D# group 0,2,3; 8 for SReg_256 / D# group 1) per the
// pseudo's InOperandList in MIMGInstructions.td:2073.
Value *marshalSgprGroup(RaiseContext &ctx, ParsedReg base, unsigned n,
                        const Twine &name) {
  auto *vecTy = FixedVectorType::get(ctx.i32Ty, n);
  Value *vec = PoisonValue::get(vecTy);
  for (unsigned i = 0; i < n; ++i) {
    Value *dword = ctx.regs.loadSGPR32(ctx.B, base.baseIdx + static_cast<int>(i));
    vec = ctx.B.CreateInsertElement(vec, dword, i, name);
  }
  return vec;
}

// Build a `<n x i32> zeroinitializer` for groups the form does not
// supply (group 4 is always reserved; groups 2 and 3 are unused
// in the `_d2` form per `MIMGInstructions.td:2099-2100`).
Value *zeroVec(RaiseContext &ctx, unsigned n) {
  auto *vecTy = FixedVectorType::get(ctx.i32Ty, n);
  return ConstantAggregateZero::get(vecTy);
}

// Read an immediate operand and zero-extend it to i32. The intrinsic
// signature carries `cachepolicy` as i32 with `ImmArg<ArgIndex<5>>`
// so the value MUST be a constant — `op.srcImm` returns the decoded
// integer directly, sidestepping any operand-read divergence path.
Value *cpolImm(RaiseContext &ctx, OpResolver &op, unsigned cpolIdx) {
  return ConstantInt::get(ctx.i32Ty, static_cast<uint32_t>(op.srcImm(cpolIdx)));
}

} // namespace

HandlerResult handleVIMAGE(RaiseContext &ctx, const DecodedInst &di,
                           OpResolver &op) {
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

  // Same-target gfx1250 -> gfx1250 intrinsic lift.
  if (ctx.targetIsa.hasTensorOps) {
    // Recover the operand-shape variant. The pseudo's InOperandList
    // (MIMGInstructions.td:2073) has 4 sources for `_d2`
    // (vaddr0, vaddr1, r128, cpol) and 6 for `_d4` (vaddr0..vaddr3,
    // r128, cpol); both `_d{2,4}_gfx1250` Reals inherit the same
    // operand list at MIMGInstructions.td:2087. Anything else means
    // a future encoding variant landed in LLVM that we have not
    // audited — refuse loudly so the drift surfaces immediately.
    const unsigned nsrcs = op.nSrcs();
    if (nsrcs != 4 && nsrcs != 6) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VIMAGE",
          Twine("unexpected source operand count ") + Twine(nsrcs) +
              " for tensor op (expected 4 for _d2 or 6 for _d4)");
      return hr;
    }
    const bool isD2 = (nsrcs == 4);

    // Vaddr operands are required to be real SGPR ranges. The
    // SReg_128_XNULL/SReg_256_XNULL operand classes nominally
    // permit the NULL sentinel, but a NULL D# pointer would mean
    // the kernel has no Tensor Descriptor to walk — there is no
    // sensible lowering that preserves observed behaviour. We
    // intentionally do NOT cross-check `pr.width` against the
    // operand-class width: `computeRegWidth32` (raise_context.cpp:66)
    // walks the disassembler-supplied sub-reg chain, and AMDGPU's
    // SReg_*_XNULL tuple classes sometimes report a width smaller
    // than the operand's nominal dword count when the high lanes
    // alias an aggregate sub-reg index. The hardware encoding still
    // reads `n` consecutive SGPRs starting at `baseIdx` regardless
    // of how the tuple chain is named, so reading via baseIdx is
    // the source of truth — it matches the decode of the encoded
    // 8-bit SGPR pointer field exactly.
    auto requireSgpr = [&](ParsedReg pr, const char *role) -> bool {
      if (pr.kind != ParsedReg::SGPR) {
        hr.failure = RaiseFailure::unsupportedShape(
            di, "VIMAGE",
            Twine("tensor ") + role + " must be a contiguous SGPR range "
                  "(got non-SGPR operand kind)");
        return false;
      }
      return true;
    };

    ParsedReg vaddr0 = op.srcReg(0);
    ParsedReg vaddr1 = op.srcReg(1);
    if (!requireSgpr(vaddr0, "vaddr0/D# group 0") ||
        !requireSgpr(vaddr1, "vaddr1/D# group 1"))
      return hr;

    Value *grp0 = marshalSgprGroup(ctx, vaddr0, 4, "td_grp0");
    Value *grp1 = marshalSgprGroup(ctx, vaddr1, 8, "td_grp1");
    Value *grp2;
    Value *grp3;
    Value *cpol;
    if (isD2) {
      grp2 = zeroVec(ctx, 4);
      grp3 = zeroVec(ctx, 4);
      cpol = cpolImm(ctx, op, 3);
    } else {
      ParsedReg vaddr2 = op.srcReg(2);
      ParsedReg vaddr3 = op.srcReg(3);
      if (!requireSgpr(vaddr2, "vaddr2/D# group 2") ||
          !requireSgpr(vaddr3, "vaddr3/D# group 3"))
        return hr;
      grp2 = marshalSgprGroup(ctx, vaddr2, 4, "td_grp2");
      grp3 = marshalSgprGroup(ctx, vaddr3, 4, "td_grp3");
      cpol = cpolImm(ctx, op, 5);
    }
    Value *grp4 = zeroVec(ctx, 8); // reserved for future targets

    Intrinsic::ID iid = (sop == SemOp::TENSOR_LOAD_TO_LDS)
                            ? Intrinsic::amdgcn_tensor_load_to_lds
                            : Intrinsic::amdgcn_tensor_store_from_lds;
    Function *fn = Intrinsic::getOrInsertDeclaration(&ctx.M, iid);
    ctx.B.CreateCall(fn, {grp0, grp1, grp2, grp3, grp4, cpol});

    hr.handled = true;
    return hr;
  }

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
