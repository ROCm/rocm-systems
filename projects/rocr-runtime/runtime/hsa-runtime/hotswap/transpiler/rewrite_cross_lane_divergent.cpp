#include "rewrite_cross_lane_divergent.hpp"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace transpiler {

namespace {

// ============================================================================
// Use-chain classifier
// ============================================================================
//
// Forward-walk the transitive uses of a cross-lane primitive's result.
// If any use reaches an SGPR-constrained operand, the backend will
// re-introduce `v_readfirstlane_b32` on the rewrite's `ds_bpermute`
// output and collapse the per-source-wave distinction the rewrite
// exists to preserve. Such kernels must be refused at raise time
// rather than emitted with a silently-broken rewrite.
//
// Over-approximation policy: unknown users (unrecognised intrinsics,
// inline asm with unrecognised constraints, non-instruction users
// like constant expressions) are classified as SGPR-forced. The
// project's no-silent-miscompile rule trades false positives for
// soundness. Each time the classifier refuses a kernel that would
// actually be VGPR-safe, the right fix is to extend one of the
// `is*Intrinsic*` tables below, not to relax the default.

enum class UseChainVerdict {
  VGPRSafe,
  SGPRForced,
};

// Per-instruction result: an intrinsic call either
//   (a) forces SGPR on the input at this operand index (refuse),
//   (b) is a VGPR-safe sink — the call consumes our value as a VGPR
//       input and its result is a fresh value we do not need to
//       track further (e.g. a memory-store intrinsic),
//   (c) is a VGPR-safe propagator — the call consumes our value as
//       a VGPR input and its result may still carry the per-source-
//       wave state into downstream uses (e.g. a VALU arithmetic
//       intrinsic), so we continue the walk through the result's
//       users.
enum class IntrinsicRole {
  SGPRForced,
  VGPRSafeSink,
  VGPRSafePropagator,
  Unknown,
};

// amdgcn.* intrinsics whose corresponding operand position *must*
// receive an SGPR. When our tracked value reaches any of these at the
// flagged operand index, the backend inserts `v_readfirstlane` on the
// input and the rewrite loses its per-source-wave state. Entries are
// keyed on the intrinsic ID; the operand-index predicate is checked
// per-call-site in `classifyIntrinsicUse`.
//
// This list is deliberately conservative — it only covers intrinsics
// the raiser is *known* to emit today (see the `git grep
// Intrinsic::amdgcn_` audit in hotswap/docs/learnings.md). An
// unrecognised amdgcn intrinsic falls through the switch below and is
// treated as Unknown (→ SGPRForced).
bool operandForcesSGPR(Intrinsic::ID id, unsigned operandIdx) {
  switch (id) {
  // `amdgcn.readfirstlane(src)` — the entire point of the intrinsic
  // is scalarising a VGPR into an SGPR. If our value reaches this,
  // it is the explicit scalarisation marker the rewrite was designed
  // to avoid; refuse unconditionally (operandIdx is ignored — every
  // operand position is an SGPR boundary in practice for this
  // intrinsic's intent).
  case Intrinsic::amdgcn_readfirstlane:
    return true;
  // `s_sendmsg(msg, val)` — the message index must be a compile-time
  // immediate, but the payload `val` is required to be in an SGPR.
  case Intrinsic::amdgcn_s_sendmsg:
    return operandIdx <= 1;
  // `s_barrier()` has no operands; including here as a scalar-context
  // sink would be noise. Omitted intentionally.
  default:
    return false;
  }
}

// amdgcn.* intrinsics whose *every* operand accepts a VGPR and whose
// result we need to continue tracking because it may still carry the
// per-source-wave state downstream (arithmetic, rotation, per-lane
// data motion). Must not overlap `operandForcesSGPR` for the same
// (intrinsic, operand) pair.
bool isIntrinsicVGPRSafePropagator(Intrinsic::ID id) {
  switch (id) {
  case Intrinsic::amdgcn_ds_bpermute:
  case Intrinsic::amdgcn_ds_swizzle:
  case Intrinsic::amdgcn_update_dpp:
  case Intrinsic::amdgcn_mbcnt_lo:
  case Intrinsic::amdgcn_mbcnt_hi:
  case Intrinsic::amdgcn_perm:
  case Intrinsic::amdgcn_cvt_f32_bf8:
  case Intrinsic::amdgcn_cvt_f32_fp8:
  case Intrinsic::amdgcn_cvt_pk_bf8_f32:
  case Intrinsic::amdgcn_cvt_pk_f32_bf8:
  case Intrinsic::amdgcn_cvt_pk_f32_fp8:
  case Intrinsic::amdgcn_cvt_pk_fp8_f32:
  case Intrinsic::amdgcn_cvt_pkrtz:
  case Intrinsic::amdgcn_cvt_scalef32_pk_fp4_f32:
  case Intrinsic::amdgcn_cvt_scale_pk8_bf16_fp4:
  case Intrinsic::amdgcn_class:
  case Intrinsic::amdgcn_rcp:
  case Intrinsic::amdgcn_sffbh:
  case Intrinsic::amdgcn_div_fixup:
  case Intrinsic::amdgcn_div_fmas:
  case Intrinsic::amdgcn_div_scale:
  // Cross-lane primitives the rewrite also rewrites in this same
  // pass: their post-rewrite shape is `select` / `ds_bpermute`, both
  // VGPR-safe. We can treat them as VGPR-safe propagators pre-
  // rewrite because the symmetry rule guarantees they will be
  // rewritten together.
  case Intrinsic::amdgcn_writelane:
  case Intrinsic::amdgcn_readlane:
    return true;
  // Generic LLVM intrinsics that the AMDGPU backend lowers to per-lane
  // VALU opcodes with no SGPR-forced operand position in any codegen
  // path.  The audit here is tight: only add intrinsics whose AMDGPU
  // lowering is a single VALU instruction (or a VALU-only expansion)
  // accepting all-VGPR operands — same bar as the `amdgcn_*` cases
  // above.  Unknown generic intrinsics stay SGPR-forced via the default
  // arm below, consistent with the "refuse when uncertain" rule in
  // hotswap/docs/wave-size-translation.md §5.6.3.
  //
  // Why these specifically, and why now: Triton's AMD backend emits
  // these in the fast-reciprocal / rsqrt Newton-iteration expansion
  // surrounding `@llvm.amdgcn.div_fixup` / `div_fmas` / `div_scale`
  // (already whitelisted above) and in the reduction post-processing
  // of layer-norm / softmax (rstd = `1 / sqrt(var + eps)`, softmax
  // normaliser `x / sum`).  All appear on every reduction-bearing
  // Triton kernel's readlane-result use chain.  Pre-audit, the
  // classifier over-approximated them as SGPR-forced, which disabled
  // the rewrite pass on the entire function (all-or-nothing per
  // §5.6.3's "mix of rewritten and preserved sites recreates the
  // Matmul128x128 asymmetric-rewrite fault" rule).  The AMDGPU
  // lowerings are:
  //
  //   * `@llvm.fma.f32`     → `v_fma_f32` (VOP3, three VGPR sources,
  //     VGPR destination; no SGPR-forced operand).  VALU, per-lane.
  //   * `@llvm.fmuladd.f32` → `v_fma_f32` / `v_mac_f32` (relaxed-
  //     precision fused-or-split multiply-add; backend chooses per
  //     target and `contract` metadata).  Operand shapes identical to
  //     `fma` — all VGPR, per-lane.
  //   * `@llvm.sqrt.f32`    → `v_sqrt_f32` (VOP1, one VGPR source,
  //     VGPR destination).  VALU, per-lane.
  //   * `@llvm.maxnum.f32`  → `v_max_f32` (VOP2, two VGPR sources,
  //     VGPR destination; IEEE max-num with NaN-propagation rules
  //     handled in the VALU expansion).  VALU, per-lane.  Appears on
  //     every softmax reduction (`m_i = max(m_i-1, x)`) — the same
  //     position `fma` occupies in layer-norm.
  //   * `@llvm.minnum.f32`  → `v_min_f32`.  Symmetric with maxnum;
  //     audited for parity so any future min-reducing kernel isn't
  //     blocked on a one-intrinsic gap.
  //   * `@llvm.fabs.f32`    → `v_and_b32` with a `0x7fffffff` mask
  //     (the backend's preferred `fabs` lowering on modern AMDGPU;
  //     see AMDGPUCombinerHelper.cpp).  Per-lane, all-VGPR.  Shows
  //     up in Triton's reduction prologues when the source is the
  //     absolute-value form of a norm.
  //   * `@llvm.exp2.f32` / `@llvm.log2.f32` → `v_exp_f32` /
  //     `v_log_f32` (VOP1).  Softmax's exponentiation and the
  //     `pow` / `log` decomposition both route here.  Per-lane,
  //     all-VGPR.
  //   * `@llvm.floor.f32` / `@llvm.ceil.f32` / `@llvm.trunc.f32` /
  //     `@llvm.rint.f32` / `@llvm.round.f32` / `@llvm.nearbyint.f32`
  //     → `v_floor_f32` / `v_ceil_f32` / `v_trunc_f32` /
  //     `v_rndne_f32` (VOP1).  Per-lane, all-VGPR.  Triton emits
  //     these from integer-float conversions and `tl.cdiv`-style
  //     ceiling division.
  //   * `@llvm.copysign.f32` → `v_bfi_b32` with a sign-bit selector
  //     (backend-canonical).  Per-lane, all-VGPR.
  //   * `@llvm.smin.i32` / `@llvm.smax.i32` / `@llvm.umin.i32` /
  //     `@llvm.umax.i32` → `v_min_i32` / `v_max_i32` / `v_min_u32` /
  //     `v_max_u32`.  Integer lane-parallel min/max; per-lane,
  //     all-VGPR.
  //   * `@llvm.abs.i32` → `v_sub_i32` / `v_max_i32` pair (backend
  //     expansion).  Per-lane, all-VGPR.
  //   * `@llvm.ctpop.i32` / `@llvm.ctlz.i32` / `@llvm.cttz.i32` /
  //     `@llvm.bitreverse.i32` → per-lane bit-counting / bit-reverse
  //     VALU instructions.  All-VGPR.
  //   * `@llvm.fshl.i32` / `@llvm.fshr.i32` → `v_alignbit_b32`
  //     (funnel shift).  Per-lane, all-VGPR.
  //
  // All carry per-source-wave state through unchanged (SIMT per-lane
  // math), so the forward walk must continue past them — hence
  // `VGPRSafePropagator` rather than `VGPRSafeSink`.  When LLVM ever
  // routes one of these operands through an SGPR-constrained form
  // (none exists today), extending `operandForcesSGPR` above would
  // shadow the per-operand entry back to SGPR-forced without having
  // to remove the intrinsic from this list.
  //
  // Intrinsics deliberately NOT whitelisted (require additional
  // audit / may decompose through an SGPR-forced helper): trig
  // functions (`sin` / `cos`), vector-reduction intrinsics
  // (`vector.reduce.*`), `experimental.constrained.*` variants,
  // anything that lowers to a library call.  Add here only after
  // confirming the AMDGPU lowering is a single per-lane VALU
  // instruction (or a VALU-only expansion) with no SGPR-forced
  // operand in any codegen path.
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::sqrt:
  case Intrinsic::maxnum:
  case Intrinsic::minnum:
  case Intrinsic::fabs:
  case Intrinsic::exp2:
  case Intrinsic::log2:
  // `@llvm.ldexp.f32.i32` → `v_ldexp_f32` (VOP2; `x * 2^n` with an
  // integer exponent).  Softmax's normalisation routes through ldexp
  // when the backend's exp2/pow decomposition picks it (LLVM r202+
  // converts `exp(x) * 2^k` patterns there).  Per-lane, all-VGPR.
  case Intrinsic::ldexp:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::round:
  case Intrinsic::nearbyint:
  case Intrinsic::copysign:
  case Intrinsic::smin:
  case Intrinsic::smax:
  case Intrinsic::umin:
  case Intrinsic::umax:
  case Intrinsic::abs:
  case Intrinsic::ctpop:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::bitreverse:
  case Intrinsic::fshl:
  case Intrinsic::fshr:
    return true;
  default:
    return false;
  }
}

// amdgcn.* intrinsics that consume our value as a VGPR input and
// produce a result (or no result) we do not need to track further —
// the result either is unrelated to our value's per-source-wave
// identity (memory-store result, WMMA / MFMA accumulator — which
// contains our value mixed with many other lanes' values, so
// "refuse downstream if WMMA output itself is SGPR-forced" is too
// conservative) or has no users the classifier would care about.
bool isIntrinsicVGPRSafeSink(Intrinsic::ID id) {
  switch (id) {
  // Stores: value consumed, no IR result to propagate.
  case Intrinsic::amdgcn_raw_buffer_store:
  case Intrinsic::amdgcn_tensor_store_from_lds:
  case Intrinsic::amdgcn_global_load_async_to_lds_b8:
  case Intrinsic::amdgcn_global_load_async_to_lds_b32:
  case Intrinsic::amdgcn_global_load_async_to_lds_b64:
  case Intrinsic::amdgcn_global_load_async_to_lds_b128:
  case Intrinsic::amdgcn_tensor_load_to_lds:
  case Intrinsic::amdgcn_global_prefetch:
  case Intrinsic::amdgcn_s_barrier:
    return true;
  // Ballot: `ballot(i1 pred)` is a proper cross-lane REDUCTION that
  // packs every active lane's predicate bit into the result mask
  // without inserting `v_readfirstlane` on the input. The mask is
  // wave-uniform by construction (same bit pattern on every lane),
  // and downstream uses consume it as EXEC manipulation / flow
  // control / mask arithmetic — none of which re-introduces a per-
  // source-wave collapse. Classifying `ballot` as a sink (rather
  // than propagating through its result) matches the AMDGPU
  // divergence analysis' own treatment and keeps the classifier
  // from over-refusing kernels that ballot an `icmp` on a rewritten
  // writelane/readlane result (common Triton reduction shape).
  case Intrinsic::amdgcn_ballot:
    return true;
  // MFMA / WMMA: accumulator inputs are VGPR. The result is a VGPR
  // fragment distributed across lanes; downstream uses of the
  // accumulator flow through `select` reconstruction or memory
  // stores — both VGPR-safe paths. Treating MFMA/WMMA as a sink
  // (rather than a propagator) keeps the classifier linear in the
  // kernel size; a theoretical "MFMA output lane-swapped into an
  // SGPR-forced consumer" shape would require re-classification on
  // the result, which no corpus kernel is known to need today.
  case Intrinsic::amdgcn_mfma_f32_16x16x16bf16_1k:
  case Intrinsic::amdgcn_mfma_f32_16x16x16f16:
  case Intrinsic::amdgcn_mfma_f32_16x16x1f32:
  case Intrinsic::amdgcn_mfma_f32_16x16x2bf16:
  case Intrinsic::amdgcn_mfma_f32_16x16x32_bf16:
  case Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_bf8:
  case Intrinsic::amdgcn_mfma_f32_16x16x32_bf8_fp8:
  case Intrinsic::amdgcn_mfma_f32_16x16x32_f16:
  case Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_bf8:
  case Intrinsic::amdgcn_mfma_f32_16x16x32_fp8_fp8:
  case Intrinsic::amdgcn_mfma_f32_16x16x4f16:
  case Intrinsic::amdgcn_mfma_f32_16x16x4f32:
  case Intrinsic::amdgcn_mfma_f32_16x16x8_xf32:
  case Intrinsic::amdgcn_mfma_f32_32x32x16_bf16:
  case Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_bf8:
  case Intrinsic::amdgcn_mfma_f32_32x32x16_bf8_fp8:
  case Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_bf8:
  case Intrinsic::amdgcn_mfma_f32_32x32x16_fp8_fp8:
  case Intrinsic::amdgcn_mfma_f32_32x32x1f32:
  case Intrinsic::amdgcn_mfma_f32_32x32x2bf16:
  case Intrinsic::amdgcn_mfma_f32_32x32x2f32:
  case Intrinsic::amdgcn_mfma_f32_32x32x4f16:
  case Intrinsic::amdgcn_mfma_f32_32x32x4_xf32:
  case Intrinsic::amdgcn_mfma_f32_32x32x8bf16_1k:
  case Intrinsic::amdgcn_mfma_f32_32x32x8f16:
  case Intrinsic::amdgcn_mfma_f32_4x4x1f32:
  case Intrinsic::amdgcn_mfma_f32_4x4x2bf16:
  case Intrinsic::amdgcn_mfma_f32_4x4x4f16:
  case Intrinsic::amdgcn_mfma_i32_16x16x32_i8:
  case Intrinsic::amdgcn_mfma_i32_16x16x4i8:
  case Intrinsic::amdgcn_mfma_i32_32x32x16_i8:
  case Intrinsic::amdgcn_mfma_i32_32x32x4i8:
  case Intrinsic::amdgcn_mfma_i32_4x4x4i8:
  case Intrinsic::amdgcn_mfma_scale_f32_16x16x128_f8f6f4:
  case Intrinsic::amdgcn_mfma_scale_f32_32x32x64_f8f6f4:
  case Intrinsic::amdgcn_wmma_f32_16x16x32_bf16:
  case Intrinsic::amdgcn_wmma_f32_16x16x32_f16:
  case Intrinsic::amdgcn_wmma_f32_16x16x4_f32:
  case Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_bf8:
  case Intrinsic::amdgcn_wmma_f32_16x16x64_bf8_fp8:
  case Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_bf8:
  case Intrinsic::amdgcn_wmma_f32_16x16x64_fp8_fp8:
  case Intrinsic::amdgcn_wmma_i32_16x16x64_iu8:
  case Intrinsic::amdgcn_wmma_scale_f32_16x16x128_f8f6f4:
    return true;
  default:
    return false;
  }
}

// Classify how `V` (our tracked value) is used by the call
// instruction `CB` at operand index `operandIdx`. Returns the call
// site's role for the purposes of the forward walk.
IntrinsicRole classifyIntrinsicUse(CallBase *CB, Value *V,
                                    unsigned operandIdx) {
  // Inline asm: we cannot audit the constraint letters cheaply here
  // (would need to walk InlineAsm::ParseConstraints and map the
  // physical arg index to the constraint tuple). Refuse — this is
  // the conservative direction and inline asm with SGPR constraints
  // is exactly the shape the rewrite needs to avoid.
  if (CB->isInlineAsm())
    return IntrinsicRole::SGPRForced;

  Function *callee = CB->getCalledFunction();
  if (!callee)
    return IntrinsicRole::SGPRForced; // indirect / unresolved

  Intrinsic::ID id = callee->getIntrinsicID();
  if (id == Intrinsic::not_intrinsic)
    return IntrinsicRole::SGPRForced; // ordinary call — unknown

  if (operandForcesSGPR(id, operandIdx))
    return IntrinsicRole::SGPRForced;

  // `raw_buffer_load` is the one mixed intrinsic in the raiser's
  // emission set: arg 0 (rsrc) is SGPR-forced, arg 3 (soffset) is
  // SGPR-forced, args 1 / 2 / 4 are VGPR-safe. Handled here so the
  // enumeration in `operandForcesSGPR` / `isIntrinsicVGPRSafeSink`
  // does not duplicate the per-operand conditional.
  if (id == Intrinsic::amdgcn_raw_buffer_load) {
    if (operandIdx == 0 || operandIdx == 3)
      return IntrinsicRole::SGPRForced;
    // The result is a fresh value (loaded from memory) that no
    // longer carries our per-source-wave identity; stop walking.
    return IntrinsicRole::VGPRSafeSink;
  }
  if (id == Intrinsic::amdgcn_raw_buffer_store) {
    if (operandIdx == 1 || operandIdx == 4)
      return IntrinsicRole::SGPRForced;
    return IntrinsicRole::VGPRSafeSink;
  }

  if (isIntrinsicVGPRSafePropagator(id))
    return IntrinsicRole::VGPRSafePropagator;
  if (isIntrinsicVGPRSafeSink(id))
    return IntrinsicRole::VGPRSafeSink;

  (void)V; // reserved for future per-operand predicates
  return IntrinsicRole::Unknown;
}

// Address spaces whose pointer operand accepts a divergent VGPR. If
// our tracked value feeds a load/store through one of these, the
// backend emits a VGPR-addressed memory op and the per-source-wave
// identity is preserved through the memory operation (the loaded
// value is a fresh VGPR the classifier no longer tracks).
//
// addrspace(4) (constant) and addrspace(6) (constant32bit) require
// uniform SGPR pointers; a divergent pointer there triggers backend
// scalarisation.
bool isVGPRAddressablePointerAS(unsigned AS) {
  switch (AS) {
  case 0: // flat
  case 1: // global
  case 3: // LDS
  case 5: // private / scratch
  case 7: // buffer fat pointer
    return true;
  default:
    // 2 (region) / 4 (constant) / 6 (constant32bit) / anything we
    // have not audited → refuse.
    return false;
  }
}

// Build a stable one-line description of the blocking use, for the
// refusal diagnostic.
std::string describeUser(const Instruction *I) {
  std::string s;
  raw_string_ostream os(s);
  if (const auto *CB = dyn_cast<CallBase>(I)) {
    if (CB->isInlineAsm())
      os << "inline asm";
    else if (Function *f = CB->getCalledFunction())
      os << "call @" << f->getName();
    else
      os << "indirect call";
  } else if (const auto *LI = dyn_cast<LoadInst>(I)) {
    os << "load (addrspace " << LI->getPointerAddressSpace() << ")";
  } else if (const auto *SI = dyn_cast<StoreInst>(I)) {
    os << "store (addrspace " << SI->getPointerAddressSpace() << ")";
  } else {
    os << I->getOpcodeName();
  }
  return s;
}

// Forward-classify the use chain of `root`. Returns VGPRSafe iff
// every transitive user is a proven-safe consumer; otherwise writes
// a single-line description of the first blocking user to
// `blockingDetail` and returns SGPRForced.
UseChainVerdict classifyForwardUseChain(Value *root,
                                         std::string &blockingDetail) {
  SmallPtrSet<Value *, 32> visited;
  SmallVector<Value *, 16> worklist;
  worklist.push_back(root);

  while (!worklist.empty()) {
    Value *V = worklist.pop_back_val();
    if (!visited.insert(V).second)
      continue;

    for (Use &U : V->uses()) {
      User *UserObj = U.getUser();
      auto *I = dyn_cast<Instruction>(UserObj);
      if (!I) {
        // ConstantExpr or similar non-instruction use. Refuse —
        // the rewrite pass runs post-mem2reg on lifted kernels
        // and the raiser does not emit constant expressions, so
        // this would indicate an unexpected user we cannot
        // prove safe.
        blockingDetail = "non-instruction user (ConstantExpr?)";
        return UseChainVerdict::SGPRForced;
      }

      if (I->isTerminator())
        continue; // br/switch/ret consume as i1/i32; AMDGPU handles via EXEC

      // Pure propagators: forward-walk the instruction's result.
      if (isa<CastInst>(I) || isa<BinaryOperator>(I) ||
          isa<UnaryOperator>(I) || isa<ICmpInst>(I) || isa<FCmpInst>(I) ||
          isa<SelectInst>(I) || isa<PHINode>(I) ||
          isa<GetElementPtrInst>(I) || isa<FreezeInst>(I) ||
          isa<ExtractElementInst>(I) || isa<InsertElementInst>(I) ||
          isa<ShuffleVectorInst>(I) || isa<ExtractValueInst>(I) ||
          isa<InsertValueInst>(I)) {
        worklist.push_back(I);
        continue;
      }

      if (auto *LI = dyn_cast<LoadInst>(I)) {
        if (!isVGPRAddressablePointerAS(LI->getPointerAddressSpace())) {
          blockingDetail = describeUser(I);
          return UseChainVerdict::SGPRForced;
        }
        // Loaded value is a fresh result disconnected from our
        // per-source-wave chain; stop walking.
        continue;
      }
      if (auto *SI = dyn_cast<StoreInst>(I)) {
        // Value operand: always VGPR-safe (hardware write port
        // accepts VGPR regardless of pointer addrspace).
        if (SI->getValueOperand() == V)
          continue;
        // Pointer operand: same addrspace rules as load.
        if (!isVGPRAddressablePointerAS(SI->getPointerAddressSpace())) {
          blockingDetail = describeUser(I);
          return UseChainVerdict::SGPRForced;
        }
        continue;
      }
      if (auto *AI = dyn_cast<AtomicRMWInst>(I)) {
        if (!isVGPRAddressablePointerAS(AI->getPointerAddressSpace())) {
          blockingDetail = describeUser(I);
          return UseChainVerdict::SGPRForced;
        }
        continue;
      }
      if (auto *CmpX = dyn_cast<AtomicCmpXchgInst>(I)) {
        if (!isVGPRAddressablePointerAS(CmpX->getPointerAddressSpace())) {
          blockingDetail = describeUser(I);
          return UseChainVerdict::SGPRForced;
        }
        continue;
      }

      if (auto *CB = dyn_cast<CallBase>(I)) {
        unsigned operandIdx = U.getOperandNo();
        switch (classifyIntrinsicUse(CB, V, operandIdx)) {
        case IntrinsicRole::SGPRForced:
          blockingDetail = describeUser(I);
          return UseChainVerdict::SGPRForced;
        case IntrinsicRole::VGPRSafeSink:
          continue;
        case IntrinsicRole::VGPRSafePropagator:
          worklist.push_back(CB);
          continue;
        case IntrinsicRole::Unknown:
          blockingDetail = describeUser(I) + " (unaudited)";
          return UseChainVerdict::SGPRForced;
        }
      }

      // Any other instruction kind we did not enumerate — refuse.
      blockingDetail = describeUser(I) + " (unaudited)";
      return UseChainVerdict::SGPRForced;
    }
  }
  return UseChainVerdict::VGPRSafe;
}

// ============================================================================
// Rewrite primitives
// ============================================================================

// Build (once per function) the target-wave absolute lane id as the
// standard two-step mbcnt idiom. Returned value dominates every use
// site because it is emitted at the head of the function's entry
// block, immediately after the terminator of `allocas-and-setup`
// prelude (we insert at the entry's first insertion point).
Value *buildTargetLaneId(Function &F) {
  Module *M = F.getParent();
  LLVMContext &C = F.getContext();
  Type *i32Ty = Type::getInt32Ty(C);
  IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
  Function *mbcntLo = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_mbcnt_lo);
  Function *mbcntHi = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_mbcnt_hi);
  // `ConstantInt::get(IntegerType*, uint64_t V, bool IsSigned=false)`
  // asserts `V < 2^BitWidth` when `!IsSigned`; implicit (int64_t)-1 ->
  // uint64_t produces `0xFFFF'FFFF'FFFF'FFFF` which blows that assert
  // for a 32-bit type. Use the unsigned 32-bit all-ones bit pattern
  // (2^32 - 1), which mbcnt hardware interprets as the wave-wide
  // exec-all mask — the standard idiom behind the two-step lane_id
  // construction.
  Value *minusOne = ConstantInt::get(i32Ty, 0xFFFFFFFFu);
  Value *zero = ConstantInt::get(i32Ty, 0);
  Value *laneLo = B.CreateCall(mbcntLo, {minusOne, zero},
                                "cwd_lane_id_lo");
  Value *laneId = B.CreateCall(mbcntHi, {minusOne, laneLo},
                                "cwd_lane_id");
  return laneId;
}

// Rewrite one `amdgcn.writelane(val, lane, old)` call to
// `select ((lane_id & (W_s-1)) == lane), val, old` in-place.
void rewriteWritelaneCall(CallInst *CI, Value *laneId,
                          unsigned sourceWaveSize) {
  IRBuilder<> B(CI);
  B.SetCurrentDebugLocation(CI->getDebugLoc());
  Type *i32Ty = B.getInt32Ty();
  Value *val = CI->getArgOperand(0);
  Value *laneIdx = CI->getArgOperand(1);
  Value *oldVal = CI->getArgOperand(2);
  Value *modMask = ConstantInt::get(i32Ty, sourceWaveSize - 1);
  Value *laneMod = B.CreateAnd(laneId, modMask, "cwd_wl_lane_mod");
  Value *selMask = B.CreateICmpEQ(laneMod, laneIdx, "cwd_wl_mask");
  Value *newVal = B.CreateSelect(selMask, val, oldVal,
                                  "cwd_writelane_rewritten");
  CI->replaceAllUsesWith(newVal);
  CI->eraseFromParent();
}

// Rewrite one `amdgcn.readlane(src, lane)` call to
// `ds_bpermute(((lane_id & ~(W_s-1)) | lane) << 2, src)`.
void rewriteReadlaneCall(CallInst *CI, Value *laneId,
                         unsigned sourceWaveSize) {
  IRBuilder<> B(CI);
  B.SetCurrentDebugLocation(CI->getDebugLoc());
  Module *M = CI->getModule();
  Type *i32Ty = B.getInt32Ty();
  Value *src = CI->getArgOperand(0);
  Value *laneIdx = CI->getArgOperand(1);

  uint32_t baseMaskImm =
      ~(static_cast<uint32_t>(sourceWaveSize) - 1u);
  Value *baseMask = ConstantInt::get(i32Ty, baseMaskImm);
  Value *srcWaveBase = B.CreateAnd(laneId, baseMask,
                                    "cwd_rl_src_wave_base");
  Value *bcastLane = B.CreateOr(srcWaveBase, laneIdx,
                                 "cwd_rl_bcast_lane");
  Value *selector = B.CreateShl(bcastLane, ConstantInt::get(i32Ty, 2),
                                 "cwd_rl_selector");
  Function *bpermute = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_ds_bpermute);
  Value *broadcast = B.CreateCall(bpermute, {selector, src},
                                   "cwd_readlane_rewritten");
  CI->replaceAllUsesWith(broadcast);
  CI->eraseFromParent();
}

// ============================================================================
// DPP rewrite helpers
// ============================================================================

// Per-target-lane source-lane mapping for a supported `dpp_ctrl`
// value.  `supported=false` signals the ctrl is outside the rewrite's
// domain and the pass must refuse the function.
struct DppLaneMap {
  // i32: within-row source-lane index (0..15) — the ds_bpermute
  // selector's intra-row bits.  Unused when `supported=false`.
  Value *srcWithinRow = nullptr;
  // i1: whether the mapping is valid for this target lane (false on
  // out-of-row references per the DPP semantics).  Unused when
  // `supported=false`.
  Value *inRange = nullptr;
  // False if `dpp_ctrl` is outside the supported family.  When
  // false, the rewrite must bail and the caller surfaces a refusal
  // via `report.unsupportedDppDetail`.
  bool supported = false;
};

// Build the per-lane source-lane + in-range mapping for `dpp_ctrl`.
//
// Supported families (covers the observed Triton reduction corpus):
//
//   * QUAD_PERM          (0x000..0x0FF)  — per-quad 4-lane permutation.
//     ctrl encodes four 2-bit selectors; lane L within its 4-lane
//     quad reads source-lane-in-quad = selector[L & 3].  Always
//     in-range.
//
//   * ROW_SL:N           (0x101..0x10F)  — row shift left by N.
//     Target-lane L (within-row W) reads source within-row W + N.
//     Out-of-range iff W + N >= 16.
//
//   * ROW_SR:N           (0x111..0x11F)  — row shift right by N.
//     Target-lane L (within-row W) reads source within-row W - N.
//     Out-of-range iff W < N.
//
// All three families keep the source lane within the same 16-lane
// row as the target lane.  Since a 16-lane row is a topology
// invariant of every AMDGPU wave size >= 16, the `rowBase(L) |
// srcWithinRow` computation produces identical source-lane indices
// on wave32 and wave64, which is precisely the wave-size-
// obliviousness property the rewrite relies on.
//
// Unsupported families (refused loudly via `supported=false`):
//
//   * 0x120..0x12F  — ROW_RR:N (row rotate right).  Rotation keeps
//     data within a 16-lane row, but requires modular arithmetic
//     that this helper could easily extend to.  Left off the
//     supported list until a corpus kernel exercises it — adding it
//     requires only another case below plus a lit fixture.
//
//   * 0x130..0x13F  — wave-wide shifts (WAVE_SHL / WAVE_ROL /
//     WAVE_SHR / WAVE_ROR).  These cross 16-lane row boundaries
//     within the source wave; under cross-widening the "wave"
//     meaning diverges (wave32 = 32 lanes, wave64 = 64 lanes) and
//     the translation is NOT the identity ctrl.  A future rewrite
//     would compute the source-wave boundary via `(L & ~(W_s-1))`
//     and clamp shifts accordingly, but today every Triton
//     reduction we have expresses wave-width reductions via
//     `row_shl/shr` + `permlane16` rather than the wave-wide DPP
//     ctrls, so this family has no corpus demand.
//
//   * 0x140 / 0x141 — ROW_MIRROR / ROW_HALF_MIRROR.  Within a
//     16-lane row, so they WOULD be expressible here — but they
//     also don't appear in the reduction corpus yet.  Easy to add
//     when needed; keeping the initial rollout narrow.
//
//   * 0x142 / 0x143 — BCAST15 / BCAST31 (gfx9-only).  These cross
//     16- and 32-lane row boundaries respectively.  gfx1250 source
//     cannot emit them (they were removed in RDNA), so refusing
//     is no regression.
//
//   * 0x150..0x15F — ROW_SHARE:N (gfx10+).  Broadcasts lane N of
//     each row to all other lanes in that row.  Expressible here
//     (set srcWithinRow = N, inRange = true), but not yet required.
//
//   * 0x160..0x16F — ROW_XMASK:N (gfx10+).  Each lane reads from
//     its XOR-N partner within the row.  Expressible here via
//     `srcWithinRow = withinRow ^ N`, but not yet required.
//
// When extending this table, prefer a one-case-per-ctrl-family
// layout and document the in-range predicate and source-lane
// formula alongside each case — the correctness argument is local
// per ctrl value.
DppLaneMap buildDppLaneMap(IRBuilder<> &B, Value *withinRow,
                            unsigned ctrl) {
  DppLaneMap out;
  Type *i32Ty = B.getInt32Ty();

  if (ctrl <= 0x0FF) {
    // QUAD_PERM family. Decode the 4 two-bit selectors on-the-fly
    // so the rewrite works for every ctrl value in [0, 0x100)
    // without a 256-way switch.  The 4-lane quad the target lane
    // sits in is `withinRow & ~3`; the target-lane's position
    // within that quad is `withinRow & 3`; the 2-bit selector
    // lives at bits `[2 * (withinRow & 3) .. 2 * (withinRow & 3) + 1]`
    // of `ctrl`.
    Value *quadBase = B.CreateAnd(withinRow, ConstantInt::get(i32Ty, ~3u),
                                    "cwd_dpp_quad_base");
    Value *quadWithin = B.CreateAnd(withinRow, ConstantInt::get(i32Ty, 3),
                                     "cwd_dpp_quad_within");
    Value *shift = B.CreateShl(quadWithin, ConstantInt::get(i32Ty, 1),
                                "cwd_dpp_quad_shift");
    Value *ctrlVal = ConstantInt::get(i32Ty, ctrl);
    Value *selector = B.CreateAnd(B.CreateLShr(ctrlVal, shift),
                                   ConstantInt::get(i32Ty, 3),
                                   "cwd_dpp_quad_sel");
    out.srcWithinRow = B.CreateOr(quadBase, selector, "cwd_dpp_quad_src");
    out.inRange = ConstantInt::getTrue(B.getContext());
    out.supported = true;
    return out;
  }

  if (ctrl >= 0x101 && ctrl <= 0x10F) {
    // ROW_SL:N.  Source within-row = withinRow + N; OOB iff the sum
    // falls outside [0, 16).  Use unsigned comparison — withinRow
    // is already masked to [0, 16) by the caller's `laneId & 0xF`,
    // so the addition cannot wrap.
    unsigned N = ctrl - 0x100;
    Value *nVal = ConstantInt::get(i32Ty, N);
    out.srcWithinRow = B.CreateAdd(withinRow, nVal, "cwd_dpp_sl_src");
    out.inRange = B.CreateICmpULT(out.srcWithinRow,
                                   ConstantInt::get(i32Ty, 16),
                                   "cwd_dpp_sl_inrange");
    out.supported = true;
    return out;
  }

  if (ctrl >= 0x111 && ctrl <= 0x11F) {
    // ROW_SR:N.  Source within-row = withinRow - N; OOB iff
    // withinRow < N.  Compute srcWithinRow as a plain i32 subtract
    // — the select on `inRange` at the caller clamps the bogus
    // wrap-around result before it feeds the ds_bpermute selector.
    unsigned N = ctrl - 0x110;
    Value *nVal = ConstantInt::get(i32Ty, N);
    out.inRange = B.CreateICmpUGE(withinRow, nVal, "cwd_dpp_sr_inrange");
    out.srcWithinRow = B.CreateSub(withinRow, nVal, "cwd_dpp_sr_src");
    out.supported = true;
    return out;
  }

  // Unsupported — leave out.supported = false.
  return out;
}

// Format a `dpp_ctrl` value as a human-readable name for the refusal
// diagnostic.  Keeps the refusal message grep-able per control family
// so triage doesn't need an ISA reference open.
std::string describeDppCtrl(unsigned ctrl) {
  std::string s;
  raw_string_ostream os(s);
  if (ctrl <= 0x0FF) {
    os << "quad_perm:[" << (ctrl & 3) << "," << ((ctrl >> 2) & 3) << ","
       << ((ctrl >> 4) & 3) << "," << ((ctrl >> 6) & 3) << "]";
  } else if (ctrl >= 0x101 && ctrl <= 0x10F) {
    os << "row_shl:" << (ctrl - 0x100);
  } else if (ctrl >= 0x111 && ctrl <= 0x11F) {
    os << "row_shr:" << (ctrl - 0x110);
  } else if (ctrl >= 0x120 && ctrl <= 0x12F) {
    os << "row_ror:" << (ctrl - 0x120);
  } else if (ctrl >= 0x130 && ctrl <= 0x13F) {
    os << "wave-wide dpp_ctrl=0x" << utohexstr(ctrl);
  } else if (ctrl == 0x140) {
    os << "row_mirror";
  } else if (ctrl == 0x141) {
    os << "row_half_mirror";
  } else if (ctrl == 0x142) {
    os << "row_bcast15 (gfx9-only)";
  } else if (ctrl == 0x143) {
    os << "row_bcast31 (gfx9-only)";
  } else if (ctrl >= 0x150 && ctrl <= 0x15F) {
    os << "row_share:" << (ctrl & 0xF);
  } else if (ctrl >= 0x160 && ctrl <= 0x16F) {
    os << "row_xmask:" << (ctrl & 0xF);
  } else {
    os << "dpp_ctrl=0x" << utohexstr(ctrl);
  }
  return s;
}

// Rewrite one `amdgcn.update.dpp.i32(old, src, dpp_ctrl, row_mask,
// bank_mask, bound_ctrl)` call.  Returns `true` on success; on
// failure (unsupported `dpp_ctrl`), leaves the call untouched,
// writes the describe-string to `*unsupportedDetail`, and returns
// `false`.  Caller must populate the report's
// `unsupportedDppDetail` and refuse the function — symmetric with
// the SGPR-forced classifier's all-or-nothing semantics.
//
// Only called for i32-overloaded DPP.  i64 DPP sites are left to
// the backend's native lowering (see the header's "@llvm.amdgcn.
// update.dpp" paragraph for the i32-only scope rationale).
bool rewriteUpdateDppI32Call(CallInst *CI, Value *laneId,
                              std::string *unsupportedDetail) {
  IRBuilder<> B(CI);
  B.SetCurrentDebugLocation(CI->getDebugLoc());
  Module *M = CI->getModule();
  Type *i32Ty = B.getInt32Ty();

  // The intrinsic's TableGen declaration marks args 2..5 as `ImmArg`,
  // so the raiser (and any well-formed caller) always passes them as
  // `ConstantInt`. Cast assertively; a non-ConstantInt here indicates
  // a caller invariant violation worth asserting loudly.
  Value *oldVal = CI->getArgOperand(0);
  Value *src = CI->getArgOperand(1);
  auto *ctrlC = cast<ConstantInt>(CI->getArgOperand(2));
  auto *rowMaskC = cast<ConstantInt>(CI->getArgOperand(3));
  auto *bankMaskC = cast<ConstantInt>(CI->getArgOperand(4));
  auto *boundCtrlC = cast<ConstantInt>(CI->getArgOperand(5));
  unsigned ctrl = ctrlC->getZExtValue();
  unsigned rowMaskImm = rowMaskC->getZExtValue();
  unsigned bankMaskImm = bankMaskC->getZExtValue();
  bool boundCtrl = boundCtrlC->getZExtValue() != 0;

  // Lane-topology values — derived once per rewrite, reused across
  // the three selects below.  `laneId` itself is memoised at the
  // function level by the caller (`buildTargetLaneId`), so the only
  // duplication across DPP sites is the and/lshr chain, which
  // instcombine folds post-pass.
  Value *withinRow = B.CreateAnd(laneId, ConstantInt::get(i32Ty, 0xF),
                                  "cwd_dpp_within_row");
  Value *rowIdx =
      B.CreateAnd(B.CreateLShr(laneId, ConstantInt::get(i32Ty, 4)),
                   ConstantInt::get(i32Ty, 3), "cwd_dpp_row");
  Value *bankIdx =
      B.CreateAnd(B.CreateLShr(laneId, ConstantInt::get(i32Ty, 2)),
                   ConstantInt::get(i32Ty, 3), "cwd_dpp_bank");
  Value *rowBase = B.CreateAnd(laneId, ConstantInt::get(i32Ty, ~0xFu),
                                "cwd_dpp_row_base");

  // Per-ctrl source mapping.
  DppLaneMap m = buildDppLaneMap(B, withinRow, ctrl);
  if (!m.supported) {
    if (unsupportedDetail)
      *unsupportedDetail = describeDppCtrl(ctrl);
    return false;
  }

  // Clamp the bogus wrap-around result on OOB so the ds_bpermute
  // selector always references a deterministic intra-row lane.  The
  // `inRange` select below discards the bpermuted value for OOB
  // lanes, so the clamp is strictly for IR clarity — lane 0's
  // selector reads row[0] instead of row[0xFFFF_FFF8 & 0x3F].
  Value *srcWithinRowSafe = B.CreateSelect(
      m.inRange, m.srcWithinRow, ConstantInt::get(i32Ty, 0),
      "cwd_dpp_src_safe");
  Value *srcLaneAbs = B.CreateOr(rowBase, srcWithinRowSafe,
                                  "cwd_dpp_src_abs");
  Value *byteAddr = B.CreateShl(srcLaneAbs, ConstantInt::get(i32Ty, 2),
                                 "cwd_dpp_selector");

  Function *bpermute = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::amdgcn_ds_bpermute);
  Value *bperm = B.CreateCall(bpermute, {byteAddr, src},
                               "cwd_dpp_bperm");

  // Out-of-range disposition.  Per the AMDGPU ISA DPP spec: an active
  // target lane whose source lane is OOB receives `0` under
  // `bound_ctrl=1` or retains `old` under `bound_ctrl=0`.
  Value *oobVal = boundCtrl ? static_cast<Value *>(ConstantInt::get(i32Ty, 0))
                             : oldVal;
  Value *dppVal = B.CreateSelect(m.inRange, bperm, oobVal,
                                  "cwd_dpp_inrange");

  // row_mask / bank_mask gating.  Fold the select away when both
  // masks are 0xF (the common "every lane participates" case) —
  // keeps the rewritten IR minimal for the overwhelmingly common
  // reduction-tree shape the corpus emits, and keeps lit-test
  // FileCheck patterns simple.
  Value *result;
  if (rowMaskImm == 0xF && bankMaskImm == 0xF) {
    result = dppVal;
  } else {
    Value *rowMaskVal = ConstantInt::get(i32Ty, rowMaskImm);
    Value *bankMaskVal = ConstantInt::get(i32Ty, bankMaskImm);
    Value *rowActive = B.CreateICmpNE(
        B.CreateAnd(B.CreateLShr(rowMaskVal, rowIdx),
                     ConstantInt::get(i32Ty, 1)),
        ConstantInt::get(i32Ty, 0), "cwd_dpp_row_active");
    Value *bankActive = B.CreateICmpNE(
        B.CreateAnd(B.CreateLShr(bankMaskVal, bankIdx),
                     ConstantInt::get(i32Ty, 1)),
        ConstantInt::get(i32Ty, 0), "cwd_dpp_bank_active");
    Value *laneActive = B.CreateAnd(rowActive, bankActive,
                                     "cwd_dpp_lane_active");
    result = B.CreateSelect(laneActive, dppVal, oldVal, "cwd_dpp_gated");
  }

  CI->replaceAllUsesWith(result);
  CI->eraseFromParent();
  return true;
}

} // namespace

CrossLaneDivergentRewriteReport rewriteCrossLaneDivergent(
    Function &F, unsigned sourceWaveSize, unsigned targetWaveSize) {
  CrossLaneDivergentRewriteReport report;

  // Direction gate. Same-wave / narrowing skip the rewrite entirely:
  // the backend's implicit readfirstlane would not collapse any per-
  // source-wave state that the target wave does not also hold.
  if (targetWaveSize <= sourceWaveSize)
    return report;

  // Pre-collect candidate call sites. Iterating the function while
  // rewriting would mutate the CFG under the iterator; the two-phase
  // shape keeps the walk O(n) and the rewrite-phase linear in the
  // number of matched sites.
  //
  // DPP collection is i32-only: the rewrite's `ds_bpermute` path is
  // i32-typed and i64 DPP sites keep their native `@llvm.amdgcn.
  // update.dpp.i64` lowering via the backend's implicit split
  // (correct but not wave-size-aware; see the header for rationale
  // of the i32-only rollout).  Any future widening of the rewrite
  // to i64 would add a split/recombine shim here alongside this
  // walk and need to update the symmetry invariant downstream.
  SmallVector<CallInst *, 16> writelaneSites;
  SmallVector<CallInst *, 16> readlaneSites;
  SmallVector<CallInst *, 16> dppI32Sites;
  for (Instruction &I : instructions(F)) {
    auto *CI = dyn_cast<CallInst>(&I);
    if (!CI)
      continue;
    Function *callee = CI->getCalledFunction();
    if (!callee)
      continue;
    switch (callee->getIntrinsicID()) {
    case Intrinsic::amdgcn_writelane:
      writelaneSites.push_back(CI);
      break;
    case Intrinsic::amdgcn_readlane:
      readlaneSites.push_back(CI);
      break;
    case Intrinsic::amdgcn_update_dpp:
      if (CI->getType() == Type::getInt32Ty(F.getContext()))
        dppI32Sites.push_back(CI);
      // i64 DPP: intentionally left unrewritten (see walk comment).
      break;
    default:
      break;
    }
  }

  if (writelaneSites.empty() && readlaneSites.empty() &&
      dppI32Sites.empty())
    return report;

  // ==== Phase A: use-chain classification =================================
  //
  // Forward-walk the uses of every writelane result and every
  // readlane result. If any site's chain reaches an SGPR-constrained
  // consumer (s_buffer_load rsrc, s_sendmsg message, readfirstlane,
  // addrspace(4) load, inline asm `"s"`, or any unaudited sink),
  // refuse the whole function rather than produce an asymmetric
  // rewrite or a backend-re-scalarised rewrite. The refusal is all-
  // or-nothing to preserve the writelane/readlane symmetry invariant
  // on shared VGPRs — a mix of rewritten and preserved sites on one
  // VGPR recreates the Matmul128x128 aperture-violation pattern
  // (hotswap/docs/learnings.md 2026-04-21 entry).
  auto classifySite = [&](CallInst *CI,
                          const char *kind) -> bool {
    std::string detail;
    if (classifyForwardUseChain(CI, detail) ==
        UseChainVerdict::VGPRSafe)
      return true;
    std::string msg;
    raw_string_ostream os(msg);
    os << "function '" << F.getName() << "' has a " << kind
       << " whose use chain reaches an SGPR-forced consumer ("
       << detail
       << "). Rewriting to `ds_bpermute` here would re-introduce "
          "`v_readfirstlane` at the SGPR boundary — refusing rather "
          "than silently miscompiling. See "
          "hotswap/docs/wave-size-translation.md \u00a75.6.3 (use-"
          "chain constraint).";
    report.sgprForcedDetail = os.str();
    return false;
  };

  for (CallInst *CI : writelaneSites)
    if (!classifySite(CI, "writelane"))
      return report;
  for (CallInst *CI : readlaneSites)
    if (!classifySite(CI, "readlane"))
      return report;
  for (CallInst *CI : dppI32Sites)
    if (!classifySite(CI, "update.dpp"))
      return report;

  // ==== Phase B: per-DPP ctrl pre-flight ==================================
  //
  // Refuse the whole function BEFORE any rewrite if ANY DPP ctrl is
  // outside the supported family.  Extends the symmetry invariant to
  // DPP: if we rewrote some DPPs to ds_bpermute while leaving others
  // as native `@llvm.amdgcn.update.dpp`, the mixed state would hit
  // the same source-wave asymmetry trap that writelane/readlane
  // symmetry exists to prevent (a shared VGPR written by one form
  // and read by the other produces divergent data).  All-or-nothing.
  //
  // Pre-flight via a non-mutating decode: walk the collected DPP
  // sites, call `buildDppLaneMap` with a dummy IRBuilder, check
  // `supported`.  Any failure populates the report and returns zero
  // rewrites across all three primitive families.
  {
    LLVMContext &ctx = F.getContext();
    IRBuilder<> preflightBuilder(ctx);
    // `buildDppLaneMap` only touches the builder for constant folding
    // / no-op scaffolding that instcombine later removes; for the
    // pre-flight we only care about the `supported` flag and the
    // describe-string, so point the builder at the end of the
    // function's entry block (guaranteed to exist) to keep any stray
    // inserted ops well-formed.  They are never committed because
    // the `DppLaneMap`-producing IRs are discarded if unsupported.
    preflightBuilder.SetInsertPoint(&F.getEntryBlock(),
                                     F.getEntryBlock().getFirstInsertionPt());
    Value *dummyWithinRow =
        ConstantInt::get(Type::getInt32Ty(ctx), 0);
    for (CallInst *CI : dppI32Sites) {
      unsigned ctrl =
          cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
      DppLaneMap probe = buildDppLaneMap(preflightBuilder,
                                          dummyWithinRow, ctrl);
      if (!probe.supported) {
        std::string msg;
        raw_string_ostream os(msg);
        os << "function '" << F.getName()
           << "' has an update.dpp site with unsupported "
           << describeDppCtrl(ctrl)
           << ". The cross-widen rewrite only covers quad_perm, "
              "row_shl:N and row_shr:N today (all stay within a "
              "single 16-lane row, hence wave-size-oblivious). "
              "Extending the supported set requires a per-ctrl "
              "correctness argument in buildDppLaneMap and a new "
              "lit fixture; refusing rather than silently miscompiling. "
              "See hotswap/docs/wave-size-translation.md \u00a75.3.";
        report.unsupportedDppDetail = os.str();
        return report;
      }
    }
  }

  // ==== Phase C: unconditional symmetric rewrite ==========================
  //
  // Every writelane, every readlane, and every i32 DPP under cross-
  // widening is rewritten — uniform operands included — because an
  // asymmetric mix (native `v_writelane_b32` + rewritten `ds_bpermute`
  // on the same VGPR, or native `v_mov_b32_dpp` + rewritten
  // `ds_bpermute + select` reading each other's outputs) is silently
  // unsound (see the header comment in
  // rewrite_cross_lane_divergent.hpp, "WRITELANE / READLANE
  // SYMMETRY" and the `@llvm.amdgcn.update.dpp` paragraph). The
  // `select` / `ds_bpermute` forms are semantically equivalent to
  // the source opcodes for every (val, old, src) divergence triple
  // so unconditional rewriting is correctness-preserving.
  Value *laneIdCached = nullptr;
  auto getLaneId = [&]() -> Value * {
    if (!laneIdCached)
      laneIdCached = buildTargetLaneId(F);
    return laneIdCached;
  };

  for (CallInst *CI : writelaneSites) {
    rewriteWritelaneCall(CI, getLaneId(), sourceWaveSize);
    ++report.writelaneRewritten;
  }
  for (CallInst *CI : readlaneSites) {
    rewriteReadlaneCall(CI, getLaneId(), sourceWaveSize);
    ++report.readlaneRewritten;
  }
  for (CallInst *CI : dppI32Sites) {
    std::string unsupported;
    // Pre-flight (Phase B) guaranteed every collected ctrl is
    // supported, so this cannot fail; still thread the out-param
    // through and assert on unexpected failure rather than risk
    // producing a half-rewritten function.
    bool ok = rewriteUpdateDppI32Call(CI, getLaneId(), &unsupported);
    (void)ok;
    assert(ok && "DPP pre-flight said all ctrls supported, but rewriteUpdateDppI32Call rejected one "
                  "— buildDppLaneMap's supported check and rewriteUpdateDppI32Call's check "
                  "disagreed, indicating a rewrite-pass invariant violation");
    ++report.dppRewritten;
  }

  return report;
}

} // namespace transpiler
