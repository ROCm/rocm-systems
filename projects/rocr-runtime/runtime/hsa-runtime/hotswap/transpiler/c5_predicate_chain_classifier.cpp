#include "c5_predicate_chain_classifier.hpp"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace transpiler {

namespace {

// ============================================================================
// Narrow-O1 classifier — see hotswap/docs/modrep-predicate-chain.md §5 (O1)
// and §9.6 for the narrowing rationale. The walker mirrors the shape of
// `rewrite_cross_lane_divergent.cpp::classifyForwardUseChain`: forward-walk
// from a source call, classify each user, refuse-on-match or continue.
// ============================================================================

// True iff `V` is a non-negative `ConstantInt` whose unsigned value lies
// strictly inside `(0, bound]`. We exclude zero because `icmp %v, 0` is
// a mask-extraction / null-check idiom independent of lane position, and
// clamping to `>= 0` keeps the LLVM "signed constant happens to fit in
// `bound`" case honest (the classifier exists to refuse specifically the
// `tid < 2^s` / `tid < W_s/2` idioms, all of which use positive constants).
bool isNonZeroConstantUleBound(const Value *V, uint64_t bound) {
  const auto *ci = dyn_cast<ConstantInt>(V);
  if (!ci)
    return false;
  const APInt &val = ci->getValue();
  if (val.isNegative())
    return false;
  if (val.isZero())
    return false;
  return val.ule(bound);
}

// Return the "other" operand of a two-operand instruction. Used for
// `and %v, K` / `icmp %v, %other` shape checks where we already know `%v`
// is on the worklist (one of the two operands) and need to inspect the
// sibling operand.
Value *otherOperand(const Instruction *I, const Value *V) {
  if (I->getNumOperands() != 2)
    return nullptr;
  Value *a = I->getOperand(0);
  Value *b = I->getOperand(1);
  return (a == V) ? b : a;
}

// True iff `I` is an `and` that AND-masks `V` against a compile-time
// constant K with `K <= sourceWaveSize - 1`. Matches the §5.6.2
// `wave_id` lift's `and X, 0x1F` mask, the SPE prelude's
// `lane_id & (execBits - 1)` mask, and the Triton-emitted
// `offs & (BLOCK_SIZE - 1)` mask when `BLOCK_SIZE <= W_s`.
bool isSourceWaveMaskAnd(const Instruction *I, const Value *V,
                          unsigned sourceWaveSize) {
  const auto *bop = dyn_cast<BinaryOperator>(I);
  if (!bop || bop->getOpcode() != Instruction::And)
    return false;
  const Value *other = otherOperand(I, V);
  if (!other)
    return false;
  // Zero is fine here because `and %v, 0` is always masked (result is
  // structurally 0, the strictest mask). Hence `isConstantUleBound`
  // with the zero case allowed, not the non-zero-only variant.
  const auto *ci = dyn_cast<ConstantInt>(other);
  if (!ci)
    return false;
  const APInt &val = ci->getValue();
  if (val.isNegative())
    return false;
  return val.ule(sourceWaveSize - 1);
}

// Pure-propagator set. Any instruction in this set keeps `V`'s per-lane
// divergence in its result; the walker should push the result onto the
// worklist so downstream uses are inspected too.
//
// Intentionally excludes `and` — that is handled by the specialised
// mask check above and must not fall through to "continue walking the
// result as TidDerived" when the `and` is the W_s-1 mask.
//
// Also excludes `icmp` (handled separately as a refusal gate) and
// memory ops (sinks — the value was consumed verbatim, no further
// tid-divergence carried by the memory op's own result).
bool isPurePropagator(const Instruction *I) {
  if (isa<CastInst>(I))
    return true;
  if (isa<PHINode>(I))
    return true;
  if (isa<SelectInst>(I))
    return true;
  if (isa<GetElementPtrInst>(I))
    return true;
  if (isa<FreezeInst>(I))
    return true;
  if (isa<ExtractElementInst>(I) || isa<InsertElementInst>(I) ||
      isa<ShuffleVectorInst>(I) || isa<ExtractValueInst>(I) ||
      isa<InsertValueInst>(I))
    return true;
  if (const auto *bop = dyn_cast<BinaryOperator>(I)) {
    // `and` is handled by `isSourceWaveMaskAnd` above; a non-masking
    // `and` (other operand is not a compile-time W_s-1 constant)
    // still propagates tid-divergence, so keep walking through it.
    switch (bop->getOpcode()) {
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::And:
      return true;
    default:
      return false;
    }
  }
  if (isa<UnaryOperator>(I))
    return true;
  return false;
}

// Build the refusal-detail string for the diagnostic. Kept short and
// stable so lit fixtures can pin the refusal on substrings without
// tying themselves to exact wording.
std::string formatRefusalDetail(const ICmpInst *cmp, const ConstantInt *K,
                                 unsigned sourceWaveSize) {
  std::string s;
  raw_string_ostream os(s);
  os << "`workitem.id.x()` reaches `icmp";
  switch (cmp->getPredicate()) {
  case ICmpInst::ICMP_EQ:  os << " eq"; break;
  case ICmpInst::ICMP_NE:  os << " ne"; break;
  case ICmpInst::ICMP_ULT: os << " ult"; break;
  case ICmpInst::ICMP_ULE: os << " ule"; break;
  case ICmpInst::ICMP_UGT: os << " ugt"; break;
  case ICmpInst::ICMP_UGE: os << " uge"; break;
  case ICmpInst::ICMP_SLT: os << " slt"; break;
  case ICmpInst::ICMP_SLE: os << " sle"; break;
  case ICmpInst::ICMP_SGT: os << " sgt"; break;
  case ICmpInst::ICMP_SGE: os << " sge"; break;
  default:                 os << " ?"; break;
  }
  os << "` against compile-time constant " << K->getValue()
     << " which is within (0, W_s-1=" << (sourceWaveSize - 1) << "]. ";
  os << "This is a lane-position-scoped predicate; under cross-widening ";
  os << "modulo-replication, source wave 0's lane L and target replica-1's ";
  os << "lane L+W_s would evaluate the predicate differently despite sharing ";
  os << "the same source EXEC bit. Refusing per hotswap/docs/"
        "modrep-predicate-chain.md \u00a75 (narrow-O1 classifier).";
  return s;
}

} // namespace

PredicateChainClassifierReport classifyPredicateChain(
    Function &F, unsigned sourceWaveSize, unsigned targetWaveSize,
    bool waveNative) {
  PredicateChainClassifierReport report;

  // Direction gate: no predicate-chain risk at same-wave or narrowing,
  // and a zero/one-lane source wave is structurally incoherent.
  if (targetWaveSize <= sourceWaveSize)
    return report;
  if (sourceWaveSize < 2)
    return report;

  // Projection gate: under WaveNativeProjection the refusal rationale
  // does not apply (each target lane is its own source lane; no
  // MODREP replica-1 sharing source wave 0's EXEC). See the
  // `waveNative` parameter docstring in the header for the detailed
  // model; this early-return is the structural consequence of that
  // reasoning. No sites tracked because the classifier should not
  // surface any end-to-end signal under a projection where its
  // diagnostic would be a false positive.
  if (waveNative)
    return report;

  // Collect every `@llvm.amdgcn.workitem.id.x()` call site. Matches
  // the pattern used by `rewrite_cross_lane_divergent.cpp` for its
  // own site collection — a single O(n) walk, worklist processing
  // separated so we iterate a stable container.
  SmallVector<CallInst *, 4> sites;
  for (Instruction &I : instructions(F)) {
    auto *CI = dyn_cast<CallInst>(&I);
    if (!CI)
      continue;
    Function *callee = CI->getCalledFunction();
    if (!callee)
      continue;
    if (callee->getIntrinsicID() == Intrinsic::amdgcn_workitem_id_x)
      sites.push_back(CI);
  }

  report.visitedCalls = static_cast<unsigned>(sites.size());

  for (CallInst *root : sites) {
    SmallPtrSet<Value *, 32> visited;
    SmallVector<Value *, 16> worklist;
    worklist.push_back(root);

    while (!worklist.empty()) {
      Value *V = worklist.pop_back_val();
      if (!visited.insert(V).second)
        continue;

      for (Use &U : V->uses()) {
        auto *I = dyn_cast<Instruction>(U.getUser());
        if (!I)
          continue;

        // Mask-by-(W_s-1) check: if the `and` instance folds our
        // value into `[0, W_s)`, downstream uses cannot re-introduce
        // per-replica divergence. Stop walking this user.
        if (isSourceWaveMaskAnd(I, V, sourceWaveSize))
          continue;

        // Refusal gate: `icmp` with a compile-time constant operand
        // in (0, W_s-1]. This is the lane-position-scoped predicate
        // shape the narrow-O1 classifier is defined to catch.
        if (auto *cmp = dyn_cast<ICmpInst>(I)) {
          const Value *other = otherOperand(cmp, V);
          if (other &&
              isNonZeroConstantUleBound(other, sourceWaveSize - 1)) {
            const auto *K = dyn_cast<ConstantInt>(other);
            report.refused = true;
            report.refusalDetail = formatRefusalDetail(cmp, K, sourceWaveSize);
            return report;
          }
          // Non-matching icmp: treat as a bounds-check / generic
          // predicate sink. The icmp result does not carry forward
          // tid's lane-position divergence in a way the narrow-O1
          // classifier refuses on, so stop walking past it.
          continue;
        }

        // Pure propagators: forward-walk through the result.
        if (isPurePropagator(I)) {
          worklist.push_back(I);
          continue;
        }

        // Anything else (call / store / load / atomic / branch /
        // return / unknown): sink. The narrow-O1 classifier scopes
        // refusal strictly to the icmp-with-small-K shape, so sinks
        // are allowed. If a future iteration widens the class, the
        // new gate goes in the `isa<ICmpInst>` arm above alongside
        // the current check.
      }
    }
  }

  return report;
}

} // namespace transpiler
