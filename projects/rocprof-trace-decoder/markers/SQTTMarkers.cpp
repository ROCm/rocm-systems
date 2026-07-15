// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "SQTTPass.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

static constexpr const char* MarkerHeaderMetadataName = "sqtt.marker_header";
static constexpr const char* RawPayloadMetadataName = "sqtt.raw_payload";
static constexpr uint32_t GFX12_SHADER_CYCLES_LO = 29;

// True for instructions that already act as a *scheduling* boundary at the
// IR/MIR level — i.e. the optimizer/scheduler will not move s_ttracedata
// across them. Only sched_barrier and IR fences qualify.
//
// NOTE: s_barrier (and the split signal/wait family) intentionally do NOT
// belong here. They are marked IntrNoMem in the AMDGPU intrinsic table, so
// from the optimizer's point of view they do not constrain reordering of
// s_ttracedata (which is IntrInaccessibleMemOnly). Without an explicit
// sched_barrier(0), the trace marker will drift away from the s_barrier and
// the recorded trace timing becomes meaningless.
static bool isHardSchedBoundary(Instruction* I)
{
    if (!I) return false;
    if (auto* CI = dyn_cast<CallInst>(I))
    {
        if (Function* F = CI->getCalledFunction())
        {
            if (F->getIntrinsicID() == Intrinsic::amdgcn_sched_barrier) return true;
        }
    }
    if (isa<FenceInst>(I)) return true;
    return false;
}

// Emit a memory-reordering barrier per Config.MemBarrier:
//   None       — nothing
//   AsmClobber — empty inline asm with "~{memory}" (compiler fence only)
//   Fence      — fence syncscope("workgroup") acq_rel with AMDGPU MMRA
//                metadata limiting hardware synchronization to LDS/local.
//                The IR fence remains a compiler-visible ordering boundary.
static void emitMemBarrier(IRBuilder<>& B, MemBarrierMode mode)
{
    if (mode == MemBarrierMode::None) return;
    LLVMContext& Ctx = B.getContext();
    if (mode == MemBarrierMode::AsmClobber)
    {
        InlineAsm* MF =
            InlineAsm::get(FunctionType::get(Type::getVoidTy(Ctx), false), "", "~{memory}", /*hasSideEffects=*/true);
        B.CreateCall(MF);
        return;
    }
    // MemBarrierMode::Fence
    SyncScope::ID WG = Ctx.getOrInsertSyncScopeID("workgroup");
    FenceInst* F = B.CreateFence(AtomicOrdering::AcquireRelease, WG);

    // SQTT markers do not read or write global memory; the fence exists to
    // keep optimizers and schedulers from moving markers away from nearby
    // memory operations.  AMDGPU's amdgpu-synchronize-as MMRA lets the backend
    // keep the fence as an ordering boundary while avoiding global cache
    // invalidation for this marker-only fence.
    Metadata* LocalSyncAS[] = {MDString::get(Ctx, "amdgpu-synchronize-as"), MDString::get(Ctx, "local")};
    F->setMetadata(LLVMContext::MD_mmra, MDNode::get(Ctx, LocalSyncAS));
}

void SQTTInstrumentPass::emitTraceBlockBoundary(IRBuilder<>& B, bool after)
{
    Module* M = B.GetInsertBlock()->getParent()->getParent();
    Type* I32 = Type::getInt32Ty(B.getContext());
    Function* SchedBarrier = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_sched_barrier);

    // Keep the complete address block between two scheduling boundaries,
    // rather than adding one around every raw payload.
    if (after)
    {
        emitMemBarrier(B, Config.MemBarrier);
        B.CreateCall(SchedBarrier, {ConstantInt::get(I32, 0)});
    }
    else
    {
        B.CreateCall(SchedBarrier, {ConstantInt::get(I32, 0)});
        emitMemBarrier(B, Config.MemBarrier);
    }
}

void SQTTInstrumentPass::insertTraceMarker(IRBuilder<>& B, uint32_t markerID, Function& F, GfxGen gen)
{
    insertTraceMarkerWithPayload(B, markerID, nullptr, F, gen);
}

void SQTTInstrumentPass::insertTraceMarkerWithPayload(
    IRBuilder<>& B, uint32_t markerID, Value* payload, Function& F, GfxGen gen
)
{
    Module* M = F.getParent();
    LLVMContext& Ctx = M->getContext();
    Type* I32 = Type::getInt32Ty(Ctx);
    Type* I16 = Type::getInt16Ty(Ctx);

    bool needsBarriers = !Config.needsScopeCheck();

    auto emitMarker = [&](IRBuilder<>& Builder)
    {
        // If the marker is being inserted adjacent to a real hard scheduling
        // boundary (s_barrier, fence, etc.), skip the redundant sched_barrier
        // on that side — it would only inhibit useful scheduling without
        // adding any reordering protection.
        BasicBlock* IBB = Builder.GetInsertBlock();
        auto IP = Builder.GetInsertPoint();
        Instruction* PrevI = (IP == IBB->begin()) ? nullptr : &*std::prev(IP);
        Instruction* NextI = (IP == IBB->end()) ? nullptr : &*IP;
        bool skipSchedBefore = isHardSchedBoundary(PrevI);
        bool skipSchedAfter = isHardSchedBoundary(NextI);

        if (needsBarriers && !skipSchedBefore)
        {
            Function* SchedBarrier = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_sched_barrier);
            Builder.CreateCall(SchedBarrier, {ConstantInt::get(I32, 0)});
        }
        emitMemBarrier(Builder, Config.MemBarrier);

        CallInst* trace;
        if (canUseImm(markerID) && supportsImmTrace(gen))
        {
            Function* TTDImm = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata_imm);
            trace = Builder.CreateCall(TTDImm, {ConstantInt::get(I16, markerID)});
        }
        else
        {
            Function* TTD = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata);
            trace = Builder.CreateCall(TTD, {ConstantInt::get(I32, markerID)});
        }
        markMarkerHeaderTrace(trace);

        if (payload)
        {
            if (payload->getType() != I32) payload = Builder.CreateZExtOrTrunc(payload, I32);
            emitRawTracePayload(Builder, payload, M);
        }

        emitMemBarrier(Builder, Config.MemBarrier);
        if (needsBarriers && !skipSchedAfter)
        {
            Function* SchedBarrier = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_sched_barrier);
            Builder.CreateCall(SchedBarrier, {ConstantInt::get(I32, 0)});
        }
    };

    if (!Config.needsScopeCheck())
    {
        emitMarker(B);
        return;
    }

    Value* Ok = getOrCreateScopeCheck(F, gen);

    Instruction* SplitPt = &*B.GetInsertPoint();
    BasicBlock* OrigBB = SplitPt->getParent();
    BasicBlock* TailBB = OrigBB->splitBasicBlock(SplitPt, "sqtt.skip");
    BasicBlock* TraceBB = BasicBlock::Create(Ctx, "sqtt.trace", &F, TailBB);

    OrigBB->getTerminator()->eraseFromParent();
    B.SetInsertPoint(OrigBB);
    B.CreateCondBr(Ok, TraceBB, TailBB);

    B.SetInsertPoint(TraceBB);
    emitMarker(B);
    B.CreateBr(TailBB);

    // Pin both ends of the trace site with sched_barrier(0). Without these,
    // late codegen / sinking happily fills the gap between the cbranch
    // fall-through (the trace marker) and the surrounding code with
    // unrelated instructions, and the recorded trace timing drifts away
    // from whatever the marker is supposed to be measuring.
    //
    //   - TraceBB tail pin: sits just before the unconditional br back to
    //     TailBB, preventing code from being sunk into TraceBB after the
    //     marker. Mostly defensive — TraceBB is conditionally entered so
    //     sinking into it is rare — but cheap to plant.
    //   - TailBB head pin: sits at the very top of TailBB (before any code
    //     that could be sunk in), anchoring the marker to the merge point
    //     rather than letting the sync drift down past sunk-in work.
    //
    // We always plant both — the per-marker scheduling cost is small and
    // marker accuracy matters universally (point markers, user markers,
    // function entry/exit are all timed events).
    Function* SchedBarrier = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_sched_barrier);
    {
        IRBuilder<> TraceTail(TraceBB->getTerminator());
        TraceTail.CreateCall(SchedBarrier, {ConstantInt::get(I32, 0)});
    }
    {
        IRBuilder<> TailHead(&*TailBB->getFirstInsertionPt());
        TailHead.CreateCall(SchedBarrier, {ConstantInt::get(I32, 0)});
    }

    B.SetInsertPoint(&*TailBB->begin());
}

Value* SQTTInstrumentPass::buildScopeCheck(IRBuilder<>& B, GfxGen gen)
{
    Module* M = B.GetInsertBlock()->getParent()->getParent();
    LLVMContext& Ctx = M->getContext();
    Type* I32 = Type::getInt32Ty(Ctx);
    HwRegEncodings hw = getHwRegEncodings(gen);
    Function* SGetReg = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_getreg);

    Value* Ok = ConstantInt::getTrue(Ctx);

    auto addCheck = [&](uint32_t mask, uint32_t fullMask, uint32_t hwReg)
    {
        if ((mask & fullMask) != fullMask)
        {
            Value* ID = B.CreateCall(SGetReg, {ConstantInt::get(I32, hwReg)});
            Value* Bit = B.CreateAnd(B.CreateLShr(ConstantInt::get(I32, mask), ID), ConstantInt::get(I32, 1));
            Ok = B.CreateAnd(Ok, B.CreateICmpNE(Bit, ConstantInt::get(I32, 0)));
        }
    };
    addCheck(Config.WaveMask, FULL_WAVE_MASK, hw.wave);
    addCheck(Config.SimdMask, FULL_SIMD_MASK, hw.simd);
    addCheck(Config.CuMask, FULL_CU_MASK, hw.cu);
    addCheck(Config.WgMask, FULL_WG_MASK, hw.wg);

    return Ok;
}

Value* SQTTInstrumentPass::getOrCreateScopeCheck(Function& F, GfxGen gen)
{
    if (CurScopeCheck) return CurScopeCheck;
    IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
    CurScopeCheck = buildScopeCheck(B, gen);
    return CurScopeCheck;
}

// True for instructions that we are willing to leave inside a scope-check
// trace block alongside grouped markers — i.e. their semantics are unchanged
// (or harmless) when they only execute on the trace path:
//   - other ttracedata calls (these are the markers themselves)
//   - sched_barrier (compile-time scheduling hint, no runtime effect)
//   - debug intrinsics, lifetime markers (no runtime effect)
// Anything else between two markers prevents coalescing — we never want to
// move arbitrary code (loads/stores/arithmetic visible after the merge
// point) into a conditionally executed region.
static bool isIgnorableBetweenMarkers(Instruction* I)
{
    auto* CI = dyn_cast<CallInst>(I);
    if (!CI) return false;
    Function* F = CI->getCalledFunction();
    if (!F) return false;
    switch (F->getIntrinsicID())
    {
        case Intrinsic::amdgcn_s_ttracedata:
        case Intrinsic::amdgcn_s_ttracedata_imm:
        case Intrinsic::amdgcn_sched_barrier:
        case Intrinsic::dbg_declare:
        case Intrinsic::dbg_value:
        case Intrinsic::dbg_label:
        case Intrinsic::lifetime_start:
        case Intrinsic::lifetime_end: return true;
        default: return false;
    }
}

bool SQTTInstrumentPass::wrapExistingMarkers(Function& F, GfxGen gen)
{
    // Collect markers per BB, preserving instruction order. Wrapping each
    // marker independently produces one diamond per marker; after the early
    // phase + inlining flattens many tiny device functions into a hot
    // region, you get a chain of N back-to-back condbrs all on the same
    // cached scope-check predicate. SimplifyCFG would normally fold that,
    // but we run at OptimizerLastEP — nothing simplifies after us.
    //
    // Instead, group runs of markers that are adjacent (only ignorable
    // instructions between them in the same BB) and wrap each run with a
    // single diamond. This is purely a CFG-shape optimization: marker
    // ordering, IDs, and the cached predicate value are unchanged.
    SmallVector<std::pair<BasicBlock*, SmallVector<CallInst*, 4>>, 8> ByBB;
    for (auto& BB : F)
    {
        SmallVector<CallInst*, 4> InBB;
        for (auto& I : BB)
        {
            auto* CI = dyn_cast<CallInst>(&I);
            if (!CI) continue;
            Function* Callee = CI->getCalledFunction();
            if (!Callee) continue;
            auto IID = Callee->getIntrinsicID();
            if (IID == Intrinsic::amdgcn_s_ttracedata || IID == Intrinsic::amdgcn_s_ttracedata_imm) InBB.push_back(CI);
        }
        if (!InBB.empty()) ByBB.push_back({&BB, std::move(InBB)});
    }
    if (ByBB.empty()) return false;

    for (auto& [BB, Markers] : ByBB)
    {
        size_t i = 0;
        while (i < Markers.size())
        {
            // Extend the group as long as Markers[j-1] and Markers[j] are
            // separated only by ignorable instructions. We only check
            // strictly-between instructions — the markers themselves are
            // the things being grouped.
            size_t j = i + 1;
            while (j < Markers.size())
            {
                bool canExtend = true;
                for (Instruction* I = Markers[j - 1]->getNextNode(); I && I != Markers[j]; I = I->getNextNode())
                {
                    if (!isIgnorableBetweenMarkers(I))
                    {
                        canExtend = false;
                        break;
                    }
                }
                if (!canExtend) break;
                ++j;
            }
            wrapRangeWithScopeCheck(Markers[i], Markers[j - 1], F, gen);
            i = j;
        }
    }
    return true;
}

void SQTTInstrumentPass::wrapRangeWithScopeCheck(CallInst* First, CallInst* Last, Function& F, GfxGen gen)
{
    Value* Ok = getOrCreateScopeCheck(F, gen);

    BasicBlock* OrigBB = First->getParent();
    BasicBlock* TraceBB = OrigBB->splitBasicBlock(First->getIterator(), "sqtt.trace");

    // After the first split, [First..Last] and everything after them lives
    // in TraceBB. Split again immediately after Last so TailBB starts with
    // the first non-grouped instruction.
    BasicBlock* TailBB = TraceBB->splitBasicBlock(Last->getNextNode()->getIterator(), "sqtt.skip");

    OrigBB->getTerminator()->eraseFromParent();
    IRBuilder<> BrB(OrigBB);
    BrB.CreateCondBr(Ok, TraceBB, TailBB);
}

bool SQTTInstrumentPass::addBarriersToExistingMarkers(Function& F)
{
    Module* M = F.getParent();
    LLVMContext& Ctx = M->getContext();
    Type* I32 = Type::getInt32Ty(Ctx);

    SmallVector<CallInst*, 8> Markers;
    for (auto& BB : F)
    {
        for (auto& I : BB)
        {
            auto* CI = dyn_cast<CallInst>(&I);
            if (!CI) continue;
            Function* Callee = CI->getCalledFunction();
            if (!Callee) continue;
            auto IID = Callee->getIntrinsicID();
            if (IID == Intrinsic::amdgcn_s_ttracedata || IID == Intrinsic::amdgcn_s_ttracedata_imm)
                Markers.push_back(CI);
        }
    }
    if (Markers.empty()) return false;

    bool needsSchedBarrier = !Config.needsScopeCheck();

    for (auto* CI : Markers)
    {
        // Adjacent hard sync (e.g. __syncthreads) is already a scheduling
        // boundary — a sched_barrier on that side is redundant and prevents
        // useful reorderings. sched_barrier itself is in the set so we also
        // dedup against barriers we just emitted around the previous marker.
        Instruction* PrevI = CI->getPrevNode();
        Instruction* NextI = CI->getNextNode();
        bool skipBefore = isHardSchedBoundary(PrevI);
        bool skipAfter = isHardSchedBoundary(NextI);

        IRBuilder<> Before(CI);
        if (needsSchedBarrier && !skipBefore)
        {
            Function* SB = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_sched_barrier);
            Before.CreateCall(SB, {ConstantInt::get(I32, 0)});
        }
        emitMemBarrier(Before, Config.MemBarrier);

        IRBuilder<> After(CI->getNextNode());
        emitMemBarrier(After, Config.MemBarrier);
        if (needsSchedBarrier && !skipAfter)
        {
            Function* SB = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_sched_barrier);
            After.CreateCall(SB, {ConstantInt::get(I32, 0)});
        }
    }
    return true;
}

CallInst* SQTTInstrumentPass::emitBareTrace(IRBuilder<>& B, uint32_t encoded, Module* M, GfxGen gen)
{
    LLVMContext& Ctx = M->getContext();
    CallInst* CI;
    if (canUseImm(encoded) && supportsImmTrace(gen))
    {
        Function* TTDImm = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata_imm);
        CI = B.CreateCall(TTDImm, {ConstantInt::get(Type::getInt16Ty(Ctx), encoded)});
    }
    else
    {
        Function* TTD = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata);
        CI = B.CreateCall(TTD, {ConstantInt::get(Type::getInt32Ty(Ctx), encoded)});
    }
    markMarkerHeaderTrace(CI);
    return CI;
}

CallInst* SQTTInstrumentPass::emitBareTraceValue(IRBuilder<>& B, Value* val, Module* M, GfxGen gen)
{
    (void) gen;
    Function* TTD = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata);
    CallInst* CI = B.CreateCall(TTD, {val});
    markRawPayloadTrace(CI);
    return CI;
}

CallInst* SQTTInstrumentPass::emitRawTracePayload(IRBuilder<>& B, Value* val, Module* M)
{
    // Full s_ttracedata lowers through M0, so its input must be scalar. The
    // intrinsic lowering normally inserts this readfirstlane for a divergent
    // named data value; retain that behavior before the explicit asm lowering.
    Type* I32 = Type::getInt32Ty(M->getContext());
    if (val->getType() != I32) val = B.CreateZExtOrTrunc(val, I32);
    if (!isa<ConstantInt>(val))
    {
        Function* ReadFirstLane =
            Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_readfirstlane, {I32});
        val = B.CreateCall(ReadFirstLane, {val});
    }
    Function* TTD = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata);
    CallInst* CI = B.CreateCall(TTD, {val});
    markRawPayloadTrace(CI);
    return CI;
}

void SQTTInstrumentPass::markRawPayloadTrace(CallInst* CI)
{
    if (!CI) return;
    CI->setMetadata(RawPayloadMetadataName, MDNode::get(CI->getContext(), {}));
}

bool SQTTInstrumentPass::isRawPayloadTrace(CallInst* CI)
{
    return CI && CI->getMetadata(RawPayloadMetadataName) != nullptr;
}

void SQTTInstrumentPass::markMarkerHeaderTrace(CallInst* CI)
{
    if (!CI) return;
    CI->setMetadata(MarkerHeaderMetadataName, MDNode::get(CI->getContext(), {}));
}

bool SQTTInstrumentPass::isMarkerHeaderTrace(CallInst* CI)
{
    return CI && CI->getMetadata(MarkerHeaderMetadataName) != nullptr;
}

bool SQTTInstrumentPass::lowerFullTracesWithM0Nop(Function& F)
{
    SmallVector<CallInst*, 16> traces;
    for (auto& BB : F)
    {
        for (auto& I : BB)
        {
            auto* CI = dyn_cast<CallInst>(&I);
            if (!CI) continue;
            Function* Callee = CI->getCalledFunction();
            if (Callee && Callee->getIntrinsicID() == Intrinsic::amdgcn_s_ttracedata) traces.push_back(CI);
        }
    }
    if (traces.empty()) return false;

    LLVMContext& Ctx = F.getContext();
    Type* I32 = Type::getInt32Ty(Ctx);
    // Model M0 as a fixed output instead of a clobber. M0 is reserved on
    // AMDGPU and LLVM diagnoses `~{m0}` as undefined behavior.
    FunctionType* TraceTy = FunctionType::get(I32, {I32}, false);
    static constexpr const char TraceAsmText[] =
        "s_mov_b32 m0, $1\n"
        "s_nop 0\n"
        "s_ttracedata";

    // Constants can be encoded directly in the M0 move. Non-constant values
    // reaching this helper are required to be scalar.
    InlineAsm* ImmediateTrace =
        InlineAsm::get(TraceTy, TraceAsmText, "={m0},i", /*hasSideEffects=*/true);
    InlineAsm* ScalarTrace = InlineAsm::get(TraceTy, TraceAsmText, "={m0},s", /*hasSideEffects=*/true);

    for (CallInst* CI : traces)
    {
        Value* value = CI->getArgOperand(0);
        InlineAsm* TraceAsm = isa<ConstantInt>(value) ? ImmediateTrace : ScalarTrace;
        IRBuilder<> B(CI);
        CallInst* replacement = B.CreateCall(TraceAsm, {value});
        replacement->setDebugLoc(CI->getDebugLoc());
        replacement->copyMetadata(*CI);
        CI->eraseFromParent();
    }
    return true;
}

bool SQTTInstrumentPass::applyShaderClockPacking(Function& F, GfxGen gen)
{
    unsigned clockBits = getShaderClockBits(Config, gen);
    if (!usesShaderClockPacking(Config, gen)) return false;

    if (clockBits > 29)
        report_fatal_error("SQTT_SHADER_CLOCK_BITS must leave at least one marker ID bit");
    if (Config.ShaderClockShift >= 32 || Config.ShaderClockShift + clockBits > 32)
        report_fatal_error("SQTT shader clock window must fit in shader_cycles_lo bits [31:0]");

    const unsigned idBits = 30 - clockBits;
    const uint32_t maxID = (uint32_t(1) << idBits) - 1u;
    const unsigned markerAndFlagBits = idBits + 2;
    const uint32_t markerAndFlagMask =
        markerAndFlagBits >= 32 ? 0xFFFFFFFFu : ((uint32_t(1) << markerAndFlagBits) - 1u);

    SmallVector<std::pair<CallInst*, Value*>, 16> Rewrites;
    for (auto& BB : F)
    {
        for (auto& I : BB)
        {
            auto* CI = dyn_cast<CallInst>(&I);
            if (!CI || isRawPayloadTrace(CI)) continue;
            Function* Callee = CI->getCalledFunction();
            if (!Callee) continue;
            auto IID = Callee->getIntrinsicID();
            if (IID != Intrinsic::amdgcn_s_ttracedata && IID != Intrinsic::amdgcn_s_ttracedata_imm) continue;

            Value* encodedValue = CI->getArgOperand(0);
            bool header = isMarkerHeaderTrace(CI);
            if (auto* Arg = dyn_cast<ConstantInt>(encodedValue))
            {
                uint32_t encoded = Arg->getZExtValue();
                // A bare exit has no ID and cannot be distinguished from the
                // numeric API in trace data, so keep all of them packed.
                if (!header && encoded != FLAG_EXIT_PREV) continue;
                uint32_t markerID = encoded >> 2;
                if (markerID > maxID)
                {
                    report_fatal_error(
                        Twine("SQTT marker ID ") + Twine(markerID) + " does not fit with SQTT_SHADER_CLOCK_BITS=" +
                        Twine(clockBits)
                    );
                }
            }
            else if (!header)
                continue;
            Rewrites.push_back({CI, encodedValue});
        }
    }
    if (Rewrites.empty()) return false;

    Module* M = F.getParent();
    LLVMContext& Ctx = M->getContext();
    Type* I32 = Type::getInt32Ty(Ctx);
    Function* SGetReg = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_getreg);
    Function* TTD = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata);
    uint32_t hwreg = GETREG_IMMED(clockBits - 1, Config.ShaderClockShift, GFX12_SHADER_CYCLES_LO);
    uint32_t clockDestShift = 32 - clockBits;

    for (auto& [CI, encoded] : Rewrites)
    {
        IRBuilder<> B(CI);
        Value* markerAndFlags = encoded;
        if (markerAndFlags->getType() != I32) markerAndFlags = B.CreateZExtOrTrunc(markerAndFlags, I32);
        markerAndFlags = B.CreateAnd(markerAndFlags, ConstantInt::get(I32, markerAndFlagMask));
        Value* clock = B.CreateCall(SGetReg, {ConstantInt::get(I32, hwreg)});
        Value* shiftedClock = B.CreateShl(clock, ConstantInt::get(I32, clockDestShift));
        Value* packed = B.CreateOr(shiftedClock, markerAndFlags);
        CallInst* packedTrace = B.CreateCall(TTD, {packed});
        packedTrace->setDebugLoc(CI->getDebugLoc());
        packedTrace->copyMetadata(*CI);
        CI->eraseFromParent();
    }

    ShaderClockBitsUsed = clockBits;
    ShaderClockShiftUsed = Config.ShaderClockShift;
    return true;
}
