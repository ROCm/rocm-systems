#include "SQTTPass.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/AtomicOrdering.h"

using namespace llvm;

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
//   Fence      — fence syncscope("workgroup") acq_rel; lowers to
//                s_waitcnt lgkmcnt(0) on AMDGPU. Free in non-LDS regions.
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
    B.CreateFence(AtomicOrdering::AcquireRelease, WG);
}

void SQTTInstrumentPass::insertTraceMarker(IRBuilder<>& B, uint32_t markerID, Function& F, GfxGen gen)
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

        if (canUseImm(markerID) && supportsImmTrace(gen))
        {
            Function* TTDImm = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata_imm);
            Builder.CreateCall(TTDImm, {ConstantInt::get(I16, markerID)});
        }
        else
        {
            Function* TTD = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata);
            Builder.CreateCall(TTD, {ConstantInt::get(I32, markerID)});
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

bool SQTTInstrumentPass::wrapExistingMarkers(Function& F, GfxGen gen)
{
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

    for (auto* CI : Markers) wrapWithScopeCheck(CI, F, gen);
    return true;
}

void SQTTInstrumentPass::wrapWithScopeCheck(CallInst* CI, Function& F, GfxGen gen)
{
    Value* Ok = getOrCreateScopeCheck(F, gen);

    BasicBlock* OrigBB = CI->getParent();
    BasicBlock* TraceBB = OrigBB->splitBasicBlock(CI->getIterator(), "sqtt.trace");

    BasicBlock* TailBB = TraceBB->splitBasicBlock(CI->getNextNode()->getIterator(), "sqtt.skip");

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

void SQTTInstrumentPass::emitBareTrace(IRBuilder<>& B, uint32_t encoded, Module* M, GfxGen gen)
{
    LLVMContext& Ctx = M->getContext();
    if (canUseImm(encoded) && supportsImmTrace(gen))
    {
        Function* TTDImm = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata_imm);
        B.CreateCall(TTDImm, {ConstantInt::get(Type::getInt16Ty(Ctx), encoded)});
    }
    else
    {
        Function* TTD = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata);
        B.CreateCall(TTD, {ConstantInt::get(Type::getInt32Ty(Ctx), encoded)});
    }
}

void SQTTInstrumentPass::emitBareTraceValue(IRBuilder<>& B, Value* val, Module* M, GfxGen gen)
{
    Function* TTD = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata);
    B.CreateCall(TTD, {val});
}
