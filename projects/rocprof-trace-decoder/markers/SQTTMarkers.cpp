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

static constexpr uint32_t GFX12_SHADER_CYCLES_LO = 29;

// Only sched_barrier and IR fences constrain ttracedata scheduling. AMDGPU
// barriers are IntrNoMem, so they still need an explicit sched_barrier(0).
static bool isHardSchedBoundary(Instruction* I)
{
    if (!I) return false;
    if (isa<FenceInst>(I)) return true;
    auto* CI = dyn_cast<CallInst>(I);
    Function* F = CI ? CI->getCalledFunction() : nullptr;
    return F && F->getIntrinsicID() == Intrinsic::amdgcn_sched_barrier;
}

static bool isCanonicalSchedBarrier(Instruction* I)
{
    auto* CI = dyn_cast_or_null<CallInst>(I);
    Function* F = CI ? CI->getCalledFunction() : nullptr;
    auto* Arg = CI && CI->arg_size() == 1 ? dyn_cast<ConstantInt>(CI->getArgOperand(0)) : nullptr;
    return F && F->getIntrinsicID() == Intrinsic::amdgcn_sched_barrier && Arg && Arg->isZero();
}

static bool isMarkerSentinel(Instruction* I)
{
    auto* CI = dyn_cast_or_null<CallInst>(I);
    Function* F = CI ? CI->getCalledFunction() : nullptr;
    return F && F->getName().starts_with("__sqtt_named_marker_");
}

static Instruction* nextRealInstruction(Instruction* I)
{
    while (I && (I->isDebugOrPseudoInst() || I->isLifetimeStartOrEnd() || isMarkerSentinel(I)))
        I = I->getNextNode();
    return I;
}

static bool isPayloadSequence(CallInst* header, CallInst* payload, MDNode* group)
{
    for (Instruction* I = header->getNextNode(); I != payload; I = I ? I->getNextNode() : nullptr)
        if (!I || (I->getMetadata(SQTT_PAYLOAD_GROUP_METADATA) != group && !I->isDebugOrPseudoInst() &&
                   !I->isLifetimeStartOrEnd()))
            return false;
    return true;
}

static void branchToScopedTrace(
    BasicBlock* source, BasicBlock* trace, BasicBlock* skip, Value* ok, bool pinSkipHead
)
{
    source->getTerminator()->eraseFromParent();
    IRBuilder<>(source).CreateCondBr(ok, trace, skip);
    if (!pinSkipHead) return;

    Module* M = source->getParent()->getParent();
    Type* I32 = Type::getInt32Ty(M->getContext());
    Function* SchedBarrier = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_sched_barrier);
    IRBuilder<>(&*skip->getFirstInsertionPt()).CreateCall(SchedBarrier, {ConstantInt::get(I32, 0)});
}

bool SQTTInstrumentPass::isSyncInstruction(Instruction* I)
{
    return isa<FenceInst>(I) || classifyBarrier(dyn_cast<CallInst>(I)) != BarrierKind::None;
}

// Emit the configured compiler ordering barrier; the fence is limited to LDS.
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

    // Keep compiler ordering without global cache invalidation.
    Metadata* LocalSyncAS[] = {MDString::get(Ctx, "amdgpu-synchronize-as"), MDString::get(Ctx, "local")};
    F->setMetadata(LLVMContext::MD_mmra, MDNode::get(Ctx, LocalSyncAS));
}

void SQTTInstrumentPass::emitTraceBoundary(IRBuilder<>& B, bool after, bool schedBarrier)
{
    Module* M = B.GetInsertBlock()->getParent()->getParent();
    Type* I32 = Type::getInt32Ty(B.getContext());
    Function* SchedBarrier =
        schedBarrier ? Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_sched_barrier) : nullptr;
    if (!after && SchedBarrier) B.CreateCall(SchedBarrier, {ConstantInt::get(I32, 0)});
    emitMemBarrier(B, Config.MemBarrier);
    if (after && SchedBarrier) B.CreateCall(SchedBarrier, {ConstantInt::get(I32, 0)});
}

void SQTTInstrumentPass::emitTraceBoundaries(
    IRBuilder<>& B, Instruction* first, Instruction* last, bool schedBarrier
)
{
    Instruction* next = last->getNextNode();
    IRBuilder<> Before(first);
    emitTraceBoundary(Before, /*after=*/false, schedBarrier && !isHardSchedBoundary(first->getPrevNode()));
    // Newly-built scoped traces end after `last`; existing traces have a tail.
    if (next) B.SetInsertPoint(next);
    emitTraceBoundary(B, /*after=*/true, schedBarrier && !isHardSchedBoundary(next));
}

void SQTTInstrumentPass::emitScopedTrace(
    IRBuilder<>& B,
    Function& F,
    GfxGen gen,
    const char* traceBlockName,
    const char* skipBlockName,
    bool pinSkipHead,
    function_ref<void(IRBuilder<>&)> emit
)
{
    if (!Config.needsScopeCheck())
    {
        emit(B);
        return;
    }

    Value* Ok = getOrCreateScopeCheck(F, gen);
    Instruction* SplitPt = &*B.GetInsertPoint();
    BasicBlock* OrigBB = SplitPt->getParent();
    BasicBlock* TailBB = OrigBB->splitBasicBlock(SplitPt, skipBlockName);
    BasicBlock* TraceBB = BasicBlock::Create(F.getContext(), traceBlockName, &F, TailBB);
    branchToScopedTrace(OrigBB, TraceBB, TailBB, Ok, pinSkipHead);

    IRBuilder<> Trace(TraceBB);
    emit(Trace);
    Trace.CreateBr(TailBB);

    B.SetInsertPoint(&*TailBB->begin());
}

void SQTTInstrumentPass::insertTraceMarker(
    IRBuilder<>& B, uint32_t markerID, Function& F, GfxGen gen, Value* payload
)
{
    Module* M = F.getParent();
    bool needsBarriers = !Config.needsScopeCheck();
    bool pinSkipHead = Config.needsScopeCheck() && isSyncInstruction(nextRealInstruction(&*B.GetInsertPoint()));

    auto emitMarker = [&](IRBuilder<>& Builder)
    {
        CallInst* first = emitBareTrace(Builder, markerID, M, gen);
        Instruction* last = payload ? emitRawTracePayload(Builder, payload, M, first) : first;
        emitTraceBoundaries(Builder, first, last, needsBarriers);
    };

    emitScopedTrace(B, F, gen, "sqtt.trace", "sqtt.skip", pinSkipHead, emitMarker);
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

// Only marker/hint/debug/lifetime instructions may move into a conditional trace block.
static bool isIgnorableBetweenMarkers(Instruction* I)
{
    auto* CI = dyn_cast<CallInst>(I);
    Function* F = CI ? CI->getCalledFunction() : nullptr;
    if (!F) return false;
    auto ID = F->getIntrinsicID();
    return ID == Intrinsic::amdgcn_s_ttracedata || ID == Intrinsic::amdgcn_s_ttracedata_imm ||
           isCanonicalSchedBarrier(I) || ID == Intrinsic::dbg_declare || ID == Intrinsic::dbg_value ||
           ID == Intrinsic::dbg_label || ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end;
}

bool SQTTInstrumentPass::finalizeExistingMarkers(Function& F, GfxGen gen)
{
    struct TraceRange
    {
        CallInst *First, *Last;
    };
    SmallVector<SmallVector<TraceRange, 4>, 8> ByBlock;
    for (auto& BB : F)
    {
        SmallVector<TraceRange, 4> InBB;
        for (auto& I : BB)
        {
            auto* CI = dyn_cast<CallInst>(&I);
            if (!isTraceDataCall(CI)) continue;
            MDNode* group = CI->getMetadata(SQTT_PAYLOAD_GROUP_METADATA);
            if (CI->getMetadata(SQTT_RAW_PAYLOAD_METADATA) && group && !InBB.empty() &&
                InBB.back().First == InBB.back().Last &&
                InBB.back().First->getMetadata(SQTT_MARKER_HEADER_METADATA) &&
                InBB.back().First->getMetadata(SQTT_PAYLOAD_GROUP_METADATA) == group &&
                isPayloadSequence(InBB.back().First, CI, group))
                InBB.back().Last = CI;
            else
                InBB.push_back({CI, CI});
        }
        if (!InBB.empty()) ByBlock.push_back(std::move(InBB));
    }
    if (ByBlock.empty()) return false;

    const bool scopeCheck = Config.needsScopeCheck();
    auto addBoundaries = [&](const TraceRange& range)
    {
        IRBuilder<> B(range.First);
        emitTraceBoundaries(B, range.First, range.Last, !scopeCheck);
    };

    for (auto& Ranges : ByBlock)
    {
        if (!scopeCheck)
        {
            for (const TraceRange& range : Ranges) addBoundaries(range);
            continue;
        }
        for (const TraceRange& range : Ranges)
        {
            while (isCanonicalSchedBarrier(range.First->getPrevNode()))
                range.First->getPrevNode()->eraseFromParent();
            while (isCanonicalSchedBarrier(range.Last->getNextNode()))
                range.Last->getNextNode()->eraseFromParent();
        }

        // Coalesce adjacent marker runs: OptimizerLastEP has no later
        // SimplifyCFG. Wrap each run before adding its boundaries.
        for (size_t first = 0; first < Ranges.size();)
        {
            size_t last = first;
            while (last + 1 < Ranges.size())
            {
                bool canExtend = true;
                for (Instruction* I = Ranges[last].Last->getNextNode(); I && I != Ranges[last + 1].First;
                     I = I->getNextNode())
                {
                    if (!isIgnorableBetweenMarkers(I))
                    {
                        canExtend = false;
                        break;
                    }
                }
                if (!canExtend) break;
                ++last;
            }
            bool pinSkipHead = isSyncInstruction(nextRealInstruction(Ranges[last].Last->getNextNode()));
            wrapRangeWithScopeCheck(Ranges[first].First, Ranges[last].Last, F, gen, pinSkipHead);
            for (size_t i = first; i <= last; ++i) addBoundaries(Ranges[i]);
            first = last + 1;
        }
    }
    return true;
}

void SQTTInstrumentPass::wrapRangeWithScopeCheck(
    CallInst* First, CallInst* Last, Function& F, GfxGen gen, bool pinSkipHead
)
{
    Value* Ok = getOrCreateScopeCheck(F, gen);
    BasicBlock* OrigBB = First->getParent();
    BasicBlock* TraceBB = OrigBB->splitBasicBlock(First->getIterator(), "sqtt.trace");

    // Split after the run so the tail begins with the first non-marker.
    BasicBlock* TailBB = TraceBB->splitBasicBlock(Last->getNextNode()->getIterator(), "sqtt.skip");
    branchToScopedTrace(OrigBB, TraceBB, TailBB, Ok, pinSkipHead);
}

CallInst* SQTTInstrumentPass::emitBareTrace(IRBuilder<>& B, uint32_t encoded, Module* M, GfxGen gen)
{
    LLVMContext& Ctx = M->getContext();
    bool useImm = canUseImm(encoded) && supportsImmTrace(gen);
    Function* TTD = Intrinsic::getOrInsertDeclaration(
        M, useImm ? Intrinsic::amdgcn_s_ttracedata_imm : Intrinsic::amdgcn_s_ttracedata
    );
    CallInst* CI = B.CreateCall(TTD, {ConstantInt::get(useImm ? Type::getInt16Ty(Ctx) : Type::getInt32Ty(Ctx), encoded)});
    CI->setMetadata(SQTT_MARKER_HEADER_METADATA, MDNode::get(Ctx, {}));
    return CI;
}

CallInst* SQTTInstrumentPass::emitBareTraceValue(IRBuilder<>& B, Value* val, Module* M)
{
    Function* TTD = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata);
    CallInst* trace = B.CreateCall(TTD, {val});
    trace->setMetadata(SQTT_RAW_PAYLOAD_METADATA, MDNode::get(B.getContext(), {}));
    return trace;
}

bool SQTTInstrumentPass::isTraceDataCall(const CallInst* CI)
{
    if (!CI) return false;
    const Function* Callee = CI->getCalledFunction();
    if (!Callee) return false;
    auto ID = Callee->getIntrinsicID();
    return ID == Intrinsic::amdgcn_s_ttracedata || ID == Intrinsic::amdgcn_s_ttracedata_imm;
}

CallInst* SQTTInstrumentPass::emitRawTracePayload(IRBuilder<>& B, Value* val, Module* M, CallInst* header)
{
    // Full s_ttracedata lowers through M0, so its input must be scalar. The
    // intrinsic lowering normally inserts this readfirstlane for a divergent
    // named data value; retain that behavior before the explicit asm lowering.
    Type* I32 = Type::getInt32Ty(M->getContext());
    MDNode* group = MDNode::getDistinct(B.getContext(), {});
    header->setMetadata(SQTT_PAYLOAD_GROUP_METADATA, group);
    if (val->getType() != I32)
    {
        val = B.CreateZExtOrTrunc(val, I32);
        if (auto* I = dyn_cast<Instruction>(val)) I->setMetadata(SQTT_PAYLOAD_GROUP_METADATA, group);
    }
    if (!isa<ConstantInt>(val))
    {
        Function* ReadFirstLane =
            Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_readfirstlane, {I32});
        CallInst* prep = B.CreateCall(ReadFirstLane, {val});
        prep->setMetadata(SQTT_PAYLOAD_GROUP_METADATA, group);
        val = prep;
    }
    CallInst* trace = emitBareTraceValue(B, val, M);
    trace->setMetadata(SQTT_PAYLOAD_GROUP_METADATA, group);
    return trace;
}

bool SQTTInstrumentPass::finalizeFullTraces(Function& F, GfxGen gen)
{
    unsigned clockBits = Config.ShaderClockBits;
    const bool packClock = gen == GfxGen::GFX12 && clockBits != 0;
    if (packClock && clockBits > 29)
        report_fatal_error("SQTT_SHADER_CLOCK_BITS must leave at least one marker ID bit");
    if (packClock && (Config.ShaderClockShift >= 32 || Config.ShaderClockShift + clockBits > 32))
        report_fatal_error("SQTT shader clock window must fit in shader_cycles_lo bits [31:0]");

    const unsigned idBits = packClock ? 30 - clockBits : 0;
    const uint32_t maxID = packClock ? (uint32_t(1) << idBits) - 1u : 0;
    const uint32_t markerAndFlagMask = packClock ? (uint32_t(1) << (idBits + 2)) - 1u : 0;

    Module* M = F.getParent();
    LLVMContext& Ctx = M->getContext();
    Type* I32 = Type::getInt32Ty(Ctx);
    const bool useShaderCyclesU64 = packClock && hasShaderCyclesU64(F);
    Function* SGetReg = nullptr;
    InlineAsm* GetShaderCycles = nullptr;
    uint32_t hwreg = packClock ? GETREG_IMMED(clockBits - 1, Config.ShaderClockShift, GFX12_SHADER_CYCLES_LO) : 0;
    uint32_t clockDestShift = packClock ? 32 - clockBits : 0;

    // Model M0 as a fixed output instead of a clobber. M0 is reserved on
    // AMDGPU and LLVM diagnoses `~{m0}` as undefined behavior.
    FunctionType* TraceTy = FunctionType::get(I32, {I32}, false);
    static constexpr const char TraceAsmText[] =
        "s_mov_b32 m0, $1\n"
        "s_nop 0\n"
        "s_ttracedata";
    InlineAsm* ImmediateTrace =
        InlineAsm::get(TraceTy, TraceAsmText, "={m0},i", /*hasSideEffects=*/true);
    InlineAsm* ScalarTrace = InlineAsm::get(TraceTy, TraceAsmText, "={m0},s", /*hasSideEffects=*/true);

    bool changed = false;
    for (auto& BB : F)
    {
        for (auto It = BB.begin(), End = BB.end(); It != End;)
        {
            auto* CI = dyn_cast<CallInst>(&*It++);
            if (!isTraceDataCall(CI)) continue;

            bool packThisTrace = false;
            if (packClock && !CI->getMetadata(SQTT_RAW_PAYLOAD_METADATA))
            {
                Value* encoded = CI->getArgOperand(0);
                if (auto* Arg = dyn_cast<ConstantInt>(encoded))
                {
                    // A bare exit has no ID and cannot be distinguished from
                    // the numeric API in trace data, so keep all of them
                    // packed.
                    if (CI->getMetadata(SQTT_MARKER_HEADER_METADATA) || Arg->getZExtValue() == FLAG_EXIT_PREV)
                    {
                        uint32_t markerID = Arg->getZExtValue() >> 2;
                        if (markerID > maxID)
                            report_fatal_error(
                                Twine("SQTT marker ID ") + Twine(markerID) +
                                " does not fit with SQTT_SHADER_CLOCK_BITS=" + Twine(clockBits)
                            );
                        packThisTrace = true;
                    }
                }
                else
                    packThisTrace = CI->getMetadata(SQTT_MARKER_HEADER_METADATA);
            }

            if (!packThisTrace && CI->getCalledFunction()->getIntrinsicID() != Intrinsic::amdgcn_s_ttracedata)
                continue;

            IRBuilder<> B(CI);
            Value* value = CI->getArgOperand(0);
            if (packThisTrace)
            {
                if (value->getType() != I32) value = B.CreateZExtOrTrunc(value, I32);
                Value* markerAndFlags = B.CreateAnd(value, ConstantInt::get(I32, markerAndFlagMask));
                Value* clock;
                if (useShaderCyclesU64)
                {
                    if (!GetShaderCycles)
                        GetShaderCycles = InlineAsm::get(
                            FunctionType::get(Type::getInt64Ty(Ctx), false),
                            "s_get_shader_cycles_u64 $0",
                            "=s",
                            /*hasSideEffects=*/true
                        );
                    Value* cyclesLo = B.CreateTrunc(B.CreateCall(GetShaderCycles), I32);
                    // The final shift below discards all but clockBits,
                    // matching the right-aligned field returned by s_getreg
                    // on gfx1200 and gfx1201.
                    clock = B.CreateLShr(cyclesLo, ConstantInt::get(I32, Config.ShaderClockShift));
                }
                else
                {
                    if (!SGetReg) SGetReg = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_getreg);
                    clock = B.CreateCall(SGetReg, {ConstantInt::get(I32, hwreg)});
                }
                value = B.CreateOr(B.CreateShl(clock, ConstantInt::get(I32, clockDestShift)), markerAndFlags);
                ShaderClockBitsUsed = clockBits;
            }

            InlineAsm* TraceAsm = isa<ConstantInt>(value) ? ImmediateTrace : ScalarTrace;
            CallInst* replacement = B.CreateCall(TraceAsm, {value});
            replacement->setDebugLoc(CI->getDebugLoc());
            replacement->copyMetadata(*CI);
            CI->eraseFromParent();
            changed = true;
        }
    }

    return changed;
}
