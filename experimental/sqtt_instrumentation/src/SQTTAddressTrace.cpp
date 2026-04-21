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
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/InlineAsm.h"

using namespace llvm;

SQTTInstrumentPass::AddrTraceKind SQTTInstrumentPass::classifyAddrTraceOp(
    Instruction* I, bool traceMemory, bool traceLDS
)
{
    // Helper lambda for pointer-based address space classification
    auto classifyAS = [&](unsigned AS) -> AddrTraceKind
    {
        if (AS == 5) return AddrTraceKind::None; // private
        if (AS == 3) return traceLDS ? AddrTraceKind::LDS : AddrTraceKind::None;
        return traceMemory ? AddrTraceKind::Memory : AddrTraceKind::None;
    };

    if (auto* LI = dyn_cast<LoadInst>(I)) return classifyAS(LI->getPointerAddressSpace());
    if (auto* SI = dyn_cast<StoreInst>(I)) return classifyAS(SI->getPointerAddressSpace());
    if (auto* AI = dyn_cast<AtomicRMWInst>(I)) return classifyAS(AI->getPointerAddressSpace());
    if (auto* AX = dyn_cast<AtomicCmpXchgInst>(I)) return classifyAS(AX->getPointerAddressSpace());

    if (auto* CI = dyn_cast<CallInst>(I))
    {
        Function* Callee = CI->getCalledFunction();
        if (!Callee) return AddrTraceKind::None;
        StringRef Name = Callee->getName();
        if (traceMemory && isBufferOp(Name)) return AddrTraceKind::Buffer;
        auto IID = Callee->getIntrinsicID();
        if (traceLDS && (IID == Intrinsic::amdgcn_ds_permute || IID == Intrinsic::amdgcn_ds_bpermute ||
                         IID == Intrinsic::amdgcn_ds_bpermute_fi_b32))
            return AddrTraceKind::Permute;
    }
    return AddrTraceKind::None;
}

Value* SQTTInstrumentPass::getMemOpPointer(Instruction* I)
{
    if (auto* LI = dyn_cast<LoadInst>(I)) return LI->getPointerOperand();
    if (auto* SI = dyn_cast<StoreInst>(I)) return SI->getPointerOperand();
    if (auto* AI = dyn_cast<AtomicRMWInst>(I)) return AI->getPointerOperand();
    if (auto* AX = dyn_cast<AtomicCmpXchgInst>(I)) return AX->getPointerOperand();
    return nullptr;
}

std::string SQTTInstrumentPass::getSourceLoc(Instruction* I)
{
    const DebugLoc& DL = I->getDebugLoc();
    if (!DL) return "";

    // Walk the inline chain innermost -> outermost.  At each level, getScope()
    // gives the file the source line lives in; getLine() gives the line.
    // getInlinedAt() walks one step outward (the call site).  Format matches
    // rocprofiler-sdk codeobj's printer: "<inner>:<line> -> <outer>:<line>".
    std::string out;
    DILocation* L = DL.get();
    while (L)
    {
        if (!out.empty()) out += " -> ";
        if (auto* Scope = L->getScope()) out += Scope->getFilename().str();
        out += ':';
        out += std::to_string(L->getLine());
        L = L->getInlinedAt();
    }
    return out;
}

std::string SQTTInstrumentPass::getFunctionSourceLoc(Function& F)
{
    DISubprogram* SP = F.getSubprogram();
    if (!SP) return "";
    StringRef File = SP->getFilename();
    unsigned Line = SP->getLine();
    if (File.empty() && Line == 0) return "";
    std::string out = File.str();
    out += ':';
    out += std::to_string(Line);
    return out;
}

const char* SQTTInstrumentPass::addrTraceKindName(AddrTraceKind kind, bool isStore, bool isAtomic)
{
    switch (kind)
    {
        case AddrTraceKind::LDS:
            if (isAtomic) return "addr_trace_lds_atomic";
            return isStore ? "addr_trace_lds_store" : "addr_trace_lds_load";
        case AddrTraceKind::Memory:
            if (isAtomic) return "addr_trace_atomic";
            return isStore ? "addr_trace_store" : "addr_trace_load";
        case AddrTraceKind::Buffer:
            // Buffer names encode struct vs raw via the caller; here we give the
            // base name which gets refined in instrumentAddressTraces.
            if (isAtomic) return "addr_trace_buffer_atomic";
            return isStore ? "addr_trace_buffer_store" : "addr_trace_buffer_load";
        case AddrTraceKind::Permute: return "addr_trace_ds_permute"; // bpermute distinguished by caller
        default: return "addr_trace_unknown";
    }
}

void SQTTInstrumentPass::emitAddressTrace(
    IRBuilder<>& B, Instruction* memOp, AddrTraceKind kind, uint32_t headerID, Function& F, GfxGen gen
)
{
    Module* M = F.getParent();
    LLVMContext& Ctx = M->getContext();
    Type* I32 = Type::getInt32Ty(Ctx);
    Type* I64 = Type::getInt64Ty(Ctx);

    // Buffer and permute ops use specialized trace protocols
    if (kind == AddrTraceKind::Buffer)
    {
        emitBufferTrace(B, cast<CallInst>(memOp), headerID, F, gen);
        return;
    }
    if (kind == AddrTraceKind::Permute)
    {
        emitPermuteTrace(B, cast<CallInst>(memOp), headerID, F, gen);
        return;
    }

    // 1. Header marker
    emitBareTrace(B, encodeMarker(headerID, false, false), M, gen);

    // 2. Get pointer and convert to integer
    Value* ptr = getMemOpPointer(memOp);
    assert(ptr && "expected Load/Store/Atomic instruction");

    bool is64bit = (kind == AddrTraceKind::Memory);
    Value *addrLo, *addrHi = nullptr;
    if (is64bit)
    {
        Value* addrI64 = B.CreatePtrToInt(ptr, I64);
        addrLo = B.CreateTrunc(addrI64, I32);
        addrHi = B.CreateTrunc(B.CreateLShr(addrI64, 32), I32);
    }
    else
    {
        // LDS: 32-bit address
        addrLo = B.CreatePtrToInt(ptr, I32);
    }

    // 3. EXEC mask
    unsigned waveSize = getWaveSize(gen);

    // gfx9: the ISA spec requires s_nop 0 between s_mov_b32 m0 and
    // s_ttracedata.  We can't insert a nop between the backend-generated
    // s_mov_b32 m0 and s_ttracedata pair, so for exec reads we emit the
    // entire sequence as a single inline asm block.
    if (gen == GfxGen::GFX9)
    {
        InlineAsm* traceExecLo = InlineAsm::get(
            FunctionType::get(Type::getVoidTy(Ctx), false),
            "s_mov_b32 m0, exec_lo\n"
            "s_nop 0\n"
            "s_ttracedata",
            "",
            /*hasSideEffects=*/true
        );
        B.CreateCall(traceExecLo);

        InlineAsm* traceExecHi = InlineAsm::get(
            FunctionType::get(Type::getVoidTy(Ctx), false),
            "s_mov_b32 m0, exec_hi\n"
            "s_nop 0\n"
            "s_ttracedata",
            "",
            /*hasSideEffects=*/true
        );
        B.CreateCall(traceExecHi);
    }
    else
    {
        InlineAsm* readExecLo =
            InlineAsm::get(FunctionType::get(I32, false), "s_mov_b32 $0, exec_lo", "=s", /*hasSideEffects=*/true);
        emitBareTraceValue(B, B.CreateCall(readExecLo), M, gen);

        InlineAsm* readExecHi =
            InlineAsm::get(FunctionType::get(I32, false), "s_mov_b32 $0, exec_hi", "=s", /*hasSideEffects=*/true);
        emitBareTraceValue(B, B.CreateCall(readExecHi), M, gen);
    }

    // 4. Readlane loop
    Function* ReadLane = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_readlane, {I32});

    BasicBlock* PreheaderBB = B.GetInsertBlock();
    BasicBlock* AfterBB;

    bool atEnd = (B.GetInsertPoint() == PreheaderBB->end());
    if (!atEnd)
    {
        Instruction* SplitPt = &*B.GetInsertPoint();
        AfterBB = PreheaderBB->splitBasicBlock(SplitPt, "sqtt.addr.after");
        PreheaderBB->getTerminator()->eraseFromParent();
    }
    else { AfterBB = BasicBlock::Create(Ctx, "sqtt.addr.after", &F, PreheaderBB->getNextNode()); }

    BasicBlock* LoopBB = BasicBlock::Create(Ctx, "sqtt.addr.loop", &F, AfterBB);

    // Preheader -> LoopBB
    IRBuilder<> PreB(PreheaderBB);
    PreB.CreateBr(LoopBB);

    // Loop body
    IRBuilder<> LoopB(LoopBB);
    PHINode* Lane = LoopB.CreatePHI(I32, 2, "lane");
    Lane->addIncoming(ConstantInt::get(I32, 0), PreheaderBB);

    Value* laneLo = LoopB.CreateCall(ReadLane, {addrLo, Lane});
    emitBareTraceValue(LoopB, laneLo, M, gen);

    if (is64bit)
    {
        Value* laneHi = LoopB.CreateCall(ReadLane, {addrHi, Lane});
        emitBareTraceValue(LoopB, laneHi, M, gen);
    }

    Value* LaneNext = LoopB.CreateAdd(Lane, ConstantInt::get(I32, 1), "lane.next");
    Lane->addIncoming(LaneNext, LoopBB);
    Value* Done = LoopB.CreateICmpEQ(LaneNext, ConstantInt::get(I32, waveSize));
    BranchInst* LoopBr = LoopB.CreateCondBr(Done, AfterBB, LoopBB);

    // Attach loop metadata to disable unrolling
    MDNode* LoopID = MDNode::getDistinct(
        Ctx,
        {nullptr, // placeholder for self-reference
         MDNode::get(Ctx, {MDString::get(Ctx, "llvm.loop.unroll.disable")})}
    );
    LoopID->replaceOperandWith(0, LoopID);
    LoopBr->setMetadata(LLVMContext::MD_loop, LoopID);

    // Restore insertion point to AfterBB
    B.SetInsertPoint(AfterBB, AfterBB->begin());
}

bool SQTTInstrumentPass::instrumentAddressTraces(Function& F, GfxGen gen)
{
    struct AddrOp
    {
        Instruction* I;
        AddrTraceKind Kind;
        bool IsStore;
        bool IsAtomic;
    };
    SmallVector<AddrOp, 16> Ops;
    for (auto& BB : F)
    {
        for (auto& I : BB)
        {
            AddrTraceKind kind = classifyAddrTraceOp(&I, Config.TraceMemoryAddrs, Config.TraceLDSAddrs);
            if (kind == AddrTraceKind::None) continue;
            bool isStore = isa<StoreInst>(&I);
            bool isAtomic = isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I);
            if (auto* CI = dyn_cast<CallInst>(&I))
            {
                Function* Callee = CI->getCalledFunction();
                if (Callee)
                {
                    StringRef Name = Callee->getName();
                    if (isBufferStore(Name)) isStore = true;
                    if (isBufferAtomic(Name)) isAtomic = true;
                }
            }
            Ops.push_back({&I, kind, isStore, isAtomic});
        }
    }
    if (Ops.empty()) return false;

    // Wave size is recorded once per module for the .sqtt_funcmap header.
    // RDNA is wave-32, CDNA is wave-64; a single AMDGPU code object normally
    // targets one or the other. If we ever see a mix, default to wave-64 —
    // the decoder treats exec_hi=0 padding as "no upper-half lanes" so the
    // wider format stays correct for both.
    unsigned waveSize = getWaveSize(gen);
    if (AddrTraceWaveSize == 0)
        AddrTraceWaveSize = waveSize;
    else if (AddrTraceWaveSize != waveSize)
        AddrTraceWaveSize = 64;

    for (auto& op : Ops)
    {
        uint32_t opID = NextEventID++;

        // Determine funcmap name — for buffers, refine with struct prefix
        std::string kindName;
        if (op.Kind == AddrTraceKind::Buffer)
        {
            auto* CI = cast<CallInst>(op.I);
            StringRef Name = CI->getCalledFunction()->getName();
            bool isStruct = isStructBuffer(Name);
            const char* base = addrTraceKindName(op.Kind, op.IsStore, op.IsAtomic);
            kindName =
                isStruct ? std::string("addr_trace_struct_") + (base + strlen("addr_trace_")) : std::string(base);
        }
        else if (op.Kind == AddrTraceKind::Permute)
        {
            auto* CI = cast<CallInst>(op.I);
            auto IID = CI->getCalledFunction()->getIntrinsicID();
            kindName = (IID == Intrinsic::amdgcn_ds_bpermute || IID == Intrinsic::amdgcn_ds_bpermute_fi_b32)
                         ? "addr_trace_ds_bpermute"
                         : "addr_trace_ds_permute";
        }
        else { kindName = addrTraceKindName(op.Kind, op.IsStore, op.IsAtomic); }
        std::string srcLoc = getSourceLoc(op.I);
        AddrTraceEntries.push_back({opID, kindName, srcLoc});

        IRBuilder<> B(op.I);
        if (Config.needsScopeCheck())
        {
            Value* Ok = getOrCreateScopeCheck(F, gen);
            Instruction* SplitPt = &*B.GetInsertPoint();
            BasicBlock* OrigBB = SplitPt->getParent();
            BasicBlock* TailBB = OrigBB->splitBasicBlock(SplitPt, "sqtt.addr.skip");
            BasicBlock* TraceBB = BasicBlock::Create(B.getContext(), "sqtt.addr.trace", &F, TailBB);

            OrigBB->getTerminator()->eraseFromParent();
            IRBuilder<> BrB(OrigBB);
            BrB.CreateCondBr(Ok, TraceBB, TailBB);

            IRBuilder<> TB(TraceBB);
            emitAddressTrace(TB, op.I, op.Kind, opID, F, gen);
            TB.CreateBr(TailBB);
        }
        else { emitAddressTrace(B, op.I, op.Kind, opID, F, gen); }
    }
    return true;
}

// ============================================================================
// Buffer intrinsic operand extraction
// ============================================================================

SQTTInstrumentPass::BufferOperands SQTTInstrumentPass::getBufferOperands(CallInst* CI)
{
    Function* Callee = CI->getCalledFunction();
    StringRef Name = Callee->getName();
    bool isStruct = isStructBuffer(Name);
    bool isCmpSwap = isBufferCmpSwap(Name);
    bool isLoad = isBufferLoad(Name);

    // Operand layout varies by intrinsic variant:
    //   raw.buffer.load:              rsrc=0, voffset=1, soffset=2
    //   raw.buffer.store:             rsrc=1, voffset=2, soffset=3
    //   raw.buffer.atomic.*:          rsrc=1, voffset=2, soffset=3
    //   raw.buffer.atomic.cmpswap:    rsrc=2, voffset=3, soffset=4
    //   struct.buffer.load:           rsrc=0, vindex=1, voffset=2, soffset=3
    //   struct.buffer.store:          rsrc=1, vindex=2, voffset=3, soffset=4
    //   struct.buffer.atomic.*:       rsrc=1, vindex=2, voffset=3, soffset=4
    //   struct.buffer.atomic.cmpswap: rsrc=2, vindex=3, voffset=4, soffset=5

    unsigned rsrcIdx;
    if (isLoad)
        rsrcIdx = 0;
    else if (isCmpSwap)
        rsrcIdx = 2; // src, cmp, rsrc, ...
    else
        rsrcIdx = 1; // vdata, rsrc, ...

    BufferOperands ops;
    ops.IsStruct = isStruct;
    ops.Rsrc = CI->getArgOperand(rsrcIdx);

    if (isStruct)
    {
        ops.VIndex = CI->getArgOperand(rsrcIdx + 1);
        ops.VOffset = CI->getArgOperand(rsrcIdx + 2);
        ops.SOffset = CI->getArgOperand(rsrcIdx + 3);
    }
    else
    {
        ops.VIndex = nullptr;
        ops.VOffset = CI->getArgOperand(rsrcIdx + 1);
        ops.SOffset = CI->getArgOperand(rsrcIdx + 2);
    }
    return ops;
}

// ============================================================================
// Buffer trace emission (component-based protocol)
// ============================================================================

void SQTTInstrumentPass::emitBufferTrace(IRBuilder<>& B, CallInst* bufOp, uint32_t headerID, Function& F, GfxGen gen)
{
    Module* M = F.getParent();
    LLVMContext& Ctx = M->getContext();
    Type* I32 = Type::getInt32Ty(Ctx);

    BufferOperands ops = getBufferOperands(bufOp);

    // 1. Header marker
    emitBareTrace(B, encodeMarker(headerID, false, false), M, gen);

    // 2. EXEC mask (same as memory/LDS traces)
    unsigned waveSize = getWaveSize(gen);
    if (gen == GfxGen::GFX9)
    {
        InlineAsm* traceExecLo = InlineAsm::get(
            FunctionType::get(Type::getVoidTy(Ctx), false),
            "s_mov_b32 m0, exec_lo\n"
            "s_nop 0\n"
            "s_ttracedata",
            "",
            /*hasSideEffects=*/true
        );
        B.CreateCall(traceExecLo);

        InlineAsm* traceExecHi = InlineAsm::get(
            FunctionType::get(Type::getVoidTy(Ctx), false),
            "s_mov_b32 m0, exec_hi\n"
            "s_nop 0\n"
            "s_ttracedata",
            "",
            /*hasSideEffects=*/true
        );
        B.CreateCall(traceExecHi);
    }
    else
    {
        InlineAsm* readExecLo =
            InlineAsm::get(FunctionType::get(I32, false), "s_mov_b32 $0, exec_lo", "=s", /*hasSideEffects=*/true);
        emitBareTraceValue(B, B.CreateCall(readExecLo), M, gen);

        InlineAsm* readExecHi =
            InlineAsm::get(FunctionType::get(I32, false), "s_mov_b32 $0, exec_hi", "=s", /*hasSideEffects=*/true);
        emitBareTraceValue(B, B.CreateCall(readExecHi), M, gen);
    }

    // 3. rsrc base (lo/hi words)
    Value* rsrc = ops.Rsrc;
    Value *rsrcLo, *rsrcHi;
    Type* rsrcTy = rsrc->getType();

    if (rsrcTy->isVectorTy())
    {
        // Legacy <4 x i32> resource descriptor
        rsrcLo = B.CreateExtractElement(rsrc, (uint64_t) 0);
        rsrcHi = B.CreateExtractElement(rsrc, (uint64_t) 1);
    }
    else
    {
        // ptr addrspace(8) — ptrtoint to i128, extract lo/hi
        Type* I128 = Type::getIntNTy(Ctx, 128);
        Value* rsrcInt = B.CreatePtrToInt(rsrc, I128);
        rsrcLo = B.CreateTrunc(rsrcInt, I32);
        rsrcHi = B.CreateTrunc(B.CreateLShr(rsrcInt, 32), I32);
    }
    emitBareTraceValue(B, rsrcLo, M, gen);
    emitBareTraceValue(B, rsrcHi, M, gen);

    // 4. Scalar offset (uniform SGPR)
    Value* soffset = ops.SOffset;
    if (soffset->getType() != I32) soffset = B.CreateZExtOrTrunc(soffset, I32);
    emitBareTraceValue(B, soffset, M, gen);

    // 5. Per-lane readlane loop over voffset
    Function* ReadLane = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_readlane, {I32});

    Value* voffset = ops.VOffset;
    if (voffset->getType() != I32) voffset = B.CreateZExtOrTrunc(voffset, I32);

    // Helper lambda: emit a readlane loop for a VGPR value
    auto emitReadlaneLoop = [&](Value* vgprVal)
    {
        BasicBlock* PreheaderBB = B.GetInsertBlock();
        BasicBlock* AfterBB;
        bool atEnd = (B.GetInsertPoint() == PreheaderBB->end());
        if (!atEnd)
        {
            Instruction* SplitPt = &*B.GetInsertPoint();
            AfterBB = PreheaderBB->splitBasicBlock(SplitPt, "sqtt.buf.after");
            PreheaderBB->getTerminator()->eraseFromParent();
        }
        else { AfterBB = BasicBlock::Create(Ctx, "sqtt.buf.after", &F, PreheaderBB->getNextNode()); }
        BasicBlock* LoopBB = BasicBlock::Create(Ctx, "sqtt.buf.loop", &F, AfterBB);

        IRBuilder<> PreB(PreheaderBB);
        PreB.CreateBr(LoopBB);

        IRBuilder<> LoopB(LoopBB);
        PHINode* Lane = LoopB.CreatePHI(I32, 2, "lane");
        Lane->addIncoming(ConstantInt::get(I32, 0), PreheaderBB);

        Value* laneVal = LoopB.CreateCall(ReadLane, {vgprVal, Lane});
        emitBareTraceValue(LoopB, laneVal, M, gen);

        Value* LaneNext = LoopB.CreateAdd(Lane, ConstantInt::get(I32, 1), "lane.next");
        Lane->addIncoming(LaneNext, LoopBB);
        Value* Done = LoopB.CreateICmpEQ(LaneNext, ConstantInt::get(I32, waveSize));
        BranchInst* LoopBr = LoopB.CreateCondBr(Done, AfterBB, LoopBB);

        MDNode* LoopID =
            MDNode::getDistinct(Ctx, {nullptr, MDNode::get(Ctx, {MDString::get(Ctx, "llvm.loop.unroll.disable")})});
        LoopID->replaceOperandWith(0, LoopID);
        LoopBr->setMetadata(LLVMContext::MD_loop, LoopID);

        B.SetInsertPoint(AfterBB, AfterBB->begin());
    };

    emitReadlaneLoop(voffset);

    // 6. For struct buffers: second readlane loop over vindex
    if (ops.VIndex)
    {
        Value* vindex = ops.VIndex;
        if (vindex->getType() != I32) vindex = B.CreateZExtOrTrunc(vindex, I32);
        emitReadlaneLoop(vindex);
    }
}

// ============================================================================
// ds_permute / ds_bpermute trace emission
// ============================================================================

void SQTTInstrumentPass::emitPermuteTrace(
    IRBuilder<>& B, CallInst* permuteOp, uint32_t headerID, Function& F, GfxGen gen
)
{
    Module* M = F.getParent();
    LLVMContext& Ctx = M->getContext();
    Type* I32 = Type::getInt32Ty(Ctx);

    // 1. Header marker
    emitBareTrace(B, encodeMarker(headerID, false, false), M, gen);

    // 2. EXEC mask
    unsigned waveSize = getWaveSize(gen);
    if (gen == GfxGen::GFX9)
    {
        InlineAsm* traceExecLo = InlineAsm::get(
            FunctionType::get(Type::getVoidTy(Ctx), false),
            "s_mov_b32 m0, exec_lo\n"
            "s_nop 0\n"
            "s_ttracedata",
            "",
            /*hasSideEffects=*/true
        );
        B.CreateCall(traceExecLo);

        InlineAsm* traceExecHi = InlineAsm::get(
            FunctionType::get(Type::getVoidTy(Ctx), false),
            "s_mov_b32 m0, exec_hi\n"
            "s_nop 0\n"
            "s_ttracedata",
            "",
            /*hasSideEffects=*/true
        );
        B.CreateCall(traceExecHi);
    }
    else
    {
        InlineAsm* readExecLo =
            InlineAsm::get(FunctionType::get(I32, false), "s_mov_b32 $0, exec_lo", "=s", /*hasSideEffects=*/true);
        emitBareTraceValue(B, B.CreateCall(readExecLo), M, gen);

        InlineAsm* readExecHi =
            InlineAsm::get(FunctionType::get(I32, false), "s_mov_b32 $0, exec_hi", "=s", /*hasSideEffects=*/true);
        emitBareTraceValue(B, B.CreateCall(readExecHi), M, gen);
    }

    // 3. Per-lane readlane loop over index operand (arg 0)
    Value* index = permuteOp->getArgOperand(0);
    if (index->getType() != I32) index = B.CreateZExtOrTrunc(index, I32);

    Function* ReadLane = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_readlane, {I32});

    BasicBlock* PreheaderBB = B.GetInsertBlock();
    BasicBlock* AfterBB;
    bool atEnd = (B.GetInsertPoint() == PreheaderBB->end());
    if (!atEnd)
    {
        Instruction* SplitPt = &*B.GetInsertPoint();
        AfterBB = PreheaderBB->splitBasicBlock(SplitPt, "sqtt.perm.after");
        PreheaderBB->getTerminator()->eraseFromParent();
    }
    else { AfterBB = BasicBlock::Create(Ctx, "sqtt.perm.after", &F, PreheaderBB->getNextNode()); }
    BasicBlock* LoopBB = BasicBlock::Create(Ctx, "sqtt.perm.loop", &F, AfterBB);

    IRBuilder<> PreB(PreheaderBB);
    PreB.CreateBr(LoopBB);

    IRBuilder<> LoopB(LoopBB);
    PHINode* Lane = LoopB.CreatePHI(I32, 2, "lane");
    Lane->addIncoming(ConstantInt::get(I32, 0), PreheaderBB);

    Value* laneVal = LoopB.CreateCall(ReadLane, {index, Lane});
    emitBareTraceValue(LoopB, laneVal, M, gen);

    Value* LaneNext = LoopB.CreateAdd(Lane, ConstantInt::get(I32, 1), "lane.next");
    Lane->addIncoming(LaneNext, LoopBB);
    Value* Done = LoopB.CreateICmpEQ(LaneNext, ConstantInt::get(I32, waveSize));
    BranchInst* LoopBr = LoopB.CreateCondBr(Done, AfterBB, LoopBB);

    MDNode* LoopID =
        MDNode::getDistinct(Ctx, {nullptr, MDNode::get(Ctx, {MDString::get(Ctx, "llvm.loop.unroll.disable")})});
    LoopID->replaceOperandWith(0, LoopID);
    LoopBr->setMetadata(LLVMContext::MD_loop, LoopID);

    B.SetInsertPoint(AfterBB, AfterBB->begin());
}
