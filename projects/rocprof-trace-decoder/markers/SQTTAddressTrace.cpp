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
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace
{

constexpr const char* MemoryTraceNames[] = {
    "addr_trace_load", "addr_trace_store", "addr_trace_atomic"
};
constexpr const char* LDSTraceNames[] = {
    "addr_trace_lds_load", "addr_trace_lds_store", "addr_trace_lds_atomic"
};
constexpr const char* BufferTraceNames[][3] = {
    {"addr_trace_buffer_load", "addr_trace_buffer_store", "addr_trace_buffer_atomic"},
    {"addr_trace_struct_buffer_load", "addr_trace_struct_buffer_store", "addr_trace_struct_buffer_atomic"}
};

unsigned traceOperationIndex(bool isStore, bool isAtomic) { return isAtomic ? 2 : isStore; }

static constexpr const char ExecTraceAsm[] =
    "s_mov_b32 m0, exec_lo\n"
    "s_nop 0\n"
    "s_ttracedata\n"
    "s_mov_b32 m0, exec_hi\n"
    "s_nop 0\n"
    "s_ttracedata";

void emitExecMaskTraces(IRBuilder<>& B)
{
    LLVMContext& Ctx = B.getContext();
    B.CreateCall(InlineAsm::get(
        FunctionType::get(Type::getInt32Ty(Ctx), false), ExecTraceAsm, "={m0}", /*hasSideEffects=*/true
    ));
}

} // namespace

SQTTInstrumentPass::AddrTraceOp SQTTInstrumentPass::classifyAddrTraceOp(
    Instruction* I, bool traceMemory, bool traceLDS
)
{
    const auto none = [=] { return AddrTraceOp{I, nullptr, AddrTraceKind::None, 0, false}; };
    if (Value* pointer = getMemoryPointer(I))
    {
        // The memory address protocol is defined only for flat (0) and
        // global (1) pointers.  Other AMDGPU address spaces have different
        // representations and must not be reinterpreted as global addresses.
        unsigned AS = cast<PointerType>(pointer->getType())->getAddressSpace();
        bool isStore = isa<StoreInst>(I);
        unsigned op = traceOperationIndex(isStore, isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I));
        if (AS == 3 && traceLDS) return {I, LDSTraceNames[op], AddrTraceKind::LDS, 0, false};
        if ((AS == 0 || AS == 1) && traceMemory)
            return {I, MemoryTraceNames[op], AddrTraceKind::Memory, 0, false};
        return none();
    }

    auto* CI = dyn_cast<CallInst>(I);
    Function* Callee = CI ? CI->getCalledFunction() : nullptr;
    if (!Callee) return none();

    StringRef Name = Callee->getName();
    BufferOpKind bufferKind = classifyBufferOp(Name);
    // buffer.load.lds has a distinct operand layout (including an LDS
    // destination pointer), so the ordinary buffer component protocol is not
    // valid for it.
    if (traceMemory && bufferKind != BufferOpKind::None && !Name.ends_with(".buffer.load.lds"))
    {
        bool isStruct = isStructBuffer(Name);
        unsigned op = static_cast<unsigned>(bufferKind) - 1;
        unsigned rsrc = bufferKind == BufferOpKind::Load ? 0 : isBufferCmpSwap(Name) ? 2 : 1;
        return {I, BufferTraceNames[isStruct][op], AddrTraceKind::Buffer, rsrc, isStruct};
    }

    auto IID = Callee->getIntrinsicID();
    if (traceLDS && (IID == Intrinsic::amdgcn_ds_permute || IID == Intrinsic::amdgcn_ds_bpermute ||
                     IID == Intrinsic::amdgcn_ds_bpermute_fi_b32))
    {
        bool isBPermute = IID == Intrinsic::amdgcn_ds_bpermute || IID == Intrinsic::amdgcn_ds_bpermute_fi_b32;
        return {I, isBPermute ? "addr_trace_ds_bpermute" : "addr_trace_ds_permute", AddrTraceKind::Permute, 0,
                false};
    }
    return none();
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

void SQTTInstrumentPass::emitAddressTrace(
    IRBuilder<>& B, const AddrTraceOp& op, uint32_t headerID, GfxGen gen
)
{
    Module* M = B.GetInsertBlock()->getParent()->getParent();
    LLVMContext& Ctx = M->getContext();
    Type* I32 = Type::getInt32Ty(Ctx);

    bool schedBarrier = !Config.needsScopeCheck();
    emitTraceBoundary(B, /*after=*/false, schedBarrier);
    emitBareTrace(B, encodeMarker(headerID, false, false), M, gen);
    unsigned waveSize = getWaveSize(gen);

    switch (op.Kind)
    {
        case AddrTraceKind::Buffer:
        {
            CallInst* bufOp = cast<CallInst>(op.I);
            Value* rsrc = bufOp->getArgOperand(op.BufferRsrcIndex);
            Value* vindex = op.StructBuffer ? bufOp->getArgOperand(op.BufferRsrcIndex + 1) : nullptr;
            Value* voffset = bufOp->getArgOperand(op.BufferRsrcIndex + (op.StructBuffer ? 2 : 1));
            Value* soffset = bufOp->getArgOperand(op.BufferRsrcIndex + (op.StructBuffer ? 3 : 2));

            // Header, EXEC, descriptor words, scalar offset, then lane data.
            emitExecMaskTraces(B);
            Value *rsrcLo, *rsrcHi;
            if (rsrc->getType()->isVectorTy())
            {
                rsrcLo = B.CreateExtractElement(rsrc, uint64_t{0});
                rsrcHi = B.CreateExtractElement(rsrc, uint64_t{1});
            }
            else
            {
                Value* rsrcInt = B.CreatePtrToInt(rsrc, Type::getIntNTy(Ctx, 128));
                rsrcLo = B.CreateTrunc(rsrcInt, I32);
                rsrcHi = B.CreateTrunc(B.CreateLShr(rsrcInt, 32), I32);
            }
            emitBareTraceValue(B, rsrcLo, M);
            emitBareTraceValue(B, rsrcHi, M);
            if (soffset->getType() != I32) soffset = B.CreateZExtOrTrunc(soffset, I32);
            emitBareTraceValue(B, soffset, M);
            emitReadlaneTraceLoop(B, voffset, nullptr, waveSize);
            if (vindex)
                emitReadlaneTraceLoop(B, vindex, nullptr, waveSize);
            break;
        }
        case AddrTraceKind::Permute:
            emitExecMaskTraces(B);
            emitReadlaneTraceLoop(
                B, cast<CallInst>(op.I)->getArgOperand(0), nullptr, waveSize
            );
            break;
        case AddrTraceKind::Memory:
        case AddrTraceKind::LDS:
        {
            Value* ptr = getMemoryPointer(op.I);
            assert(ptr && "expected Load/Store/Atomic instruction");
            Value *addrLo, *addrHi = nullptr;
            if (op.Kind == AddrTraceKind::Memory)
            {
                Value* addr = B.CreatePtrToInt(ptr, Type::getInt64Ty(Ctx));
                addrLo = B.CreateTrunc(addr, I32);
                addrHi = B.CreateTrunc(B.CreateLShr(addr, 32), I32);
            }
            else
                addrLo = B.CreatePtrToInt(ptr, I32);
            emitExecMaskTraces(B);
            emitReadlaneTraceLoop(B, addrLo, addrHi, waveSize);
            break;
        }
        default: llvm_unreachable("unsupported address trace kind");
    }

    emitTraceBoundary(B, /*after=*/true, schedBarrier);
}

void SQTTInstrumentPass::emitReadlaneTraceLoop(
    IRBuilder<>& B,
    Value* firstValue,
    Value* secondValue,
    unsigned waveSize
)
{
    Function& F = *B.GetInsertBlock()->getParent();
    Module* M = F.getParent();
    LLVMContext& Ctx = M->getContext();
    Type* I32 = Type::getInt32Ty(Ctx);
    if (firstValue->getType() != I32) firstValue = B.CreateZExtOrTrunc(firstValue, I32);
    if (secondValue && secondValue->getType() != I32) secondValue = B.CreateZExtOrTrunc(secondValue, I32);

    Function* ReadLane = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_readlane, {I32});
    BasicBlock* PreheaderBB = B.GetInsertBlock();
    BasicBlock* AfterBB;
    if (B.GetInsertPoint() == PreheaderBB->end())
        AfterBB = BasicBlock::Create(Ctx, "sqtt.lanes.after", &F, PreheaderBB->getNextNode());
    else
    {
        AfterBB = PreheaderBB->splitBasicBlock(B.GetInsertPoint(), "sqtt.lanes.after");
        PreheaderBB->getTerminator()->eraseFromParent();
    }
    BasicBlock* LoopBB = BasicBlock::Create(Ctx, "sqtt.lanes.loop", &F, AfterBB);

    IRBuilder<> PreB(PreheaderBB);
    PreB.CreateBr(LoopBB);

    IRBuilder<> LoopB(LoopBB);
    PHINode* Lane = LoopB.CreatePHI(I32, 2, "lane");
    Lane->addIncoming(ConstantInt::get(I32, 0), PreheaderBB);
    emitBareTraceValue(LoopB, LoopB.CreateCall(ReadLane, {firstValue, Lane}), M);
    if (secondValue) emitBareTraceValue(LoopB, LoopB.CreateCall(ReadLane, {secondValue, Lane}), M);

    Value* LaneNext = LoopB.CreateAdd(Lane, ConstantInt::get(I32, 1), "lane.next");
    Lane->addIncoming(LaneNext, LoopBB);
    Value* Done = LoopB.CreateICmpEQ(LaneNext, ConstantInt::get(I32, waveSize));
    auto* LoopBr = LoopB.CreateCondBr(Done, AfterBB, LoopBB);

    MDNode* LoopID =
        MDNode::getDistinct(Ctx, {nullptr, MDNode::get(Ctx, {MDString::get(Ctx, "llvm.loop.unroll.disable")})});
    LoopID->replaceOperandWith(0, LoopID);
    LoopBr->setMetadata(LLVMContext::MD_loop, LoopID);
    B.SetInsertPoint(AfterBB, AfterBB->begin());
}

bool SQTTInstrumentPass::instrumentAddressTraces(Function& F, GfxGen gen)
{
    SmallVector<AddrTraceOp, 16> Ops;
    for (auto& BB : F)
    {
        for (auto& I : BB)
        {
            if (AddrTraceOp op = classifyAddrTraceOp(&I, Config.TraceMemoryAddrs, Config.TraceLDSAddrs);
                op.Kind != AddrTraceKind::None)
                Ops.push_back(op);
        }
    }
    if (Ops.empty()) return false;

    // Wave size is recorded once per module for the .sqtt_funcmap header.
    // RDNA is wave-32, CDNA is wave-64; a single AMDGPU code object normally
    // targets one or the other. If we ever see a mix, default to wave-64 —
    // the decoder treats exec_hi=0 padding as "no upper-half lanes" so the
    // wider format stays correct for both.
    unsigned waveSize = getWaveSize(gen);
    if (waveSize > AddrTraceWaveSize) AddrTraceWaveSize = waveSize;

    for (auto& op : Ops)
    {
        uint32_t opID = NextEventID++;
        unsigned extraPayloadCount =
            2 + (op.Kind == AddrTraceKind::Buffer ? 3 : 0) +
            waveSize * ((op.Kind == AddrTraceKind::Memory || op.StructBuffer) ? 2 : 1);
        Markers.push_back({opID, MarkerKind::AddressPoint, op.Name, getSourceLoc(op.I), 0, extraPayloadCount});

        IRBuilder<> B(op.I);
        emitScopedTrace(
            B,
            F,
            gen,
            "sqtt.addr.trace",
            "sqtt.addr.skip",
            /*pinSkipHead=*/false,
            [&](IRBuilder<>& Trace) { emitAddressTrace(Trace, op, opID, gen); }
        );
    }
    return true;
}
