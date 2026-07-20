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

using namespace llvm;

SQTTInstrumentPass::BarrierKind SQTTInstrumentPass::classifyBarrier(CallInst* CI)
{
    if (!CI) return BarrierKind::None;
    Function* Callee = CI->getCalledFunction();
    if (!Callee) return BarrierKind::None;
    switch (Callee->getIntrinsicID())
    {
        case Intrinsic::amdgcn_s_barrier_signal:
        case Intrinsic::amdgcn_s_barrier_signal_var:
        case Intrinsic::amdgcn_s_barrier_signal_isfirst: return BarrierKind::Signal;
        case Intrinsic::amdgcn_s_barrier_wait: return BarrierKind::Wait;
        case Intrinsic::amdgcn_s_barrier: return BarrierKind::Full;
        default: return BarrierKind::None;
    }
}

bool SQTTInstrumentPass::instrumentBarriers(Function& F, GfxGen gen)
{
    // Snapshot all insertion points before changing the CFG.
    SmallVector<std::pair<Instruction*, uint32_t>, 8> insertions;
    auto markerID = [this](BarrierKind kind)
    {
        return kind == BarrierKind::None ? 0 : FirstBarrierID + static_cast<uint32_t>(kind);
    };
    for (auto& BB : F)
    {
        CallInst* signal = nullptr;
        for (auto& I : BB)
        {
            auto* call = dyn_cast<CallInst>(&I);
            BarrierKind kind = classifyBarrier(call);
            if (signal && (signal->getNextNode() != &I || kind != BarrierKind::Wait))
            {
                insertions.push_back({signal->getNextNode(), markerID(BarrierKind::Signal)});
                signal = nullptr;
            }
            if (kind == BarrierKind::Signal)
                signal = call;
            else if (kind == BarrierKind::Wait && signal)
            {
                insertions.push_back({call, markerID(BarrierKind::Full)});
                signal = nullptr;
            }
            else if (kind == BarrierKind::Wait || kind == BarrierKind::Full)
                insertions.push_back({call, markerID(kind)});
        }
        if (signal) insertions.push_back({signal->getNextNode(), markerID(BarrierKind::Signal)});
    }
    if (insertions.empty()) return false;
    for (auto [before, markerID] : insertions)
    {
        IRBuilder<> B(before);
        insertTraceMarker(B, encodeMarker(markerID, false, false), F, gen);
    }
    return true;
}

SQTTInstrumentPass::MemOpKind SQTTInstrumentPass::classifyMemOp(Instruction* I)
{
    if (Value* pointer = getMemoryPointer(I))
    {
        unsigned AS = cast<PointerType>(pointer->getType())->getAddressSpace();
        // Atomics are read-modify-write, so use store markers.
        if (AS == 3 || AS == 5) return MemOpKind::None;
        return isa<LoadInst>(I) ? MemOpKind::Load : MemOpKind::Store;
    }
    auto* CI = dyn_cast<CallInst>(I);
    Function* Callee = CI ? CI->getCalledFunction() : nullptr;
    if (!Callee) return MemOpKind::None;
    BufferOpKind kind = classifyBufferOp(Callee->getName());
    return kind == BufferOpKind::Load ? MemOpKind::Load
           : kind == BufferOpKind::None ? MemOpKind::None
                                        : MemOpKind::Store;
}

bool SQTTInstrumentPass::instrumentMemoryOps(Function& F, GfxGen gen)
{
    // Snapshot first: inserting a trace changes the instruction stream being chunked.
    SmallVector<std::pair<Instruction*, uint32_t>, 16> insertions;
    auto markerID = [this](MemOpKind kind)
    {
        return kind == MemOpKind::None ? 0 : FirstVmemID + static_cast<uint32_t>(kind);
    };
    for (auto& BB : F)
    {
        MemOpKind runKind = MemOpKind::None;
        Instruction* lastOp = nullptr;
        unsigned runSize = 0, gap = 0;
        auto flush = [&]
        {
            if (runSize)
                insertions.push_back({lastOp, markerID(runKind)});
            runSize = 0;
        };
        for (auto& I : BB)
        {
            MemOpKind kind = classifyMemOp(&I);
            if (kind == MemOpKind::None)
            {
                gap += lastOp != nullptr;
                continue;
            }
            if (kind != runKind || gap > Config.MemoryMaxGap) flush();

            runKind = kind;
            lastOp = &I;
            gap = 0;
            if (++runSize == Config.MemoryChunkSize) flush();
        }
        flush();
    }
    if (insertions.empty()) return false;

    for (auto [lastOp, markerID] : insertions)
    {
        IRBuilder<> B(lastOp->getNextNode());
        insertTraceMarker(B, encodeMarker(markerID, false, false), F, gen);
    }
    return true;
}
