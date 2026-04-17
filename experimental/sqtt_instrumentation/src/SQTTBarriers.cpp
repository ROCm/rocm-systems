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

using namespace llvm;

SQTTInstrumentPass::BarrierKind SQTTInstrumentPass::classifyBarrier(CallInst* CI)
{
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
    SmallVector<std::pair<CallInst*, BarrierKind>, 8> Barriers;
    for (auto& BB : F)
    {
        for (auto& I : BB)
        {
            auto* CI = dyn_cast<CallInst>(&I);
            if (!CI) continue;
            BarrierKind kind = classifyBarrier(CI);
            if (kind != BarrierKind::None) Barriers.push_back({CI, kind});
        }
    }

    if (Barriers.empty()) return false;

    bool Changed = false;

    for (unsigned i = 0; i < Barriers.size(); i++)
    {
        auto [CI, kind] = Barriers[i];

        if (kind == BarrierKind::Signal && i + 1 < Barriers.size() && Barriers[i + 1].second == BarrierKind::Wait &&
            CI->getParent() == Barriers[i + 1].first->getParent())
        {
            CallInst* WaitCI = Barriers[i + 1].first;
            IRBuilder<> B(WaitCI);
            insertTraceMarker(B, encodeMarker(BarrierFullID, false, false), F, gen);
            Changed = true;
            i++;
            continue;
        }

        if (kind == BarrierKind::Signal)
        {
            IRBuilder<> B(CI->getNextNode());
            insertTraceMarker(B, encodeMarker(BarrierSignalID, false, false), F, gen);
            Changed = true;
        }
        else if (kind == BarrierKind::Wait)
        {
            IRBuilder<> B(CI);
            insertTraceMarker(B, encodeMarker(BarrierWaitID, false, false), F, gen);
            Changed = true;
        }
        else if (kind == BarrierKind::Full)
        {
            IRBuilder<> B(CI);
            insertTraceMarker(B, encodeMarker(BarrierFullID, false, false), F, gen);
            Changed = true;
        }
    }

    return Changed;
}
