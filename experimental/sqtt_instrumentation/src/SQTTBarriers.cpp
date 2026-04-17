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
