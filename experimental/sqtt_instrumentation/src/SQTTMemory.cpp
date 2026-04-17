#include "SQTTPass.h"

#include "llvm/IR/Constants.h"

using namespace llvm;

SQTTInstrumentPass::MemOpKind SQTTInstrumentPass::classifyMemOp(Instruction* I)
{
    if (auto* LI = dyn_cast<LoadInst>(I))
    {
        unsigned AS = LI->getPointerAddressSpace();
        if (AS == 3 || AS == 5) return MemOpKind::None; // LDS or private
        return MemOpKind::Load;
    }
    if (auto* SI = dyn_cast<StoreInst>(I))
    {
        unsigned AS = SI->getPointerAddressSpace();
        if (AS == 3 || AS == 5) return MemOpKind::None; // LDS or private
        return MemOpKind::Store;
    }
    // Atomics are read-modify-write, classify as Store
    if (auto* AI = dyn_cast<AtomicRMWInst>(I))
    {
        unsigned AS = AI->getPointerAddressSpace();
        if (AS == 3 || AS == 5) return MemOpKind::None;
        return MemOpKind::Store;
    }
    if (auto* AX = dyn_cast<AtomicCmpXchgInst>(I))
    {
        unsigned AS = AX->getPointerAddressSpace();
        if (AS == 3 || AS == 5) return MemOpKind::None;
        return MemOpKind::Store;
    }
    // Buffer intrinsics
    if (auto* CI = dyn_cast<CallInst>(I))
    {
        Function* Callee = CI->getCalledFunction();
        if (!Callee) return MemOpKind::None;
        StringRef Name = Callee->getName();
        if (isBufferLoad(Name)) return MemOpKind::Load;
        if (isBufferStore(Name) || isBufferAtomic(Name)) return MemOpKind::Store;
    }
    return MemOpKind::None;
}

bool SQTTInstrumentPass::instrumentMemoryOps(Function& F, GfxGen gen)
{
    unsigned chunkSize = Config.MemoryChunkSize;
    unsigned maxGap = Config.MemoryMaxGap;

    struct MemOp
    {
        Instruction* I;
        MemOpKind Kind;
    };
    SmallVector<std::pair<BasicBlock*, SmallVector<MemOp, 8>>, 8> BBOps;

    for (auto& BB : F)
    {
        SmallVector<MemOp, 8> Ops;
        for (auto& I : BB)
        {
            MemOpKind kind = classifyMemOp(&I);
            if (kind != MemOpKind::None) Ops.push_back({&I, kind});
        }
        if (!Ops.empty()) BBOps.push_back({&BB, std::move(Ops)});
    }

    if (BBOps.empty()) return false;

    bool Changed = false;

    for (auto& [BB, Ops] : BBOps)
    {
        struct Sequence
        {
            MemOpKind Kind;
            SmallVector<Instruction*, 8> Ops;
        };
        SmallVector<Sequence, 4> Seqs;

        Sequence curSeq;
        curSeq.Kind = Ops[0].Kind;
        curSeq.Ops.push_back(Ops[0].I);

        for (unsigned i = 1; i < Ops.size(); i++)
        {
            unsigned gap = 0;
            auto It = std::next(Ops[i - 1].I->getIterator());
            for (; &*It != Ops[i].I; ++It) gap++;

            if (gap > maxGap || Ops[i].Kind != curSeq.Kind)
            {
                Seqs.push_back(std::move(curSeq));
                curSeq = Sequence();
                curSeq.Kind = Ops[i].Kind;
            }
            curSeq.Ops.push_back(Ops[i].I);
        }
        Seqs.push_back(std::move(curSeq));

        for (auto& seq : Seqs)
        {
            uint32_t markerID = (seq.Kind == MemOpKind::Load) ? VmemLoadID : VmemStoreID;
            uint32_t encoded = encodeMarker(markerID, false, false);

            for (unsigned i = 0; i < seq.Ops.size(); i += chunkSize)
            {
                unsigned end = std::min((unsigned) seq.Ops.size(), i + chunkSize);
                Instruction* lastOp = seq.Ops[end - 1];
                IRBuilder<> B(lastOp->getNextNode());
                insertTraceMarker(B, encoded, F, gen);
                Changed = true;
            }
        }
    }

    return Changed;
}
