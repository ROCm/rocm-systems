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

#include <set>

using namespace llvm;

void SQTTInstrumentPass::insertBareMarkers(Function& F, uint32_t id, GfxGen gen)
{
    Module* M = F.getParent();

    // Entry marker
    BasicBlock& EntryBB = F.getEntryBlock();
    IRBuilder<> EntryB(&*EntryBB.getFirstInsertionPt());
    emitBareTrace(EntryB, encodeMarker(id, /*enter=*/true, /*exit_prev=*/false), M, gen);

    // Exit markers before every ret.
    SmallVector<ReturnInst*, 4> Rets;
    for (auto& BB : F)
        if (auto* RI = dyn_cast<ReturnInst>(BB.getTerminator())) Rets.push_back(RI);
    for (auto* RI : Rets)
    {
        IRBuilder<> RetB(RI);
        emitBareTrace(RetB, encodeMarker(id, /*enter=*/false, /*exit_prev=*/true), M, gen);
    }
}

void SQTTInstrumentPass::storeFuncMetadata(Function& F, uint32_t id, LLVMContext& Ctx)
{
    Type* I32 = Type::getInt32Ty(Ctx);
    MDNode* MD = MDNode::get(Ctx, {ConstantAsMetadata::get(ConstantInt::get(I32, id))});
    F.setMetadata("sqtt.func.id", MD);

    unsigned preOptSize = computeFunctionSize(F, Config.Mode);
    std::string srcLoc = getFunctionSourceLoc(F);

    Module* M = F.getParent();
    NamedMDNode* NMD = M->getOrInsertNamedMetadata("sqtt.funcmap.early");
    NMD->addOperand(MDNode::get(
        Ctx,
        {ConstantAsMetadata::get(ConstantInt::get(I32, id)),
         MDString::get(Ctx, F.getName()),
         ConstantAsMetadata::get(ConstantInt::get(I32, preOptSize)),
         MDString::get(Ctx, srcLoc)}
    ));
}

void SQTTInstrumentPass::storeUserMarkerMetadata(Module& M, LLVMContext& Ctx)
{
    Type* I32 = Type::getInt32Ty(Ctx);
    NamedMDNode* NMD = M.getOrInsertNamedMetadata("sqtt.usermarkers.early");
    for (auto& entry : UserMarkers)
    {
        NMD->addOperand(MDNode::get(
            Ctx,
            {ConstantAsMetadata::get(ConstantInt::get(I32, entry.ID)),
             MDString::get(Ctx, entry.Name),
             ConstantAsMetadata::get(ConstantInt::get(I32, entry.IsPoint ? 1 : 0))}
        ));
    }
}

void SQTTInstrumentPass::recoverUserMarkerMetadata(Module& M)
{
    NamedMDNode* NMD = M.getNamedMetadata("sqtt.usermarkers.early");
    if (!NMD) return;
    for (unsigned i = 0; i < NMD->getNumOperands(); i++)
    {
        MDNode* Op = NMD->getOperand(i);
        if (Op->getNumOperands() < 3) continue;
        auto* IdC = mdconst::dyn_extract<ConstantInt>(Op->getOperand(0));
        auto* NameS = dyn_cast<MDString>(Op->getOperand(1));
        auto* PointC = mdconst::dyn_extract<ConstantInt>(Op->getOperand(2));
        if (!IdC || !NameS || !PointC) continue;
        uint32_t id = IdC->getZExtValue();
        std::string name = NameS->getString().str();
        bool isPoint = PointC->getZExtValue() != 0;
        UserMarkers.push_back({id, name, isPoint});
        UserMarkerMap[name] = id;
        if (id >= NextEventID) NextEventID = id + 1;
    }
    NMD->eraseFromParent();
}

bool SQTTInstrumentPass::filterInstrumentedFunctions(Module& M)
{
    if (Config.FunctionThreshold == 0) return false;

    bool Changed = false;

    struct EarlyEntry
    {
        std::string Name;
        unsigned PreOptSize;
        std::string SourceLoc;
    };
    std::map<uint32_t, EarlyEntry> EarlyMap;
    if (NamedMDNode* NMD = M.getNamedMetadata("sqtt.funcmap.early"))
    {
        for (unsigned i = 0; i < NMD->getNumOperands(); i++)
        {
            MDNode* Op = NMD->getOperand(i);
            if (Op->getNumOperands() < 2) continue;
            auto* IdC = mdconst::dyn_extract<ConstantInt>(Op->getOperand(0));
            auto* NameS = dyn_cast<MDString>(Op->getOperand(1));
            if (!IdC || !NameS) continue;

            unsigned preOptSize = 0;
            if (Op->getNumOperands() >= 3)
            {
                if (auto* SzC = mdconst::dyn_extract<ConstantInt>(Op->getOperand(2))) preOptSize = SzC->getZExtValue();
            }
            std::string srcLoc;
            if (Op->getNumOperands() >= 4)
            {
                if (auto* LocS = dyn_cast<MDString>(Op->getOperand(3))) srcLoc = LocS->getString().str();
            }
            EarlyMap[IdC->getZExtValue()] = {NameS->getString().str(), preOptSize, srcLoc};
        }
        NMD->eraseFromParent();
    }

    SmallVector<uint32_t, 8> RemoveIDs;

    for (auto& F : M)
    {
        if (F.isDeclaration()) continue;
        MDNode* MD = F.getMetadata("sqtt.func.id");
        if (!MD) continue;

        auto* IdC = mdconst::dyn_extract<ConstantInt>(MD->getOperand(0));
        if (!IdC) continue;
        uint32_t id = IdC->getZExtValue();

        unsigned size = computeFunctionSize(F, Config.Mode);
        if (size > Config.FunctionThreshold)
        {
            auto it = EarlyMap.find(id);
            if (it != EarlyMap.end())
            {
                // Prefer the live function's current source loc (debug info
                // may have been updated by optimization passes); fall back
                // to the value stashed pre-inline.
                std::string loc = getFunctionSourceLoc(F);
                if (loc.empty()) loc = it->second.SourceLoc;
                FuncMap.push_back({id, it->second.Name, loc});
                EarlyMap.erase(it);
            }
        }
        else
        {
            RemoveIDs.push_back(id);
            EarlyMap.erase(id);
        }

        F.setMetadata("sqtt.func.id", nullptr);
    }

    for (auto it = EarlyMap.begin(); it != EarlyMap.end();)
    {
        auto& [id, entry] = *it;
        if (entry.PreOptSize > Config.FunctionThreshold)
        {
            FuncMap.push_back({id, entry.Name, entry.SourceLoc});
            ++it;
        }
        else
        {
            RemoveIDs.push_back(id);
            it = EarlyMap.erase(it);
        }
    }

    for (uint32_t id : RemoveIDs)
    {
        removeFuncMarkersFromModule(M, id);
        Changed = true;
    }

    return Changed;
}

uint32_t SQTTInstrumentPass::compactFuncIDs(Module& M)
{
    if (FuncMap.empty() && UserMarkers.empty()) return NextEventID;

    // Count emissions per marker ID in the IR. The most-emitted markers
    // get the lowest new IDs, so they fit s_ttracedata_imm's 6-bit field.
    std::map<uint32_t, uint64_t> Counts;
    for (auto& F : M)
    {
        if (F.isDeclaration()) continue;
        for (auto& BB : F)
        {
            for (auto& I : BB)
            {
                auto* CI = dyn_cast<CallInst>(&I);
                if (!CI) continue;
                Function* Callee = CI->getCalledFunction();
                if (!Callee) continue;
                auto IID = Callee->getIntrinsicID();
                if (IID != Intrinsic::amdgcn_s_ttracedata && IID != Intrinsic::amdgcn_s_ttracedata_imm) continue;
                auto* Arg = dyn_cast<ConstantInt>(CI->getArgOperand(0));
                if (!Arg) continue;
                uint32_t val = Arg->getZExtValue();
                if (val == FLAG_EXIT_PREV) continue; // bare exit pop, no id
                uint32_t id = val >> 2;
                if (id == 0) continue;
                Counts[id]++;
            }
        }
    }

    // Build candidate set: every ID that has a record (user marker or func).
    std::set<uint32_t> Candidates;
    for (auto& entry : FuncMap) Candidates.insert(entry.ID);
    for (auto& entry : UserMarkers) Candidates.insert(entry.ID);

    // Sort by descending emission count, ties broken by old ID for stability.
    std::vector<uint32_t> Sorted(Candidates.begin(), Candidates.end());
    std::sort(
        Sorted.begin(),
        Sorted.end(),
        [&](uint32_t a, uint32_t b)
        {
            uint64_t ca = Counts.count(a) ? Counts[a] : 0;
            uint64_t cb = Counts.count(b) ? Counts[b] : 0;
            if (ca != cb) return ca > cb;
            return a < b;
        }
    );

    std::map<uint32_t, uint32_t> IDMap;
    uint32_t nextID = 1;
    for (uint32_t old : Sorted) IDMap[old] = nextID++;

    // Apply IR rewrite.
    for (auto& F : M)
    {
        if (F.isDeclaration()) continue;
        GfxGen gen = getGfxGen(F);
        if (gen == GfxGen::Unknown) continue;
        rewriteMarkerIDs(F, IDMap, gen);
    }

    // Update internal records.
    for (auto& entry : FuncMap)
    {
        auto it = IDMap.find(entry.ID);
        if (it != IDMap.end()) entry.ID = it->second;
    }
    for (auto& entry : UserMarkers)
    {
        auto it = IDMap.find(entry.ID);
        if (it != IDMap.end()) entry.ID = it->second;
    }
    for (auto& kv : UserMarkerMap)
    {
        auto it = IDMap.find(kv.second);
        if (it != IDMap.end()) kv.second = it->second;
    }

    // Keep FuncMap sorted by new ID for deterministic .sqtt_funcmap output.
    std::sort(FuncMap.begin(), FuncMap.end(), [](const FuncMapEntry& a, const FuncMapEntry& b) { return a.ID < b.ID; });
    return nextID;
}

void SQTTInstrumentPass::rewriteMarkerIDs(Function& F, const std::map<uint32_t, uint32_t>& IDMap, GfxGen gen)
{
    Module* M = F.getParent();
    LLVMContext& Ctx = M->getContext();
    Type* I32 = Type::getInt32Ty(Ctx);
    Type* I16 = Type::getInt16Ty(Ctx);

    SmallVector<std::pair<CallInst*, uint32_t>, 8> Rewrites;

    for (auto& BB : F)
    {
        for (auto& I : BB)
        {
            auto* CI = dyn_cast<CallInst>(&I);
            if (!CI) continue;
            Function* Callee = CI->getCalledFunction();
            if (!Callee) continue;

            auto IID = Callee->getIntrinsicID();
            if (IID != Intrinsic::amdgcn_s_ttracedata && IID != Intrinsic::amdgcn_s_ttracedata_imm) continue;

            auto* Arg = dyn_cast<ConstantInt>(CI->getArgOperand(0));
            if (!Arg) continue;
            uint32_t val = Arg->getZExtValue();

            uint32_t flags = val & FLAG_MASK;
            uint32_t id = val >> 2;

            auto it = IDMap.find(id);
            if (it == IDMap.end()) continue;

            uint32_t newVal;
            if (flags & FLAG_ENTER)
            {
                // Enter (and possibly fused exit+enter) — preserve flags
                newVal = (it->second << 2) | flags;
            }
            else if (flags & FLAG_EXIT_PREV)
            {
                // Pure exit pop — id payload is unused at decode time
                newVal = FLAG_EXIT_PREV;
            }
            else
            {
                // Point marker (flags == 0, id != 0) — preserve id, no flags
                newVal = (it->second << 2);
            }
            Rewrites.push_back({CI, newVal});
        }
    }

    for (auto& [CI, newVal] : Rewrites)
    {
        IRBuilder<> B(CI);
        if (canUseImm(newVal) && supportsImmTrace(gen))
        {
            Function* TTDImm = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata_imm);
            B.CreateCall(TTDImm, {ConstantInt::get(I16, newVal)});
        }
        else
        {
            Function* TTD = Intrinsic::getOrInsertDeclaration(M, Intrinsic::amdgcn_s_ttracedata);
            B.CreateCall(TTD, {ConstantInt::get(I32, newVal)});
        }
        CI->eraseFromParent();
    }
}

void SQTTInstrumentPass::removeFuncMarkersFromModule(Module& M, uint32_t id)
{
    uint32_t entryEncoded = encodeMarker(id, /*enter=*/true, /*exit_prev=*/false);
    uint32_t exitEncoded = encodeMarker(id, /*enter=*/false, /*exit_prev=*/true);

    SmallVector<CallInst*, 8> ToRemove;
    for (auto& F : M)
    {
        for (auto& BB : F)
        {
            for (auto& I : BB)
            {
                auto* CI = dyn_cast<CallInst>(&I);
                if (!CI) continue;
                Function* Callee = CI->getCalledFunction();
                if (!Callee) continue;

                auto IID = Callee->getIntrinsicID();
                if (IID != Intrinsic::amdgcn_s_ttracedata && IID != Intrinsic::amdgcn_s_ttracedata_imm) continue;

                auto* Arg = dyn_cast<ConstantInt>(CI->getArgOperand(0));
                if (!Arg) continue;
                uint32_t val = Arg->getZExtValue();
                if (val == entryEncoded || val == exitEncoded) ToRemove.push_back(CI);
            }
        }
    }

    for (auto* CI : ToRemove)
    {
        removeAdjacentBarriers(CI);
        CI->eraseFromParent();
    }
}

void SQTTInstrumentPass::removeAdjacentBarriers(CallInst* CI)
{
    auto isSQTTBarrier = [](Instruction* I) -> bool
    {
        if (!I) return false;
        auto* C = dyn_cast<CallInst>(I);
        if (!C) return false;
        Function* F = C->getCalledFunction();
        if (F && F->getIntrinsicID() == Intrinsic::amdgcn_sched_barrier) return true;
        if (C->isInlineAsm() && C->getType()->isVoidTy() && C->arg_empty())
        {
            if (auto* IA = dyn_cast<InlineAsm>(C->getCalledOperand()))
                if (IA->getAsmString().empty()) return true;
        }
        return false;
    };

    // Remove barriers after CI
    Instruction* Next = CI->getNextNode();
    while (Next && isSQTTBarrier(Next))
    {
        Instruction* AfterNext = Next->getNextNode();
        Next->eraseFromParent();
        Next = AfterNext;
    }

    // Remove barriers before CI
    BasicBlock* BB = CI->getParent();
    BasicBlock::iterator It(CI);
    while (It != BB->begin())
    {
        Instruction* Prev = &*std::prev(It);
        if (!isSQTTBarrier(Prev)) break;
        It = Prev->getIterator();
        Prev->eraseFromParent();
    }
}

bool SQTTInstrumentPass::instrumentFunctionDirect(Function& F, GfxGen gen)
{
    unsigned size = computeFunctionSize(F, Config.Mode);
    if (size <= Config.FunctionThreshold) return false;

    uint32_t id = NextEventID++;
    FuncMap.push_back({id, F.getName().str(), getFunctionSourceLoc(F)});

    BasicBlock& EntryBB = F.getEntryBlock();
    Instruction* InsertPt =
        CurScopeCheck ? cast<Instruction>(CurScopeCheck)->getNextNode() : &*EntryBB.getFirstInsertionPt();
    IRBuilder<> EntryB(InsertPt);
    insertTraceMarker(EntryB, encodeMarker(id, true, false), F, gen);

    SmallVector<ReturnInst*, 4> Rets;
    for (auto& BB : F)
        if (auto* RI = dyn_cast<ReturnInst>(BB.getTerminator())) Rets.push_back(RI);
    for (auto* RI : Rets)
    {
        IRBuilder<> RetB(RI);
        insertTraceMarker(RetB, FLAG_EXIT_PREV, F, gen);
    }
    return true;
}
