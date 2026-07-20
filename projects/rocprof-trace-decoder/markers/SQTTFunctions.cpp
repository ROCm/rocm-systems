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

#include <algorithm>

using namespace llvm;

void SQTTInstrumentPass::insertFunctionMarkers(Function& F, uint32_t id, GfxGen gen, bool useBareTrace)
{
    auto emit = [&](Instruction* I, uint32_t marker)
    {
        IRBuilder<> B(I);
        if (useBareTrace)
            emitBareTrace(B, marker, F.getParent(), gen);
        else
            insertTraceMarker(B, marker, F, gen);
    };

    Instruction* Entry = useBareTrace || !CurScopeCheck ? &*F.getEntryBlock().getFirstInsertionPt()
                                                         : cast<Instruction>(CurScopeCheck)->getNextNode();
    emit(Entry, encodeMarker(id, /*enter=*/true, /*exit_prev=*/false));

    SmallVector<ReturnInst*, 4> Rets;
    for (auto& BB : F)
        if (auto* RI = dyn_cast<ReturnInst>(BB.getTerminator())) Rets.push_back(RI);
    for (auto* RI : Rets)
        emit(RI, useBareTrace ? encodeMarker(id, /*enter=*/false, /*exit_prev=*/true) : FLAG_EXIT_PREV);
}

void SQTTInstrumentPass::storeEarlyMarkerMetadata(Module& M, LLVMContext& Ctx)
{
    Type* I32 = Type::getInt32Ty(Ctx);
    NamedMDNode* NMD = M.getOrInsertNamedMetadata("sqtt.markers.early");
    auto asInt = [&](uint32_t value) { return ConstantAsMetadata::get(ConstantInt::get(I32, value)); };
    for (const auto& entry : Markers)
        NMD->addOperand(MDNode::get(
            Ctx,
            {asInt(entry.ID),
             asInt(static_cast<unsigned>(entry.Kind)),
             MDString::get(Ctx, entry.Name),
             asInt(entry.PreOptSize),
             MDString::get(Ctx, entry.SourceLoc),
             asInt(entry.ExtraPayloadCount)}
        ));
}

bool SQTTInstrumentPass::recoverEarlyMarkerMetadata(Module& M, bool& hasEarlyFunctions)
{
    hasEarlyFunctions = false;
    NamedMDNode* NMD = M.getNamedMetadata("sqtt.markers.early");
    if (!NMD) return false;
    for (MDNode* Op : NMD->operands())
    {
        if (Op->getNumOperands() < 6) continue;
        auto* IdC = mdconst::dyn_extract<ConstantInt>(Op->getOperand(0));
        auto* KindC = mdconst::dyn_extract<ConstantInt>(Op->getOperand(1));
        auto* NameS = dyn_cast<MDString>(Op->getOperand(2));
        auto* SizeC = mdconst::dyn_extract<ConstantInt>(Op->getOperand(3));
        auto* LocS = dyn_cast<MDString>(Op->getOperand(4));
        auto* PayloadC = mdconst::dyn_extract<ConstantInt>(Op->getOperand(5));
        if (!IdC || !KindC || !NameS || !SizeC || !LocS || !PayloadC) continue;

        MarkerKind markerKind = static_cast<MarkerKind>(KindC->getZExtValue());
        if (markerKind != MarkerKind::Function && markerKind != MarkerKind::UserScope &&
            markerKind != MarkerKind::Point)
            continue;
        hasEarlyFunctions |= markerKind == MarkerKind::Function;
        uint32_t id = static_cast<uint32_t>(IdC->getZExtValue());
        Markers.push_back({id, markerKind, NameS->getString().str(), LocS->getString().str(),
                           static_cast<uint32_t>(SizeC->getZExtValue()), static_cast<uint32_t>(PayloadC->getZExtValue())});
        MarkerRecord& entry = Markers.back();
        if (markerKind == MarkerKind::UserScope || markerKind == MarkerKind::Point)
        {
            std::string key = std::string(markerKind == MarkerKind::Point ? "P:" : "U:") +
                              std::to_string(entry.ExtraPayloadCount) + ":" + entry.Name;
            UserMarkerMap[key] = id;
        }
        // Function IDs are compacted below.  Only user IDs need to advance
        // the no-op fallback counter used when every function is filtered.
        if (markerKind != MarkerKind::Function && id >= NextEventID) NextEventID = id + 1;
    }
    NMD->eraseFromParent();
    return true;
}

bool SQTTInstrumentPass::finalizeEarlyFunctionMarkers(Module& M)
{
    // One state entry owns all transient state for one early marker ID.
    struct Entry
    {
        MarkerRecord* Record;
        uint64_t Count = 0;
        uint32_t NewID = 0;
        bool Seen = false, Disabled = false;
    };
    std::map<uint32_t, Entry> entries;
    for (auto& record : Markers)
        if (record.ID) entries.emplace(record.ID, Entry{&record});
    auto find = [&](uint32_t id) -> Entry*
    {
        auto it = entries.find(id);
        return it == entries.end() ? nullptr : &it->second;
    };

    bool changed = false;
    if (Config.FunctionThreshold != 0)
    {
        // A clone group shares one early ID, so a below-threshold copy prunes
        // every copy that carries that ID.
        for (auto& F : M)
        {
            if (F.isDeclaration()) continue;
            MDNode* MD = F.getMetadata("sqtt.func.id");
            auto* IdC = MD ? mdconst::dyn_extract<ConstantInt>(MD->getOperand(0)) : nullptr;
            if (!IdC) continue;

            Entry* entry = find(IdC->getZExtValue());
            F.setMetadata("sqtt.func.id", nullptr);
            if (!entry || entry->Record->Kind != MarkerKind::Function) continue;

            bool firstCopy = !entry->Seen;
            entry->Seen = true;
            if (computeFunctionSize(F, Config.Mode) <= Config.FunctionThreshold)
            {
                changed |= !entry->Disabled;
                entry->Disabled = true;
            }
            else if (firstCopy)
            {
                std::string loc = getFunctionSourceLoc(F);
                if (!loc.empty()) entry->Record->SourceLoc = std::move(loc);
            }
        }

        for (auto& [id, entry] : entries)
            if (entry.Record->Kind == MarkerKind::Function && !entry.Seen &&
                entry.Record->PreOptSize <= Config.FunctionThreshold)
            {
                changed |= !entry.Disabled;
                entry.Disabled = true;
            }
    }

    // Snapshot first: pruning and lowering erase calls while preserving the
    // stable state pointer that owns each old ID.
    SmallVector<std::pair<CallInst*, Entry*>, 16> traces;
    for (auto& F : M)
        for (auto& BB : F)
            for (auto& I : BB)
            {
                auto* CI = dyn_cast<CallInst>(&I);
                if (!isTraceDataCall(CI) || !CI->getMetadata(SQTT_MARKER_HEADER_METADATA)) continue;
                auto* arg = dyn_cast<ConstantInt>(CI->getArgOperand(0));
                if (arg)
                    if (Entry* entry = find(static_cast<uint32_t>(arg->getZExtValue()) >> 2))
                        traces.emplace_back(CI, entry);
            }

    for (auto [call, entry] : traces)
    {
        uint32_t flags = cast<ConstantInt>(call->getArgOperand(0))->getZExtValue() & FLAG_MASK;
        if (entry->Disabled && (flags == FLAG_ENTER || flags == FLAG_EXIT_PREV))
        {
            call->eraseFromParent();
            changed = true;
            continue;
        }
        if (!entry->Disabled) ++entry->Count;
    }

    std::vector<Entry*> sorted;
    sorted.reserve(entries.size());
    for (auto& [id, entry] : entries)
        if (!entry.Disabled) sorted.push_back(&entry);
    std::sort(sorted.begin(), sorted.end(), [](const Entry* a, const Entry* b)
    {
        if (a->Count != b->Count) return a->Count > b->Count;
        return a->Record->ID < b->Record->ID;
    });
    uint32_t nextID = 1;
    for (Entry* entry : sorted) entry->NewID = nextID++;

    for (auto [call, entry] : traces)
    {
        // Disabled calls may have been erased above, so never dereference
        // their CallInst here.
        if (entry->Disabled) continue;
        GfxGen gen = getGfxGen(*call->getFunction());
        if (gen == GfxGen::Unknown) continue;
        uint32_t value = cast<ConstantInt>(call->getArgOperand(0))->getZExtValue();
        IRBuilder<> B(call);
        CallInst* replacement = emitBareTrace(
            B,
            (value & FLAG_MASK) == FLAG_EXIT_PREV ? FLAG_EXIT_PREV : (entry->NewID << 2) | (value & FLAG_MASK),
            &M,
            gen
        );
        replacement->copyMetadata(*call);
        call->eraseFromParent();
        changed = true;
    }

    Markers.erase(
        std::remove_if(Markers.begin(), Markers.end(), [&](const MarkerRecord& record)
        {
            Entry* entry = find(record.ID);
            return record.Kind == MarkerKind::Function && entry && entry->Disabled;
        }),
        Markers.end()
    );
    auto remap = [&](uint32_t& id)
    {
        if (Entry* entry = find(id)) id = entry->NewID;
    };
    for (auto& entry : Markers) remap(entry.ID);
    for (auto& [name, id] : UserMarkerMap) remap(id);
    NextEventID = nextID;
    return changed;
}

bool SQTTInstrumentPass::hasMustTailCall(const Function& F)
{
    for (const auto& BB : F)
        for (const auto& I : BB)
            if (const auto* CB = dyn_cast<CallBase>(&I); CB && CB->isMustTailCall()) return true;
    return false;
}

bool SQTTInstrumentPass::instrumentFunctionDirect(Function& F, GfxGen gen)
{
    if (hasMustTailCall(F) || computeFunctionSize(F, Config.Mode) <= Config.FunctionThreshold) return false;

    uint32_t id = NextEventID++;
    Markers.push_back({id, MarkerKind::Function, F.getName().str(), getFunctionSourceLoc(F)});
    insertFunctionMarkers(F, id, gen, /*useBareTrace=*/false);
    return true;
}
