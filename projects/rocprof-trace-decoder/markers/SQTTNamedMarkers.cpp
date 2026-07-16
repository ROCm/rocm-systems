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
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

SmallVector<SQTTInstrumentPass::MarkerCall, 8> SQTTInstrumentPass::collectSentinelCalls(Function& F)
{
    SmallVector<MarkerCall, 8> Calls;
    for (auto& BB : F)
    {
        for (auto& I : BB)
        {
            auto* CI = dyn_cast<CallInst>(&I);
            if (!CI) continue;
            Function* Callee = CI->getCalledFunction();
            if (!Callee) continue;
            StringRef Name = Callee->getName();
            if (Name == "__sqtt_named_marker_enter")
                Calls.push_back({CI, MarkerType::Enter});
            else if (Name == "__sqtt_named_marker_exit")
                Calls.push_back({CI, MarkerType::Exit});
            else if (Name == "__sqtt_named_marker_point")
                Calls.push_back({CI, MarkerType::Point});
            else if (Name == "__sqtt_named_marker_data")
                Calls.push_back({CI, MarkerType::Data});
        }
    }
    return Calls;
}

uint32_t SQTTInstrumentPass::resolveMarkerString(CallInst* CI, MarkerType type)
{
    Value* Arg = CI->getArgOperand(0)->stripPointerCasts();
    auto* GV = dyn_cast<GlobalVariable>(Arg);
    if (!GV || !GV->hasInitializer()) return 0;
    auto* CDA = dyn_cast<ConstantDataArray>(GV->getInitializer());
    if (!CDA || !CDA->isString()) return 0;

    // Exit just pops the top of the marker stack — the name string is
    // unused at the trace level, so no ID/funcmap entry is needed.
    if (type == MarkerType::Exit) return FLAG_EXIT_PREV; // value 1: pop top scope

    std::string Name = CDA->getAsString().str();
    if (!Name.empty() && Name.back() == '\0') Name.pop_back();

    bool isPoint = (type == MarkerType::Point || type == MarkerType::Data);
    uint32_t extraPayloadCount = (type == MarkerType::Data) ? 1 : 0;
    std::string key = std::string(isPoint ? "P:" : "U:") + std::to_string(extraPayloadCount) + ":" + Name;
    uint32_t id;
    auto it = UserMarkerMap.find(key);
    if (it != UserMarkerMap.end()) { id = it->second; }
    else
    {
        id = NextEventID++;
        UserMarkerMap[key] = id;
        UserMarkers.push_back({id, Name, isPoint, extraPayloadCount});
    }
    bool enter = (type == MarkerType::Enter);
    return encodeMarker(id, enter, false); // enter or point
}

bool SQTTInstrumentPass::resolveNamedMarkersEarly(Function& F, GfxGen gen)
{
    return processMarkerCalls(F, gen, /*useBareTrace=*/true);
}

bool SQTTInstrumentPass::processNamedMarkers(Function& F, GfxGen gen)
{
    return processMarkerCalls(F, gen, /*useBareTrace=*/false);
}

// Shared logic for resolveNamedMarkersEarly and processNamedMarkers.
// When useBareTrace=true (early pass): emits bare s_ttracedata, skips
// unresolvable calls.
// When useBareTrace=false (late pass): emits full markers with scope
// checks/barriers, warns on unresolvable calls.
bool SQTTInstrumentPass::processMarkerCalls(Function& F, GfxGen gen, bool useBareTrace)
{
    auto Calls = collectSentinelCalls(F);
    if (Calls.empty()) return false;

    Module* M = F.getParent();
    bool Changed = false;

    for (unsigned i = 0; i < Calls.size(); i++)
    {
        auto& [CI, Type] = Calls[i];
        if (!CI) continue; // already consumed by fusion

        // Fuse only directly adjacent exit+enter pairs.  A marker boundary
        // must not absorb work between the calls.
        if (Type == MarkerType::Exit && i + 1 < Calls.size())
        {
            auto& [NextCI, NextType] = Calls[i + 1];
            if (NextCI && NextType == MarkerType::Enter && CI->getNextNode() == NextCI)
            {
                uint32_t enterEncoded = resolveMarkerString(NextCI, MarkerType::Enter);
                if (enterEncoded)
                {
                    uint32_t id = enterEncoded >> 2;
                    uint32_t fused = encodeMarker(id, true, true);
                    IRBuilder<> B(CI);
                    if (useBareTrace)
                        emitBareTrace(B, fused, M, gen);
                    else
                        insertTraceMarker(B, fused, F, gen);
                    CI->eraseFromParent();
                    NextCI->eraseFromParent();
                    NextCI = nullptr;
                    Changed = true;
                    i++;
                    continue;
                }
            }
        }

        uint32_t encoded = resolveMarkerString(CI, Type);
        if (!encoded)
        {
            if (useBareTrace) continue; // not resolvable yet, leave for late pass
            errs() << "SQTT: warning: sqtt_marker_enter/exit/point/data() "
                      "argument is not a string literal, skipping\n";
            CI->eraseFromParent();
            continue;
        }

        IRBuilder<> B(CI);
        if (Type == MarkerType::Data)
        {
            if (useBareTrace)
            {
                emitBareTrace(B, encoded, M, gen);
                Value* payload = CI->getArgOperand(1);
                if (payload->getType() != llvm::Type::getInt32Ty(CI->getContext()))
                    payload = B.CreateZExtOrTrunc(payload, llvm::Type::getInt32Ty(CI->getContext()));
                emitRawTracePayload(B, payload, M);
            }
            else
                insertTraceMarkerWithPayload(B, encoded, CI->getArgOperand(1), F, gen);
        }
        else if (useBareTrace)
            emitBareTrace(B, encoded, M, gen);
        else
            insertTraceMarker(B, encoded, F, gen);
        CI->eraseFromParent();
        Changed = true;
    }
    return Changed || !useBareTrace;
}
