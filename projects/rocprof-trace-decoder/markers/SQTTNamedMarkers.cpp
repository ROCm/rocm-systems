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

#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace
{

enum MarkerFlag : uint8_t
{
    MarkerEnter = 1,
    MarkerExit = 2,
    MarkerPoint = 4,
    MarkerPayload = 8
};

uint8_t markerFlags(const CallInst* call)
{
    const Function* callee = call ? call->getCalledFunction() : nullptr;
    return callee ? StringSwitch<uint8_t>(callee->getName())
                        .Case("__sqtt_named_marker_enter", MarkerEnter)
                        .Case("__sqtt_named_marker_exit", MarkerExit)
                        .Case("__sqtt_named_marker_point", MarkerPoint)
                        .Case("__sqtt_named_marker_data", MarkerPoint | MarkerPayload)
                        .Default(0)
                  : 0;
}

} // namespace

uint32_t SQTTInstrumentPass::resolveMarkerString(CallInst* CI, uint8_t flags)
{
    Value* Arg = CI->getArgOperand(0)->stripPointerCasts();
    auto* GV = dyn_cast<GlobalVariable>(Arg);
    if (!GV || !GV->hasInitializer()) return 0;
    auto* CDA = dyn_cast<ConstantDataArray>(GV->getInitializer());
    if (!CDA || !CDA->isString()) return 0;

    // Exit just pops the top of the marker stack — the name string is
    // unused at the trace level, so no ID/funcmap entry is needed.
    if (flags & MarkerExit) return FLAG_EXIT_PREV; // value 1: pop top scope

    std::string Name = CDA->getAsString().str();
    if (!Name.empty() && Name.back() == '\0') Name.pop_back();

    bool isPoint = flags & MarkerPoint;
    uint32_t extraPayloadCount = (flags & MarkerPayload) ? 1 : 0;
    std::string key = std::string(isPoint ? "P:" : "U:") + std::to_string(extraPayloadCount) + ":" + Name;
    auto [it, inserted] = UserMarkerMap.try_emplace(key, NextEventID);
    uint32_t id = it->second;
    if (inserted)
    {
        ++NextEventID;
        Markers.push_back(
            {id, isPoint ? MarkerKind::Point : MarkerKind::UserScope, Name, {}, 0, extraPayloadCount}
        );
    }
    bool enter = flags & MarkerEnter;
    return encodeMarker(id, enter, false); // enter or point
}

// Early emits bare traces and leaves unresolved calls for late processing.
// Late emits scoped/bounded traces and warns for unresolved calls.
bool SQTTInstrumentPass::processMarkerCalls(Function& F, GfxGen gen, bool useBareTrace)
{
    SmallVector<CallInst*, 8> Calls;
    for (auto& BB : F)
        for (auto& I : BB)
            if (auto* CI = dyn_cast<CallInst>(&I); markerFlags(CI)) Calls.push_back(CI);
    if (Calls.empty()) return false;

    Module* M = F.getParent();
    bool Changed = false;
    auto emit = [&](IRBuilder<>& B, uint32_t encoded, Value* payload = nullptr)
    {
        if (useBareTrace)
        {
            CallInst* header = emitBareTrace(B, encoded, M, gen);
            if (payload) emitRawTracePayload(B, payload, M, header);
        }
        else
            insertTraceMarker(B, encoded, F, gen, payload);
    };

    for (unsigned i = 0; i < Calls.size(); i++)
    {
        CallInst* CI = Calls[i];
        uint8_t flags = markerFlags(CI);

        // Fuse only directly adjacent exit+enter pairs.  A marker boundary
        // must not absorb work between the calls.
        if (flags == MarkerExit && i + 1 < Calls.size())
        {
            CallInst* NextCI = Calls[i + 1];
            if (markerFlags(NextCI) == MarkerEnter && CI->getNextNode() == NextCI)
            {
                uint32_t enterEncoded = resolveMarkerString(NextCI, MarkerEnter);
                if (enterEncoded)
                {
                    uint32_t id = enterEncoded >> 2;
                    uint32_t fused = encodeMarker(id, true, true);
                    IRBuilder<> B(CI);
                    emit(B, fused);
                    CI->eraseFromParent();
                    NextCI->eraseFromParent();
                    Changed = true;
                    i++;
                    continue;
                }
            }
        }

        uint32_t encoded = resolveMarkerString(CI, flags);
        if (!encoded)
        {
            if (useBareTrace) continue; // not resolvable yet, leave for late pass
            errs() << "SQTT: warning: sqtt_marker_enter/exit/point/data() "
                      "argument is not a string literal, skipping\n";
            CI->eraseFromParent();
            continue;
        }

        IRBuilder<> B(CI);
        emit(B, encoded, flags & MarkerPayload ? CI->getArgOperand(1) : nullptr);
        CI->eraseFromParent();
        Changed = true;
    }
    return Changed || !useBareTrace;
}
