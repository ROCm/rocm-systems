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

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

#include "SQTTConfig.h"
#include "SQTTTarget.h"

// Early resolves named calls and inserts bare function markers before inlining.
// Late filters/compacts them, adds guards and boundaries, then emits the funcmap.

class SQTTInstrumentPass : public llvm::PassInfoMixin<SQTTInstrumentPass>
{
public:
    enum class Mode
    {
        Early,
        Late
    };

    SQTTInstrumentPass(SQTTConfig Cfg, Mode M) : Config(Cfg), PassMode(M) {}

    llvm::PreservedAnalyses run(llvm::Module& M, llvm::ModuleAnalysisManager& MAM);

private:
    SQTTConfig Config;
    Mode PassMode;
    uint32_t NextEventID = 1;
    // One ledger backs both the early-pass handoff and the final funcmap.
    // Function and kernel definition locations are "file:line" when available.
    enum class MarkerKind : uint8_t
    {
        Function,
        Kernel,
        UserScope,
        Point,
        SystemPoint,
        AddressPoint
    };
    struct MarkerRecord
    {
        uint32_t ID = 0; // Kernels have no shaderdata ID.
        MarkerKind Kind;
        std::string Name;
        std::string SourceLoc;
        uint32_t PreOptSize = 0;       // Function rows only, before inlining.
        uint32_t ExtraPayloadCount = 0; // Header rows only.
    };
    std::vector<MarkerRecord> Markers;
    // Name-to-ID is rebuilt by compaction along with the ledger IDs.
    std::map<std::string, uint32_t> UserMarkerMap;
    llvm::Value* CurScopeCheck = nullptr; // cached per-function scope check result
    uint32_t ShaderClockBitsUsed = 0;

    // Automatic marker groups are allocated in their classifier enum order.
    uint32_t FirstBarrierID = 0;
    uint32_t FirstVmemID = 0;

    unsigned AddrTraceWaveSize = 0; // set once during instrumentAddressTraces

    // Phase entry points.
    llvm::PreservedAnalyses runEarly(llvm::Module& M);
    llvm::PreservedAnalyses runLate(llvm::Module& M);

    // Marker insertion and scope filtering.
    void insertTraceMarker(
        llvm::IRBuilder<>& B, uint32_t markerID, llvm::Function& F, GfxGen gen, llvm::Value* payload = nullptr
    );

    llvm::Value* buildScopeCheck(llvm::IRBuilder<>& B, GfxGen gen);
    llvm::Value* getOrCreateScopeCheck(llvm::Function& F, GfxGen gen);
    bool finalizeExistingMarkers(llvm::Function& F, GfxGen gen);
    // Wrap an adjacent marker run in one scope-check diamond.
    void wrapRangeWithScopeCheck(
        llvm::CallInst* First, llvm::CallInst* Last, llvm::Function& F, GfxGen gen, bool pinSkipHead
    );

    uint32_t resolveMarkerString(llvm::CallInst* CI, uint8_t flags);

    llvm::CallInst* emitBareTrace(llvm::IRBuilder<>& B, uint32_t encoded, llvm::Module* M, GfxGen gen);
    llvm::CallInst* emitBareTraceValue(llvm::IRBuilder<>& B, llvm::Value* val, llvm::Module* M);
    llvm::CallInst* emitRawTracePayload(
        llvm::IRBuilder<>& B, llvm::Value* val, llvm::Module* M, llvm::CallInst* header
    );
    static bool isTraceDataCall(const llvm::CallInst* CI);

    bool processMarkerCalls(llvm::Function& F, GfxGen gen, bool useBareTrace);

    // Automatic barrier and memory instrumentation.
    enum class BarrierKind : uint32_t
    {
        Signal = 0,
        Wait,
        Full,
        None
    };
    static BarrierKind classifyBarrier(llvm::CallInst* CI);
    static bool isSyncInstruction(llvm::Instruction* I);
    bool instrumentBarriers(llvm::Function& F, GfxGen gen);

    enum class MemOpKind : uint32_t
    {
        Load = 0,
        Store,
        None
    };
    static MemOpKind classifyMemOp(llvm::Instruction* I);
    bool instrumentMemoryOps(llvm::Function& F, GfxGen gen);

    // Address tracing.
    enum class AddrTraceKind
    {
        Memory,
        LDS,
        Buffer,
        Permute,
        None
    };
    // Everything needed after the initial scan.  Keeping the protocol shape
    // here avoids rediscovering buffer spelling and operand layout while the
    // CFG is being rewritten.
    struct AddrTraceOp
    {
        llvm::Instruction* I;
        const char* Name;
        AddrTraceKind Kind;
        unsigned BufferRsrcIndex;
        bool StructBuffer;
    };
    static AddrTraceOp classifyAddrTraceOp(llvm::Instruction* I, bool traceMemory, bool traceLDS);
    void emitAddressTrace(
        llvm::IRBuilder<>& B,
        const AddrTraceOp& op,
        uint32_t headerID,
        GfxGen gen
    );
    void emitReadlaneTraceLoop(
        llvm::IRBuilder<>& B,
        llvm::Value* firstValue,
        llvm::Value* secondValue,
        unsigned waveSize
    );
    void emitTraceBoundary(llvm::IRBuilder<>& B, bool after, bool schedBarrier = true);
    void emitTraceBoundaries(
        llvm::IRBuilder<>& B, llvm::Instruction* first, llvm::Instruction* last, bool schedBarrier
    );
    // Emit a sequence directly or inside the configured scope-check diamond.
    void emitScopedTrace(
        llvm::IRBuilder<>& B,
        llvm::Function& F,
        GfxGen gen,
        const char* traceBlockName,
        const char* skipBlockName,
        bool pinSkipHead,
        llvm::function_ref<void(llvm::IRBuilder<>&)> emit
    );
    // Return the innermost-to-outermost inline chain, or "" without debug info.
    static std::string getSourceLoc(llvm::Instruction* I);
    static std::string getFunctionSourceLoc(llvm::Function& F);

    bool instrumentAddressTraces(llvm::Function& F, GfxGen gen);

    // Insert function entry/exit markers for either pass phase.
    void insertFunctionMarkers(llvm::Function& F, uint32_t id, GfxGen gen, bool useBareTrace);
    void storeEarlyMarkerMetadata(llvm::Module& M, llvm::LLVMContext& Ctx);
    bool recoverEarlyMarkerMetadata(llvm::Module& M, bool& hasEarlyFunctions);

    // Late phase: filtering, ID compaction, packing, and lowering.
    bool finalizeEarlyFunctionMarkers(llvm::Module& M);
    // Pack gfx12 shader-clock headers and lower full traces through M0/NOP.
    bool finalizeFullTraces(llvm::Function& F, GfxGen gen);

    // -O0 fallback.
    static bool hasMustTailCall(const llvm::Function& F);
    bool instrumentFunctionDirect(llvm::Function& F, GfxGen gen);

    void emitFuncMap(llvm::Module& M);
};
