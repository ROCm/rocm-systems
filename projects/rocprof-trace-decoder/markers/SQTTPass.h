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

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"

#include "SQTTConfig.h"
#include "SQTTTarget.h"

// ============================================================================
// The pass (two-phase: Early + Late)
//
// Early phase (before inliner):
//   - Force-inlines wrappers around named marker sentinels, then resolves
//     sentinel calls to bare s_ttracedata intrinsics. This eliminates
//     opaque extern calls before the optimizer runs, avoiding VGPR
//     pressure from conservative register allocation.
//   - Instruments all non-kernel device functions with bare entry/exit
//     markers (no scope checks or barriers). When functions are inlined,
//     their markers travel with the code.
//
// Late phase (after inliner):
//   - Evaluates the cost/size threshold on the now-optimized IR and
//     removes markers from functions that are too small.
//   - Adds scope checks and sched/mem barriers around surviving markers.
//   - Handles any remaining named marker sentinel calls (fallback).
//   - Instruments barriers and emits the .sqtt_funcmap ELF section.
//   - Reassigns compact function IDs to enable s_ttracedata_imm.
// ============================================================================

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
    uint32_t NextEventID = 1; // unified counter for all marker types
    // Instrumented device functions.  SourceLoc is the function definition
    // location (file:line) from DWARF, "" if no debug info.
    struct FuncMapEntry
    {
        uint32_t ID;
        std::string Name;
        std::string SourceLoc;
    };
    std::vector<FuncMapEntry> FuncMap;
    // Kernels (not instrumented, recorded for vaddr lookup).
    struct KernelEntry
    {
        std::string Name;
        std::string SourceLoc;
    };
    std::vector<KernelEntry> KernelNames;
    std::map<std::string, uint32_t> UserMarkerMap;
    // User markers: (id, name, is_point).  Scope markers use U: prefix,
    // point markers use P: prefix in the funcmap.
    struct UserMarkerEntry
    {
        uint32_t ID;
        std::string Name;
        bool IsPoint;
        uint32_t ExtraPayloadCount;
    };
    std::vector<UserMarkerEntry> UserMarkers;
    llvm::Value* CurScopeCheck = nullptr; // cached per-function scope check result
    uint32_t ShaderClockBitsUsed = 0;
    uint32_t ShaderClockShiftUsed = 0;

    // Dynamically allocated IDs for system events
    uint32_t BarrierSignalID = 0;
    uint32_t BarrierWaitID = 0;
    uint32_t BarrierFullID = 0;
    uint32_t VmemLoadID = 0;
    uint32_t VmemStoreID = 0;

    // Per-op unique IDs for address tracing.  Each memory op gets its own ID
    // so the decoder can correlate traces back to individual source locations.
    struct AddrTraceEntry
    {
        uint32_t ID;
        std::string Kind;             // "addr_trace_load", "addr_trace_store", etc.
        std::string SourceLoc;        // "file.hip:42" or "" if no debug info
        uint32_t ExtraPayloadCount;   // following s_ttracedata records after the header
    };
    std::vector<AddrTraceEntry> AddrTraceEntries;
    unsigned AddrTraceWaveSize = 0; // set once during instrumentAddressTraces

    // -----------------------------------------------------------------
    // Phase entry points
    // -----------------------------------------------------------------
    llvm::PreservedAnalyses runEarly(llvm::Module& M);
    llvm::PreservedAnalyses runLate(llvm::Module& M);

    // -----------------------------------------------------------------
    // Marker insertion
    // -----------------------------------------------------------------
    void insertTraceMarker(llvm::IRBuilder<>& B, uint32_t markerID, llvm::Function& F, GfxGen gen);
    void insertTraceMarkerWithPayload(
        llvm::IRBuilder<>& B, uint32_t markerID, llvm::Value* payload, llvm::Function& F, GfxGen gen
    );

    // -----------------------------------------------------------------
    // Scope check
    // -----------------------------------------------------------------
    llvm::Value* buildScopeCheck(llvm::IRBuilder<>& B, GfxGen gen);
    llvm::Value* getOrCreateScopeCheck(llvm::Function& F, GfxGen gen);
    bool wrapExistingMarkers(llvm::Function& F, GfxGen gen);
    // Wraps a contiguous run of marker calls [First..Last] (inclusive, same
    // basic block, separated only by ignorable instructions) in a single
    // scope-check diamond.  Used by wrapExistingMarkers to coalesce adjacent
    // markers that would otherwise produce a chain of identical-condition
    // branches after inlining.
    void wrapRangeWithScopeCheck(llvm::CallInst* First, llvm::CallInst* Last, llvm::Function& F, GfxGen gen);

    // -----------------------------------------------------------------
    // Barriers around existing markers
    // -----------------------------------------------------------------
    bool addBarriersToExistingMarkers(llvm::Function& F);

    // -----------------------------------------------------------------
    // Named marker helpers
    // -----------------------------------------------------------------
    enum class MarkerType
    {
        Enter,
        Exit,
        Point,
        Data
    };
    struct MarkerCall
    {
        llvm::CallInst* CI;
        MarkerType Type;
    };

    llvm::SmallVector<MarkerCall, 8> collectSentinelCalls(llvm::Function& F);
    uint32_t resolveMarkerString(llvm::CallInst* CI, MarkerType type);

    llvm::CallInst* emitBareTrace(llvm::IRBuilder<>& B, uint32_t encoded, llvm::Module* M, GfxGen gen);
    llvm::CallInst* emitBareTraceValue(llvm::IRBuilder<>& B, llvm::Value* val, llvm::Module* M, GfxGen gen);
    llvm::CallInst* emitRawTracePayload(llvm::IRBuilder<>& B, llvm::Value* val, llvm::Module* M);
    static void markMarkerHeaderTrace(llvm::CallInst* CI);
    static bool isMarkerHeaderTrace(llvm::CallInst* CI);
    static void markRawPayloadTrace(llvm::CallInst* CI);
    static bool isRawPayloadTrace(llvm::CallInst* CI);

    bool resolveNamedMarkersEarly(llvm::Function& F, GfxGen gen);
    bool processNamedMarkers(llvm::Function& F, GfxGen gen);
    bool processMarkerCalls(llvm::Function& F, GfxGen gen, bool useBareTrace);

    // -----------------------------------------------------------------
    // Barrier auto-instrumentation
    // -----------------------------------------------------------------
    enum class BarrierKind
    {
        Signal,
        Wait,
        Full,
        None
    };
    static BarrierKind classifyBarrier(llvm::CallInst* CI);
    bool instrumentBarriers(llvm::Function& F, GfxGen gen);

    // -----------------------------------------------------------------
    // Memory operation auto-instrumentation
    // -----------------------------------------------------------------
    enum class MemOpKind
    {
        Load,
        Store,
        None
    };
    static MemOpKind classifyMemOp(llvm::Instruction* I);
    bool instrumentMemoryOps(llvm::Function& F, GfxGen gen);

    // -----------------------------------------------------------------
    // Address trace instrumentation
    // -----------------------------------------------------------------
    enum class AddrTraceKind
    {
        Memory,
        LDS,
        Buffer,
        Permute,
        None
    };
    static AddrTraceKind classifyAddrTraceOp(llvm::Instruction* I, bool traceMemory, bool traceLDS);
    static llvm::Value* getMemOpPointer(llvm::Instruction* I);
    void emitAddressTrace(
        llvm::IRBuilder<>& B,
        llvm::Instruction* memOp,
        AddrTraceKind kind,
        uint32_t headerID,
        llvm::Function& F,
        GfxGen gen
    );
    // One ordering boundary on each side of a complete address-trace block.
    // This deliberately does not put a boundary around every raw payload.
    void emitTraceBlockBoundary(llvm::IRBuilder<>& B, bool after);
    // Source location for an instruction.  Walks the inline chain via
    // DILocation::getInlinedAt() and joins entries with " -> " (innermost
    // first, then each outward call site).  Matches the format used by
    // rocprofiler-sdk's codeobj DWARF inline chain printer.
    // Returns "" if no debug info.
    static std::string getSourceLoc(llvm::Instruction* I);
    // Source location for a function's definition (from DISubprogram).
    // No inline chain — returns just "file:line", or "" if no debug info.
    static std::string getFunctionSourceLoc(llvm::Function& F);
    static const char* addrTraceKindName(AddrTraceKind kind, bool isStore, bool isAtomic = false);

    // Buffer intrinsic operand extraction
    struct BufferOperands
    {
        llvm::Value* Rsrc;
        llvm::Value* VOffset;
        llvm::Value* SOffset;
        llvm::Value* VIndex; // null for raw buffers
        bool IsStruct;
    };
    static BufferOperands getBufferOperands(llvm::CallInst* CI);

    void emitBufferTrace(llvm::IRBuilder<>& B, llvm::CallInst* bufOp, uint32_t headerID, llvm::Function& F, GfxGen gen);
    void emitPermuteTrace(
        llvm::IRBuilder<>& B, llvm::CallInst* permuteOp, uint32_t headerID, llvm::Function& F, GfxGen gen
    );

    bool instrumentAddressTraces(llvm::Function& F, GfxGen gen);

    // -----------------------------------------------------------------
    // Early phase helpers
    // -----------------------------------------------------------------
    void insertBareMarkers(llvm::Function& F, uint32_t id, GfxGen gen);
    void storeFuncMetadata(llvm::Function& F, uint32_t id, llvm::LLVMContext& Ctx);
    void storeUserMarkerMetadata(llvm::Module& M, llvm::LLVMContext& Ctx);
    void recoverUserMarkerMetadata(llvm::Module& M);

    // -----------------------------------------------------------------
    // Late phase: threshold filter and ID compaction
    // -----------------------------------------------------------------
    bool filterInstrumentedFunctions(llvm::Module& M);
    uint32_t compactFuncIDs(llvm::Module& M);
    void rewriteMarkerIDs(llvm::Function& F, const std::map<uint32_t, uint32_t>& IDMap, GfxGen gen);
    bool applyShaderClockPacking(llvm::Function& F, GfxGen gen);
    // Lower full traces after all marker transformations to explicit asm with
    // the required M0 hazard spacing.
    bool lowerFullTracesWithM0Nop(llvm::Function& F);
    void removeFuncMarkersFromModule(llvm::Module& M, uint32_t id);
    void removeAdjacentBarriers(llvm::CallInst* CI);

    // -----------------------------------------------------------------
    // -O0 fallback: direct function instrumentation
    // -----------------------------------------------------------------
    static bool hasMustTailCall(const llvm::Function& F);
    bool instrumentFunctionDirect(llvm::Function& F, GfxGen gen);

    // -----------------------------------------------------------------
    // Emit .sqtt_funcmap ELF section
    // -----------------------------------------------------------------
    void emitFuncMap(llvm::Module& M);
};
