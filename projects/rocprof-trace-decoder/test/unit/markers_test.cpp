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

#include <gtest/gtest.h>

#include "SQTTConfig.h"
#include "SQTTPass.h"
#include "SQTTTarget.h"
#include "rocprof_trace_decoder/cxx/markers.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace
{

class ScopedEnv
{
public:
    ScopedEnv(std::string name, std::optional<std::string> value) : Name(std::move(name))
    {
        if (const char* old = std::getenv(Name.c_str()))
        {
            HadOldValue = true;
            OldValue = old;
        }

        if (value)
            setenv(Name.c_str(), value->c_str(), 1);
        else
            unsetenv(Name.c_str());
    }

    ~ScopedEnv()
    {
        if (HadOldValue)
            setenv(Name.c_str(), OldValue.c_str(), 1);
        else
            unsetenv(Name.c_str());
    }

    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    std::string Name;
    bool HadOldValue = false;
    std::string OldValue;
};

std::vector<std::unique_ptr<ScopedEnv>> clearSqttEnvironment()
{
    std::vector<std::unique_ptr<ScopedEnv>> env;
    for (const char* name :
         {"SQTT_INSTRUMENT_BARRIERS",
          "SQTT_MEM_BARRIER",
          "SQTT_SCOPE_WAVE",
          "SQTT_SCOPE_SIMD",
          "SQTT_SCOPE_CU",
          "SQTT_SCOPE_WG",
          "SQTT_SHADER_CLOCK_BITS",
          "SQTT_SHADER_CLOCK_SHIFT",
          "SQTT_INSTRUMENT_FUNCTIONS",
          "SQTT_INSTRUMENT_MEMORY",
          "SQTT_TRACE_ADDRESSES"})
    {
        env.push_back(std::make_unique<ScopedEnv>(name, std::nullopt));
    }
    return env;
}

std::unique_ptr<Module> makeModule(LLVMContext& ctx)
{
    auto module = std::make_unique<Module>("markers-unit", ctx);
    module->setTargetTriple(Triple("amdgcn-amd-amdhsa"));
    return module;
}

Function* makeFunction(Module& module, StringRef name, StringRef cpu, FunctionType* type)
{
    Function* function = Function::Create(type, GlobalValue::ExternalLinkage, name, module);
    function->addFnAttr("target-cpu", cpu);
    return function;
}

Function* makeVoidFunction(Module& module, StringRef name, StringRef cpu)
{
    LLVMContext& ctx = module.getContext();
    Function* function = makeFunction(module, name, cpu, FunctionType::get(Type::getVoidTy(ctx), false));
    BasicBlock* entry = BasicBlock::Create(ctx, "entry", function);
    IRBuilder<> builder(entry);
    builder.CreateRetVoid();
    return function;
}

void useFullScopeMasks(SQTTConfig& config)
{
    config.WaveMask = FULL_WAVE_MASK;
    config.SimdMask = FULL_SIMD_MASK;
    config.CuMask = FULL_CU_MASK;
    config.WgMask = FULL_WG_MASK;
    config.MemBarrier = MemBarrierMode::None;
}

CallInst* insertTraceCallBefore(Instruction* insertPt, uint32_t encoded, bool passHeader = false)
{
    Module* module = insertPt->getModule();
    LLVMContext& ctx = module->getContext();
    IRBuilder<> builder(insertPt);
    Function* trace = Intrinsic::getOrInsertDeclaration(module, Intrinsic::amdgcn_s_ttracedata);
    CallInst* call = builder.CreateCall(trace, {ConstantInt::get(Type::getInt32Ty(ctx), encoded)});
    if (passHeader) call->setMetadata("sqtt.marker_header", MDNode::get(ctx, {}));
    return call;
}

Function* makeNamedMarkerSentinel(Module& module, StringRef name)
{
    LLVMContext& ctx = module.getContext();
    auto* type = FunctionType::get(Type::getVoidTy(ctx), {PointerType::get(ctx, 0)}, false);
    return Function::Create(type, GlobalValue::ExternalLinkage, name, module);
}

GlobalVariable* makeMarkerString(Module& module, StringRef value)
{
    LLVMContext& ctx = module.getContext();
    auto* initializer = ConstantDataArray::getString(ctx, value, true);
    return new GlobalVariable(
        module,
        initializer->getType(),
        true,
        GlobalValue::PrivateLinkage,
        initializer,
        ".sqtt.marker.string"
    );
}

void addEarlyFunctionMetadata(Function& function, uint32_t id, unsigned preOptSize, StringRef sourceLoc)
{
    Module* module = function.getParent();
    LLVMContext& ctx = module->getContext();
    Type* i32 = Type::getInt32Ty(ctx);
    MDNode* idNode = MDNode::get(ctx, {ConstantAsMetadata::get(ConstantInt::get(i32, id))});
    function.setMetadata("sqtt.func.id", idNode);

    NamedMDNode* earlyMap = module->getOrInsertNamedMetadata("sqtt.funcmap.early");
    earlyMap->addOperand(MDNode::get(
        ctx,
        {ConstantAsMetadata::get(ConstantInt::get(i32, id)),
         MDString::get(ctx, function.getName()),
         ConstantAsMetadata::get(ConstantInt::get(i32, preOptSize)),
         MDString::get(ctx, sourceLoc)}
    ));
}

void addEarlyFunctionMapEntry(Module& module, uint32_t id, StringRef name, unsigned preOptSize, StringRef sourceLoc)
{
    LLVMContext& ctx = module.getContext();
    Type* i32 = Type::getInt32Ty(ctx);
    NamedMDNode* earlyMap = module.getOrInsertNamedMetadata("sqtt.funcmap.early");
    earlyMap->addOperand(MDNode::get(
        ctx,
        {ConstantAsMetadata::get(ConstantInt::get(i32, id)),
         MDString::get(ctx, name),
         ConstantAsMetadata::get(ConstantInt::get(i32, preOptSize)),
         MDString::get(ctx, sourceLoc)}
    ));
}

std::string getFuncMap(const Module& module)
{
    for (const GlobalVariable& global : module.globals())
    {
        if (global.getSection() != ".sqtt_funcmap" || !global.hasInitializer()) continue;
        if (auto* data = dyn_cast<ConstantDataArray>(global.getInitializer()))
        {
            if (data->isString()) return data->getAsCString().str();
        }
    }
    return {};
}

std::string printModule(const Module& module)
{
    std::string text;
    raw_string_ostream os(text);
    module.print(os, nullptr);
    return os.str();
}

size_t countIntrinsicCalls(const Module& module, Intrinsic::ID id)
{
    size_t count = 0;
    for (const Function& function : module)
    {
        for (const BasicBlock& block : function)
        {
            for (const Instruction& inst : block)
            {
                auto* call = dyn_cast<CallInst>(&inst);
                if (!call) continue;
                Function* callee = call->getCalledFunction();
                if (callee && callee->getIntrinsicID() == id) ++count;
            }
        }
    }
    return count;
}

size_t countIntrinsicCalls(const Function& function, Intrinsic::ID id)
{
    size_t count = 0;
    for (const BasicBlock& block : function)
    {
        for (const Instruction& inst : block)
        {
            auto* call = dyn_cast<CallInst>(&inst);
            if (!call) continue;
            Function* callee = call->getCalledFunction();
            if (callee && callee->getIntrinsicID() == id) ++count;
        }
    }
    return count;
}

size_t countFences(const Function& function)
{
    size_t count = 0;
    for (const BasicBlock& block : function)
        for (const Instruction& inst : block)
            if (isa<FenceInst>(inst)) ++count;
    return count;
}

std::optional<unsigned> earlyFunctionPreOptSize(const Module& module, StringRef name)
{
    const NamedMDNode* entries = module.getNamedMetadata("sqtt.funcmap.early");
    if (!entries) return std::nullopt;
    for (const MDNode* entry : entries->operands())
    {
        if (entry->getNumOperands() < 3) continue;
        auto* entryName = dyn_cast<MDString>(entry->getOperand(1));
        auto* size = mdconst::dyn_extract<ConstantInt>(entry->getOperand(2));
        if (entryName && size && entryName->getString() == name) return size->getZExtValue();
    }
    return std::nullopt;
}

std::vector<uint32_t> traceMarkerValues(const Function& function)
{
    std::vector<uint32_t> values;
    for (const BasicBlock& block : function)
    {
        for (const Instruction& inst : block)
        {
            auto* call = dyn_cast<CallInst>(&inst);
            if (!call) continue;
            Function* callee = call->getCalledFunction();
            if (!callee) continue;
            auto id = callee->getIntrinsicID();
            if (id != Intrinsic::amdgcn_s_ttracedata && id != Intrinsic::amdgcn_s_ttracedata_imm) continue;
            auto* arg = dyn_cast<ConstantInt>(call->getArgOperand(0));
            if (arg) values.push_back(arg->getZExtValue());
        }
    }
    return values;
}

const CallInst* findM0NopTrace(const Function& function)
{
    constexpr const char TraceAsm[] =
        "s_mov_b32 m0, $1\n"
        "s_nop 0\n"
        "s_ttracedata";
    for (const BasicBlock& block : function)
    {
        for (const Instruction& inst : block)
        {
            auto* call = dyn_cast<CallInst>(&inst);
            if (!call) continue;
            auto* asmCall = dyn_cast<InlineAsm>(call->getCalledOperand());
            if (asmCall && asmCall->hasSideEffects() && asmCall->getAsmString() == TraceAsm) return call;
        }
    }
    return nullptr;
}

bool hasPtrToIntFromAddressSpace(const Module& module, unsigned addressSpace, unsigned resultBits)
{
    for (const Function& function : module)
    {
        for (const BasicBlock& block : function)
        {
            for (const Instruction& inst : block)
            {
                if (inst.getOpcode() != Instruction::PtrToInt) continue;
                if (inst.getType()->getIntegerBitWidth() != resultBits) continue;
                if (inst.getOperand(0)->getType()->getPointerAddressSpace() == addressSpace) return true;
            }
        }
    }
    return false;
}

void addExistingLlvmUsed(Module& module)
{
    LLVMContext& ctx = module.getContext();
    Type* i32 = Type::getInt32Ty(ctx);
    auto* dummy = new GlobalVariable(
        module, i32, false, GlobalValue::InternalLinkage, ConstantInt::get(i32, 0), "existing_used_global"
    );
    Constant* dummyPtr = ConstantExpr::getPointerBitCastOrAddrSpaceCast(dummy, PointerType::getUnqual(ctx));
    ArrayType* usedTy = ArrayType::get(PointerType::getUnqual(ctx), 1);
    auto* used = new GlobalVariable(
        module, usedTy, false, GlobalValue::AppendingLinkage, ConstantArray::get(usedTy, {dummyPtr}), "llvm.used"
    );
    used->setSection("llvm.metadata");
}

unsigned llvmUsedOperandCount(const Module& module)
{
    const GlobalVariable* used = module.getGlobalVariable("llvm.used");
    if (!used || !used->hasInitializer()) return 0;
    auto* values = dyn_cast<ConstantArray>(used->getInitializer());
    return values ? values->getNumOperands() : 0;
}

void expectContains(const std::string& text, StringRef needle)
{
    EXPECT_NE(text.find(needle.str()), std::string::npos) << "missing: " << needle.str();
}

void expectNotContains(const std::string& text, StringRef needle)
{
    EXPECT_EQ(text.find(needle.str()), std::string::npos) << "unexpected: " << needle.str();
}

std::optional<unsigned> pointEntryId(const std::string& funcMap, StringRef name)
{
    SmallVector<StringRef, 32> lines;
    StringRef(funcMap).split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef line : lines)
    {
        line = line.rtrim("\r");
        if (!line.consume_front("P:")) continue;

        auto [idText, rest] = line.split(':');
        unsigned id = 0;
        if (idText.getAsInteger(10, id)) continue;

        auto [entryName, sourceLoc] = rest.split('@');
        (void) sourceLoc;
        if (entryName == name) return id;
    }
    return std::nullopt;
}

size_t countPointEntries(const std::string& funcMap, StringRef name)
{
    size_t count = 0;
    SmallVector<StringRef, 32> lines;
    StringRef(funcMap).split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef line : lines)
    {
        line = line.rtrim("\r");
        if (!line.consume_front("P:")) continue;
        auto [idText, rest] = line.split(':');
        unsigned id = 0;
        if (idText.getAsInteger(10, id)) continue;
        auto [entryName, sourceLoc] = rest.split('@');
        (void) sourceLoc;
        if (entryName == name) ++count;
    }
    return count;
}

std::optional<unsigned> extraPayloadCountForId(const std::string& funcMap, unsigned markerId)
{
    SmallVector<StringRef, 32> lines;
    StringRef(funcMap).split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef line : lines)
    {
        line = line.rtrim("\r");
        if (!line.consume_front("R:")) continue;

        auto [idText, metadata] = line.split(':');
        unsigned id = 0;
        if (idText.getAsInteger(10, id) || id != markerId) continue;

        SmallVector<StringRef, 4> fields;
        metadata.split(fields, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
        for (StringRef field : fields)
        {
            if (!field.consume_front("extra_payload_count=")) continue;
            unsigned count = 0;
            if (!field.getAsInteger(10, count)) return count;
        }
    }
    return std::nullopt;
}

void expectPointEntryWithPayload(const std::string& funcMap, StringRef name, unsigned expectedPayloadCount)
{
    std::optional<unsigned> id = pointEntryId(funcMap, name);
    ASSERT_TRUE(id.has_value()) << "missing point funcmap entry: " << name.str();

    std::optional<unsigned> payloadCount = extraPayloadCountForId(funcMap, *id);
    ASSERT_TRUE(payloadCount.has_value()) << "missing payload metadata for funcmap entry: " << name.str();
    EXPECT_EQ(*payloadCount, expectedPayloadCount) << "wrong payload metadata for funcmap entry: " << name.str();
}

} // namespace

TEST(MarkerPublicHeader, HostScopeConfigParsesMasks)
{
    ScopedEnv wave("SQTT_SCOPE_WAVE", "0x5");
    ScopedEnv simd("SQTT_SCOPE_SIMD", "-1");
    ScopedEnv cu("SQTT_SCOPE_CU", "bad");
    ScopedEnv wg("SQTT_SCOPE_WG", std::nullopt);

    EXPECT_EQ(sqtt::parse_env_mask("SQTT_SCOPE_WAVE", 0), 0x5u);
    EXPECT_EQ(sqtt::parse_env_mask("SQTT_SCOPE_SIMD", 0), 0xFFFFFFFFu);
    EXPECT_EQ(sqtt::parse_env_mask("SQTT_SCOPE_CU", 0x3), 0x3u);
    EXPECT_EQ(sqtt::parse_env_mask("SQTT_SCOPE_WG", 0x9), 0x9u);

    sqtt::ScopeConfig config = sqtt::ScopeConfig::from_env();
    EXPECT_EQ(config.wave_mask, 0x5u);
    EXPECT_EQ(config.simd_mask, 0xFFFFFFFFu);
    EXPECT_EQ(config.cu_mask, 0x3u);
    EXPECT_EQ(config.wg_mask, 0xFFFFFFFFu);
}

TEST(MarkerConfig, ParsesEnvironmentAndRejectsConflictingModes)
{
    auto cleanEnv = clearSqttEnvironment();
    ScopedEnv barriers("SQTT_INSTRUMENT_BARRIERS", "YES");
    ScopedEnv memBarrier("SQTT_MEM_BARRIER", "clobber");
    ScopedEnv functions("SQTT_INSTRUMENT_FUNCTIONS", "cost:42");
    ScopedEnv memory("SQTT_INSTRUMENT_MEMORY", "4:7");
    ScopedEnv addrs("SQTT_TRACE_ADDRESSES", "memory, lds, bogus");
    ScopedEnv shaderBits("SQTT_SHADER_CLOCK_BITS", "not-a-number");
    ScopedEnv shaderShift("SQTT_SHADER_CLOCK_SHIFT", "8");
    ScopedEnv scopeWave("SQTT_SCOPE_WAVE", "not-a-mask");
    ScopedEnv scopeSimd("SQTT_SCOPE_SIMD", "0x5");
    ScopedEnv scopeCu("SQTT_SCOPE_CU", "-1");

    SQTTConfig config = SQTTConfig::fromEnvironment();

    EXPECT_TRUE(config.InstrumentBarriers);
    EXPECT_EQ(config.MemBarrier, MemBarrierMode::AsmClobber);
    EXPECT_EQ(config.Mode, CostMode::WeightedCost);
    EXPECT_EQ(config.FunctionThreshold, 42u);
    EXPECT_TRUE(config.InstrumentMemory);
    EXPECT_EQ(config.MemoryChunkSize, 4u);
    EXPECT_EQ(config.MemoryMaxGap, 7u);
    EXPECT_FALSE(config.TraceMemoryAddrs);
    EXPECT_FALSE(config.TraceLDSAddrs);
    EXPECT_EQ(config.ShaderClockBits, SQTTConfig::AutoShaderClockBits);
    EXPECT_EQ(config.ShaderClockShift, 8u);
    EXPECT_EQ(config.WaveMask, 0xFFFFFFFFu);
    EXPECT_EQ(config.SimdMask, 0x5u);
    EXPECT_EQ(config.CuMask, 0xFFFFFFFFu);

    ScopedEnv invalidMemory("SQTT_INSTRUMENT_MEMORY", "4");
    ScopedEnv traceOnly("SQTT_TRACE_ADDRESSES", "lds");
    ScopedEnv invalidMemBarrier("SQTT_MEM_BARRIER", "bad-mode");
    config = SQTTConfig::fromEnvironment();
    EXPECT_EQ(config.MemBarrier, MemBarrierMode::Fence);
    EXPECT_FALSE(config.InstrumentMemory);
    EXPECT_TRUE(config.TraceLDSAddrs);
    EXPECT_FALSE(config.TraceMemoryAddrs);
}

TEST(MarkerTarget, ClassifiesArchitecturesAndInstructionCosts)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* i32 = Type::getInt32Ty(ctx);

    EXPECT_EQ(getGfxGen(*makeVoidFunction(*module, "gfx9_func", "gfx90a")), GfxGen::GFX9);
    EXPECT_EQ(getGfxGen(*makeVoidFunction(*module, "gfx10_func", "gfx1030")), GfxGen::RDNA);
    EXPECT_EQ(getGfxGen(*makeVoidFunction(*module, "gfx11_func", "gfx1100")), GfxGen::RDNA);
    EXPECT_EQ(getGfxGen(*makeVoidFunction(*module, "gfx12_func", "gfx1200")), GfxGen::GFX12);
    EXPECT_EQ(getGfxGen(*makeVoidFunction(*module, "unknown_func", "notgfx")), GfxGen::Unknown);

    EXPECT_EQ(getWaveSize(GfxGen::GFX9), 64u);
    EXPECT_EQ(getWaveSize(GfxGen::RDNA), 32u);
    EXPECT_FALSE(supportsImmTrace(GfxGen::GFX9));
    EXPECT_TRUE(supportsImmTrace(GfxGen::GFX12));

    SQTTConfig config;
    EXPECT_EQ(getShaderClockBits(config, GfxGen::GFX12), 12u);
    EXPECT_EQ(getShaderClockBits(config, GfxGen::RDNA), 0u);
    config.ShaderClockBits = 5;
    EXPECT_EQ(getShaderClockBits(config, GfxGen::RDNA), 5u);
    EXPECT_TRUE(usesShaderClockPacking(config, GfxGen::GFX12));
    EXPECT_FALSE(usesShaderClockPacking(config, GfxGen::RDNA));

    Function* costed =
        makeFunction(*module, "costed", "gfx1100", FunctionType::get(Type::getVoidTy(ctx), {i32}, false));
    BasicBlock* entry = BasicBlock::Create(ctx, "entry", costed);
    IRBuilder<> builder(entry);
    builder.CreateAlloca(i32);
    Value* loaded = builder.CreateLoad(i32, UndefValue::get(PointerType::get(ctx, 1)));
    builder.CreateStore(loaded, UndefValue::get(PointerType::get(ctx, 3)));
    Function* mfma = Function::Create(
        FunctionType::get(i32, false), GlobalValue::ExternalLinkage, "llvm.amdgcn.mfma.unit", module.get()
    );
    builder.CreateCall(mfma);
    builder.CreateRetVoid();

    EXPECT_EQ(computeFunctionSize(*costed, CostMode::InstructionCount), 4u);
    EXPECT_EQ(computeFunctionSize(*costed, CostMode::WeightedCost), 31u);
}

TEST(MarkerPass, AddressTracingHandlesBuffersAndPermutes)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* i32 = Type::getInt32Ty(ctx);
    Type* i16 = Type::getInt16Ty(ctx);
    Type* i64 = Type::getInt64Ty(ctx);
    Type* voidTy = Type::getVoidTy(ctx);
    auto* rsrcVecTy = FixedVectorType::get(i32, 4);
    auto* bufferPtrTy = PointerType::get(ctx, 8);

    Function* function =
        makeFunction(*module, "buffer_traces", "gfx1100", FunctionType::get(voidTy, {bufferPtrTy}, false));
    BasicBlock* entry = BasicBlock::Create(ctx, "entry", function);
    IRBuilder<> builder(entry);
    Value* rsrcVec = ConstantAggregateZero::get(rsrcVecTy);
    Value* rsrcPtr = function->getArg(0);

    Function* rawLoad = Function::Create(
        FunctionType::get(i32, {rsrcVecTy, i64, i16}, false),
        GlobalValue::ExternalLinkage,
        "llvm.amdgcn.raw.buffer.load.unit",
        module.get()
    );
    builder.CreateCall(rawLoad, {rsrcVec, ConstantInt::get(i64, 11), ConstantInt::get(i16, 3)});

    Function* structStore = Function::Create(
        FunctionType::get(voidTy, {i32, rsrcVecTy, i16, i16, i64}, false),
        GlobalValue::ExternalLinkage,
        "llvm.amdgcn.struct.buffer.store.unit",
        module.get()
    );
    builder.CreateCall(
        structStore,
        {ConstantInt::get(i32, 17),
         rsrcVec,
         ConstantInt::get(i16, 5),
         ConstantInt::get(i16, 7),
         ConstantInt::get(i64, 9)}
    );

    Function* rawPtrCmpSwap = Function::Create(
        FunctionType::get(i32, {i32, i32, bufferPtrTy, i16, i16}, false),
        GlobalValue::ExternalLinkage,
        "llvm.amdgcn.raw.ptr.buffer.atomic.cmpswap.unit",
        module.get()
    );
    builder.CreateCall(
        rawPtrCmpSwap,
        {ConstantInt::get(i32, 1),
         ConstantInt::get(i32, 2),
         rsrcPtr,
         ConstantInt::get(i16, 4),
         ConstantInt::get(i16, 6)}
    );

    FunctionCallee bpermute = Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_ds_bpermute);
    builder.CreateCall(bpermute, {ConstantInt::get(i32, 16), ConstantInt::get(i32, 33)});
    builder.CreateRetVoid();

    SQTTConfig config;
    useFullScopeMasks(config);
    config.TraceMemoryAddrs = true;
    config.TraceLDSAddrs = true;

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    std::string funcMap = getFuncMap(*module);
    expectContains(funcMap, "W:32");
    expectPointEntryWithPayload(funcMap, "addr_trace_buffer_load", 37);
    expectPointEntryWithPayload(funcMap, "addr_trace_struct_buffer_store", 69);
    expectPointEntryWithPayload(funcMap, "addr_trace_buffer_atomic", 37);
    expectPointEntryWithPayload(funcMap, "addr_trace_ds_bpermute", 34);

    std::string ir = printModule(*module);
    expectContains(ir, "sqtt.buf.loop");
    expectContains(ir, "sqtt.perm.loop");
    EXPECT_TRUE(hasPtrToIntFromAddressSpace(*module, 8, 128));
    EXPECT_EQ(countIntrinsicCalls(*module, Intrinsic::amdgcn_readlane), 5u);
}

TEST(MarkerPass, AddressTracingOnlyUsesFlatAndGlobalAddressSpaces)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* i32 = Type::getInt32Ty(ctx);
    Type* voidTy = Type::getVoidTy(ctx);
    FunctionType* type = FunctionType::get(
        voidTy,
        {PointerType::get(ctx, 0),
         PointerType::get(ctx, 1),
         PointerType::get(ctx, 2),
         PointerType::get(ctx, 4),
         PointerType::get(ctx, 5)},
        false
    );
    Function* function = makeFunction(*module, "address_spaces", "gfx1100", type);
    BasicBlock* entry = BasicBlock::Create(ctx, "entry", function);
    IRBuilder<> builder(entry);
    builder.CreateLoad(i32, function->getArg(0));
    builder.CreateStore(ConstantInt::get(i32, 1), function->getArg(1));
    builder.CreateLoad(i32, function->getArg(2));
    builder.CreateLoad(i32, function->getArg(3));
    builder.CreateLoad(i32, function->getArg(4));
    builder.CreateRetVoid();

    SQTTConfig config;
    useFullScopeMasks(config);
    config.TraceMemoryAddrs = true;

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    const std::string funcMap = getFuncMap(*module);
    EXPECT_EQ(countPointEntries(funcMap, "addr_trace_load"), 1u);
    EXPECT_EQ(countPointEntries(funcMap, "addr_trace_store"), 1u);
    EXPECT_EQ(countPointEntries(funcMap, "addr_trace_atomic"), 0u);
}

TEST(MarkerPass, AddressTracingSkipsBufferLoadLds)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* voidTy = Type::getVoidTy(ctx);
    Function* function = makeVoidFunction(*module, "buffer_load_lds", "gfx1200");
    IRBuilder<> builder(function->getEntryBlock().getTerminator());
    Function* bufferLoadLds = Function::Create(
        FunctionType::get(voidTy, false),
        GlobalValue::ExternalLinkage,
        "llvm.amdgcn.raw.buffer.load.lds",
        module.get()
    );
    builder.CreateCall(bufferLoadLds);
    Function* point = makeNamedMarkerSentinel(*module, "__sqtt_named_marker_point");
    builder.CreateCall(point, {makeMarkerString(*module, "ordinary_point")});

    SQTTConfig config;
    useFullScopeMasks(config);
    config.TraceMemoryAddrs = true;
    // This would fail if .buffer.load.lds were misclassified as an address
    // block during the preflight scan.
    config.ShaderClockBits = 12;

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    const std::string funcMap = getFuncMap(*module);
    expectContains(funcMap, "M:shader_clock_bits=12;shader_clock_shift=4");
    expectNotContains(funcMap, "addr_trace_buffer_load");
    expectNotContains(funcMap, "W:");
}

TEST(MarkerPass, AddressTraceBlocksDisableAutomaticGfx12ClockPackingAndUseBlockBoundaries)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* i32 = Type::getInt32Ty(ctx);
    Type* voidTy = Type::getVoidTy(ctx);
    Function* function = makeFunction(
        *module,
        "gfx12_address_trace",
        "gfx1200",
        FunctionType::get(voidTy, {PointerType::get(ctx, 1)}, false)
    );
    BasicBlock* entry = BasicBlock::Create(ctx, "entry", function);
    IRBuilder<> builder(entry);
    builder.CreateLoad(i32, function->getArg(0));
    builder.CreateRetVoid();

    SQTTConfig config;
    useFullScopeMasks(config);
    config.MemBarrier = MemBarrierMode::Fence;
    config.TraceMemoryAddrs = true;

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    const std::string funcMap = getFuncMap(*module);
    expectContains(funcMap, "W:32");
    expectPointEntryWithPayload(funcMap, "addr_trace_load", 66);
    expectNotContains(funcMap, "M:shader_clock_bits=");
    EXPECT_EQ(countIntrinsicCalls(*function, Intrinsic::amdgcn_s_getreg), 0u);
    EXPECT_EQ(countFences(*function), 2u);
    EXPECT_EQ(countIntrinsicCalls(*function, Intrinsic::amdgcn_sched_barrier), 2u);
}

TEST(MarkerPass, ForcedGfx12ClockPackingFailsForAddressTraceBlocks)
{
    EXPECT_DEATH(
        {
            LLVMContext ctx;
            auto module = makeModule(ctx);
            Type* i32 = Type::getInt32Ty(ctx);
            Type* voidTy = Type::getVoidTy(ctx);
            Function* function = makeFunction(
                *module,
                "forced_clock_address_trace",
                "gfx1200",
                FunctionType::get(voidTy, {PointerType::get(ctx, 1)}, false)
            );
            BasicBlock* entry = BasicBlock::Create(ctx, "entry", function);
            IRBuilder<> builder(entry);
            builder.CreateLoad(i32, function->getArg(0));
            builder.CreateRetVoid();

            SQTTConfig config;
            useFullScopeMasks(config);
            config.TraceMemoryAddrs = true;
            config.ShaderClockBits = 12;

            ModuleAnalysisManager analysisManager;
            SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
            pass.run(*module, analysisManager);
        },
        "SQTT_TRACE_ADDRESSES emits multi-payload records"
    );
}

TEST(MarkerPass, OnePayloadNamedDataRemainsEligibleForGfx12ClockPacking)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* i32 = Type::getInt32Ty(ctx);
    Type* voidTy = Type::getVoidTy(ctx);
    Function* data = Function::Create(
        FunctionType::get(voidTy, {PointerType::get(ctx, 0), i32}, false),
        GlobalValue::ExternalLinkage,
        "__sqtt_named_marker_data",
        module.get()
    );
    GlobalVariable* name = makeMarkerString(*module, "one_payload");
    Function* function = makeVoidFunction(*module, "gfx12_named_data", "gfx1200");
    IRBuilder<> builder(function->getEntryBlock().getTerminator());
    builder.CreateCall(data, {name, ConstantInt::get(i32, 17)});

    SQTTConfig config;
    useFullScopeMasks(config);

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    const std::string funcMap = getFuncMap(*module);
    expectContains(funcMap, "M:shader_clock_bits=12;shader_clock_shift=4");
    expectPointEntryWithPayload(funcMap, "one_payload", 1);
    EXPECT_EQ(countIntrinsicCalls(*function, Intrinsic::amdgcn_s_getreg), 1u);
}

TEST(MarkerPass, Gfx9BufferAndPermuteAddressTracesUseInlineAsmExecProtocol)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* i32 = Type::getInt32Ty(ctx);
    Type* i16 = Type::getInt16Ty(ctx);
    auto* rsrcVecTy = FixedVectorType::get(i32, 4);

    Function* function = makeVoidFunction(*module, "gfx9_buffer_traces", "gfx90a");
    Instruction* ret = function->getEntryBlock().getTerminator();
    IRBuilder<> builder(ret);

    Function* rawLoad = Function::Create(
        FunctionType::get(i32, {rsrcVecTy, i32, i16}, false),
        GlobalValue::ExternalLinkage,
        "llvm.amdgcn.raw.buffer.load.unit",
        module.get()
    );
    builder.CreateCall(
        rawLoad, {ConstantAggregateZero::get(rsrcVecTy), ConstantInt::get(i32, 1), ConstantInt::get(i16, 2)}
    );

    FunctionCallee permute = Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_ds_permute);
    builder.CreateCall(permute, {ConstantInt::get(i32, 8), ConstantInt::get(i32, 13)});

    SQTTConfig config;
    useFullScopeMasks(config);
    config.TraceMemoryAddrs = true;
    config.TraceLDSAddrs = true;

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    std::string funcMap = getFuncMap(*module);
    expectContains(funcMap, "W:64");
    expectContains(funcMap, "addr_trace_buffer_load");
    expectContains(funcMap, "addr_trace_ds_permute");

    std::string ir = printModule(*module);
    expectContains(ir, "s_mov_b32 m0, exec_lo");
    expectContains(ir, "s_nop 0");
    expectContains(ir, "s_ttracedata");
    expectContains(ir, "={m0}");
}

TEST(MarkerPass, FullM0TracesUseExplicitNopSequenceOnEveryArchitecture)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    const uint32_t markerValue = encodeMarker(64, false, false);
    MDNode* traceMetadata = MDNode::get(ctx, {});

    std::vector<Function*> functions;
    for (const auto& [name, cpu] : {std::pair{"gfx9_trace", "gfx90a"},
                                    std::pair{"gfx10_trace", "gfx1030"},
                                    std::pair{"gfx12_trace", "gfx1200"}})
    {
        Function* function = makeVoidFunction(*module, name, cpu);
        CallInst* trace = insertTraceCallBefore(function->getEntryBlock().getTerminator(), markerValue);
        trace->setMetadata("sqtt.test.trace", traceMetadata);
        functions.push_back(function);
    }

    SQTTConfig config;
    useFullScopeMasks(config);

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    for (Function* function : functions)
    {
        const CallInst* trace = findM0NopTrace(*function);
        ASSERT_NE(trace, nullptr) << function->getName().str();
        auto* inlineAsm = dyn_cast<InlineAsm>(trace->getCalledOperand());
        ASSERT_NE(inlineAsm, nullptr);
        EXPECT_EQ(inlineAsm->getConstraintString(), "={m0},i");
        EXPECT_EQ(trace->getMetadata("sqtt.test.trace"), traceMetadata);
    }
}

TEST(MarkerPass, Gfx12PacksOnlyPassGeneratedHeaders)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* i32 = Type::getInt32Ty(ctx);
    Function* function = makeFunction(*module, "numeric_marker", "gfx1200", FunctionType::get(i32, {i32}, false));
    BasicBlock* entry = BasicBlock::Create(ctx, "entry", function);
    IRBuilder<> builder(entry);
    Value* value = builder.CreateAdd(function->getArg(0), ConstantInt::get(i32, 1));
    builder.CreateAdd(value, ConstantInt::get(i32, 2));
    Instruction* ret = builder.CreateRet(value);

    // This mirrors the numeric marker API: it has no funcmap entry and must
    // keep its full legacy value even when the same module has packed headers.
    insertTraceCallBefore(ret, encodeMarker(1u << 20, false, false));
    Function* trace = Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_s_ttracedata);
    builder.SetInsertPoint(ret);
    builder.CreateCall(trace, {value});

    SQTTConfig config;
    useFullScopeMasks(config);
    config.FunctionThreshold = 1;

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    // Entry and exit function markers read the shader clock; neither
    // unregistered numeric marker reads it or trips the packed-ID range check.
    EXPECT_EQ(countIntrinsicCalls(*function, Intrinsic::amdgcn_s_getreg), 2u);
    expectContains(getFuncMap(*module), "M:shader_clock_bits=12;shader_clock_shift=4");
}

TEST(MarkerPass, BarrierInstrumentationHandlesSplitAndStandaloneBarriers)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* i32 = Type::getInt32Ty(ctx);
    Type* i16 = Type::getInt16Ty(ctx);

    Function* function = makeVoidFunction(*module, "barrier_traces", "gfx1100");
    Instruction* ret = function->getEntryBlock().getTerminator();
    IRBuilder<> builder(ret);

    FunctionCallee signal = Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_s_barrier_signal);
    FunctionCallee wait = Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_s_barrier_wait);
    FunctionCallee full = Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_s_barrier);
    Function* work = Function::Create(
        FunctionType::get(Type::getVoidTy(ctx), false),
        GlobalValue::ExternalLinkage,
        "barrier_work",
        module.get()
    );

    builder.CreateCall(signal, {ConstantInt::get(i32, 0)});
    builder.CreateCall(wait, {ConstantInt::get(i16, 0)});
    builder.CreateCall(signal, {ConstantInt::get(i32, 0)});
    builder.CreateCall(work);
    builder.CreateCall(wait, {ConstantInt::get(i16, 0)});
    builder.CreateCall(full);

    SQTTConfig config;
    useFullScopeMasks(config);
    config.InstrumentBarriers = true;

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    std::string funcMap = getFuncMap(*module);
    std::optional<unsigned> signalId = pointEntryId(funcMap, "barrier_signal");
    std::optional<unsigned> waitId = pointEntryId(funcMap, "barrier_wait");
    std::optional<unsigned> fullId = pointEntryId(funcMap, "barrier");
    ASSERT_TRUE(signalId.has_value());
    ASSERT_TRUE(waitId.has_value());
    ASSERT_TRUE(fullId.has_value());
    EXPECT_NE(signalId, waitId);
    EXPECT_NE(signalId, fullId);
    EXPECT_NE(waitId, fullId);

    size_t traceCount = countIntrinsicCalls(*module, Intrinsic::amdgcn_s_ttracedata) +
                        countIntrinsicCalls(*module, Intrinsic::amdgcn_s_ttracedata_imm);
    EXPECT_EQ(traceCount, 4u);
}

TEST(MarkerPass, NamedExitEnterFusionRequiresDirectAdjacency)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* i32 = Type::getInt32Ty(ctx);
    Function* exit = makeNamedMarkerSentinel(*module, "__sqtt_named_marker_exit");
    Function* enter = makeNamedMarkerSentinel(*module, "__sqtt_named_marker_enter");
    GlobalVariable* oldName = makeMarkerString(*module, "old");
    GlobalVariable* newName = makeMarkerString(*module, "new");

    Function* separated = makeFunction(
        *module, "separated_named_markers", "gfx1100", FunctionType::get(Type::getVoidTy(ctx), {i32}, false)
    );
    BasicBlock* separatedEntry = BasicBlock::Create(ctx, "entry", separated);
    IRBuilder<> separatedBuilder(separatedEntry);
    separatedBuilder.CreateCall(exit, {oldName});
    separatedBuilder.CreateAdd(separated->getArg(0), ConstantInt::get(i32, 1));
    separatedBuilder.CreateCall(enter, {newName});
    separatedBuilder.CreateRetVoid();

    Function* adjacent = makeVoidFunction(*module, "adjacent_named_markers", "gfx1100");
    Instruction* adjacentRet = adjacent->getEntryBlock().getTerminator();
    IRBuilder<> adjacentBuilder(adjacentRet);
    adjacentBuilder.CreateCall(exit, {oldName});
    adjacentBuilder.CreateCall(enter, {newName});

    SQTTConfig config;
    useFullScopeMasks(config);

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    std::vector<uint32_t> separatedMarkers = traceMarkerValues(*separated);
    ASSERT_EQ(separatedMarkers.size(), 2u);
    EXPECT_EQ(separatedMarkers[0], FLAG_EXIT_PREV);
    EXPECT_EQ(separatedMarkers[1] & FLAG_MASK, FLAG_ENTER);

    std::vector<uint32_t> adjacentMarkers = traceMarkerValues(*adjacent);
    ASSERT_EQ(adjacentMarkers.size(), 1u);
    EXPECT_EQ(adjacentMarkers[0] & FLAG_MASK, FLAG_ENTER | FLAG_EXIT_PREV);
}

TEST(MarkerPass, NumericMarkersAlwaysReceiveConfiguredMemoryBarriers)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Function* function = makeVoidFunction(*module, "numeric_marker", "gfx1100");
    insertTraceCallBefore(function->getEntryBlock().getTerminator(), encodeMarker(17, false, false));

    SQTTConfig config;
    useFullScopeMasks(config);
    config.MemBarrier = MemBarrierMode::Fence;

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    EXPECT_EQ(countFences(*function), 2u);
}

TEST(MarkerPass, DirectFunctionInstrumentationHandlesO0Fallback)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* i32 = Type::getInt32Ty(ctx);

    Function* large = makeFunction(*module, "direct_large", "gfx1100", FunctionType::get(i32, {i32}, false));
    BasicBlock* entry = BasicBlock::Create(ctx, "entry", large);
    BasicBlock* thenBlock = BasicBlock::Create(ctx, "then", large);
    BasicBlock* elseBlock = BasicBlock::Create(ctx, "else", large);
    IRBuilder<> builder(entry);
    Value* arg = large->getArg(0);
    builder.CreateCondBr(builder.CreateICmpUGT(arg, ConstantInt::get(i32, 10)), thenBlock, elseBlock);
    builder.SetInsertPoint(thenBlock);
    builder.CreateRet(builder.CreateAdd(arg, ConstantInt::get(i32, 1)));
    builder.SetInsertPoint(elseBlock);
    builder.CreateRet(builder.CreateSub(arg, ConstantInt::get(i32, 1)));

    Function* small = makeVoidFunction(*module, "direct_small", "gfx1100");
    Function* kernel = makeVoidFunction(*module, "direct_kernel", "gfx1100");
    kernel->setCallingConv(CallingConv::AMDGPU_KERNEL);

    SQTTConfig config;
    useFullScopeMasks(config);
    config.FunctionThreshold = 3;

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    std::string funcMap = getFuncMap(*module);
    expectContains(funcMap, "F:1:direct_large");
    expectContains(funcMap, "K:direct_kernel");
    expectNotContains(funcMap, "direct_small");

    std::vector<uint32_t> largeMarkers = traceMarkerValues(*large);
    EXPECT_EQ(std::count(largeMarkers.begin(), largeMarkers.end(), encodeMarker(1, true, false)), 1);
    EXPECT_EQ(std::count(largeMarkers.begin(), largeMarkers.end(), FLAG_EXIT_PREV), 2);
    EXPECT_TRUE(traceMarkerValues(*small).empty());
    EXPECT_TRUE(traceMarkerValues(*kernel).empty());
}

TEST(MarkerPass, FunctionThresholdIgnoresPassMarkersBeforeAndAfterOptimization)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Function* function = makeVoidFunction(*module, "small_function", "gfx1100");

    SQTTConfig config;
    useFullScopeMasks(config);
    config.FunctionThreshold = 1;

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass early(config, SQTTInstrumentPass::Mode::Early);
    early.run(*module, analysisManager);

    EXPECT_EQ(earlyFunctionPreOptSize(*module, function->getName()), 1u);

    SQTTInstrumentPass late(config, SQTTInstrumentPass::Mode::Late);
    late.run(*module, analysisManager);

    EXPECT_TRUE(traceMarkerValues(*function).empty());
    expectNotContains(getFuncMap(*module), "small_function");
}

TEST(MarkerPass, AutomaticFunctionInstrumentationSkipsMustTailFunctions)
{
    auto makeMustTailFunction = [](Module& module, StringRef name)
    {
        LLVMContext& ctx = module.getContext();
        Type* i32 = Type::getInt32Ty(ctx);
        FunctionType* type = FunctionType::get(i32, {i32}, false);
        Function* callee =
            Function::Create(type, GlobalValue::ExternalLinkage, name.str() + ".callee", &module);
        Function* function = makeFunction(module, name, "gfx1100", type);
        BasicBlock* entry = BasicBlock::Create(ctx, "entry", function);
        IRBuilder<> builder(entry);
        CallInst* call = builder.CreateCall(callee, {function->getArg(0)});
        call->setTailCallKind(CallInst::TCK_MustTail);
        builder.CreateRet(call);
        return function;
    };

    SQTTConfig config;
    useFullScopeMasks(config);
    config.FunctionThreshold = 1;
    ModuleAnalysisManager analysisManager;

    LLVMContext earlyCtx;
    auto earlyModule = makeModule(earlyCtx);
    Function* earlyFunction = makeMustTailFunction(*earlyModule, "early_musttail");
    SQTTInstrumentPass early(config, SQTTInstrumentPass::Mode::Early);
    early.run(*earlyModule, analysisManager);
    EXPECT_TRUE(traceMarkerValues(*earlyFunction).empty());
    EXPECT_FALSE(earlyModule->getNamedMetadata("sqtt.funcmap.early"));

    LLVMContext lateCtx;
    auto lateModule = makeModule(lateCtx);
    Function* lateFunction = makeMustTailFunction(*lateModule, "late_musttail");
    SQTTInstrumentPass late(config, SQTTInstrumentPass::Mode::Late);
    late.run(*lateModule, analysisManager);
    EXPECT_TRUE(traceMarkerValues(*lateFunction).empty());
    expectNotContains(getFuncMap(*lateModule), "late_musttail");
}

TEST(MarkerPass, FunctionThresholdPrunesMarkersAndPreservesExistingLlvmUsed)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* i32 = Type::getInt32Ty(ctx);
    uint32_t smallId = 7;
    uint32_t largeId = 8;

    Function* small = makeVoidFunction(*module, "small_function", "gfx1100");
    Instruction* smallRet = small->getEntryBlock().getTerminator();
    insertTraceCallBefore(smallRet, encodeMarker(smallId, true, false), true);
    insertTraceCallBefore(smallRet, encodeMarker(smallId, false, true), true);
    addEarlyFunctionMetadata(*small, smallId, 1, "small.hip:3");

    Function* large =
        makeFunction(*module, "large_function", "gfx1100", FunctionType::get(Type::getVoidTy(ctx), {i32}, false));
    BasicBlock* largeEntry = BasicBlock::Create(ctx, "entry", large);
    IRBuilder<> builder(largeEntry);
    Value* value = large->getArg(0);
    for (unsigned i = 0; i < 30; ++i) value = builder.CreateAdd(value, ConstantInt::get(i32, i + 1));
    builder.CreateRetVoid();
    Instruction* firstLargeInst = &*large->getEntryBlock().getFirstInsertionPt();
    Instruction* largeRet = large->getEntryBlock().getTerminator();
    insertTraceCallBefore(firstLargeInst, encodeMarker(largeId, true, false), true);
    insertTraceCallBefore(largeRet, encodeMarker(largeId, false, true), true);
    addEarlyFunctionMetadata(*large, largeId, 40, "large.hip:17");

    // This unregistered numeric marker happens to use the pruned function's
    // old ID. It must not be removed or rewritten with pass-owned headers.
    Function* numeric = makeVoidFunction(*module, "numeric_marker", "gfx90a");
    insertTraceCallBefore(numeric->getEntryBlock().getTerminator(), encodeMarker(smallId, true, false));

    addEarlyFunctionMapEntry(*module, 99, "inlined_large_function", 40, "inlined.hip:21");
    addEarlyFunctionMapEntry(*module, 100, "inlined_small_function", 1, "inlined.hip:4");
    addExistingLlvmUsed(*module);

    SQTTConfig config;
    useFullScopeMasks(config);
    config.FunctionThreshold = 20;

    ModuleAnalysisManager analysisManager;
    SQTTInstrumentPass pass(config, SQTTInstrumentPass::Mode::Late);
    pass.run(*module, analysisManager);

    std::string funcMap = getFuncMap(*module);
    expectContains(funcMap, "F:1:large_function@large.hip:17");
    expectContains(funcMap, "F:2:inlined_large_function@inlined.hip:21");
    expectNotContains(funcMap, "small_function");
    expectNotContains(funcMap, "inlined_small_function");
    EXPECT_EQ(llvmUsedOperandCount(*module), 2u);

    EXPECT_EQ(countIntrinsicCalls(*small, Intrinsic::amdgcn_s_ttracedata), 0u);
    EXPECT_EQ(countIntrinsicCalls(*small, Intrinsic::amdgcn_s_ttracedata_imm), 0u);
    const CallInst* numericTrace = findM0NopTrace(*numeric);
    ASSERT_NE(numericTrace, nullptr);
    auto* numericValue = dyn_cast<ConstantInt>(numericTrace->getArgOperand(0));
    ASSERT_NE(numericValue, nullptr);
    EXPECT_EQ(numericValue->getZExtValue(), encodeMarker(smallId, true, false));
}
