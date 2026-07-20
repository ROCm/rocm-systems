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

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
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
        if (const char* old = std::getenv(Name.c_str())) OldValue = old;
        if (value)
            setenv(Name.c_str(), value->c_str(), 1);
        else
            unsetenv(Name.c_str());
    }
    ~ScopedEnv() { OldValue ? setenv(Name.c_str(), OldValue->c_str(), 1) : unsetenv(Name.c_str()); }

    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    std::string Name;
    std::optional<std::string> OldValue;
};

std::vector<std::unique_ptr<ScopedEnv>> clearSqttEnvironment()
{
    std::vector<std::unique_ptr<ScopedEnv>> env;
    for (const char* name : {"SQTT_INSTRUMENT_BARRIERS", "SQTT_MEM_BARRIER", "SQTT_SCOPE_WAVE", "SQTT_SCOPE_SIMD",
                             "SQTT_SCOPE_CU", "SQTT_SCOPE_WG", "SQTT_SHADER_CLOCK_BITS", "SQTT_SHADER_CLOCK_SHIFT",
                             "SQTT_INSTRUMENT_FUNCTIONS", "SQTT_INSTRUMENT_MEMORY", "SQTT_TRACE_ADDRESSES"})
        env.push_back(std::make_unique<ScopedEnv>(name, std::nullopt));
    return env;
}

std::unique_ptr<Module> makeModule(LLVMContext& ctx)
{
    auto module = std::make_unique<Module>("markers-unit", ctx); module->setTargetTriple(Triple("amdgcn-amd-amdhsa"));
    return module;
}

Function* makeFunction(Module& module, StringRef name, StringRef cpu, FunctionType* type)
{
    auto* function = Function::Create(type, GlobalValue::ExternalLinkage, name, module); function->addFnAttr("target-cpu", cpu);
    return function;
}

Function* declareFunction(Module& module, StringRef name, Type* result, ArrayRef<Type*> args)
{
    return Function::Create(FunctionType::get(result, args, false), GlobalValue::ExternalLinkage, name, &module);
}

Function* makeVoidFunction(Module& module, StringRef name, StringRef cpu)
{
    LLVMContext& ctx = module.getContext();
    Function* function = makeFunction(module, name, cpu, FunctionType::get(Type::getVoidTy(ctx), false));
    IRBuilder<>(BasicBlock::Create(ctx, "entry", function)).CreateRetVoid();
    return function;
}

Function* makeGlobalLoadFunction(Module& module, StringRef name, StringRef cpu)
{
    LLVMContext& ctx = module.getContext();
    Type* i32 = Type::getInt32Ty(ctx);
    Function* function = makeFunction(module, name, cpu,
                                      FunctionType::get(Type::getVoidTy(ctx), {PointerType::get(ctx, 1)}, false));
    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", function));
    builder.CreateLoad(i32, function->getArg(0));
    builder.CreateRetVoid();
    return function;
}

Function* makeBufferTraceFunction(Module& module, StringRef cpu, bool bpermute)
{
    LLVMContext& ctx = module.getContext();
    Type *i32 = Type::getInt32Ty(ctx), *i16 = Type::getInt16Ty(ctx), *i64 = Type::getInt64Ty(ctx);
    Type* voidTy = Type::getVoidTy(ctx);
    auto* rsrcTy = FixedVectorType::get(i32, 4);
    auto* rsrcPtrTy = PointerType::get(ctx, 8);
    Function* function = makeFunction(module, "buffer_traces", cpu, FunctionType::get(voidTy, {rsrcPtrTy}, false));
    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", function));
    Value* rsrc = ConstantAggregateZero::get(rsrcTy);

    Type* offsetTy = bpermute ? i64 : i32;
    builder.CreateCall(declareFunction(module, "llvm.amdgcn.raw.buffer.load.unit", i32, {rsrcTy, offsetTy, i16}),
                       {rsrc, ConstantInt::get(offsetTy, 11), ConstantInt::get(i16, 3)});
    builder.CreateCall(
        declareFunction(module, "llvm.amdgcn.struct.buffer.store.unit", voidTy, {i32, rsrcTy, i16, i16, i64}),
        {ConstantInt::get(i32, 17), rsrc, ConstantInt::get(i16, 5), ConstantInt::get(i16, 7),
         ConstantInt::get(i64, 9)}
    );
    builder.CreateCall(
        declareFunction(module, "llvm.amdgcn.raw.ptr.buffer.atomic.cmpswap.unit", i32,
                        {i32, i32, rsrcPtrTy, i16, i16}),
        {ConstantInt::get(i32, 1), ConstantInt::get(i32, 2), function->getArg(0), ConstantInt::get(i16, 4),
         ConstantInt::get(i16, 6)}
    );
    builder.CreateCall(Intrinsic::getOrInsertDeclaration(
                           &module, bpermute ? Intrinsic::amdgcn_ds_bpermute : Intrinsic::amdgcn_ds_permute
                       ),
                       {ConstantInt::get(i32, 16), ConstantInt::get(i32, 33)});
    builder.CreateRetVoid();

    Function* addresses = makeFunction(
        module, "address_spaces", cpu,
        FunctionType::get(voidTy, {PointerType::get(ctx, 0), PointerType::get(ctx, 1), PointerType::get(ctx, 2),
                                    PointerType::get(ctx, 4), PointerType::get(ctx, 5)}, false)
    );
    IRBuilder<> addressBuilder(BasicBlock::Create(ctx, "entry", addresses));
    addressBuilder.CreateLoad(i32, addresses->getArg(0));
    addressBuilder.CreateStore(ConstantInt::get(i32, 1), addresses->getArg(1));
    for (unsigned i : {2u, 3u, 4u}) addressBuilder.CreateLoad(i32, addresses->getArg(i));
    addressBuilder.CreateRetVoid();
    return function;
}

SQTTConfig fullScopeConfig()
{
    SQTTConfig config;
    config.WaveMask = FULL_WAVE_MASK; config.SimdMask = FULL_SIMD_MASK;
    config.CuMask = FULL_CU_MASK; config.WgMask = FULL_WG_MASK;
    config.MemBarrier = MemBarrierMode::None;
    return config;
}

void runPass(Module& module, const SQTTConfig& config, SQTTInstrumentPass::Mode mode = SQTTInstrumentPass::Mode::Late)
{
    ModuleAnalysisManager analysisManager; SQTTInstrumentPass(config, mode).run(module, analysisManager);
}

CallInst* insertTraceCallBefore(Instruction* insertPt, uint32_t encoded, bool passHeader = false)
{
    Module* module = insertPt->getModule();
    IRBuilder<> builder(insertPt);
    CallInst* call = builder.CreateCall(Intrinsic::getOrInsertDeclaration(module, Intrinsic::amdgcn_s_ttracedata),
                                        {ConstantInt::get(Type::getInt32Ty(module->getContext()), encoded)});
    if (passHeader) call->setMetadata("sqtt.marker_header", MDNode::get(module->getContext(), {}));
    return call;
}

Function* makeNamedMarkerSentinel(Module& module, StringRef name)
{
    LLVMContext& ctx = module.getContext(); return declareFunction(module, name, Type::getVoidTy(ctx), {PointerType::get(ctx, 0)});
}

GlobalVariable* makeMarkerString(Module& module, StringRef value)
{
    LLVMContext& ctx = module.getContext();
    auto* initializer = ConstantDataArray::getString(ctx, value, true);
    return new GlobalVariable(module, initializer->getType(), true, GlobalValue::PrivateLinkage, initializer,
                              ".sqtt.marker.string");
}

void addEarlyFunctionMapEntry(Module& module, uint32_t id, StringRef name, unsigned preOptSize, StringRef sourceLoc)
{
    LLVMContext& ctx = module.getContext();
    Type* i32 = Type::getInt32Ty(ctx);
    NamedMDNode* earlyMap = module.getOrInsertNamedMetadata("sqtt.markers.early");
    earlyMap->addOperand(MDNode::get(
        ctx,
        {ConstantAsMetadata::get(ConstantInt::get(i32, id)),
         ConstantAsMetadata::get(ConstantInt::get(i32, 0)), // MarkerKind::Function
         MDString::get(ctx, name),
         ConstantAsMetadata::get(ConstantInt::get(i32, preOptSize)),
         MDString::get(ctx, sourceLoc),
         ConstantAsMetadata::get(ConstantInt::get(i32, 0))}
    ));
}

void addEarlyFunctionMetadata(Function& function, uint32_t id, unsigned preOptSize, StringRef sourceLoc)
{
    Module* module = function.getParent();
    LLVMContext& ctx = module->getContext();
    Type* i32 = Type::getInt32Ty(ctx);
    function.setMetadata("sqtt.func.id", MDNode::get(ctx, {ConstantAsMetadata::get(ConstantInt::get(i32, id))}));
    addEarlyFunctionMapEntry(*module, id, function.getName(), preOptSize, sourceLoc);
}

void addPassOwnedFunctionMarkers(Function& function, uint32_t id)
{
    BasicBlock& entry = function.getEntryBlock();
    insertTraceCallBefore(&*entry.getFirstInsertionPt(), encodeMarker(id, true, false), true);
    insertTraceCallBefore(entry.getTerminator(), encodeMarker(id, false, true), true);
}

Function* makeLargePassOwnedFunction(Module& module, StringRef name, uint32_t id, StringRef sourceLoc)
{
    LLVMContext& ctx = module.getContext();
    Type* i32 = Type::getInt32Ty(ctx);
    Function* function = makeFunction(module, name, "gfx1100", FunctionType::get(Type::getVoidTy(ctx), {i32}, false));
    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", function));
    Value* value = function->getArg(0);
    for (unsigned i = 0; i < 30; ++i) value = builder.CreateAdd(value, ConstantInt::get(i32, i + 1));
    builder.CreateRetVoid();
    addPassOwnedFunctionMarkers(*function, id);
    addEarlyFunctionMetadata(*function, id, 40, sourceLoc);
    return function;
}

Function* makeMustTailFunction(Module& module, StringRef name)
{
    LLVMContext& ctx = module.getContext();
    Type* i32 = Type::getInt32Ty(ctx); FunctionType* type = FunctionType::get(i32, {i32}, false);
    Function* callee = declareFunction(module, name.str() + ".callee", i32, {i32});
    Function* function = makeFunction(module, name, "gfx1100", type);
    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", function));
    CallInst* call = builder.CreateCall(callee, {function->getArg(0)});
    call->setTailCallKind(CallInst::TCK_MustTail);
    builder.CreateRet(call);
    return function;
}

std::string getFuncMap(const Module& module)
{
    for (const GlobalVariable& global : module.globals())
        if (global.getSection() == ".sqtt_funcmap" && global.hasInitializer())
            if (auto* data = dyn_cast<ConstantDataArray>(global.getInitializer()); data && data->isString())
                return data->getAsCString().str();
    return {};
}

std::string runPassAndGetFuncMap(
    Module& module, const SQTTConfig& config, SQTTInstrumentPass::Mode mode = SQTTInstrumentPass::Mode::Late
)
{
    runPass(module, config, mode); return getFuncMap(module);
}

std::string printModule(const Module& module)
{
    std::string text; raw_string_ostream os(text); module.print(os, nullptr); return os.str();
}

template <typename Visitor>
void forEachCall(const Function& function, Visitor visit)
{
    for (const BasicBlock& block : function)
        for (const Instruction& inst : block)
            if (const auto* call = dyn_cast<CallInst>(&inst)) visit(*call);
}

size_t countIntrinsicCalls(const Function& function, Intrinsic::ID id)
{
    size_t count = 0;
    forEachCall(function, [&](const CallInst& call)
    {
        const Function* callee = call.getCalledFunction();
        count += callee && callee->getIntrinsicID() == id;
    });
    return count;
}

size_t countIntrinsicCalls(const Module& module, Intrinsic::ID id)
{
    size_t count = 0;
    for (const Function& function : module) count += countIntrinsicCalls(function, id);
    return count;
}

size_t countFences(const Function& function)
{
    size_t count = 0;
    for (const BasicBlock& block : function) for (const Instruction& inst : block) count += isa<FenceInst>(inst);
    return count;
}

std::vector<uint32_t> traceMarkerValues(const Function& function)
{
    std::vector<uint32_t> values;
    forEachCall(function, [&](const CallInst& call)
    {
        const Function* callee = call.getCalledFunction();
        if (!callee) return;
        auto id = callee->getIntrinsicID();
        if (id != Intrinsic::amdgcn_s_ttracedata && id != Intrinsic::amdgcn_s_ttracedata_imm) return;
        if (auto* arg = dyn_cast<ConstantInt>(call.getArgOperand(0))) values.push_back(arg->getZExtValue());
    });
    return values;
}

std::optional<uint32_t> markerAfter(Instruction* instruction)
{
    for (instruction = instruction->getNextNode(); instruction; instruction = instruction->getNextNode())
    {
        auto* call = dyn_cast<CallInst>(instruction);
        auto* callee = call ? call->getCalledFunction() : nullptr;
        if (!callee) return std::nullopt;
        if (callee->getIntrinsicID() == Intrinsic::amdgcn_sched_barrier) continue;
        if (callee->getIntrinsicID() != Intrinsic::amdgcn_s_ttracedata_imm) return std::nullopt;
        if (auto* value = dyn_cast<ConstantInt>(call->getArgOperand(0))) return value->getZExtValue();
        return std::nullopt;
    }
    return std::nullopt;
}

const CallInst* findM0NopTrace(const Function& function)
{
    constexpr const char TraceAsm[] = "s_mov_b32 m0, $1\n"
                                      "s_nop 0\n"
                                      "s_ttracedata";
    const CallInst* trace = nullptr;
    forEachCall(function, [&](const CallInst& call)
    {
        auto* asmCall = dyn_cast<InlineAsm>(call.getCalledOperand());
        if (!trace && asmCall && asmCall->hasSideEffects() && asmCall->getAsmString() == TraceAsm) trace = &call;
    });
    return trace;
}

bool isTraceCall(const CallInst& call)
{
    if (const Function* callee = call.getCalledFunction())
        return callee->getIntrinsicID() == Intrinsic::amdgcn_s_ttracedata ||
               callee->getIntrinsicID() == Intrinsic::amdgcn_s_ttracedata_imm;
    if (const auto* asmCall = dyn_cast<InlineAsm>(call.getCalledOperand()))
        return asmCall->getAsmString().contains("s_ttracedata");
    return false;
}

CallInst* findTraceWithMetadata(Function& function, StringRef metadata)
{
    for (BasicBlock& block : function)
        for (Instruction& instruction : block)
            if (auto* call = dyn_cast<CallInst>(&instruction); call && call->getMetadata(metadata) && isTraceCall(*call))
                return call;
    return nullptr;
}

void expectScopedSkipPin(Function& function)
{
    const BasicBlock* skip = nullptr;
    for (const BasicBlock& block : function)
        if (block.getName().starts_with("sqtt.skip")) skip = &block;
    ASSERT_NE(skip, nullptr);
    auto it = skip->begin();
    ASSERT_NE(it, skip->end());
    auto* pin = dyn_cast<CallInst>(&*it++);
    ASSERT_NE(pin, nullptr);
    ASSERT_EQ(pin->getCalledFunction()->getIntrinsicID(), Intrinsic::amdgcn_sched_barrier);
    ASSERT_NE(it, skip->end());
    auto* sync = dyn_cast<CallInst>(&*it);
    ASSERT_NE(sync, nullptr);
    EXPECT_EQ(sync->getCalledFunction()->getIntrinsicID(), Intrinsic::amdgcn_s_barrier);
}

void expectScopedMarkerCase(bool early, bool sync)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Function* function = makeVoidFunction(*module, "scoped_marker", "gfx1100");
    IRBuilder<> builder(function->getEntryBlock().getTerminator());
    builder.CreateCall(makeNamedMarkerSentinel(*module, "__sqtt_named_marker_point"),
                       {makeMarkerString(*module, "scoped")});
    if (sync) builder.CreateCall(Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_s_barrier));

    SQTTConfig config = fullScopeConfig();
    config.CuMask = 0x1;
    if (early) runPass(*module, config, SQTTInstrumentPass::Mode::Early);
    runPass(*module, config);
    EXPECT_EQ(countIntrinsicCalls(*function, Intrinsic::amdgcn_sched_barrier), sync ? 1u : 0u);
    if (sync) expectScopedSkipPin(*function);
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

void expectContains(const std::string& text, StringRef needle)
{
    EXPECT_NE(text.find(needle.str()), std::string::npos) << "missing: " << needle.str();
}

void expectNotContains(const std::string& text, StringRef needle)
{
    EXPECT_EQ(text.find(needle.str()), std::string::npos) << "unexpected: " << needle.str();
}

template <typename Visitor>
void forEachFuncmapLine(StringRef funcMap, Visitor visit)
{
    SmallVector<StringRef, 32> lines;
    funcMap.split(lines, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    for (StringRef line : lines) visit(line.rtrim("\r"));
}

std::vector<unsigned> pointEntryIds(const std::string& funcMap, StringRef name)
{
    std::vector<unsigned> ids;
    forEachFuncmapLine(funcMap, [&](StringRef line)
    {
        if (!line.consume_front("P:")) return;
        auto [idText, rest] = line.split(':');
        unsigned id = 0;
        if (idText.getAsInteger(10, id)) return;
        auto [entryName, sourceLoc] = rest.split('@');
        (void) sourceLoc;
        if (entryName == name) ids.push_back(id);
    });
    return ids;
}

std::optional<unsigned> pointEntryId(const std::string& funcMap, StringRef name)
{
    std::vector<unsigned> ids = pointEntryIds(funcMap, name);
    return ids.empty() ? std::nullopt : std::optional<unsigned>(ids.front());
}

size_t countPointEntries(const std::string& funcMap, StringRef name) { return pointEntryIds(funcMap, name).size(); }

std::optional<unsigned> extraPayloadCountForId(const std::string& funcMap, unsigned markerId)
{
    std::optional<unsigned> result;
    forEachFuncmapLine(funcMap, [&](StringRef line)
    {
        if (result || !line.consume_front("R:")) return;
        auto [idText, metadata] = line.split(':');
        unsigned id = 0;
        if (idText.getAsInteger(10, id) || id != markerId) return;
        SmallVector<StringRef, 4> fields;
        metadata.split(fields, ';', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
        for (StringRef field : fields)
        {
            if (!field.consume_front("extra_payload_count=")) continue;
            unsigned count = 0;
            if (!field.getAsInteger(10, count)) result = count;
        }
    });
    return result;
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

class MarkerPass : public ::testing::Test
{
protected:
    LLVMContext ctx;
    std::unique_ptr<Module> module = makeModule(ctx);
};

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
    auto env = clearSqttEnvironment();
    for (const auto& [name, value] : {std::pair{"SQTT_INSTRUMENT_BARRIERS", "YES"},
                                      std::pair{"SQTT_MEM_BARRIER", "clobber"},
                                      std::pair{"SQTT_INSTRUMENT_FUNCTIONS", "cost:42"},
                                      std::pair{"SQTT_INSTRUMENT_MEMORY", "4:7"},
                                      std::pair{"SQTT_TRACE_ADDRESSES", "memory, lds, bogus"},
                                      std::pair{"SQTT_SHADER_CLOCK_BITS", "not-a-number"},
                                      std::pair{"SQTT_SHADER_CLOCK_SHIFT", "8"},
                                      std::pair{"SQTT_SCOPE_WAVE", "not-a-mask"},
                                      std::pair{"SQTT_SCOPE_SIMD", "0x5"}, std::pair{"SQTT_SCOPE_CU", "-1"}})
        env.push_back(std::make_unique<ScopedEnv>(name, value));

    SQTTConfig config = SQTTConfig::fromEnvironment();

    EXPECT_TRUE(config.InstrumentBarriers);
    EXPECT_EQ(config.MemBarrier, MemBarrierMode::AsmClobber);
    EXPECT_EQ(config.Mode, CostMode::WeightedCost);
    EXPECT_EQ(config.FunctionThreshold, 42u);
    EXPECT_NE(config.MemoryChunkSize, 0u);
    EXPECT_EQ(config.MemoryChunkSize, 4u);
    EXPECT_EQ(config.MemoryMaxGap, 7u);
    EXPECT_FALSE(config.TraceMemoryAddrs);
    EXPECT_FALSE(config.TraceLDSAddrs);
    EXPECT_EQ(config.ShaderClockBits, 0u);
    EXPECT_EQ(config.ShaderClockShift, 8u);
    EXPECT_EQ(config.WaveMask, 0xFFFFFFFFu);
    EXPECT_EQ(config.SimdMask, 0x5u);
    EXPECT_EQ(config.CuMask, 0xFFFFFFFFu);

    for (const auto& [name, value] : {std::pair{"SQTT_INSTRUMENT_MEMORY", "4"},
                                      std::pair{"SQTT_TRACE_ADDRESSES", "lds"},
                                      std::pair{"SQTT_MEM_BARRIER", "bad-mode"}})
        env.push_back(std::make_unique<ScopedEnv>(name, value));
    config = SQTTConfig::fromEnvironment();
    EXPECT_EQ(config.MemBarrier, MemBarrierMode::Fence);
    EXPECT_EQ(config.MemoryChunkSize, 0u);
    EXPECT_TRUE(config.TraceLDSAddrs);
    EXPECT_FALSE(config.TraceMemoryAddrs);
}

TEST(MarkerTarget, ClassifiesArchitecturesAndInstructionCosts)
{
    LLVMContext ctx;
    auto module = makeModule(ctx);
    Type* i32 = Type::getInt32Ty(ctx);

    for (const auto& [cpu, expected] : {std::pair{"gfx90a", GfxGen::GFX9}, std::pair{"gfx1030", GfxGen::RDNA},
                                         std::pair{"gfx1100", GfxGen::RDNA}, std::pair{"gfx1200", GfxGen::GFX12},
                                         std::pair{"notgfx", GfxGen::Unknown}})
        EXPECT_EQ(getGfxGen(*makeVoidFunction(*module, cpu, cpu)), expected);
    for (const auto& [cpu, expected] : {std::pair{"gfx1200", false}, std::pair{"gfx1201", false},
                                         std::pair{"gfx1202", true}, std::pair{"gfx1250", true},
                                         std::pair{"gfx1250:xnack+", true}})
        EXPECT_EQ(hasShaderCyclesU64(*makeVoidFunction(*module, cpu, cpu)), expected);

    EXPECT_EQ(getWaveSize(GfxGen::GFX9), 64u);
    EXPECT_EQ(getWaveSize(GfxGen::RDNA), 32u);
    EXPECT_FALSE(supportsImmTrace(GfxGen::GFX9));
    EXPECT_TRUE(supportsImmTrace(GfxGen::GFX12));

    Function* costed =
        makeFunction(*module, "costed", "gfx1100", FunctionType::get(Type::getVoidTy(ctx), {i32}, false));
    BasicBlock* entry = BasicBlock::Create(ctx, "entry", costed);
    IRBuilder<> builder(entry);
    builder.CreateAlloca(i32);
    Value* loaded = builder.CreateLoad(i32, UndefValue::get(PointerType::get(ctx, 1)));
    builder.CreateStore(loaded, UndefValue::get(PointerType::get(ctx, 3)));
    builder.CreateCall(declareFunction(*module, "llvm.amdgcn.mfma.unit", i32, {}));
    builder.CreateRetVoid();

    EXPECT_EQ(computeFunctionSize(*costed, CostMode::InstructionCount), 4u);
    EXPECT_EQ(computeFunctionSize(*costed, CostMode::WeightedCost), 31u);
}

TEST_F(MarkerPass, FuncmapLedgerPreservesProtocolOrderAndDebugLocations)
{
    Type* i32 = Type::getInt32Ty(ctx);
    Function* device = makeFunction(
        *module, "ledger_device", "gfx1100", FunctionType::get(i32, {i32}, false)
    );
    IRBuilder<> deviceBuilder(BasicBlock::Create(ctx, "entry", device));
    Value* sum = deviceBuilder.CreateAdd(device->getArg(0), ConstantInt::get(i32, 1));
    sum = deviceBuilder.CreateAdd(sum, ConstantInt::get(i32, 2));
    deviceBuilder.CreateRet(sum);

    Function* kernel = makeFunction(
        *module,
        "ledger_kernel",
        "gfx1100",
        FunctionType::get(
            Type::getVoidTy(ctx), {PointerType::get(ctx, 1), PointerType::get(ctx, 3)}, false
        )
    );
    kernel->setCallingConv(CallingConv::AMDGPU_KERNEL);

    module->addModuleFlag(Module::Warning, "Debug Info Version", DEBUG_METADATA_VERSION);
    DIBuilder debug(*module);
    DIFile* file = debug.createFile("ledger.hip", "/source");
    DICompileUnit* unit = debug.createCompileUnit(
        dwarf::DW_LANG_C_plus_plus_14, file, "marker-unit", false, "", 0
    );
    DISubroutineType* debugType = debug.createSubroutineType(debug.getOrCreateTypeArray({}));
    DISubprogram* kernelScope = debug.createFunction(
        unit,
        "ledger_kernel",
        "ledger_kernel",
        file,
        10,
        debugType,
        10,
        DINode::FlagZero,
        DISubprogram::SPFlagDefinition
    );
    DISubprogram* inlinedScope = debug.createFunction(
        unit,
        "inlined_load",
        "inlined_load",
        file,
        7,
        debugType,
        7,
        DINode::FlagZero,
        DISubprogram::SPFlagDefinition
    );
    kernel->setSubprogram(kernelScope);

    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", kernel));
    Function* enter = makeNamedMarkerSentinel(*module, "__sqtt_named_marker_enter");
    Function* exit = makeNamedMarkerSentinel(*module, "__sqtt_named_marker_exit");
    GlobalVariable* scopeName = makeMarkerString(*module, "ledger_scope");
    builder.CreateCall(enter, {scopeName});
    Instruction* globalLoad = builder.CreateLoad(i32, kernel->getArg(0));
    Instruction* ldsStore = builder.CreateStore(globalLoad, kernel->getArg(1));
    builder.CreateCall(Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_s_barrier));
    builder.CreateCall(exit, {scopeName});
    builder.CreateRetVoid();

    DILocation* callSite = DILocation::get(ctx, 20, 1, kernelScope);
    globalLoad->setDebugLoc(DILocation::get(ctx, 7, 1, inlinedScope, callSite));
    ldsStore->setDebugLoc(DILocation::get(ctx, 30, 1, kernelScope));
    debug.finalize();

    SQTTConfig config = fullScopeConfig();
    config.FunctionThreshold = 1;
    config.InstrumentBarriers = true;
    config.TraceMemoryAddrs = config.TraceLDSAddrs = true;
    const std::string funcMap = runPassAndGetFuncMap(*module, config);

    EXPECT_FALSE(verifyModule(*module));
    const size_t functionRow = funcMap.find("F:4:ledger_device");
    const size_t kernelRow = funcMap.find("K:ledger_kernel@ledger.hip:10");
    const size_t namedRow = funcMap.find("U:5:ledger_scope");
    const size_t systemRow = funcMap.find("P:1:barrier_signal");
    const size_t waveRow = funcMap.find("W:32");
    const size_t globalRow = funcMap.find("P:6:addr_trace_load@ledger.hip:7 -> ledger.hip:20");
    const size_t ldsRow = funcMap.find("P:7:addr_trace_lds_store@ledger.hip:30");
    for (size_t row : {functionRow, kernelRow, namedRow, systemRow, waveRow, globalRow, ldsRow})
        ASSERT_NE(row, std::string::npos) << funcMap;
    EXPECT_LT(functionRow, kernelRow);
    EXPECT_LT(kernelRow, namedRow);
    EXPECT_LT(namedRow, systemRow);
    EXPECT_LT(systemRow, waveRow);
    EXPECT_LT(waveRow, globalRow);
    EXPECT_LT(globalRow, ldsRow);
    EXPECT_EQ(extraPayloadCountForId(funcMap, 6), std::optional<unsigned>(66));
    EXPECT_EQ(extraPayloadCountForId(funcMap, 7), std::optional<unsigned>(34));
}

TEST_F(MarkerPass, AddressTracingCoversBufferProtocolsAcrossWaveSizes)
{
    struct Case
    {
        const char* Cpu;
        const char* WaveSize;
        const char* PermuteName;
        unsigned Lanes;
        bool Barriers;
    };
    for (const Case& test : {Case{"gfx1100", "W:32", "addr_trace_ds_bpermute", 32, true},
                             Case{"gfx90a", "W:64", "addr_trace_ds_permute", 64, false}})
    {
        SCOPED_TRACE(test.Cpu);
        LLVMContext ctx;
        auto module = makeModule(ctx);
        Function* function = makeBufferTraceFunction(*module, test.Cpu, test.Barriers);
        SQTTConfig config = fullScopeConfig();
        config.InstrumentBarriers = test.Barriers;
        config.TraceMemoryAddrs = config.TraceLDSAddrs = true;

        const std::string funcMap = runPassAndGetFuncMap(*module, config);
        expectContains(funcMap, test.WaveSize);
        expectPointEntryWithPayload(funcMap, "addr_trace_buffer_load", 5 + test.Lanes);
        expectPointEntryWithPayload(funcMap, "addr_trace_struct_buffer_store", 5 + 2 * test.Lanes);
        expectPointEntryWithPayload(funcMap, "addr_trace_buffer_atomic", 5 + test.Lanes);
        expectPointEntryWithPayload(funcMap, test.PermuteName, 2 + test.Lanes);
        EXPECT_EQ(countPointEntries(funcMap, "addr_trace_load"), 1u);
        EXPECT_EQ(countPointEntries(funcMap, "addr_trace_store"), 1u);
        EXPECT_EQ(countPointEntries(funcMap, "addr_trace_atomic"), 0u);
        if (test.Barriers)
        {
            EXPECT_LT(funcMap.find("barrier_signal"), funcMap.find(test.WaveSize));
            EXPECT_LT(funcMap.find(test.WaveSize), funcMap.find("addr_trace_buffer_load"));
        }

        const std::string ir = printModule(*module);
        expectContains(ir, "sqtt.lanes.loop");
        expectContains(ir, "s_mov_b32 m0, exec_lo");
        expectContains(ir, "s_nop 0");
        expectContains(ir, "s_ttracedata");
        expectContains(ir, "={m0}");
        bool hasDescriptorPtrToInt = false;
        for (const Function& candidate : *module)
            for (const BasicBlock& block : candidate)
                for (const Instruction& instruction : block)
                    hasDescriptorPtrToInt |= instruction.getOpcode() == Instruction::PtrToInt &&
                                             instruction.getType()->getIntegerBitWidth() == 128 &&
                                             instruction.getOperand(0)->getType()->getPointerAddressSpace() == 8;
        EXPECT_TRUE(hasDescriptorPtrToInt);
        EXPECT_EQ(countIntrinsicCalls(*module, Intrinsic::amdgcn_readlane), 9u);
        std::vector<uint32_t> readlanes;
        forEachCall(*function, [&](const CallInst& call)
        {
            if (const Function* callee = call.getCalledFunction(); callee && callee->getIntrinsicID() == Intrinsic::amdgcn_readlane)
                if (const auto* value = dyn_cast<ConstantInt>(call.getArgOperand(0))) readlanes.push_back(value->getZExtValue());
        });
        EXPECT_EQ(readlanes, (std::vector<uint32_t>{11, 7, 5, 4, 16}));
    }

    LLVMContext scopeCtx;
    auto scopeModule = makeModule(scopeCtx);
    Function* scoped = makeGlobalLoadFunction(*scopeModule, "scoped_address_trace", "gfx1100");
    SQTTConfig scopeConfig = fullScopeConfig();
    scopeConfig.CuMask = 0x1;
    scopeConfig.MemBarrier = MemBarrierMode::Fence;
    scopeConfig.TraceMemoryAddrs = true;
    runPass(*scopeModule, scopeConfig);
    EXPECT_EQ(countFences(*scoped), 2u);
    EXPECT_EQ(countIntrinsicCalls(*scoped, Intrinsic::amdgcn_sched_barrier), 0u);
}

TEST_F(MarkerPass, PayloadMarkersControlGfx12ClockPacking)
{
    Function* function = makeGlobalLoadFunction(*module, "gfx12_address_trace", "gfx1200");
    SQTTConfig config = fullScopeConfig();
    config.MemBarrier = MemBarrierMode::Fence;
    config.TraceMemoryAddrs = true;

    const std::string funcMap = runPassAndGetFuncMap(*module, config);
    expectContains(funcMap, "W:32");
    expectPointEntryWithPayload(funcMap, "addr_trace_load", 66);
    expectNotContains(funcMap, "M:shader_clock_bits=");
    EXPECT_EQ(countIntrinsicCalls(*function, Intrinsic::amdgcn_s_getreg), 0u);
    EXPECT_EQ(countFences(*function), 2u);
    EXPECT_EQ(countIntrinsicCalls(*function, Intrinsic::amdgcn_sched_barrier), 2u);

    // buffer.load.lds must not create an address payload block, so packing remains valid.
    LLVMContext ldsCtx;
    auto ldsModule = makeModule(ldsCtx);
    Function* ldsFunction = makeVoidFunction(*ldsModule, "buffer_load_lds", "gfx1200");
    IRBuilder<> ldsBuilder(ldsFunction->getEntryBlock().getTerminator());
    Function* point = makeNamedMarkerSentinel(*ldsModule, "__sqtt_named_marker_point");
    ldsBuilder.CreateCall(declareFunction(*ldsModule, "llvm.amdgcn.raw.buffer.load.lds", Type::getVoidTy(ldsCtx), {}));
    ldsBuilder.CreateCall(point, {makeMarkerString(*ldsModule, "ordinary_point")});
    // An ordinary numeric shaderdata value must stay legacy even while
    // pass-owned headers carry clock bits.
    insertTraceCallBefore(ldsFunction->getEntryBlock().getTerminator(), encodeMarker(1u << 20, false, false));
    Function* u64Function = makeVoidFunction(*ldsModule, "shader_cycles_u64", "gfx1250");
    IRBuilder<>(u64Function->getEntryBlock().getTerminator()).CreateCall(
        point, {makeMarkerString(*ldsModule, "ordinary_point_u64")}
    );
    SQTTConfig ldsConfig = fullScopeConfig();
    ldsConfig.TraceMemoryAddrs = true;
    ldsConfig.ShaderClockBits = 12;
    const std::string ldsMap = runPassAndGetFuncMap(*ldsModule, ldsConfig);
    expectContains(ldsMap, "M:shader_clock_bits=12;shader_clock_shift=4");
    expectNotContains(ldsMap, "addr_trace_buffer_load");
    expectNotContains(ldsMap, "W:");
    EXPECT_EQ(countIntrinsicCalls(*ldsFunction, Intrinsic::amdgcn_s_getreg), 1u);
    expectContains(printModule(*ldsModule), "s_get_shader_cycles_u64 $0");

    for (bool namedData : {false, true})
    {
        SCOPED_TRACE(namedData ? "named data" : "address block");
        EXPECT_DEATH(
            {
                LLVMContext localCtx;
                auto localModule = makeModule(localCtx);
                SQTTConfig localConfig = fullScopeConfig();
                if (namedData)
                {
                    Type* i32 = Type::getInt32Ty(localCtx);
                    Function* function = makeVoidFunction(*localModule, "gfx12_named_data", "gfx1200");
                    IRBuilder<>(function->getEntryBlock().getTerminator()).CreateCall(
                        declareFunction(*localModule, "__sqtt_named_marker_data", Type::getVoidTy(localCtx),
                                        {PointerType::get(localCtx, 0), i32}),
                        {makeMarkerString(*localModule, "one_payload"), ConstantInt::get(i32, 17)}
                    );
                }
                else
                {
                    makeGlobalLoadFunction(*localModule, "forced_clock_address_trace", "gfx1200");
                    localConfig.TraceMemoryAddrs = true;
                }
                localConfig.ShaderClockBits = 12;
                runPass(*localModule, localConfig);
            },
            "SQTT payload markers require SQTT_SHADER_CLOCK_BITS=0"
        );
    }
}

TEST_F(MarkerPass, ShaderClockPackingLeavesUnregisteredNumericDataUntouched)
{
    Type* i32 = Type::getInt32Ty(ctx);
    Function* function = makeFunction(
        *module, "mixed_clock_headers", "gfx1200", FunctionType::get(i32, {i32}, false)
    );
    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", function));
    Value* dynamicValue = builder.CreateAdd(function->getArg(0), ConstantInt::get(i32, 1));
    Value* result = builder.CreateAdd(dynamicValue, ConstantInt::get(i32, 2));
    ReturnInst* ret = builder.CreateRet(result);

    const uint32_t numericValue = encodeMarker(1u << 20, false, false);
    insertTraceCallBefore(ret, numericValue);
    IRBuilder<>(ret).CreateCall(
        Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_s_ttracedata), {dynamicValue}
    );

    SQTTConfig config = fullScopeConfig();
    config.FunctionThreshold = 1;
    config.ShaderClockBits = 12;
    const std::string funcMap = runPassAndGetFuncMap(*module, config);

    // Only the pass-generated function entry and exit are clock-packed. The
    // constant numeric ID is deliberately too large for the packed layout,
    // and the dynamic value has no funcmap identity at all.
    EXPECT_EQ(countIntrinsicCalls(*function, Intrinsic::amdgcn_s_getreg), 2u);
    expectContains(funcMap, "M:shader_clock_bits=12;shader_clock_shift=4");
    expectContains(funcMap, "F:1:mixed_clock_headers");

    bool foundConstant = false, foundDynamic = false;
    forEachCall(*function, [&](const CallInst& call)
    {
        const auto* asmCall = dyn_cast<InlineAsm>(call.getCalledOperand());
        if (!asmCall || !asmCall->getAsmString().contains("s_ttracedata")) return;
        if (const auto* value = dyn_cast<ConstantInt>(call.getArgOperand(0)))
            foundConstant |= value->getZExtValue() == numericValue;
        foundDynamic |= call.getArgOperand(0) == dynamicValue;
    });
    EXPECT_TRUE(foundConstant);
    EXPECT_TRUE(foundDynamic);
}

TEST_F(MarkerPass, ShaderClockPackingRejectsInvalidLayoutsAndOversizedHeaderIds)
{
    struct LayoutCase
    {
        unsigned Bits;
        unsigned Shift;
        const char* Message;
    };
    for (const LayoutCase& test : {LayoutCase{30, 0, "leave at least one marker ID bit"},
                                   LayoutCase{12, 21, "window must fit"}})
    {
        SCOPED_TRACE(test.Message);
        EXPECT_DEATH(
            {
                LLVMContext localCtx;
                auto localModule = makeModule(localCtx);
                makeVoidFunction(*localModule, "invalid_clock_layout", "gfx1200");
                SQTTConfig localConfig = fullScopeConfig();
                localConfig.ShaderClockBits = test.Bits;
                localConfig.ShaderClockShift = test.Shift;
                runPass(*localModule, localConfig);
            },
            test.Message
        );
    }

    EXPECT_DEATH(
        {
            LLVMContext localCtx;
            auto localModule = makeModule(localCtx);
            Function* function = makeVoidFunction(*localModule, "oversized_clock_id", "gfx1200");
            insertTraceCallBefore(
                function->getEntryBlock().getTerminator(), encodeMarker(1u << 18, false, false), true
            );
            SQTTConfig localConfig = fullScopeConfig();
            localConfig.ShaderClockBits = 12;
            runPass(*localModule, localConfig);
        },
        "marker ID .* does not fit"
    );
}

TEST_F(MarkerPass, NumericMarkerLoweringAndBoundaries)
{
    const uint32_t markerValue = encodeMarker(64, false, false);
    MDNode* traceMetadata = MDNode::get(ctx, {});

    std::vector<Function*> functions;
    for (const auto& [name, cpu] : {
             std::pair{"gfx9_trace",  "gfx90a" },
             std::pair{"gfx10_trace", "gfx1030"},
             std::pair{"gfx12_trace", "gfx1200"}
    })
    {
        Function* function = makeVoidFunction(*module, name, cpu);
        CallInst* trace = insertTraceCallBefore(function->getEntryBlock().getTerminator(), markerValue);
        trace->setMetadata("sqtt.test.trace", traceMetadata);
        functions.push_back(function);
    }

    SQTTConfig config = fullScopeConfig();

    runPass(*module, config);

    for (Function* function : functions)
    {
        const CallInst* trace = findM0NopTrace(*function);
        ASSERT_NE(trace, nullptr) << function->getName().str();
        auto* inlineAsm = dyn_cast<InlineAsm>(trace->getCalledOperand());
        ASSERT_NE(inlineAsm, nullptr);
        EXPECT_EQ(inlineAsm->getConstraintString(), "={m0},i");
        EXPECT_EQ(trace->getMetadata("sqtt.test.trace"), traceMetadata);
    }

    LLVMContext fenceCtx;
    auto fenceModule = makeModule(fenceCtx);
    Function* fenced = makeVoidFunction(*fenceModule, "numeric_marker", "gfx1100");
    insertTraceCallBefore(fenced->getEntryBlock().getTerminator(), encodeMarker(17, false, false));
    SQTTConfig fenceConfig = fullScopeConfig();
    fenceConfig.MemBarrier = MemBarrierMode::Fence;
    runPass(*fenceModule, fenceConfig);
    EXPECT_EQ(countFences(*fenced), 2u);

    LLVMContext scopeCtx;
    auto scopeModule = makeModule(scopeCtx);
    Function* scoped = makeVoidFunction(*scopeModule, "scoped_numeric_markers", "gfx1100");
    Type* scopedI32 = Type::getInt32Ty(scopeCtx);
    IRBuilder<> scopedBuilder(scoped->getEntryBlock().getTerminator());
    FunctionCallee hint = Intrinsic::getOrInsertDeclaration(scopeModule.get(), Intrinsic::amdgcn_sched_barrier);
    FunctionCallee trace = Intrinsic::getOrInsertDeclaration(scopeModule.get(), Intrinsic::amdgcn_s_ttracedata);
    for (uint32_t id : {17u, 18u})
    {
        scopedBuilder.CreateCall(hint, {ConstantInt::get(scopedI32, 0)});
        scopedBuilder.CreateCall(trace, {ConstantInt::get(scopedI32, encodeMarker(id, false, false))});
        scopedBuilder.CreateCall(hint, {ConstantInt::get(scopedI32, 0)});
    }
    scopedBuilder.CreateCall(Intrinsic::getOrInsertDeclaration(scopeModule.get(), Intrinsic::amdgcn_s_barrier));
    SQTTConfig scopeConfig;
    scopeConfig.CuMask = 0x1;
    scopeConfig.MemBarrier = MemBarrierMode::Fence;
    runPass(*scopeModule, scopeConfig);
    size_t traceBlocks = 0;
    for (const BasicBlock& block : *scoped) traceBlocks += block.getName() == "sqtt.trace";
    EXPECT_EQ(traceBlocks, 1u);
    EXPECT_EQ(countFences(*scoped), 4u);
    EXPECT_EQ(countIntrinsicCalls(*scoped, Intrinsic::amdgcn_sched_barrier), 1u);
    expectScopedSkipPin(*scoped);
}

TEST_F(MarkerPass, ScopedMarkerCoalescingKeepsUserSchedulerBarriersUnconditional)
{
    Function* function = makeVoidFunction(*module, "scoped_user_sched_barrier", "gfx1100");
    Instruction* ret = function->getEntryBlock().getTerminator();
    IRBuilder<> builder(ret);
    Type* i32 = Type::getInt32Ty(ctx);
    FunctionCallee trace = Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_s_ttracedata);
    FunctionCallee sched = Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_sched_barrier);
    builder.CreateCall(trace, {ConstantInt::get(i32, encodeMarker(17, false, false))});
    CallInst* userBarrier = builder.CreateCall(sched, {ConstantInt::get(i32, 1)});
    builder.CreateCall(trace, {ConstantInt::get(i32, encodeMarker(18, false, false))});

    SQTTConfig config = fullScopeConfig();
    config.CuMask = 0x1;
    runPass(*module, config);

    EXPECT_FALSE(verifyModule(*module));
    size_t traceBlocks = 0;
    for (const BasicBlock& block : *function) traceBlocks += block.getName().starts_with("sqtt.trace");
    EXPECT_EQ(traceBlocks, 2u);
    EXPECT_FALSE(userBarrier->getParent()->getName().starts_with("sqtt.trace"));
    EXPECT_EQ(countIntrinsicCalls(*function, Intrinsic::amdgcn_sched_barrier), 1u);
}

TEST_F(MarkerPass, ScopedMarkerBoundariesStayUniformAndPayloadsStayAtomic)
{
    SQTTConfig config = fullScopeConfig();
    config.CuMask = 0x1;
    for (const auto& [early, sync] : {std::tuple{false, false}, std::tuple{true, false},
                                      std::tuple{false, true}, std::tuple{true, true}})
        expectScopedMarkerCase(early, sync);

    struct PayloadCase { bool Scoped, TailUse; const char* Name; };
    for (const PayloadCase& test : {PayloadCase{false, false, "full data"},
                                    PayloadCase{true, false, "scoped data"},
                                    PayloadCase{true, true, "scoped data with tail use"}})
    {
        SCOPED_TRACE(test.Name);
        LLVMContext payloadCtx;
        auto payloadModule = makeModule(payloadCtx);
        Type* i32 = Type::getInt32Ty(payloadCtx);
        Function* payloadFunction = makeFunction(*payloadModule, "named_data", "gfx1100",
                                                  FunctionType::get(test.TailUse ? i32 : Type::getVoidTy(payloadCtx),
                                                                    {i32}, false));
        IRBuilder<> payloadBuilder(BasicBlock::Create(payloadCtx, "entry", payloadFunction));
        payloadBuilder.CreateCall(
            declareFunction(*payloadModule, "__sqtt_named_marker_data", Type::getVoidTy(payloadCtx),
                            {PointerType::get(payloadCtx, 0), i32}),
            {makeMarkerString(*payloadModule, "payload"), payloadFunction->getArg(0)}
        );
        ReturnInst* tail = test.TailUse ? payloadBuilder.CreateRet(payloadFunction->getArg(0))
                                        : payloadBuilder.CreateRetVoid();
        SQTTConfig payloadConfig = test.Scoped ? config : fullScopeConfig();
        payloadConfig.MemBarrier = MemBarrierMode::Fence;
        runPass(*payloadModule, payloadConfig, SQTTInstrumentPass::Mode::Early);
        Instruction* userBetween = nullptr;
        if (test.TailUse)
        {
            CallInst* earlyPayload = findTraceWithMetadata(*payloadFunction, "sqtt.raw_payload");
            ASSERT_NE(earlyPayload, nullptr);
            userBetween = cast<Instruction>(
                IRBuilder<>(earlyPayload).CreateAdd(payloadFunction->getArg(0), ConstantInt::get(i32, 1))
            );
            tail->setOperand(0, userBetween);
        }
        runPass(*payloadModule, payloadConfig);

        EXPECT_FALSE(verifyModule(*payloadModule));
        CallInst* header = findTraceWithMetadata(*payloadFunction, "sqtt.marker_header");
        CallInst* payload = findTraceWithMetadata(*payloadFunction, "sqtt.raw_payload");
        ASSERT_NE(header, nullptr);
        ASSERT_NE(payload, nullptr);
        if (test.TailUse)
        {
            EXPECT_FALSE(userBetween->getParent()->getName().starts_with("sqtt.trace"));
            continue;
        }
        ASSERT_EQ(header->getParent(), payload->getParent());
        EXPECT_EQ(countFences(*payloadFunction), 2u);
        EXPECT_EQ(countIntrinsicCalls(*payloadFunction, Intrinsic::amdgcn_sched_barrier), test.Scoped ? 0u : 2u);
        for (Instruction* instruction = header->getNextNode(); instruction != payload;
             instruction = instruction->getNextNode())
        {
            ASSERT_NE(instruction, nullptr);
            EXPECT_FALSE(isa<FenceInst>(instruction));
            auto* call = dyn_cast<CallInst>(instruction);
            Function* callee = call ? call->getCalledFunction() : nullptr;
            EXPECT_FALSE(callee && callee->getIntrinsicID() == Intrinsic::amdgcn_sched_barrier);
            EXPECT_FALSE(callee && (callee->getIntrinsicID() == Intrinsic::amdgcn_s_ttracedata ||
                                    callee->getIntrinsicID() == Intrinsic::amdgcn_s_ttracedata_imm));
            if (auto* asmCall = call ? dyn_cast<InlineAsm>(call->getCalledOperand()) : nullptr)
                EXPECT_EQ(asmCall->getAsmString().find("s_ttracedata"), StringRef::npos);
        }
    }
}

TEST_F(MarkerPass, MemoryInstrumentationPreservesChunksKindsAndGaps)
{
    Type* i32 = Type::getInt32Ty(ctx);
    Function* function = makeFunction(
        *module, "memory_chunks", "gfx1100", FunctionType::get(Type::getVoidTy(ctx), {PointerType::get(ctx, 1)}, false)
    );
    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", function));
    Value* pointer = function->getArg(0);
    Instruction* load1 = builder.CreateLoad(i32, pointer);
    Instruction* load2 = builder.CreateLoad(i32, pointer);
    Instruction* store1 = builder.CreateStore(load1, pointer);
    Instruction* store2 = builder.CreateStore(load2, pointer);
    Instruction* gapLoad1 = builder.CreateLoad(i32, pointer);
    builder.CreateAdd(gapLoad1, ConstantInt::get(i32, 1));
    Instruction* gapLoad2 = builder.CreateLoad(i32, pointer);
    builder.CreateRetVoid();

    SQTTConfig config = fullScopeConfig();
    config.MemoryChunkSize = 2;
    config.MemoryMaxGap = 0;
    const std::string funcMap = runPassAndGetFuncMap(*module, config);
    auto loadID = pointEntryId(funcMap, "vmem_load");
    auto storeID = pointEntryId(funcMap, "vmem_store");
    ASSERT_TRUE(loadID.has_value());
    ASSERT_TRUE(storeID.has_value());
    EXPECT_EQ(*loadID, 1u);
    EXPECT_EQ(*storeID, 2u);

    uint32_t loadMarker = encodeMarker(*loadID, false, false);
    uint32_t storeMarker = encodeMarker(*storeID, false, false);
    EXPECT_EQ(traceMarkerValues(*function), (std::vector<uint32_t>{loadMarker, storeMarker, loadMarker, loadMarker}));
    EXPECT_FALSE(markerAfter(load1));
    EXPECT_EQ(markerAfter(load2), std::optional<uint32_t>(loadMarker));
    EXPECT_FALSE(markerAfter(store1));
    EXPECT_EQ(markerAfter(store2), std::optional<uint32_t>(storeMarker));
    EXPECT_EQ(markerAfter(gapLoad1), std::optional<uint32_t>(loadMarker));
    EXPECT_EQ(markerAfter(gapLoad2), std::optional<uint32_t>(loadMarker));

    LLVMContext scopeCtx;
    auto scopeModule = makeModule(scopeCtx);
    Function* scoped = makeGlobalLoadFunction(*scopeModule, "scoped_memory_chunks", "gfx1100");
    IRBuilder<> scopedBuilder(scoped->getEntryBlock().getTerminator());
    Type* scopedI32 = Type::getInt32Ty(scopeCtx);
    scopedBuilder.CreateLoad(scopedI32, scoped->getArg(0));
    scopedBuilder.CreateLoad(scopedI32, scoped->getArg(0));
    SQTTConfig scopeConfig = fullScopeConfig();
    scopeConfig.CuMask = 0x1;
    scopeConfig.MemoryChunkSize = 1;
    runPass(*scopeModule, scopeConfig);
    EXPECT_EQ(traceMarkerValues(*scoped).size(), 3u);
}

TEST_F(MarkerPass, BarrierInstrumentationHandlesSplitAndStandaloneBarriers)
{
    Type* i32 = Type::getInt32Ty(ctx);
    Type* i16 = Type::getInt16Ty(ctx);

    Function* function = makeVoidFunction(*module, "barrier_traces", "gfx1100");
    Instruction* ret = function->getEntryBlock().getTerminator();
    IRBuilder<> builder(ret);

    FunctionCallee signal = Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_s_barrier_signal);
    FunctionCallee wait = Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_s_barrier_wait);
    FunctionCallee full = Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_s_barrier);
    Function* work = Function::Create(
        FunctionType::get(Type::getVoidTy(ctx), false), GlobalValue::ExternalLinkage, "barrier_work", module.get()
    );

    builder.CreateCall(signal, {ConstantInt::get(i32, 0)});
    builder.CreateCall(wait, {ConstantInt::get(i16, 0)});
    builder.CreateCall(signal, {ConstantInt::get(i32, 0)});
    builder.CreateCall(work);
    builder.CreateCall(wait, {ConstantInt::get(i16, 0)});
    builder.CreateCall(full);

    SQTTConfig config = fullScopeConfig();
    config.InstrumentBarriers = true;
    config.MemoryChunkSize = 1;

    std::string funcMap = runPassAndGetFuncMap(*module, config);
    std::optional<unsigned> signalId = pointEntryId(funcMap, "barrier_signal");
    std::optional<unsigned> waitId = pointEntryId(funcMap, "barrier_wait");
    std::optional<unsigned> fullId = pointEntryId(funcMap, "barrier");
    ASSERT_TRUE(signalId.has_value());
    ASSERT_TRUE(waitId.has_value());
    ASSERT_TRUE(fullId.has_value());
    EXPECT_EQ(*signalId, 1u);
    EXPECT_EQ(*waitId, 2u);
    EXPECT_EQ(*fullId, 3u);
    EXPECT_EQ(pointEntryId(funcMap, "vmem_load"), std::optional<unsigned>(4));
    EXPECT_EQ(pointEntryId(funcMap, "vmem_store"), std::optional<unsigned>(5));

    size_t traceCount = countIntrinsicCalls(*module, Intrinsic::amdgcn_s_ttracedata) +
                        countIntrinsicCalls(*module, Intrinsic::amdgcn_s_ttracedata_imm);
    EXPECT_EQ(traceCount, 4u);
}

TEST_F(MarkerPass, NamedExitEnterFusionRequiresDirectAdjacency)
{
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

    SQTTConfig config = fullScopeConfig();

    runPass(*module, config);

    std::vector<uint32_t> separatedMarkers = traceMarkerValues(*separated);
    ASSERT_EQ(separatedMarkers.size(), 2u);
    EXPECT_EQ(separatedMarkers[0], FLAG_EXIT_PREV);
    EXPECT_EQ(separatedMarkers[1] & FLAG_MASK, FLAG_ENTER);

    std::vector<uint32_t> adjacentMarkers = traceMarkerValues(*adjacent);
    ASSERT_EQ(adjacentMarkers.size(), 1u);
    EXPECT_EQ(adjacentMarkers[0] & FLAG_MASK, FLAG_ENTER | FLAG_EXIT_PREV);
}

TEST_F(MarkerPass, NamedMarkerWrappersInlineAtEveryCallSiteBeforeResolution)
{
    Function* point = makeNamedMarkerSentinel(*module, "__sqtt_named_marker_point");
    Type* ptr = PointerType::get(ctx, 0);
    Function* wrapper = makeFunction(
        *module, "user_marker_wrapper", "gfx1100", FunctionType::get(Type::getVoidTy(ctx), {ptr}, false)
    );
    wrapper->setLinkage(GlobalValue::InternalLinkage);
    IRBuilder<> wrapperBuilder(BasicBlock::Create(ctx, "entry", wrapper));
    wrapperBuilder.CreateCall(point, {wrapper->getArg(0)});
    wrapperBuilder.CreateRetVoid();

    GlobalVariable* name = makeMarkerString(*module, "wrapped_point");
    std::vector<Function*> callers;
    for (StringRef callerName : {StringRef("wrapper_caller_one"), StringRef("wrapper_caller_two")})
    {
        Function* caller = makeVoidFunction(*module, callerName, "gfx1100");
        IRBuilder<>(caller->getEntryBlock().getTerminator()).CreateCall(wrapper, {name});
        callers.push_back(caller);
    }

    SQTTConfig config = fullScopeConfig();
    runPass(*module, config, SQTTInstrumentPass::Mode::Early);

    for (Function* caller : callers)
    {
        EXPECT_EQ(traceMarkerValues(*caller), (std::vector<uint32_t>{encodeMarker(1, false, false)}));
        forEachCall(*caller, [&](const CallInst& call) { EXPECT_NE(call.getCalledFunction(), wrapper); });
    }
    const NamedMDNode* early = module->getNamedMetadata("sqtt.markers.early");
    ASSERT_NE(early, nullptr);
    EXPECT_EQ(early->getNumOperands(), 1u);
    EXPECT_FALSE(verifyModule(*module));
}

TEST_F(MarkerPass, DirectFunctionInstrumentationHandlesO0Fallback)
{
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
    Function* mustTail = makeMustTailFunction(*module, "late_musttail");

    SQTTConfig config = fullScopeConfig();
    config.FunctionThreshold = 3;

    std::string funcMap = runPassAndGetFuncMap(*module, config);
    expectContains(funcMap, "F:1:direct_large");
    expectContains(funcMap, "K:direct_kernel");
    expectNotContains(funcMap, "direct_small");
    expectNotContains(funcMap, "late_musttail");

    std::vector<uint32_t> largeMarkers = traceMarkerValues(*large);
    EXPECT_EQ(std::count(largeMarkers.begin(), largeMarkers.end(), encodeMarker(1, true, false)), 1);
    EXPECT_EQ(std::count(largeMarkers.begin(), largeMarkers.end(), FLAG_EXIT_PREV), 2);
    EXPECT_TRUE(traceMarkerValues(*small).empty());
    EXPECT_TRUE(traceMarkerValues(*kernel).empty());
    EXPECT_TRUE(traceMarkerValues(*mustTail).empty());
}

TEST_F(MarkerPass, FunctionThresholdIgnoresPassMarkersBeforeAndAfterOptimization)
{
    Function* function = makeVoidFunction(*module, "small_function", "gfx1100");

    SQTTConfig config = fullScopeConfig();
    config.FunctionThreshold = 1;

    runPass(*module, config, SQTTInstrumentPass::Mode::Early);

    const NamedMDNode* early = module->getNamedMetadata("sqtt.markers.early");
    ASSERT_NE(early, nullptr);
    auto* size = mdconst::dyn_extract<ConstantInt>(early->getOperand(0)->getOperand(3));
    ASSERT_NE(size, nullptr);
    EXPECT_EQ(size->getZExtValue(), 1u);

    runPass(*module, config);

    EXPECT_TRUE(traceMarkerValues(*function).empty());
    expectNotContains(getFuncMap(*module), "small_function");

    LLVMContext mustTailCtx;
    auto mustTailModule = makeModule(mustTailCtx);
    Function* mustTail = makeMustTailFunction(*mustTailModule, "early_musttail");
    runPass(*mustTailModule, config, SQTTInstrumentPass::Mode::Early);
    EXPECT_TRUE(traceMarkerValues(*mustTail).empty());
    EXPECT_FALSE(mustTailModule->getNamedMetadata("sqtt.markers.early"));
}

TEST_F(MarkerPass, FunctionThresholdPrunesMarkersAndPreservesExistingLlvmUsed)
{
    Type* i32 = Type::getInt32Ty(ctx);
    constexpr uint32_t smallId = 7, largeId = 8;

    Function* small = makeVoidFunction(*module, "small_function", "gfx1100");
    addPassOwnedFunctionMarkers(*small, smallId);
    IRBuilder<> smallBuilder(&*small->getEntryBlock().begin());
    smallBuilder.CreateCall(
        Intrinsic::getOrInsertDeclaration(module.get(), Intrinsic::amdgcn_sched_barrier),
        {ConstantInt::get(i32, 0)}
    );
    addEarlyFunctionMetadata(*small, smallId, 1, "small.hip:3");

    Function* large = makeLargePassOwnedFunction(*module, "large_function", largeId, "large.hip:17");

    constexpr uint32_t cloneId = 101;
    Function* largeClone = makeLargePassOwnedFunction(*module, "large_clone", cloneId, "clone.hip:10");
    Function* smallClone = makeVoidFunction(*module, "small_clone", "gfx1100");
    addPassOwnedFunctionMarkers(*smallClone, cloneId);
    smallClone->setMetadata("sqtt.func.id", MDNode::get(ctx, {ConstantAsMetadata::get(ConstantInt::get(i32, cloneId))}));

    // This unregistered numeric marker happens to use the pruned function's
    // old ID. It must not be removed or rewritten with pass-owned headers.
    Function* numeric = makeVoidFunction(*module, "numeric_marker", "gfx90a");
    insertTraceCallBefore(numeric->getEntryBlock().getTerminator(), encodeMarker(smallId, true, false));

    addEarlyFunctionMapEntry(*module, 99, "inlined_large_function", 40, "inlined.hip:21");
    addEarlyFunctionMapEntry(*module, 100, "inlined_small_function", 1, "inlined.hip:4");
    addExistingLlvmUsed(*module);

    SQTTConfig config = fullScopeConfig();
    config.FunctionThreshold = 20;

    std::string funcMap = runPassAndGetFuncMap(*module, config);
    expectContains(funcMap, "F:1:large_function@large.hip:17");
    expectContains(funcMap, "F:2:inlined_large_function@inlined.hip:21");
    expectNotContains(funcMap, "small_function");
    expectNotContains(funcMap, "inlined_small_function");
    expectNotContains(funcMap, "large_clone");
    const GlobalVariable* used = module->getGlobalVariable("llvm.used");
    ASSERT_NE(used, nullptr);
    const auto* usedValues = dyn_cast<ConstantArray>(used->getInitializer());
    ASSERT_NE(usedValues, nullptr);
    EXPECT_EQ(usedValues->getNumOperands(), 2u);

    EXPECT_EQ(countIntrinsicCalls(*small, Intrinsic::amdgcn_s_ttracedata), 0u);
    EXPECT_EQ(countIntrinsicCalls(*small, Intrinsic::amdgcn_s_ttracedata_imm), 0u);
    EXPECT_EQ(countIntrinsicCalls(*small, Intrinsic::amdgcn_sched_barrier), 1u);
    EXPECT_EQ(countIntrinsicCalls(*largeClone, Intrinsic::amdgcn_s_ttracedata), 0u);
    EXPECT_EQ(countIntrinsicCalls(*smallClone, Intrinsic::amdgcn_s_ttracedata), 0u);
    EXPECT_EQ(findM0NopTrace(*largeClone), nullptr);
    EXPECT_EQ(findM0NopTrace(*smallClone), nullptr);
    const CallInst* numericTrace = findM0NopTrace(*numeric);
    ASSERT_NE(numericTrace, nullptr);
    auto* numericValue = dyn_cast<ConstantInt>(numericTrace->getArgOperand(0));
    ASSERT_NE(numericValue, nullptr);
    EXPECT_EQ(numericValue->getZExtValue(), encodeMarker(smallId, true, false));
}

TEST_F(MarkerPass, LateNamedMarkersReuseTheirCompactedEarlyID)
{
    Type* i32 = Type::getInt32Ty(ctx);
    Function* point = makeNamedMarkerSentinel(*module, "__sqtt_named_marker_point");
    GlobalVariable* name = makeMarkerString(*module, "reused_point");
    Function* function = makeFunction(
        *module, "late_named_marker", "gfx1100", FunctionType::get(Type::getVoidTy(ctx), {i32}, false)
    );
    IRBuilder<> builder(BasicBlock::Create(ctx, "entry", function));
    builder.CreateCall(point, {name});
    Value* value = function->getArg(0);
    for (unsigned i = 0; i < 8; ++i) value = builder.CreateAdd(value, ConstantInt::get(i32, i));
    builder.CreateRetVoid();

    SQTTConfig config = fullScopeConfig();
    config.FunctionThreshold = 1;
    runPass(*module, config, SQTTInstrumentPass::Mode::Early);

    // This models a literal marker exposed only after the early pass. The
    // original declaration has been erased along with its resolved call.
    point = makeNamedMarkerSentinel(*module, "__sqtt_named_marker_point");
    builder.SetInsertPoint(function->getEntryBlock().getTerminator());
    builder.CreateCall(point, {name});

    const std::string funcMap = runPassAndGetFuncMap(*module, config);
    const std::optional<unsigned> id = pointEntryId(funcMap, "reused_point");
    ASSERT_TRUE(id.has_value());
    EXPECT_EQ(countPointEntries(funcMap, "reused_point"), 1u);
    const std::vector<uint32_t> markers = traceMarkerValues(*function);
    EXPECT_EQ(std::count(markers.begin(), markers.end(), encodeMarker(*id, false, false)), 2);
}
