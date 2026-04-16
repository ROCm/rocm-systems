//===-- PayloadCompiler.cpp - LLVM IR Payload Compiler -----------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Compiles trampoline payloads from LLVM IR through the AMDGPU backend.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/PayloadCompiler.h"

#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include <cstring>

using namespace llvm;

extern "C" {
void LLVMInitializeAMDGPUTargetInfo();
void LLVMInitializeAMDGPUTarget();
void LLVMInitializeAMDGPUTargetMC();
void LLVMInitializeAMDGPUAsmPrinter();
void LLVMInitializeAMDGPUAsmParser();
}

namespace aegisbit {

PayloadCompiler::PayloadCompiler() = default;
PayloadCompiler::~PayloadCompiler() = default;

Expected<std::unique_ptr<PayloadCompiler>>
PayloadCompiler::create(StringRef GPUArch) {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
  LLVMInitializeAMDGPUAsmParser();

  // Use mesa3d for compilation: amdgpu_cs is classified as a shader
  // convention and is rejected by the amdhsa backend. The compiled .text
  // bytes are ISA-identical — we only extract raw instruction bytes.
  Triple TT("amdgcn-amd-mesa3d");
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TT, Error);
  if (!TheTarget)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to lookup AMDGPU target: " + Error);

  TargetOptions Options;
  auto TM = std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
      TT, GPUArch, "+wavefrontsize64", Options, Reloc::PIC_,
      CodeModel::Small, CodeGenOptLevel::Default));
  if (!TM)
    return createStringError(inconvertibleErrorCode(),
                             "Failed to create TargetMachine for " +
                                 GPUArch.str());

  auto PC = std::unique_ptr<PayloadCompiler>(new PayloadCompiler());
  PC->Ctx = std::make_unique<LLVMContext>();
  PC->TM = std::move(TM);
  PC->Arch = GPUArch.str();
  return PC;
}

LLVMContext &PayloadCompiler::getContext() { return *Ctx; }

const std::string &PayloadCompiler::getGPUArch() const { return Arch; }

// Strip the compiler-generated function prologue and epilogue.
// For amdgpu_cs (compute shader entry), the epilogue is s_endpgm.
// The prologue may include s_waitcnt and/or mode register setup.
static std::vector<uint8_t> stripPrologueEpilogue(ArrayRef<uint8_t> Bytes) {
  if (Bytes.size() < 4)
    return {Bytes.begin(), Bytes.end()};

  size_t Start = 0;
  size_t End = Bytes.size();

  // Strip leading s_waitcnt: encoding 0xBF8Cxxxx (SOPP, opcode 0x0C)
  if (Bytes.size() >= 4) {
    uint32_t First;
    std::memcpy(&First, Bytes.data(), 4);
    if ((First & 0xFFFF0000u) == 0xBF8C0000u)
      Start = 4;
  }

  // Strip trailing s_endpgm: encoding 0xBF810000 (SOPP, opcode 0x01)
  if (End - Start >= 4) {
    uint32_t Last;
    std::memcpy(&Last, Bytes.data() + End - 4, 4);
    if (Last == 0xBF810000u)
      End -= 4;
  }

  // Also handle s_setpc_b64 at the end (for non-entry-point conventions).
  // SOP1 encoding: [31:23]=101111101 [22:16]=SDST [15:8]=OPCODE [7:0]=SSRC0
  // s_setpc_b64 opcode = 0x1D. Must check specifically to avoid stripping
  // other SOP1 instructions (e.g. s_mov_b32 used by our inline asm).
  if (End - Start >= 4) {
    uint32_t Last;
    std::memcpy(&Last, Bytes.data() + End - 4, 4);
    if ((Last & 0xFF00FF00u) == 0xBE001D00u)
      End -= 4;
  }

  return {Bytes.begin() + Start, Bytes.begin() + End};
}

Expected<std::vector<uint8_t>>
PayloadCompiler::compile(std::unique_ptr<Module> M) {
  M->setDataLayout(TM->createDataLayout());
  M->setTargetTriple(TM->getTargetTriple());

  if (verifyModule(*M, &errs()))
    return createStringError(inconvertibleErrorCode(),
                             "PayloadCompiler: IR module verification failed");

  SmallVector<char, 4096> ObjBuffer;
  raw_svector_ostream ObjStream(ObjBuffer);

  legacy::PassManager PM;
  if (TM->addPassesToEmitFile(PM, ObjStream, nullptr,
                              CodeGenFileType::ObjectFile))
    return createStringError(inconvertibleErrorCode(),
                             "PayloadCompiler: addPassesToEmitFile failed");

  PM.run(*M);

  auto MemBuf =
      MemoryBuffer::getMemBufferCopy(StringRef(ObjBuffer.data(), ObjBuffer.size()));
  auto ObjOrErr =
      object::ObjectFile::createObjectFile(MemBuf->getMemBufferRef());
  if (!ObjOrErr)
    return ObjOrErr.takeError();

  auto &Obj = *ObjOrErr.get();

  for (const auto &Section : Obj.sections()) {
    auto NameOrErr = Section.getName();
    if (!NameOrErr)
      continue;
    if (*NameOrErr != ".text")
      continue;

    auto ContentsOrErr = Section.getContents();
    if (!ContentsOrErr)
      return ContentsOrErr.takeError();

    auto Raw = *ContentsOrErr;
    if (Raw.empty())
      return createStringError(inconvertibleErrorCode(),
                               "PayloadCompiler: .text section is empty");

    return stripPrologueEpilogue(
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(Raw.data()),
                          Raw.size()));
  }

  return createStringError(inconvertibleErrorCode(),
                           "PayloadCompiler: no .text section in compiled object");
}

//===----------------------------------------------------------------------===//
// IR Builders
//===----------------------------------------------------------------------===//

std::unique_ptr<Module>
PayloadCompiler::buildCountingLoop(LLVMContext &Ctx) {
  auto M = std::make_unique<Module>("aegis_counting_loop", Ctx);

  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I1 = Type::getInt1Ty(Ctx);

  // declare i64 @llvm.cttz.i64(i64, i1 immarg)
  Function *CttzI64 =
      Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::cttz, {I64});

  // declare i32 @llvm.amdgcn.readlane.i32(i32, i32)
  Function *ReadLane =
      Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::amdgcn_readlane, {I32});

  Type *Void = Type::getVoidTy(Ctx);

  // define amdgpu_cs void @aegis_count_unique(i32 inreg %exec_lo,
  //                                           i32 inreg %exec_hi,
  //                                           i32 %addr_val)
  // Using amdgpu_cs: no callee-saved registers, no scratch spills.
  // inreg args map to SGPRs (s0, s1), non-inreg to VGPRs (v0).
  // Return value written to v0 via inline asm at exit.
  FunctionType *FTy = FunctionType::get(Void, {I32, I32, I32}, false);
  Function *F = Function::Create(FTy, GlobalValue::ExternalLinkage,
                                 "aegis_count_unique", M.get());
  F->setCallingConv(CallingConv::AMDGPU_CS);
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr("amdgpu-flat-work-group-size", "1,1024");

  // For amdgpu_cs, inreg args come first (SGPRs), then VGPR args.
  // Arg order: s0=exec_lo, s1=exec_hi, v0=addr_val
  auto *ExecLo = F->getArg(0);
  auto *ExecHi = F->getArg(1);
  auto *AddrVal = F->getArg(2);
  ExecLo->setName("exec_lo");
  ExecHi->setName("exec_hi");
  AddrVal->setName("addr_val");
  ExecLo->addAttr(Attribute::InReg);
  ExecHi->addAttr(Attribute::InReg);

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", F);
  BasicBlock *Loop = BasicBlock::Create(Ctx, "loop", F);
  BasicBlock *Process = BasicBlock::Create(Ctx, "process", F);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "exit", F);

  IRBuilder<> B(Ctx);

  // entry:
  B.SetInsertPoint(Entry);
  Value *Lo64 = B.CreateZExt(ExecLo, I64, "lo64");
  Value *Hi64 = B.CreateZExt(ExecHi, I64, "hi64");
  Value *HiShifted = B.CreateShl(Hi64, 32, "hi_shifted");
  Value *Working = B.CreateOr(Lo64, HiShifted, "working");
  B.CreateBr(Loop);

  // loop:
  B.SetInsertPoint(Loop);
  PHINode *W = B.CreatePHI(I64, 2, "w");
  PHINode *Count = B.CreatePHI(I32, 2, "count");
  Value *Done = B.CreateICmpEQ(W, ConstantInt::get(I64, 0), "done");
  B.CreateCondBr(Done, Exit, Process);

  // process:
  B.SetInsertPoint(Process);
  Value *Lane64 = B.CreateCall(CttzI64, {W, ConstantInt::getTrue(I1)}, "lane64");
  Value *Lane = B.CreateTrunc(Lane64, I32, "lane");
  Value *Val = B.CreateCall(ReadLane, {AddrVal, Lane}, "val");
  // Use inline asm to force VOP3 encoding (v_cmp_eq_u32_e64) which writes
  // to an SGPR pair instead of VCC.  The VOPC (e32) encoding writes VCC,
  // and on gfx950 VCC writes inside a trampoline cause corruption that
  // persists even through v_readlane-based VCC restore.
  auto *CmpAsmTy = FunctionType::get(I64, {I32, I32}, false);
  auto *CmpAsm = InlineAsm::get(CmpAsmTy,
      "v_cmp_eq_u32_e64 $0, $1, $2", "=s,s,v", true);
  Value *Match = B.CreateCall(CmpAsm, {Val, AddrVal}, "match");
  Value *MatchMasked = B.CreateAnd(Match, W, "match_masked");
  Value *NotMatch = B.CreateXor(MatchMasked, ConstantInt::getSigned(I64, -1),
                                "not_match");
  Value *WNext = B.CreateAnd(W, NotMatch, "w_next");
  Value *CountNext = B.CreateAdd(Count, ConstantInt::get(I32, 1), "count_next");
  B.CreateBr(Loop);

  // Wire up PHIs
  W->addIncoming(Working, Entry);
  W->addIncoming(WNext, Process);
  Count->addIncoming(ConstantInt::get(I32, 0), Entry);
  Count->addIncoming(CountNext, Process);

  // exit: write count to s0 via inline asm, then return.
  // The count is written to s0 (not v0) to avoid clobbering the address VGPR
  // which the displaced instruction may need. The glue layer reads s0.
  B.SetInsertPoint(Exit);
  auto *AsmFTy = FunctionType::get(Void, {I32}, false);
  auto *IA = InlineAsm::get(AsmFTy, "s_mov_b32 s0, $0", "s", true);
  B.CreateCall(IA, {Count});
  B.CreateRetVoid();

  return M;
}

std::unique_ptr<Module>
PayloadCompiler::buildMaxPopCountLoop(LLVMContext &Ctx) {
  auto M = std::make_unique<Module>("aegis_max_popcount_loop", Ctx);

  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I1 = Type::getInt1Ty(Ctx);
  Type *Void = Type::getVoidTy(Ctx);

  Function *CttzI64 =
      Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::cttz, {I64});
  Function *ReadLane =
      Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::amdgcn_readlane, {I32});
  Function *CtpopI64 =
      Intrinsic::getOrInsertDeclaration(M.get(), Intrinsic::ctpop, {I64});

  FunctionType *FTy = FunctionType::get(Void, {I32, I32, I32}, false);
  Function *F = Function::Create(FTy, GlobalValue::ExternalLinkage,
                                 "aegis_max_popcount", M.get());
  F->setCallingConv(CallingConv::AMDGPU_CS);
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr("amdgpu-flat-work-group-size", "1,1024");

  auto *ExecLo = F->getArg(0);
  auto *ExecHi = F->getArg(1);
  auto *AddrVal = F->getArg(2);
  ExecLo->setName("exec_lo");
  ExecHi->setName("exec_hi");
  AddrVal->setName("addr_val");
  ExecLo->addAttr(Attribute::InReg);
  ExecHi->addAttr(Attribute::InReg);

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", F);
  BasicBlock *Loop = BasicBlock::Create(Ctx, "loop", F);
  BasicBlock *Process = BasicBlock::Create(Ctx, "process", F);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "exit", F);

  IRBuilder<> B(Ctx);

  // entry:
  B.SetInsertPoint(Entry);
  Value *Lo64 = B.CreateZExt(ExecLo, I64, "lo64");
  Value *Hi64 = B.CreateZExt(ExecHi, I64, "hi64");
  Value *HiShifted = B.CreateShl(Hi64, 32, "hi_shifted");
  Value *Working = B.CreateOr(Lo64, HiShifted, "working");
  B.CreateBr(Loop);

  // loop:
  B.SetInsertPoint(Loop);
  PHINode *W = B.CreatePHI(I64, 2, "w");
  PHINode *MaxCount = B.CreatePHI(I32, 2, "max_count");
  Value *Done = B.CreateICmpEQ(W, ConstantInt::get(I64, 0), "done");
  B.CreateCondBr(Done, Exit, Process);

  // process:
  B.SetInsertPoint(Process);
  Value *Lane64 = B.CreateCall(CttzI64, {W, ConstantInt::getTrue(I1)}, "lane64");
  Value *Lane = B.CreateTrunc(Lane64, I32, "lane");
  Value *Val = B.CreateCall(ReadLane, {AddrVal, Lane}, "val");

  auto *CmpAsmTy = FunctionType::get(I64, {I32, I32}, false);
  auto *CmpAsm = InlineAsm::get(CmpAsmTy,
      "v_cmp_eq_u32_e64 $0, $1, $2", "=s,s,v", true);
  Value *Match = B.CreateCall(CmpAsm, {Val, AddrVal}, "match");
  Value *MatchMasked = B.CreateAnd(Match, W, "match_masked");

  // popcount: how many lanes share this value (compiles to s_bcnt1_i32_b64)
  Value *Pop64 = B.CreateCall(CtpopI64, {MatchMasked}, "pop64");
  Value *Pop = B.CreateTrunc(Pop64, I32, "pop");

  // max_next = max(max_count, pop)
  Value *IsBigger = B.CreateICmpUGT(Pop, MaxCount, "is_bigger");
  Value *MaxNext = B.CreateSelect(IsBigger, Pop, MaxCount, "max_next");

  Value *NotMatch = B.CreateXor(MatchMasked, ConstantInt::getSigned(I64, -1),
                                "not_match");
  Value *WNext = B.CreateAnd(W, NotMatch, "w_next");
  B.CreateBr(Loop);

  // Wire up PHIs
  W->addIncoming(Working, Entry);
  W->addIncoming(WNext, Process);
  MaxCount->addIncoming(ConstantInt::get(I32, 0), Entry);
  MaxCount->addIncoming(MaxNext, Process);

  // exit: write max_count to s0
  B.SetInsertPoint(Exit);
  auto *AsmFTy = FunctionType::get(Void, {I32}, false);
  auto *IA = InlineAsm::get(AsmFTy, "s_mov_b32 s0, $0", "s", true);
  B.CreateCall(IA, {MaxCount});
  B.CreateRetVoid();

  return M;
}

std::unique_ptr<Module>
PayloadCompiler::buildAtomicAccumulator(LLVMContext &Ctx, bool UseAtomics) {
  auto M = std::make_unique<Module>("aegis_atomic_accum", Ctx);

  Type *I32 = Type::getInt32Ty(Ctx);
  Type *I8 = Type::getInt8Ty(Ctx);
  Type *Void = Type::getVoidTy(Ctx);
  // Global address space = 1
  PointerType *GlobPtr = PointerType::get(Ctx, 1);

  // define amdgpu_cs void @aegis_atomic_accum(ptr addrspace(1) inreg %base,
  //                                           i32 inreg %count)
  // Using amdgpu_cs: no callee-saved registers, no scratch spills.
  FunctionType *FTy = FunctionType::get(Void, {GlobPtr, I32}, false);
  Function *F = Function::Create(FTy, GlobalValue::ExternalLinkage,
                                 "aegis_atomic_accum", M.get());
  F->setCallingConv(CallingConv::AMDGPU_CS);
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr("amdgpu-flat-work-group-size", "1,1024");

  auto *Base = F->getArg(0);
  auto *CountArg = F->getArg(1);
  Base->setName("base");
  CountArg->setName("count");
  Base->addAttr(Attribute::InReg);
  CountArg->addAttr(Attribute::InReg);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  IRBuilder<> B(BB);

  // Compute pointer to total_samples at base + 4
  Value *SampPtr = B.CreateGEP(I8, Base, ConstantInt::get(I32, 4), "samp_ptr");

  if (UseAtomics) {
    B.CreateAtomicRMW(AtomicRMWInst::Add, Base, CountArg, MaybeAlign(),
                      AtomicOrdering::Monotonic);
    B.CreateAtomicRMW(AtomicRMWInst::Add, SampPtr, ConstantInt::get(I32, 1),
                      MaybeAlign(), AtomicOrdering::Monotonic);
  } else {
    // Non-atomic load-add-store (benign races acceptable for profiling)
    Value *CL = B.CreateLoad(I32, Base, "cl");
    Value *CLNew = B.CreateAdd(CL, CountArg, "cl_new");
    B.CreateStore(CLNew, Base);

    Value *Samp = B.CreateLoad(I32, SampPtr, "samp");
    Value *SampNew = B.CreateAdd(Samp, ConstantInt::get(I32, 1), "samp_new");
    B.CreateStore(SampNew, SampPtr);
  }

  B.CreateRetVoid();
  return M;
}

} // namespace aegisbit
