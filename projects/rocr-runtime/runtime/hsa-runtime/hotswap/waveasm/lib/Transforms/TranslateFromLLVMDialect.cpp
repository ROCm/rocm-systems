// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// TranslateFromLLVM: Strict LLVM dialect -> WaveASM translation.
//
// Consumes gpu.module { llvm.func @kernel ... } with rocdl intrinsics.
// Fails on any unhandled op — no silent fallthrough.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/AssemblyEmitter.h"
#include "waveasm/Transforms/Passes.h"
#include "waveasm/Transforms/TranslateFromMLIR.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-translate-llvm"

using namespace mlir;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMTRANSLATEFROMLLVM
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace waveasm {

//===----------------------------------------------------------------------===//
// LLVM Translation State
//===----------------------------------------------------------------------===//

// AMDGPU SRD (Shader Resource Descriptor) constants.
// SRD is 4 consecutive SGPRs: [base_lo, base_hi|stride, num_records, flags].

/// Mask for SRD word 1 to keep only base_addr[47:32] (lower 16 bits).
static constexpr int64_t kSRDWord1BaseMask = 0xFFFF;
/// Default SRD word 3 flags set by the prologue (OOB_SELECT=2).
static constexpr int64_t kSRDDefaultFlags = 0x20000;
/// Default num_records when buffer size is unknown (max 4-byte-aligned value).
static constexpr int64_t kSRDDefaultNumRecords = 0x7FFFFFFC;

/// Tracks decomposed buffer pointer info from GEP operations.
/// A GEP on ptr<7> decomposes into (SRD, byte-offset-vgpr).
/// TODO: consider a separate decomposition pass for ptr<7>.
struct BufferPtrInfo {
  Value srd;     // The SRD (4×SGPR) from rocdl.make.buffer.rsrc.
  Value voffset; // Byte offset VGPR.
};

/// State for LLVM->WaveASM translation, layered on top of TranslationContext.
class LLVMTranslationState {
public:
  explicit LLVMTranslationState(TranslationContext &ctx) : ctx(ctx) {}

  TranslationContext &ctx;

  /// Map rocdl.make.buffer.rsrc result -> SRD SGPR value from prologue.
  void mapBufferRsrc(Value rsrc, Value srd) { rsrcToSRD[rsrc] = srd; }
  Value lookupSRD(Value rsrc) const { return rsrcToSRD.lookup(rsrc); }

  /// Map GEP result -> decomposed (SRD, voffset).
  void mapGEP(Value gep, BufferPtrInfo info) { gepMap[gep] = info; }
  std::optional<BufferPtrInfo> lookupGEP(Value gep) const {
    auto it = gepMap.find(gep);
    if (it != gepMap.end())
      return it->second;
    return std::nullopt;
  }

  /// Track base-pointer byte offset from bare-pointer GEPs.
  /// These offsets accumulate and get added to voffset when the pointer
  /// is used via make.buffer.rsrc + buffer GEP.
  void setBaseOffset(Value ptr, Value offset) { baseOffsets[ptr] = offset; }
  Value lookupBaseOffset(Value ptr) const { return baseOffsets.lookup(ptr); }

private:
  DenseMap<Value, Value> rsrcToSRD;
  DenseMap<Value, BufferPtrInfo> gepMap;
  DenseMap<Value, Value> baseOffsets;
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Extract workgroup size from llvm.func attributes.
/// Returns failure if both gpu.known_block_size and
/// rocdl.reqd_work_group_size are present and disagree.
static FailureOr<std::tuple<int64_t, int64_t, int64_t>>
getWorkgroupSize(LLVM::LLVMFuncOp func) {
  auto gpuAttr = func->getAttrOfType<DenseI32ArrayAttr>("gpu.known_block_size");
  auto rocdlAttr =
      func->getAttrOfType<DenseI32ArrayAttr>("rocdl.reqd_work_group_size");

  if (gpuAttr && rocdlAttr && gpuAttr.asArrayRef() != rocdlAttr.asArrayRef())
    return func->emitOpError("contradicting workgroup size attributes: "
                             "gpu.known_block_size and "
                             "rocdl.reqd_work_group_size disagree");

  DenseI32ArrayAttr attr = gpuAttr ? gpuAttr : rocdlAttr;
  if (!attr)
    return std::tuple<int64_t, int64_t, int64_t>{64, 1, 1};

  auto vals = attr.asArrayRef();
  int64_t x = vals.size() > 0 ? vals[0] : 64;
  int64_t y = vals.size() > 1 ? vals[1] : 1;
  int64_t z = vals.size() > 2 ? vals[2] : 1;
  return std::tuple<int64_t, int64_t, int64_t>{x, y, z};
}

/// Create a waveasm.program from an llvm.func kernel.
static ProgramOp createProgramFromLLVMFunc(LLVM::LLVMFuncOp func,
                                           OpBuilder &builder,
                                           StringRef targetId) {
  auto *mlirCtx = builder.getContext();
  auto loc = func.getLoc();

  // Code object version 5: supports kernel argument preloading.
  auto targetAttr =
      TargetAttr::get(mlirCtx, getTargetKindAttr(mlirCtx, targetId),
                      /*code_object_version=*/5);
  auto abiAttr = KernelABIAttr::get(mlirCtx, /*tid=*/0, /*kernarg=*/0,
                                    /*wg_id_x=*/std::nullopt,
                                    /*wg_id_y=*/std::nullopt,
                                    /*wg_id_z=*/std::nullopt);

  FailureOr<std::tuple<int64_t, int64_t, int64_t>> wgSize =
      getWorkgroupSize(func);
  if (failed(wgSize))
    return {};
  auto [wgX, wgY, wgZ] = *wgSize;
  std::array<Attribute, 3> sizes = {builder.getI64IntegerAttr(wgX),
                                    builder.getI64IntegerAttr(wgY),
                                    builder.getI64IntegerAttr(wgZ)};

  // Mangle the program name to avoid symbol collision with the original
  // llvm.func (which we keep alive for gpu.launch_func verification).
  // Store the original kernel name for assembly emission.
  std::string programName = (func.getName() + "__waveasm").str();
  auto program =
      ProgramOp::create(builder, loc, programName, targetAttr, abiAttr,
                        /*vgprs=*/int64_t{256},
                        /*sgprs=*/int64_t{104},
                        /*workgroup_size=*/builder.getArrayAttr(sizes),
                        /*lds_size=*/IntegerAttr{});

  program->setAttr(WaveASMDialect::getKernelNameAttrName(),
                   builder.getStringAttr(func.getName()));
  assert(!program.getBody().empty() && "ProgramOp builder must create a block");
  return program;
}

/// Look up the WaveASM value that an LLVM value was translated to.
/// Returns failure if the value was never mapped -- silently returning
/// the original (soon-to-be-erased) LLVM value is a use-after-free bug.
static FailureOr<Value> resolve(Value v, TranslationContext &ctx) {
  if (auto mapped = ctx.getMapper().getMapped(v))
    return *mapped;
  // Block arguments (func params) are mapped during prologue setup.
  // If we get here, an LLVM op was skipped or handled incorrectly.
  return failure();
}

/// Truncate an i64 WaveASM value to i32 via an arith.trunc pseudo-op.
/// Returns the value unchanged if the LLVM source type is already <= 32 bits.
static Value truncToI32(Value v, Type llvmType, OpBuilder &builder,
                        Location loc, TranslationContext &ctx) {
  auto intTy = dyn_cast<IntegerType>(llvmType);
  if (!intTy || intTy.getWidth() <= 32)
    return v;
  Type resTy = isVGPRType(v.getType()) ? (Type)ctx.createVRegType()
                                       : (Type)ctx.createSRegType();
  return ArithTruncOp::create(builder, loc, resTy, v);
}

/// Infer the pseudo-op result type from operand types.
/// If any operand is VGPR -> VReg; otherwise SReg.
static Type inferResultType(ValueRange operands, TranslationContext &ctx) {
  for (Value v : operands)
    if (isVGPRType(v.getType()))
      return ctx.createVRegType();
  return ctx.createSRegType();
}

//===----------------------------------------------------------------------===//
// Op handlers
//===----------------------------------------------------------------------===//

static LogicalResult handlePoison(LLVM::PoisonOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  // Poison is undefined -- materialize as zero.  Must be mapped because
  // downstream ops (e.g. GEP, add) may reference the poison result via
  // resolve(), which now requires every LLVM value to have a mapping.
  auto intType = dyn_cast<IntegerType>(op.getResult().getType());
  if (!intType)
    return op->emitOpError("expected integer poison");

  auto immTy = ctx.createImmType(0);
  auto zeroImm = ConstantOp::create(builder, loc, immTy, int64_t{0});
  auto vregTy = ctx.createVRegType();

  if (intType.getWidth() <= 32) {
    Value mov = V_MOV_B32::create(builder, loc, vregTy, zeroImm);
    ctx.getMapper().mapValue(op.getResult(), mov);
    return success();
  }

  if (intType.getWidth() <= 64) {
    Value loMov = V_MOV_B32::create(builder, loc, vregTy, zeroImm);
    Value hiMov = V_MOV_B32::create(builder, loc, vregTy, zeroImm);
    auto wideTy = ctx.createVRegType(2, 2);
    Value packed =
        PackOp::create(builder, loc, wideTy, ValueRange{loMov, hiMov});
    ctx.getMapper().mapValue(op.getResult(), packed);
    return success();
  }

  return op->emitOpError("unsupported poison width (expected i32 or i64)");
}

static LogicalResult handleConstant(LLVM::ConstantOp op,
                                    LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  auto valAttr = op.getValue();
  int64_t intVal = 0;
  if (auto intAttr = dyn_cast<IntegerAttr>(valAttr))
    intVal = intAttr.getValue().getSExtValue();
  else
    return op->emitOpError("unsupported constant type");

  auto intType = dyn_cast<IntegerType>(op.getResult().getType());
  if (!intType)
    return op->emitOpError("expected integer constant");

  if (intType.getWidth() <= 32) {
    auto immTy = ctx.createImmType(intVal);
    auto immOp = ConstantOp::create(builder, loc, immTy, intVal);
    auto vregTy = ctx.createVRegType();
    auto mov = V_MOV_B32::create(builder, loc, vregTy, immOp);
    ctx.getMapper().mapValue(op.getResult(), mov);
    return success();
  }

  if (intType.getWidth() <= 64) {
    // Split i64 constant into lo/hi halves and pack into vreg<2>.
    int32_t lo = static_cast<int32_t>(intVal & 0xFFFFFFFF);
    int32_t hi = static_cast<int32_t>(static_cast<uint64_t>(intVal) >> 32);
    auto vregTy = ctx.createVRegType();
    auto loImm = ConstantOp::create(builder, loc, ctx.createImmType(lo), lo);
    auto hiImm = ConstantOp::create(builder, loc, ctx.createImmType(hi), hi);
    Value loMov = V_MOV_B32::create(builder, loc, vregTy, loImm);
    Value hiMov = V_MOV_B32::create(builder, loc, vregTy, hiImm);
    auto wideTy = ctx.createVRegType(2, 2);
    Value packed =
        PackOp::create(builder, loc, wideTy, ValueRange{loMov, hiMov});
    ctx.getMapper().mapValue(op.getResult(), packed);
    return success();
  }

  return op->emitOpError("unsupported constant width (expected i32 or i64)");
}

static LogicalResult handleThreadIdX(ROCDL::ThreadIdXOp op,
                                     LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  // rocdl.workitem.id.x -> hardware v0 (flat workitem ID).
  ctx.setUsesWorkitemId(true);
  auto vregTy = ctx.createVRegType();
  auto v0 = PrecoloredVRegOp::create(builder, loc, vregTy, /*regIndex=*/0,
                                     /*size=*/1);
  ctx.getMapper().mapValue(op.getResult(), v0);
  return success();
}

// rocdl.workgroup.id.{x,y,z} -> system SGPRs (set by hardware dispatch).
template <typename OpTy>
static LogicalResult handleWorkgroupId(OpTy op, LLVMTranslationState &st,
                                       int dimIndex) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  if (dimIndex == 0)
    ctx.setUsesWorkgroupIdX(true);
  else if (dimIndex == 1)
    ctx.setUsesWorkgroupIdY(true);
  else
    ctx.setUsesWorkgroupIdZ(true);

  int64_t sgprIndex = ctx.getWorkgroupIdSgprIndex(dimIndex);
  auto sregType = ctx.createSRegType();
  auto blockId = PrecoloredSRegOp::create(builder, loc, sregType, sgprIndex, 1);
  ctx.getMapper().mapValue(op.getResult(), blockId);
  return success();
}

// Emit arith pseudo-ops for i32/i64 casts -- legalization pass handles width.
/// Translate an LLVM cast op to a WaveASM arithmetic pseudo-op.
template <typename LLVMOp, typename WaveASMOp>
static LogicalResult handleCastOp(LLVMOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  FailureOr<Value> src = resolve(op.getOperand(), ctx);
  if (failed(src))
    return op->emitOpError("unmapped operand in cast");
  Type resTy = isVGPRType(src->getType()) ? (Type)ctx.createVRegType()
                                          : (Type)ctx.createSRegType();
  Value pseudo = WaveASMOp::create(builder, op.getLoc(), resTy, *src);
  ctx.getMapper().mapValue(op.getResult(), pseudo);
  return success();
}

/// Map LLVM ICmpPredicate to WaveASM CmpPredicate.
static CmpPredicate mapLLVMPredicate(LLVM::ICmpPredicate pred) {
  using LP = LLVM::ICmpPredicate;
  switch (pred) {
  case LP::eq:
    return CmpPredicate::eq;
  case LP::ne:
    return CmpPredicate::ne;
  case LP::slt:
    return CmpPredicate::slt;
  case LP::sle:
    return CmpPredicate::sle;
  case LP::sgt:
    return CmpPredicate::sgt;
  case LP::sge:
    return CmpPredicate::sge;
  case LP::ult:
    return CmpPredicate::ult;
  case LP::ule:
    return CmpPredicate::ule;
  case LP::ugt:
    return CmpPredicate::ugt;
  case LP::uge:
    return CmpPredicate::uge;
  }
  llvm_unreachable("unhandled LLVM ICmpPredicate");
}

static LogicalResult handleICmp(LLVM::ICmpOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  FailureOr<Value> lhs = resolve(op.getLhs(), ctx);
  FailureOr<Value> rhs = resolve(op.getRhs(), ctx);
  if (failed(lhs) || failed(rhs))
    return op->emitOpError("unmapped operand in icmp");
  Type resTy = inferResultType({*lhs, *rhs}, ctx);
  auto pred = mapLLVMPredicate(op.getPredicate());
  Value cmp = ArithCmpOp::create(builder, op.getLoc(), resTy, pred, *lhs, *rhs);
  ctx.getMapper().mapValue(op.getResult(), cmp);
  return success();
}

static LogicalResult handleSelect(LLVM::SelectOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  FailureOr<Value> cond = resolve(op.getCondition(), ctx);
  FailureOr<Value> trueVal = resolve(op.getTrueValue(), ctx);
  FailureOr<Value> falseVal = resolve(op.getFalseValue(), ctx);
  if (failed(cond) || failed(trueVal) || failed(falseVal))
    return op->emitOpError("unmapped operand in select");
  Type resTy = inferResultType({*trueVal, *falseVal}, ctx);
  // ODS declaration order: (falseVal, trueVal, condition).
  Value sel = ArithSelectOp::create(builder, op.getLoc(), resTy, *falseVal,
                                    *trueVal, *cond);
  ctx.getMapper().mapValue(op.getResult(), sel);
  return success();
}

/// Translate an LLVM binary op to a WaveASM arithmetic pseudo-op.
/// Width validation is deferred to ArithLegalization.
template <typename LLVMOp, typename WaveASMOp>
static LogicalResult handleBinaryOp(LLVMOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  FailureOr<Value> lhs = resolve(op.getLhs(), ctx);
  FailureOr<Value> rhs = resolve(op.getRhs(), ctx);
  if (failed(lhs) || failed(rhs))
    return op->emitOpError("unmapped operand in binary op");
  Type resTy = inferResultType({*lhs, *rhs}, ctx);
  Value result = WaveASMOp::create(builder, op.getLoc(), resTy, *lhs, *rhs);
  ctx.getMapper().mapValue(op.getResult(), result);
  return success();
}

static LogicalResult handleMakeBufferRsrc(ROCDL::MakeBufferRsrcOp op,
                                          LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  // The base pointer was set up as an SRD in the prologue via queueSRDSetup.
  Value basePtr = op.getBase();
  auto srdVal = ctx.getMapper().getMapped(basePtr);
  if (!srdVal)
    return op->emitOpError("SRD not found for base pointer");

  // The prologue used s_mov_b64 to copy the 64-bit pointer into SRD[0:1].
  // This corrupts SRD word 1 bits [31:16] (stride/swizzle) with pointer bits.
  // Also, the prologue hardcodes SRD[3]=0x20000 but make.buffer.rsrc may
  // want different flags. Rebuild the SRD with corrected words via PackOp.
  bool needsPatch = (srdVal->getDefiningOp() != nullptr);
  if (needsPatch) {
    SRegType sregTy = ctx.createSRegType();

    // Extract individual SRD words from the source.
    Value word0 = ExtractOp::create(builder, loc, sregTy, *srdVal, 0);
    Value word1 = ExtractOp::create(builder, loc, sregTy, *srdVal, 1);
    Value word2 = ExtractOp::create(builder, loc, sregTy, *srdVal, 2);
    Value word3 = ExtractOp::create(builder, loc, sregTy, *srdVal, 3);

    // Clear stride/swizzle bits in SRD word 1 (keep only base_addr[47:32]).
    auto maskImm = ctx.createImmType(kSRDWord1BaseMask);
    auto maskVal = ConstantOp::create(builder, loc, maskImm, kSRDWord1BaseMask);
    word1 = S_AND_B32::create(builder, loc, sregTy, ctx.createSCCType(), word1,
                              maskVal)
                .getDst();

    // Patch SRD[3] with the actual flags from make.buffer.rsrc.
    auto flags = getConstantIntValue(op.getFlags());
    if (flags && *flags != kSRDDefaultFlags) {
      auto flagsImm = ctx.createImmType(*flags);
      auto flagsVal = ConstantOp::create(builder, loc, flagsImm, *flags);
      word3 = S_MOV_B32::create(builder, loc, sregTy, flagsVal);
    }

    // If bare-pointer GEPs accumulated a byte offset before make.buffer.rsrc,
    // fold it into the SRD base address (64-bit SALU add to SRD[0:1]).
    Value baseOff = st.lookupBaseOffset(basePtr);
    if (baseOff) {
      // Make the offset scalar, preserving width (i32 or i64).
      int64_t width = getRegSize(baseOff.getType());
      SRegType scalarTy = ctx.createSRegType(width, width);
      Value offScalar =
          ArithReadFirstLaneOp::create(builder, loc, scalarTy, baseOff);

      // Split into lo/hi 32-bit halves for the 64-bit SRD base add.
      Value offLo = ArithTruncOp::create(builder, loc, sregTy, offScalar);
      Value offHi;
      if (width > 1)
        offHi = ExtractOp::create(builder, loc, sregTy, offScalar, 1);
      else
        offHi = ConstantOp::create(builder, loc, ctx.createImmType(0), 0);

      // Adjust base: S_ADD_U32 sets SCC, S_ADDC_U32 reads it.
      SCCType sccTy = ctx.createSCCType();
      auto addLo = S_ADD_U32::create(builder, loc, sregTy, sccTy, word0, offLo);
      word0 = addLo.getDst();
      word1 = S_ADDC_U32::create(builder, loc, sregTy, sccTy, addLo.getScc(),
                                 word1, offHi)
                  .getDst();
    }

    // Pack into a 4-wide SGPR SRD.
    auto srdType = ctx.createSRegType(4, 4);
    Value newSrd = PackOp::create(builder, loc, srdType,
                                  ValueRange{word0, word1, word2, word3});
    st.mapBufferRsrc(op.getResult(), newSrd);
  } else {
    st.mapBufferRsrc(op.getResult(), *srdVal);
  }

  return success();
}

static LogicalResult handleGEP(LLVM::GEPOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();
  Value base = op.getBase();

  // GEP index is a dynamic Value (not a constant attr).
  auto indices = op.getIndices();
  if (indices.size() != 1)
    return op->emitOpError("GEP with multiple indices not yet supported");
  auto idx = indices[0].dyn_cast<Value>();
  if (!idx)
    return op->emitOpError("GEP with constant index attr not yet supported");

  FailureOr<Value> resolved = resolve(idx, ctx);
  if (failed(resolved))
    return op->emitOpError("unmapped GEP index");
  Value newOffset = *resolved;

  // Bare-pointer GEP (!llvm.ptr, not <7>): 64-bit pointer arithmetic before
  // make.buffer.rsrc. Propagate the mapper entry and accumulate the byte
  // offset so it can be folded into the SRD base later.
  auto baseTy = op.getBase().getType();
  unsigned addrSpace = 0;
  if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(baseTy))
    addrSpace = ptrTy.getAddressSpace();

  if (addrSpace != 0 && addrSpace != 7)
    return op->emitOpError("unsupported address space ") << addrSpace;

  if (addrSpace == 0) {
    // Forward mapper entry so make.buffer.rsrc can find the SRD.
    if (auto mapped = ctx.getMapper().getMapped(base))
      ctx.getMapper().mapValue(op.getResult(), *mapped);

    // Accumulate base offset with 64-bit add (pointer arithmetic).
    Value prevOffset = st.lookupBaseOffset(base);
    if (prevOffset) {
      Type resTy = inferResultType({prevOffset, newOffset}, ctx);
      newOffset =
          ArithAddOp::create(builder, loc, resTy, prevOffset, newOffset);
    }
    st.setBaseOffset(op.getResult(), newOffset);
    return success();
  }

  // Buffer voffsets are 32-bit. Truncate i64 GEP indices.
  newOffset = truncToI32(newOffset, idx.getType(), builder, loc, ctx);

  // Buffer GEP (ptr<7>): decompose into (SRD, voffset).
  // Check gepMap first -- covers both chained buffer GEPs and
  // make.buffer.rsrc entries seeded with a bare-pointer base offset.
  if (std::optional<BufferPtrInfo> baseGEP = st.lookupGEP(base)) {
    auto vregTy = ctx.createVRegType();
    Value sum =
        V_ADD_U32::create(builder, loc, vregTy, baseGEP->voffset, newOffset);
    st.mapGEP(op.getResult(), {baseGEP->srd, sum});
    return success();
  }

  // First buffer GEP directly on make.buffer.rsrc (no bare-pointer offset).
  Value srd = st.lookupSRD(base);
  if (srd) {
    st.mapGEP(op.getResult(), {srd, newOffset});
    return success();
  }

  return op->emitOpError("GEP base is not a tracked buffer resource");
}

/// Compute buffer load/store size from the LLVM element type.
static int64_t getBufferAccessBytes(Type ty) {
  if (auto vecTy = dyn_cast<VectorType>(ty))
    return vecTy.getNumElements() *
           vecTy.getElementType().getIntOrFloatBitWidth() / 8;
  if (ty.isIntOrFloat())
    return ty.getIntOrFloatBitWidth() / 8;
  return 0;
}

static LogicalResult handleLoad(LLVM::LoadOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  std::optional<BufferPtrInfo> ptr = st.lookupGEP(op.getAddr());
  if (!ptr)
    return op->emitOpError("load address not from a tracked GEP");

  int64_t numBytes = getBufferAccessBytes(op.getResult().getType());

  auto soffsetTy = ctx.createImmType(0);
  auto zeroSoffset = ConstantOp::create(builder, loc, soffsetTy, 0);
  auto vregTy = ctx.createVRegType();

  Operation *loadOp = nullptr;
  if (numBytes == 2)
    loadOp = BUFFER_LOAD_USHORT::create(builder, loc, TypeRange{vregTy},
                                        ptr->srd, ptr->voffset, zeroSoffset,
                                        /*instOffset=*/0);
  else if (numBytes == 4)
    loadOp =
        BUFFER_LOAD_DWORD::create(builder, loc, TypeRange{vregTy}, ptr->srd,
                                  ptr->voffset, zeroSoffset, /*instOffset=*/0);
  else if (numBytes == 8) {
    auto wideTy = ctx.createVRegType(2, 2);
    loadOp = BUFFER_LOAD_DWORDX2::create(builder, loc, TypeRange{wideTy},
                                         ptr->srd, ptr->voffset, zeroSoffset,
                                         /*instOffset=*/0);
  } else if (numBytes == 12) {
    auto wideTy = ctx.createVRegType(3, 3);
    loadOp = BUFFER_LOAD_DWORDX3::create(builder, loc, TypeRange{wideTy},
                                         ptr->srd, ptr->voffset, zeroSoffset,
                                         /*instOffset=*/0);
  } else if (numBytes == 16) {
    auto wideTy = ctx.createVRegType(4, 4);
    loadOp = BUFFER_LOAD_DWORDX4::create(builder, loc, TypeRange{wideTy},
                                         ptr->srd, ptr->voffset, zeroSoffset,
                                         /*instOffset=*/0);
  } else
    return op->emitOpError("unsupported load size: ") << numBytes << " bytes";

  ctx.getMapper().mapValue(op.getResult(), loadOp->getResult(0));
  return success();
}

static LogicalResult handleStore(LLVM::StoreOp op, LLVMTranslationState &st) {
  auto &ctx = st.ctx;
  auto &builder = ctx.getBuilder();
  auto loc = op.getLoc();

  std::optional<BufferPtrInfo> ptr = st.lookupGEP(op.getAddr());
  if (!ptr)
    return op->emitOpError("store address not from a tracked GEP");

  FailureOr<Value> data = resolve(op.getValue(), ctx);
  if (failed(data))
    return op->emitOpError("unmapped store value");
  int64_t numBytes = getBufferAccessBytes(op.getValue().getType());

  if (numBytes == 2)
    BUFFER_STORE_SHORT::create(builder, loc, *data, ptr->srd, ptr->voffset,
                               /*instOffset=*/0);
  else if (numBytes == 4)
    BUFFER_STORE_DWORD::create(builder, loc, *data, ptr->srd, ptr->voffset,
                               /*instOffset=*/0);
  else if (numBytes == 8)
    BUFFER_STORE_DWORDX2::create(builder, loc, *data, ptr->srd, ptr->voffset,
                                 /*instOffset=*/0);
  else if (numBytes == 12)
    BUFFER_STORE_DWORDX3::create(builder, loc, *data, ptr->srd, ptr->voffset,
                                 /*instOffset=*/0);
  else if (numBytes == 16)
    BUFFER_STORE_DWORDX4::create(builder, loc, *data, ptr->srd, ptr->voffset,
                                 /*instOffset=*/0);
  else
    return op->emitOpError("unsupported store size: ") << numBytes << " bytes";

  return success();
}

//===----------------------------------------------------------------------===//
// Op dispatch
//===----------------------------------------------------------------------===//

static LogicalResult translateOp(Operation *op, LLVMTranslationState &st) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](LLVM::ConstantOp o) { return handleConstant(o, st); })
      .Case([&](LLVM::PoisonOp o) { return handlePoison(o, st); })
      .Case([&](ROCDL::ThreadIdXOp o) { return handleThreadIdX(o, st); })
      .Case([&](ROCDL::BlockIdXOp o) { return handleWorkgroupId(o, st, 0); })
      .Case([&](ROCDL::BlockIdYOp o) { return handleWorkgroupId(o, st, 1); })
      .Case([&](ROCDL::BlockIdZOp o) { return handleWorkgroupId(o, st, 2); })
      .Case([&](LLVM::SExtOp o) {
        return handleCastOp<LLVM::SExtOp, ArithSExtOp>(o, st);
      })
      .Case([&](LLVM::ZExtOp o) {
        return handleCastOp<LLVM::ZExtOp, ArithZExtOp>(o, st);
      })
      .Case([&](LLVM::TruncOp o) {
        return handleCastOp<LLVM::TruncOp, ArithTruncOp>(o, st);
      })
      .Case([&](LLVM::ICmpOp o) { return handleICmp(o, st); })
      .Case([&](LLVM::SelectOp o) { return handleSelect(o, st); })
      .Case([&](LLVM::MulOp o) {
        return handleBinaryOp<LLVM::MulOp, ArithMulOp>(o, st);
      })
      .Case([&](LLVM::AddOp o) {
        return handleBinaryOp<LLVM::AddOp, ArithAddOp>(o, st);
      })
      .Case([&](LLVM::OrOp o) {
        return handleBinaryOp<LLVM::OrOp, ArithOrOp>(o, st);
      })
      .Case([&](LLVM::AndOp o) {
        return handleBinaryOp<LLVM::AndOp, ArithAndOp>(o, st);
      })
      .Case([&](ROCDL::MakeBufferRsrcOp o) {
        return handleMakeBufferRsrc(o, st);
      })
      .Case([&](LLVM::GEPOp o) { return handleGEP(o, st); })
      .Case([&](LLVM::LoadOp o) { return handleLoad(o, st); })
      .Case([&](LLVM::StoreOp o) { return handleStore(o, st); })
      .Default([](Operation *op) {
        return op->emitOpError("unhandled op in LLVM->WaveASM translation");
      });
}

//===----------------------------------------------------------------------===//
// Core translation logic
//===----------------------------------------------------------------------===//

static LogicalResult translateLLVMModule(Operation *rootOp,
                                         StringRef targetId) {
  auto target = getTargetKindAttr(rootOp->getContext(), targetId);
  if (!target)
    return rootOp->emitError() << "unknown target: " << targetId;

  if (!isa<GFX950TargetAttr>(target))
    return rootOp->emitError()
           << "LLVM->WaveASM translation only supports gfx950, got "
           << targetId;

  SmallVector<LLVM::LLVMFuncOp> kernels;
  rootOp->walk([&](LLVM::LLVMFuncOp func) {
    if (func->hasAttr("gpu.kernel") || func->hasAttr("rocdl.kernel"))
      kernels.push_back(func);
  });

  if (kernels.empty())
    return success();

  for (LLVM::LLVMFuncOp func : kernels) {
    OpBuilder builder(rootOp->getContext());
    builder.setInsertionPointAfter(func);

    ProgramOp program = createProgramFromLLVMFunc(func, builder, targetId);
    if (!program)
      return failure();
    builder.setInsertionPointToStart(&program.getBodyBlock());
    TranslationContext ctx(builder, program, target);
    LLVMTranslationState st(ctx);

    // Map llvm.func arguments: pointers get SRD setup, scalars get mapped
    // to their preloaded SGPR positions directly.
    SmallVector<BlockArgument> scalarArgs;
    for (auto arg : func.getBody().getArguments()) {
      if (isa<LLVM::LLVMPointerType>(arg.getType())) {
        int64_t argIdx = arg.getArgNumber();
        ctx.queueSRDSetup(arg, argIdx, /*bufferSize=*/kSRDDefaultNumRecords);
      } else {
        scalarArgs.push_back(arg);
        ctx.queueScalarArgLoad(arg, arg.getArgNumber());
      }
    }

    ctx.emitSRDPrologue();

    // Map scalar (non-pointer) args to their SGPR positions.
    // gfx950 hardware preloads arg N into s[2+N*2 : 2+N*2+1] (64-bit each).
    // Assumes all scalar args fit in the preload window (no overflow).
    for (auto arg : scalarArgs) {
      int64_t argIdx = arg.getArgNumber();
      int64_t preloadBase = 2 + argIdx * 2;
      auto sregTy = ctx.createSRegType(2, 2);
      auto sreg = PrecoloredSRegOp::create(builder, arg.getLoc(), sregTy,
                                           preloadBase, /*size=*/2);
      ctx.getMapper().mapValue(arg, sreg);
    }

    // Enable all workgroup IDs so the SGPR layout is predictable.
    // Note: LLVM enables them selectively via amdgpu-no-workgroup-id-{y,z}
    // attributes. We enable all three unconditionally for simplicity.
    ctx.enableAllWorkgroupIds();

    for (Operation &op : func.getBody().front()) {
      if (isa<LLVM::ReturnOp>(op))
        continue;
      if (failed(translateOp(&op, st)))
        return failure();
    }

    S_ENDPGM::create(builder, func.getLoc());

    program->setAttr("num_kernel_args",
                     builder.getI64IntegerAttr(func.getNumArguments()));

    int64_t ldsSize = ctx.getTotalLDSSize();
    if (ldsSize > 0)
      program->setAttr("lds_size", builder.getI64IntegerAttr(ldsSize));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

namespace {

struct WAVEASMTranslateFromLLVMPass
    : impl::WAVEASMTranslateFromLLVMBase<WAVEASMTranslateFromLLVMPass> {
  using WAVEASMTranslateFromLLVMBase::WAVEASMTranslateFromLLVMBase;

  void runOnOperation() override {
    if (failed(translateLLVMModule(getOperation(), targetArch)))
      return signalPassFailure();
  }
};

} // namespace

} // namespace waveasm
