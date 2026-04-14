// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_TRANSLATEFROMMLIR_H
#define WaveASM_TRANSFORMS_TRANSLATEFROMMLIR_H

#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <functional>
#include <optional>

namespace waveasm {

//===----------------------------------------------------------------------===//
// Value Mapping (Pure SSA)
//===----------------------------------------------------------------------===//

/// Maps MLIR values from input dialect to waveasm SSA values
class ValueMapper {
public:
  ValueMapper() = default;

  /// Map an MLIR value to an waveasm value
  void mapValue(mlir::Value inputValue, mlir::Value asmValue) {
    valueMap[inputValue] = asmValue;
  }

  /// Get the mapped value for an MLIR value
  std::optional<mlir::Value> getMapped(mlir::Value mlirValue) const {
    auto it = valueMap.find(mlirValue);
    if (it != valueMap.end())
      return it->second;
    return std::nullopt;
  }

  /// Check if a value is mapped
  bool isMapped(mlir::Value mlirValue) const {
    return valueMap.contains(mlirValue);
  }

private:
  llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
};

//===----------------------------------------------------------------------===//
// Expression Key for CSE in Expression Emitter
//===----------------------------------------------------------------------===//

/// Key for identifying equivalent expressions during translation
struct ExprKey {
  llvm::StringRef opName;
  llvm::SmallVector<mlir::Value, 4> operands;
  llvm::SmallVector<int64_t, 2> constants;

  bool operator==(const ExprKey &rhs) const {
    return opName == rhs.opName && operands == rhs.operands &&
           constants == rhs.constants;
  }
};

//===----------------------------------------------------------------------===//
// SRD (Shader Resource Descriptor) Info
//===----------------------------------------------------------------------===//

/// Information about a buffer's SRD setup
struct SRDInfo {
  mlir::Value baseSGPR; // SGPR pair holding base address
  mlir::Value srdSGPR;  // SGPR quad holding full SRD
  int64_t bindingIndex; // Which kernel argument this refers to
  int64_t bufferSize;   // Buffer size in bytes (for bounds checking)
  int64_t elementSize;  // Element size in bytes

  SRDInfo() : bindingIndex(-1), bufferSize(0), elementSize(4) {}
  SRDInfo(mlir::Value base, mlir::Value srd, int64_t binding, int64_t size,
          int64_t elemSize)
      : baseSGPR(base), srdSGPR(srd), bindingIndex(binding), bufferSize(size),
        elementSize(elemSize) {}
};

//===----------------------------------------------------------------------===//
// Bit Range for OR vs ADD optimization
//===----------------------------------------------------------------------===//

/// Represents a range of bits [lowBit, highBit] that an expression can occupy.
/// Used to determine if two additions can be fused into OR when bits don't
/// overlap.
struct BitRange {
  int64_t lowBit;  // Lowest bit index used (0-based)
  int64_t highBit; // Highest bit index used (0-based)

  BitRange() : lowBit(0), highBit(31) {} // Default: full 32-bit range
  BitRange(int64_t low, int64_t high) : lowBit(low), highBit(high) {}

  /// Check if this range overlaps with another
  bool overlaps(const BitRange &other) const {
    return highBit >= other.lowBit && other.highBit >= lowBit;
  }

  /// Merge two non-overlapping ranges (for OR result)
  BitRange merge(const BitRange &other) const {
    return BitRange(std::min(lowBit, other.lowBit),
                    std::max(highBit, other.highBit));
  }

  /// Extend range for ADD result (accounts for carry)
  BitRange extendForAdd(const BitRange &other) const {
    return BitRange(std::min(lowBit, other.lowBit),
                    std::max(highBit, other.highBit) + 1);
  }

  /// Shift range left by n bits
  BitRange shiftLeft(int64_t n) const {
    return BitRange(lowBit + n, highBit + n);
  }

  /// Shift range right by n bits
  BitRange shiftRight(int64_t n) const {
    return BitRange(std::max(int64_t(0), lowBit - n),
                    std::max(int64_t(0), highBit - n));
  }

  /// Create range from a constant value.
  /// Negative constants use their unsigned 32-bit representation so the
  /// OR-for-add optimisation in AffineHandlers sees the actually-set bits.
  static BitRange fromConstant(int64_t value) {
    if (value == 0)
      return BitRange(0, 0);
    // Work with the 32-bit unsigned representation so negative values
    // (e.g. -16 = 0xFFFFFFF0) are handled correctly.
    uint32_t uval = static_cast<uint32_t>(value);
    if (uval == 0)
      return BitRange(0, 0);
    int low = llvm::countr_zero(uval);
    int high = 31 - llvm::countl_zero(uval);
    return BitRange(low, high);
  }

  /// Create range for a value with known max
  static BitRange fromMaxValue(int64_t maxValue) {
    if (maxValue <= 0)
      return BitRange(0, 0);
    int64_t bits = 0;
    int64_t v = maxValue;
    while (v > 0) {
      bits++;
      v >>= 1;
    }
    return BitRange(0, bits - 1);
  }
};

//===----------------------------------------------------------------------===//
// Operation Handlers
//===----------------------------------------------------------------------===//

class TranslationContext;

/// Handler function type for translating operations
using OpHandler =
    std::function<mlir::LogicalResult(mlir::Operation *, TranslationContext &)>;

/// Registry of operation handlers
class OpHandlerRegistry {
public:
  explicit OpHandlerRegistry(mlir::MLIRContext *ctx);

  /// Register a handler for an operation
  void registerHandler(mlir::OperationName opName, OpHandler handler);

  /// Get the handler for an operation
  std::optional<OpHandler> getHandler(mlir::OperationName opName) const;

  /// Check if an operation has a handler
  bool hasHandler(mlir::OperationName opName) const;

  /// Get the default registry with built-in handlers
  static OpHandlerRegistry &getDefault();

private:
  void registerDefaultHandlers(mlir::MLIRContext *ctx);

  llvm::DenseMap<mlir::OperationName, OpHandler> handlers;
};

//===----------------------------------------------------------------------===//
// Translation Options
//===----------------------------------------------------------------------===//

/// Options for MLIR to waveasm translation
struct TranslationOptions {
  /// Target architecture (gfx942, gfx950, gfx1250)
  std::string targetId = "gfx942";

  /// Workgroup size (x, y, z). If any dimension is 0, use defaults.
  int64_t workgroupSizeX = 0;
  int64_t workgroupSizeY = 0;
  int64_t workgroupSizeZ = 0;

  /// Subgroup (wavefront) size
  int64_t subgroupSize = 64;

  /// Check if workgroup size is explicitly specified
  bool hasExplicitWorkgroupSize() const {
    return workgroupSizeX > 0 || workgroupSizeY > 0 || workgroupSizeZ > 0;
  }

  /// Get normalized workgroup size (with defaults for 0 values)
  std::tuple<int64_t, int64_t, int64_t> getWorkgroupSize() const {
    return {workgroupSizeX > 0 ? workgroupSizeX : 64,
            workgroupSizeY > 0 ? workgroupSizeY : 1,
            workgroupSizeZ > 0 ? workgroupSizeZ : 1};
  }
};

//===----------------------------------------------------------------------===//
// Translation Context (Pure SSA)
//===----------------------------------------------------------------------===//

/// Context for translating MLIR to waveasm pure SSA IR
class TranslationContext {
public:
  TranslationContext(mlir::OpBuilder &builder, ProgramOp program,
                     TargetAttrInterface target,
                     const TranslationOptions &options = TranslationOptions());

  /// Get the MLIR builder
  mlir::OpBuilder &getBuilder() { return builder; }

  /// Get the target program
  ProgramOp getProgram() { return program; }

  /// Get the target architecture
  TargetAttrInterface getTarget() { return target; }

  /// Get the value mapper
  ValueMapper &getMapper() { return mapper; }

  /// Create a virtual VGPR type
  VRegType createVRegType(int64_t size = 1, int64_t alignment = 1);

  /// Create a virtual SGPR type
  SRegType createSRegType(int64_t size = 1, int64_t alignment = 1);

  /// Create a virtual AGPR type
  ARegType createARegType(int64_t size = 1, int64_t alignment = 1);

  /// Create an SCC flag type
  SCCType createSCCType();

  /// Create an immediate type with a value
  ImmType createImmType(int64_t value);

  /// Emit a comment
  CommentOp emitComment(llvm::StringRef text);

  /// Emit a raw assembly line
  RawOp emitRaw(llvm::StringRef code);

  /// Get the current insertion point
  mlir::Block::iterator getInsertionPoint();

  /// Track binding information for IREE stream.binding.subspan
  void trackBinding(mlir::Value value, int64_t bindingIndex) {
    bindingMap[value] = bindingIndex;
  }

  /// Get binding index for a value
  std::optional<int64_t> getBindingIndex(mlir::Value value) const {
    auto it = bindingMap.find(value);
    if (it != bindingMap.end())
      return it->second;
    return std::nullopt;
  }

  /// Set cache swizzle stride for a buffer
  void setCacheSwizzleStride(mlir::Value value, int64_t stride) {
    cacheSwizzleMap[value] = stride;
  }

  /// Get cache swizzle stride for a buffer
  std::optional<int64_t> getCacheSwizzleStride(mlir::Value value) const {
    auto it = cacheSwizzleMap.find(value);
    if (it != cacheSwizzleMap.end())
      return it->second;
    return std::nullopt;
  }

  //===--------------------------------------------------------------------===//
  // SRD Management
  //===--------------------------------------------------------------------===//

  /// Register an SRD for a memref binding
  void registerSRD(mlir::Value memref, const SRDInfo &info) {
    srdMap[memref] = info;
  }

  /// Get SRD info for a memref
  std::optional<SRDInfo> getSRD(mlir::Value memref) const {
    auto it = srdMap.find(memref);
    if (it != srdMap.end())
      return it->second;
    return std::nullopt;
  }

  /// Check if SRD is already set up for a memref
  bool hasSRD(mlir::Value memref) const { return srdMap.contains(memref); }

  //===--------------------------------------------------------------------===//
  // Expression CSE (Common Subexpression Elimination)
  //===--------------------------------------------------------------------===//

  /// Check if an expression result is already cached
  std::optional<mlir::Value>
  getCachedExpr(llvm::StringRef opName, mlir::ValueRange operands,
                llvm::ArrayRef<int64_t> constants = {}) const;

  /// Cache an expression result
  void cacheExpr(llvm::StringRef opName, mlir::ValueRange operands,
                 llvm::ArrayRef<int64_t> constants, mlir::Value result);

  /// Clear expression cache (e.g., at loop boundaries)
  void clearExprCache() { exprCache.clear(); }

  //===--------------------------------------------------------------------===//
  // Pending SRD Setup (Deferred to Prologue)
  //===--------------------------------------------------------------------===//

  /// Information about a pending SRD setup
  struct PendingSRD {
    mlir::Value memref;   // The memref value this SRD is for
    int64_t argIndex;     // Which kernel argument (0, 1, 2, ...)
    int64_t bufferSize;   // Size in bytes for SRD[2]
    int64_t srdBaseIndex; // SGPR index for SRD (e.g., 8 for s[8:11])
  };

  /// Information about a pending scalar kernel argument load (index, i32, etc.)
  struct PendingScalarArg {
    mlir::Value blockArg; // The MLIR block argument
    int64_t argIndex;     // Position in function signature
  };

  /// Queue an SRD setup for a binding
  void queueSRDSetup(mlir::Value memref, int64_t argIndex, int64_t bufferSize);

  /// Queue a scalar argument load from the kernarg buffer
  void queueScalarArgLoad(mlir::Value blockArg, int64_t argIndex);

  /// Emit all pending SRD setup instructions (called at start of kernel body)
  void emitSRDPrologue();

  /// Get the SRD SGPR index for a memref
  std::optional<int64_t> getSRDIndex(mlir::Value memref) const;

  /// Set SRD base index for a memref
  void setSRDIndex(mlir::Value memref, int64_t srdBaseIndex);

  /// Get the SRD SSA value for a memref (from PackOp-based construction).
  std::optional<mlir::Value> getSRDValue(mlir::Value memref) const;

  /// Set SRD SSA value for a memref.
  void setSRDValue(mlir::Value memref, mlir::Value srd);

  /// Get next available SRD index
  int64_t getNextSRDIndex();

  /// Update buffer size for a pending SRD (called when we see reinterpret_cast)
  void updateSRDBufferSize(mlir::Value memref, int64_t bufferSize);

  /// Get the number of kernel arguments (bindings + scalar args).
  size_t getNumKernelArgs() const {
    return pendingSRDs.size() + pendingScalarArgs.size();
  }

  //===--------------------------------------------------------------------===//
  // Split Vector Result Tracking
  //===--------------------------------------------------------------------===//

  /// Register split results from a vector load (for later use in stores)
  void registerSplitResults(mlir::Value originalValue,
                            llvm::ArrayRef<mlir::Value> splitResults) {
    for (auto result : splitResults) {
      splitResultsMap[originalValue].push_back(result);
    }
  }

  /// Get split results for a value (returns empty if not a split vector)
  llvm::ArrayRef<mlir::Value> getSplitResults(mlir::Value value) const {
    auto it = splitResultsMap.find(value);
    if (it != splitResultsMap.end())
      return it->second;
    return {};
  }

  /// Check if a value has split results
  bool hasSplitResults(mlir::Value value) const {
    return splitResultsMap.contains(value);
  }

  //===--------------------------------------------------------------------===//
  // Thread ID Upper Bound Tracking (for affine simplification)
  //===--------------------------------------------------------------------===//

  /// Register thread ID upper bound for a value
  void setThreadIdUpperBound(mlir::Value value, int64_t upperBound) {
    threadIdUpperBounds[value] = upperBound;
  }

  /// Get thread ID upper bound for a value (returns 0 if not tracked)
  int64_t getThreadIdUpperBound(mlir::Value value) const {
    auto it = threadIdUpperBounds.find(value);
    if (it != threadIdUpperBounds.end())
      return it->second;
    return 0;
  }

  /// Check if a value is a thread ID with known upper bound
  bool hasThreadIdUpperBound(mlir::Value value) const {
    return threadIdUpperBounds.contains(value);
  }

  //===--------------------------------------------------------------------===//
  // Bit Range Tracking (for OR vs ADD optimization)
  //===--------------------------------------------------------------------===//

  /// Set bit range for a value
  void setBitRange(mlir::Value value, const BitRange &range) {
    bitRangeMap[value] = range;
  }

  /// Get bit range for a value (returns full 32-bit range if not tracked)
  BitRange getBitRange(mlir::Value value) const {
    auto it = bitRangeMap.find(value);
    if (it != bitRangeMap.end())
      return it->second;
    return BitRange(); // Default: full 32-bit range (conservative)
  }

  /// Check if a value has a tracked bit range
  bool hasBitRange(mlir::Value value) const {
    return bitRangeMap.contains(value);
  }

  //===--------------------------------------------------------------------===//
  // Constant Offset Tracking (for buffer store offset:N optimization)
  //===--------------------------------------------------------------------===//

  /// Set constant offset for a value (for use in buffer store offset:N)
  void setConstOffset(mlir::Value value, int64_t offset) {
    constOffsetMap[value] = offset;
  }

  /// Get constant offset for a value (returns 0 if not tracked)
  int64_t getConstOffset(mlir::Value value) const {
    auto it = constOffsetMap.find(value);
    if (it != constOffsetMap.end())
      return it->second;
    return 0;
  }

  /// Check if a value has a tracked constant offset
  bool hasConstOffset(mlir::Value value) const {
    return constOffsetMap.contains(value);
  }

  //===--------------------------------------------------------------------===//
  // LDS Base Offset Tracking (for memref.view byte offsets)
  //===--------------------------------------------------------------------===//

  /// Set LDS base offset for a memref (byte offset from memref.view)
  void setLDSBaseOffset(mlir::Value memref, mlir::Value offset) {
    ldsBaseOffsetMap[memref] = offset;
  }

  /// Get LDS base offset for a memref (returns nullopt if not tracked)
  std::optional<mlir::Value> getLDSBaseOffset(mlir::Value memref) const {
    auto it = ldsBaseOffsetMap.find(memref);
    if (it != ldsBaseOffsetMap.end())
      return it->second;
    return std::nullopt;
  }

  /// Check if a memref has a tracked LDS base offset
  bool hasLDSBaseOffset(mlir::Value memref) const {
    return ldsBaseOffsetMap.contains(memref);
  }

  //===--------------------------------------------------------------------===//
  // Dynamic Stride Tracking (for memref.reinterpret_cast with runtime strides)
  //===--------------------------------------------------------------------===//

  /// Store a dynamic (runtime) stride value for a memref dimension.
  /// \p strideValue is the mapped WaveASM SSA value holding the element stride.
  void setDynamicStride(mlir::Value memref, unsigned dim,
                        mlir::Value strideValue) {
    dynamicStrideMap[memref][dim] = strideValue;
  }

  /// Get the dynamic stride value for a memref dimension.
  /// Returns nullopt if the stride is static.
  std::optional<mlir::Value> getDynamicStride(mlir::Value memref,
                                              unsigned dim) const {
    auto it = dynamicStrideMap.find(memref);
    if (it == dynamicStrideMap.end())
      return std::nullopt;
    auto dimIt = it->second.find(dim);
    if (dimIt == it->second.end())
      return std::nullopt;
    return dimIt->second;
  }

  /// Track a pending per-workgroup SRD base adjustment for a linearized memref
  struct PendingSRDBaseAdjust {
    mlir::Value elementOffset;
    int64_t srcSrdBase;
    int64_t elementBytes;
    // When non-null, use this value as NUM_RECORDS instead of
    // computeBufferSizeFromMemRef.  Set by handleFatRawBufferCast when
    // the MLIR specifies a tighter validBytes (e.g. for epilogue
    // elimination OOB protection on direct buffer loads).
    mlir::Value numRecordsOverride;
    // SSA value of the source (kernel-arg) SRD from the prologue.
    // Keeps the SRD live in the register allocator so its physical
    // registers are not reused between the prologue and the adjustment
    // point.
    mlir::Value srcSrdValue;
  };

  void setPendingSRDBaseAdjust(mlir::Value memref, mlir::Value elemOffset,
                               int64_t srcSrdBase, int64_t elementBytes,
                               mlir::Value numRecordsOverride = {},
                               mlir::Value srcSrdValue = {}) {
    pendingSRDBaseAdjustMap[memref] = {elemOffset, srcSrdBase, elementBytes,
                                       numRecordsOverride, srcSrdValue};
  }

  /// Get a pending SRD base adjustment (returns nullptr if none)
  const PendingSRDBaseAdjust *
  getPendingSRDBaseAdjust(mlir::Value memref) const {
    auto it = pendingSRDBaseAdjustMap.find(memref);
    if (it != pendingSRDBaseAdjustMap.end())
      return &it->second;
    return nullptr;
  }

  /// Clear a pending SRD base adjustment after it has been applied
  void clearPendingSRDBaseAdjust(mlir::Value memref) {
    pendingSRDBaseAdjustMap.erase(memref);
  }

  /// Default max buffer size: sentinel-safe limit ensuring the OOB sentinel
  /// at byte offset 0x7FFFFFFF falls outside the SRD range.
  static constexpr int64_t kDefaultMaxBufferSize = 0x7FFFFFFE;

  /// Look up buffer size for an SRD by its SGPR base index
  int64_t getBufferSizeForSRD(int64_t srdBase) const {
    for (const auto &pending : pendingSRDs) {
      if (pending.srdBaseIndex == srdBase)
        return pending.bufferSize;
    }
    return kDefaultMaxBufferSize;
  }

  //===--------------------------------------------------------------------===//
  // LDS Size Tracking (for kernel descriptor)
  //===--------------------------------------------------------------------===//

  /// Add to the total LDS size requirement
  void addLDSSize(int64_t size) { totalLDSSize += size; }

  /// Get the total LDS size requirement
  int64_t getTotalLDSSize() const { return totalLDSSize; }

  /// Get the current LDS allocation byte offset (for typed allocs)
  int64_t getLDSAllocOffset() const { return ldsAllocOffset; }

  /// Advance the LDS allocation offset by the given number of bytes
  void advanceLDSAllocOffset(int64_t size) { ldsAllocOffset += size; }

  //===--------------------------------------------------------------------===//
  // System Register Usage Tracking (for kernel descriptor metadata)
  //===--------------------------------------------------------------------===//

  /// Mark that the kernel uses workgroup ID in dimension x/y/z
  void setUsesWorkgroupIdX(bool uses = true) { usesWorkgroupIdX = uses; }
  void setUsesWorkgroupIdY(bool uses = true) { usesWorkgroupIdY = uses; }
  void setUsesWorkgroupIdZ(bool uses = true) { usesWorkgroupIdZ = uses; }

  /// Mark that the kernel uses workitem/thread ID (x dimension)
  void setUsesWorkitemId(bool uses = true) { usesWorkitemId = uses; }

  /// Query system register usage
  bool getUsesWorkgroupIdX() const { return usesWorkgroupIdX; }
  bool getUsesWorkgroupIdY() const { return usesWorkgroupIdY; }
  bool getUsesWorkgroupIdZ() const { return usesWorkgroupIdZ; }
  bool getUsesWorkitemId() const { return usesWorkitemId; }

  /// Get the number of user SGPRs (for computing system SGPR offsets)
  /// User SGPRs include: kernarg ptr (2) + preloaded args (on gfx950+)
  int64_t getUserSgprCount() const {
    // Base: 2 SGPRs for kernarg_segment_ptr
    int64_t count = 2;
    // On gfx950+ with kernarg preloading, add preloaded args
    if (llvm::isa<GFX950TargetAttr>(target)) {
      // Each kernel arg uses 2 SGPRs, capped at 14 (hardware max 16 total).
      count += std::min(size_t(14), getNumKernelArgs() * 2);
    }
    return count;
  }

  /// Get SGPR index for workgroup ID in the given dimension (0=x, 1=y, 2=z).
  /// System SGPRs are only allocated for enabled dimensions, so we count
  /// only the enabled IDs before the requested dimension.
  int64_t getWorkgroupIdSgprIndex(int dimension) const {
    int64_t baseIndex = getUserSgprCount();
    // When all three IDs are enabled (e.g., via enableAllWorkgroupIds()),
    // the layout is simply base + dimension.
    if (usesWorkgroupIdX && usesWorkgroupIdY && usesWorkgroupIdZ)
      return baseIndex + dimension;
    int64_t offset = 0;
    if (dimension > 0 && usesWorkgroupIdX)
      offset++;
    if (dimension > 1 && usesWorkgroupIdY)
      offset++;
    return baseIndex + offset;
  }

  /// Enable all three workgroup IDs so that the SGPR layout is predictable.
  /// Call this before translating any ops to avoid ordering dependencies.
  void enableAllWorkgroupIds() {
    usesWorkgroupIdX = true;
    usesWorkgroupIdY = true;
    usesWorkgroupIdZ = true;
  }

  /// Check if this is a multi-wave kernel (more than 64 threads per workgroup)
  bool isMultiWaveKernel() const {
    auto [wgX, wgY, wgZ] = getWorkgroupSize();
    return (wgX * wgY * wgZ) > 64;
  }

  /// Get workgroup size as (x, y, z) tuple
  std::tuple<int64_t, int64_t, int64_t> getWorkgroupSize() const {
    // Get workgroup_size attribute (using const-safe accessor)
    mlir::ArrayAttr workgroupSizeAttr =
        program->getAttrOfType<mlir::ArrayAttr>("workgroup_size");
    int64_t wgX = 64, wgY = 1, wgZ = 1; // defaults
    if (workgroupSizeAttr && workgroupSizeAttr.size() >= 1) {
      if (auto intAttr =
              llvm::dyn_cast<mlir::IntegerAttr>(workgroupSizeAttr[0])) {
        wgX = intAttr.getInt();
      }
      if (workgroupSizeAttr.size() >= 2) {
        if (auto intAttr =
                llvm::dyn_cast<mlir::IntegerAttr>(workgroupSizeAttr[1])) {
          wgY = intAttr.getInt();
        }
      }
      if (workgroupSizeAttr.size() >= 3) {
        if (auto intAttr =
                llvm::dyn_cast<mlir::IntegerAttr>(workgroupSizeAttr[2])) {
          wgZ = intAttr.getInt();
        }
      }
    }
    return {wgX, wgY, wgZ};
  }

  const OpHandlerRegistry &getRegistry() const { return registry; }

  const TranslationOptions &getOptions() const { return options; }

private:
  mlir::OpBuilder &builder;
  OpHandlerRegistry registry;
  ProgramOp program;
  TargetAttrInterface target;
  TranslationOptions options;
  ValueMapper mapper;
  llvm::DenseMap<mlir::Value, int64_t> bindingMap;
  llvm::DenseMap<mlir::Value, int64_t> cacheSwizzleMap;
  llvm::DenseMap<mlir::Value, SRDInfo> srdMap;
  llvm::DenseMap<mlir::Value, int64_t>
      srdIndexMap; // memref -> SRD SGPR base index
  llvm::DenseMap<mlir::Value, mlir::Value>
      srdValueMap; // memref -> SRD SSA value (PackOp-based)
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 4>>
      splitResultsMap; // originalValue -> split results
  llvm::DenseMap<mlir::Value, int64_t>
      threadIdUpperBounds; // thread ID -> upper bound
  llvm::DenseMap<mlir::Value, BitRange>
      bitRangeMap; // value -> bit range (for OR vs ADD opt)
  llvm::DenseMap<mlir::Value, int64_t>
      constOffsetMap; // value -> constant offset (for buffer store offset:N)
  llvm::DenseMap<mlir::Value, mlir::Value>
      ldsBaseOffsetMap; // memref -> LDS byte offset from memref.view

  llvm::DenseMap<mlir::Value, PendingSRDBaseAdjust> pendingSRDBaseAdjustMap;
  llvm::SmallVector<PendingSRD, 4> pendingSRDs;
  llvm::SmallVector<PendingScalarArg, 2> pendingScalarArgs;
  llvm::DenseMap<mlir::Value, llvm::DenseMap<unsigned, mlir::Value>>
      dynamicStrideMap;
  llvm::StringMap<mlir::Value> exprCache;
  int64_t nextSRDIndex =
      -1; // Will be computed lazily, starts after user+system SGPRs
  int64_t totalLDSSize = 0; // Total LDS allocation size in bytes
  // Running byte offset for typed LDS allocations (pattern b in
  // handleMemRefAlloc). Reset implicitly: a fresh TranslationContext is
  // created per gpu.func in translateModule / createProgramFromFunc.
  int64_t ldsAllocOffset = 0;
  bool srdPrologueEmitted = false;
  // System register usage tracking
  bool usesWorkgroupIdX = false;
  bool usesWorkgroupIdY = false;
  bool usesWorkgroupIdZ = false;
  bool usesWorkitemId = false;
};

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Check if a memref type represents LDS memory
bool isLDSMemRef(mlir::MemRefType memrefType);

/// Compute buffer size from memref type
int64_t computeBufferSizeFromMemRef(mlir::MemRefType memrefType);

/// Get constant value from an MLIR value if it's a constant
std::optional<int64_t> getArithConstantValue(mlir::Value value);

/// Check if a value is a power of 2
bool isPowerOf2(int64_t value);

/// Compute log2 of a value (assumes value is a power of 2)
int64_t log2(int64_t value);

//===----------------------------------------------------------------------===//
// Shared Buffer-Access Helpers
//===----------------------------------------------------------------------===//

struct VOffsetResult {
  mlir::Value voffset;
  int64_t instOffset;
};

VOffsetResult computeVOffsetFromIndices(mlir::MemRefType memrefType,
                                        mlir::ValueRange indices,
                                        TranslationContext &ctx,
                                        mlir::Location loc,
                                        mlir::Value base = nullptr);

/// Emit inline SRD base adjustment for per-workgroup buffer addressing.
/// Allocates a new SRD (5 SGPRs: base pair, hi/lo temporaries, offset temp),
/// copies the source SRD base, multiplies the element offset by elementBytes,
/// adds the byte offset into the base, and fills num_records + stride.
/// Returns a precolored SGPR quad representing the adjusted SRD.
mlir::Value
emitSRDBaseAdjustment(const TranslationContext::PendingSRDBaseAdjust &adj,
                      mlir::Value memref, TranslationContext &ctx,
                      mlir::Location loc);

mlir::Value lookupSRD(mlir::Value memref, TranslationContext &ctx,
                      mlir::Location loc);

llvm::SmallVector<mlir::Value, 4>
emitBufferLoads(mlir::Value srd, mlir::Value voffset, int64_t instOffset,
                int64_t numBytes, TranslationContext &ctx, mlir::Location loc);

//===----------------------------------------------------------------------===//
// Translation Functions
//===----------------------------------------------------------------------===//

/// Translate a single MLIR operation to waveasm
mlir::LogicalResult translateOperation(mlir::Operation *op,
                                       TranslationContext &ctx);

/// Translate an MLIR module containing GPU kernels to waveasm programs
mlir::LogicalResult translateModule(mlir::ModuleOp module,
                                    llvm::StringRef targetId);

/// Translate an MLIR module with explicit options
mlir::LogicalResult translateModule(mlir::ModuleOp module,
                                    const TranslationOptions &options);

/// Create an waveasm.program from a GPU function
ProgramOp createProgramFromGPUFunc(mlir::gpu::GPUFuncOp gpuFunc,
                                   mlir::OpBuilder &builder,
                                   llvm::StringRef targetId);

/// Create an waveasm.program from a func.func
ProgramOp createProgramFromFunc(mlir::func::FuncOp funcOp,
                                mlir::OpBuilder &builder,
                                llvm::StringRef targetId);

} // namespace waveasm

#endif // WaveASM_TRANSFORMS_TRANSLATEFROMMLIR_H
