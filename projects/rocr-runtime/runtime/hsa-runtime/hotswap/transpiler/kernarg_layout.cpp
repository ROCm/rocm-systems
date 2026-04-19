#include "kernarg_layout.hpp"

#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"

#include <algorithm>
#include <string>

using namespace llvm;

namespace transpiler {

llvm::Value *extractKernargDword(const KernargLayout &layout,
                                 llvm::IRBuilder<> &B,
                                 llvm::Function *F,
                                 int byteOffset,
                                 std::string *whyNot) {
  Type *i32Ty = B.getInt32Ty();
  Type *i64Ty = B.getInt64Ty();

  for (auto &p : layout.params) {
    int pEnd = p.byteOffset + p.byteSize;
    if (byteOffset < p.byteOffset || byteOffset + 4 > pEnd)
      continue;

    int relOff = byteOffset - p.byteOffset;
    Value *arg = F->getArg(p.paramIdx);
    Type *argTy = arg->getType();

    // Pointer / i64 slot: the low dword corresponds to relOff == 0,
    // the high dword to relOff == 4. Both source-ABI and target-ABI
    // place the low dword in the low SGPR (little-endian kernarg
    // buffer), so the lshr direction here is unambiguous.
    if (p.isPointer || argTy == i64Ty) {
      Value *asI64 =
          (argTy == i64Ty) ? arg : B.CreatePtrToInt(arg, i64Ty, "ka_p2i");
      if (relOff == 0)
        return B.CreateTrunc(asI64, i32Ty, "ka_lo");
      if (relOff == 4)
        return B.CreateTrunc(B.CreateLShr(asI64, 32, "ka_hi_shr"), i32Ty,
                             "ka_hi");
      if (whyNot) {
        *whyNot = "kernarg byte offset " + std::to_string(byteOffset) +
                 " has unsupported sub-offset " + std::to_string(relOff) +
                 " into 64-bit param idx " + std::to_string(p.paramIdx) +
                 " (param byteOffset=" + std::to_string(p.byteOffset) +
                 ", byteSize=" + std::to_string(p.byteSize) + ")";
      }
      return nullptr;
    }

    // i32 slot: `byteSize == 4` is enforced by KernargParam construction
    // in raiser.cpp (size==4 args, decomposed dwords of by_value > 8).
    // A non-zero relOff would indicate a layout bug.
    if (argTy == i32Ty) {
      if (relOff != 0) {
        if (whyNot) {
          *whyNot = "kernarg byte offset " + std::to_string(byteOffset) +
                   " has non-zero sub-offset " + std::to_string(relOff) +
                   " into i32 param idx " + std::to_string(p.paramIdx) +
                   " (param byteOffset=" + std::to_string(p.byteOffset) +
                   ", byteSize=" + std::to_string(p.byteSize) + "); raiser bug";
        }
        return nullptr;
      }
      return arg;
    }

    if (whyNot) {
      *whyNot = "kernarg byte offset " + std::to_string(byteOffset) +
               " hits param idx " + std::to_string(p.paramIdx) +
               " with unsupported IR arg type";
    }
    return nullptr;
  }

  // No named kernarg slot covers [byteOffset, byteOffset+4). Two cases:
  //
  // (a) The dword lies in *trailing alignment padding* within the
  //     kernarg segment — i.e. it sits past the last named arg but
  //     still within `kernargSegmentSize`. HSA leaves those bytes
  //     uninitialised by contract, so the LLVM-correct representation
  //     is `undef i32`. Tensile-style GEMM kernels routinely emit a
  //     16-byte `s_load_b128` over the last 12 bytes of named args
  //     plus 4 bytes of padding because the aligned vector load is
  //     cheaper than a split 12+4 sequence; the trailing dword lands
  //     in an SGPR the kernel never reads, and emitting `undef`
  //     preserves that "any value is allowed" semantics without
  //     fabricating a concrete zero. The runtime never observes the
  //     value either way.
  //
  // (b) The dword would overrun the segment, or it straddles two
  //     params with a hole / partial overlap that the layout cannot
  //     describe. Both are real errors and we refuse loudly with a
  //     diagnostic that names the offset, the segment size, and the
  //     last named-arg end.
  if (layout.kernargSegmentSize > 0 &&
      byteOffset + 4 <= layout.kernargSegmentSize) {
    int lastEnd = 0;
    for (auto &p : layout.params)
      lastEnd = std::max(lastEnd, p.byteOffset + p.byteSize);
    if (byteOffset >= lastEnd)
      return UndefValue::get(i32Ty);
  }

  if (whyNot) {
    int lastEnd = 0;
    for (auto &p : layout.params)
      lastEnd = std::max(lastEnd, p.byteOffset + p.byteSize);
    *whyNot = "kernarg byte offset " + std::to_string(byteOffset) +
             " is not covered by any explicit kernarg slot "
             "(last named arg ends at " + std::to_string(lastEnd) +
             ", kernarg_segment_size=" +
             std::to_string(layout.kernargSegmentSize) + ")";
  }
  return nullptr;
}

} // namespace transpiler
