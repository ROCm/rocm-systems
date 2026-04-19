#include "kernarg_layout.hpp"

#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"

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

  if (whyNot) {
    *whyNot = "kernarg byte offset " + std::to_string(byteOffset) +
             " is not covered by any explicit kernarg slot";
  }
  return nullptr;
}

} // namespace transpiler
