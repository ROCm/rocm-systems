#ifndef HOTSWAP_TRANSPILER_KERNARG_LAYOUT_HPP
#define HOTSWAP_TRANSPILER_KERNARG_LAYOUT_HPP

#include "code_object_utils.hpp"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include <string>
#include <utility>
#include <vector>

namespace transpiler {

struct KernargParam {
  // Byte offset within the kernarg segment.
  int byteOffset;
  // Width of the slot in bytes. Always one of {4, 8}: pointer / i64 slots
  // are 8 bytes, every other slot (including the per-dword decomposition
  // of `by_value` aggregates with size > 8) is 4 bytes. The
  // raiser is responsible for splitting any larger `by_value` aggregate
  // into 4-byte i32 slots so the IR-level kernarg buffer layout matches
  // the source binary's byte layout. See raiser.cpp for the splitter.
  int byteSize;
  // Index into the IR-level kernel function's argument list.
  int paramIdx;
  // True iff the source metadata declared this slot as `global_buffer`
  // (so the IR arg type is `ptr addrspace(1)`); false for i32 / i64 /
  // dword-of-aggregate slots.
  bool isPointer;
};

struct KernargLayout {
  std::vector<KernargParam> params;
  int implicitArgsBase = 0;

  // Legacy "fully-contained slot list" resolver; kept around for any
  // future caller that wants the slot-coarse-grained view, but
  // `handle_smem.cpp` no longer relies on it (it uses the per-dword
  // `extractKernargDword` helper below for an exact byte-by-byte tile).
  void resolveLoad(
      int byteOffset, int loadBytes,
      std::vector<std::pair<int, int>> &out) const {
    int loadEnd = byteOffset + loadBytes;
    for (auto &p : params) {
      int pEnd = p.byteOffset + p.byteSize;
      if (p.byteOffset >= byteOffset && pEnd <= loadEnd)
        out.push_back({p.byteSize / 4, p.paramIdx});
    }
  }
};

// Materialise the i32 dword that lives at `byteOffset` in the kernarg
// segment, by extracting it from the matching IR-level Function arg.
//
// Handles the three slot shapes that `KernargLayout::params` records:
//   * i32 slot at the requested offset → return the arg directly.
//   * i64 / pointer slot covering [p.byteOffset, p.byteOffset+8) →
//     return either the low or high dword via lshr+trunc.
//
// Returns the materialised i32 Value on success.  On failure returns
// `nullptr` and (if `whyNot != nullptr`) writes a precise diagnostic
// describing why the load could not be served — the caller turns that
// into a `RaiseFailure::smemKernargMiss` (handle_smem.cpp).  Diagnostics
// always name the byte offset, the conflicting param's offset/size, and
// the sub-offset that defeated the lift, so the message is actionable
// without re-running with a debugger attached.
//
// Does not emit any IR for the failure path; the caller decides.
//
// Test back-reference: lit_tests/s_load_b96_kernarg/ exercises this
// helper end-to-end via an `s_load_b96` over a 16-byte `by_value`
// aggregate.  Any change to the slot-walk logic above (i32 vs.
// i64-or-ptr branch, sub-offset handling, error paths) must keep the
// IR-arg signature shape that fixture pins; conversely, that fixture
// is the only regression gate for the per-dword resolution contract
// across raiser / handler boundaries.
llvm::Value *extractKernargDword(const KernargLayout &layout,
                                 llvm::IRBuilder<> &B,
                                 llvm::Function *F,
                                 int byteOffset,
                                 std::string *whyNot = nullptr);

} // namespace transpiler

#endif
