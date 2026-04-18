#ifndef HOTSWAP_TRANSPILER_DECODE_HPP
#define HOTSWAP_TRANSPILER_DECODE_HPP

#include "decoded_inst.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <set>

namespace transpiler {

struct MCState;
class OpcodeMap;

// Output of the decode phase: a linearised stream of decoded instructions
// plus the set of basic-block start offsets the CFG recovery discovered.
//
// `blockStarts` uses `std::set` because Phase 3 iterates it in ascending
// order to assign deterministic BB labels and because the decode loop
// relies on `upper_bound` to decide whether an `s_endpgm` terminates
// the scan or is an early-return that still has later reachable code.
//
// `insts` uses `SmallVector<T, 0>` — the LLVM-native escape hatch for
// "I want the SmallVector API but no inline storage because
// `sizeof(T)` is too large for the default inline budget."
// `DecodedInst` contains three `std::string`s plus an inline MCInst
// operand array, which exceeds LLVM's 256-byte default cap; the zero
// inline capacity explicitly opts out of inline storage while still
// getting SmallVector's API and move-only guarantees.
struct DecodeResult {
  llvm::SmallVector<DecodedInst, 0> insts;
  std::set<uint64_t> blockStarts;
};

// Decode `textBytes` starting at `kernelOffset` using the caller-owned
// MC + OpcodeMap. Produces a fully populated `DecodedInst` per MCInst
// (semOp, tsFlags, srcMap/modMap, implicit-def classification, branch
// targets) and the set of block-start offsets.
//
// Fails loudly via `report_fatal_error` on any MC/TableGen invariant
// violation (unknown tied-to-def OpName, srcMap vs OpName::srcN drift,
// kMaxSrcs overflow). This is the LLVM-version-drift guard surface —
// every check here catches an upstream LLVM change before it can silently
// corrupt a handler's view of an instruction.
DecodeResult decodeKernel(const MCState &mc,
                          const OpcodeMap &opcMap,
                          llvm::ArrayRef<uint8_t> textBytes,
                          uint64_t kernelOffset);

} // namespace transpiler

#endif
