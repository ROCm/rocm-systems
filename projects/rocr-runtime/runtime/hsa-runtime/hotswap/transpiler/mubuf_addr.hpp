#ifndef HOTSWAP_TRANSPILER_MUBUF_ADDR_HPP
#define HOTSWAP_TRANSPILER_MUBUF_ADDR_HPP

#include "decoded_inst.hpp"
#include "parsed_reg.hpp"
#include "raise_context.hpp"

#include "llvm/IR/Value.h"

namespace transpiler {

// Decoded addressing shape of a MUBUF / VBUFFER load or store.
//
// `srd` is a <4 x i32> raw buffer descriptor built from the SRSRC
// 128-bit SGPR tuple, routed through `amdgcn.readfirstlane` so each
// word lands in an SGPR (avoiding the waterfall loop the backend
// would otherwise insert). `voffset` is the per-lane i32 byte offset
// (vaddr + imm). `soffset` is the SGPR byte offset (defaults to 0).
// `auxFlags` is always a zero i32 — the raw buffer intrinsics take it
// but today's raiser doesn't surface cpol/th/scope.
//
// For stores, `stData` is the VGPR source carrying the data. For
// loads it's a default-constructed ParsedReg (kind == OTHER).
struct MubufAddr {
  llvm::Value *srd = nullptr;
  llvm::Value *voffset = nullptr;
  llvm::Value *soffset = nullptr;
  llvm::Value *auxFlags = nullptr;
  ParsedReg stData;
  // Raw SRSRC dwords, retained for callers that need to reconstruct
  // the base pointer / numRec out-of-band (the flat-store OOB-sink
  // pattern in handle_mubuf.cpp):
  //   dw0 = base_lo, dw1 = base_hi (low 16 bits), dw2 = numRec.
  // Not needed by users that only consume `srd` through the raw-
  // buffer intrinsics.
  llvm::Value *dw0 = nullptr;
  llvm::Value *dw1 = nullptr;
  llvm::Value *dw2 = nullptr;
  bool haveSoffset = false;
};

// Decode a MUBUF / VBUFFER load or store's addressing operands into a
// fully-materialised `MubufAddr`. Recognises the LLVM-MC operand
// layout:
//
//   load:  [dst(vdata), srsrc(SGPR4), vaddr(VGPR), soff(SGPR?), imm?, cpol?]
//   store: [srsrc(SGPR4), vdata(VGPR), vaddr(VGPR), soff(SGPR?), imm?, cpol?]
//
// Both MUBUF and VBUFFER encodings share this shape — the operand
// order inside an encoding can vary between LLVM versions so the
// classifier keys on `ParsedReg::Kind` rather than position.
//
// Fails loudly via `report_fatal_error` on unrecognised shapes (no
// SRSRC found, or cannot read SRSRC dwords). `diagLabel` names the
// caller for the error message (e.g. `"MUBUF"` or `"MUBUF_LDS"`).
MubufAddr decodeMubufAddr(RaiseContext &ctx, const DecodedInst &di,
                          OpResolver &op, bool isStore,
                          llvm::StringRef diagLabel);

// Atomic form of MUBUF: the SRSRC is used as a plain 64-bit base
// pointer (lo + masked hi16) rather than a raw-buffer descriptor.
// `ptr` is the resulting flat pointer in address space 0.
struct MubufAtomicAddr {
  llvm::Value *ptr = nullptr;
};

// Decode a `buffer_atomic_*` instruction's addressing operands.
// `op.srcReg(0)` MUST be the SRSRC SGPR tuple. Fails loudly on shape
// mismatch.
MubufAtomicAddr decodeMubufAtomicAddr(RaiseContext &ctx,
                                       const DecodedInst &di,
                                       OpResolver &op,
                                       llvm::StringRef diagLabel);

} // namespace transpiler

#endif
