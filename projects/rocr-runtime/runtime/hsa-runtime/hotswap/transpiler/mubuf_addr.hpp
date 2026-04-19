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
// but today's raiser doesn't surface cpol/th/scope (see the
// "auxFlags is hard-coded to 0" item in
// hotswap/docs/buffer-store-lowering.md §"Known limitations").
//
// For stores, `stData` is the VGPR source carrying the data. For
// loads it's a default-constructed ParsedReg (kind == OTHER).
//
// Historical note: an earlier revision of this struct also exposed
// the raw SRSRC dwords (`dw0` / `dw1` / `dw2`) for the flat-store
// OOB-sink pattern in handle_mubuf.cpp.  That pattern was removed
// when the BUFFER_STORE handler was rewritten to use
// `llvm.amdgcn.raw.buffer.store` directly (R1 fix; see
// hotswap/docs/buffer-store-lowering.md), so the dword fields became
// dead — every consumer now goes through `srd`.  The fields were
// removed in the same cleanup pass that added the lit regression
// guard (lit_tests/buffer_store_no_scratch_alloca/).
struct MubufAddr {
  llvm::Value *srd = nullptr;
  llvm::Value *voffset = nullptr;
  llvm::Value *soffset = nullptr;
  llvm::Value *auxFlags = nullptr;
  ParsedReg stData;
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
