#ifndef HOTSWAP_TRANSPILER_FLAT_ADDR_HPP
#define HOTSWAP_TRANSPILER_FLAT_ADDR_HPP

#include "decoded_inst.hpp"
#include "parsed_reg.hpp"
#include "raise_context.hpp"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Value.h"

namespace transpiler {

// Decoded address-shape of a FLAT / GLOBAL load or store.
//
// `ptr` is the final address-space-1 pointer with any byte memOffset
// already folded in via GEP, ready to feed to `CreateLoad` / `CreateStore`.
// `memOffset` is retained for debugging / diagnostics only — callers
// should not re-GEP it onto `ptr`.
struct FlatAddr {
  llvm::Value *ptr = nullptr;
  int64_t memOffset = 0;
  bool hasSaddr = false;
  // For stores: the VGPR source register carrying the store data. Unused
  // (OTHER kind) for loads.
  ParsedReg stData;
};

// Decode a GLOBAL_LOAD addressing operand shape. Recognised forms:
//
//   plain form: vaddr(VGPR64), [imms...]
//   SADDR form: saddr(SGPR64), vaddr(VGPR32), [imms...]
//
// `elemBytes` is the access element size (used only when
// `di.hasScaleOffset` is set — then the per-lane VGPR vaddr is
// multiplied by `elemBytes` before being added to the SGPR base).
// `diagLabel` is used in the `report_fatal_error` message if neither
// form matches (e.g. `"GLOBAL_LOAD sub-dword"`).
//
// Fails loudly on unrecognised shapes.
FlatAddr decodeGlobalLoadAddr(RaiseContext &ctx, const DecodedInst &di,
                               OpResolver &op, int elemBytes,
                               llvm::StringRef diagLabel);

// Decode a GLOBAL_STORE addressing operand shape. Recognised forms:
//
//   plain form: vaddr(VGPR64), vdata(VGPR*), [imms...]
//   SADDR form: vaddr(VGPR32), vdata(VGPR*), saddr(SGPR64), [imms...]
//
// On success, `.stData` is populated with the vdata register. Other
// behaviour matches `decodeGlobalLoadAddr`.
FlatAddr decodeGlobalStoreAddr(RaiseContext &ctx, const DecodedInst &di,
                                OpResolver &op, int elemBytes,
                                llvm::StringRef diagLabel);

} // namespace transpiler

#endif
