#ifndef HOTSWAP_TRANSPILER_TDM_RUNTIME_HPP
#define HOTSWAP_TRANSPILER_TDM_RUNTIME_HPP

// Bitcode-embedded HIP runtime helper for cross-target lifts of gfx1250
// Tensor Data Mover instructions onto gfx942.
//
// At CMake configure time, when `hipcc` is available, the project compiles
// `runtime/tdm.hip` to `tdm_runtime.gfx942.bc` and embeds the bytes via
// `cmake/EmbedBinary.cmake`. At raise time, `linkTDMRuntime` parses the
// embedded bitcode into an `llvm::Module` and link-merges it into the
// per-kernel module before `verifyModule` runs (raiser.cpp), so `llc`
// sees a self-contained module.
//
// Symbol naming is stable across builds — `handle_vimage.cpp` declares
// the helpers via `declareTDMLoad/Store` and emits `CallInst`s by name;
// `linkTDMRuntime` resolves the calls by linking the bodies in.
//
// When the transpiler was built without hipcc, `tdmRuntimeAvailable()`
// returns false and the cross-target VIMAGE handler falls back to the
// pre-existing loud refusal path. This preserves today's no-hipcc build
// behaviour exactly.

#include "llvm/ADT/StringRef.h"

namespace llvm {
class FunctionCallee;
class Module;
} // namespace llvm

namespace transpiler {

// Stable C-linkage names emitted by `runtime/tdm.hip`.
inline constexpr llvm::StringRef kTDMLoadSymbol  = "salmon_tdm_load_to_lds";
inline constexpr llvm::StringRef kTDMStoreSymbol = "salmon_tdm_store_from_lds";

// Declare (without body) the helper functions in `M`. Signature carries
// the four D# groups the emulation walk actually reads:
//   void(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>)
// The gfx1250 LLVM intrinsic's trailing `<8 x i32> grp4` (reserved) and
// `i32 cpol` arguments are deliberately NOT forwarded — the walker does
// not consume them, carrying them through would only bloat the helper's
// IR prologue, and `marshalTDMArgs` in `handle_vimage.cpp` still
// produces the full six-tuple for the same-target intrinsic emit, so
// the elision is localised to the cross-target `CreateCall` in the
// handler.
llvm::FunctionCallee declareTDMLoad(llvm::Module &M);
llvm::FunctionCallee declareTDMStore(llvm::Module &M);

// True iff the embedded bitcode blob is non-empty (i.e. hipcc was found
// at the transpiler's CMake configure time).
bool tdmRuntimeAvailable();

// True iff `M` already declares either of the helper symbols. Used by
// the raiser to decide whether a link-merge is required.
bool moduleUsesTDMRuntime(const llvm::Module &M);

// Parse the embedded `.bc` and link it into `M`. `targetISA` is
// informational today (we ship gfx942-only bitcode for v1); future
// per-target dispatch will key on it. Returns true on success; on
// failure logs to `llvm::errs()` and returns false. The caller MUST
// treat a false return as a hard raise failure — the unresolved
// `CallInst`s emitted by the handler would otherwise reach `llc` as
// undefined externals.
bool linkTDMRuntime(llvm::Module &M, llvm::StringRef targetISA);

} // namespace transpiler

#endif // HOTSWAP_TRANSPILER_TDM_RUNTIME_HPP
