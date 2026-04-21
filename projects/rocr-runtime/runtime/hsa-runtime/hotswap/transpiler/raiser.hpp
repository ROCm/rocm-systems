#ifndef HOTSWAP_TRANSPILER_RAISER_HPP
#define HOTSWAP_TRANSPILER_RAISER_HPP

#include "code_object_utils.hpp"
#include "raise_failure.hpp"

#include <memory>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace transpiler {

struct RaiseResult {
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<llvm::Module> module;
  int liftedCount = 0;
  int totalCount = 0;
  std::string irText;
  std::string disasmText;
  // Structured failure description. `failure.reason == None` iff `success`.
  RaiseFailure failure;
  bool success = false;
  bool hasDivergentExec = false;
};

// `enableWritelaneRewrite` toggles the post-raise rewrite pass that
// replaces cross-widen-divergent `v_writelane_b32` / `v_readlane_b32`
// sites with a per-source-wave `select` / `ds_bpermute` pair (see
// `rewrite_cross_lane_divergent.{hpp,cpp}` and wave-size-translation.md
// §5.6.3). Default off — opt-in per call during the graduation
// rollout; the classifier's `WaveIdLiftScalarized` refusal keeps the
// silent-miscompile gate closed on callers that do not set the flag.
//
// `enableWaveNative` toggles `WaveNativeProjection` for wave32 source
// → wave64 target cross-widening. Default off; under the default,
// `ModuloReplicationProjection` is used (the long-standing shape,
// see wave-size-translation.md §§2.2 and 5.1). Under wave-native the
// kernel body runs with hardware EXEC = -1 after an `@llvm.amdgcn.
// init_whole_wave` prologue captures the original per-lane active
// mask into the EXEC alloca. This preserves lanes 32..63 through
// every WMMA → MFMA collective (the pipeline in `wmma_lowering.cpp`
// requires full EXEC on both halves of the Wave64 for correct
// redistribute / MFMA / collect), at the cost of a semantic change
// to source-width EXEC writes (`s_mov_b32 exec_lo, v` now replicates
// into both halves of the widened EXEC; see
// `WaveProjection::broadcastNarrowExecLoWrite`). Gated per-caller
// during the rollout until the corpus sweep confirms no regression
// on kernels that rely on the old half-write semantics.
RaiseResult raiseToIR(const std::vector<uint8_t> &textBytes,
                      const std::string &sourceISA,
                      const std::string &kernelName,
                      const KernelMeta &meta,
                      uint64_t kernelOffset = 0,
                      const std::string &compilationTargetISA = "",
                      bool enableWritelaneRewrite = false,
                      bool enableWaveNative = false);

} // namespace transpiler

#endif
