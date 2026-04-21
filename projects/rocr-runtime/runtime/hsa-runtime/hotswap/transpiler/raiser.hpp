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
RaiseResult raiseToIR(const std::vector<uint8_t> &textBytes,
                      const std::string &sourceISA,
                      const std::string &kernelName,
                      const KernelMeta &meta,
                      uint64_t kernelOffset = 0,
                      const std::string &compilationTargetISA = "",
                      bool enableWritelaneRewrite = false);

} // namespace transpiler

#endif
