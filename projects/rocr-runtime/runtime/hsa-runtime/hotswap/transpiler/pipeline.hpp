#ifndef HOTSWAP_TRANSPILER_PIPELINE_HPP
#define HOTSWAP_TRANSPILER_PIPELINE_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace transpiler {

struct PipelineResult {
  std::vector<uint8_t> hsaco;
  std::string irText;
  std::string asmText;
  std::string failMnemonic;
  int liftedCount = 0;
  int totalCount = 0;
  bool success = false;
};

/// End-to-end pipeline: HSACO binary → raise to LLVM IR → llc → HSACO.
/// Single-ISA: raises and lowers using the same ISA.
///
/// `enableWritelaneRewrite` opt-in plumbs through to
/// `raiseToIR(..., enableWritelaneRewrite)` — see raiser.hpp and
/// `rewrite_cross_lane_divergent.{hpp,cpp}` for the rewrite contract
/// and wave-size-translation.md §5.6.3 for the principled derivation.
/// Default off.
PipelineResult runPipeline(const std::vector<uint8_t> &codeObjectData,
                           const std::string &targetISA,
                           const std::string &kernelName,
                           bool enableWritelaneRewrite = false);

/// Cross-architecture pipeline: raises using sourceISA, lowers to targetISA.
PipelineResult runPipeline(const std::vector<uint8_t> &codeObjectData,
                           const std::string &sourceISA,
                           const std::string &targetISA,
                           const std::string &kernelName,
                           bool enableWritelaneRewrite = false);

/// Raise and lower ALL kernels in a code object, producing a single merged
/// HSACO containing every kernel.  Returns success only if every kernel was
/// raised and compiled.  On failure, PipelineResult::failMnemonic identifies
/// the unsupported instruction (if any).
PipelineResult runPipelineAllKernels(const std::vector<uint8_t> &codeObjectData,
                                     const std::string &sourceISA,
                                     const std::string &targetISA,
                                     bool enableWritelaneRewrite = false);

} // namespace transpiler

#endif
