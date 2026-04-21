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
/// Default **on** (post-Triton-corpus graduation; the rewrite's
/// §5.6.3 symmetry contract makes it correctness-preserving on every
/// cross-widening kernel, and corpus_layernorm_fp32 /
/// corpus_softmax_fp32 in compare_correctness are the corpus-wide
/// bit-exactness regression gates).  Callers that specifically want
/// the pre-rewrite path pass `false`.
///
/// `enableWaveNative` opt-in selects `WaveNativeProjection` for
/// wave32 → wave64 cross-widening (see `wave_projection.{hpp,cpp}`
/// and wave-size-translation.md §2.2 projection ladder). Default off
/// — `ModuloReplicationProjection` remains the canonical choice
/// until the corpus sweep under wave-native confirms no regressions.
PipelineResult runPipeline(const std::vector<uint8_t> &codeObjectData,
                           const std::string &targetISA,
                           const std::string &kernelName,
                           bool enableWritelaneRewrite = true,
                           bool enableWaveNative = false);

/// Cross-architecture pipeline: raises using sourceISA, lowers to targetISA.
PipelineResult runPipeline(const std::vector<uint8_t> &codeObjectData,
                           const std::string &sourceISA,
                           const std::string &targetISA,
                           const std::string &kernelName,
                           bool enableWritelaneRewrite = true,
                           bool enableWaveNative = false);

/// Raise and lower ALL kernels in a code object, producing a single merged
/// HSACO containing every kernel.  Returns success only if every kernel was
/// raised and compiled.  On failure, PipelineResult::failMnemonic identifies
/// the unsupported instruction (if any).
PipelineResult runPipelineAllKernels(const std::vector<uint8_t> &codeObjectData,
                                     const std::string &sourceISA,
                                     const std::string &targetISA,
                                     bool enableWritelaneRewrite = true,
                                     bool enableWaveNative = false);

/// Process-global "strict mode" toggle, controlled by the
/// `HSA_SALMON_STRICT` environment variable. When set to a non-empty
/// value the salmon transpiler promotes known silent-miscompile sites
/// to honest refusals instead of warning-and-continue. Today this
/// covers two sites flagged by the corpus runner's `INTEGRATION_GAP.md`
/// investigation:
///
///   * `s_setreg_imm32_b32 mode, imm` writes to the MODE register
///     (handle_sopk.cpp): silently dropped in non-strict mode but the
///     kernel may rely on the FP rounding / denormal / IEEE / FTZ bits
///     being set.
///   * `llvm.amdgcn.implicitarg.ptr` lifts (handle_smem.cpp): emit
///     `gep = implicitarg_ptr + sourceImplOffset; load i32, gep` which
///     bakes the source ISA's hidden-arg byte layout into a load against
///     the target ISA's hidden-arg block. Any mismatch in base or
///     layout produces wrong values for `hidden_block_count_*`,
///     `hidden_grid_dims`, etc.
///
/// Parsed once on first call (`std::getenv("HSA_SALMON_STRICT")`); the
/// callers (handler implementations) read the flag without round-tripping
/// through the OS allocator on every instruction. The runner sets
/// `HSA_SALMON_STRICT=1` in its salmon `ModeSpec`; `compare_correctness`
/// and the gtest binary do not, so existing GPU tests stay passing.
bool isStrictMode();

} // namespace transpiler

#endif
