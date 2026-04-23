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
/// Raises using `sourceISA` and lowers to `targetISA`.  Single-ISA
/// callers pass the same string for both — see the single-ISA note
/// below for why there is no separate 3-string overload.
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
/// `enableWaveNative` selects `WaveNativeProjection` for wave32 →
/// wave64 cross-widening (see `wave_projection.{hpp,cpp}` and
/// wave-size-translation.md §2.2 projection ladder). Default **on**
/// as of the WaveNative graduation; pass `false` explicitly to
/// opt into `ModuloReplicationProjection`. See `raiser.hpp`'s
/// `enableWaveNative` parameter docstring for the full rationale.
///
/// Single-ISA convention: pass the same ISA string for both
/// `sourceISA` and `targetISA` (e.g. `runPipeline(data, "gfx942",
/// "gfx942", "kernel", ...)`).  Earlier revisions exposed a separate
/// 3-string overload `(data, targetISA, kernel, ...)` to elide the
/// repetition, but that overload silently misbehaved under C++
/// overload resolution: a cross-arch call `runPipeline(data,
/// "gfx1250", "gfx942", "kernel_name")` with four `const char *`
/// literals picked up the 3-string overload (because `const char *
/// → bool` is a *standard* pointer-to-bool conversion that outranks
/// the user-defined `const char * → std::string` conversion needed
/// for the 4-string cross-arch overload).  The 4th literal was then
/// bound to `enableWritelaneRewrite` as `true` and `kernelName`
/// silently became `"gfx942"`, which downstream surfaced as
/// `UserSgprLayout::fromKernelMeta: kernel 'gfx942' has no parsed
/// kernel descriptor` — an easy-to-miss silent miscompile of the
/// API contract itself.  Removing the 3-string overload and
/// requiring the canonical 4-string form is the only way to make
/// the ambiguity unrepresentable (SFINAE / `string_view`-typed
/// alternatives were evaluated and rejected: every candidate
/// preserved some path where `const char *` beats `std::string`
/// under standard conversions).  Callers repeat the ISA string
/// instead; the ~5 single-ISA call sites that did this pre-fix are
/// all updated to the two-string form.
///
/// `enablePermLane16Xor3PartnerRewrite` plumbs through to
/// `raiseToIR(..., enablePermLane16Xor3PartnerRewrite)` — see
/// raiser.hpp for the full TRANSITIONAL rationale and
/// `rewrite_permlane16_xor3_partner.{hpp,cpp}` for the pattern.
/// Default **on**.  Callers auditing the pre-rewrite shape pass
/// `false` explicitly.
PipelineResult runPipeline(const std::vector<uint8_t> &codeObjectData,
                           const std::string &sourceISA,
                           const std::string &targetISA,
                           const std::string &kernelName,
                           bool enableWritelaneRewrite = true,
                           bool enableWaveNative = true,
                           bool enablePermLane16Xor3PartnerRewrite = true);

/// Raise and lower ALL kernels in a code object, producing a single merged
/// HSACO containing every kernel.  Returns success only if every kernel was
/// raised and compiled.  On failure, PipelineResult::failMnemonic identifies
/// the unsupported instruction (if any).
PipelineResult runPipelineAllKernels(const std::vector<uint8_t> &codeObjectData,
                                     const std::string &sourceISA,
                                     const std::string &targetISA,
                                     bool enableWritelaneRewrite = true,
                                     bool enableWaveNative = true,
                                     bool enablePermLane16Xor3PartnerRewrite = true);

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
