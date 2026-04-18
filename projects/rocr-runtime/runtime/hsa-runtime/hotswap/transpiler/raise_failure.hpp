#ifndef HOTSWAP_TRANSPILER_RAISE_FAILURE_HPP
#define HOTSWAP_TRANSPILER_RAISE_FAILURE_HPP

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <string>

namespace transpiler {

struct DecodedInst;

// Structured reason for a raise failure. Lives in its own header so the
// handler layer (`raise_context.hpp`) can depend on failure values
// without pulling in `RaiseResult` and the rest of the top-level
// `raiser.hpp` interface.
enum class RaiseFailureReason : uint16_t {
  None = 0,
  // Main loop: no handler matched on TSFlags, or every matching handler
  // returned unhandled without setting a more specific failure. The
  // `mnemonic` / `format` / `offset` triple locates the instruction.
  UnsupportedOpcode,
  // A handler matched on SemOp but the specific operand shape /
  // encoding variant it saw is not yet modelled. Today's format-
  // specific failure sites (handle_valu, handle_flat, handle_mubuf,
  // handle_mfma, handle_vopd) all use this category. `detail` carries
  // shape-specific context when available.
  UnsupportedShape,
  // Phase 1.5 gate: an EXEC-writing instruction whose SemOp does not
  // have `routesExecThroughStoreExec` set in `sem_op_attrs.cpp`.
  SPEUnsafeExecWriter,
  // SMEM load's scalar base does not match any known kernarg slot.
  SMEMKernargMiss,
  // Phase 2: `TargetRegistry::createTargetMachine` returned null.
  TargetMachineCreationFailed,
  // Phase 7: `verifyModule` rejected the emitted IR.
  IRVerificationFailed,
  // Phase 1.4.5 wave-size-obstruction classifier (SPE_DESIGN.md §4).
  // One reason per refusal *decision* so `batch_raise_test` and
  // `corpus_test` can bucket failures without parsing the failure
  // text. See `wave_size_obstruction.hpp` for the classifier
  // taxonomy and the mapping between these reasons and the more
  // specific `ObstructionKind` values.
  //
  // The §3 Class 1..4 grouping is preserved as cross-references in
  // the comments below; it is not part of the enum-value identity.
  CrossWaveLaneIdLeak,             // §3 Class 1: MbcntHiLaneIdLeak / OutOfRangeLaneOperand.
  CrossWaveUnrewritableShuffle,    // §3 Class 2: FullWaveRotate (no §4 rewrite available).
  CrossWaveShuffleRewritePending,  // §3 Class 2: shuffle with CROSS_LANE_SURVEY P-item, handler not landed.
  CrossWaveReplicaRace,            // §3 Class 3: NonCommutativeAtomic.
  CrossWaveLanePredicatedExec,     // §3 Class 4: CmpxFromLaneId / SaveExecFromLaneId.
};

// Human-readable name for a `RaiseFailureReason`. Stable enough for
// test fixtures (`batch_raise_test`, `corpus_test`) to bucket on.
const char *reasonString(RaiseFailureReason r);

struct RaiseFailure {
  RaiseFailureReason reason = RaiseFailureReason::None;
  // Offending instruction mnemonic (e.g. `global_store_dwordx4`).
  std::string mnemonic;
  // Encoding-format category (e.g. `VALU`, `FLAT`, `MUBUF`) — stable
  // bucketing key for the batch / corpus test summaries. For non-
  // decode-level failures (e.g. `TargetMachineCreationFailed`) this
  // is the `reasonString` of `reason`.
  std::string format;
  // Byte offset inside the disassembled text section, in host order.
  // Zero for failures not tied to a specific instruction.
  uint64_t offset = 0;
  // Optional human-readable context; may include shape hints,
  // attempted rewrites, etc.
  std::string detail;

  bool hasFailed() const { return reason != RaiseFailureReason::None; }

  // Factory constructors. These are the canonical way to build a
  // `RaiseFailure`: aggregate initialisation was error-prone because
  // it allowed `reason = None` with non-empty strings, which
  // `hasFailed()` would then lie about.
  //
  // Handler layer.

  // Handler recognised the SemOp but refused the specific operand
  // shape. `di` supplies the mnemonic and source offset.
  static RaiseFailure unsupportedShape(const DecodedInst &di,
                                        llvm::StringRef format,
                                        const llvm::Twine &detail = {});

  // SMEM load's scalar base doesn't match any known kernarg slot.
  static RaiseFailure smemKernargMiss(const DecodedInst &di);

  // Raiser main loop / pre-translation gates. These are only built by
  // `raiser.cpp` — the factories live here so every reason is
  // constructed consistently, not via aggregate init that could leave
  // `hasFailed()` disagreeing with the field contents.

  // Main loop: no handler claimed the SemOp (either no TSFlags match
  // or every matching handler returned `handled=false` without
  // setting a more specific failure). `di` supplies the mnemonic /
  // offset; `format` is the human-readable encoding label.
  static RaiseFailure unsupportedOpcode(const DecodedInst &di,
                                         llvm::StringRef format);

  // Phase 1.5 gate: an EXEC-writing instruction whose SemOp does not
  // have `routesExecThroughStoreExec` declared in any handler's
  // `get*Attrs()` registration.
  static RaiseFailure speUnsafeExecWriter(const DecodedInst &di);

  // Phase 2: `TargetRegistry::createTargetMachine` returned null.
  static RaiseFailure targetMachineCreationFailed();

  // Phase 7: `verifyModule` rejected the emitted IR.
  // `err` carries the verifier's diagnostic text for the `detail` field.
  static RaiseFailure irVerificationFailed(const llvm::Twine &err);

  // Phase 1.4.5 wave-size-obstruction classifier (SPE_DESIGN.md §4).
  // `di` supplies the offending mnemonic + offset. `kindDetail`
  // should carry the human-readable `ObstructionKind` name (from
  // `obstructionKindName`), the CROSS_LANE_SURVEY.md P-item (where
  // applicable), and any operand-level context the classifier
  // extracted (e.g. "operand value N >= W_s=M"). The resulting
  // failure is renderable by `reasonString` for batch-test bucketing
  // without parsing `detail`.
  static RaiseFailure crossWaveLaneIdLeak(const DecodedInst &di,
                                           const llvm::Twine &kindDetail);
  static RaiseFailure crossWaveUnrewritableShuffle(const DecodedInst &di,
                                                    const llvm::Twine &kindDetail);
  static RaiseFailure crossWaveShuffleRewritePending(const DecodedInst &di,
                                                      const llvm::Twine &kindDetail);
  static RaiseFailure crossWaveReplicaRace(const DecodedInst &di,
                                            const llvm::Twine &kindDetail);
  static RaiseFailure crossWaveLanePredicatedExec(const DecodedInst &di,
                                                   const llvm::Twine &kindDetail);
};

} // namespace transpiler

#endif
