// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file probe_callable.h
/// @brief Turn a resolved probe symbol into a self-contained, callable probe
///        body: its instruction words plus its verified ABI.

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/register_set.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rocjitsu {

class AmdGpuCodeObject;
struct ResolvedProbeSymbol;

/// @brief The return-link rule a verified probe body conforms to.
///
/// @details Names only how the body returns. The registers a convention
/// implies are reported by derive_probe_abi(), so this enum does not carry one
/// enumerator per combination of the properties an ABI has.
enum class ProbeCallingConvention {
  Unknown,                ///< Not yet verified / unrecognized.
  AmdGpuFuncReturnS30S31, ///< Returns via s_setpc_b64 s[30:31].
};

/// @brief A link-pair base no calling convention can select.
///
/// @details A 64-bit scalar operand must name an even-aligned register pair, so
/// an odd base is invalid by construction rather than by convention. Used as
/// ProbeAbi's default so the field reads as obviously unset to anyone inspecting
/// one, rather than plausible. Validity does not turn on it -- that is
/// is_valid_probe_abi()'s job.
inline constexpr uint16_t kUnsetLinkPairBase = 1;

/// @brief Where a verified convention places the values the framework, rather
///        than the probe body, is responsible for producing.
///
/// @details Derived from the convention by derive_probe_abi(). A POD, so a
/// caller can still build one by hand — the fail-closed tests rely on being
/// able to — which is why consumers ask is_valid_probe_abi() rather than
/// trusting the fields.
struct ProbeAbi {
  ProbeCallingConvention cc = ProbeCallingConvention::Unknown;
  uint16_t link_pair_base = kUnsetLinkPairBase; ///< Base of the return-link SGPR pair.

  constexpr bool operator==(const ProbeAbi &) const = default;
};

/// @brief The ABI @p cc implies, or std::nullopt if it is unrecognized.
[[nodiscard]] inline constexpr std::optional<ProbeAbi> derive_probe_abi(ProbeCallingConvention cc) {
  // No default case: a convention added to the enum should fail the build here
  // rather than silently derive to nullopt. The trailing return keeps a value
  // cast from outside the enumerators from running off the end.
  switch (cc) {
  case ProbeCallingConvention::AmdGpuFuncReturnS30S31:
    return ProbeAbi{.cc = cc, .link_pair_base = 30};
  case ProbeCallingConvention::Unknown:
    break;
  }
  return std::nullopt;
}

/// @brief Could @p abi have come out of derive_probe_abi()?
///
/// @details Answered by re-deriving and comparing, so the question the name asks
/// is the question that gets tested. A weaker screen -- "recognized convention,
/// even-aligned pair" -- would admit an ABI whose link pair its own convention
/// never chooses, which is the shape a hand-built one takes.
[[nodiscard]] inline constexpr bool is_valid_probe_abi(const ProbeAbi &abi) {
  return derive_probe_abi(abi.cc) == abi;
}

/// @brief The return-link register pair @p abi returns through.
///
/// @details Meaningful only for an @p abi that passes is_valid_probe_abi(),
/// like the field it wraps. Exists so consumers reason about "the link pair"
/// rather than re-deriving that it is two SGPRs starting at link_pair_base.
[[nodiscard]] inline constexpr RegisterRef probe_link_pair(const ProbeAbi &abi) {
  return RegisterRef{RegClass::SGPR, abi.link_pair_base, 2};
}

/// @brief The registers @p abi has the framework supply to the probe body.
///
/// @details A body that reads one of these is not depending on state that only
/// exists at kernel entry, so a live-in analysis subtracts this set before
/// concluding a probe cannot be called from an arbitrary site. Collecting the
/// per-convention register knowledge behind one query keeps that analysis from
/// having to enumerate the conventions itself.
///
/// An @p abi that fails is_valid_probe_abi() supplies nothing.
[[nodiscard]] RegisterSet supplied_registers(const ProbeAbi &abi);

/// @brief A probe body extracted from a code object and verified to be safe to
///        relocate verbatim into the instrumented code object.
struct ProbeCallable {
  std::string symbol;                                        ///< Resolved symbol name.
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;          ///< ISA the body decodes as.
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID; ///< Concrete ISA target, if known.
  std::vector<uint32_t> body_words; ///< The body's instruction words, in order.
  ProbeAbi abi;                     ///< Verified convention and the registers it implies.

  /// Byte offset of the body once it has been laid out in the instrumented
  /// code object's text.
  /// Currently zero here because that placement depends on the trampoline
  /// size and the final cave-placement API, neither of which exists yet.
  uint64_t output_text_offset = 0;
};

/// @brief Extract and verify the body of @p sym (already resolved out of
///        @p probe_obj) as a self-contained callable probe decoded for @p arch.
///
/// Verification is intentionally conservative (fail-closed). The body must:
///   - copy in-bounds out of the code object image,
///   - have no relocation applied anywhere inside it,
///   - decode cleanly as a sequence of 4- or 8-byte @p arch instructions that
///     exactly tiles the body (no partial trailing word),
///   - contain no call (s_swappc_b64 / s_call_b64) and no explicit scratch
///     access (FLAT scratch_* / SMEM s_scratch_*); note private access via FLAT
///     addressing is not statically detectable here and is not rejected, and
///   - end in `s_setpc_b64 s[30:31]`.
///
/// On success the returned ProbeCallable carries the ABI derived from
/// AmdGpuFuncReturnS30S31. Returns std::nullopt (with a reason written to
/// @p error_out, if non-null) on any failure.
[[nodiscard]] std::optional<ProbeCallable> build_probe_callable(const AmdGpuCodeObject &probe_obj,
                                                                const ResolvedProbeSymbol &sym,
                                                                rj_code_arch_t arch,
                                                                std::string *error_out = nullptr);

} // namespace rocjitsu
