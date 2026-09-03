// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file probe_callable.h
/// @brief Turn a resolved probe symbol into a self-contained, callable probe
///        body: its instruction words plus a verified calling convention.

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

/// @brief The ABI a verified probe body conforms to.
///
/// Only one shape is supported today: a no-argument function that returns
/// through the s[30:31] link pair. More enums will be added as more
/// arguments are passed
enum class ProbeCallingConvention {
  Unknown,                      ///< Not yet verified / unrecognized.
  AmdGpuFuncNoArgsReturnS30S31, ///< void(void), returns via s_setpc_b64 s[30:31].
};

/// @brief The return-link SGPR pair base for a verified calling convention.
[[nodiscard]] inline constexpr std::optional<uint16_t> link_pair_for(ProbeCallingConvention cc) {
  switch (cc) {
  case ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31:
    return 30;
  case ProbeCallingConvention::Unknown:
  default:
    return std::nullopt;
  }
}

/// @brief A link-pair base no calling convention can select.
///
/// @details A 64-bit scalar operand must name an even-aligned register pair, so
/// an odd base is invalid by construction rather than by convention. Used as
/// ProbeAbi's default so an ABI that never went through derive_probe_abi() is
/// detectable instead of merely wrong.
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
};

/// @brief Could @p abi have come out of derive_probe_abi()?
///
/// @details Rejects both an unrecognized convention and a link pair that no
/// convention could have chosen, so a default-constructed or hand-built ABI is
/// caught alongside an unverified one.
[[nodiscard]] inline constexpr bool is_valid_probe_abi(const ProbeAbi &abi) {
  return abi.cc != ProbeCallingConvention::Unknown && abi.link_pair_base % 2 == 0;
}

/// @brief The ABI @p cc implies, or std::nullopt if it is unrecognized.
[[nodiscard]] inline constexpr std::optional<ProbeAbi> derive_probe_abi(ProbeCallingConvention cc) {
  const std::optional<uint16_t> link_pair = link_pair_for(cc);
  if (!link_pair)
    return std::nullopt;
  return ProbeAbi{.cc = cc, .link_pair_base = *link_pair};
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
  ProbeCallingConvention cc = ProbeCallingConvention::Unknown; ///< Verified ABI.

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
/// On success the returned ProbeCallable has cc ==
/// AmdGpuFuncNoArgsReturnS30S31. Returns std::nullopt (with a reason written to
/// @p error_out, if non-null) on any failure.
[[nodiscard]] std::optional<ProbeCallable> build_probe_callable(const AmdGpuCodeObject &probe_obj,
                                                                const ResolvedProbeSymbol &sym,
                                                                rj_code_arch_t arch,
                                                                std::string *error_out = nullptr);

} // namespace rocjitsu
