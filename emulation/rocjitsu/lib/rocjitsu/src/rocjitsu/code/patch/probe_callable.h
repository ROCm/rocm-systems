// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file probe_callable.h
/// @brief Turn a resolved probe symbol into a self-contained, callable probe
///        body: its instruction words plus a verified calling convention.

#pragma once

#include "rocjitsu/code/rj_code.h"

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

/// @brief A probe body extracted from a code object and verified to be safe to
///        relocate verbatim into the instrumented code object.
struct ProbeCallable {
  std::string symbol;                               ///< Resolved symbol name.
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID; ///< ISA the body decodes as.
  std::vector<uint32_t> body_words;                 ///< The body's instruction words, in order.
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
