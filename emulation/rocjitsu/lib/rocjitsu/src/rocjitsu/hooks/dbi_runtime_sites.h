// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbi_runtime_sites.h
/// @brief Site selection and patching for the DBI runtime hook, free of HSA.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {
class AmdGpuCodeObject;
} // namespace rocjitsu

namespace rocjitsu::hooks {

/// @brief Whether the DBI runtime instruments a code object, and if not, why.
///
/// PatchFailed stays last: kDbiLoadDecisionCount counts through it.
enum class DbiLoadDecision : uint8_t {
  Instrumented,           ///< The patched object replaces the original.
  NotAmdGpuObject,        ///< The bytes do not parse as an AMDGPU code object.
  UnsupportedArch,        ///< DBI is not validated end to end on the object's target.
  ProbeTargetMismatch,    ///< The probe was built for a different target than the object.
  NotSingleKernel,        ///< The object has zero or several kernels.
  UnsupportedControlFlow, ///< The kernel makes a call or has unresolved control flow.
  NoSites,                ///< No reachable block has an instruction that can anchor a site.
  PatchFailed,            ///< The Instrumentor refused the object; see the errors.
};

inline constexpr size_t kDbiLoadDecisionCount =
    static_cast<size_t>(DbiLoadDecision::PatchFailed) + 1;

/// @brief Stable name of @p decision, for counters and diagnostics.
[[nodiscard]] const char *dbi_load_decision_name(DbiLoadDecision decision);

/// @brief The probe called at every site.
struct DbiProbe {
  const AmdGpuCodeObject *object = nullptr; ///< Borrowed; must outlive the call.
  std::string symbol;
};

/// @brief Outcome of instrumenting one code object.
struct DbiInstrumentation {
  DbiLoadDecision decision = DbiLoadDecision::NotAmdGpuObject;
  std::vector<uint8_t> elf_bytes;   ///< The patched object when Instrumented.
  uint32_t sites = 0;               ///< Sites requested; applied only when Instrumented.
  uint32_t blocks_without_site = 0; ///< Blocks with no instruction that can anchor one.
  std::vector<std::string> errors;  ///< Instrumentor diagnostics when PatchFailed.
};

/// @brief Instrument each basic block of @p elf with a call to @p probe.
///
/// @details Covers the blocks the kernel can reach from its entry along
/// successors(); dead code is left alone. A block's site is its first
/// instruction that can anchor a trampoline (Instrumentor::first_anchor());
/// blocks without one are counted and skipped. Only single-kernel objects on a
/// target DBI is validated on are instrumented, and only when the reachable
/// code makes no call and has no unresolved control flow, because the liveness
/// the trampolines rely on follows neither. Patching is all-or-nothing, so one
/// refused site leaves the whole object unpatched.
///
/// @pre @p probe.object is a valid code object.
[[nodiscard]] DbiInstrumentation instrument_each_block(std::span<const uint8_t> elf,
                                                       const DbiProbe &probe);

} // namespace rocjitsu::hooks
