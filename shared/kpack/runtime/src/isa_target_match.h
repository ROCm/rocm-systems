// Copyright (c) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// ISA target matching with feature-flag awareness.
//
// GPU architectures may include ISA feature flags after a colon separator
// (e.g., "gfx942:sramecc+:xnack-"). When loading code objects, the runtime
// receives the agent's full ISA from HSA, but the code object may have been
// compiled with fewer (or no) feature flags. Omitted features mean "Any" —
// compatible with either On or Off.
//
// This matches the semantics used throughout the ROCm stack:
//   - LLVM:  clang/lib/Basic/TargetID.cpp  isCompatibleTargetID()
//   - comgr: amd/comgr/src/comgr-metadata.cpp  isCompatibleIsaName()
//   - HSA:   core/runtime/isa.cpp  Isa::IsCompatible()
//   - CLR:   rocclr/device/device.cpp  Isa::isCompatible()
//
// Example: agent reports "gfx942:sramecc+:xnack-". Compatible archives may
// be named with any subset of those features:
//   gfx942:sramecc+:xnack-  (fully qualified)
//   gfx942:sramecc+          (xnack=Any, e.g. compiled without xnack spec)
//   gfx942:xnack-            (sramecc=Any)
//   gfx942                   (all features=Any, typical release build)

#ifndef KPACK_ISA_TARGET_MATCH_H
#define KPACK_ISA_TARGET_MATCH_H

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace kpack {

// Parsed ISA target: processor name + individual feature flags.
struct ParsedTarget {
  std::string processor;              // e.g., "gfx942"
  std::vector<std::string> features;  // e.g., ["sramecc+", "xnack-"]
};

// Strip "amdgcn-amd-amdhsa--" prefix if present.
// "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-" → "gfx942:sramecc+:xnack-"
// "gfx1201" → "gfx1201" (no-op)
std::string strip_target_prefix(const char* isa);

// Parse a target ID string into processor + features.
// "gfx942:sramecc+:xnack-" → { "gfx942", ["sramecc+", "xnack-"] }
// "gfx1201" → { "gfx1201", [] }
// "" → { "", [] }
ParsedTarget parse_target_id(const std::string& target);

// Iterate compatible ISA target strings from most specific to least.
//
// Calls callback(target_string) for each compatible target. Stops early if
// the callback returns true (match found). Returns true if any callback
// returned true.
//
// Given agent ISA "gfx942:sramecc+:xnack-", calls back with:
//   "gfx942:sramecc+:xnack-"  (all features — exact match)
//   "gfx942:sramecc+"          (drop xnack — xnack=Any archive)
//   "gfx942:xnack-"            (drop sramecc — sramecc=Any archive)
//   "gfx942"                    (bare processor — all features=Any)
//
// For consumer cards with no features (e.g., "gfx1201"), calls back once
// with just the bare processor name.
template <typename Fn>
bool for_each_compatible_target(const char* agent_isa, Fn&& callback) {
  if (!agent_isa || !agent_isa[0]) return false;

  std::string stripped = strip_target_prefix(agent_isa);
  if (stripped.empty()) return false;

  ParsedTarget parsed = parse_target_id(stripped);
  if (parsed.processor.empty()) return false;

  const size_t n = parsed.features.size();

  if (n == 0) {
    // No features — single callback with bare processor
    return callback(parsed.processor);
  }

  // Iterate power set of features, descending cardinality (most specific first).
  // Bit (n-1-i) corresponds to features[i], so descending mask order drops
  // features from the right first — preserving the original feature ordering.
  const unsigned full_mask = (1u << n) - 1;
  for (unsigned mask = full_mask;; --mask) {
    std::string candidate = parsed.processor;
    for (size_t i = 0; i < n; ++i) {
      if (mask & (1u << (n - 1 - i))) {
        candidate += ':';
        candidate += parsed.features[i];
      }
    }

    if (callback(candidate)) return true;

    if (mask == 0) break;
  }

  return false;
}

}  // namespace kpack

#endif  // KPACK_ISA_TARGET_MATCH_H
