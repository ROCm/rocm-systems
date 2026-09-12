/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ABCE_TYPES_H_
#define ABCE_TYPES_H_

#include <cstddef>
#include <cstdint>

// Host+device qualifier (mirrors abce_builder.h / abce_ring_core.h; guarded so
// any of them may define it first).
#ifndef ABCE_HD
#if defined(__HIPCC__) || defined(__CUDACC__)
#define ABCE_HD __host__ __device__
#else
#define ABCE_HD
#endif
#endif  // ABCE_HD

namespace abce {

namespace detail {

// Bit counts used by engine-mask handling and rect tiling. GCC/Clang (including
// Clang in MSVC-compatibility mode and during device compilation) get the
// builtins; anything else gets a portable loop. The masks involved are at most
// kMaxEngines bits wide and neither caller is on a hot path, so the fallback
// costs nothing worth an ISA-specific intrinsic.

/// Index of the lowest set bit. @p value MUST be non-zero.
ABCE_HD inline int CountTrailingZeros64(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_ctzll(value);
#else
  int count = 0;
  while ((value & 1u) == 0) {
    value >>= 1;
    ++count;
  }
  return count;
#endif
}

ABCE_HD inline int PopCount64(uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(value);
#else
  int count = 0;
  for (; value != 0; value >>= 1) count += static_cast<int>(value & 1u);
  return count;
#endif
}

}  // namespace detail

inline constexpr uint32_t kKi = 1024;
// One choice per possible engine (matches kMaxEngines in abce_host.h) so the
// fan-out balancer can see and spread across every registered ring, not a
// truncated subset.
inline constexpr uint32_t kMaxEngineChoices = 16;

/// Hardware instruction-set version used to select SDMA packet layouts.
struct IsaVersion {
  uint32_t major = 0;
  uint32_t minor = 0;
  uint32_t stepping = 0;
};

enum class SdmaIpVersion : uint8_t {
  kOSS4,  ///< Legacy packets, no explicit GCR.
  kOSS5,  ///< Legacy packets with GCR invalidate/writeback.
  kOSS7,  ///< Scope-bearing packets, no explicit GCR.
};

struct PacketCaps {
  SdmaIpVersion version = SdmaIpVersion::kOSS4;
  bool use_gcr = false;
  bool scope_fields = false;
  bool gfx125plus = false;
  size_t max_linear_copy_size = 0x3fffe0;
};

/// Derive inherent packet/coherency capabilities from the gfx IP.
constexpr PacketCaps DetectPacketCaps(const IsaVersion& isa) {
  if (isa.major == 9) {
    const size_t max_copy_size =
        isa.minor >= 4 || (isa.minor == 0 && isa.stepping == 10) ? 0x3fffffff : 0x3fffff;
    return {SdmaIpVersion::kOSS4, false, false, false, max_copy_size};
  }
  if (isa.major == 10) {
    const size_t max_copy_size = isa.minor < 3 ? 0x3fffff : 0x3fffffff;
    return {SdmaIpVersion::kOSS5, true, false, false, max_copy_size};
  }
  if ((isa.major == 11 || isa.major == 12) && isa.minor < 5)
    return {SdmaIpVersion::kOSS5, true, false, false, 0x3fffffff};
  if ((isa.major == 11 || isa.major == 12) && isa.minor >= 5)
    return {SdmaIpVersion::kOSS7, false, true, isa.major == 12 && isa.minor >= 5, 0x3fffffff};
  return {};
}

struct BuilderConfig {
  bool use_copy_size_override = true;
  size_t max_linear_copy_size = 0;
  size_t max_fill_size = 0;
};

enum class LinearBatchMode : uint8_t {
  kAutomatic,
  kForceBackToBack,
  kForceFanOut,
};

enum class MulticastMode : uint8_t {
  kAutomatic,
  kForceMulticast,
  kForceFanOut,
};

/// Capabilities and policy decisions supplied by the surrounding runtime.
///
/// These values cannot be inferred solely from the gfx IP. For example, the
/// same IP may be exposed with or without HDP flush support.
struct PlatformCaps {
  bool device_atomic_support = false;
  bool emit_hdp_flush = false;
  bool driver_manages_gcr = false;
};

/// ROCr-compatible IP defaults. A runtime should override these when link
/// topology says otherwise (for example an xGMI host link disables HDP flush).
constexpr PlatformCaps DetectDefaultPlatformCaps(const IsaVersion& isa) {
  const bool atomic_support = !(isa.major == 7 && isa.minor == 0 && isa.stepping == 1);
  const bool hdp_flush = isa.major >= 9 && !(isa.major == 10 && isa.minor == 1);
  return {atomic_support, hdp_flush, false};
}

struct Dim3 {
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t z = 0;
};

struct PitchedPtr {
  void* base = nullptr;
  size_t pitch = 0;
  size_t slice = 0;
};

}  // namespace abce

#endif  // ABCE_TYPES_H_
