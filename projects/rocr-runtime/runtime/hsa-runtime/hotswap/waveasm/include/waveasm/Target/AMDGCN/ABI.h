// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// AMDGCN ABI Constants
//===----------------------------------------------------------------------===//
//
// This file centralizes all ABI-related constants for AMDGCN targets.
// Instead of hardcoding magic numbers throughout the codebase, all ABI
// constants should be defined here and referenced by name.
//
// References:
// - AMD ROCm ABI specification
// - LLVM AMDGPU backend documentation
// - HSA runtime specification
//
//===----------------------------------------------------------------------===//

#ifndef WAVEASM_TARGET_AMDGCN_ABI_H
#define WAVEASM_TARGET_AMDGCN_ABI_H

#include <cstdint>

namespace waveasm {
namespace abi {

//===----------------------------------------------------------------------===//
// User SGPR Layout (Kernel Arguments)
//===----------------------------------------------------------------------===//

/// SGPR index for kernarg_segment_ptr (always s[0:1])
constexpr int64_t kKernargPtrSGPR = 0;

/// Size of kernarg pointer in SGPRs (64-bit pointer = 2 SGPRs)
constexpr int64_t kKernargPtrSize = 2;

/// Base index for first preloaded kernel argument (gfx950+ only)
/// On gfx950+, kernel args can be preloaded into s[2:3], s[4:5], etc.
constexpr int64_t kPreloadedArgsBase = 2;

/// Number of SGPRs per preloaded pointer argument
constexpr int64_t kSGPRsPerPointer = 2;

//===----------------------------------------------------------------------===//
// System SGPR Layout (Workgroup IDs)
//===----------------------------------------------------------------------===//

/// Offset from user SGPRs to workgroup ID X
/// Workgroup IDs are system SGPRs that come after all user SGPRs
constexpr int64_t kWorkgroupIdXOffset = 0;

/// Offset from user SGPRs to workgroup ID Y
constexpr int64_t kWorkgroupIdYOffset = 1;

/// Offset from user SGPRs to workgroup ID Z
constexpr int64_t kWorkgroupIdZOffset = 2;

/// Number of system SGPRs for workgroup IDs (x, y, z)
constexpr int64_t kNumWorkgroupIdSGPRs = 3;

//===----------------------------------------------------------------------===//
// SRD (Shader Resource Descriptor) Layout
//===----------------------------------------------------------------------===//

/// Size of a buffer SRD in SGPRs (4 consecutive SGPRs)
constexpr int64_t kSRDSize = 4;

/// SRD alignment requirement (must be aligned to 4 SGPRs)
constexpr int64_t kSRDAlignment = 4;

/// SRD word 2: buffer size field (number of bytes)
/// Format: s[srdBase+2] = buffer_size
constexpr int64_t kSRDSizeWordOffset = 2;

/// SRD word 3: stride and cache swizzle descriptor
/// Format: s[srdBase+3] = stride_descriptor
constexpr int64_t kSRDStrideWordOffset = 3;

/// Default stride descriptor value (0x20000 = stride mode)
/// Bit 17 (NUM_FORMAT) = 1 indicates buffer access
constexpr int64_t kDefaultSRDStride = 0x20000;

//===----------------------------------------------------------------------===//
// VGPR Layout (Thread State)
//===----------------------------------------------------------------------===//

/// VGPR index for flat workitem ID (v0) in multi-wave kernels
constexpr int64_t kFlatWorkitemIdVGPR = 0;

/// Scratch VGPR used for literal constant loads
/// (Used when VOP3 ops need literals outside inline constant range)
constexpr int64_t kScratchVGPR = 255;

//===----------------------------------------------------------------------===//
// Thread ID Bit Packing (for multi-wave kernels)
//===----------------------------------------------------------------------===//

/// Number of bits per dimension for packed thread ID
/// Thread ID X uses bits [0:9], Y uses [10:19], Z uses [20:29]
constexpr int64_t kThreadIdBitsPerDim = 10;

/// Bit offset for thread ID X in packed format
constexpr int64_t kThreadIdXBitOffset = 0;

/// Bit offset for thread ID Y in packed format
constexpr int64_t kThreadIdYBitOffset = 10;

/// Bit offset for thread ID Z in packed format
constexpr int64_t kThreadIdZBitOffset = 20;

//===----------------------------------------------------------------------===//
// Wavefront Properties
//===----------------------------------------------------------------------===//

/// Default wavefront size for AMDGCN (wave64)
constexpr int64_t kDefaultWaveSize = 64;

/// Wave size for gfx10/gfx11+ targets that support wave32
constexpr int64_t kWave32Size = 32;

//===----------------------------------------------------------------------===//
// Register Limits
//===----------------------------------------------------------------------===//

/// Maximum VGPRs available on gfx9 targets
constexpr int64_t kMaxVGPRsGFX9 = 256;

/// Maximum SGPRs available on gfx9 targets
constexpr int64_t kMaxSGPRsGFX9 = 102;

/// Maximum VGPRs available on gfx10/gfx11 targets
constexpr int64_t kMaxVGPRsGFX10 = 256;

/// VGPR allocation granularity for gfx942/gfx950
constexpr int64_t kVGPRGranularityGFX94x = 8;

/// VGPR allocation granularity for other GFX9
constexpr int64_t kVGPRGranularityGFX9 = 4;

/// SGPR allocation granularity (all targets)
constexpr int64_t kSGPRGranularity = 8;

//===----------------------------------------------------------------------===//
// Instruction Encoding Limits
//===----------------------------------------------------------------------===//

/// Inline constant range: -16 to 64 (inclusive)
constexpr int64_t kInlineConstantMin = -16;
constexpr int64_t kInlineConstantMax = 64;

/// Maximum instruction offset for buffer operations (12-bit unsigned)
constexpr int64_t kMaxBufferOffset = 4095;

/// Maximum instruction offset for LDS operations
constexpr int64_t kMaxLDSOffset = 65535;

//===----------------------------------------------------------------------===//
// LDS (Local Data Share) Properties
//===----------------------------------------------------------------------===//

/// LDS allocation granularity in bytes
constexpr int64_t kLDSGranularity = 128;

/// Maximum LDS size per workgroup (64KB for most targets)
constexpr int64_t kMaxLDSSize = 65536;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if a value is within the inline constant range
inline bool isInlineConstant(int64_t value) {
  return value >= kInlineConstantMin && value <= kInlineConstantMax;
}

/// Compute user SGPR count for a given number of kernel arguments
/// On gfx950+, kernel args are preloaded into SGPRs
inline int64_t computeUserSGPRCount(int64_t numArgs, bool usePreloading) {
  int64_t count = kKernargPtrSize; // Always have kernarg pointer
  if (usePreloading) {
    count += numArgs * kSGPRsPerPointer;
  }
  return count;
}

/// Compute SGPR index for workgroup ID in dimension (0=x, 1=y, 2=z)
inline int64_t computeWorkgroupIdSGPR(int64_t userSGPRCount,
                                      int64_t dimension) {
  return userSGPRCount + dimension;
}

/// Compute first SRD SGPR index (aligned to 4)
inline int64_t computeFirstSRDIndex(int64_t userSGPRCount) {
  int64_t afterSystem = userSGPRCount + kNumWorkgroupIdSGPRs;
  return (afterSystem + kSRDAlignment - 1) & ~(kSRDAlignment - 1);
}

/// Align VGPR count to target granularity
inline int64_t alignVGPRCount(int64_t count, int64_t granularity) {
  return ((count + granularity - 1) / granularity) * granularity;
}

/// Align SGPR count to target granularity
inline int64_t alignSGPRCount(int64_t count) {
  return ((count + kSGPRGranularity - 1) / kSGPRGranularity) * kSGPRGranularity;
}

/// Compute LDS block count (for kernel descriptor)
inline int64_t computeLDSBlocks(int64_t ldsSize) {
  return (ldsSize + kLDSGranularity - 1) / kLDSGranularity;
}

} // namespace abi
} // namespace waveasm

#endif // WAVEASM_TARGET_AMDGCN_ABI_H
