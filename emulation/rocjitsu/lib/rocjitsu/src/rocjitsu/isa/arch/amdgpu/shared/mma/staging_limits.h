// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_STAGING_LIMITS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_STAGING_LIMITS_H_

#include <cstddef>
#include <cstdint>

namespace rocjitsu::amdgpu {

// Shared by the scalar adapter and SIMD backends. SIMD rows pad N to the
// native vector width; larger shapes retain their existing scalar fallback.
constexpr size_t MFMA_SIMD_MAX_AB = 2048;      // max M*K
constexpr size_t MFMA_SIMD_MAX_BSTRIDE = 4096; // max K*stride
constexpr size_t MFMA_SIMD_MAX_C = 1024;       // max M*stride
// F64 has half as many SIMD lanes and retains its smaller padded B buffer.
constexpr size_t MFMA_F64_SIMD_MAX_BSTRIDE = 2048;

static_assert((MFMA_SIMD_MAX_AB + MFMA_SIMD_MAX_BSTRIDE + MFMA_SIMD_MAX_C) * sizeof(float) <=
                  48 * 1024,
              "MFMA staging buffers exceed the 48 KiB stack budget");
static_assert((MFMA_SIMD_MAX_AB + MFMA_SIMD_MAX_BSTRIDE + MFMA_SIMD_MAX_C) * sizeof(int32_t) <=
                  48 * 1024,
              "MFMA i8 staging buffers exceed the 48 KiB stack budget");
static_assert((MFMA_SIMD_MAX_AB + MFMA_F64_SIMD_MAX_BSTRIDE + MFMA_SIMD_MAX_C) * sizeof(double) <=
                  48 * 1024,
              "MFMA f64 staging buffers exceed the 48 KiB stack budget");

// Bounds every real WMMA shape (M <= 32, N <= 16, K <= 128), including
// column padding to a SIMD-width multiple.
constexpr size_t WMMA_SIMD_MAX_AB = 4096;      // max M*K
constexpr size_t WMMA_SIMD_MAX_BSTRIDE = 4096; // max K*stride
constexpr size_t WMMA_SIMD_MAX_C = 1024;       // max M*stride
// Tripwire against silent stack-frame growth in WMMA/SWMMAC staging.
static_assert((WMMA_SIMD_MAX_AB + WMMA_SIMD_MAX_BSTRIDE + WMMA_SIMD_MAX_C) * sizeof(float) <=
                  48 * 1024,
              "WMMA SIMD staging buffers exceed the 48 KiB stack budget");

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_STAGING_LIMITS_H_
