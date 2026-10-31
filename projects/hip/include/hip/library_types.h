/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_LIBRARY_TYPES_H
#define HIP_INCLUDE_HIP_LIBRARY_TYPES_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#endif

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

typedef enum hipDataType {
  HIP_R_32F = 0,
  HIP_R_64F = 1,
  HIP_R_16F = 2,
  HIP_R_8I = 3,
  HIP_C_32F = 4,
  HIP_C_64F = 5,
  HIP_C_16F = 6,
  HIP_C_8I = 7,
  HIP_R_8U = 8,
  HIP_C_8U = 9,
  HIP_R_32I = 10,
  HIP_C_32I = 11,
  HIP_R_32U = 12,
  HIP_C_32U = 13,
  HIP_R_16BF = 14,
  HIP_C_16BF = 15,
  HIP_R_4I = 16,
  HIP_C_4I = 17,
  HIP_R_4U = 18,
  HIP_C_4U = 19,
  HIP_R_16I = 20,
  HIP_C_16I = 21,
  HIP_R_16U = 22,
  HIP_C_16U = 23,
  HIP_R_64I = 24,
  HIP_C_64I = 25,
  HIP_R_64U = 26,
  HIP_C_64U = 27,
  HIP_R_8F_E4M3 = 28,
  HIP_R_8F_E5M2 = 29,
  HIP_R_8F_UE8M0 = 30,
  HIP_R_6F_E2M3 = 31,
  HIP_R_6F_E3M2 = 32,
  HIP_R_4F_E2M1 = 33,
  // HIP specific Data Types
  HIP_R_8F_E4M3_FNUZ = 1000,
  HIP_R_8F_E5M2_FNUZ = 1001,
} hipDataType;

typedef enum hipLibraryPropertyType {
  HIP_LIBRARY_MAJOR_VERSION,
  HIP_LIBRARY_MINOR_VERSION,
  HIP_LIBRARY_PATCH_LEVEL
} hipLibraryPropertyType;

/** Enum to configure when libraries use floating-point emulation. */
typedef enum hipEmulationStrategy_t {
  /** Default strategy. Equivalent to HIP_EMULATION_STRATEGY_PERFORMANT unless
   *  overridden by a library-dependent environment variable. */
  HIP_EMULATION_STRATEGY_DEFAULT = 0,
  /** Use emulation when it provides a performance benefit. */
  HIP_EMULATION_STRATEGY_PERFORMANT = 1,
  /** Use emulation whenever possible. */
  HIP_EMULATION_STRATEGY_EAGER = 2,
} hipEmulationStrategy;

/** Enum to configure mantissa-related parameters of floating-point emulation algorithms. */
typedef enum hipEmulationMantissaControl_t {
  /** The number of retained mantissa bits is computed at runtime to ensure the
   *  same or better accuracy than the floating-point format being emulated. */
  HIP_EMULATION_MANTISSA_CONTROL_DYNAMIC = 0,
  /** The number of retained mantissa bits is fixed (set by the user). */
  HIP_EMULATION_MANTISSA_CONTROL_FIXED = 1,
} hipEmulationMantissaControl;

/** Bitmask selecting which floating-point special values emulation algorithms
 *  must handle. Bit 0 = signed infinities, bit 1 = NaN. Values may be combined
 *  with bitwise OR. */
typedef enum hipEmulationSpecialValuesSupport_t {
  /** Default mask: support signed infinities and NaN values. */
  HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT = 0xFFFF,
  /** No special-value handling required (faster; Inf/NaN inputs may silently
   *  corrupt output). */
  HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NONE = 0,
  /** Handle signed infinity inputs and outputs. */
  HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY = 1,
  /** Handle NaN inputs and outputs. */
  HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN = 2,
} hipEmulationSpecialValuesSupport;

#ifndef __cplusplus
typedef enum hipEmulationStrategy_t hipEmulationStrategy_t;
typedef enum hipEmulationMantissaControl_t hipEmulationMantissaControl_t;
typedef enum hipEmulationSpecialValuesSupport_t hipEmulationSpecialValuesSupport_t;
#endif

#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include <library_types.h>
#if !defined(__HIPCC_RTC__)
#include <cuda_runtime_api.h>
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
#define hipEmulationStrategy cudaEmulationStrategy
#define hipEmulationStrategy_t cudaEmulationStrategy_t
#define HIP_EMULATION_STRATEGY_DEFAULT CUDA_EMULATION_STRATEGY_DEFAULT
#define HIP_EMULATION_STRATEGY_PERFORMANT CUDA_EMULATION_STRATEGY_PERFORMANT
#define HIP_EMULATION_STRATEGY_EAGER CUDA_EMULATION_STRATEGY_EAGER
#endif  // CUDART_VERSION >= 13000

/* cudaEmulationMantissaControl and cudaEmulationSpecialValuesSupport first
 * appeared in CUDA 13.0.2, but CUDART_VERSION is 13000 for all 13.0.x
 * releases, so require CUDA 13.1 or later to be safe. */
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13010
#define hipEmulationMantissaControl cudaEmulationMantissaControl
#define hipEmulationMantissaControl_t cudaEmulationMantissaControl_t
#define HIP_EMULATION_MANTISSA_CONTROL_DYNAMIC CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC
#define HIP_EMULATION_MANTISSA_CONTROL_FIXED CUDA_EMULATION_MANTISSA_CONTROL_FIXED

#define hipEmulationSpecialValuesSupport cudaEmulationSpecialValuesSupport
#define hipEmulationSpecialValuesSupport_t cudaEmulationSpecialValuesSupport_t
#define HIP_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_DEFAULT
#define HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NONE CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NONE
#define HIP_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_INFINITY
#define HIP_EMULATION_SPECIAL_VALUES_SUPPORT_NAN CUDA_EMULATION_SPECIAL_VALUES_SUPPORT_NAN
#endif  // CUDART_VERSION >= 13010

#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#endif
