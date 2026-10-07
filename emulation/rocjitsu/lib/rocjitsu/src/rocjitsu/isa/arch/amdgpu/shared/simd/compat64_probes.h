// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_COMPAT64_PROBES_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_COMPAT64_PROBES_H_

#define ROCJITSU_TRY_SIMD_VOP2_FMA_F64() static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_VOP2_BINARY_FP64(...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_VOP1_UNARY_F64(T, ...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_CVT_F64_TO_B32(Tout, ...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_CVT_B32_TO_F64(Tin, ...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_CVT_VOP3_F64_TO_B32_FP(...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_VOPC64(T, ...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_VOPC_CLASS_F64(...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_VOP3_CLASS_F64(SM, ...) static_cast<void>(inst)
#define ROCJITSU_TRY_SIMD_VOP3_CLASS_F64_RESULT(WRITE_RESULT, SM, ...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_VOPC64_VOP3_INT(T, ...) static_cast<void>(inst)
#define ROCJITSU_TRY_SIMD_VOPC64_VOP3_INT_RESULT(WRITE_RESULT, T, ...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_FMA_VOP3_FP64() static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP64(...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_FMAC_VOP3_MODE_FP64() static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_FMAC_VOP3_FP64(...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_LDEXP_VOP3_FP64(...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_DIV_FMAS_VOP3_FP64() static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_VOP3_BINARY_FP64(...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_VOP3_UNARY_FP64(...) static_cast<void>(inst)

#define ROCJITSU_TRY_SIMD_MAD_WIDE64_VOP3(...) static_cast<void>(inst)
#define ROCJITSU_TRY_SIMD_MAD_WIDE64_VOP3_RESULT(WRITE_RESULT, ...) static_cast<void>(inst)

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_COMPAT64_PROBES_H_
