// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_NATIVE64_PROBES_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_NATIVE64_PROBES_H_

#define ROCJITSU_TRY_SIMD_VOP2_FMA_F64()                                                           \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop2_f64_simd<double>(inst, wf))                     \
  return

#define ROCJITSU_TRY_SIMD_VOP2_BINARY_FP64(...)                                                    \
  if (::rocjitsu::amdgpu::try_execute_binary_vop2_f64_simd(inst, wf, __VA_ARGS__))                 \
  return

#define ROCJITSU_TRY_SIMD_VOP1_UNARY_F64(T, ...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_unary_vop1_f64_simd<T>(inst, wf, __VA_ARGS__))               \
  return

#define ROCJITSU_TRY_SIMD_CVT_F64_TO_B32(Tout, ...)                                                \
  if (::rocjitsu::amdgpu::try_execute_cvt_f64_to_b32_simd<Tout>(inst, wf, __VA_ARGS__))            \
  return

#define ROCJITSU_TRY_SIMD_CVT_B32_TO_F64(Tin, ...)                                                 \
  if (::rocjitsu::amdgpu::try_execute_cvt_b32_to_f64_simd<Tin>(inst, wf, __VA_ARGS__))             \
  return

#define ROCJITSU_TRY_SIMD_CVT_VOP3_F64_TO_B32_FP(...)                                              \
  if (::rocjitsu::amdgpu::try_execute_cvt_vop3_f64_to_b32_fp_simd(inst, wf, __VA_ARGS__))          \
  return

#define ROCJITSU_TRY_SIMD_VOPC64(T, ...)                                                           \
  if (::rocjitsu::amdgpu::try_execute_vopc64_simd<T>(inst, wf, __VA_ARGS__))                       \
  return

#define ROCJITSU_TRY_SIMD_VOPC_CLASS_F64(...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_vopc_class_f64_simd(inst, wf, __VA_ARGS__))                  \
  return

#define ROCJITSU_TRY_SIMD_VOP3_CLASS_F64(SM, ...)                                                  \
  if (::rocjitsu::amdgpu::try_execute_vop3_class_f64_simd(                                         \
          inst, wf, SM, __VA_ARGS__, [&](uint64_t result) {                                        \
            ::rocjitsu::amdgpu::write_explicit_lane_mask(inst.vdst, wf, result);                   \
          }))                                                                                      \
  return
#define ROCJITSU_TRY_SIMD_VOP3_CLASS_F64_RESULT(WRITE_RESULT, SM, ...)                             \
  if (::rocjitsu::amdgpu::try_execute_vop3_class_f64_simd(inst, wf, SM, __VA_ARGS__,               \
                                                          WRITE_RESULT))                           \
  return

#define ROCJITSU_TRY_SIMD_VOPC64_VOP3_INT(T, ...)                                                  \
  if (::rocjitsu::amdgpu::try_execute_vopc64_vop3_int_simd<T>(                                     \
          inst, wf, __VA_ARGS__, [&](uint64_t result) {                                            \
            ::rocjitsu::amdgpu::write_explicit_lane_mask(inst.vdst, wf, result);                   \
          }))                                                                                      \
  return
#define ROCJITSU_TRY_SIMD_VOPC64_VOP3_INT_RESULT(WRITE_RESULT, T, ...)                             \
  if (::rocjitsu::amdgpu::try_execute_vopc64_vop3_int_simd<T>(inst, wf, __VA_ARGS__,               \
                                                              WRITE_RESULT))                       \
  return

#define ROCJITSU_TRY_SIMD_FMA_VOP3_FP64()                                                          \
  if (::rocjitsu::amdgpu::try_execute_fma_vop3_fp64_simd(inst, wf))                                \
  return

#define ROCJITSU_TRY_SIMD_VOP3_TERNARY_FP64(...)                                                   \
  if (::rocjitsu::amdgpu::try_execute_ternary_vop3_fp64_simd(inst, wf, __VA_ARGS__))               \
  return

#define ROCJITSU_TRY_SIMD_FMAC_VOP3_MODE_FP64()                                                    \
  if (::rocjitsu::amdgpu::try_execute_fmac_vop3_fp64_mode_simd(inst, wf))                          \
  return

#define ROCJITSU_TRY_SIMD_FMAC_VOP3_FP64(...)                                                      \
  if (::rocjitsu::amdgpu::try_execute_fmac_vop3_fp64_simd(inst, wf, __VA_ARGS__))                  \
  return

#define ROCJITSU_TRY_SIMD_LDEXP_VOP3_FP64(...)                                                     \
  if (::rocjitsu::amdgpu::try_execute_ldexp_vop3_fp64_simd(inst, wf, __VA_ARGS__))                 \
  return

#define ROCJITSU_TRY_SIMD_DIV_FMAS_VOP3_FP64()                                                     \
  if (::rocjitsu::amdgpu::try_execute_div_fmas_f64_simd(inst, wf))                                 \
  return

#define ROCJITSU_TRY_SIMD_VOP3_BINARY_FP64(...)                                                    \
  if (::rocjitsu::amdgpu::try_execute_binary_vop3_fp64_simd(inst, wf, __VA_ARGS__))                \
  return

#define ROCJITSU_TRY_SIMD_VOP3_UNARY_FP64(...)                                                     \
  if (::rocjitsu::amdgpu::try_execute_unary_vop3_fp64_simd(inst, wf, __VA_ARGS__))                 \
  return

#define ROCJITSU_TRY_SIMD_MAD_WIDE64_VOP3(...)                                                     \
  if (::rocjitsu::amdgpu::try_execute_mad_wide64_vop3_simd(inst, wf, __VA_ARGS__))                 \
  return
#define ROCJITSU_TRY_SIMD_MAD_WIDE64_VOP3_RESULT(WRITE_RESULT, ...)                                \
  if (::rocjitsu::amdgpu::try_execute_mad_wide64_vop3_result_simd(inst, wf, __VA_ARGS__,           \
                                                                  WRITE_RESULT))                   \
  return

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_SIMD_NATIVE64_PROBES_H_
