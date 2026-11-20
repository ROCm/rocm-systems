/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Included by fakes/sym_kernels_host_include.h: the generated sym_kernels_host.cc declares 86
// kernels as __global__, which that header neuters before including this and the generated file.
// Its callers compile into separate binaries, so including it from each is not an ODR or link
// collision.

#ifndef RCCL_TEST_HOST_FAKES_SYM_KERNELS_TEST_STUBS_H_
#define RCCL_TEST_HOST_FAKES_SYM_KERNELS_TEST_STUBS_H_

#define RCCL_SYMK_TEST_KERNEL_IDS(X) \
  X(ncclSymkDevKernel_AllGather_LL) \
  X(ncclSymkDevKernel_AllGather_ST) \
  X(ncclSymkDevKernel_AllGather_RailRing_LsaST) \
  X(ncclSymkDevKernel_AllGather_HierLsaST) \
  X(ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f32) \
  X(ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_f32) \
  X(ncclSymkDevKernel_ReduceScatter_LL_sum_f32) \
  X(ncclSymkDevKernel_ReduceScatter_LD_sum_f32) \
  X(ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_f32) \
  X(ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f16) \
  X(ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_f16) \
  X(ncclSymkDevKernel_ReduceScatter_LL_sum_f16) \
  X(ncclSymkDevKernel_ReduceScatter_LD_sum_f16) \
  X(ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_f16) \
  X(ncclSymkDevKernel_AllReduce_AGxLL_R_sum_bf16) \
  X(ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_bf16) \
  X(ncclSymkDevKernel_ReduceScatter_LL_sum_bf16) \
  X(ncclSymkDevKernel_ReduceScatter_LD_sum_bf16) \
  X(ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_bf16) \
  X(ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f8e4m3) \
  X(ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_f8e4m3) \
  X(ncclSymkDevKernel_ReduceScatter_LL_sum_f8e4m3) \
  X(ncclSymkDevKernel_ReduceScatter_LD_sum_f8e4m3) \
  X(ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_f8e4m3) \
  X(ncclSymkDevKernel_AllReduce_AGxLL_R_sum_f8e5m2) \
  X(ncclSymkDevKernel_AllReduce_RSxLD_AGxST_sum_f8e5m2) \
  X(ncclSymkDevKernel_ReduceScatter_LL_sum_f8e5m2) \
  X(ncclSymkDevKernel_ReduceScatter_LD_sum_f8e5m2) \
  X(ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_sum_f8e5m2) \
  X(ncclSymkDevKernel_ReduceScatter_LL_avg_f32) \
  X(ncclSymkDevKernel_ReduceScatter_LD_avg_f32) \
  X(ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_f32) \
  X(ncclSymkDevKernel_ReduceScatter_LL_avg_f16) \
  X(ncclSymkDevKernel_ReduceScatter_LD_avg_f16) \
  X(ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_f16) \
  X(ncclSymkDevKernel_ReduceScatter_LL_avg_bf16) \
  X(ncclSymkDevKernel_ReduceScatter_LD_avg_bf16) \
  X(ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_bf16) \
  X(ncclSymkDevKernel_ReduceScatter_LL_avg_f8e4m3) \
  X(ncclSymkDevKernel_ReduceScatter_LD_avg_f8e4m3) \
  X(ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_f8e4m3) \
  X(ncclSymkDevKernel_ReduceScatter_LL_avg_f8e5m2) \
  X(ncclSymkDevKernel_ReduceScatter_LD_avg_f8e5m2) \
  X(ncclSymkDevKernel_ReduceScatter_RailA2A_LsaLD_avg_f8e5m2) \
  RCCL_SYMK_TEST_TMA_KERNEL_IDS(X)

// The DMA-staged kernels, emitted only when gfx1250 is in GPU_TARGETS. Listed unconditionally:
// elsewhere they are simply unreferenced and --gc-sections drops them.
#define RCCL_SYMK_TEST_TMA_KERNEL_IDS(X) \
  X(ncclSymkDevKernel_AllGather_TmaST) \
  X(ncclSymkDevKernel_AllGather_HierLsaTmaST) \
  X(ncclSymkDevKernel_AllReduce_RSxTmaLD_AGxTmaST_sum_f32) \
  X(ncclSymkDevKernel_AllReduce_RSxTmaLD_AGxTmaST_sum_f16) \
  X(ncclSymkDevKernel_AllReduce_RSxTmaLD_AGxTmaST_sum_bf16) \
  X(ncclSymkDevKernel_AllReduce_RSxTmaLD_AGxTmaST_sum_f8e4m3) \
  X(ncclSymkDevKernel_AllReduce_RSxTmaLD_AGxTmaST_sum_f8e5m2) \
  X(ncclSymkDevKernel_ReduceScatter_TmaLD_sum_f32) \
  X(ncclSymkDevKernel_ReduceScatter_TmaLD_sum_f16) \
  X(ncclSymkDevKernel_ReduceScatter_TmaLD_sum_bf16) \
  X(ncclSymkDevKernel_ReduceScatter_TmaLD_sum_f8e4m3) \
  X(ncclSymkDevKernel_ReduceScatter_TmaLD_sum_f8e5m2) \
  X(ncclSymkDevKernel_ReduceScatter_TmaLD_avg_f32) \
  X(ncclSymkDevKernel_ReduceScatter_TmaLD_avg_f16) \
  X(ncclSymkDevKernel_ReduceScatter_TmaLD_avg_bf16) \
  X(ncclSymkDevKernel_ReduceScatter_TmaLD_avg_f8e4m3) \
  X(ncclSymkDevKernel_ReduceScatter_TmaLD_avg_f8e5m2)

// One stub per kernel plus its _profile sibling: both arrays sym_kernels_host.cc defines need a real address.
#define RCCL_SYMK_TEST_STUB(name) \
  void name(ncclSymkDevWorkArgs4K const) {} \
  void name##_profile(ncclSymkDevWorkArgs4K const) {}
RCCL_SYMK_TEST_KERNEL_IDS(RCCL_SYMK_TEST_STUB)
#undef RCCL_SYMK_TEST_STUB
#undef RCCL_SYMK_TEST_KERNEL_IDS
#undef RCCL_SYMK_TEST_TMA_KERNEL_IDS

#endif  // RCCL_TEST_HOST_FAKES_SYM_KERNELS_TEST_STUBS_H_
