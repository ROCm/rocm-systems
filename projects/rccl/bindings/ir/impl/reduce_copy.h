/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

/*
 * IR-only ReduceCopy API declarations - moved from nccl_device_wrapper__impl.h
 */

#ifndef _NCCL_IR_IMPL_REDUCE_COPY_H_
#define _NCCL_IR_IMPL_REDUCE_COPY_H_

#include "util.h"

/* LSA reduce operations */
#define NCCL_IR_DEFINE_ncclLsaReduceSum(suffix, type) \
  extern "C" __device__ void ncclLsaReduceSum_##suffix( \
      ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset, \
      type* dst, size_t count, ncclTeam team);
NCCL_IR_DEFINE_API_ALL_TYPES(ncclLsaReduceSum)

/* Multimem reduce operations */
#define NCCL_IR_DEFINE_ncclMultimemReduceSum(suffix, type) \
  extern "C" __device__ void ncclMultimemReduceSum_##suffix( \
      ncclCoopAny coop, type* mcSrc, type* dst, size_t count);
NCCL_IR_DEFINE_API_MULTIMEM_TYPES(ncclMultimemReduceSum)

/* LSA copy operations */
#define NCCL_IR_DEFINE_ncclLsaCopy(suffix, type) \
  extern "C" __device__ void ncclLsaCopy_##suffix( \
      ncclCoopAny coop, type* src, ncclWindow_t dstWindow, \
      size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_DEFINE_API_ALL_TYPES(ncclLsaCopy)

/* Multimem copy operations */
#define NCCL_IR_DEFINE_ncclMultimemCopy(suffix, type) \
  extern "C" __device__ void ncclMultimemCopy_##suffix( \
      ncclCoopAny coop, type* src, type* mcDst, size_t count);
NCCL_IR_DEFINE_API_MULTIMEM_TYPES(ncclMultimemCopy)

/* LSA reduce+copy operations */
#define NCCL_IR_DEFINE_ncclLsaReduceSumCopy(suffix, type) \
  extern "C" __device__ void ncclLsaReduceSumCopy_##suffix( \
      ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset, \
      ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_DEFINE_API_ALL_TYPES(ncclLsaReduceSumCopy)

/* Multimem reduce+copy operations */
#define NCCL_IR_DEFINE_ncclMultimemReduceSumCopy(suffix, type) \
  extern "C" __device__ void ncclMultimemReduceSumCopy_##suffix( \
      ncclCoopAny coop, type* mcSrc, type* mcDst, size_t count);
NCCL_IR_DEFINE_API_MULTIMEM_TYPES(ncclMultimemReduceSumCopy)

/* Local reduce+copy operations */
#define NCCL_IR_DEFINE_ncclLocalReduceSumCopy(suffix, type) \
  extern "C" __device__ void ncclLocalReduceSumCopy_##suffix( \
      ncclCoopAny coop, int nSrc, type* srcBase, size_t srcDispl, \
      int nDst, type* dstBase, size_t dstDispl, size_t count);
NCCL_IR_DEFINE_API_ALL_TYPES(ncclLocalReduceSumCopy)

#endif // _NCCL_IR_IMPL_REDUCE_COPY_H_
