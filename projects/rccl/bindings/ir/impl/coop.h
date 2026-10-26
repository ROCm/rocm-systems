/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

/*
 * IR-only cooperative groups ABI declarations - moved from nccl_device_wrapper__impl.h
 */

#ifndef _NCCL_IR_IMPL_COOP_H_
#define _NCCL_IR_IMPL_COOP_H_

/* Cooperative group initialization */
extern "C" __device__ void ncclCoopAnyInitThread(ncclCoopAny* coop);
extern "C" __device__ void ncclCoopAnyInitWarp(ncclCoopAny* coop);
extern "C" __device__ void ncclCoopAnyInitLanes(ncclCoopAny* coop, ncclCoopMask_t lane_mask);
extern "C" __device__ void ncclCoopAnyInitWarpSpan(
    ncclCoopAny* coop, int warp0, int nWarps, int id, void* barrierLds);
extern "C" __device__ void ncclCoopAnyInitCta(ncclCoopAny* coop);

/* Cooperative group utility */
extern "C" __device__ int ncclCoopThreadRank(const ncclCoopAny* coop);
extern "C" __device__ int ncclCoopSize(const ncclCoopAny* coop);
extern "C" __device__ int ncclCoopNumThreads(const ncclCoopAny* coop);
extern "C" __device__ void ncclCoopSync(const ncclCoopAny* coop);

#endif // _NCCL_IR_IMPL_COOP_H_
