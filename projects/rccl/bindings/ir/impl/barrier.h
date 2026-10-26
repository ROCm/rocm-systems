/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

/*
 * IR-only barrier session API declarations - moved from nccl_device_wrapper__impl.h
 */

#ifndef _NCCL_IR_IMPL_BARRIER_H_
#define _NCCL_IR_IMPL_BARRIER_H_

/* LSA barrier session */
extern "C" __device__ void ncclLsaBarrierSessionInit(
    ncclLsaBarrierSession_C* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclLsaBarrierHandle handle,
    uint32_t index,
    bool multimem,
    ncclMultimemHandle mmHandle);
extern "C" __device__ void ncclLsaBarrierSessionArrive(
    ncclLsaBarrierSession_C* session, ncclCoopAny coop, cuda::memory_order order);
extern "C" __device__ void ncclLsaBarrierSessionWait(
    ncclLsaBarrierSession_C* session, ncclCoopAny coop, cuda::memory_order order);
extern "C" __device__ void ncclLsaBarrierSessionSync(
    ncclLsaBarrierSession_C* session, ncclCoopAny coop, cuda::memory_order order);
extern "C" __device__ void ncclLsaBarrierSessionDestroy(ncclLsaBarrierSession_C* session);

/* GIN barrier session */
extern "C" __device__ void ncclGinBarrierSessionInit(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    ncclGin_C const* net,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index);
extern "C" __device__ void ncclGinBarrierSessionInitAllContexts(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index);
extern "C" __device__ void ncclGinBarrierSessionSync(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence);
extern "C" __device__ void ncclGinBarrierSessionDestroy(ncclGinBarrierSession_C* session);

/* Composite barrier session */
extern "C" __device__ void ncclBarrierSessionInit(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    ncclTeam innerTeam,
    ncclTeam outerTeam,
    ncclGin_C const* net,
    ncclLsaBarrierHandle const innerBarHandle,
    ncclGinBarrierHandle const outerBarHandle,
    uint32_t index,
    bool multimem,
    ncclMultimemHandle const innerMmHandle);
extern "C" __device__ void ncclBarrierSessionSync(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence);
extern "C" __device__ void ncclBarrierSessionDestroy(ncclBarrierSession_C* session);

#endif // _NCCL_IR_IMPL_BARRIER_H_
