/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

// Thin adapter over RCCL's generated internal device headers.
// We reuse the source-of-truth struct layouts and only provide a tiny subset
// of device-side helpers that those headers intentionally hide in this context.

#include <hip/hip_runtime.h>

#include "nccl_device/core_tmp.h"
#include "nccl_device/impl/core__types.h"
#include "nccl_device/impl/comm__types.h"

#undef NCCL_DEVICE_INLINE
#undef NCCL_HOST_DEVICE_INLINE
#define NCCL_DEVICE_INLINE __device__ __forceinline__
#define NCCL_HOST_DEVICE_INLINE __host__ __device__ __forceinline__

namespace cuda
{
enum memory_order
{
    memory_order_relaxed,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel
};
} // namespace cuda

struct ncclCoopCta
{
    NCCL_DEVICE_INLINE int thread_rank() const { return threadIdx.x; }
    NCCL_DEVICE_INLINE int size() const { return blockDim.x; }
    NCCL_DEVICE_INLINE int num_threads() const { return blockDim.x; }
    NCCL_DEVICE_INLINE void sync() const { __syncthreads(); }
};

NCCL_DEVICE_INLINE char* ncclAdd4G(char* base, int delta4G)
{
    union
    {
        uint32_t u32[2];
        char*    ptr;
    } tmp;

    tmp.ptr = base;
    tmp.u32[1] += delta4G;
    return tmp.ptr;
}

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamLsaLocal(ncclDevComm const& comm)
{
    ncclTeam ans;
    ans.nRanks = comm.lsaSize;
    ans.rank   = comm.lsaRank;
    ans.stride = 1;
    return ans;
}

NCCL_DEVICE_INLINE void* ncclGetLsaPointerLocal(ncclWindow_t window, size_t offset, int peer)
{
    return static_cast<void*>(ncclAdd4G(window->lsaFlatBase, peer * window->stride4G) + offset);
}

#define ncclTeamLsa ncclTeamLsaLocal
#define ncclGetLsaPointer ncclGetLsaPointerLocal

template <typename Coop>
struct ncclLsaBarrierSession
{
    Coop               coop;
    ncclDevComm const& comm;
    ncclTeam           team;
    ncclLsaBarrierHandle handle;
    int                index;
    bool               multimem;
    ncclMultimemHandle mmHandle;

    NCCL_DEVICE_INLINE ncclLsaBarrierSession(
        Coop coop_,
        ncclDevComm const& comm_,
        ncclTeam team_,
        ncclLsaBarrierHandle handle_,
        uint32_t index_,
        bool multimem_ = false,
        ncclMultimemHandle mmHandle_ = {}
    )
        : coop(coop_)
        , comm(comm_)
        , team(team_)
        , handle(handle_)
        , index(static_cast<int>(index_))
        , multimem(multimem_)
        , mmHandle(mmHandle_)
    {
    }

    NCCL_DEVICE_INLINE ncclLsaBarrierSession(
        Coop coop_, ncclDevComm const& comm_, ncclTeamTagLsa, uint32_t index_, bool multimem_ = false
    )
        : ncclLsaBarrierSession(
              coop_, comm_, ncclTeamLsa(comm_), comm_.lsaBarrier, index_, multimem_, comm_.lsaMultimem
          )
    {
    }

    NCCL_DEVICE_INLINE ~ncclLsaBarrierSession() { coop.sync(); }

    NCCL_DEVICE_INLINE void arrive(Coop, cuda::memory_order)
    {
        coop.sync();
        __threadfence_system();
    }

    NCCL_DEVICE_INLINE void wait(Coop, cuda::memory_order) { coop.sync(); }

    NCCL_DEVICE_INLINE void sync(Coop c, cuda::memory_order order)
    {
        arrive(c, order);
        wait(c, order);
    }
};
