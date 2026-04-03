/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

// Minimal AMD-friendly device API shim for unit tests.
// This mirrors the working local header pattern used by rccl-tests.

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t ncclDevResourceHandle;
typedef ncclDevResourceHandle ncclDevResourceHandle_t;

struct ncclDevComm;
typedef struct ncclDevComm ncclDevComm_t;

struct ncclTeam;
typedef struct ncclTeam ncclTeam_t;

struct ncclMultimemHandle;
typedef struct ncclMultimemHandle ncclMultimemHandle_t;

struct ncclLsaBarrierHandle;
typedef struct ncclLsaBarrierHandle ncclLsaBarrierHandle_t;

struct ncclDevResourceRequirements;
typedef struct ncclDevResourceRequirements ncclDevResourceRequirements_t;

struct ncclTeamRequirements;
typedef struct ncclTeamRequirements ncclTeamRequirements_t;

struct ncclDevCommRequirements;
typedef struct ncclDevCommRequirements ncclDevCommRequirements_t;

ncclResult_t ncclDevCommCreate(
    ncclComm_t comm, ncclDevCommRequirements_t const* reqs, ncclDevComm_t* outDevComm
);
ncclResult_t ncclDevCommDestroy(ncclComm_t comm, ncclDevComm_t const* devComm);

#ifdef __cplusplus
}
#endif

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

struct ncclTeam
{
    int nRanks, rank, stride;
};

struct ncclTeamTagLsa
{};

struct ncclDevCommRequirements
{
    ncclDevResourceRequirements_t* resourceRequirementsList;
    ncclTeamRequirements_t*        teamRequirementsList;
    bool                           lsaMultimem;
    int                            lsaBarrierCount;
};

struct ncclDevResourceRequirements
{
    ncclDevResourceRequirements_t* next;
    size_t                         bufferSize, bufferAlign;
    ncclDevResourceHandle_t*       outBufferHandle;
};

struct ncclTeamRequirements
{
    ncclTeamRequirements_t* next;
    ncclTeam_t             team;
    bool                   multimem;
    ncclMultimemHandle_t*  outMultimemHandle;
};

struct ncclWindow_vidmem
{
    void*    winHost;
    char*    lsaFlatBase;
    int      lsaRank;
    int      worldRank;
    uint32_t stride4G;
    uint32_t mcOffset4K;
};

struct ncclMultimemHandle
{
    void* mcBasePtr;
};

struct ncclLsaBarrierHandle
{
    ncclDevResourceHandle_t bufHandle;
    int                     nBarriers;
};

struct ncclDevCommWindowTable
{
    struct Entry
    {
        uintptr_t     base, size;
        ncclWindow_t  window;
    } entries[32];

    struct ncclDevCommWindowTable* next;
};

struct ncclDevComm
{
    int      rank, nRanks;
    uint32_t nRanks_rcp32;
    int      lsaRank, lsaSize;
    uint32_t lsaSize_rcp32;

    struct ncclDevCommWindowTable* windowTable;

    ncclWindow_t            resourceWindow;
    struct ncclWindow_vidmem resourceWindow_inlined;

    ncclMultimemHandle_t   lsaMultimem;
    ncclLsaBarrierHandle_t lsaBarrier;
};

struct ncclCoopCta
{
    NCCL_DEVICE_INLINE int  thread_rank() const { return threadIdx.x; }
    NCCL_DEVICE_INLINE int  size() const { return blockDim.x; }
    NCCL_DEVICE_INLINE void sync() const { __syncthreads(); }
};

NCCL_HOST_DEVICE_INLINE ncclTeam ncclTeamLsa(ncclDevComm const& comm)
{
    ncclTeam ans;
    ans.nRanks = comm.lsaSize;
    ans.rank   = comm.lsaRank;
    ans.stride = 1;
    return ans;
}

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

NCCL_DEVICE_INLINE void* ncclGetLsaPointer(ncclWindow_t window, size_t offset, int peer)
{
    return static_cast<void*>(ncclAdd4G(window->lsaFlatBase, peer * window->stride4G) + offset);
}

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
