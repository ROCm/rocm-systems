/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

/*
 * IR-only device ABI declarations - moved from src/include/nccl_device/core.h
 *
 * These are IR-specific declarations that don't belong in the public headers
 * because they're only needed by the LLVM bitcode generation, not by regular
 * RCCL compilation.
 */

#ifndef _NCCL_IR_IMPL_CORE_H_
#define _NCCL_IR_IMPL_CORE_H_

/* Team API */
extern "C" __device__ ncclTeam ncclTeamWorld(ncclDevComm const&);
extern "C" __device__ ncclTeam ncclTeamLsa(ncclDevComm const&);
extern "C" __device__ ncclTeam ncclTeamCft(ncclDevComm const&, ncclCftTeamMode_t mode = NCCL_CFT_TEAM_FLAT);
extern "C" __device__ ncclTeam ncclTeamCftMultimem(ncclDevComm const&);
extern "C" __device__ int ncclTeamRankToWorld(ncclDevComm const&, ncclTeam, int rank);
extern "C" __device__ int ncclTeamRankToLsa(ncclDevComm const&, ncclTeam, int rank);
extern "C" __device__ ncclTeam ncclTeamRail(ncclDevComm const&);

/* Window API - pointer queries */
extern "C" __device__ void* ncclGetLocalPointer(ncclWindow_t w, size_t offset);
extern "C" __device__ void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int peer);
extern "C" __device__ void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer);
extern "C" __device__ void* ncclGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle mmHandle);
extern "C" __device__ void* ncclGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const&);

/* CFT handle based query functions */
extern "C" __device__ void ncclGetCftLeInfo(ncclWindow_t w, size_t offset, int peerCft, ncclTeam cftTeam,
                                            ncclDevComm const& comm, ncclCftLeId* leId, size_t* leOffset);
extern "C" __device__ void ncclGetPeerLeInfo(
    ncclWindow_t w, size_t offset, int peerWorld, ncclDevComm const& comm, ncclCftLeId* leId, size_t* leOffset);
extern "C" __device__ void ncclGetMultimemLeInfo(ncclWindow_t w, size_t offset, ncclDevComm const&,
                                                  ncclCftLeId* leId, size_t* leOffset);

/* Resource buffer accessors */
extern "C" __device__ void* ncclGetResourceBufferLocalPointer(ncclDevComm const&, ncclDevResourceHandle);
extern "C" __device__ void* ncclGetResourceBufferLsaPointer(ncclDevComm const&, ncclDevResourceHandle, int peer);
extern "C" __device__ void* ncclGetResourceBufferPeerPointer(ncclDevComm const&, ncclDevResourceHandle,
                                                              ncclTeam, int peer);
extern "C" __device__ void* ncclGetResourceBufferMultimemPointer(
    ncclDevComm const&, ncclDevResourceHandle, ncclMultimemHandle);
extern "C" __device__ void* ncclGetResourceBufferLsaMultimemPointer(ncclDevComm const&,
                                                                     ncclDevResourceHandle);

/* Resource buffer CFT LE info */
extern "C" __device__ void ncclGetResourceBufferCftLeInfo(
    ncclDevComm const&, ncclDevResourceHandle, int peerCft, ncclCftLeId* leId, size_t* leOffset);
extern "C" __device__ void ncclGetResourceBufferPeerLeInfo(
    ncclDevComm const&, ncclDevResourceHandle, int peerWorld, ncclCftLeId* leId, size_t* leOffset);
extern "C" __device__ void ncclGetResourceBufferMultimemLeInfo(ncclDevComm const&, ncclDevResourceHandle,
                                                                ncclCftLeId* leId, size_t* leOffset);

/* ncclGetPeerPointerTeam wrapper (uses ncclTeam directly) */
extern "C" __device__ void* ncclGetPeerPointerTeam(ncclWindow_t w, size_t offset, ncclTeam tm, int peer);

#endif // _NCCL_IR_IMPL_CORE_H_
