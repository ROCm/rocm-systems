/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ABCE_IPC_TABLE_H_
#define _NCCL_DEVICE_GIN_ABCE_IPC_TABLE_H_

#include <stddef.h>
#include <stdint.h>

/** GIN-owned IPC lookup table for abce-sdma peer VA resolution. */
#define NCCL_GIN_ABCE_IPC_MAX_BUFS 16
#define NCCL_GIN_ABCE_IPC_MAX_RANKS 16

struct ncclGinAbceIpcBufEntry {
  uintptr_t local_base;
  uintptr_t remote_bases[NCCL_GIN_ABCE_IPC_MAX_RANKS];
  size_t length;
};

struct ncclGinAbceSdmaGPUContext;

#ifdef __cplusplus
extern "C" {
#endif

int ncclGinAbceIpcTableRegisterVmm(void* localBase, size_t length, int myRank, int nRanks, ptrdiff_t strideBytes);
int ncclGinAbceIpcTableRegisterExplicit(void* localBase, const uintptr_t* remoteBases, int nRanks, size_t length);
int ncclGinAbceIpcTableUnregister(void* localBase);
void ncclGinAbceIpcTableGetDevice(const ncclGinAbceIpcBufEntry** outTable, int* outCount);
void ncclGinAbceIpcTableTrackContext(struct ncclGinAbceSdmaGPUContext* hostCtx,
                                      struct ncclGinAbceSdmaGPUContext* devCtx);
void ncclGinAbceIpcTableUntrackContext(struct ncclGinAbceSdmaGPUContext* hostCtx);
void ncclGinAbceIpcTableTestReset(void);

#ifdef __cplusplus
}
#endif

#endif  // _NCCL_DEVICE_GIN_ABCE_IPC_TABLE_H_
