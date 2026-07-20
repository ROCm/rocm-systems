/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ANVIL_IPC_TABLE_H_
#define _NCCL_DEVICE_GIN_ANVIL_IPC_TABLE_H_

#include <stddef.h>
#include <stdint.h>

/** GIN-owned IPC lookup table for anvil-sdma peer VA resolution. */
#define NCCL_GIN_ANVIL_IPC_MAX_BUFS 16
#define NCCL_GIN_ANVIL_IPC_MAX_RANKS 16

struct ncclGinAnvilIpcBufEntry {
  uintptr_t local_base;
  uintptr_t remote_bases[NCCL_GIN_ANVIL_IPC_MAX_RANKS];
  size_t length;
};

struct ncclGinAnvilSdmaGPUContext;

#ifdef __cplusplus
extern "C" {
#endif

int ncclGinAnvilIpcTableRegisterVmm(void* localBase, size_t length, int myRank, int nRanks, ptrdiff_t strideBytes);
int ncclGinAnvilIpcTableRegisterExplicit(void* localBase, const uintptr_t* remoteBases, int nRanks, size_t length);
int ncclGinAnvilIpcTableUnregister(void* localBase);
void ncclGinAnvilIpcTableGetDevice(const ncclGinAnvilIpcBufEntry** outTable, int* outCount);
void ncclGinAnvilIpcTableTrackContext(struct ncclGinAnvilSdmaGPUContext* hostCtx,
                                      struct ncclGinAnvilSdmaGPUContext* devCtx);
void ncclGinAnvilIpcTableUntrackContext(struct ncclGinAnvilSdmaGPUContext* hostCtx);
void ncclGinAnvilIpcTableTestReset(void);

#ifdef __cplusplus
}
#endif

#endif  // _NCCL_DEVICE_GIN_ANVIL_IPC_TABLE_H_
