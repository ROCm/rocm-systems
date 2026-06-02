/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Per-block CPU execution context mirroring device ncclShmem state.
 ************************************************************************/

#ifndef RCCL_CPU_DEVICE_CONTEXT_H_
#define RCCL_CPU_DEVICE_CONTEXT_H_

#include "device.h"
#include "cpu_memory_model.h"

#include <pthread.h>
#include <stdint.h>

struct rcclCpuBlockContext {
  struct ncclDevKernelArgs args;
  struct ncclKernelComm* comm;
  struct ncclDevChannel* channel;
  int channelId;
  int blockId;
  int threadCount;
  int warpSize;
  int cudaDev;
  volatile uint32_t* hostAbortFlag;
  unsigned funcId;
  int workType;
  int workSize;
  int nWorks;
  int nextBatchIx;
  char workStorage[4096];
  uint64_t groupBarriers[NCCL_MAX_GROUPS];
  uint64_t barrierPat;
  int aborted;
};

struct rcclCpuBlockBarrier {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int arrived;
  int generation;
  int expected;
};

void rcclCpuBlockBarrierInit(struct rcclCpuBlockBarrier* b);
void rcclCpuBlockBarrierDestroy(struct rcclCpuBlockBarrier* b);
void rcclCpuBlockBarrierWait(struct rcclCpuBlockBarrier* b, int tid, int tn);

void rcclCpuBlockContextInit(struct rcclCpuBlockContext* ctx, int warpSize);
int rcclCpuMapBlockToChannel(struct ncclDevKernelArgs const* args, int blockId, int warpSize);

#endif  // RCCL_CPU_DEVICE_CONTEXT_H_
