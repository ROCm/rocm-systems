/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#ifndef RCCL_CPU_DEV_COMM_MIRROR_H_
#define RCCL_CPU_DEV_COMM_MIRROR_H_

#include "comm.h"
#include "device.h"

ncclResult_t rcclCpuMirrorDevComm(struct ncclComm* comm, struct ncclKernelComm** outHostComm);
void rcclCpuReleaseCommMirror();
bool rcclCpuCommMirrorValid();
ncclResult_t rcclCpuWritebackChannelCounters(struct ncclComm* comm, int channelId, uint64_t workCounter);

#endif
