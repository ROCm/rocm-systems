/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#ifndef RCCL_CPU_DEV_COMM_MIRROR_H_
#define RCCL_CPU_DEV_COMM_MIRROR_H_

#include "comm.h"
#include "device.h"

#include <vector>

struct rcclCpuCommMirrorState {
  struct ncclKernelCommAndChannels hostBlob;
  std::vector<struct ncclDevChannelPeer> peerStorage;
  std::vector<struct ncclDevChannelPeer*> peerPtrStorage;
  std::vector<int> ringUserRanksStorage;
  bool valid = false;
};

ncclResult_t rcclCpuMirrorDevComm(
    struct ncclComm* comm, struct rcclCpuCommMirrorState* mirror, struct ncclKernelComm** outHostComm);
ncclResult_t rcclCpuWritebackChannelCounters(
    struct ncclComm* comm, struct rcclCpuCommMirrorState* mirror, int channelId, uint64_t workCounter);

// Returns true when mirrored P2P channels have usable SIMPLE connections for ring peers.
bool rcclCpuMirrorChannelsReady(
    struct ncclComm* comm, struct rcclCpuCommMirrorState* mirror, struct ncclDevKernelArgs* args);

#endif
