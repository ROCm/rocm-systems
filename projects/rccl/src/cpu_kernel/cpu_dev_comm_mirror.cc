/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#include "cpu_dev_comm_mirror.h"

#include "checks.h"

#include <hip/hip_runtime.h>
#include <cstddef>
#include <vector>

namespace {

struct rcclCpuCommMirror {
  struct ncclKernelCommAndChannels hostBlob;
  std::vector<struct ncclDevChannelPeer> peerStorage;
  std::vector<struct ncclDevChannelPeer*> peerPtrStorage;
  std::vector<struct ncclDevChannelPeer**> channelPeerLists;
  bool valid = false;
};

static thread_local rcclCpuCommMirror t_mirror;

static struct ncclKernelCommAndChannels* devCommBase(struct ncclComm* comm) {
  return reinterpret_cast<struct ncclKernelCommAndChannels*>(
      reinterpret_cast<char*>(comm->devComm) - offsetof(struct ncclKernelCommAndChannels, comm));
}

}  // namespace

ncclResult_t rcclCpuMirrorDevComm(struct ncclComm* comm, struct ncclKernelComm** outHostComm) {
  if (comm == nullptr || comm->devComm == nullptr) return ncclInvalidArgument;

  struct ncclKernelCommAndChannels* devBase = devCommBase(comm);
  t_mirror.valid = false;
  t_mirror.peerStorage.clear();
  t_mirror.peerPtrStorage.clear();
  t_mirror.channelPeerLists.clear();

  CUDACHECK(hipMemcpy(&t_mirror.hostBlob, devBase, sizeof(t_mirror.hostBlob), hipMemcpyDeviceToHost));

  struct ncclDevChannel* devChannels = devBase->channels;
  int nChannels = MAXCHANNELS;
  int nRanks = comm->nRanks;

  t_mirror.peerStorage.resize(static_cast<size_t>(nChannels) * nRanks);
  t_mirror.peerPtrStorage.resize(static_cast<size_t>(nChannels) * nRanks);
  t_mirror.channelPeerLists.resize(nChannels, nullptr);

  for (int c = 0; c < nChannels; c++) {
    struct ncclDevChannelPeer** devPeerList = nullptr;
    CUDACHECK(hipMemcpy(&devPeerList, &devChannels[c].peers, sizeof(devPeerList), hipMemcpyDeviceToHost));
    if (devPeerList == nullptr) continue;

    std::vector<struct ncclDevChannelPeer*> devPeerPtrs(nRanks);
    CUDACHECK(hipMemcpy(devPeerPtrs.data(), devPeerList, sizeof(struct ncclDevChannelPeer*) * nRanks,
                        hipMemcpyDeviceToHost));

    for (int r = 0; r < nRanks; r++) {
      size_t idx = static_cast<size_t>(c) * nRanks + r;
      t_mirror.peerPtrStorage[idx] = &t_mirror.peerStorage[idx];
      if (devPeerPtrs[r] != nullptr) {
        CUDACHECK(hipMemcpy(&t_mirror.peerStorage[idx], devPeerPtrs[r], sizeof(struct ncclDevChannelPeer),
                            hipMemcpyDeviceToHost));
      }
    }
    t_mirror.channelPeerLists[c] = &t_mirror.peerPtrStorage[static_cast<size_t>(c) * nRanks];
    t_mirror.hostBlob.channels[c].peers = t_mirror.channelPeerLists[c];
  }

  t_mirror.hostBlob.comm.channels = t_mirror.hostBlob.channels;
  t_mirror.valid = true;
  *outHostComm = &t_mirror.hostBlob.comm;
  return ncclSuccess;
}

void rcclCpuReleaseCommMirror() {
  t_mirror.valid = false;
}

bool rcclCpuCommMirrorValid() {
  return t_mirror.valid;
}

ncclResult_t rcclCpuWritebackChannelCounters(struct ncclComm* comm, int channelId, uint64_t workCounter) {
  if (!t_mirror.valid) return ncclInternalError;
  t_mirror.hostBlob.channels[channelId].workCounter = workCounter;

  struct ncclKernelCommAndChannels* devBase = devCommBase(comm);
  CUDACHECK(hipMemcpy(&devBase->channels[channelId].workCounter, &workCounter, sizeof(workCounter),
                      hipMemcpyHostToDevice));
  return ncclSuccess;
}
