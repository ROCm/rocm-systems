/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#include "cpu_dev_comm_mirror.h"
#include "cpu_device_guard.h"

#include "checks.h"

#include <hip/hip_runtime.h>
#include <cstddef>

namespace {

static struct ncclKernelCommAndChannels* devCommBase(struct ncclComm* comm) {
  return reinterpret_cast<struct ncclKernelCommAndChannels*>(
      reinterpret_cast<char*>(comm->devComm) - offsetof(struct ncclKernelCommAndChannels, comm));
}

static char* devChannelField(struct ncclKernelCommAndChannels* devBase, int channelId, size_t fieldOffset) {
  return reinterpret_cast<char*>(devBase) + offsetof(struct ncclKernelCommAndChannels, channels) +
         static_cast<size_t>(channelId) * sizeof(struct ncclDevChannel) + fieldOffset;
}

}  // namespace

ncclResult_t rcclCpuMirrorDevComm(
    struct ncclComm* comm, struct rcclCpuCommMirrorState* mirror, struct ncclKernelComm** outHostComm) {
  if (comm == nullptr || comm->devComm == nullptr || mirror == nullptr) return ncclInvalidArgument;

  rcclCpuDeviceGuard guard(comm->cudaDev);
  struct ncclKernelCommAndChannels* devBase = devCommBase(comm);
  mirror->valid = false;
  mirror->peerStorage.clear();
  mirror->peerPtrStorage.clear();
  mirror->ringUserRanksStorage.clear();

  CUDACHECK(hipMemcpy(&mirror->hostBlob, devBase, sizeof(mirror->hostBlob), hipMemcpyDeviceToHost));

  int nRanks = comm->nRanks;
  mirror->peerStorage.resize(static_cast<size_t>(MAXCHANNELS) * nRanks);
  mirror->peerPtrStorage.resize(static_cast<size_t>(MAXCHANNELS) * nRanks, nullptr);

  for (int c = 0; c < MAXCHANNELS; c++) {
    size_t base = static_cast<size_t>(c) * nRanks;
    for (int r = 0; r < nRanks; r++) {
      mirror->peerPtrStorage[base + r] = &mirror->peerStorage[base + r];
    }
    mirror->hostBlob.channels[c].peers = &mirror->peerPtrStorage[base];

    struct ncclDevChannelPeer** devPeerList = nullptr;
    CUDACHECK(hipMemcpy(&devPeerList, devChannelField(devBase, c, offsetof(struct ncclDevChannel, peers)),
                        sizeof(devPeerList), hipMemcpyDeviceToHost));
    if (devPeerList == nullptr) continue;

    std::vector<struct ncclDevChannelPeer*> devPeerPtrs(nRanks);
    CUDACHECK(hipMemcpy(devPeerPtrs.data(), devPeerList, sizeof(struct ncclDevChannelPeer*) * nRanks,
                        hipMemcpyDeviceToHost));

    for (int r = 0; r < nRanks; r++) {
      if (devPeerPtrs[r] != nullptr) {
        size_t idx = base + r;
        CUDACHECK(hipMemcpy(&mirror->peerStorage[idx], devPeerPtrs[r], sizeof(struct ncclDevChannelPeer),
                            hipMemcpyDeviceToHost));
      }
    }

    int* devUserRanks = nullptr;
    CUDACHECK(hipMemcpy(&devUserRanks, devChannelField(devBase, c, offsetof(struct ncclDevChannel, ring.userRanks)),
                        sizeof(devUserRanks), hipMemcpyDeviceToHost));
    if (devUserRanks != nullptr) {
      size_t ringBase = mirror->ringUserRanksStorage.size();
      mirror->ringUserRanksStorage.resize(ringBase + static_cast<size_t>(nRanks));
      CUDACHECK(hipMemcpy(mirror->ringUserRanksStorage.data() + ringBase, devUserRanks,
                          sizeof(int) * nRanks, hipMemcpyDeviceToHost));
      mirror->hostBlob.channels[c].ring.userRanks = mirror->ringUserRanksStorage.data() + ringBase;
    } else {
      mirror->hostBlob.channels[c].ring.userRanks = nullptr;
    }
  }

  mirror->hostBlob.comm.channels = mirror->hostBlob.channels;
  mirror->hostBlob.comm.abortFlag = comm->abortFlag;
  mirror->valid = true;
  *outHostComm = &mirror->hostBlob.comm;
  return ncclSuccess;
}

bool rcclCpuMirrorChannelsReady(
    struct ncclComm* comm, struct rcclCpuCommMirrorState* mirror, struct ncclDevKernelArgs* args) {
  if (comm == nullptr || mirror == nullptr || !mirror->valid || args == nullptr) return false;

  auto peerConnReady = [&](int channelId, int peer, bool recv) -> bool {
    if (peer < 0 || peer >= comm->nRanks) return true;
    struct ncclDevChannel* ch = &mirror->hostBlob.channels[channelId];
    if (ch->peers == nullptr || ch->peers[peer] == nullptr) return false;
    struct ncclConnInfo* conn = recv ? &ch->peers[peer]->recv[0] : &ch->peers[peer]->send[0];
    if (conn->stepSize <= 0 || conn->buffs[NCCL_PROTO_SIMPLE] == nullptr) return false;
    if (recv && conn->head == nullptr) return false;
    if (!recv && conn->tail == nullptr) return false;
    if ((conn->flags & (NCCL_P2P_READ | NCCL_P2P_WRITE)) == 0) return false;
    return true;
  };

  for (int c = 0; c < MAXCHANNELS; c++) {
    if (!(args->channelMask.masks[c / 64] & (1ull << (c % 64)))) continue;
    int prev = mirror->hostBlob.channels[c].ring.prev;
    int next = mirror->hostBlob.channels[c].ring.next;
    if (!peerConnReady(c, prev, true)) return false;
    if (!peerConnReady(c, next, false)) return false;
    return true;
  }
  return false;
}

ncclResult_t rcclCpuWritebackChannelCounters(
    struct ncclComm* comm, struct rcclCpuCommMirrorState* mirror, int channelId, uint64_t workCounter) {
  if (mirror == nullptr || !mirror->valid) return ncclInternalError;
  mirror->hostBlob.channels[channelId].workCounter = workCounter;

  rcclCpuDeviceGuard guard(comm->cudaDev);
  struct ncclKernelCommAndChannels* devBase = devCommBase(comm);
  CUDACHECK(hipMemcpy(devChannelField(devBase, channelId, offsetof(struct ncclDevChannel, workCounter)),
                      &workCounter, sizeof(workCounter), hipMemcpyHostToDevice));
  return ncclSuccess;
}
