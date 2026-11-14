/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#include "gin/gin_anvil_sdma_factory.h"

#include "sdma/anvil.hpp"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <algorithm>

#include "debug.h"

struct gin_anvil_sdma_opaque {
  int nRanks;
  int numChannels;
  int myRank;
  int myDeviceId;
  int sdmaChannelStride;
  sdma_anvil::SdmaQueueDeviceHandle** deviceHandles_d;
  uint64_t* sdmaDirty_d;
};

extern "C" int gin_anvil_sdma_probe(void) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) return 0;
  return sdma_anvil::initEndpoint() ? 1 : 0;
}

static int checkHip(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    WARN("GIN anvil-sdma: %s: %s", what, hipGetErrorString(e));
    return -1;
  }
  return 0;
}

static int ginAnvilSpreadChannelsFromEnv() {
  const char* e = getenv("NCCL_GIN_ANVIL_SDMA_SPREAD_CHANNELS");
  if (!e || !e[0]) return 1;
  if (e[0] == '0' && e[1] == '\0') return 0;
  return atoi(e) != 0 ? 1 : 0;
}

// Names a GPU on another host for Anvil's remote-peer queues. A collision would
// only make two such peers share a queue.
static uint64_t ginAnvilRemotePeerKey(const gin_anvil_sdma_peer_t& p) {
  return p.host_id ^ ((static_cast<uint64_t>(p.device_id) + 1) * 0x9E3779B97F4A7C15ull);
}

extern "C" int gin_anvil_sdma_create(int nRanks, int myRank, int my_device_id, uint64_t my_host_id,
                                     int remote_hosts_ok,
                                     int (*allgather)(void* ctx, void* buf, size_t bytes_per_rank),
                                     void* allgather_ctx, int num_channels,
                                     gin_anvil_sdma_handle_t* out_handle, void** out_gpu_handles,
                                     uint64_t** out_sdma_dirty) {
  if (!out_handle || !out_gpu_handles || !out_sdma_dirty || !allgather || nRanks < 1 || myRank < 0 ||
      myRank >= nRanks)
    return -1;

  const int numChannels = std::max(1, std::min(8, num_channels));

  std::vector<gin_anvil_sdma_peer_t> peers(static_cast<size_t>(nRanks), gin_anvil_sdma_peer_t{-1, 0, 0});
  peers[static_cast<size_t>(myRank)] = gin_anvil_sdma_peer_t{my_device_id, 0, my_host_id};
  if (allgather(allgather_ctx, peers.data(), sizeof(gin_anvil_sdma_peer_t)) != 0) {
    WARN("GIN anvil-sdma: allgather(device ids) failed");
    return -1;
  }

  for (int i = 0; i < nRanks; ++i) {
    if (peers[static_cast<size_t>(i)].device_id < 0) {
      WARN("GIN anvil-sdma: invalid device id for rank %d", i);
      return -1;
    }
  }

  const int myDev = peers[static_cast<size_t>(myRank)].device_id;

  // A peer on another host is keyed by host and ordinal and gets no peer access.
  std::vector<char> sameHost(static_cast<size_t>(nRanks), 1);
  std::vector<uint64_t> remoteKeys(static_cast<size_t>(nRanks), 0);
  int remotePeers = 0;
  for (int i = 0; i < nRanks; ++i) {
    const gin_anvil_sdma_peer_t& p = peers[static_cast<size_t>(i)];
    const bool remote = my_host_id != 0 && p.host_id != 0 && p.host_id != my_host_id;
    if (remote && !remote_hosts_ok) {
      WARN("GIN anvil-sdma: rank %d is on another host, outside this GPU's SDMA reach", i);
      return -1;
    }
    sameHost[static_cast<size_t>(i)] = remote ? 0 : 1;
    if (remote) remoteKeys[static_cast<size_t>(i)] = ginAnvilRemotePeerKey(p);
    remotePeers += remote ? 1 : 0;
  }
  if (remotePeers > 0) {
    INFO(NCCL_INIT, "GIN anvil-sdma: %d of %d peers are on other hosts; each gets its own SDMA queue",
         remotePeers, nRanks);
  }

  if (!sdma_anvil::initEndpoint()) {
    WARN("GIN anvil-sdma: Anvil SDMA init failed");
    return -1;
  }

  // Queues live in the process-global singleton, so a failed create has to give back the ones it
  // took or they keep charging the budget. It must give back no more than that either: each
  // communicator gets its own ginState and the loop below deliberately reuses queues a previous
  // create left behind, whose device handles are already published to the GPU.
  std::vector<int> connectedDevs;
  std::vector<uint64_t> connectedRemotes;
  auto abandon = [&connectedDevs, &connectedRemotes]() {
    for (int dev : connectedDevs) sdma_anvil::anvil.disconnectDevice(dev);
    for (uint64_t key : connectedRemotes) sdma_anvil::anvil.disconnectRemotePeer(key);
    return -1;
  };

  try {
    for (int local_pe = 0; local_pe < nRanks; ++local_pe) {
      if (!sameHost[static_cast<size_t>(local_pe)]) {
        const uint64_t key = remoteKeys[static_cast<size_t>(local_pe)];
        if (sdma_anvil::anvil.getRemotePeerQueue(key, 0) != nullptr) continue;
        if (!sdma_anvil::anvil.connectRemotePeer(myDev, key, numChannels)) {
          WARN("GIN anvil-sdma: connect(%d -> rank %d on another host) failed", myDev, local_pe);
          return abandon();
        }
        connectedRemotes.push_back(key);
        continue;
      }
      const int remoteDev = peers[static_cast<size_t>(local_pe)].device_id;
      if (sdma_anvil::anvil.getSdmaQueue(myDev, remoteDev, 0) != nullptr) continue;
      if (myDev != remoteDev) sdma_anvil::EnablePeerAccess(myDev, remoteDev);
      if (!sdma_anvil::anvil.connect(myDev, remoteDev, numChannels)) {
        WARN("GIN anvil-sdma: connect(%d -> %d) failed", myDev, remoteDev);
        return abandon();
      }
      connectedDevs.push_back(remoteDev);
    }
  } catch (const std::exception& e) {
    WARN("GIN anvil-sdma: SDMA connect failed: %s", e.what());
    return abandon();
  }

  const int total = nRanks * numChannels;
  std::vector<sdma_anvil::SdmaQueueDeviceHandle*> host_handles(
      static_cast<size_t>(total), nullptr);
  int validHandles = 0;
  for (int local_pe = 0; local_pe < nRanks; ++local_pe) {
    const bool local = sameHost[static_cast<size_t>(local_pe)];
    const int remoteDev = peers[static_cast<size_t>(local_pe)].device_id;
    for (int c = 0; c < numChannels; ++c) {
      sdma_anvil::SdmaQueue* q =
          local ? sdma_anvil::anvil.getSdmaQueue(myDev, remoteDev, c)
                : sdma_anvil::anvil.getRemotePeerQueue(remoteKeys[static_cast<size_t>(local_pe)], c);
      auto* h = q ? q->deviceHandle() : nullptr;
      host_handles[static_cast<size_t>(local_pe * numChannels + c)] = h;
      if (h != nullptr) validHandles++;
    }
  }
  if (validHandles == 0) {
    WARN("GIN anvil-sdma: no SDMA queue handles for device %d", myDev);
    return abandon();
  }

  sdma_anvil::SdmaQueueDeviceHandle** dev_row = nullptr;
  if (checkHip(hipMalloc(&dev_row, static_cast<size_t>(total) * sizeof(void*)), "hipMalloc handles") != 0)
    return abandon();
  if (checkHip(hipMemcpy(dev_row, host_handles.data(), static_cast<size_t>(total) * sizeof(void*),
                         hipMemcpyHostToDevice),
               "hipMemcpy handles") != 0) {
    checkHip(hipFree(dev_row), "hipFree handles (cleanup)");
    return abandon();
  }

  uint64_t* dirty = nullptr;
  if (checkHip(hipExtMallocWithFlags((void**)&dirty, sizeof(uint64_t), hipDeviceMallocFinegrained),
               "hipExtMallocWithFlags sdmaDirty") != 0) {
    checkHip(hipFree(dev_row), "hipFree handles (cleanup)");
    return abandon();
  }
  if (checkHip(hipMemset(dirty, 0, sizeof(uint64_t)), "hipMemset sdmaDirty") != 0) {
    checkHip(hipFree(dev_row), "hipFree handles (cleanup)");
    checkHip(hipFree(dirty), "hipFree dirty (cleanup)");
    return abandon();
  }

  auto* impl = new gin_anvil_sdma_opaque{};
  impl->nRanks = nRanks;
  impl->numChannels = numChannels;
  impl->myRank = myRank;
  impl->myDeviceId = myDev;
  impl->sdmaChannelStride = ginAnvilSpreadChannelsFromEnv();
  impl->deviceHandles_d = dev_row;
  impl->sdmaDirty_d = dirty;

  *out_handle = impl;
  *out_gpu_handles = dev_row;
  *out_sdma_dirty = dirty;
  return 0;
}

extern "C" void gin_anvil_sdma_destroy(gin_anvil_sdma_handle_t handle) {
  if (!handle) return;
  auto* impl = handle;
  if (impl->deviceHandles_d) checkHip(hipFree(impl->deviceHandles_d), "hipFree handles (destroy)");
  if (impl->sdmaDirty_d) checkHip(hipFree(impl->sdmaDirty_d), "hipFree dirty (destroy)");
  delete impl;
}

extern "C" int gin_anvil_sdma_get_n_ranks(gin_anvil_sdma_handle_t handle) {
  return handle ? handle->nRanks : 0;
}

extern "C" int gin_anvil_sdma_get_num_channels(gin_anvil_sdma_handle_t handle) {
  return handle ? handle->numChannels : 0;
}

extern "C" int gin_anvil_sdma_get_channel_stride(gin_anvil_sdma_handle_t handle) {
  return handle ? handle->sdmaChannelStride : 0;
}

#endif // ENABLE_ROCSHMEM_GIN
