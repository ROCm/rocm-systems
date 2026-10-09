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

extern "C" int gin_anvil_sdma_create(int nRanks, int myRank, int my_device_id,
                                     int (*allgather)(void* ctx, void* buf, size_t bytes_per_rank),
                                     void* allgather_ctx, int num_channels,
                                     gin_anvil_sdma_handle_t* out_handle, void** out_gpu_handles,
                                     uint64_t** out_sdma_dirty) {
  if (!out_handle || !out_gpu_handles || !out_sdma_dirty || !allgather || nRanks < 1 || myRank < 0 ||
      myRank >= nRanks)
    return -1;

  const int numChannels = std::max(1, std::min(8, num_channels));

  std::vector<int> devs(static_cast<size_t>(nRanks), -1);
  devs[static_cast<size_t>(myRank)] = my_device_id;
  if (allgather(allgather_ctx, devs.data(), sizeof(int)) != 0) {
    WARN("GIN anvil-sdma: allgather(device ids) failed");
    return -1;
  }

  for (int i = 0; i < nRanks; ++i) {
    if (devs[static_cast<size_t>(i)] < 0) {
      WARN("GIN anvil-sdma: invalid device id for rank %d", i);
      return -1;
    }
  }

  const int myDev = devs[static_cast<size_t>(myRank)];

  if (!sdma_anvil::initEndpoint()) {
    WARN("GIN anvil-sdma: Anvil SDMA init failed");
    return -1;
  }

  try {
    for (int local_pe = 0; local_pe < nRanks; ++local_pe) {
      const int remoteDev = devs[static_cast<size_t>(local_pe)];
      if (sdma_anvil::anvil.getSdmaQueue(myDev, remoteDev, 0) != nullptr) continue;
      if (myDev != remoteDev) sdma_anvil::EnablePeerAccess(myDev, remoteDev);
      if (!sdma_anvil::anvil.connect(myDev, remoteDev, numChannels)) {
        WARN("GIN anvil-sdma: connect(%d -> %d) failed", myDev, remoteDev);
        return -1;
      }
    }
  } catch (const std::exception& e) {
    WARN("GIN anvil-sdma: SDMA connect failed: %s", e.what());
    return -1;
  }

  const int total = nRanks * numChannels;
  std::vector<sdma_anvil::SdmaQueueDeviceHandle*> host_handles(
      static_cast<size_t>(total), nullptr);
  int validHandles = 0;
  for (int local_pe = 0; local_pe < nRanks; ++local_pe) {
    const int remoteDev = devs[static_cast<size_t>(local_pe)];
    for (int c = 0; c < numChannels; ++c) {
      sdma_anvil::SdmaQueue* q = sdma_anvil::anvil.getSdmaQueue(myDev, remoteDev, c);
      auto* h = q ? q->deviceHandle() : nullptr;
      host_handles[static_cast<size_t>(local_pe * numChannels + c)] = h;
      if (h != nullptr) validHandles++;
    }
  }
  if (validHandles == 0) {
    WARN("GIN anvil-sdma: no SDMA queue handles for device %d", myDev);
    return -1;
  }

  sdma_anvil::SdmaQueueDeviceHandle** dev_row = nullptr;
  if (checkHip(hipMalloc(&dev_row, static_cast<size_t>(total) * sizeof(void*)), "hipMalloc handles") != 0)
    return -1;
  if (checkHip(hipMemcpy(dev_row, host_handles.data(), static_cast<size_t>(total) * sizeof(void*),
                         hipMemcpyHostToDevice),
               "hipMemcpy handles") != 0) {
    checkHip(hipFree(dev_row), "hipFree handles (cleanup)");
    return -1;
  }

  uint64_t* dirty = nullptr;
  if (checkHip(hipExtMallocWithFlags((void**)&dirty, sizeof(uint64_t), hipDeviceMallocFinegrained),
               "hipExtMallocWithFlags sdmaDirty") != 0) {
    checkHip(hipFree(dev_row), "hipFree handles (cleanup)");
    return -1;
  }
  if (checkHip(hipMemset(dirty, 0, sizeof(uint64_t)), "hipMemset sdmaDirty") != 0) {
    checkHip(hipFree(dev_row), "hipFree handles (cleanup)");
    checkHip(hipFree(dirty), "hipFree dirty (cleanup)");
    return -1;
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
