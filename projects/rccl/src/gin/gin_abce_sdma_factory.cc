/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef ENABLE_ROCSHMEM_GIN

#include "gin/gin_abce_sdma_factory.h"

#include <hip/hip_runtime.h>
#include <hip/hip_ext_copy_engine.h>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <algorithm>

#include "debug.h"

// ABCE copy-engine backend. Unlike anvil (a process-global SdmaQueue singleton
// with connect()/getSdmaQueue()), ABCE hands out per-(device,src,dst) device
// handles via hipExtCopyEngineGetDeviceHandles(). A handle is a 16-byte value
// struct passed to kernels by value, not a device pointer; we still store a
// device-memory row of them indexed [local_pe * numChannels + ch] so the device
// code indexes exactly as the anvil backend did.
struct gin_abce_sdma_opaque {
  int nRanks;
  int numChannels;
  int myRank;
  int myDeviceId;
  int sdmaChannelStride;
  hipExtCopyEngineDeviceHandle* deviceHandles_d;  // [nRanks*numChannels] by value
  uint64_t* sdmaDirty_d;
};

extern "C" int gin_abce_sdma_probe(void) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) return 0;
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) return 0;
  // A self-copy handle on the current device is the cheapest liveness check:
  // succeeds only when the kernel-side copy engine is on and supported.
  hipExtCopyEngineDeviceHandle h{};
  int n = 0;
  if (hipExtCopyEngineGetDeviceHandles(dev, dev, dev, 1, &h, &n) != hipSuccess || n < 1) return 0;
  return 1;
}

static int checkHip(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    WARN("GIN abce-sdma: %s: %s", what, hipGetErrorString(e));
    return -1;
  }
  return 0;
}

static int ginAbceSpreadChannelsFromEnv() {
  const char* e = getenv("NCCL_GIN_ABCE_SDMA_SPREAD_CHANNELS");
  if (!e || !e[0]) return 1;
  if (e[0] == '0' && e[1] == '\0') return 0;
  return atoi(e) != 0 ? 1 : 0;
}

extern "C" int gin_abce_sdma_create(int nRanks, int myRank, int my_device_id,
                                     int (*allgather)(void* ctx, void* buf, size_t bytes_per_rank),
                                     void* allgather_ctx, int num_channels,
                                     gin_abce_sdma_handle_t* out_handle, void** out_gpu_handles,
                                     uint64_t** out_sdma_dirty) {
  if (!out_handle || !out_gpu_handles || !out_sdma_dirty || !allgather || nRanks < 1 || myRank < 0 ||
      myRank >= nRanks)
    return -1;

  const int numChannels = std::max(1, std::min(8, num_channels));

  std::vector<int> devs(static_cast<size_t>(nRanks), -1);
  devs[static_cast<size_t>(myRank)] = my_device_id;
  if (allgather(allgather_ctx, devs.data(), sizeof(int)) != 0) {
    WARN("GIN abce-sdma: allgather(device ids) failed");
    return -1;
  }

  for (int i = 0; i < nRanks; ++i) {
    if (devs[static_cast<size_t>(i)] < 0) {
      WARN("GIN abce-sdma: invalid device id for rank %d", i);
      return -1;
    }
  }

  const int myDev = devs[static_cast<size_t>(myRank)];

  // One copy-engine handle set per peer. ABCE ranks engines for the (src,dst)
  // pair and returns up to numChannels handles; we fan them across channels,
  // repeating the last if ABCE offered fewer than requested.
  const int total = nRanks * numChannels;
  std::vector<hipExtCopyEngineDeviceHandle> host_handles(static_cast<size_t>(total));
  std::memset(host_handles.data(), 0, host_handles.size() * sizeof(hipExtCopyEngineDeviceHandle));
  int validHandles = 0;
  for (int local_pe = 0; local_pe < nRanks; ++local_pe) {
    const int remoteDev = devs[static_cast<size_t>(local_pe)];
    if (myDev != remoteDev) {
      // ABCE reaches the peer's memory over xGMI; HIP peer access must be on.
      int canAccess = 0;
      if (hipDeviceCanAccessPeer(&canAccess, myDev, remoteDev) == hipSuccess && canAccess) {
        int prev = 0;
        if (hipGetDevice(&prev) == hipSuccess) {
          if (hipSetDevice(myDev) == hipSuccess) {
            hipError_t pe = hipDeviceEnablePeerAccess(remoteDev, 0);
            if (pe != hipSuccess && pe != hipErrorPeerAccessAlreadyEnabled)
              WARN("GIN abce-sdma: enable peer access %d->%d: %s", myDev, remoteDev,
                   hipGetErrorString(pe));
            hipSetDevice(prev);
          }
        }
      }
    }
    std::vector<hipExtCopyEngineDeviceHandle> handles(static_cast<size_t>(numChannels));
    int n = 0;
    hipError_t e =
        hipExtCopyEngineGetDeviceHandles(myDev, myDev, remoteDev, numChannels, handles.data(), &n);
    if (e != hipSuccess || n < 1) {
      WARN("GIN abce-sdma: GetDeviceHandles(%d -> %d) failed (%s, n=%d)", myDev, remoteDev,
           hipGetErrorString(e), n);
      return -1;
    }
    for (int c = 0; c < numChannels; ++c) {
      host_handles[static_cast<size_t>(local_pe * numChannels + c)] =
          handles[static_cast<size_t>(c < n ? c : n - 1)];
      ++validHandles;
    }
  }
  if (validHandles == 0) {
    WARN("GIN abce-sdma: no copy-engine handles for device %d", myDev);
    return -1;
  }

  hipExtCopyEngineDeviceHandle* dev_row = nullptr;
  if (checkHip(hipMalloc(&dev_row, static_cast<size_t>(total) * sizeof(hipExtCopyEngineDeviceHandle)),
               "hipMalloc handles") != 0)
    return -1;
  if (checkHip(hipMemcpy(dev_row, host_handles.data(),
                         static_cast<size_t>(total) * sizeof(hipExtCopyEngineDeviceHandle),
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

  auto* impl = new gin_abce_sdma_opaque{};
  impl->nRanks = nRanks;
  impl->numChannels = numChannels;
  impl->myRank = myRank;
  impl->myDeviceId = myDev;
  impl->sdmaChannelStride = ginAbceSpreadChannelsFromEnv();
  impl->deviceHandles_d = dev_row;
  impl->sdmaDirty_d = dirty;

  *out_handle = impl;
  *out_gpu_handles = dev_row;
  *out_sdma_dirty = dirty;
  return 0;
}

extern "C" void gin_abce_sdma_destroy(gin_abce_sdma_handle_t handle) {
  if (!handle) return;
  auto* impl = handle;
  if (impl->deviceHandles_d) checkHip(hipFree(impl->deviceHandles_d), "hipFree handles (destroy)");
  if (impl->sdmaDirty_d) checkHip(hipFree(impl->sdmaDirty_d), "hipFree dirty (destroy)");
  delete impl;
}

extern "C" int gin_abce_sdma_get_n_ranks(gin_abce_sdma_handle_t handle) {
  return handle ? handle->nRanks : 0;
}

extern "C" int gin_abce_sdma_get_num_channels(gin_abce_sdma_handle_t handle) {
  return handle ? handle->numChannels : 0;
}

extern "C" int gin_abce_sdma_get_channel_stride(gin_abce_sdma_handle_t handle) {
  return handle ? handle->sdmaChannelStride : 0;
}

#endif // ENABLE_ROCSHMEM_GIN
