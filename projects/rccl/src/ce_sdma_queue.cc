/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ce_sdma_queue.h"

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsakmt/hsakmt.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "debug.h"

struct ncclCeSdmaHostQueue {
  uint32_t* ring;
  volatile uint64_t* rptr;
  volatile uint64_t* wptr;
  volatile uint64_t* doorbell;
  uint64_t cachedWptr;  // guarded by lock
  std::mutex lock;
  int srcDev, dstDev;
};

namespace {
constexpr uint64_t kRingBytes = 1 << 20;
constexpr uint32_t kOpNop = 0;
constexpr int64_t kSubmitTimeoutUs = 10 * 1000 * 1000;

std::mutex gLock;  // guards gQueues and gAgents
std::vector<ncclCeSdmaHostQueue*> gQueues;  // never freed
std::vector<hsa_agent_t> gAgents;

hsa_status_t collectGpuAgents(hsa_agent_t agent, void* data) {
  hsa_device_type_t type;
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU)
    static_cast<std::vector<hsa_agent_t>*>(data)->push_back(agent);
  return HSA_STATUS_SUCCESS;
}

// HSA agent of a HIP device, matched by PCI domain:bus:device.function.
bool agentOf(int hipDev, hsa_agent_t* out) {
  if (gAgents.empty()) hsa_iterate_agents(collectGpuAgents, &gAgents);
  char id[32] = {};
  unsigned dom, bus, dev, fn;
  if (hipDeviceGetPCIBusId(id, sizeof(id), hipDev) != hipSuccess ||
      sscanf(id, "%x:%x:%x.%x", &dom, &bus, &dev, &fn) != 4)
    return false;
  for (const hsa_agent_t& a : gAgents) {
    uint32_t d = 0, bdf = 0;
    if (hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_DOMAIN, &d) == HSA_STATUS_SUCCESS &&
        hsa_agent_get_info(a, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_BDFID, &bdf) == HSA_STATUS_SUCCESS && d == dom &&
        ((bdf >> 8) & 0xff) == bus && ((bdf >> 3) & 0x1f) == dev && (bdf & 7) == fn) {
      *out = a;
      return true;
    }
  }
  return false;
}
// The queue is created through the KFD thunk (libhsakmt), which is linked statically.
bool openKfd() {
  static const bool ok = [] {
    HsaSystemProperties props = {};
    return hsa_init() == HSA_STATUS_SUCCESS && hsaKmtOpenKFD() == HSAKMT_STATUS_SUCCESS &&
           hsaKmtAcquireSystemProperties(&props) == HSAKMT_STATUS_SUCCESS;
  }();
  return ok;
}

bool enablePeerAccess(int srcDev, int dstDev) {
  int cur = 0;
  if (hipGetDevice(&cur) != hipSuccess || hipSetDevice(srcDev) != hipSuccess) return false;
  const hipError_t err = hipDeviceEnablePeerAccess(dstDev, 0);
  (void)hipSetDevice(cur);
  return err == hipSuccess || err == hipErrorPeerAccessAlreadyEnabled;
}
}  // namespace

ncclResult_t ncclCeSdmaHostQueueGet(int srcDev, int dstDev, struct ncclCeSdmaHostQueue** out) {
  std::lock_guard<std::mutex> g(gLock);
  for (auto* q : gQueues)
    if (q->srcDev == srcDev && q->dstDev == dstDev) {
      *out = q;
      return ncclSuccess;
    }
  if (!openKfd()) {
    WARN("CE SDMA: could not open the KFD");
    return ncclSystemError;
  }
  hsa_agent_t srcAgent, dstAgent;
  uint32_t engineMask = 1;  // a queue to the own device: any engine
  uint32_t node = 0;
  bool ok = agentOf(srcDev, &srcAgent) && agentOf(dstDev, &dstAgent) &&
            hsa_agent_get_info(srcAgent, HSA_AGENT_INFO_NODE, &node) == HSA_STATUS_SUCCESS;
  if (ok && srcDev != dstDev)
    ok = hsa_amd_memory_get_preferred_copy_engine(dstAgent, srcAgent, &engineMask) == HSA_STATUS_SUCCESS &&
         engineMask != 0;
  if (!ok || (srcDev != dstDev && !enablePeerAccess(srcDev, dstDev))) {
    WARN("CE SDMA: no HSA agent, SDMA engine or peer access for devices %d -> %d", srcDev, dstDev);
    return ncclSystemError;
  }

  // The ring is device memory the CPU can write and the SDMA engine executes.
  HsaMemFlags flags = {};
  flags.ui32.NonPaged = 1;
  flags.ui32.HostAccess = 1;
  flags.ui32.PageSize = HSA_PAGE_SIZE_4KB;
  flags.ui32.NoNUMABind = 1;
  flags.ui32.ExecuteAccess = 1;
  flags.ui32.Uncached = 1;
  void* ring = nullptr;
  HsaQueueResource res = {};
  if (hsaKmtAllocMemory(node, kRingBytes, flags, &ring) != HSAKMT_STATUS_SUCCESS) {
    WARN("CE SDMA: could not allocate the ring for devices %d -> %d", srcDev, dstDev);
    return ncclSystemError;
  }
  const bool mapped = hsaKmtMapMemoryToGPU(ring, kRingBytes, nullptr) == HSAKMT_STATUS_SUCCESS;
  if (!mapped || hsaKmtCreateQueueExt(node, HSA_QUEUE_SDMA_BY_ENG_ID, 100, HSA_QUEUE_PRIORITY_NORMAL,
                                      (uint32_t)__builtin_ctz(engineMask), ring, kRingBytes, nullptr,
                                      &res) != HSAKMT_STATUS_SUCCESS) {
    if (mapped) hsaKmtUnmapMemoryToGPU(ring);
    hsaKmtFreeMemory(ring, kRingBytes);
    WARN("CE SDMA: could not create the queue for devices %d -> %d", srcDev, dstDev);
    return ncclSystemError;
  }

  auto* q = new ncclCeSdmaHostQueue();
  q->ring = static_cast<uint32_t*>(ring);
  q->rptr = reinterpret_cast<volatile uint64_t*>(res.Queue_read_ptr_aql);
  q->wptr = reinterpret_cast<volatile uint64_t*>(res.Queue_write_ptr_aql);
  q->doorbell = reinterpret_cast<volatile uint64_t*>(res.Queue_DoorBell_aql);
  q->cachedWptr = *q->wptr;
  q->srcDev = srcDev;
  q->dstDev = dstDev;
  gQueues.push_back(q);
  *out = q;
  return ncclSuccess;
}

ncclResult_t ncclCeSdmaHostQueueSubmit(struct ncclCeSdmaHostQueue* q, const uint32_t* dwords, size_t nDwords) {
  const uint64_t bytes = nDwords * sizeof(uint32_t);
  if (bytes >= kRingBytes / 2) {
    WARN("CE SDMA: submission of %llu bytes does not fit the ring", (unsigned long long)bytes);
    return ncclInvalidArgument;
  }
  std::lock_guard<std::mutex> guard(q->lock);

  uint64_t cur = q->cachedWptr;
  const uint64_t off = cur % kRingBytes;
  const uint64_t pad = off + bytes > kRingBytes ? kRingBytes - off : 0;  // a packet never wraps: pad with NOPs

  const auto t0 = std::chrono::steady_clock::now();
  while (cur + pad + bytes - *q->rptr >= kRingBytes) {
    if (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count() >
        kSubmitTimeoutUs) {
      WARN("CE SDMA: ring full for 10 s");
      return ncclSystemError;
    }
    std::this_thread::yield();
  }

  if (pad) {
    std::fill_n(q->ring + off / sizeof(uint32_t), pad / sizeof(uint32_t), kOpNop);
    cur += pad;
  }
  std::memcpy(q->ring + (cur % kRingBytes) / sizeof(uint32_t), dwords, bytes);

  cur += bytes;
  std::atomic_thread_fence(std::memory_order_release);
  *q->wptr = cur;
  std::atomic_thread_fence(std::memory_order_release);
  *q->doorbell = cur;
  q->cachedWptr = cur;
  return ncclSuccess;
}
