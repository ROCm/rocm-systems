/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ce_sdma_sync.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <vector>

#include "ce_sdma_queue.h"
#include "comm.h"
#include "debug.h"
#include "dev_runtime.h"
#include "utils.h"

struct ncclCeSdmaState {
  int lsaSelf;
  std::vector<struct ncclCeSdmaHostQueue*> queues;  // by LSA index; the own index is the control queue
  uint64_t* release;                                // signal memory polled by the stream's CP wait
  uint64_t seq = 0;                                 // all-gathers launched so far
};

struct ncclCeSdmaPlan {
  struct ncclComm* comm;
  uint64_t n;                             // call number, 1-based
  std::vector<std::vector<uint32_t>> dw;  // packets per queue
};

namespace {
constexpr uint32_t kOpCopy = 1, kOpWrite = 2, kOpPollRegmem = 8, kOpAtomic = 10;
constexpr uint32_t kAtomicAdd64 = 47, kPollGeq = 5, kPollInterval = 10, kPollRetryForever = 0xFFF;
constexpr size_t kMaxCopyBytes = 1ull << 30;  // COPY_LINEAR count is 30 bits

uint32_t lo(uint64_t v) {
  return (uint32_t)v;
}
uint32_t hi(uint64_t v) {
  return (uint32_t)(v >> 32);
}
ncclCeSdmaState* stateOf(const ncclCeSdmaPlan* p) {
  return p->comm->ceColl.sdma;
}
int lsaSizeOf(const ncclCeSdmaPlan* p) {
  return (int)stateOf(p)->queues.size();
}

void emit(ncclCeSdmaPlan* p, int queue, std::initializer_list<uint32_t> words) {
  p->dw[queue].insert(p->dw[queue].end(), words);
}

void pollGeq(ncclCeSdmaPlan* p, int queue, const void* counter, uint64_t expected) {
  const uint64_t a = (uintptr_t)counter;
  emit(p, queue,
       {kOpPollRegmem | (kPollGeq << 28) | (1u << 31), lo(a), hi(a), lo(expected), 0xFFFFFFFFu,
        kPollInterval | (kPollRetryForever << 16)});
}

void copy(ncclCeSdmaPlan* p, int queue, void* dst, const void* src, size_t bytes) {
  for (size_t n; bytes > 0; bytes -= n) {
    n = std::min(bytes, kMaxCopyBytes);
    const uint64_t s = (uintptr_t)src, d = (uintptr_t)dst;
    emit(p, queue, {kOpCopy, (uint32_t)(n - 1), 0, lo(s), hi(s), lo(d), hi(d)});
    src = (const uint8_t*)src + n;
    dst = (uint8_t*)dst + n;
  }
}

void add64(ncclCeSdmaPlan* p, int queue, void* dst, uint64_t delta) {
  const uint64_t a = (uintptr_t)dst;
  emit(p, queue, {kOpAtomic | (kAtomicAdd64 << 25), lo(a), hi(a), lo(delta), hi(delta), 0, 0, 0});
}

// Gate counters follow the flag arrays in the CE sync window: ready[lsaSize], then arrival.
uint64_t* myGate(ncclComm* comm, size_t index) {
  return (uint64_t*)((uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.sdmaGateOffset) + index;
}

ncclResult_t peerGate(ncclComm* comm, size_t index, int lsaPeer, uint64_t** out) {
  void* p;
  NCCLCHECK(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, comm->ceColl.sdmaGateOffset + index * sizeof(uint64_t),
                                  lsaPeer, &p));
  *out = (uint64_t*)p;
  return ncclSuccess;
}

// Runs when the stream reaches the collective: writes the packets and rings the doorbells.
void armChains(void* arg) {
  std::unique_ptr<ncclCeSdmaPlan> p((ncclCeSdmaPlan*)arg);
  ncclCeSdmaState* st = stateOf(p.get());
  for (size_t q = 0; q < st->queues.size(); q++) {
    if (!p->dw[q].empty() &&
        ncclCeSdmaHostQueueSubmit(st->queues[q], p->dw[q].data(), p->dw[q].size()) != ncclSuccess) {
      // Release the stream to avoid a hang; the output is incomplete.
      WARN("CE SDMA sync: could not arm call %llu", (unsigned long long)p->n);
      *(volatile uint64_t*)st->release = p->n;
      return;
    }
  }
}
}  // namespace

size_t ncclCeSdmaGateBytes(int lsaSize) {
  return (size_t)(lsaSize + 1) * sizeof(uint64_t);
}

ncclResult_t ncclCeSdmaInit(struct ncclComm* comm) {
  const char* cpWait = getenv("GPU_STREAMOPS_CP_WAIT");
  if (cpWait == nullptr || cpWait[0] == '\0' || !strcmp(cpWait, "0")) {
    WARN("CE SDMA sync: GPU_STREAMOPS_CP_WAIT=1 is required");
    return ncclInvalidUsage;
  }
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, comm->cudaDev) != hipSuccess ||
      (strncmp(prop.gcnArchName, "gfx90a", 6) && strncmp(prop.gcnArchName, "gfx942", 6) &&
       strncmp(prop.gcnArchName, "gfx950", 6))) {
    WARN("CE SDMA sync: unsupported GPU architecture");
    return ncclInvalidUsage;
  }
  auto st = std::make_unique<ncclCeSdmaState>();
  st->lsaSelf = comm->devrState.lsaSelf;
  for (int i = 0; i < comm->devrState.lsaSize; i++) {
    // Device ordinal of the LSA rank in this process, matched by PCI bus id.
    char busId[64];
    int dev;
    if (int64ToBusId(comm->peerInfo[comm->devrState.lsaRankList[i]].busId, busId) != ncclSuccess ||
        hipDeviceGetByPCIBusId(&dev, busId) != hipSuccess) {
      WARN("CE SDMA sync: LSA rank %d is not visible to this process", i);
      return ncclInvalidUsage;
    }
    st->queues.emplace_back();
    NCCLCHECK(ncclCeSdmaHostQueueGet(comm->cudaDev, dev, &st->queues.back()));
  }
  if (hipExtMallocWithFlags((void**)&st->release, sizeof(uint64_t), hipMallocSignalMemory) != hipSuccess) {
    WARN("CE SDMA sync: signal memory allocation failed");
    return ncclUnhandledCudaError;
  }
  *(volatile uint64_t*)st->release = 0;
  INFO(NCCL_INIT, "CE SDMA sync: enabled for rank %d (%zu LSA queues)", comm->rank, st->queues.size());
  comm->ceColl.sdma = st.release();
  return ncclSuccess;
}

void ncclCeSdmaDestroy(struct ncclComm* comm) {
  std::unique_ptr<ncclCeSdmaState> st(comm->ceColl.sdma);
  if (!st) return;
  comm->ceColl.sdma = nullptr;
  // Queued chains and the CP wait still reference the gate window and the release word.
  (void)hipSetDevice(comm->cudaDev);
  (void)hipDeviceSynchronize();
  (void)hipFree(st->release);
}

bool ncclCeSdmaActive(struct ncclComm* comm) {
  const ncclCeSdmaState* st = comm->ceColl.sdma;
  // The next call's arrival target must fit the 32-bit compare; every rank sees the same count.
  return st != nullptr && (st->seq + 1) * 2 * (st->queues.size() - 1) <= 0xFFFFFFFFull;
}

ncclResult_t ncclCeSdmaPlanNew(struct ncclComm* comm, struct ncclCeSdmaPlan** out) {
  const ncclCeSdmaState* st = comm->ceColl.sdma;
  *out = new ncclCeSdmaPlan{comm, st->seq + 1, std::vector<std::vector<uint32_t>>(st->queues.size())};
  return ncclSuccess;
}

void ncclCeSdmaPlanFree(struct ncclCeSdmaPlan* plan) {
  delete plan;
}

void ncclCeSdmaPlanWait(struct ncclCeSdmaPlan* plan, const void* counter64, uint64_t expected, bool fanOut) {
  const int self = stateOf(plan)->lsaSelf;
  if (!fanOut) return pollGeq(plan, self, counter64, expected);
  for (int q = 0; q < lsaSizeOf(plan); q++)
    if (q != self) pollGeq(plan, q, counter64, expected);
}

void ncclCeSdmaPlanWrite(struct ncclCeSdmaPlan* plan, void* dst, uint64_t value) {
  const uint64_t d = (uintptr_t)dst;
  emit(plan, stateOf(plan)->lsaSelf, {kOpWrite, lo(d), hi(d), 1, lo(value), hi(value)});
}

ncclResult_t ncclCeSdmaAnnounceReady(struct ncclCeSdmaPlan* plan) {
  const int self = stateOf(plan)->lsaSelf, size = lsaSizeOf(plan);
  for (int r = 1; r < size; r++) {
    const int peer = (self + r) % size;
    uint64_t* gate;
    NCCLCHECK(peerGate(plan->comm, self, peer, &gate));
    add64(plan, peer, gate, 1);
  }
  return ncclSuccess;
}

ncclResult_t ncclCeSdmaBroadcast(struct ncclCeSdmaPlan* plan, struct ncclDevrWindow* recvWin, size_t offset,
                                 const void* src, size_t bytes) {
  const int self = stateOf(plan)->lsaSelf, size = lsaSizeOf(plan);
  void* own = (uint8_t*)recvWin->userPtr + offset;
  if (own != src) copy(plan, self, own, src, bytes);
  for (int r = 1; r < size; r++) {
    const int peer = (self + r) % size;
    void* dst;
    NCCLCHECK(ncclDevrGetLsaRankPtr(plan->comm, recvWin, offset, peer, &dst));
    pollGeq(plan, peer, myGate(plan->comm, peer), plan->n);
    copy(plan, peer, dst, src, bytes);
  }
  return ncclSuccess;
}

ncclResult_t ncclCeSdmaScatter(struct ncclCeSdmaPlan* plan, struct ncclDevrWindow* recvWin, size_t offset,
                               const void* src, size_t bytes) {
  const int self = stateOf(plan)->lsaSelf, size = lsaSizeOf(plan);
  for (int r = 1; r < size; r++) {
    const int peer = (self + r) % size;
    void* dst;
    NCCLCHECK(ncclDevrGetLsaRankPtr(plan->comm, recvWin, offset, peer, &dst));
    copy(plan, peer, dst, src, bytes);
  }
  return ncclSuccess;
}

ncclResult_t ncclCeSdmaFinish(struct ncclCeSdmaPlan* plan, hipStream_t stream) {
  std::unique_ptr<ncclCeSdmaPlan> p(plan);
  ncclComm* comm = p->comm;
  ncclCeSdmaState* st = stateOf(p.get());
  const int self = st->lsaSelf, size = lsaSizeOf(p.get());
  // After its last copy, each peer queue adds 1 to the peer's arrival counter and 1 to ours.
  uint64_t* myArrival = myGate(comm, size);
  for (int r = 1; r < size; r++) {
    const int peer = (self + r) % size;
    uint64_t* peerArrival;
    NCCLCHECK(peerGate(comm, size, peer, &peerArrival));
    add64(p.get(), peer, peerArrival, 1);
    add64(p.get(), peer, myArrival, 1);
  }
  pollGeq(p.get(), self, myArrival, p->n * 2 * (size - 1));
  add64(p.get(), self, st->release, 1);

  const uint64_t n = p->n;
  if (hipLaunchHostFunc(stream, armChains, p.get()) != hipSuccess) {
    WARN("CE SDMA sync: hipLaunchHostFunc failed");
    return ncclUnhandledCudaError;
  }
  p.release();  // freed by armChains
  st->seq++;
  if (hipStreamWaitValue64(stream, st->release, n, hipStreamWaitValueGte, ~0ull) != hipSuccess) {
    WARN("CE SDMA sync: hipStreamWaitValue64 failed; the communicator must be aborted");
    return ncclUnhandledCudaError;
  }
  return ncclSuccess;
}
