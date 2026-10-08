/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Clear-to-send (CTS) telemetry for the IB transport.
//
// Enabled with RCCL_IB_CTS_TELEMETRY=1. When disabled, ncclIbNetCommBase::ctsTel
// is NULL and every hook below is a single predictable branch.
//
// Each metric keeps its latest RCCL_IB_CTS_TELEMETRY_SAMPLES values in host
// memory. When the comm is closed, statistics are computed and printed as one
// INFO line (NCCL_DEBUG_SUBSYS=NET) per connection, i.e. per channel and peer.
// count/mean/min/max cover every value; percentiles cover the kept values.
//
// Sender metrics (local monotonic clock):
//   ctsWait   first ncclIbIsend() attempt that found the CTS slot empty ->
//             the poll that found it filled (data ready, waiting for receiver)
//   react     CTS seen -> RDMA writes posted
//   sendCmpl  RDMA writes posted -> last send completion
// Receiver metrics:
//   ctsRtt    CTS RDMA write posted -> its completion (receiver NIC, wire,
//             sender NIC DMA into host memory, ACK). Sampled on signaled CTS.
//   firstData CTS posted -> first data completion
//   recvDone  CTS posted -> receive request complete
//
// A WARN is printed when a CTS or a receive is outstanding for more than
// RCCL_IB_CTS_TELEMETRY_STALL_MS.

#ifndef NET_IB_CTS_TELEMETRY_H_
#define NET_IB_CTS_TELEMETRY_H_

#include <stdint.h>
#include <time.h>
#include "nccl.h"

#define NCCL_IB_CTS_MAX_SLOTS 256
// Set in wr_id of CTS writes signaled only for telemetry. Their completions
// are consumed before request lookup: recvReqs[slot] may be stale by then.
#define NCCL_IB_CTS_TEL_WR_ID (1ull << 62)

enum ncclIbCtsMetric {
  NCCL_IB_CTS_SEND_WAIT = 0,
  NCCL_IB_CTS_SEND_REACT = 1,
  NCCL_IB_CTS_SEND_CMPL = 2,
  NCCL_IB_CTS_RECV_RTT = 0,
  NCCL_IB_CTS_RECV_FIRST_DATA = 1,
  NCCL_IB_CTS_RECV_DONE = 2,
  NCCL_IB_CTS_NMETRICS = 3,
};

struct ncclIbCtsSamples {
  uint64_t count;
  uint64_t sumNs;
  uint32_t minNs;
  uint32_t maxNs;
  uint32_t* ns; // ring of the latest ncclIbCtsTelemetry::cap values
};

struct ncclIbCtsTelemetry {
  bool isSend;
  int channelId; // -1 until ncclIbCtsTelemetrySetConn()
  int rank;
  int peerRank;
  uint64_t cap;
  struct ncclIbCtsSamples m[NCCL_IB_CTS_NMETRICS];
  uint64_t nCts;          // CTS posted (recv) / CTS consumed (send)
  uint64_t nSampled;      // signaled CTS completions measured (recv)
  uint64_t nReadyOnFirst; // CTS already present on first ncclIbIsend() (send)
  uint64_t nStalls;
  uint64_t bytes;

  // Sender state
  uint64_t waitIdx;
  uint64_t waitStartNs;
  bool waitStallWarned;
  uint64_t seenIdx[NCCL_IB_CTS_MAX_SLOTS];
  uint64_t seenNs[NCCL_IB_CTS_MAX_SLOTS];
  uint64_t postNs[NCCL_IB_CTS_MAX_SLOTS];

  // Receiver state
  uint64_t ctsPostNs[NCCL_IB_CTS_MAX_SLOTS];
  uint64_t ctsId[NCCL_IB_CTS_MAX_SLOTS];
  uint8_t ctsSignaled[NCCL_IB_CTS_MAX_SLOTS];
  uint8_t firstDataSeen[NCCL_IB_CTS_MAX_SLOTS];
  uint8_t recvStallWarned[NCCL_IB_CTS_MAX_SLOTS];

  uint64_t stallNs;
  int signalEvery;
};

struct ncclIbNetCommBase;
struct ncclIbRequest;

ncclResult_t ncclIbCtsTelemetryInit(struct ncclIbNetCommBase* base);
// Prints the statistics line, then frees.
ncclResult_t ncclIbCtsTelemetryClose(struct ncclIbNetCommBase* base);
// Frees without printing (comm setup failure paths).
void ncclIbCtsTelemetryFree(struct ncclIbNetCommBase* base);

void ncclIbCtsStallWarnSend(struct ncclIbNetCommBase* base, uint64_t idx, uint64_t slotIdx, uint64_t waitedNs);
void ncclIbCtsStallWarnRecv(struct ncclIbNetCommBase* base, struct ncclIbRequest* req, uint64_t waitedNs);

static inline uint64_t ncclIbCtsNowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static inline void ncclIbCtsRecord(struct ncclIbCtsTelemetry* t, int metric, uint64_t ns) {
  struct ncclIbCtsSamples* s = &t->m[metric];
  uint32_t v = ns >= UINT32_MAX ? UINT32_MAX : (uint32_t)ns;
  if (s->count == 0 || v < s->minNs) s->minNs = v;
  if (v > s->maxNs) s->maxNs = v;
  s->sumNs += ns;
  if (s->ns) s->ns[s->count % t->cap] = v;
  s->count++;
}

// ---------------------------------------------------------------------------
// Sender hooks (ncclIbIsend / ncclIbRequestComplete)
// ---------------------------------------------------------------------------

// The CTS for fifo index idx is not there yet.
static inline void ncclIbCtsSendNotReady(struct ncclIbNetCommBase* base, struct ncclIbCtsTelemetry* t, uint64_t idx,
                                         uint64_t slotIdx) {
  uint64_t now = ncclIbCtsNowNs();
  if (t->waitIdx != idx) {
    t->waitIdx = idx;
    t->waitStartNs = now;
    t->waitStallWarned = false;
    return;
  }
  if (t->stallNs && !t->waitStallWarned && now - t->waitStartNs > t->stallNs) {
    t->waitStallWarned = true;
    t->nStalls++;
    ncclIbCtsStallWarnSend(base, idx, slotIdx, now - t->waitStartNs);
  }
}

// The CTS for fifo index idx was found in slot.
static inline void ncclIbCtsSendSeen(struct ncclIbCtsTelemetry* t, uint64_t idx, int slot) {
  if (t->seenIdx[slot] == idx) return; // multi-recv: one CTS row, several isend calls
  uint64_t now = ncclIbCtsNowNs();
  t->seenIdx[slot] = idx;
  t->seenNs[slot] = now;
  t->nCts++;
  uint64_t wait = 0;
  if (t->waitIdx == idx) {
    wait = now - t->waitStartNs;
  } else {
    t->nReadyOnFirst++;
  }
  ncclIbCtsRecord(t, NCCL_IB_CTS_SEND_WAIT, wait);
}

static inline void ncclIbCtsSendPosted(struct ncclIbCtsTelemetry* t, int slot, uint64_t bytes) {
  uint64_t now = ncclIbCtsNowNs();
  t->postNs[slot] = now;
  t->bytes += bytes;
  ncclIbCtsRecord(t, NCCL_IB_CTS_SEND_REACT, now - t->seenNs[slot]);
}

static inline void ncclIbCtsSendDone(struct ncclIbCtsTelemetry* t, int slot) {
  ncclIbCtsRecord(t, NCCL_IB_CTS_SEND_CMPL, ncclIbCtsNowNs() - t->postNs[slot]);
}

// ---------------------------------------------------------------------------
// Receiver hooks (ncclIbPostFifo / ncclIbCompletionEventProcess / ncclIbTest)
// ---------------------------------------------------------------------------

// Called right before the CTS is posted. Returns true if this CTS should be
// signaled so its round trip can be measured.
static inline bool ncclIbCtsRecvPost(struct ncclIbCtsTelemetry* t, uint64_t id, int slot, bool signaled) {
  bool sample = signaled || (t->signalEvery > 0 && (id % t->signalEvery) == 0);
  t->ctsPostNs[slot] = ncclIbCtsNowNs();
  t->ctsId[slot] = id;
  t->ctsSignaled[slot] = sample;
  t->firstDataSeen[slot] = 0;
  t->recvStallWarned[slot] = 0;
  t->nCts++;
  return sample;
}

static inline void ncclIbCtsRecvCtsCompletion(struct ncclIbCtsTelemetry* t, uint64_t wrId) {
  wrId &= ~NCCL_IB_CTS_TEL_WR_ID;
  if (wrId >= NCCL_IB_CTS_MAX_SLOTS || !t->ctsSignaled[wrId]) return;
  int slot = (int)wrId;
  t->ctsSignaled[slot] = 0;
  t->nSampled++;
  ncclIbCtsRecord(t, NCCL_IB_CTS_RECV_RTT, ncclIbCtsNowNs() - t->ctsPostNs[slot]);
}

static inline void ncclIbCtsRecvData(struct ncclIbCtsTelemetry* t, uint64_t id) {
  int slot = (int)(id % NCCL_IB_CTS_MAX_SLOTS);
  if (t->ctsId[slot] != id || t->firstDataSeen[slot]) return;
  t->firstDataSeen[slot] = 1;
  ncclIbCtsRecord(t, NCCL_IB_CTS_RECV_FIRST_DATA, ncclIbCtsNowNs() - t->ctsPostNs[slot]);
}

static inline void ncclIbCtsRecvDone(struct ncclIbCtsTelemetry* t, uint64_t id, uint64_t bytes) {
  int slot = (int)(id % NCCL_IB_CTS_MAX_SLOTS);
  if (t->ctsId[slot] != id) return;
  ncclIbCtsRecord(t, NCCL_IB_CTS_RECV_DONE, ncclIbCtsNowNs() - t->ctsPostNs[slot]);
  t->bytes += bytes;
}

static inline void ncclIbCtsRecvCheckStall(struct ncclIbNetCommBase* base, struct ncclIbCtsTelemetry* t,
                                           struct ncclIbRequest* req, uint64_t id) {
  if (t->stallNs == 0) return;
  int slot = (int)(id % NCCL_IB_CTS_MAX_SLOTS);
  if (t->ctsId[slot] != id || t->recvStallWarned[slot]) return;
  uint64_t waited = ncclIbCtsNowNs() - t->ctsPostNs[slot];
  if (waited > t->stallNs) {
    t->recvStallWarned[slot] = 1;
    t->nStalls++;
    ncclIbCtsStallWarnRecv(base, req, waited);
  }
}

#endif
