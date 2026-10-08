/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cts_telemetry.h"
#include "common.h"

#include <algorithm>

RCCL_PARAM(IbCtsTelemetry, "IB_CTS_TELEMETRY", 0);
// Additionally signal every Nth CTS so its round trip can be timed (0 = only
// the CTS that are signaled anyway to drain the send queue).
RCCL_PARAM(IbCtsTelemetrySignalEvery, "IB_CTS_TELEMETRY_SIGNAL_EVERY", 16);
RCCL_PARAM(IbCtsTelemetryStallMs, "IB_CTS_TELEMETRY_STALL_MS", 10000);
// Values kept per metric per connection for percentiles (4 bytes each).
RCCL_PARAM(IbCtsTelemetrySamples, "IB_CTS_TELEMETRY_SAMPLES", 65536);

static_assert(NET_IB_MAX_REQUESTS == NCCL_IB_CTS_MAX_SLOTS, "CTS telemetry slots must match NET_IB_MAX_REQUESTS");

static const char* sendMetricNames[NCCL_IB_CTS_NMETRICS] = {"ctsWait", "react", "sendCmpl"};
static const char* recvMetricNames[NCCL_IB_CTS_NMETRICS] = {"ctsRtt", "firstData", "recvDone"};

ncclResult_t ncclIbCtsTelemetryInit(struct ncclIbNetCommBase* base) {
  base->ctsTel = NULL;
  if (rcclParamIbCtsTelemetry() <= 0) return ncclSuccess;
  struct ncclIbCtsTelemetry* t;
  NCCLCHECK(ncclCalloc(&t, 1));
  t->isSend = base->isSend;
  t->channelId = t->rank = t->peerRank = -1;
  t->signalEvery = (int)rcclParamIbCtsTelemetrySignalEvery();
  t->stallNs = (uint64_t)rcclParamIbCtsTelemetryStallMs() * 1000000ull;
  t->cap = std::max<int64_t>(rcclParamIbCtsTelemetrySamples(), 0);
  t->waitIdx = UINT64_MAX;
  for (int s = 0; s < NCCL_IB_CTS_MAX_SLOTS; s++) {
    t->seenIdx[s] = UINT64_MAX;
    t->ctsId[s] = UINT64_MAX;
  }
  for (int m = 0; m < NCCL_IB_CTS_NMETRICS && t->cap; m++) {
    if (ncclCalloc(&t->m[m].ns, t->cap) != ncclSuccess) {
      WARN("NET/IB: CTS telemetry: cannot keep %lu samples per metric, reporting count/mean/min/max only", t->cap);
      for (int i = 0; i < m; i++) {
        free(t->m[i].ns);
        t->m[i].ns = NULL;
      }
      t->cap = 0;
    }
  }
  base->ctsTel = t;
  return ncclSuccess;
}

void ncclIbCtsTelemetrySetConn(void* netComm, int channelId, int rank, int peerRank) {
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)netComm;
  if (base == NULL || base->ctsTel == NULL || base->ctsTel->channelId >= 0) return;
  base->ctsTel->channelId = channelId;
  base->ctsTel->rank = rank;
  base->ctsTel->peerRank = peerRank;
}

void ncclIbCtsTelemetryFree(struct ncclIbNetCommBase* base) {
  if (base->ctsTel == NULL) return;
  for (int m = 0; m < NCCL_IB_CTS_NMETRICS; m++) free(base->ctsTel->m[m].ns);
  free(base->ctsTel);
  base->ctsTel = NULL;
}

static const char* ctsDevName(struct ncclIbNetCommBase* base) {
  struct ncclIbNetCommDevBase* devBase = ncclIbGetNetCommDevBase(base, 0);
  return (devBase && devBase->pd) ? devBase->pd->context->device->name : "?";
}

// Appends " name[n= mean= p50= p90= p99= p999= max=]" in microseconds. Sorts the kept samples.
static int ctsFormatMetric(char* buf, size_t len, const char* name, struct ncclIbCtsSamples* s, uint64_t cap) {
  if (s->count == 0) return snprintf(buf, len, " %s[n=0]", name);
  uint64_t kept = std::min(s->count, cap);
  double p[4] = {0, 0, 0, 0};
  static const double q[4] = {0.50, 0.90, 0.99, 0.999};
  if (s->ns && kept) {
    std::sort(s->ns, s->ns + kept);
    for (int i = 0; i < 4; i++) p[i] = s->ns[std::min<uint64_t>(kept - 1, (uint64_t)(q[i] * kept))] / 1e3;
  }
  return snprintf(buf, len, " %s[n=%lu mean=%.2f min=%.2f p50=%.2f p90=%.2f p99=%.2f p999=%.2f max=%.2f]", name,
                  s->count, (double)s->sumNs / s->count / 1e3, s->minNs / 1e3, p[0], p[1], p[2], p[3],
                  s->maxNs / 1e3);
}

ncclResult_t ncclIbCtsTelemetryClose(struct ncclIbNetCommBase* base) {
  struct ncclIbCtsTelemetry* t = base->ctsTel;
  if (t == NULL) return ncclSuccess;
  if (t->nCts > 0) {
    const char** names = t->isSend ? sendMetricNames : recvMetricNames;
    char line[1024];
    int o = snprintf(line, sizeof(line), "ch %d %s rank %d %s %d dev %s nqps %d cts %lu bytes %lu ", t->channelId,
                     t->isSend ? "send" : "recv", t->rank, t->isSend ? "->" : "<-", t->peerRank, ctsDevName(base),
                     base->nqps, t->nCts, t->bytes);
    if (t->isSend) {
      o += snprintf(line + o, sizeof(line) - o, "readyOnFirstPoll %lu", t->nReadyOnFirst);
    } else {
      o += snprintf(line + o, sizeof(line) - o, "sampled %lu", t->nSampled);
    }
    o += snprintf(line + o, sizeof(line) - o, " stalls %lu samplesKept %lu (us):", t->nStalls, t->cap);
    for (int m = 0; m < NCCL_IB_CTS_NMETRICS && o < (int)sizeof(line); m++) {
      o += ctsFormatMetric(line + o, sizeof(line) - o, names[m], &t->m[m], t->cap);
    }
    INFO(NCCL_NET, "NET/IB: CTS %s", line);
  }
  ncclIbCtsTelemetryFree(base);
  return ncclSuccess;
}

void ncclIbCtsStallWarnSend(struct ncclIbNetCommBase* base, uint64_t idx, uint64_t slotIdx, uint64_t waitedNs) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)base;
  struct ncclIbCtsTelemetry* t = base->ctsTel;
  int slot = (int)((idx - 1) % NET_IB_MAX_REQUESTS);
  WARN("NET/IB: CTS stall (sender): ch %d rank %d -> %d waiting %.3f s for CTS idx %lu in slot %d, slot holds idx "
       "%lu; fifoHead %lu, sendReqsCnt[slot] %d, cts consumed %lu, dev %s nqps %d",
       t->channelId, t->rank, t->peerRank, waitedNs / 1e9, idx, slot, slotIdx, base->fifoHead,
       comm->sendReqsCnt[slot], t->nCts, ctsDevName(base), base->nqps);
}

void ncclIbCtsStallWarnRecv(struct ncclIbNetCommBase* base, struct ncclIbRequest* req, uint64_t waitedNs) {
  struct ncclIbCtsTelemetry* t = base->ctsTel;
  int slot = (int)(req->id % NET_IB_MAX_REQUESTS);
  WARN("NET/IB: CTS stall (receiver): ch %d rank %d <- %d request id %lu slot %d not complete %.3f s after its CTS "
       "was posted; CTS completion %s, first data %s, events {%d,%d,%d,%d}, fifoHead %lu, cts posted %lu, dev %s "
       "nqps %d",
       t->channelId, t->rank, t->peerRank, req->id, slot, waitedNs / 1e9,
       t->ctsSignaled[slot] ? "pending" : "seen or not sampled", t->firstDataSeen[slot] ? "received" : "not received",
       req->events[0], req->events[1], req->events[2], req->events[3], base->fifoHead, t->nCts, ctsDevName(base),
       base->nqps);
}
