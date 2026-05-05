#include "common_cast.h"
extern int64_t ncclParamIbCastQpsPerConn();


struct ncclIbQpSchedParms castGlobalQpSchedParms;
pthread_mutex_t ncclIbQpSchedParmsLock = PTHREAD_MUTEX_INITIALIZER;
struct ncclIbQpSchedParmsCB stagedSchedParms { ncclNumFuncs };
ncclFunc_t IbCastQpSchedProxyPrevCollType = ncclNumFuncs;
size_t IbCastQpSchedProxyPrevMsgSz;

extern int64_t ncclParamIbCastSplitDataOnQps();

RCCL_PARAM_NCCL_ALIAS(IbCastQpSchedEnable, "IB_QP_SCHED_ENABLE", QP_SCHED_ENABLE_DEF);
RCCL_PARAM_NCCL_ALIAS(IbQpSchedWrrEnable, "IB_QP_SCHED_WRR_ENABLE", QP_SCHED_WRR_ENABLE_DEF);
RCCL_PARAM_NCCL_ALIAS(IbQpSchedResetInterval, "IB_QP_SCHED_RESET_INTERVAL", -1);
RCCL_PARAM_NCCL_ALIAS(IbQpSchedUpdateInterval, "IB_QP_SCHED_UPDATE_INTERVAL", -1);
RCCL_PARAM_NCCL_ALIAS(IbQpSchedSplitDataMin, "IB_QP_SCHED_SPLIT_DATA_MIN", -1);
RCCL_PARAM_NCCL_ALIAS(IbQpSchedLogInterval, "IB_QP_SCHED_LOG_INTERVAL", -1);

FILE *IbCastQpSchedLogStream;

ncclResult_t IbCastQpSchedInitParms(struct ncclIbQpSchedParms *parms) {
    char *str, *logFileName = NULL;
    int val;
    double weight;
    uint64_t nsec;
    ncclResult_t ret = ncclSuccess;
  
    parms->enable = QP_SCHED_ENABLE_DEF;
    parms->wrrEnable = QP_SCHED_WRR_ENABLE_DEF;
    parms->resetInterval = QP_SCHED_RESET_DEF;
    parms->updateInterval = QP_SCHED_UPDATE_DEF;
    parms->weightNew = QP_SCHED_WEIGHT_DEF;
    parms->splitData = ncclParamIbCastSplitDataOnQps();
    parms->splitDataMin = QP_SCHED_SPLIT_DATA_MIN_DEF;
    parms->logInterval = QP_SCHED_LOG_DEF;
    parms->logEnable = false;
    if (!parms->enable)
      parms->doWrr = false;
    else if (!parms->splitData)
      parms->doWrr = true;
    else if (parms->wrrEnable)
      parms->doWrr = true;
    else
      parms->doWrr = false;
  
    if (rcclParamIbCastQpSchedEnable()) {
      parms->enable = true;
      INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_ENABLE set to enabled");
    } else {
      parms->enable = false;
      INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_ENABLE set to disabled");
      goto exit;
    }
  
    if (parms->splitData) {
      if (rcclParamIbQpSchedWrrEnable()) {
        parms->doWrr = true;
        INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_WRR_ENABLE set to enabled");
      }
    } else
      parms->doWrr = true;
  
    val = (int)rcclParamIbQpSchedResetInterval();
    if (val >= 0) {
      nsec = (val * NSEC_PER_MSEC);
      if (nsec > 0 && nsec < QP_SCHED_RESET_MIN)
        goto getUpdateParm;
      parms->resetInterval = nsec;
      INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_RESET_INTERVAL set to %lu nsec", nsec);
    }
  
  getUpdateParm:
    val = (int)rcclParamIbQpSchedUpdateInterval();
    if (val >= 0) {
      nsec = (val * NSEC_PER_USEC);
      if ((nsec >= QP_SCHED_UPDATE_MIN) && (nsec <= QP_SCHED_UPDATE_MAX)) {
        parms->updateInterval = nsec;
        INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_UPDATE_INTERVAL set to %lu nsec", nsec);
      }
    }
  
    str = getenv(QP_SCHED_WEIGHT_ENV_VAR);
    if (!str)
      str = getenv(QP_SCHED_WEIGHT_ENV_VAR_ALIAS);
    if (str) {
      weight = atof(str);
      if (weight != QP_SCHED_WEIGHT_NONE) {
        if (weight <= QP_SCHED_WEIGHT_MAX && weight >= QP_SCHED_WEIGHT_MIN) {
          parms->weightNew = weight;
          INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_WEIGHT set to %f", weight);
        }
      }
    }
  
    val = (int)rcclParamIbQpSchedSplitDataMin();
    if (val > 0) {
      parms->splitDataMin = val;
      INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_SPLIT_DATA_MIN set to %d bytes", val);
    }
  
    str = getenv(QP_SCHED_LOG_PATH_ENV_VAR);
    if (!str)
      str = getenv(QP_SCHED_LOG_PATH_ENV_VAR_ALIAS);
    if (str != NULL) {
      char hostName[HOST_NAME_MAX + 1], pid[32];
      size_t fileNameLen;
  
      gethostname(hostName, HOST_NAME_MAX);
      snprintf(pid, sizeof(pid), "%d", getpid());
      fileNameLen = strlen(str) + 1 + strlen(QP_SCHED_LOG_FILE_NAME_PREFIX) +
                strlen(hostName) + 1 + strlen(pid);
      logFileName = (char *) calloc(1, fileNameLen + 1);
      if (logFileName == NULL) {
        WARN("(IB-CAST) NCCL_IB_QP_SCHED_LOG_PATH: calloc failed");
        goto err_exit;
      }
      snprintf(logFileName, fileNameLen + 1, "%s/%s%s_%s", str, QP_SCHED_LOG_FILE_NAME_PREFIX, hostName, pid);
      IbCastQpSchedLogStream = fopen(logFileName, "w");
      if (IbCastQpSchedLogStream == NULL) {
        WARN("(IB-CAST) NCCL_IB_QP_SCHED_LOG_PATH: fopen failed: %s (logging disabled)", strerror(errno));
        parms->logEnable = false;
        goto exit;
      }
      INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) NCCL_IB_QP_SCHED_LOG_PATH: opened %s", logFileName);
      parms->logEnable = true;
  
      val = (int)rcclParamIbQpSchedLogInterval();
      if (val >= 0) {
        nsec = (val * NSEC_PER_USEC);
        if ((nsec >= QP_SCHED_UPDATE_MIN) && (nsec <= QP_SCHED_UPDATE_MAX)) {
          parms->logInterval = nsec;
          INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) RCCL_IB_QP_SCHED_LOG_INTERVAL set to %lu nsec", nsec);
        }
      }
    }
  
    INFO(NCCL_NET|NCCL_ENV, "(IB-CAST) NCCL_IB_QPS_PER_CONNECTION set to %d", ncclParamIbCastQpsPerConn());
  
  exit:
    free(logFileName);
    return ret;
  
  err_exit:
    ret = ncclInternalError;
    goto exit;
  }
  
  void IbCastLogSched(struct ncclIbSendComm *comm) {
    
    fprintf(IbCastQpSchedLogStream, "comm %p: num qp's %d ndevs %d\n", comm, comm->base.nqps, comm->base.vProps.ndevs);
    for (int i = 0; i < comm->base.nqps; i++) {
      ncclIbQp* qp = comm->base.qps + i;
      int devArrayIdx = qp->devIndex;
      int ibDevN = comm->devs[devArrayIdx].base.ibDevN;
      fprintf(IbCastQpSchedLogStream, "  qp[%02d]: dev[%d]=%s (ibDev %d): minW=%.4f curW=%.4f maxW=%.4f lat=%.2fus\n",
              i,
              devArrayIdx,
              IbCastDevs[ibDevN].devName,
              ibDevN,
              comm->base.qpTxSched[i].minWeight,
              comm->base.qpTxSched[i].weight,
              comm->base.qpTxSched[i].maxWeight,
              comm->base.qpTxStats[i].rtt / 1000.0);  // Convert ns to us
    }
    fflush(IbCastQpSchedLogStream);
  }

void IbCastUpdateSchedParmsTry(struct ncclIbNetCommBase *base, int nreqs, int size) {
  if (!base->schedParmsInit) {
    base->schedParms = castGlobalQpSchedParms;
    base->schedParmsInit = true;
  }
  if (base->stagedParmsConEpoch < stagedSchedParms.prodEpoch) {
    bool collTypeChanged = false, msgSzChanged = false;
    ncclFunc_t collType;
    size_t msgSz;
    pthread_mutex_lock(&ncclIbQpSchedParmsLock);
    collType = stagedSchedParms.collType;
    if (IbCastQpSchedProxyPrevCollType != collType) {
      IbCastQpSchedProxyPrevCollType = collType;
      collTypeChanged = true;
    }
    msgSz = stagedSchedParms.msgSz;
    if (IbCastQpSchedProxyPrevMsgSz != msgSz) {
      IbCastQpSchedProxyPrevMsgSz = msgSz;
      msgSzChanged = true;
    }
    base->schedParms = stagedSchedParms.parms;
    base->stagedParmsConEpoch = stagedSchedParms.prodEpoch;
    if (base->schedParms.resetRtt)
      base->resetRttDone = false;
    pthread_mutex_unlock(&ncclIbQpSchedParmsLock);
    if (msgSzChanged || collTypeChanged) {
      const char *collStr;
      switch (collType) {
      case ncclFuncBroadcast:
        collStr = "Broadcast";
        break;
      case ncclFuncReduce:
        collStr = "Reduce";
        break;
      case ncclFuncAllGather:
        collStr = "All_Gather";
        break;
      case ncclFuncReduceScatter:
        collStr = "Reduce_Scatter";
        break;
      case ncclFuncAllReduce:
        collStr = "All_Reduce";
        break;
      default:
        collStr = NULL;
        break;
      }
      if (collStr == NULL)
        INFO(NCCL_NET, "PID %d: CollType = %d  MsgSz = %lu  ChunkSz = %d  NumChunks = %d",
             getpid(), collType, msgSz, size, nreqs);
      else
        INFO(NCCL_NET, "PID %d: CollType = %s  MsgSz = %lu  ChunkSz = %d  NumChunks = %d",
             getpid(), collStr, msgSz, size, nreqs);
    }
  }
}
void IbCastQpSchedUpdateTx(struct ncclIbNetCommBase *base) {
  int qp;
  uint64_t minRttSample;
  double minRtt = DBL_MAX, rttSum = 0.0;
  struct ncclIbQpTxSchedScratchpad s;
  struct ncclIbRrTokens *tokens;

  for (qp = 0; qp < base->nqps; qp++) {
    if (base->qpTxStats[qp].rtt < minRtt) {
      minRtt = base->qpTxStats[qp].rtt;
      minRttSample = base->qpTxStats[qp].minRttSample;
    }
  }

  for (qp = 0; qp < base->nqps; qp++) {
    if (base->qpTxStats[qp].rtt == 0.0)
      return;
    double denom = (base->qpTxStats[qp].rtt - minRtt) + minRttSample;
    if (denom <= 0.0)
      return;
    s.rtt[qp] = 1.0 / denom;
    rttSum += s.rtt[qp];
  }

  if (rttSum == 0.0) {
    for (qp = 0; qp < base->nqps; qp++)
      base->qpTxSched[qp].weight = 1.0 / ((double) base->nqps);
  } else {
    for (qp = 0; qp < base->nqps; qp++)
        base->qpTxSched[qp].weight = s.rtt[qp] / rttSum;
  }
  if (base->schedParms.logEnable) {
    if (!base->qpTxSchedInit) {
      for (qp = 0; qp < base->nqps; qp++)
        base->qpTxSched[qp].minWeight = DBL_MAX;
    }

    for (qp = 0; qp < base->nqps; qp++) {
      if (base->qpTxSched[qp].weight < base->qpTxSched[qp].minWeight)
        base->qpTxSched[qp].minWeight = base->qpTxSched[qp].weight;
      if (base->qpTxSched[qp].weight > base->qpTxSched[qp].maxWeight)
        base->qpTxSched[qp].maxWeight = base->qpTxSched[qp].weight;
    }
  }

  tokens = &base->rrQpTxSched.initTokens;
  tokens->totTokens = 0;
  for (qp = 0; qp < base->nqps; qp++) {
    double temp, temp3;
    int temp2;
    temp = ((double) NCCL_IB_TARGET_TOT_TOKENS) *
           base->qpTxSched[qp].weight;
    temp2 = (int) temp;
    temp3 = (double) temp2;
    if (((temp - temp3) > 0.5) || (temp2 == 0))
      temp2++;
    tokens->qpTokens[qp] = temp2;
    tokens->totTokens += temp2;
  }

  base->qpTxSchedInit = true;
}


void IbCastQpSchedUpdateTxStats(struct ncclIbRemapWrId *remap,
                            struct ncclIbNetCommBase *base) {
  uint64_t nowNs, sampleNs, sampleTxTimeNs, rtt, resetInterval;
  double sampleTxTime, weightNew, weightPrev;
  struct timespec now;
  struct ncclIbQpTxStats *qpTxStats;

  if (remap->tx.startTimeNs == 0)
    return;

  if (clock_gettime(CLOCK_MONOTONIC, &now))
    return;

  nowNs = TIMESPEC_TO_NSEC(&now);
  bool resetRtt = false;
  if (remap->parms.resetRtt && (!base->resetRttDone)) {
    resetRtt = true;
    base->resetRttDone = true;
  }
  resetInterval = remap->parms.resetInterval;
  if ((resetInterval != QP_SCHED_RESET_NEVER) &&
      (nowNs >= base->nextQpTxStatsResetNs))
    resetRtt = true;
  if (resetRtt) {
    for (int qp = 0; qp < base->nqps; qp++) {
      base->qpTxStats[qp].totRtt = 0;
      base->qpTxStats[qp].numMeasurements = 0;
    }
    base->nextQpTxStatsResetNs = nowNs + resetInterval;
  }

  sampleNs = nowNs - remap->tx.startTimeNs;
  qpTxStats = &base->qpTxStats[remap->qpIndex];
  // Get the IB device index: qp->devIndex is index into vProps.devs[], which contains actual IB device indices
  int qpDevArrayIndex = base->qps[remap->qpIndex].devIndex;
  int ibDevIndex = base->vProps.devs[qpDevArrayIndex];
  sampleTxTime = (((double) remap->tx.bytes) * ((double) BITS_PER_BYTE)) /
                 (((double) IbCastDevs[ibDevIndex].speed) * ((double) MEG));
  sampleTxTimeNs = (uint64_t) (sampleTxTime * ((double) NSEC_PER_SEC));
  rtt = sampleNs - sampleTxTimeNs;
  qpTxStats->totRtt += rtt;
  qpTxStats->numMeasurements++;
  if ((qpTxStats->minRttSample == 0) || (rtt < qpTxStats->minRttSample))
    qpTxStats->minRttSample = rtt;
  if (qpTxStats->numMeasurements == 1) {
    qpTxStats->rtt = (double) rtt;
    return;
  }
  weightNew = remap->parms.weightNew;
  weightPrev = 1.0 - weightNew;
  if (weightNew == ((double) QP_SCHED_WEIGHT_NONE))
    qpTxStats->rtt = ((double) qpTxStats->totRtt) /
                     ((double) qpTxStats->numMeasurements);
  else
    qpTxStats->rtt = (qpTxStats->rtt * weightPrev) +
                     (((double) rtt) * weightNew);
}

int IbCastQpSchedGetEffectiveTxNqps(struct ncclIbRequest* req, int *startQpIndex, bool *wrrSched) {
  int dataPerQp, qpIndex = req->base->qpIndex, nqps = req->base->nqps;
  struct ncclIbQpSchedParms *parms = &req->desc.parms;

  *wrrSched = false;

  if (req->base->nqps == 1)
    goto exit;

  if (!parms->splitData)
    goto oneQp;

  dataPerQp = (req->send.size * req->nreqs) / req->base->nqps;
  if ((dataPerQp >= parms->splitDataMin) ||  !parms->enable) {
    goto exit;
  }

oneQp:
  nqps = 1;
  if (parms->doWrr && req->base->qpTxSchedInit) {
    if (req->base->rrQpTxSched.activeTokens.totTokens == 0)
      req->base->rrQpTxSched.activeTokens = req->base->rrQpTxSched.initTokens;
    qpIndex = req->base->rrQpTxSched.qpIndex;
    while (1) {
      if (req->base->rrQpTxSched.activeTokens.qpTokens[qpIndex]) {
        req->base->rrQpTxSched.activeTokens.qpTokens[qpIndex]--;
        req->base->rrQpTxSched.activeTokens.totTokens--;
        req->base->rrQpTxSched.qpIndex = (qpIndex+1) % req->base->nqps;
        *wrrSched = true;
        break;
      }
      qpIndex = (qpIndex+1) % req->base->nqps;
    }
  }

exit:
  *startQpIndex = qpIndex;
  return nqps;
}

ncclResult_t IbCastQpSchedGetRemap(struct ncclIbNetCommBase* base, uint64_t wrId, int qpIndex, struct ncclIbRemapWrId** remap) {
  for (int i=0; i<NET_IB_MAX_REQUESTS; i++) {
    struct ncclIbRemapWrId* r = base->remapWrId+i;
    if (r->state == NCCL_NET_IB_REMAP_UNUSED) {
      r->origWrId = wrId;
      r->qpIndex = qpIndex;
      r->state = NCCL_NET_IB_REMAP_USED;
      *remap = r;
      return ncclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate remap buffer");
  *remap = NULL;
  return ncclInternalError;
}
ncclResult_t IbCastQpSchedFreeRemap(struct ncclIbRemapWrId* r) {
  r->state = NCCL_NET_IB_REMAP_UNUSED;
  return ncclSuccess;
}

// =============================================================================
// Test introspection API — exposes internal WRR scheduler state from a
// sendComm handle.  Only intended for unit tests; not part of the public net
// plugin ABI.
// Struct definition and function prototypes live in src/include/net_ib_cast_inspect.h.
// =============================================================================

// ncclIbCastGetSchedState — copy WRR scheduler state out of a sendComm.
// sendComm must be a valid ncclIbSendComm* obtained from IbCastConnect/IbCastAccept.
// Returns ncclInvalidArgument if sendComm or out is null.
extern "C" ncclResult_t ncclIbCastGetSchedState(void* sendComm, struct ncclIbCastSchedState* out) {
  if (!sendComm || !out) return ncclInvalidArgument;
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*) sendComm;
  struct ncclIbNetCommBase* base = &comm->base;

  out->nqps      = base->nqps;
  out->schedInit = base->qpTxSchedInit;
  out->qpIndex   = base->rrQpTxSched.qpIndex;

  out->initTotTokens   = base->rrQpTxSched.initTokens.totTokens;
  out->activeTotTokens = base->rrQpTxSched.activeTokens.totTokens;

  int n = (base->nqps < NCCL_IB_MAX_QPS) ? base->nqps : NCCL_IB_MAX_QPS;
  for (int i = 0; i < n; i++) {
    out->initQpTokens[i]   = base->rrQpTxSched.initTokens.qpTokens[i];
    out->activeQpTokens[i] = base->rrQpTxSched.activeTokens.qpTokens[i];
  }

  out->schedEnable  = base->schedParms.enable;
  out->doWrr        = base->schedParms.doWrr;
  out->splitData    = base->schedParms.splitData;
  out->splitDataMin = base->schedParms.splitDataMin;

  return ncclSuccess;
}

// ncclIbCastSetTokens — force-initialize the WRR token table for testing.
// Bypasses the RTT-based IbCastQpSchedUpdateTx; immediately arms the scheduler.
// qpTokens must have nqps entries; totTokens is computed as their sum.
extern "C" ncclResult_t ncclIbCastSetTokens(void* sendComm, const int* qpTokens, int nqps) {
  if (!sendComm || !qpTokens || nqps <= 0 || nqps > NCCL_IB_MAX_QPS)
    return ncclInvalidArgument;
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*) sendComm;
  struct ncclIbNetCommBase* base = &comm->base;

  // If the connection is already established, nqps must match the real QP count.
  if (base->nqps > 0 && nqps != base->nqps)
    return ncclInvalidArgument;

  struct ncclIbRrTokens* t = &base->rrQpTxSched.initTokens;
  t->totTokens = 0;
  for (int i = 0; i < nqps; i++) {
    t->qpTokens[i] = qpTokens[i];
    t->totTokens  += qpTokens[i];
  }
  // Zero out entries beyond nqps so stale values from a previous call cannot
  // be observed by the WRR cursor if base->nqps ever changes.
  for (int i = nqps; i < NCCL_IB_MAX_QPS; i++)
    t->qpTokens[i] = 0;

  base->rrQpTxSched.activeTokens = *t;
  base->rrQpTxSched.qpIndex      = 0;
  base->qpTxSchedInit = true;

  return ncclSuccess;
}


// ncclIbCastSetSchedParms — override schedParms fields for testing.
// Takes effect on the very next isend; does not require re-connection.
// Only the four fields most relevant to path-selection are exposed.
extern "C" ncclResult_t ncclIbCastSetSchedParms(void* sendComm,
                                                bool schedEnable,
                                                bool doWrr,
                                                bool splitData,
                                                uint32_t splitDataMin) {
    if (!sendComm) return ncclInvalidArgument;
    struct ncclIbSendComm* comm = (struct ncclIbSendComm*) sendComm;
    struct ncclIbNetCommBase* base = &comm->base;
    base->schedParms.enable       = schedEnable;
    base->schedParms.doWrr        = doWrr;
    base->schedParms.splitData    = splitData;
    base->schedParms.splitDataMin = splitDataMin;
    return ncclSuccess;
}