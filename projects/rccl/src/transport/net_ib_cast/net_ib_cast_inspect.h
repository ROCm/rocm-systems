/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_CAST_INSPECT_H_
#define RCCL_NET_IB_CAST_INSPECT_H_

#include <stdint.h>
#include "net_ib_limits.h"

#ifdef __cplusplus
#include "nccl.h"  /* ncclResult_t */
extern "C" {
#else
#include "nccl.h"
#endif

/*
 * Test-only introspection API for the net-ib-cast WRR scheduler.
 *
 * Implementation lives in src/transport/net_ib_cast/scheduler.cc and is
 * compiled into librccl.so.  Both the library and the unit tests include this
 * one header so the struct layout and function signatures can never diverge.
 */

struct ncclIbCastSchedState {
  int      nqps;
  int      qpIndex;          /* current WRR cursor */
  int      initTotTokens;
  int      initQpTokens[NCCL_IB_MAX_QPS];
  int      activeTotTokens;
  int      activeQpTokens[NCCL_IB_MAX_QPS];
  uint32_t splitDataMin;
  bool     schedInit;        /* true once IbCastQpSchedUpdateTx has fired */
  bool     schedEnable;
  bool     doWrr;
  bool     splitData;
};

/* Copy WRR scheduler state out of a sendComm.
 * sendComm must be a valid ncclIbSendComm* from IbCastConnect/IbCastAccept.
 * Returns ncclInvalidArgument if either pointer is null. */
ncclResult_t ncclIbCastGetSchedState(void* sendComm, struct ncclIbCastSchedState* out);

/* Force-initialize the WRR token table, bypassing RTT-based scheduling.
 * Immediately sets qpTxSchedInit=true.
 * nqps must equal the connection's actual nqps (base->nqps). */
ncclResult_t ncclIbCastSetTokens(void* sendComm, const int* qpTokens, int nqps);

/* Override schedParms fields for mid-test toggling.
 * Takes effect on the very next isend; does not require re-connection. */
ncclResult_t ncclIbCastSetSchedParms(void* sendComm,
                                      bool schedEnable,
                                      bool doWrr,
                                      bool splitData,
                                      uint32_t splitDataMin);

/* ── Resiliency state introspection (requires ENABLE_FAULT_INJECTION) ── */
#ifdef ENABLE_FAULT_INJECTION

struct ncclIbCastResiliencyState {
  bool recoveryEnabled;
  bool inProgress;
  int  outstandingRequests;
  int  outstandingRecovery;
  int  ndevs;
  int  devState[4];
};

/* Fills out with the current resiliency state of the communicator.
 * sendComm must have resiliency enabled (NCCL_IB_RESILIENCY_PORT_FAILOVER=1).
 * Returns ncclInvalidArgument if resiliency context is NULL. */
ncclResult_t ncclIbCastGetResiliencyState(void* sendComm, struct ncclIbCastResiliencyState* out);

/* Returns the number of times IbCastResiliencyRepostRequest was called
 * (i.e. selective retransmit count). Counter is stored in the resiliency
 * context and incremented in p2p_resiliency.cc. */
ncclResult_t ncclIbCastGetRepostCount(void* sendComm, int* out);

#endif /* ENABLE_FAULT_INJECTION */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RCCL_NET_IB_CAST_INSPECT_H_ */
