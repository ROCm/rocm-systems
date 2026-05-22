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
 * Shared between librccl.so and the unit tests so the struct layout
 * and signatures cannot diverge.
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

/* Copy scheduler state out of a connected sendComm.
 * Returns ncclInvalidArgument on null pointers. */
ncclResult_t ncclIbCastGetSchedState(void* sendComm, struct ncclIbCastSchedState* out);

/* Force-initialize the WRR token table, bypassing RTT-driven scheduling.
 * nqps must match the connection's nqps. */
ncclResult_t ncclIbCastSetTokens(void* sendComm, const int* qpTokens, int nqps);

/* Override schedParms; takes effect on the next isend, no reconnect needed. */
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
