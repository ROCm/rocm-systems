/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_CAST_INSPECT_H_
#define RCCL_NET_IB_CAST_INSPECT_H_

#include <stdint.h>

#ifdef __cplusplus
#include "nccl.h"  /* ncclResult_t */
extern "C" {
#else
#include "nccl.h"
#endif

/*
 * Test-only introspection API for the net-ib-cast WRR scheduler.
 *
 * Implementation lives in src/transport/net_ib_cast.cc and is compiled into
 * librccl.so.  Both the library and the unit tests include this one header so
 * the struct layout and function signatures can never diverge.
 *
 * NCCL_IB_MAX_QPS is the maximum number of QPs per connection.
 * Defined here so both the library and the tests share the same value.
 */
#define NCCL_IB_MAX_QPS 128

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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RCCL_NET_IB_CAST_INSPECT_H_ */
