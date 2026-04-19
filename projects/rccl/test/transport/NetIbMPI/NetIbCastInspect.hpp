/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstdint>
#include "nccl.h" // ncclResult_t — must match the definitions in net_ib_cast.cc exactly

// Test-only introspection API for the net-ib-cast WRR scheduler.
// The implementation lives in src/transport/net_ib_cast.cc and is compiled
// into librccl.so.  This header just declares the structs and the functions.

#ifdef MPI_TESTS_ENABLED

#define NCCL_IB_CAST_INSPECT_MAX_QPS 128

struct ncclIbCastSchedState {
  int      nqps;
  bool     schedInit;        // true once IbCastQpSchedUpdateTx has fired
  int      qpIndex;          // current WRR cursor

  // initTokens snapshot
  int      initTotTokens;
  int      initQpTokens[NCCL_IB_CAST_INSPECT_MAX_QPS];

  // activeTokens snapshot
  int      activeTotTokens;
  int      activeQpTokens[NCCL_IB_CAST_INSPECT_MAX_QPS];

  // schedParms snapshot (subset most useful for tests)
  bool     schedEnable;
  bool     doWrr;
  bool     splitData;
  uint32_t splitDataMin;
};

// Declared in net_ib_cast.cc; linked from librccl.so in debug builds.
extern "C" ncclResult_t ncclIbCastGetSchedState(void* sendComm, struct ncclIbCastSchedState* out);

// Force-initialize the WRR token table, bypassing RTT-based scheduling.
// Immediately arms schedInit=true.
// nqps must equal the connection's actual nqps (base->nqps).
extern "C" ncclResult_t ncclIbCastSetTokens(void* sendComm, const int* qpTokens, int nqps);

// Override schedParms fields for mid-test toggling.
// Takes effect on the very next isend; does not require re-connection.
extern "C" ncclResult_t ncclIbCastSetSchedParms(void* sendComm,
                                                 bool schedEnable,
                                                 bool doWrr,
                                                 bool splitData,
                                                 uint32_t splitDataMin);

#endif // MPI_TESTS_ENABLED
