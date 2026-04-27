/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_FAULT_INJECT_H_
#define RCCL_NET_IB_FAULT_INJECT_H_

#ifdef ENABLE_FAULT_INJECTION

#include <stdint.h>

#ifdef __cplusplus
#include "nccl.h"  /* ncclResult_t */
extern "C" {
#else
#include <stdbool.h>
#include "nccl.h"
#endif

/*
 * Test-only per-QP fault injection API for the net-ib CAST transport.
 *
 * Implemented in src/transport/net_ib_cast.cc (CAST multi-QP path).
 *
 * NCCL_IB_MAX_QPS is defined in net_ib_cast_inspect.h; guard against
 * double-definition when both headers are included together.
 */
#ifndef NCCL_IB_MAX_QPS
#define NCCL_IB_MAX_QPS 128
#endif

/* ── CAST path (net_ib_cast.cc) ───────────────────────────────────────── */

/* Set an artificial delay (microseconds) on a specific QP index.
 * qpIdx must be in [0, NCCL_IB_MAX_QPS).
 * Set delayUs=0 to clear the delay. */
ncclResult_t ncclIbCastFaultSetQpDelay(void* sendComm, int qpIdx, uint32_t delayUs);

/* Arm error injection on a specific QP index.
 * When armed, the hook calls ncclIbStatsFatalError and then returns
 * ncclSystemError instead of calling wrap_ibv_post_send.
 * Set inject=false to disarm. */
ncclResult_t ncclIbCastFaultSetQpError(void* sendComm, int qpIdx, bool inject);

/* Clear all fault state (delays and errors) on the connection. */
ncclResult_t ncclIbCastFaultClear(void* sendComm);

/* Return the current fatalErrorCount from the connection's stats. */
ncclResult_t ncclIbCastFaultGetFatalCount(void* sendComm, int* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ENABLE_FAULT_INJECTION */

#endif /* RCCL_NET_IB_FAULT_INJECT_H_ */
