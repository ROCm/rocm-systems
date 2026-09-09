/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_FLUSH_FAULT_INJECT_H_
#define RCCL_NET_IB_FLUSH_FAULT_INJECT_H_

#ifdef ENABLE_FAULT_INJECTION

#ifdef __cplusplus
#include "nccl.h"  /* ncclResult_t */
extern "C" {
#else
#include <stdbool.h>
#include "nccl.h"
#endif

/*
 * Test-only fault injection for the base net-ib GDR flush path.
 *
 * Implemented in src/transport/net_ib/p2p.cc (ncclIbIflush).
 */

/* Force ncclIbIflush to re-issue the pre-fix scratchpad RDMA_WRITE before the
 * flush READ, reproducing the dma-buf QP async-fatal for the regression test;
 * enable=false restores shipped read-only behaviour. Toggle is process-global. */
ncclResult_t ncclIbFlushFaultForceScratchpadWrite(void* recvComm, bool enable);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ENABLE_FAULT_INJECTION */

#endif /* RCCL_NET_IB_FLUSH_FAULT_INJECT_H_ */
