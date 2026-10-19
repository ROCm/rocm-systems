/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_IB_FLUSH_INSPECT_H_
#define RCCL_NET_IB_FLUSH_INSPECT_H_

#ifdef ENABLE_FAULT_INJECTION

#include "nccl.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Test-only introspection of the GDR flush setup that net_ib accept leaves on a
 * recv comm. Only valid for comms created by ncclNetIb, not net_ib_cast.
 */

#define NCCL_IB_FLUSH_INSPECT_MAX_DEVS 8

struct ncclIbFlushState {
  int flushEnabled;
  int ndevs;
  /* RO=0 scratchpad that ncclIbIflush writes before the read; per comm device. */
  bool scratchpadRegistered[NCCL_IB_FLUSH_INSPECT_MAX_DEVS];
  bool scratchpadDmaBuf[NCCL_IB_FLUSH_INSPECT_MAX_DEVS];
};

ncclResult_t ncclIbFlushGetState(void* recvComm, struct ncclIbFlushState* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ENABLE_FAULT_INJECTION */

#endif /* RCCL_NET_IB_FLUSH_INSPECT_H_ */
