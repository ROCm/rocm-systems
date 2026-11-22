/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// net_ib's parameter definitions, which IB-MSEG declares extern; the fork binaries link no net_ib unit.
// Hand-copied from src/transport/net_ib/*.cc; NetIbMsegParamCopies (connect-test.cc) fails when they drift.

#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

// src/transport/net_ib/common.cc
NCCL_PARAM(IbSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 0);
NCCL_PARAM(IbPrepostReceiveWorkRequests, "IB_PREPOST_RECEIVE_WORK_REQUESTS", -2);
NCCL_PARAM(IbAsyncEvents, "IB_RETURN_ASYNC_EVENTS", 1);
NCCL_PARAM(IbEventBasedLb, "IB_EVENT_BASED_LB", 0);
NCCL_PARAM(IbEventBasedLbRemote, "IB_EVENT_BASED_LB_REMOTE", 1);

// src/transport/net_ib/gdr.cc
RCCL_PARAM(ForceEnableGdrdma, "FORCE_ENABLE_GDRDMA", -1);

// src/transport/net_ib/gin.cc
#define NCCL_RMA_IB_WR_BATCHSIZE 64
NCCL_PARAM(GinType, "GIN_TYPE", -1);
NCCL_PARAM(GinIbTc, "GIN_IB_TC", -1);
NCCL_PARAM(RmaIbWrBatchsize, "RMA_IB_WR_BATCHSIZE", NCCL_RMA_IB_WR_BATCHSIZE);

// src/transport/net_ib/init.cc
NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);
NCCL_PARAM(IbAdaptiveRouting, "IB_ADAPTIVE_ROUTING", -2);
NCCL_PARAM(IbDataDirect, "IB_DATA_DIRECT", 1);
NCCL_PARAM(IbOooRq, "IB_OOO_RQ", 0)
NCCL_PARAM(IbDisable, "IB_DISABLE", 0);
NCCL_PARAM(IbMergeVfs, "IB_MERGE_VFS", 1);
NCCL_PARAM(IbMergeNics, "IB_MERGE_NICS", 1);
NCCL_PARAM(IbDevicePciOrder, "IB_DEVICE_PCI_ORDER", 1);
NCCL_PARAM(IbSortMergeNics, "IB_SORT_MERGE_NICS", 0);

// src/transport/net_ib/p2p_resiliency.cc
NCCL_PARAM(IbResiliencyPortFailoverMaxAttempts, "IB_RESILIENCY_PORT_FAILOVER_MAX_ATTEMPTS", 1);
NCCL_PARAM(IbResiliencyPortFailoverProbeDelay, "IB_RESILIENCY_PORT_FAILOVER_PROBE_DELAY", 10);

// src/transport/net_ib/p2p_resiliency_recovery.cc
NCCL_PARAM(IbResiliencyPortRecovery, "IB_RESILIENCY_PORT_RECOVERY", 0);
NCCL_PARAM(IbResiliencyPortRecoveryStartDelay, "IB_RESILIENCY_PORT_RECOVERY_START_DELAY", 200);
NCCL_PARAM(IbResiliencyPortRecoveryAliveMsgBatchInterval, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_INTERVAL",
           500);
NCCL_PARAM(IbResiliencyPortRecoveryAliveMsgBatchSize, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE", 5);
NCCL_PARAM(IbResiliencyPortRecoveryAliveMsgSequenceSize, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_SEQUENCE_SIZE", 5);
NCCL_PARAM(IbResiliencyPortRecoveryAliveMsgTimeout, "IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_TIMEOUT",
           4000);
NCCL_PARAM(IbResiliencyPortRecoveryAckTimeout, "IB_RESILIENCY_PORT_RECOVERY_ACK_TIMEOUT", 5000);
NCCL_PARAM(IbResiliencyPortRecoveryAttemptsMax, "IB_RESILIENCY_PORT_RECOVERY_ATTEMPTS_MAX", 20);

// src/transport/net_ib/wqe_lat_mon.cc
NCCL_PARAM(IbWqeLatencyThresholdNs, "IB_WQE_LATENCY_THRESHOLD_NS", 0);
NCCL_PARAM(IbWqeLatencyReport, "IB_WQE_LATENCY_REPORT", 1);
