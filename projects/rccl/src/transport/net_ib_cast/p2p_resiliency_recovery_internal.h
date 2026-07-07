/*************************************************************************
 * Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/* Internal types and constants shared between p2p_resiliency_recovery.cc
 * and p2p_resiliency_recovery_ainic.cc. Not part of the public plugin API. */

#ifndef NET_IB_P2P_RESILIENCY_RECOVERY_INTERNAL_H_
#define NET_IB_P2P_RESILIENCY_RECOVERY_INTERNAL_H_

#include "p2p_resiliency_recovery_cast.h"

#define NCCL_IB_RESILIENCY_PORT_RECOVERY_ACK_MSG_ID (0x1234)
#define NCCL_IB_RECOVERY_QKEY                        0x1234ABCD
#define NCCL_IB_MAX_RECOVERY_QPN_ENTRIES             NCCL_IB_MAX_QPS

enum ncclIbPortRecoveryState {
  ncclIbPortRecoveryStateInit,
  ncclIbPortRecoveryStateAliveMessages,
  ncclIbPortRecoveryStateAck,
  ncclIbPortRecoveryStateSuccess,
  ncclIbPortRecoveryStateFailed,
};

struct ncclIbPortRecoveryContext {
  struct ncclIbResiliency* resCtx;
  int devIndex;
  enum ncclIbPortRecoveryState state;

  struct ibv_cq* recoveryCq;

  uint64_t timeInit;
  int nFailedAttempts;
  uint64_t timeLastMsg;
  uint32_t aliveMsgNextId;

  bool ackReceived;
  bool ackPosted;
  bool ackCompleted;

  union {
    struct {
      bool aliveMsgPosted;
      bool aliveMsgCompleted;
    } send;
    struct {
      uint32_t windowBase;
      uint32_t receivedBits;
    } recv;
  };

  /* AINIC destroy+recreate: local and remote QPN exchange buffers. */
  struct RecoveryQpnEntry localQpns[NCCL_IB_MAX_QPS];
  int nLocalQpns;
  struct RecoveryQpnEntry remoteQpns[NCCL_IB_MAX_QPS];
  int nRemoteQpns;
};

#endif /* NET_IB_P2P_RESILIENCY_RECOVERY_INTERNAL_H_ */
