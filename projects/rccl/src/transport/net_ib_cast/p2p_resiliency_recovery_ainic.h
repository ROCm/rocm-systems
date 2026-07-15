/*************************************************************************
 * Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_P2P_RESILIENCY_RECOVERY_AINIC_H_
#define NET_IB_P2P_RESILIENCY_RECOVERY_AINIC_H_

#include "nccl.h"
#include "p2p_resiliency_cast.h"

/* Forward declaration — ncclIbPortRecoveryContext is defined in p2p_resiliency_recovery_internal.h */
struct ncclIbPortRecoveryContext;
struct ibv_wc;

/* Phase A: destroy+recreate QPs on AINIC (ibv_modify_qp RESET is broken).
 * Records new QPNs in recoveryContext->localQpns for exchange with remote. */
ncclResult_t IbCastPortRecoveryQpsRestoreAinic(struct ncclIbPortRecoveryContext* recoveryContext);

/* Phase B: transition recreated QPs to RTR+RTS using remote QPNs from ACK.
 * No-op on non-AINIC paths. */
ncclResult_t IbCastPortRecoveryQpsToRtsAinic(struct ncclIbPortRecoveryContext* recoveryContext);

/* Post a UD ACK message carrying local QPN entries as payload. */
ncclResult_t IbCastPortRecoveryPostAckWithQpnsAinic(struct ncclIbPortRecoveryContext* recoveryContext, bool* success);

/* Extract remote QPNs from a received ACK completion buffer (after GRH).
 * No-op on non-AINIC paths where ACK messages carry no QPN payload. */
void IbCastPortRecoveryExtractQpnsAinic(struct ncclIbPortRecoveryContext* recoveryContext, struct ibv_wc* wc);

#endif /* NET_IB_P2P_RESILIENCY_RECOVERY_AINIC_H_ */
