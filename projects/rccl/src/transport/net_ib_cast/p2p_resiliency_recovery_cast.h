/*************************************************************************
 * Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_P2P_RESILIENCY_RECOVERY_H_
#define NET_IB_P2P_RESILIENCY_RECOVERY_H_

#include "nccl.h" // For ncclResult_t
#include "p2p_resiliency_cast.h"

// Determines the size of the work queue on the recovery QPs
#define NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX 10

// The asynchronous thread should be be able to handle the CQ fast enough so
// no more than two batches of "alive" messsages should be pending in the CQ at
// any time.
#define NCCL_IB_RESILIENCY_PORT_RECOVERY_CQ_SIZE (NCCL_IB_RESILIENCY_PORT_RECOVERY_ALIVE_MSG_BATCH_SIZE_MAX * 2)

ncclResult_t IbCastPortRecoveryThreadStart();
ncclResult_t IbCastPortRecoveryThreadStop();
ncclResult_t IbCastPortRecoveryInit(struct ncclIbResiliency* resCtx);
ncclResult_t IbCastPortRecoveryClose(struct ncclIbResiliency* resCtx);

ncclResult_t IbCastPortRecoveryDevInit(struct ncclIbResiliency* resCtx, int devIndex, ncclIbDev* ibDev);
ncclResult_t IbCastPortRecoveryDevDestroy(struct ncclIbResiliency* resCtx, int devIndex);

ncclResult_t IbCastPortRecoverySenderQpsCreate(struct ncclIbResiliency* resCtx,
                                               struct ncclIbQpInfo* localResiliencyInfo, int nQps);
ncclResult_t IbCastPortRecoverySenderQpsToRts(struct ncclIbResiliency* resCtx, struct ncclIbConnectionMetadata* remInfo,
                                              int nQps);

ncclResult_t IbCastPortRecoveryReceiverQpsCreateToRts(struct ncclIbResiliency* resCtx,
                                                      struct ncclIbConnectionMetadata* remInfo,
                                                      struct ncclIbQpInfo* localPortRecoveryQpsInfo, int nQps);

ncclResult_t IbCastPortRecoveryQpsDestroy(struct ncclIbResiliency* resCtx, int nQps);

// The INIT, RTR and RTS transitions of a recovery UD QP, done in one go.
ncclResult_t IbCastPortRecoveryQpUdToRts(struct ncclIbQp* qp, int portNum);

ncclResult_t IbCastPortRecoveryHandleFailure(struct ncclIbResiliency* resCtx, int devIndex);

#endif // NET_IB_P2P_RESILIENCY_RECOVERY_H_
