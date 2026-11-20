/*************************************************************************
 * Copyright (c) 2016-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_P2P_RESILIENCY_RECOVERY_MSEG_H_
#define NET_IB_P2P_RESILIENCY_RECOVERY_MSEG_H_

#include "nccl.h" // For ncclResult_t
#include "p2p_resiliency_mseg.h"

ncclResult_t IbMsegPortRecoveryThreadStart();
ncclResult_t IbMsegPortRecoveryThreadStop();
ncclResult_t IbMsegPortRecoveryInit(struct IbMsegResiliency* resCtx);
ncclResult_t IbMsegPortRecoveryClose(struct IbMsegResiliency* resCtx);

ncclResult_t IbMsegPortRecoveryDevInit(struct IbMsegResiliency* resCtx, int devIndex, IbMsegDev* ibDev);
ncclResult_t IbMsegPortRecoveryDevDestroy(struct IbMsegResiliency* resCtx, int devIndex);

ncclResult_t IbMsegPortRecoverySenderQpsCreate(struct IbMsegResiliency* resCtx,
                                               struct IbMsegQpInfo* localResiliencyInfo, int nQps);
ncclResult_t IbMsegPortRecoverySenderQpsToRts(struct IbMsegResiliency* resCtx, struct IbMsegConnectionMetadata* remInfo,
                                              int nQps);

ncclResult_t IbMsegPortRecoveryReceiverQpsCreateToRts(struct IbMsegResiliency* resCtx,
                                                      struct IbMsegConnectionMetadata* remInfo,
                                                      struct IbMsegQpInfo* localPortRecoveryQpsInfo, int nQps);

ncclResult_t IbMsegPortRecoveryQpsDestroy(struct IbMsegResiliency* resCtx, int nQps);

ncclResult_t IbMsegPortRecoveryHandleFailure(struct IbMsegResiliency* resCtx, int devIndex);

ncclResult_t IbMsegPortRecoveryQpsReconfigure(struct IbMsegResiliency* resCtx, int devIndex, bool* success);

#endif // NET_IB_P2P_RESILIENCY_RECOVERY_MSEG_H_
