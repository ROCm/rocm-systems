/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_CONNECT_MSEG_H_
#define NET_IB_CONNECT_MSEG_H_

#include "common_mseg.h"
#include "ibvwrap.h"

struct IbMsegQpCreateAttr {
  void* qpContext;
  enum ibv_qp_type type;
  bool oooRq;
  struct ibv_cq* cq;
  struct ibv_pd* pd;
  uint32_t maxRecvWorkRequest;
  uint32_t maxSendWorkRequest;
  uint32_t maxInlineData;
};

// Per-QP connection metatdata
struct IbMsegQpInfo {
  uint32_t qpn;

  // Fields needed for ece (enhanced connection establishment)
  struct ibv_ece ece;
  int ece_supported;

  // The index of the device on which the QP was created. Allows the sender and
  // receiver side to have asymmetric device configuration, meaning the sender
  // and receiver can use different number of devices.
  int devIndex;
};

struct IbMsegResiliencyInfo {
  // QPs used for probing of data transfers in case of QP/device failures.
  struct IbMsegQpInfo probingQpsInfo[NCCL_IB_MAX_DEVS_PER_NIC];
  // QPs used for recovery protocol after QP/device failures.
  struct IbMsegQpInfo portRecoveryQpsInfo[NCCL_IB_MAX_DEVS_PER_NIC];
};

// Structure used to hold information needed to establish the communication
// between the sender and receiver.
// The structure is populated during the connection establishment phase and
// populated by each side of the connection before being sent to the remote
// peer. The remote peer uses the information passed to it from its peer to
// create and initialize its local resources.
struct IbMsegConnectionMetadata {
  struct IbMsegQpInfo qpInfo[NCCL_IB_MAX_QPS];
  struct IbMsegResiliencyInfo resiliencyInfo;
  struct IbMsegDevInfo devs[NCCL_IB_MAX_DEVS_PER_NIC];
  char devName[MAX_MERGED_DEV_NAME];
  // An address for a registered memory to be accessed by the peer. The address
  // can be accessed using RDMA using the key specified in IbMsegDevInfo::rkey.
  // The sender side gets in this member, from the receiver, the address of the
  // memory to which the sender writes the sizes of the data transfers that
  // the sender sends.
  // The receiver side gets in this member, from the sender, the address of the
  // memory to which the receiver writes the CTS messages.
  uint64_t addr;
  uint64_t remSpeedBufAddr;
  uint32_t remSpeedBufRkey;
  int ndevs;
  int tc;
  int sl;
  int isP2p;
  // Receiver to sender: the GDR flush flags to write after each flagged
  // payload, or 0 when this connection keeps the RDMA read flush.
  uint32_t flagFlushRkey;
  uint64_t flagFlushAddr;
};

ncclResult_t IbMsegQpCreate(struct IbMsegQp* qp, struct IbMsegQpCreateAttr* createQpAttrs);
ncclResult_t IbMsegQpInit(struct IbMsegQp* qp);
ncclResult_t IbMsegQpRtr(struct IbMsegQp* qp);
ncclResult_t IbMsegQpRts(struct IbMsegQp* qp);
ncclResult_t IbMsegQpReset(struct IbMsegQp* qp);
ncclResult_t IbMsegQpError(struct IbMsegQp* qp);

ncclResult_t IbMsegPostReceiveWorkRequestsOnQp(struct IbMsegRecvComm* recvComm, IbMsegQp* dataQp);

#endif // NET_IB_CONNECT_MSEG_H_
