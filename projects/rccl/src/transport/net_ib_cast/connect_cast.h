/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_CONNECT_H_
#define NET_IB_CONNECT_H_

#include "common_cast.h"
#include "ibvwrap.h"

enum ncclIbCommState {
  ncclIbCommStateStart = 0,
  ncclIbCommStateConnect = 1,
  ncclIbCommStateAccept = 3,
  ncclIbCommStateSend = 4,
  ncclIbCommStateRecv = 5,
  ncclIbCommStateConnecting = 6,
  ncclIbCommStateConnected = 7,
  ncclIbCommStatePendingReady = 8,
  ncclIbCommStateSendDevList = 9,
  ncclIbCommStateRecvDevList = 10,
};

struct ncclIbCommStage {
  enum ncclIbCommState state;
  int offset;
  void* buffer;
  void* comm;
};

struct ncclIbHandle {
  union ncclSocketAddress connectAddr; // Filled by the target
  uint64_t magic; // random number to help debugging
  int isP2p;
  bool isRMA;
  struct ncclIbCommStage stage; // Used by the other side when connecting
};

struct ncclIbQpCreateAttr {
  void* qpContext;
  enum ibv_qp_type type;
  bool oooRq;
  struct ibv_cq* cq;
  struct ibv_pd* pd;
  uint32_t maxRecvWorkRequest;
  uint32_t maxSendWorkRequest;
  bool isDataQp;
  bool isCtsEnabled;
  int8_t ctsQpSlot;
  int channelId;
  int ibDevN;
  bool useIonic;
};

// Per-QP connection metatdata
struct ncclIbQpInfo {
  uint32_t qpn;

  // Fields needed for ece (enhanced connection establishment)
  struct ibv_ece ece;
  int ece_supported;

  // The index of the device on which the QP was created. Allows the sender and
  // receiver side to have asymmetric device configuration, meaning the sender
  // and receiver can use different number of devices.
  int devIndex;
  int ibv_dev_index;
};

struct ncclIbResiliencyInfo {
  // QPs used for probing of data transfers in case of QP/device failures.
  struct ncclIbQpInfo probingQpsInfo[NCCL_IB_MAX_DEVS_PER_NIC];
  // QPs used for recovery protocol after QP/device failures.
  struct ncclIbQpInfo portRecoveryQpsInfo[NCCL_IB_MAX_DEVS_PER_NIC];
};

// Structure used to hold information needed to establish the communication
// between the sender and receiver.
// The structure is populated during the connection establishment phase and
// populated by each side of the connection before being sent to the remote
// peer. The remote peer uses the information passed to it from its peer to
// create and initialize its local resources.
struct ncclIbConnectionMetadata {
  struct ncclIbQpInfo qpInfo[NCCL_IB_MAX_QPS];
  struct ncclIbResiliencyInfo resiliencyInfo;
  struct ncclIbDevInfo devs[NCCL_IB_MAX_DEVS_PER_NIC];
  char devName[MAX_MERGED_DEV_NAME];
  // An address for a registered memory to be accessed by the peer. The address
  // can be accessed using RDMA using the key specified in ncclIbDevInfo::rkey.
  // The sender side gets in this member, from the receiver, the address of the
  // memory to which the sender writes the sizes of the data transfers that
  // the sender sends.
  // The receiver side gets in this member, from the sender, the address of the
  // memory to which the receiver writes the CTS messages.
  uint64_t addr;
  int ndevs;
  int tc;
  int sl;
  int isP2p;
  bool isRMA;
};

ncclResult_t IbCastQpCreate(struct ncclIbQp* qp, struct ncclIbQpCreateAttr* createQpAttrs);
void IbCastBuildDataQpCreateAttr(struct ncclIbNetCommBase* base, int devIndex, struct ncclIbQpCreateAttr* out);
ncclResult_t IbCastQpInit(struct ncclIbQp* qp);
ncclResult_t IbCastQpRtr(struct ncclIbQp* qp);
ncclResult_t IbCastQpRts(struct ncclIbQp* qp);
ncclResult_t IbCastQpReset(struct ncclIbQp* qp);
ncclResult_t IbCastQpError(struct ncclIbQp* qp);

ncclResult_t IbCastPostReceiveWorkRequestsOnQp(struct ncclIbRecvComm* recvComm, ncclIbQp* dataQp);

#endif // NET_IB_CONNECT_H_
