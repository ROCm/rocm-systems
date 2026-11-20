/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_P2P_RESILIENCY_MSEG_H_
#define NET_IB_P2P_RESILIENCY_MSEG_H_

#include "common_mseg.h"
#include "connect_mseg.h"

enum IbMsegResiliencyDevState {
  // The device is operating normally.
  IbMsegResiliencyDevStateOk = 0,
  // The device encountered an error and and can be tried to be recovered.
  IbMsegResiliencyDevStateError,
  // The device is currently undergoing recovery.
  IbMsegResiliencyDevStateRecoveryInProgress,
  // The device recovery failed.
  IbMsegResiliencyDevStateRecoveryFailed,
  // The device was recovered.
  IbMsegResiliencyDevStateRecovered,
  // The device will not be attempted to be recovered any more.
  IbMsegResiliencyDevStateErrorPermanent
};

struct IbMsegResiliencyDev {
  // Atomic to allow lock-free access from multiple threads (main + recovery).
  std::atomic<IbMsegResiliencyDevState> state;
  // CQ to get CQEs on the sender side for probing operations.
  // Receiver side is not expected to get any CQEs on the this CQ but Verbs
  // requires a CQ to be associated with a QP.
  struct ibv_cq* probingCq;
  int nOutstandingProbes;
  // MR to allow the RDMA Read to write to local memory on the sender side.
  // Note that this memory registration is not required to be communicated to
  // the receiver side as the memory is only used as a target for RDMA Read
  // operations initiated by the sender side.
  struct ibv_mr* probingResultMr;
  // CQ to get CQEs for recovery protocol messages.
  struct ibv_cq* portRecoveryCq;
};

struct IbMsegResiliency {
  // Back pointer to the base communicator.
  struct IbMsegNetCommBase* baseComm;

  bool recoveryEnabled;

  struct IbMsegResiliencyDev devs[NCCL_IB_MAX_DEVS_PER_NIC];
  int ndevs;

  // QPs used for probing of data transfers in case of QP/device failures.
  // Note that the receiver and sender might have different number of devices
  // (asymmetric device configuration), so the number of probing QPs might be
  // configured differently on the receiver and sender and might be even larger
  // than the number of local devices. In case that the number of probing QPs
  // is larger than the number of local devices, some of these QPs are only
  // created to allow the remote side to connect to them and issue RDMA Read
  // operations through them but no local operations are performed on these
  // QPs.
  struct IbMsegQp probingQps[NCCL_IB_MAX_DEVS_PER_NIC];
  int nProbingQps;

  // As long as this variable is true, it means that resiliency operations are
  // in progress.
  bool inProgress;

  // Number of requests that are currently being handled by the resiliency
  // module.
  int outstandingRequests;

  // QPs used for recovery protocol messages.
  struct IbMsegQp portRecoveryQps[NCCL_IB_MAX_DEVS_PER_NIC];
  int nPortRecoveryQps;

  // Number of outstanding devices that are currently undergoing recovery.
  int outstandingRecovery;
};

enum IbMsegResiliencyRequestSendState {
  // The request just encountered an error and waiting for probing to be
  // posted.
  IbMsegResiliencyRequestStatePending = 0,
  // A probe was posted for this request.
  IbMsegResiliencyRequestStateProbePosted,
  // The probe was completed. After this state, the request is either completed
  // or replayed.
  IbMsegResiliencyRequestStateProbeCompleted
};

struct IbMsegResiliencyErrorInfo {
  // The index of the device on which the error occurred.
  int devIndex;
  // The time on which the error occurred. Can be used to verify if sufficient
  // time has elapsed since the error to send the read probe.
  uint64_t time;
};

// Structure to hold information about a send request that encountered an
// error.
struct IbMsegResiliencyRequestSend {
  enum IbMsegResiliencyRequestSendState state;
  struct IbMsegRequest* request;
  struct IbMsegResiliencyErrorInfo errorInfo;
  int failedAttempts;

  // It might be that an old CQE that belongs to a request that was already
  // handled is still in the CQ. To safely ignore such old CQEs, this ever
  // incrementing "generation ID" used to make sure that the CQE belongs to
  // a request that was already handled so the CQE can be ignored.
  uint64_t id;
};

struct IbMsegResiliencyRemoteCompletionRecordsInfo {
  // The address of the completion records structure on the receiver side.
  uint64_t addr;
  // For accessing the completion records structure on the receiver side.
  uint32_t rkey;
};

struct IbMsegResiliencySend {
  struct IbMsegResiliency base;
  // Array of requests that encountered an error. Requests in this array can be
  // waiting for probing and then possibly replay. Any request that encounters
  // an error is added to this array. The request is removed from this array
  // when it is either completed (if the probe shows that the receiver
  // completed the request) or when it is replayed.
  // Note that multi-recv/send requests are considered as a single request.
  // Meaning, if one send request of a multi-send request encounters an error,
  // only the first send request is added here as a "representative" of the
  // whole multi-send request.
  // Note that iterating in order to progress these requests is not efficient
  // but on the other hand, if there are requests here, it means that requests
  // are in a transient state and might be replayed over different devices.
  // At a "steady state" there should be no requests here.
  struct IbMsegResiliencyRequestSend failedRequests[NET_IB_MAX_REQUESTS];

  // Stores the results of probe for this request
  // Note, this memory must be registered with an MR to allow RDMA Read
  // operations to write to it. Although each 1D array in this 2D array
  // "belongs" to a request, a 2D array is allocated to have a contiguous
  // memory allocation, so that the MR registration is easier and can be done
  // *simply* under a single MR (per device).
  bool probingResults[NET_IB_MAX_REQUESTS][NCCL_IB_MAX_QPS];

  // Stores information, per-device, used to access the completion records
  // structure on the receiver side.
  struct IbMsegResiliencyRemoteCompletionRecordsInfo remCmplRecordsInfo[NCCL_IB_MAX_DEVS_PER_NIC];
};

// -----------------------------
// Data path APIs
// -----------------------------

ncclResult_t IbMsegResiliencyRequestIsComplete(struct IbMsegRequest* request, bool* isComplete);

// First checks if the error is recoverable or not. If yes, performs QPs
// replacement on the communicator for all QPs that are associated
// with the given device index and also initiates probing if needed.
// Note: The error might occur on a "new" request or on a request that is
// already waiting for a probe posting/completion or replay is already ongoing.
ncclResult_t IbMsegResiliencyHandleCompletionError(struct IbMsegResiliency* resCtx, struct ibv_wc* wc, int devIndex);

// Progresses all operations on the resiliency context.
ncclResult_t IbMsegResiliencyProgress(struct IbMsegResiliency* resCtx);

// -----------------------------
// Control path APIs
// -----------------------------

// Initializes/Destroys the resiliency context.
ncclResult_t IbMsegResiliencyInit(struct IbMsegNetCommBase* baseComm, struct IbMsegResiliency** resCtx);
ncclResult_t IbMsegResiliencyDestroy(struct IbMsegResiliency** resCtx);

// Initializes/Destroys device-related resources.
ncclResult_t IbMsegResiliencyDevInit(struct IbMsegResiliency* resCtx, uint devIndex, IbMsegDev* ibDev);
ncclResult_t IbMsegResiliencyDevDestroy(struct IbMsegResiliency* resCtx, uint devIndex);

// Gets the size of the CQ that is associated with the data QPs. This CQ size
// accomodates the number of devices that are supported for failover.
ncclResult_t IbMsegResiliencyDataCqSizeGet(struct IbMsegResiliency* resCtx, uint devIndex, int* cqSize);
// Gets the size of the Receive Queue (RQ) that is expected to receive data.
// The RQ size accomodates the number of devices that are supported for
// failover.
ncclResult_t IbMsegResiliencyDataRqSizeGet(struct IbMsegResiliency* resCtx, uint devIndex, uint32_t* rqSize);

// Set the number of local devices and remote devices for the resiliency
// context. This function must be called BEFORE creating the resiliency QPs
ncclResult_t IbMsegResiliencyDeviceNumSet(struct IbMsegResiliency* resCtx, int nLocalDevs, int nRemDevs);

// The local info should be populated by the function with the information of
// the QPs created so it could be passed to the receiver side.
ncclResult_t IbMsegResiliencySenderCreateQps(struct IbMsegResiliency* resCtx,
                                             struct IbMsegResiliencyInfo* localResiliencyInfo);
// The remote info should be used for modifying the QPs required for resiliency
// on the sender side to RTS state.
ncclResult_t IbMsegResiliencySenderQpsToRts(struct IbMsegResiliency* resCtx, struct IbMsegConnectionMetadata* remInfo);
// The local info should be populated with the information of the QPs created
// so it could be passed to the sender side.
ncclResult_t IbMsegResiliencyReceiverQpsCreateToRts(struct IbMsegResiliency* resCtx,
                                                    struct IbMsegConnectionMetadata* remInfo,
                                                    struct IbMsegResiliencyInfo* localResiliencyInfo);

ncclResult_t IbMsegResiliencyQpsReconfigure(struct IbMsegResiliency* resCtx, int devIndex, bool* success);

// Reconfigure a single QP through RESET -> INIT -> ECE -> RTR -> RTS.
ncclResult_t IbMsegResiliencyQpReconfigure(struct IbMsegResiliency* resCtx, struct IbMsegQp* qp,
                                           struct IbMsegNetCommDevBase* devBase, int devIndex, bool* success);

ncclResult_t IbMsegResiliencyClose(struct IbMsegResiliency* resCtx);

// Allow resiliency context to reuse the memory registration
// Receiver side registers the memory for completion records and sends the
// memory info to the sender side. This function should be called on the sender
// side to allow the resiliency context to access the completion records
// structure on the receiver side.
ncclResult_t IbMsegResiliencyRemoteCompletionRecordsSet(struct IbMsegResiliency* resCtx, uint32_t cmplsRecordsRkey,
                                                        uint64_t cmplsRecordsAddr, uint devIndex);

#endif // NET_IB_P2P_RESILIENCY_MSEG_H_
