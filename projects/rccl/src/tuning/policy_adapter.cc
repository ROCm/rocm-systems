/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "policy_adapter.h"

#include "archinfo.h"
#include "comm.h"
#include "debug.h"
#include "device.h"
#include "param.h"
#include "transport.h"

#include <algorithm>
#include <stdint.h>
#include <string.h>

extern int64_t ncclParamMinNchannels();
extern int64_t ncclParamMaxNchannels();
extern int64_t ncclParamP2pDisable();
extern int64_t ncclParamP2pReadEnable();
extern int64_t ncclParamShmDisable();

namespace {

rcclExecutionTransport rcclConfiguredPolicyTransport(bool p2pDisabled) {
  if (!p2pDisabled) return RCCL_EXECUTION_TRANSPORT_IPC;
  if (ncclParamShmDisable() == 0) return RCCL_EXECUTION_TRANSPORT_SHM;
  return RCCL_EXECUTION_TRANSPORT_UNKNOWN;
}

bool rcclPolicyRuntimeToggleRequested() {
  const char* toggle = ncclGetEnv("RCCL_RUNTIME_TRANSPORT_TOGGLE");
  return toggle != nullptr && strcmp(toggle, "1") == 0;
}

rcclExecutionPolicyResolution rcclInvalidPolicyResolution() {
  rcclExecutionPolicyResolution resolution{};
  resolution.status = RCCL_EXECUTION_POLICY_INVALID_INPUT;
  resolution.validationError = RCCL_EXECUTION_POLICY_VALIDATION_NONE;
  resolution.selectedRuleId = RCCL_EXECUTION_POLICY_NO_RULE;
  resolution.firstRejectedRuleId = RCCL_EXECUTION_POLICY_NO_RULE;
  resolution.firstRejectedError = RCCL_EXECUTION_POLICY_VALIDATION_NONE;
  rcclDefaultCollectiveExecutionPolicy(&resolution.policy);
  return resolution;
}

struct ncclTransportComm* rcclPolicyTransportComm(
  rcclExecutionTransport transport, bool isSendNotRecv) {
  switch (transport) {
  case RCCL_EXECUTION_TRANSPORT_IPC:
    return isSendNotRecv ? &p2pTransport.send : &p2pTransport.recv;
  case RCCL_EXECUTION_TRANSPORT_SHM:
    return isSendNotRecv ? &shmTransport.send : &shmTransport.recv;
  default:
    return nullptr;
  }
}

bool rcclConnectorUsesPolicyTransport(
  const ncclConnector& connector, rcclExecutionTransport transport,
  bool isSendNotRecv, bool requireConnected) {
  return (!requireConnected || connector.connected != 0) &&
         connector.transportComm ==
           rcclPolicyTransportComm(transport, isSendNotRecv);
}

int rcclPlannedP2pConnectorIndex(
  const ncclComm* comm, rcclExecutionTransport transport) {
  if (transport == RCCL_EXECUTION_TRANSPORT_IPC) return 1;
  if (transport == RCCL_EXECUTION_TRANSPORT_SHM)
    return rcclPolicyCanonicalShmEligible(comm)
             ? RCCL_CONN_IDX_P2P_SHM
             : 1;
  return -1;
}

// Default connectors carry the configured transport and run every shape.
// Another transport runs only on its dedicated slot, which the collective
// connector adapter materializes for Ring/Simple.
bool rcclPolicyTransportHasDedicatedConnector(
  const ncclComm* comm, rcclExecutionTransport transport) {
  return transport == RCCL_EXECUTION_TRANSPORT_SHM &&
         rcclPolicyCanonicalShmEligible(comm);
}

void rcclSetCollectiveTransportCapabilities(
  const ncclComm* comm, rcclExecutionPolicyValidationContext* validation) {
  rcclExecutionTransport primary =
    rcclConfiguredPolicyTransport(ncclParamP2pDisable());
  validation->validateTransportCapabilities = true;
  for (int t = 0; t < RCCL_EXECUTION_TRANSPORT_COUNT; t++) {
    rcclExecutionTransport transport = static_cast<rcclExecutionTransport>(t);
    bool isPrimary =
      transport != RCCL_EXECUTION_TRANSPORT_UNKNOWN && transport == primary;
    bool dedicated = rcclPolicyTransportHasDedicatedConnector(comm, transport);
    for (int algorithm = 0; algorithm < NCCL_NUM_ALGORITHMS; algorithm++) {
      for (int protocol = 0; protocol < NCCL_NUM_PROTOCOLS; protocol++) {
        validation->transportAlgoProtoAvailable[t][algorithm][protocol] =
          validation->algoProtoAvailable[algorithm][protocol] &&
          (isPrimary || (dedicated && algorithm == NCCL_ALGO_RING &&
                         protocol == NCCL_PROTO_SIMPLE));
      }
    }
  }
}

// Read/write transfer selection applies to IPC connectors only.
void rcclSetP2pTransportCapabilities(
  const ncclComm* comm, bool p2pDisabled,
  rcclExecutionPolicyValidationContext* validation) {
  rcclExecutionTransport primary = rcclConfiguredPolicyTransport(p2pDisabled);
  uint32_t ipcTransfers = uint32_t{1} << RCCL_P2P_TRANSFER_WRITE;
  if (ncclParamP2pReadEnable() != 0)
    ipcTransfers |= uint32_t{1} << RCCL_P2P_TRANSFER_READ;
  validation->validateTransportCapabilities = true;
  for (int t = 0; t < RCCL_EXECUTION_TRANSPORT_COUNT; t++)
    validation->transportP2pTransferMask[t] = 0;
  if (primary == RCCL_EXECUTION_TRANSPORT_IPC ||
      rcclPolicyTransportHasDedicatedConnector(
        comm, RCCL_EXECUTION_TRANSPORT_IPC))
    validation->transportP2pTransferMask[RCCL_EXECUTION_TRANSPORT_IPC] =
      ipcTransfers;
}

void rcclBuildCollectiveExecutionPolicyValidationContext(
  const ncclComm* comm, const ncclTaskColl* info,
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS],
  rcclExecutionPolicyValidationContext* validation) {
  rcclDefaultExecutionPolicyValidationContext(validation);

  int minChannels = 1;
  if (info->minCTAs > 0)
    minChannels = std::max(minChannels, info->minCTAs);
  int userMinChannels = ncclParamMinNchannels();
  if (userMinChannels > 0)
    minChannels = std::max(minChannels, userMinChannels);

  int maxChannels = comm->nChannels;
  if (info->maxCTAs > 0)
    maxChannels = std::min(maxChannels, info->maxCTAs);
  int userMaxChannels = ncclParamMaxNchannels();
  if (userMaxChannels > 0)
    maxChannels = std::min(maxChannels, userMaxChannels);

  validation->minChannels = minChannels;
  validation->maxChannels = maxChannels;
  validation->baselineAlgorithm = info->algorithm;
  validation->baselineProtocol = info->protocol;
  validation->validateAlgoProto = true;
  validation->validateTransport = true;
  validation->transportMask =
    rcclPolicyAvailableTransportMask(comm, ncclParamP2pDisable());
  for (int algorithm = 0; algorithm < NCCL_NUM_ALGORITHMS; algorithm++) {
    bool algorithmAllowedByTask =
      info->algMask == 0 ||
      (info->algMask & (uint64_t{1} << algorithm)) != 0;
    for (int protocol = 0; protocol < NCCL_NUM_PROTOCOLS; protocol++) {
      validation->algoProtoAvailable[algorithm][protocol] =
        algorithmAllowedByTask && table[algorithm][protocol] >= 0.0f;
    }
  }
  rcclSetCollectiveTransportCapabilities(comm, validation);
}

} // namespace

rcclExecutionTransportIntent rcclPolicyTransportIntent(bool p2pDisabled) {
  const char* p2pDisable = ncclGetEnv("NCCL_P2P_DISABLE");
  if (p2pDisable == nullptr || p2pDisable[0] == '\0')
    return RCCL_EXECUTION_TRANSPORT_INTENT_AUTO;
  return p2pDisabled ? RCCL_EXECUTION_TRANSPORT_INTENT_SHM
                     : RCCL_EXECUTION_TRANSPORT_INTENT_IPC;
}

bool rcclPolicyCanonicalTransportEligible(const ncclComm* comm) {
  return comm != nullptr && rcclPolicyRuntimeToggleRequested() &&
         comm->nNodes == 1 && comm->nRanks == 8 &&
         IsArchMatch(comm->archName, "gfx1201");
}

bool rcclPolicyTransportToggleEligible(const ncclComm* comm) {
  return rcclPolicyCanonicalTransportEligible(comm) &&
         rcclPolicyTransportIntent(ncclParamP2pDisable()) ==
           RCCL_EXECUTION_TRANSPORT_INTENT_AUTO &&
         ncclParamP2pDisable() == 0 && ncclParamShmDisable() == 0;
}

bool rcclPolicyCanonicalShmEligible(const ncclComm* comm) {
  return rcclPolicyCanonicalTransportEligible(comm) &&
         ncclParamShmDisable() == 0 &&
         rcclPolicyTransportIntent(ncclParamP2pDisable()) !=
           RCCL_EXECUTION_TRANSPORT_INTENT_IPC;
}

uint32_t rcclPolicyAvailableTransportMask(
  const ncclComm* comm, bool p2pDisabled) {
  rcclExecutionTransport configured =
    rcclConfiguredPolicyTransport(p2pDisabled);
  uint32_t mask =
    configured == RCCL_EXECUTION_TRANSPORT_UNKNOWN
      ? 0
      : rcclExecutionTransportBit(configured);
  if (rcclPolicyTransportIntent(p2pDisabled) ==
        RCCL_EXECUTION_TRANSPORT_INTENT_AUTO &&
      rcclPolicyTransportToggleEligible(comm)) {
    mask |= rcclExecutionTransportBit(RCCL_EXECUTION_TRANSPORT_IPC) |
            rcclExecutionTransportBit(RCCL_EXECUTION_TRANSPORT_SHM);
  }
  return mask;
}

bool rcclCollectiveTaskBuffersOverlap(
  const ncclComm* comm, const ncclTaskColl* task) {
  if (comm == nullptr || task == nullptr || task->sendbuff == nullptr ||
      task->recvbuff == nullptr || comm->nRanks <= 0)
    return false;

  size_t sendCount = task->count;
  size_t recvCount = task->count;
  if (task->func == ncclFuncAllGather) {
    recvCount = task->count > SIZE_MAX / static_cast<size_t>(comm->nRanks)
                  ? SIZE_MAX
                  : task->count * static_cast<size_t>(comm->nRanks);
  } else if (task->func == ncclFuncReduceScatter) {
    sendCount = task->count > SIZE_MAX / static_cast<size_t>(comm->nRanks)
                  ? SIZE_MAX
                  : task->count * static_cast<size_t>(comm->nRanks);
  }

  int typeSize = ncclTypeSize(task->datatype);
  if (typeSize <= 0) return false;
  size_t elementBytes = static_cast<size_t>(typeSize);
  size_t sendBytes =
    sendCount > SIZE_MAX / elementBytes ? SIZE_MAX : sendCount * elementBytes;
  size_t recvBytes =
    recvCount > SIZE_MAX / elementBytes ? SIZE_MAX : recvCount * elementBytes;
  return rcclBuffersOverlap(
    task->sendbuff, sendBytes, task->recvbuff, recvBytes);
}

int rcclGetCollectiveExecutionPolicyRequiredChannels(
  ncclComm* comm, bool p2pDisabled) {
  if (comm == nullptr) return 0;
  // Canonical mode provisions for every toggle transport regardless of
  // intent, so forced modes share AUTO's channel pools.
  bool canonical =
    rcclPolicyCanonicalTransportEligible(comm) && ncclParamShmDisable() == 0;
  rcclExecutionPolicyProvisioningInput input = {
    comm->archName,
    comm->nNodes,
    comm->nRanks,
    canonical
      ? rcclExecutionTransportBit(RCCL_EXECUTION_TRANSPORT_IPC) |
          rcclExecutionTransportBit(RCCL_EXECUTION_TRANSPORT_SHM)
      : rcclPolicyAvailableTransportMask(comm, p2pDisabled),
    canonical ? RCCL_EXECUTION_TRANSPORT_INTENT_AUTO
              : rcclPolicyTransportIntent(p2pDisabled),
  };
  return rcclResolveExecutionPolicyRequiredChannels(&input);
}

rcclExecutionPolicyResolution rcclGetCollectiveExecutionPolicyWithValidation(
  const ncclComm* comm, rcclExecutionScope scope, ncclFunc_t collType,
  size_t dataSize, bool p2pDisabled, bool inPlace,
  const rcclExecutionPolicyValidationContext* validation) {
  if (comm == nullptr) return rcclInvalidPolicyResolution();

  rcclExecutionPolicyValidationContext transportValidation;
  if (validation == nullptr) {
    rcclDefaultExecutionPolicyValidationContext(&transportValidation);
    transportValidation.validateTransport = true;
    transportValidation.transportMask =
      rcclPolicyAvailableTransportMask(comm, p2pDisabled);
    validation = &transportValidation;
  }
  const rcclCollectivePolicyInput input = {
    scope,
    comm->archName,
    collType,
    dataSize,
    comm->nNodes,
    comm->nRanks,
    scope == RCCL_EXECUTION_SCOPE_P2P
      ? comm->p2pnChannels
      : comm->nChannels,
    rcclPolicyAvailableTransportMask(comm, p2pDisabled),
    inPlace,
    rcclPolicyTransportIntent(p2pDisabled),
  };
  return rcclResolveCollectiveExecutionPolicyWithValidation(
    &input, validation);
}

bool rcclGetCollectiveExecutionPolicy(
  const ncclComm* comm, rcclExecutionScope scope, ncclFunc_t collType,
  size_t dataSize, bool p2pDisabled, bool inPlace,
  rcclCollectiveExecutionPolicy* policy) {
  if (policy == nullptr) return false;
  rcclExecutionPolicyResolution resolution =
    rcclGetCollectiveExecutionPolicyWithValidation(
      comm, scope, collType, dataSize, p2pDisabled, inPlace, nullptr);
  *policy = resolution.policy;
  return resolution.status == RCCL_EXECUTION_POLICY_SELECTED;
}

bool rcclExecutionPolicyForP2pTask(
  ncclComm* comm, ncclTaskP2p* task, bool p2pDisabled,
  rcclCollectiveExecutionPolicy* policy) {
  if (comm == nullptr || task == nullptr || policy == nullptr ||
      comm->nRanks <= 0)
    return false;
  size_t aggregateBytes =
    task->bytes > SIZE_MAX / comm->nRanks
      ? SIZE_MAX
      : task->bytes * static_cast<size_t>(comm->nRanks);

  rcclExecutionPolicyValidationContext validation;
  rcclDefaultExecutionPolicyValidationContext(&validation);
  validation.minChannels = 1;
  int userMinChannels = ncclParamMinNchannels();
  if (userMinChannels > 0)
    validation.minChannels =
      std::max(validation.minChannels, userMinChannels);
  validation.maxChannels = comm->p2pnChannels;
  int userMaxChannels = ncclParamMaxNchannels();
  if (userMaxChannels > 0)
    validation.maxChannels =
      std::min(validation.maxChannels, userMaxChannels);

  uint32_t protocolMask = 0;
  if (task->collAPI >= 0 && task->collAPI < NCCL_NUM_FUNCTIONS) {
    for (int protocol = 0; protocol < NCCL_NUM_PROTOCOLS; protocol++) {
      for (int algorithm = 0; algorithm < NCCL_NUM_ALGORITHMS; algorithm++) {
        if (comm->bandwidths[task->collAPI][algorithm][protocol] > 0.0f) {
          protocolMask |= uint32_t{1} << protocol;
          break;
        }
      }
    }
  }
  validation.validateP2pProtocol = protocolMask != 0;
  validation.p2pProtocolMask = protocolMask;
  if (ncclParamP2pReadEnable() == 0) {
    validation.validateP2pTransfer = true;
    validation.p2pTransferMask =
      uint32_t{1} << RCCL_P2P_TRANSFER_WRITE;
  }
  validation.validateTransport = true;
  validation.transportMask =
    rcclPolicyAvailableTransportMask(comm, p2pDisabled);
  rcclSetP2pTransportCapabilities(comm, p2pDisabled, &validation);

  rcclExecutionPolicyResolution resolution =
    rcclGetCollectiveExecutionPolicyWithValidation(
      comm, RCCL_EXECUTION_SCOPE_P2P, task->collAPI, aggregateBytes,
      p2pDisabled, task->inPlace, &validation);
  *policy = resolution.policy;
  if (resolution.status != RCCL_EXECUTION_POLICY_SELECTED &&
      (resolution.status == RCCL_EXECUTION_POLICY_VALIDATION_FAILED ||
       resolution.status == RCCL_EXECUTION_POLICY_INVALID_INPUT)) {
    INFO(NCCL_COLL,
         "RCCL P2P execution policy rejected: coll=%d bytes=%zu status=%d "
         "rule=%u error=%d",
         (int)task->collAPI, aggregateBytes, (int)resolution.status,
         (unsigned)resolution.firstRejectedRuleId,
         (int)(resolution.status == RCCL_EXECUTION_POLICY_INVALID_INPUT
                 ? resolution.validationError
                 : resolution.firstRejectedError));
  }
  return resolution.status == RCCL_EXECUTION_POLICY_SELECTED;
}

bool rcclApplyCollectiveExecutionPolicy(
  ncclComm* comm, ncclTaskColl* info, size_t dataSize,
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS],
  int* policyChannels) {
  if (comm == nullptr || info == nullptr || table == nullptr ||
      policyChannels == nullptr)
    return false;

  info->executionTransport = RCCL_EXECUTION_TRANSPORT_UNKNOWN;
  rcclExecutionPolicyValidationContext validation;
  rcclBuildCollectiveExecutionPolicyValidationContext(
    comm, info, table, &validation);
  rcclExecutionPolicyResolution resolution =
    rcclGetCollectiveExecutionPolicyWithValidation(
      comm, RCCL_EXECUTION_SCOPE_COLLECTIVE, info->func, dataSize,
      ncclParamP2pDisable(), rcclCollectiveTaskBuffersOverlap(comm, info),
      &validation);
  if (resolution.status != RCCL_EXECUTION_POLICY_SELECTED) {
    if (resolution.status == RCCL_EXECUTION_POLICY_VALIDATION_FAILED ||
        resolution.status == RCCL_EXECUTION_POLICY_INVALID_INPUT) {
      INFO(NCCL_TUNING,
           "RCCL collective execution policy rejected: coll=%d bytes=%zu "
           "status=%d rule=%u error=%d",
           (int)info->func, dataSize, (int)resolution.status,
           (unsigned)resolution.firstRejectedRuleId,
           (int)(resolution.status == RCCL_EXECUTION_POLICY_INVALID_INPUT
                   ? resolution.validationError
                   : resolution.firstRejectedError));
    }
    return false;
  }
  const rcclCollectiveExecutionPolicy& policy = resolution.policy;

  int algorithm =
    policy.algorithm >= 0 ? policy.algorithm : info->algorithm;
  int protocol =
    policy.protocol >= 0 ? policy.protocol : info->protocol;
  if (policy.algorithm >= 0 || policy.protocol >= 0) {
    info->algorithm = algorithm;
    info->protocol = protocol;
  }
  info->executionTransport = policy.transport;

  *policyChannels = policy.nChannels;
  INFO(NCCL_TUNING,
       "RCCL collective execution policy: arch=%s coll=%d bytes=%zu rule=%u "
       "channels=%d algorithm=%d protocol=%d transport=%d",
       comm->archName, (int)info->func, dataSize,
       (unsigned)resolution.selectedRuleId, policy.nChannels,
       policy.algorithm, policy.protocol, (int)policy.transport);
  return true;
}

int rcclPolicyCollectiveConnectorIndex(
  const ncclComm* comm, const ncclTaskColl* info,
  int channelId, size_t nBytes) {
  if (comm == nullptr || info == nullptr) return -1;
  if (channelId < 0 || channelId >= comm->nChannels ||
      info->algorithm != NCCL_ALGO_RING ||
      info->protocol != NCCL_PROTO_SIMPLE || info->regBufType != 0)
    return -1;

  rcclExecutionTransport transport =
    static_cast<rcclExecutionTransport>(info->executionTransport);
  if (rcclPolicyTransportComm(transport, true) == nullptr) return -1;

  const ncclChannel& channel = comm->channels[channelId];
  if (channel.ring.next < 0 || channel.ring.next >= comm->nRanks ||
      channel.ring.prev < 0 || channel.ring.prev >= comm->nRanks)
    return -1;
  bool canonicalShm = transport == RCCL_EXECUTION_TRANSPORT_SHM &&
                      rcclPolicyCanonicalShmEligible(comm);
  for (int connIndex = canonicalShm ? RCCL_CONN_IDX_COLL_SHM : 0;
       connIndex < (canonicalShm ? RCCL_CONN_IDX_COLL_SHM + 1 : NCCL_MAX_CONNS);
       connIndex++) {
    const ncclConnector& send =
      channel.peers[channel.ring.next]->send[connIndex];
    const ncclConnector& recv =
      channel.peers[channel.ring.prev]->recv[connIndex];
    if (rcclConnectorUsesPolicyTransport(
          send, transport, /*isSendNotRecv=*/true, /*requireConnected=*/true) &&
        rcclConnectorUsesPolicyTransport(
          recv, transport, /*isSendNotRecv=*/false, /*requireConnected=*/true)) {
      if (comm->rank == 0) {
        INFO(NCCL_TUNING,
             "RCCL execution-policy transport: bytes=%zu transport=%s "
             "connIndex=%d",
             nBytes,
             transport == RCCL_EXECUTION_TRANSPORT_SHM ? "SHM" : "IPC",
             connIndex);
      }
      return connIndex;
    }
  }
  if (comm->rank == 0) {
    INFO(NCCL_TUNING,
         "RCCL connector adapter found no connected transport=%d "
         "for channel=%d",
         (int)transport, channelId);
  }
  return -1;
}

int rcclPolicyP2pConnectorIndex(
  const ncclComm* comm, int channelId, int peer, bool isSendNotRecv,
  rcclExecutionTransport transport) {
  if (comm == nullptr || channelId < 0 || channelId >= MAXCHANNELS ||
      peer < 0 || peer >= comm->nRanks ||
      rcclPolicyTransportComm(transport, isSendNotRecv) == nullptr)
    return -1;

  if (transport == RCCL_EXECUTION_TRANSPORT_SHM &&
      rcclPolicyCanonicalShmEligible(comm))
    return RCCL_CONN_IDX_P2P_SHM;

  const ncclChannelPeer* channelPeer = comm->channels[channelId].peers[peer];
  if (channelPeer != nullptr) {
    for (int connIndex = 1; connIndex < NCCL_MAX_CONNS; connIndex++) {
      const ncclConnector& connector =
        isSendNotRecv ? channelPeer->send[connIndex]
                      : channelPeer->recv[connIndex];
      if (rcclConnectorUsesPolicyTransport(
            connector, transport, isSendNotRecv, /*requireConnected=*/true))
        return connIndex;
    }
  }
  return rcclPlannedP2pConnectorIndex(comm, transport);
}

// IPC registration bypasses proxy connectors and is incompatible with the
// policy-selected SHM path.
bool rcclPolicyCollectiveAllowsRegistration(const ncclTaskColl* info) {
  return info == nullptr ||
         info->executionTransport != RCCL_EXECUTION_TRANSPORT_SHM;
}

bool rcclPolicyAllToAllUsesSendRecvPath(
  const ncclComm* comm, size_t aggregateBytes, bool inPlace) {
  rcclCollectiveExecutionPolicy policy;
  return rcclGetCollectiveExecutionPolicy(
           comm, RCCL_EXECUTION_SCOPE_P2P, ncclFuncAlltoAll, aggregateBytes,
           ncclParamP2pDisable(), inPlace, &policy) &&
         policy.path == RCCL_P2P_PATH_SENDRECV;
}

void rcclPolicyResolveP2pTask(ncclComm* comm, ncclTaskP2p* task) {
  task->executionPolicyMatched = rcclExecutionPolicyForP2pTask(
    comm, task, ncclParamP2pDisable(), &task->executionPolicy);
  if (!task->executionPolicyMatched)
    rcclDefaultCollectiveExecutionPolicy(&task->executionPolicy);
}

void rcclPolicyPlanP2pWork(
  const ncclComm* comm, ncclTaskP2p* const tasks[2], bool logSelection,
  rcclP2pPolicyWorkPlan* plan) {
  bool hasTask = false;
  bool uniformSendRecv = true;
  int activeChannels = comm->p2pnChannels;
  for (int dir = 0; dir < 2; dir++) {
    plan->matched[dir] = tasks[dir] != nullptr && tasks[dir]->executionPolicyMatched;
    if (plan->matched[dir]) plan->policy[dir] = tasks[dir]->executionPolicy;
    else rcclDefaultCollectiveExecutionPolicy(&plan->policy[dir]);
    plan->channels[dir] = plan->matched[dir] ? plan->policy[dir].nChannels : -1;
    if (tasks[dir] == nullptr) continue;
    hasTask = true;
    // A work item has one channel namespace shared by send and receive. Never
    // cap a mixed item containing an ordinary grouped P2P operation.
    if (!plan->matched[dir] || plan->policy[dir].path != RCCL_P2P_PATH_SENDRECV) {
      uniformSendRecv = false;
    } else if (plan->policy[dir].nChannels > 0) {
      activeChannels = std::min(activeChannels, plan->policy[dir].nChannels);
    }
  }
  plan->activeChannels =
    hasTask && uniformSendRecv ? activeChannels : comm->p2pnChannels;

  if (!logSelection || (!plan->matched[0] && !plan->matched[1])) return;
  int dir = plan->matched[0] ? 0 : 1;
  const rcclCollectiveExecutionPolicy& policy = plan->policy[dir];
  size_t taskBytes = tasks[dir]->bytes;
  size_t aggregateBytes =
    taskBytes > SIZE_MAX / comm->nRanks ? SIZE_MAX : taskBytes * static_cast<size_t>(comm->nRanks);
  const char* transferName = policy.transferMode == RCCL_P2P_TRANSFER_READ    ? "read"
                             : policy.transferMode == RCCL_P2P_TRANSFER_WRITE ? "write"
                                                                             : "auto";
  INFO(NCCL_COLL,
       "RCCL P2P execution policy: arch=%s coll=%d aggregateBytes=%zu inPlace=%d channels=%d->%d path=SendRecv "
       "protocol=%s transfer=%s transport=%d",
       comm->archName, (int)tasks[dir]->collAPI, aggregateBytes, (int)tasks[dir]->inPlace,
       comm->p2pnChannels, plan->activeChannels,
       policy.protocol >= 0 ? ncclProtoStr[policy.protocol] : "auto", transferName,
       (int)policy.transport);
}

int rcclPolicyP2pWorkConnectorIndex(
  const ncclComm* comm, const rcclP2pPolicyWorkPlan* plan, int dir,
  int channelId, int peer, int defaultConnIndex) {
  if (!plan->matched[dir]) return defaultConnIndex;
  const rcclCollectiveExecutionPolicy& policy = plan->policy[dir];
  bool isSendNotRecv = dir != 0;
  int connIndex = defaultConnIndex;
  if (policy.transport != RCCL_EXECUTION_TRANSPORT_UNKNOWN) {
    int transportConnIndex =
      rcclPolicyP2pConnectorIndex(comm, channelId, peer, isSendNotRecv, policy.transport);
    if (transportConnIndex >= 0) connIndex = transportConnIndex;
  }
  if (policy.transport == RCCL_EXECUTION_TRANSPORT_SHM ||
      policy.transferMode == RCCL_P2P_TRANSFER_AUTO)
    return connIndex;

  int requiredFlag =
    policy.transferMode == RCCL_P2P_TRANSFER_READ ? NCCL_P2P_READ : NCCL_P2P_WRITE;
  constexpr int candidateConnIndices[] = {1, RCCL_CONN_IDX_P2P_ALT};
  for (int candidate : candidateConnIndices) {
    if (candidate == RCCL_CONN_IDX_P2P_ALT && comm->p2pNet) continue;
    const ncclChannelPeer* channelPeer = comm->channels[channelId].peers[peer];
    const ncclConnector& conn =
      isSendNotRecv ? channelPeer->send[candidate] : channelPeer->recv[candidate];
    if (conn.connected && (conn.conn.flags & requiredFlag)) return candidate;
  }
  INFO(NCCL_COLL,
       "RCCL P2P execution policy transfer mode unavailable; preserving default connector "
       "(dir=%s requested=%s)",
       isSendNotRecv ? "send" : "recv",
       policy.transferMode == RCCL_P2P_TRANSFER_READ ? "read" : "write");
  return connIndex;
}

int rcclPolicyP2pWorkProtocol(
  const rcclP2pPolicyWorkPlan* plan, int dir, int selectedProtocol,
  bool useLL128, bool latencyBufferAvailable) {
  int requestedProtocol = plan->matched[dir] ? plan->policy[dir].protocol : -1;
  if (requestedProtocol < 0) return selectedProtocol;
  if (requestedProtocol == NCCL_PROTO_SIMPLE) return NCCL_PROTO_SIMPLE;
  if (requestedProtocol == NCCL_PROTO_LL && !useLL128 && latencyBufferAvailable) return NCCL_PROTO_LL;
  if (requestedProtocol == NCCL_PROTO_LL128 && useLL128 && latencyBufferAvailable) return NCCL_PROTO_LL128;
  return selectedProtocol;
}

bool rcclPolicyP2pWorkAllowsRegistration(
  const rcclP2pPolicyWorkPlan* plan, int dir) {
  return !plan->matched[dir] ||
         plan->policy[dir].transport != RCCL_EXECUTION_TRANSPORT_SHM;
}

bool rcclPolicyP2pTaskAllowsRegistration(const ncclTaskP2p* task) {
  return !task->executionPolicyMatched ||
         task->executionPolicy.transport != RCCL_EXECUTION_TRANSPORT_SHM;
}

bool rcclPolicyP2pPreconnect(
  const ncclComm* comm, const ncclTaskP2p* task, rcclP2pPolicyPreconnect* preconnect) {
  preconnect->nChannels = comm->p2pnChannels;
  preconnect->nChannelsPerPeer = comm->p2pnChannelsPerPeer;
  preconnect->nConnIndices = 0;
  if (!task->executionPolicyMatched) return false;
  const rcclCollectiveExecutionPolicy& policy = task->executionPolicy;

  bool usesShm = policy.transport == RCCL_EXECUTION_TRANSPORT_SHM;
  if (policy.nChannels <= 0 && policy.transferMode == RCCL_P2P_TRANSFER_AUTO && !usesShm)
    return true;
  if (policy.nChannels > 0)
    preconnect->nChannels = std::min(comm->p2pnChannels, policy.nChannels);
  preconnect->nChannelsPerPeer =
    std::min(comm->p2pnChannelsPerPeer, preconnect->nChannels);
  int transportConnIndex = rcclPlannedP2pConnectorIndex(comm, policy.transport);
  if (usesShm && transportConnIndex == RCCL_CONN_IDX_P2P_SHM) {
    preconnect->connIndices[preconnect->nConnIndices++] = RCCL_CONN_IDX_P2P_SHM;
    return true;
  }
  preconnect->connIndices[preconnect->nConnIndices++] = 1;
  if (!usesShm && policy.transferMode != RCCL_P2P_TRANSFER_AUTO && !comm->p2pNet)
    preconnect->connIndices[preconnect->nConnIndices++] = RCCL_CONN_IDX_P2P_ALT;
  return true;
}
