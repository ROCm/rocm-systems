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
    return rcclPolicyTransportToggleEligible(comm)
             ? RCCL_CONN_IDX_P2P_SHM
             : 1;
  return -1;
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
}

} // namespace

rcclExecutionTransportIntent rcclPolicyTransportIntent(bool p2pDisabled) {
  const char* p2pDisable = ncclGetEnv("NCCL_P2P_DISABLE");
  if (p2pDisable == nullptr || p2pDisable[0] == '\0')
    return RCCL_EXECUTION_TRANSPORT_INTENT_AUTO;
  return p2pDisabled ? RCCL_EXECUTION_TRANSPORT_INTENT_SHM
                     : RCCL_EXECUTION_TRANSPORT_INTENT_IPC;
}

bool rcclPolicyTransportToggleEligible(const ncclComm* comm) {
  return comm != nullptr &&
         rcclPolicyTransportIntent(ncclParamP2pDisable()) ==
           RCCL_EXECUTION_TRANSPORT_INTENT_AUTO &&
         rcclPolicyRuntimeToggleRequested() &&
         ncclParamP2pDisable() == 0 && ncclParamShmDisable() == 0 &&
         comm->nNodes == 1 && comm->nRanks == 8 &&
         IsArchMatch(comm->archName, "gfx1201");
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
  rcclExecutionPolicyProvisioningInput input = {
    comm->archName,
    comm->nNodes,
    comm->nRanks,
    rcclPolicyAvailableTransportMask(comm, p2pDisabled),
    rcclPolicyTransportIntent(p2pDisabled),
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
  for (int connIndex = 0; connIndex < NCCL_MAX_CONNS; connIndex++) {
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
