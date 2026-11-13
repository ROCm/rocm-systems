/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "policy_adapter.h"

#include "archinfo.h"
#include "channel.h"
#include "comm.h"
#include "debug.h"
#include "device.h"
#include "param.h"
#include "transport.h"

#include <algorithm>
#include <atomic>
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

// Read once per process, like NCCL_PARAM: the enqueue path queries these for
// every task and channel. -1 means not yet read.
std::atomic<int> rcclPolicyToggleRequestedCache{-1};
std::atomic<int> rcclPolicyP2pDisableSetCache{-1};

bool rcclPolicyP2pDisableSet() {
  int cached = rcclPolicyP2pDisableSetCache.load(std::memory_order_relaxed);
  if (cached < 0) {
    const char* p2pDisable = ncclGetEnv("NCCL_P2P_DISABLE");
    cached = p2pDisable != nullptr && p2pDisable[0] != '\0';
    rcclPolicyP2pDisableSetCache.store(cached, std::memory_order_relaxed);
  }
  return cached != 0;
}

rcclExecutionPolicyResolution rcclPolicyResolutionWithStatus(rcclExecutionPolicyResolutionStatus status) {
  rcclExecutionPolicyResolution resolution;
  rcclInitExecutionPolicyResolution(&resolution, status);
  return resolution;
}

void rcclLogRejectedPolicy(uint64_t subsystem, const char* scope, ncclFunc_t collType, size_t dataSize,
                           const rcclExecutionPolicyResolution& resolution) {
  if (resolution.status != RCCL_EXECUTION_POLICY_VALIDATION_FAILED &&
      resolution.status != RCCL_EXECUTION_POLICY_INVALID_INPUT)
    return;
  INFO(subsystem, "RCCL %s execution policy rejected: coll=%d bytes=%zu status=%d rule=%u error=%d", scope,
       (int)collType, dataSize, (int)resolution.status, (unsigned)resolution.firstRejectedRuleId,
       (int)(resolution.status == RCCL_EXECUTION_POLICY_INVALID_INPUT ? resolution.validationError :
                                                                        resolution.firstRejectedError));
}

size_t rcclP2pAggregateBytes(const ncclComm* comm, size_t taskBytes) {
  return rcclSaturatingMultiply(taskBytes, static_cast<size_t>(std::max(comm->nRanks, 0)));
}

struct ncclTransportComm* rcclPolicyTransportComm(rcclExecutionTransport transport, bool isSendNotRecv) {
  switch (transport) {
  case RCCL_EXECUTION_TRANSPORT_IPC:
    return isSendNotRecv ? &p2pTransport.send : &p2pTransport.recv;
  case RCCL_EXECUTION_TRANSPORT_SHM:
    return isSendNotRecv ? &shmTransport.send : &shmTransport.recv;
  default:
    return nullptr;
  }
}

bool rcclConnectorUsesPolicyTransport(const ncclConnector& connector, rcclExecutionTransport transport,
                                      bool isSendNotRecv, bool requireConnected) {
  return (!requireConnected || connector.connected != 0) &&
         connector.transportComm == rcclPolicyTransportComm(transport, isSendNotRecv);
}

int rcclPlannedP2pConnectorIndex(const ncclComm* comm, rcclExecutionTransport transport) {
  if (transport == RCCL_EXECUTION_TRANSPORT_IPC) return 1;
  if (transport == RCCL_EXECUTION_TRANSPORT_SHM)
    return rcclPolicyCanonicalShmEligible(comm) ? RCCL_CONN_IDX_P2P_SHM : 1;
  return -1;
}

// Default connectors carry the configured transport and run every shape.
// Another transport runs only on its dedicated slot, which the collective
// connector adapter materializes for Ring/Simple.
bool rcclPolicyTransportHasDedicatedConnector(const ncclComm* comm, rcclExecutionTransport transport) {
  return transport == RCCL_EXECUTION_TRANSPORT_SHM && rcclPolicyCanonicalShmEligible(comm);
}

void rcclSetCollectiveTransportCapabilities(const ncclComm* comm, rcclExecutionPolicyValidationContext* validation) {
  rcclExecutionTransport primary = rcclConfiguredPolicyTransport(ncclParamP2pDisable());
  validation->validateTransportCapabilities = true;
  for (int t = 0; t < RCCL_EXECUTION_TRANSPORT_COUNT; t++) {
    rcclExecutionTransport transport = static_cast<rcclExecutionTransport>(t);
    bool isPrimary = transport != RCCL_EXECUTION_TRANSPORT_UNKNOWN && transport == primary;
    bool dedicated = rcclPolicyTransportHasDedicatedConnector(comm, transport);
    for (int algorithm = 0; algorithm < NCCL_NUM_ALGORITHMS; algorithm++) {
      for (int protocol = 0; protocol < NCCL_NUM_PROTOCOLS; protocol++) {
        validation->transportAlgoProtoAvailable[t][algorithm][protocol] =
          validation->algoProtoAvailable[algorithm][protocol] &&
          (isPrimary || (dedicated && algorithm == NCCL_ALGO_RING && protocol == NCCL_PROTO_SIMPLE));
      }
    }
  }
}

// Read/write transfer selection applies to IPC connectors only.
void rcclSetP2pTransportCapabilities(const ncclComm* comm, bool p2pDisabled,
                                     rcclExecutionPolicyValidationContext* validation) {
  rcclExecutionTransport primary = rcclConfiguredPolicyTransport(p2pDisabled);
  uint32_t ipcTransfers = uint32_t{1} << RCCL_P2P_TRANSFER_WRITE;
  if (ncclParamP2pReadEnable() != 0) ipcTransfers |= uint32_t{1} << RCCL_P2P_TRANSFER_READ;
  validation->validateTransportCapabilities = true;
  for (int t = 0; t < RCCL_EXECUTION_TRANSPORT_COUNT; t++) validation->transportP2pTransferMask[t] = 0;
  if (primary == RCCL_EXECUTION_TRANSPORT_IPC ||
      rcclPolicyTransportHasDedicatedConnector(comm, RCCL_EXECUTION_TRANSPORT_IPC))
    validation->transportP2pTransferMask[RCCL_EXECUTION_TRANSPORT_IPC] = ipcTransfers;
}

void rcclBuildCollectiveExecutionPolicyValidationContext(const ncclComm* comm, const ncclTaskColl* info,
                                                         float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS],
                                                         rcclExecutionPolicyValidationContext* validation) {
  rcclDefaultExecutionPolicyValidationContext(validation);

  // The bounds collective channel tuning honors: per-call CTAs and the
  // NCCL_MIN/MAX_NCHANNELS overrides, within the communicator's channels.
  int minChannels = 1;
  if (info->minCTAs > 0) minChannels = std::max(minChannels, info->minCTAs);
  int userMinChannels = ncclParamMinNchannels();
  if (userMinChannels > 0) minChannels = std::max(minChannels, userMinChannels);

  int maxChannels = comm->nChannels;
  if (info->maxCTAs > 0) maxChannels = std::min(maxChannels, info->maxCTAs);
  int userMaxChannels = ncclParamMaxNchannels();
  if (userMaxChannels > 0) maxChannels = std::min(maxChannels, userMaxChannels);

  validation->minChannels = minChannels;
  validation->maxChannels = maxChannels;
  validation->baselineAlgorithm = info->algorithm;
  validation->baselineProtocol = info->protocol;
  validation->validateAlgoProto = true;
  validation->validateTransport = true;
  validation->transportMask = rcclPolicyAvailableTransportMask(comm, ncclParamP2pDisable());
  for (int algorithm = 0; algorithm < NCCL_NUM_ALGORITHMS; algorithm++) {
    bool algorithmAllowedByTask = info->algMask == 0 || (info->algMask & (uint64_t{1} << algorithm)) != 0;
    for (int protocol = 0; protocol < NCCL_NUM_PROTOCOLS; protocol++) {
      validation->algoProtoAvailable[algorithm][protocol] =
        algorithmAllowedByTask && table[algorithm][protocol] >= 0.0f;
    }
  }
  rcclSetCollectiveTransportCapabilities(comm, validation);
}

// Both ends of a P2P link must reach one verdict, so every bound comes from
// communicator state rather than rank-local overrides.
void rcclBuildP2pExecutionPolicyValidationContext(const ncclComm* comm, ncclFunc_t collAPI, bool p2pDisabled,
                                                  rcclExecutionPolicyValidationContext* validation) {
  rcclDefaultExecutionPolicyValidationContext(validation);
  validation->minChannels = 1;
  validation->maxChannels = comm->p2pnChannels;

  uint32_t protocolMask = 0;
  if (collAPI >= 0 && collAPI < NCCL_NUM_FUNCTIONS) {
    for (int protocol = 0; protocol < NCCL_NUM_PROTOCOLS; protocol++) {
      for (int algorithm = 0; algorithm < NCCL_NUM_ALGORITHMS; algorithm++) {
        if (comm->bandwidths[collAPI][algorithm][protocol] > 0.0f) {
          protocolMask |= uint32_t{1} << protocol;
          break;
        }
      }
    }
  }
  validation->validateP2pProtocol = protocolMask != 0;
  validation->p2pProtocolMask = protocolMask;
  if (ncclParamP2pReadEnable() == 0) {
    validation->validateP2pTransfer = true;
    validation->p2pTransferMask = uint32_t{1} << RCCL_P2P_TRANSFER_WRITE;
  }
  validation->validateTransport = true;
  validation->transportMask = rcclPolicyAvailableTransportMask(comm, p2pDisabled);
  rcclSetP2pTransportCapabilities(comm, p2pDisabled, validation);
}

rcclExecutionPolicyResolution rcclResolveP2pExecutionPolicy(const ncclComm* comm, ncclFunc_t collAPI,
                                                            size_t aggregateBytes, bool inPlace, bool p2pDisabled) {
  rcclExecutionPolicyValidationContext validation;
  rcclBuildP2pExecutionPolicyValidationContext(comm, collAPI, p2pDisabled, &validation);
  return rcclGetCollectiveExecutionPolicyWithValidation(comm, RCCL_EXECUTION_SCOPE_P2P, collAPI, aggregateBytes,
                                                        p2pDisabled, inPlace, &validation);
}

} // namespace

bool rcclPolicyRuntimeToggleRequested() {
  int cached = rcclPolicyToggleRequestedCache.load(std::memory_order_relaxed);
  if (cached < 0) {
    const char* toggle = ncclGetEnv("RCCL_RUNTIME_TRANSPORT_TOGGLE");
    cached = toggle != nullptr && strcmp(toggle, "1") == 0;
    rcclPolicyToggleRequestedCache.store(cached, std::memory_order_relaxed);
  }
  return cached != 0;
}

void rcclPolicyResetEnvCache() {
  rcclPolicyToggleRequestedCache.store(-1, std::memory_order_relaxed);
  rcclPolicyP2pDisableSetCache.store(-1, std::memory_order_relaxed);
}

rcclExecutionTransportIntent rcclPolicyTransportIntent(bool p2pDisabled) {
  if (!rcclPolicyP2pDisableSet()) return RCCL_EXECUTION_TRANSPORT_INTENT_AUTO;
  return p2pDisabled ? RCCL_EXECUTION_TRANSPORT_INTENT_SHM : RCCL_EXECUTION_TRANSPORT_INTENT_IPC;
}

bool rcclPolicyCanonicalTransportEligible(const ncclComm* comm) {
  return comm != nullptr && rcclPolicyRuntimeToggleRequested() && comm->nNodes == 1 && comm->nRanks == 8 &&
         IsArchMatch(comm->archName, "gfx1201");
}

bool rcclPolicyTransportToggleEligible(const ncclComm* comm) {
  return rcclPolicyCanonicalTransportEligible(comm) &&
         rcclPolicyTransportIntent(ncclParamP2pDisable()) == RCCL_EXECUTION_TRANSPORT_INTENT_AUTO &&
         ncclParamP2pDisable() == 0 && ncclParamShmDisable() == 0;
}

// Mirrors shmCanConnect's host and /dev/shm checks over the all-gathered peer
// table, so every rank reaches the same verdict before provisioning SHM.
static bool rcclPolicyAllPeersShareShm(const ncclComm* comm) {
  if (comm->peerInfo == nullptr) return false;
  const ncclPeerInfo& first = comm->peerInfo[0];
  for (int r = 1; r < comm->nRanks; r++) {
    if (comm->peerInfo[r].hostHash != first.hostHash || comm->peerInfo[r].shmDev != first.shmDev) return false;
  }
  return true;
}

bool rcclPolicyCanonicalShmEligible(const ncclComm* comm) {
  return rcclPolicyCanonicalTransportEligible(comm) && ncclParamShmDisable() == 0 && rcclPolicyAllPeersShareShm(comm) &&
         rcclPolicyTransportIntent(ncclParamP2pDisable()) != RCCL_EXECUTION_TRANSPORT_INTENT_IPC;
}

uint32_t rcclPolicyAvailableTransportMask(const ncclComm* comm, bool p2pDisabled) {
  rcclExecutionTransport configured = rcclConfiguredPolicyTransport(p2pDisabled);
  uint32_t mask = configured == RCCL_EXECUTION_TRANSPORT_UNKNOWN ? 0 : rcclExecutionTransportBit(configured);
  if (rcclPolicyTransportIntent(p2pDisabled) == RCCL_EXECUTION_TRANSPORT_INTENT_AUTO &&
      rcclPolicyTransportToggleEligible(comm)) {
    mask |= rcclExecutionTransportBit(RCCL_EXECUTION_TRANSPORT_IPC);
    if (rcclPolicyAllPeersShareShm(comm)) mask |= rcclExecutionTransportBit(RCCL_EXECUTION_TRANSPORT_SHM);
  }
  return mask;
}

bool rcclCollectiveTaskBuffersOverlap(const ncclComm* comm, const ncclTaskColl* task) {
  if (comm == nullptr || task == nullptr) return false;
  int typeSize = ncclTypeSize(task->datatype);
  if (typeSize <= 0) return false;
  return rcclCollectiveBuffersOverlap(task->func, task->sendbuff, task->recvbuff, task->count,
                                      static_cast<size_t>(typeSize), comm->nRanks, /*allToAllvSizes=*/nullptr);
}

int rcclGetCollectiveExecutionPolicyRequiredChannels(ncclComm* comm, bool p2pDisabled) {
  if (comm == nullptr || !rcclPolicyRuntimeToggleRequested()) return 0;
  // Canonical mode provisions for every toggle transport regardless of
  // intent, so forced modes share AUTO's channel pools.
  bool canonical = rcclPolicyCanonicalTransportEligible(comm) && ncclParamShmDisable() == 0;
  rcclExecutionPolicyProvisioningInput input = {
    comm->archName,
    comm->nNodes,
    comm->nRanks,
    canonical ? rcclExecutionTransportBit(RCCL_EXECUTION_TRANSPORT_IPC) |
                  rcclExecutionTransportBit(RCCL_EXECUTION_TRANSPORT_SHM) :
                rcclPolicyAvailableTransportMask(comm, p2pDisabled),
    canonical ? RCCL_EXECUTION_TRANSPORT_INTENT_AUTO : rcclPolicyTransportIntent(p2pDisabled),
  };
  return rcclResolveExecutionPolicyRequiredChannels(&input);
}

rcclExecutionPolicyResolution rcclGetCollectiveExecutionPolicyWithValidation(
  const ncclComm* comm, rcclExecutionScope scope, ncclFunc_t collType, size_t dataSize, bool p2pDisabled, bool inPlace,
  const rcclExecutionPolicyValidationContext* validation) {
  if (comm == nullptr) return rcclPolicyResolutionWithStatus(RCCL_EXECUTION_POLICY_INVALID_INPUT);
  // Execution policies are opt-in; without the toggle RCCL keeps its own choices.
  if (!rcclPolicyRuntimeToggleRequested()) return rcclPolicyResolutionWithStatus(RCCL_EXECUTION_POLICY_NO_MATCH);

  rcclExecutionPolicyValidationContext transportValidation;
  if (validation == nullptr) {
    rcclDefaultExecutionPolicyValidationContext(&transportValidation);
    transportValidation.validateTransport = true;
    transportValidation.transportMask = rcclPolicyAvailableTransportMask(comm, p2pDisabled);
    validation = &transportValidation;
  }
  const rcclCollectivePolicyInput input = {
    scope,
    comm->archName,
    collType,
    dataSize,
    comm->nNodes,
    comm->nRanks,
    scope == RCCL_EXECUTION_SCOPE_P2P ? comm->p2pnChannels : comm->nChannels,
    rcclPolicyAvailableTransportMask(comm, p2pDisabled),
    inPlace,
    rcclPolicyTransportIntent(p2pDisabled),
  };
  return rcclResolveCollectiveExecutionPolicyWithValidation(&input, validation);
}

bool rcclGetCollectiveExecutionPolicy(const ncclComm* comm, rcclExecutionScope scope, ncclFunc_t collType,
                                      size_t dataSize, bool p2pDisabled, bool inPlace,
                                      rcclCollectiveExecutionPolicy* policy) {
  if (policy == nullptr) return false;
  rcclExecutionPolicyResolution resolution =
    rcclGetCollectiveExecutionPolicyWithValidation(comm, scope, collType, dataSize, p2pDisabled, inPlace, nullptr);
  *policy = resolution.policy;
  return resolution.status == RCCL_EXECUTION_POLICY_SELECTED;
}

bool rcclExecutionPolicyForP2pTask(ncclComm* comm, ncclTaskP2p* task, bool p2pDisabled,
                                   rcclCollectiveExecutionPolicy* policy) {
  if (comm == nullptr || task == nullptr || policy == nullptr || comm->nRanks <= 0) return false;
  size_t aggregateBytes = rcclP2pAggregateBytes(comm, task->bytes);
  rcclExecutionPolicyResolution resolution =
    rcclResolveP2pExecutionPolicy(comm, task->collAPI, aggregateBytes, task->inPlace, p2pDisabled);
  *policy = resolution.policy;
  rcclLogRejectedPolicy(NCCL_COLL, "P2P", task->collAPI, aggregateBytes, resolution);
  return resolution.status == RCCL_EXECUTION_POLICY_SELECTED;
}

bool rcclApplyCollectiveExecutionPolicy(ncclComm* comm, ncclTaskColl* info, size_t dataSize,
                                        float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS], int* policyChannels) {
  if (comm == nullptr || info == nullptr || table == nullptr || policyChannels == nullptr) return false;

  info->executionTransport = RCCL_EXECUTION_TRANSPORT_UNKNOWN;
  rcclExecutionPolicyValidationContext validation;
  rcclBuildCollectiveExecutionPolicyValidationContext(comm, info, table, &validation);
  rcclExecutionPolicyResolution resolution =
    rcclGetCollectiveExecutionPolicyWithValidation(comm, RCCL_EXECUTION_SCOPE_COLLECTIVE, info->func, dataSize,
                                                   ncclParamP2pDisable(), rcclCollectiveTaskBuffersOverlap(comm, info),
                                                   &validation);
  if (resolution.status != RCCL_EXECUTION_POLICY_SELECTED) {
    rcclLogRejectedPolicy(NCCL_TUNING, "collective", info->func, dataSize, resolution);
    return false;
  }
  const rcclCollectiveExecutionPolicy& policy = resolution.policy;

  int algorithm = policy.algorithm >= 0 ? policy.algorithm : info->algorithm;
  int protocol = policy.protocol >= 0 ? policy.protocol : info->protocol;
  if (policy.algorithm >= 0 || policy.protocol >= 0) {
    info->algorithm = algorithm;
    info->protocol = protocol;
  }
  info->executionTransport = policy.transport;

  *policyChannels = policy.nChannels;
  INFO(NCCL_TUNING,
       "RCCL collective execution policy: arch=%s coll=%d bytes=%zu rule=%u "
       "channels=%d algorithm=%d protocol=%d transport=%d",
       comm->archName, (int)info->func, dataSize, (unsigned)resolution.selectedRuleId, policy.nChannels,
       policy.algorithm, policy.protocol, (int)policy.transport);
  return true;
}

int rcclPolicyCollectiveConnectorIndex(const ncclComm* comm, const ncclTaskColl* info, int channelId, size_t nBytes) {
  if (comm == nullptr || info == nullptr) return -1;
  if (channelId < 0 || channelId >= comm->nChannels || info->algorithm != NCCL_ALGO_RING ||
      info->protocol != NCCL_PROTO_SIMPLE || info->regBufType != 0)
    return -1;

  // Validation admits a transport only on the default connectors, which carry
  // the configured transport, or on its dedicated slot. Only the latter
  // constrains the connector; otherwise the runtime's choice stands.
  rcclExecutionTransport transport = static_cast<rcclExecutionTransport>(info->executionTransport);
  if (!rcclPolicyTransportHasDedicatedConnector(comm, transport)) return -1;

  const ncclChannel& channel = comm->channels[channelId];
  if (channel.ring.next < 0 || channel.ring.next >= comm->nRanks || channel.ring.prev < 0 ||
      channel.ring.prev >= comm->nRanks)
    return -1;
  const int connIndex = RCCL_CONN_IDX_COLL_SHM;
  const ncclConnector& send = channel.peers[channel.ring.next]->send[connIndex];
  const ncclConnector& recv = channel.peers[channel.ring.prev]->recv[connIndex];
  if (rcclConnectorUsesPolicyTransport(send, transport, /*isSendNotRecv=*/true, /*requireConnected=*/true) &&
      rcclConnectorUsesPolicyTransport(recv, transport, /*isSendNotRecv=*/false, /*requireConnected=*/true)) {
    if (comm->rank == 0) {
      INFO(NCCL_TUNING, "RCCL execution-policy transport: bytes=%zu transport=SHM connIndex=%d", nBytes, connIndex);
    }
    return connIndex;
  }
  if (comm->rank == 0) {
    INFO(NCCL_TUNING,
         "RCCL connector adapter found no connected transport=%d "
         "for channel=%d",
         (int)transport, channelId);
  }
  return -1;
}

int rcclPolicyP2pConnectorIndex(const ncclComm* comm, int channelId, int peer, bool isSendNotRecv,
                                rcclExecutionTransport transport) {
  if (comm == nullptr || channelId < 0 || channelId >= MAXCHANNELS || peer < 0 || peer >= comm->nRanks ||
      rcclPolicyTransportComm(transport, isSendNotRecv) == nullptr)
    return -1;

  if (transport == RCCL_EXECUTION_TRANSPORT_SHM && rcclPolicyCanonicalShmEligible(comm)) return RCCL_CONN_IDX_P2P_SHM;

  const ncclChannelPeer* channelPeer = comm->channels[channelId].peers[peer];
  if (channelPeer != nullptr) {
    for (int connIndex = 1; connIndex < NCCL_MAX_CONNS; connIndex++) {
      const ncclConnector& connector = isSendNotRecv ? channelPeer->send[connIndex] : channelPeer->recv[connIndex];
      if (rcclConnectorUsesPolicyTransport(connector, transport, isSendNotRecv, /*requireConnected=*/true))
        return connIndex;
    }
  }
  return rcclPlannedP2pConnectorIndex(comm, transport);
}

// IPC registration bypasses proxy connectors and is incompatible with the
// policy-selected SHM path.
bool rcclPolicyCollectiveAllowsRegistration(const ncclTaskColl* info) {
  return info == nullptr || info->executionTransport != RCCL_EXECUTION_TRANSPORT_SHM;
}

// Resolves exactly as the AllToAll's P2P tasks will, so the path is chosen by
// the policy those tasks run.
bool rcclPolicyAllToAllUsesSendRecvPath(const ncclComm* comm, size_t aggregateBytes, bool inPlace) {
  if (comm == nullptr) return false;
  rcclExecutionPolicyResolution resolution =
    rcclResolveP2pExecutionPolicy(comm, ncclFuncAlltoAll, aggregateBytes, inPlace, ncclParamP2pDisable());
  return resolution.status == RCCL_EXECUTION_POLICY_SELECTED && resolution.policy.path == RCCL_P2P_PATH_SENDRECV;
}

void rcclPolicyResolveP2pTask(ncclComm* comm, ncclTaskP2p* task) {
  task->executionPolicyMatched =
    rcclExecutionPolicyForP2pTask(comm, task, ncclParamP2pDisable(), &task->executionPolicy);
  if (!task->executionPolicyMatched) rcclDefaultCollectiveExecutionPolicy(&task->executionPolicy);
}

int rcclPolicyP2pTaskPool(const ncclComm* comm, const ncclTaskP2p* task) {
  const rcclCollectiveExecutionPolicy& policy = task->executionPolicy;
  if (task->executionPolicyMatched && policy.path == RCCL_P2P_PATH_SENDRECV && policy.nChannels > 0)
    return std::min(comm->p2pnChannels, policy.nChannels);
  return comm->p2pnChannels;
}

bool rcclPolicyP2pWorkSplitsDirections(const ncclComm* comm, ncclTaskP2p* const tasks[2], int sendRank) {
  return tasks[0] != nullptr && tasks[1] != nullptr && sendRank != comm->rank &&
         rcclPolicyP2pTaskPool(comm, tasks[0]) != rcclPolicyP2pTaskPool(comm, tasks[1]);
}

void rcclPolicyPlanP2pWork(const ncclComm* comm, ncclTaskP2p* const tasks[2], bool logSelection,
                           rcclP2pPolicyWorkPlan* plan) {
  int activeChannels = -1;
  for (int dir = 0; dir < 2; dir++) {
    plan->matched[dir] = tasks[dir] != nullptr && tasks[dir]->executionPolicyMatched;
    if (plan->matched[dir]) plan->policy[dir] = tasks[dir]->executionPolicy;
    else rcclDefaultCollectiveExecutionPolicy(&plan->policy[dir]);
    plan->channels[dir] = plan->matched[dir] ? plan->policy[dir].nChannels : -1;
    if (tasks[dir] == nullptr) continue;
    // Peer directions with different pools are planned as separate work items,
    // so min() only combines pools for a self send/recv pair.
    int pool = rcclPolicyP2pTaskPool(comm, tasks[dir]);
    activeChannels = activeChannels < 0 ? pool : std::min(activeChannels, pool);
  }
  plan->activeChannels = activeChannels > 0 ? activeChannels : comm->p2pnChannels;

  if (!logSelection || (!plan->matched[0] && !plan->matched[1])) return;
  int dir = plan->matched[0] ? 0 : 1;
  const rcclCollectiveExecutionPolicy& policy = plan->policy[dir];
  size_t aggregateBytes = rcclP2pAggregateBytes(comm, tasks[dir]->bytes);
  const char* transferName = policy.transferMode == RCCL_P2P_TRANSFER_READ  ? "read" :
                             policy.transferMode == RCCL_P2P_TRANSFER_WRITE ? "write" :
                                                                              "auto";
  INFO(NCCL_COLL,
       "RCCL P2P execution policy: arch=%s coll=%d aggregateBytes=%zu inPlace=%d channels=%d->%d path=SendRecv "
       "protocol=%s transfer=%s transport=%d",
       comm->archName, (int)tasks[dir]->collAPI, aggregateBytes, (int)tasks[dir]->inPlace, comm->p2pnChannels,
       plan->activeChannels, policy.protocol >= 0 ? ncclProtoStr[policy.protocol] : "auto", transferName,
       (int)policy.transport);
}

namespace {

// Connected connector slot carrying the policy's transport and transfer mode
// on one channel, or -1 when the channel has none. Policies that constrain
// neither keep defaultConnIndex.
int rcclPolicyP2pChannelConnector(const ncclComm* comm, const rcclCollectiveExecutionPolicy& policy, int channelId,
                                  int peer, bool isSendNotRecv, int defaultConnIndex) {
  bool constrainsTransport = policy.transport != RCCL_EXECUTION_TRANSPORT_UNKNOWN;
  bool constrainsTransfer =
    policy.transport != RCCL_EXECUTION_TRANSPORT_SHM && policy.transferMode != RCCL_P2P_TRANSFER_AUTO;
  if (!constrainsTransport && !constrainsTransfer) return defaultConnIndex;
  if (channelId < 0 || channelId >= MAXCHANNELS || peer < 0 || peer >= comm->nRanks) return -1;
  const ncclChannelPeer* channelPeer = comm->channels[channelId].peers[peer];
  if (channelPeer == nullptr) return -1;
  auto connectorAt = [&](int connIndex) -> const ncclConnector& {
    return isSendNotRecv ? channelPeer->send[connIndex] : channelPeer->recv[connIndex];
  };

  int connIndex = defaultConnIndex;
  if (constrainsTransport) {
    connIndex = rcclPolicyP2pConnectorIndex(comm, channelId, peer, isSendNotRecv, policy.transport);
    if (connIndex < 0 || !rcclConnectorUsesPolicyTransport(connectorAt(connIndex), policy.transport, isSendNotRecv,
                                                           /*requireConnected=*/true))
      return -1;
  }
  if (!constrainsTransfer) return connIndex;

  int requiredFlag = policy.transferMode == RCCL_P2P_TRANSFER_READ ? NCCL_P2P_READ : NCCL_P2P_WRITE;
  constexpr int candidateConnIndices[] = {1, RCCL_CONN_IDX_P2P_ALT};
  for (int candidate : candidateConnIndices) {
    if (rcclConnIndexIsIntraNet(comm, candidate)) continue;
    const ncclConnector& conn = connectorAt(candidate);
    if (!conn.connected || !(conn.conn.flags & requiredFlag)) continue;
    if (constrainsTransport && !rcclConnectorUsesPolicyTransport(conn, policy.transport, isSendNotRecv,
                                                                 /*requireConnected=*/true))
      continue;
    return candidate;
  }
  return -1;
}

// One slot must satisfy the policy on every channel, since a work item uses a
// single connector index for all of its parts.
int rcclPolicyP2pChannelsConnector(const ncclComm* comm, const rcclCollectiveExecutionPolicy& policy,
                                   const int* channelIds, int nChannelIds, int peer, bool isSendNotRecv,
                                   int defaultConnIndex, int* failedChannel) {
  int connIndex = -1;
  for (int i = 0; i < nChannelIds; i++) {
    int channelConnIndex =
      rcclPolicyP2pChannelConnector(comm, policy, channelIds[i], peer, isSendNotRecv, defaultConnIndex);
    if (channelConnIndex < 0 || (i > 0 && channelConnIndex != connIndex)) {
      *failedChannel = channelIds[i];
      return -1;
    }
    connIndex = channelConnIndex;
  }
  return connIndex < 0 ? defaultConnIndex : connIndex;
}

} // namespace

int rcclPolicyP2pWorkConnectorIndex(const ncclComm* comm, const rcclP2pPolicyWorkPlan* plan, int dir,
                                    const int* channelIds, int nChannelIds, int peer, int defaultConnIndex) {
  if (!plan->matched[dir]) return defaultConnIndex;
  int failedChannel = -1;
  int connIndex = rcclPolicyP2pChannelsConnector(comm, plan->policy[dir], channelIds, nChannelIds, peer,
                                                 /*isSendNotRecv=*/dir != 0, defaultConnIndex, &failedChannel);
  if (connIndex >= 0) return connIndex;
  // Validation already checked every channel of this task's pool, which is the
  // pool its work item uses.
  INFO(NCCL_COLL,
       "RCCL P2P execution policy connector unavailable on channel %d; using default connector "
       "(dir=%s peer=%d transport=%d transfer=%d)",
       failedChannel, dir ? "send" : "recv", peer, (int)plan->policy[dir].transport,
       (int)plan->policy[dir].transferMode);
  return defaultConnIndex;
}

void rcclPolicyValidateP2pTask(ncclComm* comm, ncclTaskP2p* task, bool isSendNotRecv) {
  int peer = task->root;
  if (!task->executionPolicyMatched || peer < 0 || peer >= comm->nRanks || peer == comm->rank) return;
  rcclP2pPolicyPreconnect preconnect;
  rcclPolicyP2pPreconnect(comm, task, &preconnect);

  int round = 0;
  while (round < comm->nRanks &&
         peer != (isSendNotRecv ? comm->p2pSchedule[round].sendRank : comm->p2pSchedule[round].recvRank))
    round++;
  int failedChannel = -1;
  if (round < comm->nRanks && preconnect.nChannels > 0) {
    int base = ncclP2pChannelBaseForRound(comm, round, rcclEffectiveP2pBatchEnable(comm)) % preconnect.nChannels;
    int channelIds[MAXCHANNELS];
    int nChannelIds = std::min(preconnect.nChannelsPerPeer, MAXCHANNELS);
    for (int part = 0; part < nChannelIds; part++)
      channelIds[part] =
        ncclP2pChannelForPart(preconnect.nChannels, base, part, nChannelIds, comm->nNodes, comm->p2pChannelShiftSize);
    if (rcclPolicyP2pChannelsConnector(comm, task->executionPolicy, channelIds, nChannelIds, peer, isSendNotRecv,
                                       /*defaultConnIndex=*/1, &failedChannel) >= 0)
      return;
  }
  INFO(NCCL_COLL,
       "RCCL P2P execution policy dropped: coll=%d peer=%d dir=%s transport=%d transfer=%d has no "
       "connected connector on channel %d; task runs without a policy",
       (int)task->collAPI, peer, isSendNotRecv ? "send" : "recv", (int)task->executionPolicy.transport,
       (int)task->executionPolicy.transferMode, failedChannel);
  task->executionPolicyMatched = false;
  rcclDefaultCollectiveExecutionPolicy(&task->executionPolicy);
}

void rcclPolicyValidateP2pTasks(ncclComm* comm) {
  for (int peer = 0; peer < comm->nRanks; peer++) {
    for (ncclTaskP2p* task = ncclIntruQueueHead(&comm->planner.peers[peer].sendQueue); task != nullptr;
         task = task->next)
      rcclPolicyValidateP2pTask(comm, task, /*isSendNotRecv=*/true);
    for (ncclTaskP2p* task = ncclIntruQueueHead(&comm->planner.peers[peer].recvQueue); task != nullptr;
         task = task->next)
      rcclPolicyValidateP2pTask(comm, task, /*isSendNotRecv=*/false);
  }
}

int rcclPolicyP2pWorkProtocol(const rcclP2pPolicyWorkPlan* plan, int dir, int selectedProtocol, bool useLL128,
                              bool latencyBufferAvailable) {
  int requestedProtocol = plan->matched[dir] ? plan->policy[dir].protocol : -1;
  if (requestedProtocol < 0) return selectedProtocol;
  if (requestedProtocol == NCCL_PROTO_SIMPLE) return NCCL_PROTO_SIMPLE;
  if (requestedProtocol == NCCL_PROTO_LL && !useLL128 && latencyBufferAvailable) return NCCL_PROTO_LL;
  if (requestedProtocol == NCCL_PROTO_LL128 && useLL128 && latencyBufferAvailable) return NCCL_PROTO_LL128;
  return selectedProtocol;
}

bool rcclPolicyP2pWorkAllowsRegistration(const rcclP2pPolicyWorkPlan* plan, int dir) {
  return !plan->matched[dir] || plan->policy[dir].transport != RCCL_EXECUTION_TRANSPORT_SHM;
}

bool rcclPolicyP2pTaskAllowsRegistration(const ncclTaskP2p* task) {
  return !task->executionPolicyMatched || task->executionPolicy.transport != RCCL_EXECUTION_TRANSPORT_SHM;
}

bool rcclPolicyP2pPreconnect(const ncclComm* comm, const ncclTaskP2p* task, rcclP2pPolicyPreconnect* preconnect) {
  preconnect->nChannels = comm->p2pnChannels;
  preconnect->nChannelsPerPeer = comm->p2pnChannelsPerPeer;
  preconnect->nConnIndices = 0;
  if (!task->executionPolicyMatched) return false;
  const rcclCollectiveExecutionPolicy& policy = task->executionPolicy;

  bool usesShm = policy.transport == RCCL_EXECUTION_TRANSPORT_SHM;
  preconnect->nChannels = rcclPolicyP2pTaskPool(comm, task);
  if (preconnect->nChannels == comm->p2pnChannels && policy.transferMode == RCCL_P2P_TRANSFER_AUTO && !usesShm)
    return true;
  preconnect->nChannelsPerPeer = std::min(comm->p2pnChannelsPerPeer, preconnect->nChannels);
  int transportConnIndex = rcclPlannedP2pConnectorIndex(comm, policy.transport);
  if (usesShm && transportConnIndex == RCCL_CONN_IDX_P2P_SHM) {
    preconnect->connIndices[preconnect->nConnIndices++] = RCCL_CONN_IDX_P2P_SHM;
    return true;
  }
  preconnect->connIndices[preconnect->nConnIndices++] = 1;
  if (!usesShm && policy.transferMode != RCCL_P2P_TRANSFER_AUTO &&
      !rcclConnIndexIsIntraNet(comm, RCCL_CONN_IDX_P2P_ALT))
    preconnect->connIndices[preconnect->nConnIndices++] = RCCL_CONN_IDX_P2P_ALT;
  return true;
}

int rcclPolicyP2pPreconnects(ncclComm* comm, const ncclTaskP2p* task, rcclP2pPolicyPreconnect preconnects[2]) {
  ncclTaskP2p placements[2] = {};
  int nPreconnects = 0;
  for (int i = 0; i < 2; i++) {
    placements[i].collAPI = task->collAPI;
    placements[i].bytes = task->bytes;
    placements[i].inPlace = i == 0 ? task->inPlace : !task->inPlace;
    if (i == 0) {
      placements[i].executionPolicyMatched = task->executionPolicyMatched;
      placements[i].executionPolicy = task->executionPolicy;
    } else {
      rcclPolicyResolveP2pTask(comm, &placements[i]);
    }
    rcclP2pPolicyPreconnect& preconnect = preconnects[nPreconnects];
    if (!rcclPolicyP2pPreconnect(comm, &placements[i], &preconnect) || preconnect.nConnIndices == 0) continue;
    if (nPreconnects == 1 && preconnects[0].nChannels == preconnect.nChannels &&
        preconnects[0].nChannelsPerPeer == preconnect.nChannelsPerPeer &&
        preconnects[0].nConnIndices == preconnect.nConnIndices &&
        std::equal(preconnect.connIndices, preconnect.connIndices + preconnect.nConnIndices,
                   preconnects[0].connIndices))
      continue;
    nPreconnects++;
  }
  return nPreconnects;
}
