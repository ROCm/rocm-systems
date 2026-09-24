/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_POLICY_ADAPTER_H_
#define RCCL_POLICY_ADAPTER_H_

#include "collective_execution_policy.h"

struct ncclComm;
struct ncclTaskColl;
struct ncclTaskP2p;

// Input boundary: normalize mutable RCCL state into policy-specific values.
enum rcclExecutionTransportIntent rcclPolicyTransportIntent(bool p2pDisabled);
uint32_t rcclPolicyAvailableTransportMask(
  const struct ncclComm* comm, bool p2pDisabled);
// Canonical transport mode (RCCL_RUNTIME_TRANSPORT_TOGGLE=1 on an eligible
// topology): every intent shares one provisioning and connector layout.
bool rcclPolicyCanonicalTransportEligible(const struct ncclComm* comm);
// AUTO intent with more than one selectable transport.
bool rcclPolicyTransportToggleEligible(const struct ncclComm* comm);
// Policy SHM work uses the dedicated SHM connector slot.
bool rcclPolicyCanonicalShmEligible(const struct ncclComm* comm);
bool rcclCollectiveTaskBuffersOverlap(
  const struct ncclComm* comm, const struct ncclTaskColl* task);

int rcclGetCollectiveExecutionPolicyRequiredChannels(
  struct ncclComm* comm, bool p2pDisabled);
struct rcclExecutionPolicyResolution rcclGetCollectiveExecutionPolicyWithValidation(
  const struct ncclComm* comm, enum rcclExecutionScope scope,
  ncclFunc_t collType, size_t dataSize, bool p2pDisabled, bool inPlace,
  const struct rcclExecutionPolicyValidationContext* validation);
bool rcclGetCollectiveExecutionPolicy(
  const struct ncclComm* comm, enum rcclExecutionScope scope,
  ncclFunc_t collType, size_t dataSize, bool p2pDisabled, bool inPlace,
  struct rcclCollectiveExecutionPolicy* policy);
inline bool rcclGetCollectiveExecutionPolicy(
  const struct ncclComm* comm, enum rcclExecutionScope scope,
  ncclFunc_t collType, size_t dataSize, bool p2pDisabled,
  struct rcclCollectiveExecutionPolicy* policy) {
  return rcclGetCollectiveExecutionPolicy(
    comm, scope, collType, dataSize, p2pDisabled, /*inPlace=*/false, policy);
}

bool rcclExecutionPolicyForP2pTask(
  struct ncclComm* comm, struct ncclTaskP2p* task, bool p2pDisabled,
  struct rcclCollectiveExecutionPolicy* policy);
bool rcclApplyCollectiveExecutionPolicy(
  struct ncclComm* comm, struct ncclTaskColl* info, size_t dataSize,
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS],
  int* policyChannels);

// Output boundary: materialize a logical policy transport as an RCCL connector.
// The adapter discovers connected transport types and keeps connector indices
// out of policy inputs, outputs, and rule data.
int rcclPolicyCollectiveConnectorIndex(
  const struct ncclComm* comm, const struct ncclTaskColl* info,
  int channelId, size_t nBytes);
int rcclPolicyP2pConnectorIndex(
  const struct ncclComm* comm, int channelId, int peer,
  bool isSendNotRecv, enum rcclExecutionTransport transport);

// Output boundary: runtime parameters derived from a selected policy. Callers
// apply these values and never interpret policy fields themselves.
bool rcclPolicyCollectiveAllowsRegistration(const struct ncclTaskColl* info);
bool rcclPolicyAllToAllUsesSendRecvPath(
  const struct ncclComm* comm, size_t aggregateBytes, bool inPlace);

// One P2P work item; index 0 is receive and 1 is send.
struct rcclP2pPolicyWorkPlan {
  bool matched[2];
  struct rcclCollectiveExecutionPolicy policy[2];
  int channels[2];    // Per-direction channel cap; -1 keeps runtime selection.
  int activeChannels; // Channel pool shared by both directions.
};
void rcclPolicyPlanP2pWork(
  struct ncclComm* comm, struct ncclTaskP2p* const tasks[2], bool logSelection,
  struct rcclP2pPolicyWorkPlan* plan);
int rcclPolicyP2pWorkConnectorIndex(
  const struct ncclComm* comm, const struct rcclP2pPolicyWorkPlan* plan, int dir,
  int channelId, int peer, int defaultConnIndex);
int rcclPolicyP2pWorkProtocol(
  const struct rcclP2pPolicyWorkPlan* plan, int dir, int selectedProtocol,
  bool useLL128, bool latencyBufferAvailable);
bool rcclPolicyP2pWorkAllowsRegistration(
  const struct rcclP2pPolicyWorkPlan* plan, int dir);
bool rcclPolicyP2pTaskAllowsRegistration(
  struct ncclComm* comm, struct ncclTaskP2p* task);

// Connector slots a P2P task's policy needs beyond the peer's default mapping,
// marked over the policy's channel pool.
struct rcclP2pPolicyPreconnect {
  int nChannels;
  int nChannelsPerPeer;
  int nConnIndices;
  int connIndices[2];
};
bool rcclPolicyP2pPreconnect(
  struct ncclComm* comm, struct ncclTaskP2p* task,
  struct rcclP2pPolicyPreconnect* preconnect);

#endif // RCCL_POLICY_ADAPTER_H_
