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
bool rcclPolicyTransportToggleEligible(const struct ncclComm* comm);
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

#endif // RCCL_POLICY_ADAPTER_H_
