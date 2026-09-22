/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collective_execution_policy.h"
#include "collective_execution_policy_internal.h"

#include "archinfo.h"
#include "comm.h"
#include "debug.h"
#include "device.h"
#include "param.h"
#include "rccl_common.h"

#include <algorithm>
#include <limits.h>
#include <stdint.h>
#include <string.h>

extern int64_t ncclParamShmDisable();

namespace {

// Policy resolution architecture
// ------------------------------
// A fact is an immutable, typed observation available during one resolution.
// Facts may be copied from operation arguments (scope, collective, data size),
// read from communicator state (architecture, node/rank/channel counts), or
// derived by the caller (transport and in-place buffer overlap). They
// form a per-call snapshot, not mutable global program state.
//
// Inputs are normalized into these facts, then resolved through three pure
// decision stages:
//
//   match -> validate -> select
//
// 1. Match evaluates every rule's conditions against the same immutable facts.
//    A missing fact, type mismatch, or failed comparison produces NoMatch.
// 2. Validate applies one matched rule's assignments to a fresh default policy.
//    Every assignment must be structurally valid (field, type, and range), then
//    the complete candidate is checked atomically against the caller's optional
//    normalized constraint snapshot. Candidates are never merged.
// 3. Select retains the first valid candidate. Rule-array index is both ID and
//    priority, so the lowest valid index wins. The scan continues to validate
//    every later match, but later candidates cannot override the winner.
//
// No match, or only invalid matches, returns false with the default policy and
// RCCL_EXECUTION_POLICY_NO_RULE. Architecture rules are declared as data in
// collective_execution_policy_rules.cc and pre-expanded once into generic rule
// rows; custom rule arrays use the same resolver.

// Finds one normalized input fact by field.
const rcclExecutionPolicyFact* rcclFindExecutionPolicyFact(const rcclExecutionPolicyFact* facts, size_t factCount,
                                                           enum rcclExecutionPolicyInputField field) {
  for (size_t i = 0; i < factCount; i++) {
    if (facts[i].field == field) return &facts[i];
  }
  return nullptr;
}

// Applies a typed ordering operator to a three-way comparison result.
bool rcclCompareOrdered(int comparison, enum rcclExecutionPolicyCompareOp op) {
  switch (op) {
  case RCCL_EXECUTION_COMPARE_EQ: return comparison == 0;
  case RCCL_EXECUTION_COMPARE_NE: return comparison != 0;
  case RCCL_EXECUTION_COMPARE_LT: return comparison < 0;
  case RCCL_EXECUTION_COMPARE_LE: return comparison <= 0;
  case RCCL_EXECUTION_COMPARE_GT: return comparison > 0;
  case RCCL_EXECUTION_COMPARE_GE: return comparison >= 0;
  default: return false;
  }
}

// Compares two policy values, including architecture-pattern matching.
bool rcclExecutionPolicyValueMatches(const rcclExecutionPolicyValue& actual,
                                     enum rcclExecutionPolicyCompareOp op,
                                     const rcclExecutionPolicyValue& expected) {
  if (actual.type != expected.type) return false;

  switch (actual.type) {
  case RCCL_EXECUTION_VALUE_SIGNED:
    return rcclCompareOrdered((actual.signedValue > expected.signedValue) -
                                (actual.signedValue < expected.signedValue),
                              op);
  case RCCL_EXECUTION_VALUE_UNSIGNED:
    return rcclCompareOrdered((actual.unsignedValue > expected.unsignedValue) -
                                (actual.unsignedValue < expected.unsignedValue),
                              op);
  case RCCL_EXECUTION_VALUE_STRING:
    if (actual.stringValue == nullptr || expected.stringValue == nullptr) return false;
    if (op == RCCL_EXECUTION_COMPARE_ARCH_MATCH) return IsArchMatch(actual.stringValue, expected.stringValue);
    return rcclCompareOrdered(strcmp(actual.stringValue, expected.stringValue), op);
  case RCCL_EXECUTION_VALUE_BOOL:
    return rcclCompareOrdered(static_cast<int>(actual.boolValue) - static_cast<int>(expected.boolValue), op);
  }
  return false;
}

// Normalizes the transport preference produced by RCCL's enable/disable
// controls. A disabled P2P transport exposes SHM only while SHM itself remains
// enabled; otherwise the next configured fallback is NET.
rcclExecutionTransport rcclConfiguredExecutionTransport(bool p2pDisabled) {
  if (!p2pDisabled) return RCCL_EXECUTION_TRANSPORT_IPC;
  if (ncclParamShmDisable() == 0) return RCCL_EXECUTION_TRANSPORT_SHM;
  return RCCL_EXECUTION_TRANSPORT_NET;
}

// Preserve the distinction between an absent setting (automatic policy
// selection) and an explicit setting whose normalized value happens to equal
// RCCL's default. The adapter records this as an immutable input fact.
rcclExecutionTransport rcclRequestedExecutionTransport(bool p2pDisabled) {
  const char* p2pDisable = ncclGetEnv("NCCL_P2P_DISABLE");
  if (p2pDisable == nullptr || p2pDisable[0] == '\0')
    return RCCL_EXECUTION_TRANSPORT_UNKNOWN;
  return rcclConfiguredExecutionTransport(p2pDisabled);
}

uint32_t rcclAvailableExecutionTransportMask(const struct ncclComm* comm,
                                             bool p2pDisabled) {
  uint32_t mask = 1u << rcclConfiguredExecutionTransport(p2pDisabled);
  if (rcclRequestedExecutionTransport(p2pDisabled) ==
        RCCL_EXECUTION_TRANSPORT_UNKNOWN &&
      rcclRuntimeTransportToggleEligible(comm)) {
    mask |= (1u << RCCL_EXECUTION_TRANSPORT_IPC) |
            (1u << RCCL_EXECUTION_TRANSPORT_SHM);
  }
  return mask;
}

// Stage 1: match one rule against the normalized input facts.
bool rcclCollectiveExecutionRuleMatches(const rcclCollectiveExecutionRule& rule,
                                        const rcclExecutionPolicyFact* facts, size_t factCount) {
  if (rule.conditionCount != 0 && rule.conditions == nullptr) return false;
  for (size_t i = 0; i < rule.conditionCount; i++) {
    const rcclExecutionPolicyCondition& condition = rule.conditions[i];
    const rcclExecutionPolicyFact* fact = rcclFindExecutionPolicyFact(facts, factCount, condition.field);
    if (fact == nullptr || !rcclExecutionPolicyValueMatches(fact->value, condition.op, condition.value)) return false;
  }
  return true;
}

// Validates and applies one output assignment to a candidate policy.
bool rcclApplyCollectivePolicyAssignment(const rcclExecutionPolicyAssignment& assignment,
                                         rcclCollectiveExecutionPolicy* policy) {
  if (assignment.value.type != RCCL_EXECUTION_VALUE_SIGNED) return false;
  int64_t value = assignment.value.signedValue;
  if (value < INT_MIN || value > INT_MAX) return false;

  switch (assignment.field) {
  case RCCL_EXECUTION_OUTPUT_N_CHANNELS:
    if (value < -1 || value > MAXCHANNELS) return false;
    policy->nChannels = static_cast<int>(value);
    return true;
  case RCCL_EXECUTION_OUTPUT_ALGORITHM:
    if (value < -1 || value >= NCCL_NUM_ALGORITHMS) return false;
    policy->algorithm = static_cast<int>(value);
    return true;
  case RCCL_EXECUTION_OUTPUT_PROTOCOL:
    if (value < -1 || value >= NCCL_NUM_PROTOCOLS) return false;
    policy->protocol = static_cast<int>(value);
    return true;
  case RCCL_EXECUTION_OUTPUT_PATH:
    if (value < RCCL_P2P_PATH_AUTO || value > RCCL_P2P_PATH_SENDRECV) return false;
    policy->path = static_cast<enum rcclP2pExecutionPath>(value);
    return true;
  case RCCL_EXECUTION_OUTPUT_TRANSFER_MODE:
    if (value < RCCL_P2P_TRANSFER_AUTO || value > RCCL_P2P_TRANSFER_READ) return false;
    policy->transferMode = static_cast<enum rcclP2pTransferMode>(value);
    return true;
  case RCCL_EXECUTION_OUTPUT_TRANSPORT:
    if (value <= RCCL_EXECUTION_TRANSPORT_UNKNOWN ||
        value >= RCCL_EXECUTION_TRANSPORT_COUNT)
      return false;
    policy->transport = static_cast<enum rcclExecutionTransport>(value);
    return true;
  }
  return false;
}

// Stage 2a: structurally validate one matched rule and materialize its atomic
// candidate. Semantic validation runs only after every assignment succeeds.
rcclExecutionPolicyValidationError rcclValidateCollectiveExecutionRule(
  const rcclCollectiveExecutionRule& rule, rcclCollectiveExecutionPolicy* policy) {
  if (policy == nullptr || (rule.assignmentCount != 0 && rule.assignments == nullptr))
    return RCCL_EXECUTION_POLICY_VALIDATION_INVALID_ASSIGNMENT;

  rcclCollectiveExecutionPolicy candidate;
  rcclDefaultCollectiveExecutionPolicy(&candidate);
  for (size_t assignmentIndex = 0; assignmentIndex < rule.assignmentCount; assignmentIndex++) {
    if (!rcclApplyCollectivePolicyAssignment(rule.assignments[assignmentIndex], &candidate))
      return RCCL_EXECUTION_POLICY_VALIDATION_INVALID_ASSIGNMENT;
  }
  *policy = candidate;
  return RCCL_EXECUTION_POLICY_VALIDATION_NONE;
}

bool rcclExecutionPolicyScopeFromFacts(const rcclExecutionPolicyFact* facts, size_t factCount,
                                       rcclExecutionScope* scope) {
  const rcclExecutionPolicyFact* fact =
    rcclFindExecutionPolicyFact(facts, factCount, RCCL_EXECUTION_INPUT_SCOPE);
  if (fact == nullptr || fact->value.type != RCCL_EXECUTION_VALUE_SIGNED ||
      fact->value.signedValue < RCCL_EXECUTION_SCOPE_P2P ||
      fact->value.signedValue > RCCL_EXECUTION_SCOPE_COLLECTIVE)
    return false;
  *scope = static_cast<rcclExecutionScope>(fact->value.signedValue);
  return true;
}

bool rcclExecutionPolicyRequestedTransportFromFacts(
  const rcclExecutionPolicyFact* facts, size_t factCount,
  rcclExecutionTransport* requestedTransport) {
  const rcclExecutionPolicyFact* fact =
    rcclFindExecutionPolicyFact(
      facts, factCount, RCCL_EXECUTION_INPUT_REQUESTED_TRANSPORT);
  if (fact == nullptr || fact->value.type != RCCL_EXECUTION_VALUE_SIGNED ||
      fact->value.signedValue < RCCL_EXECUTION_TRANSPORT_UNKNOWN ||
      fact->value.signedValue >= RCCL_EXECUTION_TRANSPORT_COUNT)
    return false;
  *requestedTransport =
    static_cast<rcclExecutionTransport>(fact->value.signedValue);
  return true;
}

bool rcclAlgoProtoShapeAvailable(const rcclExecutionPolicyValidationContext& validation,
                                 int algorithm, int protocol) {
  if (algorithm < -1 || algorithm >= NCCL_NUM_ALGORITHMS ||
      protocol < -1 || protocol >= NCCL_NUM_PROTOCOLS)
    return false;
  if (algorithm >= 0 && protocol >= 0)
    return validation.algoProtoAvailable[algorithm][protocol];
  if (algorithm >= 0) {
    for (int candidateProtocol = 0; candidateProtocol < NCCL_NUM_PROTOCOLS; candidateProtocol++) {
      if (validation.algoProtoAvailable[algorithm][candidateProtocol]) return true;
    }
    return false;
  }
  if (protocol >= 0) {
    for (int candidateAlgorithm = 0; candidateAlgorithm < NCCL_NUM_ALGORITHMS; candidateAlgorithm++) {
      if (validation.algoProtoAvailable[candidateAlgorithm][protocol]) return true;
    }
    return false;
  }
  return true;
}

bool rcclExecutionPolicyValidationContextIsValid(
  const rcclExecutionPolicyValidationContext* validation) {
  if (validation == nullptr) return true;
  if (validation->minChannels > 0 && validation->maxChannels > 0 &&
      validation->minChannels > validation->maxChannels)
    return false;
  if (validation->validateAlgoProto &&
      (validation->baselineAlgorithm < -1 ||
       validation->baselineAlgorithm >= NCCL_NUM_ALGORITHMS ||
       validation->baselineProtocol < -1 ||
       validation->baselineProtocol >= NCCL_NUM_PROTOCOLS))
    return false;
  return true;
}

// Stage 2b: validate the complete candidate against one immutable snapshot of
// user constraints and resolved capabilities.
rcclExecutionPolicyValidationError rcclValidateCollectiveExecutionPolicy(
  const rcclCollectiveExecutionPolicy& policy, const rcclExecutionPolicyFact* facts,
  size_t factCount, const rcclExecutionPolicyValidationContext* validation) {
  rcclExecutionTransport requestedTransport =
    RCCL_EXECUTION_TRANSPORT_UNKNOWN;
  if (rcclExecutionPolicyRequestedTransportFromFacts(
        facts, factCount, &requestedTransport) &&
      requestedTransport != RCCL_EXECUTION_TRANSPORT_UNKNOWN &&
      policy.transport != RCCL_EXECUTION_TRANSPORT_UNKNOWN &&
      policy.transport != requestedTransport)
    return RCCL_EXECUTION_POLICY_VALIDATION_TRANSPORT_REQUEST_CONFLICT;

  if (validation == nullptr) return RCCL_EXECUTION_POLICY_VALIDATION_NONE;

  rcclExecutionScope scope = RCCL_EXECUTION_SCOPE_COLLECTIVE;
  bool hasScope = rcclExecutionPolicyScopeFromFacts(facts, factCount, &scope);
  if (policy.transferMode != RCCL_P2P_TRANSFER_AUTO &&
      policy.path != RCCL_P2P_PATH_SENDRECV)
    return RCCL_EXECUTION_POLICY_VALIDATION_INVALID_PATH_TRANSFER;
  if (hasScope && scope == RCCL_EXECUTION_SCOPE_COLLECTIVE &&
      (policy.path != RCCL_P2P_PATH_AUTO ||
       policy.transferMode != RCCL_P2P_TRANSFER_AUTO))
    return RCCL_EXECUTION_POLICY_VALIDATION_INVALID_SCOPE_OUTPUT;
  if (hasScope && scope == RCCL_EXECUTION_SCOPE_P2P && policy.algorithm >= 0)
    return RCCL_EXECUTION_POLICY_VALIDATION_INVALID_SCOPE_OUTPUT;

  if (policy.nChannels > 0 &&
      ((validation->minChannels > 0 && policy.nChannels < validation->minChannels) ||
       (validation->maxChannels > 0 && policy.nChannels > validation->maxChannels)))
    return RCCL_EXECUTION_POLICY_VALIDATION_CHANNELS_UNAVAILABLE;

  if (validation->validateAlgoProto &&
      (policy.algorithm >= 0 || policy.protocol >= 0)) {
    int algorithm =
      policy.algorithm >= 0 ? policy.algorithm : validation->baselineAlgorithm;
    int protocol =
      policy.protocol >= 0 ? policy.protocol : validation->baselineProtocol;
    if (!rcclAlgoProtoShapeAvailable(*validation, algorithm, protocol))
      return RCCL_EXECUTION_POLICY_VALIDATION_ALGO_PROTO_UNAVAILABLE;
  }

  if (hasScope && scope == RCCL_EXECUTION_SCOPE_P2P && policy.protocol >= 0 &&
      validation->validateP2pProtocol &&
      (validation->p2pProtocolMask & (1u << policy.protocol)) == 0)
    return RCCL_EXECUTION_POLICY_VALIDATION_P2P_PROTOCOL_UNAVAILABLE;

  if (hasScope && scope == RCCL_EXECUTION_SCOPE_P2P &&
      policy.transferMode != RCCL_P2P_TRANSFER_AUTO &&
      validation->validateP2pTransfer &&
      (validation->p2pTransferMask & (1u << policy.transferMode)) == 0)
    return RCCL_EXECUTION_POLICY_VALIDATION_P2P_TRANSFER_UNAVAILABLE;

  if (policy.transport != RCCL_EXECUTION_TRANSPORT_UNKNOWN &&
      validation->validateTransport &&
      (validation->transportMask & (1u << policy.transport)) == 0)
    return RCCL_EXECUTION_POLICY_VALIDATION_TRANSPORT_UNAVAILABLE;

  return RCCL_EXECUTION_POLICY_VALIDATION_NONE;
}

using rcclExecutionPolicyRuleAt =
  bool (*)(const void* ruleContext, size_t ruleId, rcclCollectiveExecutionRule* rule);

// Adapts caller-provided generic rule arrays to the resolver.
bool rcclArrayExecutionPolicyRuleAt(const void* ruleContext, size_t ruleId,
                                    rcclCollectiveExecutionRule* rule) {
  if (ruleContext == nullptr || rule == nullptr) return false;
  *rule = static_cast<const rcclCollectiveExecutionRule*>(ruleContext)[ruleId];
  return true;
}

// Adapts the pre-expanded architecture rule table to the resolver.
bool rcclBuiltInExecutionPolicyRuleAt(const void*, size_t ruleId, rcclCollectiveExecutionRule* rule) {
  const rcclExecutionPolicyRules::RuleTableView& table =
    rcclExecutionPolicyRules::kCollectiveExecutionRuleTable;
  if (rule == nullptr || ruleId >= table.count) return false;
  const rcclExecutionPolicyRules::RuleRow& row = table.rows[ruleId];
  *rule = {row.conditions, row.conditionCount, row.assignments, row.assignmentCount};
  return true;
}

void rcclDefaultExecutionPolicyResolution(rcclExecutionPolicyResolution* resolution) {
  resolution->status = RCCL_EXECUTION_POLICY_NO_MATCH;
  resolution->validationError = RCCL_EXECUTION_POLICY_VALIDATION_NONE;
  resolution->selectedRuleId = RCCL_EXECUTION_POLICY_NO_RULE;
  resolution->firstRejectedRuleId = RCCL_EXECUTION_POLICY_NO_RULE;
  resolution->firstRejectedError = RCCL_EXECUTION_POLICY_VALIDATION_NONE;
  rcclDefaultCollectiveExecutionPolicy(&resolution->policy);
}

void rcclRecordRejectedExecutionPolicyRule(
  rcclExecutionPolicyResolution* resolution, size_t ruleId,
  rcclExecutionPolicyValidationError error) {
  if (resolution->firstRejectedRuleId != RCCL_EXECUTION_POLICY_NO_RULE) return;
  resolution->firstRejectedRuleId = static_cast<rcclExecutionPolicyRuleId>(ruleId);
  resolution->firstRejectedError = error;
}

// Stage 3: select the lowest-index valid candidate without merging profiles.
void rcclSelectCollectiveExecutionPolicy(
  const rcclExecutionPolicyFact* facts, size_t factCount, size_t ruleCount,
  rcclExecutionPolicyRuleAt ruleAt, const void* ruleContext,
  const rcclExecutionPolicyValidationContext* validation,
  rcclExecutionPolicyResolution* resolution) {
  rcclDefaultExecutionPolicyResolution(resolution);
  if (ruleAt == nullptr || ruleCount > RCCL_EXECUTION_POLICY_NO_RULE) {
    resolution->status = RCCL_EXECUTION_POLICY_INVALID_INPUT;
    return;
  }

  bool selected = false;
  bool matched = false;
  rcclCollectiveExecutionPolicy selectedPolicy;
  rcclDefaultCollectiveExecutionPolicy(&selectedPolicy);
  for (size_t ruleId = 0; ruleId < ruleCount; ruleId++) {
    rcclCollectiveExecutionRule rule;
    if (!ruleAt(ruleContext, ruleId, &rule) ||
        !rcclCollectiveExecutionRuleMatches(rule, facts, factCount))
      continue;
    matched = true;

    rcclCollectiveExecutionPolicy candidate;
    rcclExecutionPolicyValidationError error =
      rcclValidateCollectiveExecutionRule(rule, &candidate);
    if (error == RCCL_EXECUTION_POLICY_VALIDATION_NONE)
      error = rcclValidateCollectiveExecutionPolicy(
        candidate, facts, factCount, validation);
    if (error != RCCL_EXECUTION_POLICY_VALIDATION_NONE) {
      rcclRecordRejectedExecutionPolicyRule(resolution, ruleId, error);
      continue;
    }

    // Keep evaluating matched rules so matching and validation remain
    // independent stages. Selection itself is deterministic: the first valid
    // candidate has the lowest index and therefore the highest priority.
    if (!selected) {
      selectedPolicy = candidate;
      resolution->selectedRuleId =
        static_cast<rcclExecutionPolicyRuleId>(ruleId);
      selected = true;
    }
  }

  if (selected) {
    resolution->policy = selectedPolicy;
    resolution->status = RCCL_EXECUTION_POLICY_SELECTED;
  } else if (matched) {
    resolution->status = RCCL_EXECUTION_POLICY_VALIDATION_FAILED;
    resolution->validationError = resolution->firstRejectedError;
  }
}

// Matches only stable communicator facts used during channel provisioning.
bool rcclCollectiveExecutionRuleMatchesProvisioningFacts(const rcclCollectiveExecutionRule& rule,
                                                         const rcclExecutionPolicyFact* facts,
                                                         size_t factCount) {
  for (size_t i = 0; i < rule.conditionCount; i++) {
    const rcclExecutionPolicyCondition& condition = rule.conditions[i];
    if (condition.field == RCCL_EXECUTION_INPUT_SCOPE ||
        condition.field == RCCL_EXECUTION_INPUT_COLL_TYPE ||
        condition.field == RCCL_EXECUTION_INPUT_DATA_SIZE ||
        condition.field == RCCL_EXECUTION_INPUT_N_CHANNELS ||
        condition.field == RCCL_EXECUTION_INPUT_IN_PLACE)
      continue;
    const rcclExecutionPolicyFact* fact = rcclFindExecutionPolicyFact(facts, factCount, condition.field);
    if (fact == nullptr || !rcclExecutionPolicyValueMatches(fact->value, condition.op, condition.value)) return false;
  }
  return true;
}

} // namespace

// Detects overlap between two non-empty host address ranges.
bool rcclBuffersOverlap(const void* firstBuffer, size_t firstBytes, const void* secondBuffer, size_t secondBytes) {
  if (firstBuffer == nullptr || secondBuffer == nullptr || firstBytes == 0 || secondBytes == 0) return false;
  uintptr_t first = reinterpret_cast<uintptr_t>(firstBuffer);
  uintptr_t second = reinterpret_cast<uintptr_t>(secondBuffer);
  return first <= second ? second - first < firstBytes : first - second < secondBytes;
}

// Initializes a policy to preserve all normal RCCL decisions.
void rcclDefaultCollectiveExecutionPolicy(struct rcclCollectiveExecutionPolicy* policy) {
  if (policy != nullptr)
    *policy = {-1, -1, -1, RCCL_P2P_PATH_AUTO, RCCL_P2P_TRANSFER_AUTO,
               RCCL_EXECUTION_TRANSPORT_UNKNOWN};
}

void rcclDefaultExecutionPolicyValidationContext(
  struct rcclExecutionPolicyValidationContext* context) {
  if (context == nullptr) return;
  memset(context, 0, sizeof(*context));
  context->baselineAlgorithm = -1;
  context->baselineProtocol = -1;
}

// Resolves an arbitrary caller-provided rule array with an optional immutable
// validation snapshot.
struct rcclExecutionPolicyResolution rcclEvaluateCollectiveExecutionRulesWithValidation(
  const struct rcclExecutionPolicyFact* facts, size_t factCount,
  const struct rcclCollectiveExecutionRule* rules, size_t ruleCount,
  const struct rcclExecutionPolicyValidationContext* validation) {
  rcclExecutionPolicyResolution resolution;
  rcclDefaultExecutionPolicyResolution(&resolution);
  if ((factCount != 0 && facts == nullptr) ||
      (ruleCount != 0 && rules == nullptr) ||
      !rcclExecutionPolicyValidationContextIsValid(validation)) {
    resolution.status = RCCL_EXECUTION_POLICY_INVALID_INPUT;
    return resolution;
  }
  rcclSelectCollectiveExecutionPolicy(
    facts, factCount, ruleCount, rcclArrayExecutionPolicyRuleAt, rules,
    validation, &resolution);
  return resolution;
}

bool rcclEvaluateCollectiveExecutionRules(
  const struct rcclExecutionPolicyFact* facts, size_t factCount,
  const struct rcclCollectiveExecutionRule* rules, size_t ruleCount,
  struct rcclCollectiveExecutionPolicy* policy,
  rcclExecutionPolicyRuleId* selectedRuleId) {
  if (selectedRuleId != nullptr)
    *selectedRuleId = RCCL_EXECUTION_POLICY_NO_RULE;
  if (policy == nullptr) return false;
  rcclExecutionPolicyResolution resolution =
    rcclEvaluateCollectiveExecutionRulesWithValidation(
      facts, factCount, rules, ruleCount, nullptr);
  *policy = resolution.policy;
  if (selectedRuleId != nullptr)
    *selectedRuleId = resolution.selectedRuleId;
  return resolution.status == RCCL_EXECUTION_POLICY_SELECTED;
}

// Normalizes a policy input and resolves the built-in architecture rules with
// an optional immutable validation snapshot.
struct rcclExecutionPolicyResolution rcclResolveCollectiveExecutionPolicyWithValidation(
  const struct rcclCollectivePolicyInput* input,
  const struct rcclExecutionPolicyValidationContext* validation) {
  rcclExecutionPolicyResolution resolution;
  rcclDefaultExecutionPolicyResolution(&resolution);
  if (input == nullptr ||
      input->transport < RCCL_EXECUTION_TRANSPORT_UNKNOWN ||
      input->transport >= RCCL_EXECUTION_TRANSPORT_COUNT ||
      input->requestedTransport < RCCL_EXECUTION_TRANSPORT_UNKNOWN ||
      input->requestedTransport >= RCCL_EXECUTION_TRANSPORT_COUNT ||
      !rcclExecutionPolicyValidationContextIsValid(validation)) {
    resolution.status = RCCL_EXECUTION_POLICY_INVALID_INPUT;
    return resolution;
  }

  const rcclExecutionPolicyFact inputFacts[] = {
    {RCCL_EXECUTION_INPUT_SCOPE, rcclExecutionPolicyValue::Signed(input->scope)},
    {RCCL_EXECUTION_INPUT_GFX_ARCH, rcclExecutionPolicyValue::String(input->gfxArch)},
    {RCCL_EXECUTION_INPUT_COLL_TYPE, rcclExecutionPolicyValue::Signed(static_cast<int64_t>(input->collType))},
    {RCCL_EXECUTION_INPUT_DATA_SIZE, rcclExecutionPolicyValue::Unsigned(input->dataSize)},
    {RCCL_EXECUTION_INPUT_N_NODES, rcclExecutionPolicyValue::Signed(input->nNodes)},
    {RCCL_EXECUTION_INPUT_N_RANKS, rcclExecutionPolicyValue::Signed(input->nRanks)},
    {RCCL_EXECUTION_INPUT_N_CHANNELS, rcclExecutionPolicyValue::Signed(input->nChannels)},
    {RCCL_EXECUTION_INPUT_TRANSPORT, rcclExecutionPolicyValue::Signed(input->transport)},
    {RCCL_EXECUTION_INPUT_IN_PLACE, rcclExecutionPolicyValue::Boolean(input->inPlace)},
    {RCCL_EXECUTION_INPUT_REQUESTED_TRANSPORT,
     rcclExecutionPolicyValue::Signed(input->requestedTransport)},
  };
  rcclSelectCollectiveExecutionPolicy(
    inputFacts, sizeof(inputFacts) / sizeof(inputFacts[0]),
    rcclExecutionPolicyRules::kCollectiveExecutionRuleTable.count, rcclBuiltInExecutionPolicyRuleAt, nullptr,
    validation, &resolution);
  return resolution;
}

bool rcclResolveCollectiveExecutionPolicy(
  const struct rcclCollectivePolicyInput* input,
  struct rcclCollectiveExecutionPolicy* policy,
  rcclExecutionPolicyRuleId* selectedRuleId) {
  if (selectedRuleId != nullptr)
    *selectedRuleId = RCCL_EXECUTION_POLICY_NO_RULE;
  if (policy == nullptr) return false;
  rcclExecutionPolicyResolution resolution =
    rcclResolveCollectiveExecutionPolicyWithValidation(input, nullptr);
  *policy = resolution.policy;
  if (selectedRuleId != nullptr)
    *selectedRuleId = resolution.selectedRuleId;
  return resolution.status == RCCL_EXECUTION_POLICY_SELECTED;
}

// Computes the maximum channel pool needed by matching built-in rules.
int rcclGetCollectiveExecutionPolicyRequiredChannels(struct ncclComm* comm, bool p2pDisabled) {
  if (comm == nullptr) return 0;
  rcclExecutionTransport requestedTransport =
    rcclRequestedExecutionTransport(p2pDisabled);
  const rcclExecutionPolicyFact provisioningFacts[] = {
    {RCCL_EXECUTION_INPUT_SCOPE, rcclExecutionPolicyValue::Signed(RCCL_EXECUTION_SCOPE_COLLECTIVE)},
    {RCCL_EXECUTION_INPUT_GFX_ARCH, rcclExecutionPolicyValue::String(comm->archName)},
    {RCCL_EXECUTION_INPUT_N_NODES, rcclExecutionPolicyValue::Signed(comm->nNodes)},
    {RCCL_EXECUTION_INPUT_N_RANKS, rcclExecutionPolicyValue::Signed(comm->nRanks)},
    {RCCL_EXECUTION_INPUT_TRANSPORT,
     rcclExecutionPolicyValue::Signed(
       rcclConfiguredExecutionTransport(p2pDisabled))},
    {RCCL_EXECUTION_INPUT_REQUESTED_TRANSPORT,
     rcclExecutionPolicyValue::Signed(requestedTransport)},
  };

  int requiredChannels = 0;
  const rcclExecutionPolicyRules::RuleTableView& table =
    rcclExecutionPolicyRules::kCollectiveExecutionRuleTable;
  for (size_t ruleId = 0; ruleId < table.count; ruleId++) {
    const rcclExecutionPolicyRules::RuleRow& row = table.rows[ruleId];
    const rcclCollectiveExecutionRule ruleView = {
      row.conditions,
      row.conditionCount,
      row.assignments,
      row.assignmentCount,
    };
    if (!rcclCollectiveExecutionRuleMatchesProvisioningFacts(
          ruleView, provisioningFacts, sizeof(provisioningFacts) / sizeof(provisioningFacts[0])))
      continue;
    bool p2pScope = false;
    bool remapsActivePool = false;
    bool conflictsWithRequestedTransport = false;
    for (size_t conditionIndex = 0; conditionIndex < ruleView.conditionCount; conditionIndex++) {
      const rcclExecutionPolicyCondition& condition = ruleView.conditions[conditionIndex];
      p2pScope =
        p2pScope ||
        (condition.field == RCCL_EXECUTION_INPUT_SCOPE &&
         condition.value.type == RCCL_EXECUTION_VALUE_SIGNED &&
         condition.value.signedValue == RCCL_EXECUTION_SCOPE_P2P);
    }
    for (size_t assignmentIndex = 0; assignmentIndex < ruleView.assignmentCount; assignmentIndex++) {
      const rcclExecutionPolicyAssignment& assignment = ruleView.assignments[assignmentIndex];
      remapsActivePool =
        remapsActivePool ||
        (assignment.field == RCCL_EXECUTION_OUTPUT_PATH &&
         assignment.value.type == RCCL_EXECUTION_VALUE_SIGNED &&
         assignment.value.signedValue == RCCL_P2P_PATH_SENDRECV);
      conflictsWithRequestedTransport =
        conflictsWithRequestedTransport ||
        (requestedTransport != RCCL_EXECUTION_TRANSPORT_UNKNOWN &&
         assignment.field == RCCL_EXECUTION_OUTPUT_TRANSPORT &&
         assignment.value.type == RCCL_EXECUTION_VALUE_SIGNED &&
         assignment.value.signedValue != requestedTransport);
    }
    if (conflictsWithRequestedTransport) continue;
    // Direction-local P2P caps such as gfx110x's one-channel AllToAll rule
    // operate within the communicator's existing pool and require no widening.
    if (p2pScope && !remapsActivePool) continue;
    for (size_t assignmentIndex = 0; assignmentIndex < ruleView.assignmentCount; assignmentIndex++) {
      const rcclExecutionPolicyAssignment& assignment = ruleView.assignments[assignmentIndex];
      if (assignment.field != RCCL_EXECUTION_OUTPUT_N_CHANNELS ||
          assignment.value.type != RCCL_EXECUTION_VALUE_SIGNED)
        continue;
      int64_t value = assignment.value.signedValue;
      if (value > requiredChannels && value <= MAXCHANNELS) requiredChannels = static_cast<int>(value);
    }
  }
  return requiredChannels;
}

// Adapts read-only communicator state to the policy resolver input and optional
// normalized validation snapshot.
rcclExecutionPolicyResolution rcclGetCollectiveExecutionPolicyWithValidation(
  const struct ncclComm* comm, enum rcclExecutionScope scope,
  ncclFunc_t collType, size_t dataSize, bool p2pDisabled, bool inPlace,
  const rcclExecutionPolicyValidationContext* validation) {
  if (comm == nullptr) {
    rcclExecutionPolicyResolution resolution;
    rcclDefaultExecutionPolicyResolution(&resolution);
    resolution.status = RCCL_EXECUTION_POLICY_INVALID_INPUT;
    return resolution;
  }
  rcclExecutionPolicyValidationContext transportValidation;
  if (validation == nullptr) {
    rcclDefaultExecutionPolicyValidationContext(&transportValidation);
    transportValidation.validateTransport = true;
    transportValidation.transportMask =
      rcclAvailableExecutionTransportMask(comm, p2pDisabled);
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
    rcclConfiguredExecutionTransport(p2pDisabled),
    inPlace,
    rcclRequestedExecutionTransport(p2pDisabled),
  };
  return rcclResolveCollectiveExecutionPolicyWithValidation(&input, validation);
}

bool rcclGetCollectiveExecutionPolicy(const struct ncclComm* comm, enum rcclExecutionScope scope,
                                      ncclFunc_t collType, size_t dataSize, bool p2pDisabled, bool inPlace,
                                      struct rcclCollectiveExecutionPolicy* policy) {
  if (policy == nullptr) return false;
  rcclExecutionPolicyResolution resolution =
    rcclGetCollectiveExecutionPolicyWithValidation(
      comm, scope, collType, dataSize, p2pDisabled, inPlace, nullptr);
  *policy = resolution.policy;
  return resolution.status == RCCL_EXECUTION_POLICY_SELECTED;
}

extern int64_t ncclParamMinNchannels();
extern int64_t ncclParamMaxNchannels();
extern int64_t ncclParamP2pDisable();
extern int64_t ncclParamP2pReadEnable();

bool rcclExecutionPolicyForP2pTask(
  struct ncclComm* comm, struct ncclTaskP2p* task, bool p2pDisabled,
  struct rcclCollectiveExecutionPolicy* policy) {
  if (comm == nullptr || task == nullptr || policy == nullptr || comm->nRanks <= 0)
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

  // NCCL_PROTO parsing has already normalized protocol filters into the
  // communicator's tuning bandwidths. Some P2P-only collective kinds have no
  // tuning entries; leave this check disabled when no protocol mask exists.
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
    rcclAvailableExecutionTransportMask(comm, p2pDisabled);

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

static void rcclBuildCollectiveExecutionPolicyValidationContext(
  const struct ncclComm* comm, const struct ncclTaskColl* info,
  float table[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS],
  struct rcclExecutionPolicyValidationContext* validation) {
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
    rcclAvailableExecutionTransportMask(comm, ncclParamP2pDisable());
  for (int algorithm = 0; algorithm < NCCL_NUM_ALGORITHMS; algorithm++) {
    bool algorithmAllowedByTask =
      info->algMask == 0 ||
      (info->algMask & (uint64_t{1} << algorithm)) != 0;
    for (int protocol = 0; protocol < NCCL_NUM_PROTOCOLS; protocol++) {
      // NCCL_ALGO/NCCL_PROTO, topology capabilities, and tuner constraints
      // have already been normalized into the cost table. The task mask
      // carries the resolved per-call algorithm selection.
      validation->algoProtoAvailable[algorithm][protocol] =
        algorithmAllowedByTask && table[algorithm][protocol] >= 0.0f;
    }
  }
}

bool rcclApplyCollectiveExecutionPolicy(
  struct ncclComm* comm, struct ncclTaskColl* info, size_t dataSize,
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
      ncclParamP2pDisable(), /*inPlace=*/false, &validation);
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
