/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collective_execution_policy.h"
#include "collective_execution_policy_internal.h"

#include "archinfo.h"
#include "device.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

namespace {

// Policy resolution architecture
// ------------------------------
// A fact is an immutable, typed observation available during one resolution.
// Facts may be copied from operation arguments (scope, collective, data size),
// read from communicator state (architecture, node/rank/channel counts), or
// derived by the caller (available/requested transports and buffer overlap). They
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
//    every later match, but later candidates cannot override the winner. When
//    no full profile is valid, a second scan retries the matched rules that
//    have optional assignments without them.
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
  case RCCL_EXECUTION_COMPARE_EQ:
    return comparison == 0;
  case RCCL_EXECUTION_COMPARE_NE:
    return comparison != 0;
  case RCCL_EXECUTION_COMPARE_LT:
    return comparison < 0;
  case RCCL_EXECUTION_COMPARE_LE:
    return comparison <= 0;
  case RCCL_EXECUTION_COMPARE_GT:
    return comparison > 0;
  case RCCL_EXECUTION_COMPARE_GE:
    return comparison >= 0;
  default:
    return false;
  }
}

// Compares two policy values, including architecture-pattern matching.
bool rcclExecutionPolicyValueMatches(const rcclExecutionPolicyValue& actual, enum rcclExecutionPolicyCompareOp op,
                                     const rcclExecutionPolicyValue& expected) {
  if (actual.type != expected.type) return false;

  switch (actual.type) {
  case RCCL_EXECUTION_VALUE_SIGNED:
    if (op == RCCL_EXECUTION_COMPARE_MASK_CONTAINS) return false;
    return rcclCompareOrdered((actual.signedValue > expected.signedValue) - (actual.signedValue < expected.signedValue),
                              op);
  case RCCL_EXECUTION_VALUE_UNSIGNED:
    if (op == RCCL_EXECUTION_COMPARE_MASK_CONTAINS)
      return (actual.unsignedValue & expected.unsignedValue) == expected.unsignedValue;
    return rcclCompareOrdered(
      (actual.unsignedValue > expected.unsignedValue) - (actual.unsignedValue < expected.unsignedValue), op);
  case RCCL_EXECUTION_VALUE_STRING:
    if (actual.stringValue == nullptr || expected.stringValue == nullptr) return false;
    if (op == RCCL_EXECUTION_COMPARE_ARCH_MATCH) return IsArchMatch(actual.stringValue, expected.stringValue);
    if (op == RCCL_EXECUTION_COMPARE_MASK_CONTAINS) return false;
    return rcclCompareOrdered(strcmp(actual.stringValue, expected.stringValue), op);
  case RCCL_EXECUTION_VALUE_BOOL:
    if (op == RCCL_EXECUTION_COMPARE_MASK_CONTAINS) return false;
    return rcclCompareOrdered(static_cast<int>(actual.boolValue) - static_cast<int>(expected.boolValue), op);
  }
  return false;
}

bool rcclExecutionTransportMaskIsValid(uint32_t mask) {
  return mask != 0 && (mask & ~RCCL_EXECUTION_TRANSPORT_VALID_MASK) == 0;
}

uint32_t rcclEligibleExecutionTransportMask(uint32_t availableMask, rcclExecutionTransportIntent intent) {
  switch (intent) {
  case RCCL_EXECUTION_TRANSPORT_INTENT_AUTO:
    return availableMask;
  case RCCL_EXECUTION_TRANSPORT_INTENT_IPC:
    return availableMask & rcclExecutionTransportBit(RCCL_EXECUTION_TRANSPORT_IPC);
  case RCCL_EXECUTION_TRANSPORT_INTENT_SHM:
    return availableMask & rcclExecutionTransportBit(RCCL_EXECUTION_TRANSPORT_SHM);
  default:
    return 0;
  }
}

constexpr uint32_t rcclExecutionPolicyFieldBit(rcclExecutionPolicyInputField field) {
  return uint32_t{1} << static_cast<unsigned>(field);
}

// Stage 1: match one rule against the normalized input facts. Conditions on
// ignoredFields always hold.
bool rcclCollectiveExecutionRuleMatches(const rcclCollectiveExecutionRule& rule, const rcclExecutionPolicyFact* facts,
                                        size_t factCount, uint32_t ignoredFields = 0) {
  if (rule.conditionCount != 0 && rule.conditions == nullptr) return false;
  for (size_t i = 0; i < rule.conditionCount; i++) {
    const rcclExecutionPolicyCondition& condition = rule.conditions[i];
    if (ignoredFields & rcclExecutionPolicyFieldBit(condition.field)) continue;
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
    if (value <= RCCL_EXECUTION_TRANSPORT_UNKNOWN || value >= RCCL_EXECUTION_TRANSPORT_COUNT) return false;
    policy->transport = static_cast<enum rcclExecutionTransport>(value);
    return true;
  }
  return false;
}

// Stage 2a: structurally validate one matched rule and materialize its atomic
// candidate. Semantic validation runs only after every assignment succeeds.
rcclExecutionPolicyValidationError rcclValidateCollectiveExecutionRule(
  const rcclCollectiveExecutionRule& rule, bool skipOptional, rcclCollectiveExecutionPolicy* policy) {
  if (policy == nullptr || (rule.assignmentCount != 0 && rule.assignments == nullptr))
    return RCCL_EXECUTION_POLICY_VALIDATION_INVALID_ASSIGNMENT;

  rcclCollectiveExecutionPolicy candidate;
  rcclDefaultCollectiveExecutionPolicy(&candidate);
  for (size_t assignmentIndex = 0; assignmentIndex < rule.assignmentCount; assignmentIndex++) {
    const rcclExecutionPolicyAssignment& assignment = rule.assignments[assignmentIndex];
    if (skipOptional && assignment.optional) continue;
    if (!rcclApplyCollectivePolicyAssignment(assignment, &candidate))
      return RCCL_EXECUTION_POLICY_VALIDATION_INVALID_ASSIGNMENT;
  }
  *policy = candidate;
  return RCCL_EXECUTION_POLICY_VALIDATION_NONE;
}

bool rcclRuleHasOptionalAssignment(const rcclCollectiveExecutionRule& rule) {
  for (size_t i = 0; i < rule.assignmentCount; i++)
    if (rule.assignments[i].optional) return true;
  return false;
}

bool rcclExecutionPolicyScopeFromFacts(const rcclExecutionPolicyFact* facts, size_t factCount,
                                       rcclExecutionScope* scope) {
  const rcclExecutionPolicyFact* fact = rcclFindExecutionPolicyFact(facts, factCount, RCCL_EXECUTION_INPUT_SCOPE);
  if (fact == nullptr || fact->value.type != RCCL_EXECUTION_VALUE_SIGNED ||
      fact->value.signedValue < RCCL_EXECUTION_SCOPE_P2P || fact->value.signedValue > RCCL_EXECUTION_SCOPE_COLLECTIVE)
    return false;
  *scope = static_cast<rcclExecutionScope>(fact->value.signedValue);
  return true;
}

bool rcclExecutionPolicyTransportIntentFromFacts(const rcclExecutionPolicyFact* facts, size_t factCount,
                                                 rcclExecutionTransportIntent* transportIntent) {
  const rcclExecutionPolicyFact* fact =
    rcclFindExecutionPolicyFact(facts, factCount, RCCL_EXECUTION_INPUT_TRANSPORT_INTENT);
  if (fact == nullptr || fact->value.type != RCCL_EXECUTION_VALUE_SIGNED ||
      fact->value.signedValue < RCCL_EXECUTION_TRANSPORT_INTENT_AUTO ||
      fact->value.signedValue >= RCCL_EXECUTION_TRANSPORT_INTENT_COUNT)
    return false;
  *transportIntent = static_cast<rcclExecutionTransportIntent>(fact->value.signedValue);
  return true;
}

rcclExecutionTransport rcclExecutionPolicyRequestedTransport(rcclExecutionTransportIntent intent) {
  switch (intent) {
  case RCCL_EXECUTION_TRANSPORT_INTENT_IPC:
    return RCCL_EXECUTION_TRANSPORT_IPC;
  case RCCL_EXECUTION_TRANSPORT_INTENT_SHM:
    return RCCL_EXECUTION_TRANSPORT_SHM;
  default:
    return RCCL_EXECUTION_TRANSPORT_UNKNOWN;
  }
}

bool rcclAlgoProtoShapeAvailable(const bool available[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS], int algorithm,
                                 int protocol) {
  if (algorithm < -1 || algorithm >= NCCL_NUM_ALGORITHMS || protocol < -1 || protocol >= NCCL_NUM_PROTOCOLS)
    return false;
  if (algorithm >= 0 && protocol >= 0) return available[algorithm][protocol];
  for (int candidateAlgorithm = 0; candidateAlgorithm < NCCL_NUM_ALGORITHMS; candidateAlgorithm++) {
    if (algorithm >= 0 && candidateAlgorithm != algorithm) continue;
    for (int candidateProtocol = 0; candidateProtocol < NCCL_NUM_PROTOCOLS; candidateProtocol++) {
      if (protocol >= 0 && candidateProtocol != protocol) continue;
      if (available[candidateAlgorithm][candidateProtocol]) return true;
    }
  }
  return false;
}

bool rcclExecutionPolicyValidationContextIsValid(const rcclExecutionPolicyValidationContext* validation) {
  if (validation == nullptr) return true;
  if (validation->minChannels > 0 && validation->maxChannels > 0 && validation->minChannels > validation->maxChannels)
    return false;
  if (validation->validateAlgoProto &&
      (validation->baselineAlgorithm < -1 || validation->baselineAlgorithm >= NCCL_NUM_ALGORITHMS ||
       validation->baselineProtocol < -1 || validation->baselineProtocol >= NCCL_NUM_PROTOCOLS))
    return false;
  if (validation->validateTransport && !rcclExecutionTransportMaskIsValid(validation->transportMask)) return false;
  return true;
}

// Stage 2b: validate the complete candidate against one immutable snapshot of
// user constraints and resolved capabilities.
rcclExecutionPolicyValidationError rcclValidateCollectiveExecutionPolicy(
  const rcclCollectiveExecutionPolicy& policy, const rcclExecutionPolicyFact* facts, size_t factCount,
  const rcclExecutionPolicyValidationContext* validation) {
  rcclExecutionTransportIntent transportIntent = RCCL_EXECUTION_TRANSPORT_INTENT_AUTO;
  rcclExecutionTransport requestedTransport = RCCL_EXECUTION_TRANSPORT_UNKNOWN;
  if (rcclExecutionPolicyTransportIntentFromFacts(facts, factCount, &transportIntent))
    requestedTransport = rcclExecutionPolicyRequestedTransport(transportIntent);
  if (requestedTransport != RCCL_EXECUTION_TRANSPORT_UNKNOWN && policy.transport != RCCL_EXECUTION_TRANSPORT_UNKNOWN &&
      policy.transport != requestedTransport)
    return RCCL_EXECUTION_POLICY_VALIDATION_TRANSPORT_REQUEST_CONFLICT;

  if (validation == nullptr) return RCCL_EXECUTION_POLICY_VALIDATION_NONE;

  rcclExecutionScope scope = RCCL_EXECUTION_SCOPE_COLLECTIVE;
  bool hasScope = rcclExecutionPolicyScopeFromFacts(facts, factCount, &scope);
  if (policy.transferMode != RCCL_P2P_TRANSFER_AUTO && policy.path != RCCL_P2P_PATH_SENDRECV)
    return RCCL_EXECUTION_POLICY_VALIDATION_INVALID_PATH_TRANSFER;
  if (hasScope && scope == RCCL_EXECUTION_SCOPE_COLLECTIVE &&
      (policy.path != RCCL_P2P_PATH_AUTO || policy.transferMode != RCCL_P2P_TRANSFER_AUTO))
    return RCCL_EXECUTION_POLICY_VALIDATION_INVALID_SCOPE_OUTPUT;
  if (hasScope && scope == RCCL_EXECUTION_SCOPE_P2P && policy.algorithm >= 0)
    return RCCL_EXECUTION_POLICY_VALIDATION_INVALID_SCOPE_OUTPUT;

  if (policy.nChannels > 0 && ((validation->minChannels > 0 && policy.nChannels < validation->minChannels) ||
                               (validation->maxChannels > 0 && policy.nChannels > validation->maxChannels)))
    return RCCL_EXECUTION_POLICY_VALIDATION_CHANNELS_UNAVAILABLE;

  if (validation->validateAlgoProto && (policy.algorithm >= 0 || policy.protocol >= 0)) {
    int algorithm = policy.algorithm >= 0 ? policy.algorithm : validation->baselineAlgorithm;
    int protocol = policy.protocol >= 0 ? policy.protocol : validation->baselineProtocol;
    if (!rcclAlgoProtoShapeAvailable(validation->algoProtoAvailable, algorithm, protocol))
      return RCCL_EXECUTION_POLICY_VALIDATION_ALGO_PROTO_UNAVAILABLE;
  }

  if (hasScope && scope == RCCL_EXECUTION_SCOPE_P2P && policy.protocol >= 0 && validation->validateP2pProtocol &&
      (validation->p2pProtocolMask & (1u << policy.protocol)) == 0)
    return RCCL_EXECUTION_POLICY_VALIDATION_P2P_PROTOCOL_UNAVAILABLE;

  if (hasScope && scope == RCCL_EXECUTION_SCOPE_P2P && policy.transferMode != RCCL_P2P_TRANSFER_AUTO &&
      validation->validateP2pTransfer && (validation->p2pTransferMask & (1u << policy.transferMode)) == 0)
    return RCCL_EXECUTION_POLICY_VALIDATION_P2P_TRANSFER_UNAVAILABLE;

  if (policy.transport != RCCL_EXECUTION_TRANSPORT_UNKNOWN && validation->validateTransport &&
      (validation->transportMask & rcclExecutionTransportBit(policy.transport)) == 0)
    return RCCL_EXECUTION_POLICY_VALIDATION_TRANSPORT_UNAVAILABLE;

  if (policy.transport != RCCL_EXECUTION_TRANSPORT_UNKNOWN && policy.transport < RCCL_EXECUTION_TRANSPORT_COUNT &&
      validation->validateTransportCapabilities && hasScope) {
    if (scope == RCCL_EXECUTION_SCOPE_COLLECTIVE) {
      int algorithm = policy.algorithm >= 0 ? policy.algorithm : validation->baselineAlgorithm;
      int protocol = policy.protocol >= 0 ? policy.protocol : validation->baselineProtocol;
      if (!rcclAlgoProtoShapeAvailable(validation->transportAlgoProtoAvailable[policy.transport], algorithm, protocol))
        return RCCL_EXECUTION_POLICY_VALIDATION_TRANSPORT_CAPABILITY_UNAVAILABLE;
    } else if (policy.transferMode != RCCL_P2P_TRANSFER_AUTO &&
               (validation->transportP2pTransferMask[policy.transport] & (1u << policy.transferMode)) == 0) {
      return RCCL_EXECUTION_POLICY_VALIDATION_TRANSPORT_CAPABILITY_UNAVAILABLE;
    }
  }

  return RCCL_EXECUTION_POLICY_VALIDATION_NONE;
}

using rcclExecutionPolicyRuleAt = bool (*)(const void* ruleContext, size_t ruleId, rcclCollectiveExecutionRule* rule);

// Adapts caller-provided generic rule arrays to the resolver.
bool rcclArrayExecutionPolicyRuleAt(const void* ruleContext, size_t ruleId, rcclCollectiveExecutionRule* rule) {
  if (ruleContext == nullptr || rule == nullptr) return false;
  *rule = static_cast<const rcclCollectiveExecutionRule*>(ruleContext)[ruleId];
  return true;
}

// Adapts the pre-expanded architecture rule table to the resolver.
bool rcclBuiltInExecutionPolicyRuleAt(const void*, size_t ruleId, rcclCollectiveExecutionRule* rule) {
  const rcclExecutionPolicyRules::RuleTableView& table = rcclExecutionPolicyRules::kCollectiveExecutionRuleTable;
  if (rule == nullptr || ruleId >= table.count) return false;
  const rcclExecutionPolicyRules::RuleRow& row = table.rows[ruleId];
  *rule = {row.conditions, row.conditionCount, row.assignments, row.assignmentCount};
  return true;
}

void rcclRecordRejectedExecutionPolicyRule(rcclExecutionPolicyResolution* resolution, size_t ruleId,
                                           rcclExecutionPolicyValidationError error) {
  if (resolution->firstRejectedRuleId != RCCL_EXECUTION_POLICY_NO_RULE) return;
  resolution->firstRejectedRuleId = static_cast<rcclExecutionPolicyRuleId>(ruleId);
  resolution->firstRejectedError = error;
}

// Stage 3: select the lowest-index valid candidate without merging profiles.
void rcclSelectCollectiveExecutionPolicy(const rcclExecutionPolicyFact* facts, size_t factCount, size_t ruleCount,
                                         rcclExecutionPolicyRuleAt ruleAt, const void* ruleContext,
                                         const rcclExecutionPolicyValidationContext* validation,
                                         rcclExecutionPolicyResolution* resolution) {
  rcclInitExecutionPolicyResolution(resolution, RCCL_EXECUTION_POLICY_NO_MATCH);
  if (ruleAt == nullptr || ruleCount > RCCL_EXECUTION_POLICY_NO_RULE) {
    resolution->status = RCCL_EXECUTION_POLICY_INVALID_INPUT;
    return;
  }

  bool selected = false;
  bool matched = false;
  bool hasOptional = false;
  rcclCollectiveExecutionPolicy selectedPolicy;
  rcclDefaultCollectiveExecutionPolicy(&selectedPolicy);
  for (int pass = 0; pass < 2 && !selected; pass++) {
    bool skipOptional = pass == 1;
    if (skipOptional && !hasOptional) break;
    for (size_t ruleId = 0; ruleId < ruleCount; ruleId++) {
      rcclCollectiveExecutionRule rule;
      if (!ruleAt(ruleContext, ruleId, &rule) || !rcclCollectiveExecutionRuleMatches(rule, facts, factCount)) continue;
      matched = true;
      bool ruleHasOptional = rcclRuleHasOptionalAssignment(rule);
      hasOptional = hasOptional || ruleHasOptional;
      if (skipOptional && !ruleHasOptional) continue;

      rcclCollectiveExecutionPolicy candidate;
      rcclExecutionPolicyValidationError error = rcclValidateCollectiveExecutionRule(rule, skipOptional, &candidate);
      if (error == RCCL_EXECUTION_POLICY_VALIDATION_NONE)
        error = rcclValidateCollectiveExecutionPolicy(candidate, facts, factCount, validation);
      if (error != RCCL_EXECUTION_POLICY_VALIDATION_NONE) {
        rcclRecordRejectedExecutionPolicyRule(resolution, ruleId, error);
        continue;
      }

      // Keep evaluating matched rules so matching and validation remain
      // independent stages. Selection itself is deterministic: the first valid
      // candidate has the lowest index and therefore the highest priority.
      if (!selected) {
        selectedPolicy = candidate;
        resolution->selectedRuleId = static_cast<rcclExecutionPolicyRuleId>(ruleId);
        selected = true;
      }
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

// Fields provisioning cannot know: they vary per operation, or are the
// channel count being provisioned.
constexpr uint32_t kPerOperationPolicyFields = rcclExecutionPolicyFieldBit(RCCL_EXECUTION_INPUT_SCOPE) |
                                               rcclExecutionPolicyFieldBit(RCCL_EXECUTION_INPUT_COLL_TYPE) |
                                               rcclExecutionPolicyFieldBit(RCCL_EXECUTION_INPUT_DATA_SIZE) |
                                               rcclExecutionPolicyFieldBit(RCCL_EXECUTION_INPUT_N_CHANNELS) |
                                               rcclExecutionPolicyFieldBit(RCCL_EXECUTION_INPUT_IN_PLACE);

} // namespace

void rcclInitExecutionPolicyResolution(struct rcclExecutionPolicyResolution* resolution,
                                       enum rcclExecutionPolicyResolutionStatus status) {
  if (resolution == nullptr) return;
  resolution->status = status;
  resolution->validationError = RCCL_EXECUTION_POLICY_VALIDATION_NONE;
  resolution->selectedRuleId = RCCL_EXECUTION_POLICY_NO_RULE;
  resolution->firstRejectedRuleId = RCCL_EXECUTION_POLICY_NO_RULE;
  resolution->firstRejectedError = RCCL_EXECUTION_POLICY_VALIDATION_NONE;
  rcclDefaultCollectiveExecutionPolicy(&resolution->policy);
}

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
    *policy = {-1, -1, -1, RCCL_P2P_PATH_AUTO, RCCL_P2P_TRANSFER_AUTO, RCCL_EXECUTION_TRANSPORT_UNKNOWN};
}

void rcclDefaultExecutionPolicyValidationContext(struct rcclExecutionPolicyValidationContext* context) {
  if (context == nullptr) return;
  memset(context, 0, sizeof(*context));
  context->baselineAlgorithm = -1;
  context->baselineProtocol = -1;
}

// Resolves an arbitrary caller-provided rule array with an optional immutable
// validation snapshot.
struct rcclExecutionPolicyResolution rcclEvaluateCollectiveExecutionRulesWithValidation(
  const struct rcclExecutionPolicyFact* facts, size_t factCount, const struct rcclCollectiveExecutionRule* rules,
  size_t ruleCount, const struct rcclExecutionPolicyValidationContext* validation) {
  rcclExecutionPolicyResolution resolution;
  rcclInitExecutionPolicyResolution(&resolution, RCCL_EXECUTION_POLICY_NO_MATCH);
  if ((factCount != 0 && facts == nullptr) || (ruleCount != 0 && rules == nullptr) ||
      !rcclExecutionPolicyValidationContextIsValid(validation)) {
    resolution.status = RCCL_EXECUTION_POLICY_INVALID_INPUT;
    return resolution;
  }
  rcclSelectCollectiveExecutionPolicy(facts, factCount, ruleCount, rcclArrayExecutionPolicyRuleAt, rules, validation,
                                      &resolution);
  return resolution;
}

namespace {

// Every P2P task of one collective resolves the same input, so a small
// per-thread memo of the last resolutions skips the rule scan. Resolution is a
// pure function of the input, the validation snapshot and the built-in table;
// validation is compared bytewise, so differing padding can only cause a miss.
struct rcclPolicyResolutionMemoEntry {
  bool valid;
  rcclCollectivePolicyInput input;
  char gfxArch[32];
  bool hasValidation;
  rcclExecutionPolicyValidationContext validation;
  rcclExecutionPolicyResolution resolution;
};
constexpr int kPolicyResolutionMemoEntries = 4;
thread_local rcclPolicyResolutionMemoEntry rcclPolicyResolutionMemo[kPolicyResolutionMemoEntries];
thread_local int rcclPolicyResolutionMemoNext;

bool rcclPolicyMemoInputMatches(const rcclPolicyResolutionMemoEntry& entry, const rcclCollectivePolicyInput* input,
                                const rcclExecutionPolicyValidationContext* validation) {
  const rcclCollectivePolicyInput& key = entry.input;
  if (!entry.valid || key.scope != input->scope || key.collType != input->collType || key.dataSize != input->dataSize ||
      key.nNodes != input->nNodes || key.nRanks != input->nRanks || key.nChannels != input->nChannels ||
      key.availableTransportMask != input->availableTransportMask || key.inPlace != input->inPlace ||
      key.transportIntent != input->transportIntent || (input->gfxArch == nullptr) != (key.gfxArch == nullptr) ||
      (input->gfxArch != nullptr && strcmp(entry.gfxArch, input->gfxArch) != 0) ||
      entry.hasValidation != (validation != nullptr))
    return false;
  return validation == nullptr || memcmp(&entry.validation, validation, sizeof(*validation)) == 0;
}

void rcclPolicyMemoStore(const rcclCollectivePolicyInput* input, const rcclExecutionPolicyValidationContext* validation,
                         const rcclExecutionPolicyResolution& resolution) {
  if (input->gfxArch != nullptr && strlen(input->gfxArch) >= sizeof(rcclPolicyResolutionMemo[0].gfxArch)) return;
  rcclPolicyResolutionMemoEntry& entry = rcclPolicyResolutionMemo[rcclPolicyResolutionMemoNext];
  rcclPolicyResolutionMemoNext = (rcclPolicyResolutionMemoNext + 1) % kPolicyResolutionMemoEntries;
  entry.input = *input;
  // Non-null marks a present arch; the memo compares its own copy.
  entry.input.gfxArch = input->gfxArch != nullptr ? entry.gfxArch : nullptr;
  if (input->gfxArch != nullptr) strcpy(entry.gfxArch, input->gfxArch);
  entry.hasValidation = validation != nullptr;
  if (validation != nullptr) memcpy(&entry.validation, validation, sizeof(*validation));
  entry.resolution = resolution;
  entry.valid = true;
}

} // namespace

// Normalizes a policy input and resolves the built-in architecture rules with
// an optional immutable validation snapshot.
struct rcclExecutionPolicyResolution rcclResolveCollectiveExecutionPolicyWithValidation(
  const struct rcclCollectivePolicyInput* input, const struct rcclExecutionPolicyValidationContext* validation) {
  rcclExecutionPolicyResolution resolution;
  rcclInitExecutionPolicyResolution(&resolution, RCCL_EXECUTION_POLICY_NO_MATCH);
  if (input == nullptr || !rcclExecutionTransportMaskIsValid(input->availableTransportMask) ||
      input->transportIntent < RCCL_EXECUTION_TRANSPORT_INTENT_AUTO ||
      input->transportIntent >= RCCL_EXECUTION_TRANSPORT_INTENT_COUNT ||
      !rcclExecutionPolicyValidationContextIsValid(validation)) {
    resolution.status = RCCL_EXECUTION_POLICY_INVALID_INPUT;
    return resolution;
  }
  uint32_t eligibleTransportMask =
    rcclEligibleExecutionTransportMask(input->availableTransportMask, input->transportIntent);
  if (eligibleTransportMask == 0) {
    resolution.status = RCCL_EXECUTION_POLICY_INVALID_INPUT;
    return resolution;
  }
  for (const rcclPolicyResolutionMemoEntry& entry : rcclPolicyResolutionMemo) {
    if (rcclPolicyMemoInputMatches(entry, input, validation)) return entry.resolution;
  }

  const rcclExecutionPolicyFact inputFacts[] = {
    {RCCL_EXECUTION_INPUT_SCOPE, rcclExecutionPolicyValue::Signed(input->scope)},
    {RCCL_EXECUTION_INPUT_GFX_ARCH, rcclExecutionPolicyValue::String(input->gfxArch)},
    {RCCL_EXECUTION_INPUT_COLL_TYPE, rcclExecutionPolicyValue::Signed(static_cast<int64_t>(input->collType))},
    {RCCL_EXECUTION_INPUT_DATA_SIZE, rcclExecutionPolicyValue::Unsigned(input->dataSize)},
    {RCCL_EXECUTION_INPUT_N_NODES, rcclExecutionPolicyValue::Signed(input->nNodes)},
    {RCCL_EXECUTION_INPUT_N_RANKS, rcclExecutionPolicyValue::Signed(input->nRanks)},
    {RCCL_EXECUTION_INPUT_N_CHANNELS, rcclExecutionPolicyValue::Signed(input->nChannels)},
    {RCCL_EXECUTION_INPUT_AVAILABLE_TRANSPORTS, rcclExecutionPolicyValue::Unsigned(input->availableTransportMask)},
    {RCCL_EXECUTION_INPUT_ELIGIBLE_TRANSPORTS, rcclExecutionPolicyValue::Unsigned(eligibleTransportMask)},
    {RCCL_EXECUTION_INPUT_IN_PLACE, rcclExecutionPolicyValue::Boolean(input->inPlace)},
    {RCCL_EXECUTION_INPUT_TRANSPORT_INTENT, rcclExecutionPolicyValue::Signed(input->transportIntent)},
  };
  rcclSelectCollectiveExecutionPolicy(inputFacts, sizeof(inputFacts) / sizeof(inputFacts[0]),
                                      rcclExecutionPolicyRules::kCollectiveExecutionRuleTable.count,
                                      rcclBuiltInExecutionPolicyRuleAt, nullptr, validation, &resolution);
  rcclPolicyMemoStore(input, validation, resolution);
  return resolution;
}

// Computes the maximum channel pool needed by matching built-in rules from a
// normalized, runtime-independent provisioning snapshot.
int rcclResolveExecutionPolicyRequiredChannels(const rcclExecutionPolicyProvisioningInput* input) {
  if (input == nullptr || !rcclExecutionTransportMaskIsValid(input->availableTransportMask) ||
      input->transportIntent < RCCL_EXECUTION_TRANSPORT_INTENT_AUTO ||
      input->transportIntent >= RCCL_EXECUTION_TRANSPORT_INTENT_COUNT)
    return 0;
  uint32_t eligibleTransportMask =
    rcclEligibleExecutionTransportMask(input->availableTransportMask, input->transportIntent);
  if (eligibleTransportMask == 0) return 0;
  const rcclExecutionPolicyFact provisioningFacts[] = {
    {RCCL_EXECUTION_INPUT_SCOPE, rcclExecutionPolicyValue::Signed(RCCL_EXECUTION_SCOPE_COLLECTIVE)},
    {RCCL_EXECUTION_INPUT_GFX_ARCH, rcclExecutionPolicyValue::String(input->gfxArch)},
    {RCCL_EXECUTION_INPUT_N_NODES, rcclExecutionPolicyValue::Signed(input->nNodes)},
    {RCCL_EXECUTION_INPUT_N_RANKS, rcclExecutionPolicyValue::Signed(input->nRanks)},
    {RCCL_EXECUTION_INPUT_AVAILABLE_TRANSPORTS, rcclExecutionPolicyValue::Unsigned(input->availableTransportMask)},
    {RCCL_EXECUTION_INPUT_ELIGIBLE_TRANSPORTS, rcclExecutionPolicyValue::Unsigned(eligibleTransportMask)},
    {RCCL_EXECUTION_INPUT_TRANSPORT_INTENT, rcclExecutionPolicyValue::Signed(input->transportIntent)},
  };

  int requiredChannels = 0;
  const rcclExecutionPolicyRules::RuleTableView& table = rcclExecutionPolicyRules::kCollectiveExecutionRuleTable;
  for (size_t ruleId = 0; ruleId < table.count; ruleId++) {
    const rcclExecutionPolicyRules::RuleRow& row = table.rows[ruleId];
    const rcclCollectiveExecutionRule ruleView = {
      row.conditions,
      row.conditionCount,
      row.assignments,
      row.assignmentCount,
    };
    if (!rcclCollectiveExecutionRuleMatches(ruleView, provisioningFacts,
                                            sizeof(provisioningFacts) / sizeof(provisioningFacts[0]),
                                            kPerOperationPolicyFields))
      continue;
    bool p2pScope = false;
    bool remapsActivePool = false;
    for (size_t conditionIndex = 0; conditionIndex < ruleView.conditionCount; conditionIndex++) {
      const rcclExecutionPolicyCondition& condition = ruleView.conditions[conditionIndex];
      p2pScope = p2pScope || (condition.field == RCCL_EXECUTION_INPUT_SCOPE &&
                              condition.value.type == RCCL_EXECUTION_VALUE_SIGNED &&
                              condition.value.signedValue == RCCL_EXECUTION_SCOPE_P2P);
    }
    for (size_t assignmentIndex = 0; assignmentIndex < ruleView.assignmentCount; assignmentIndex++) {
      const rcclExecutionPolicyAssignment& assignment = ruleView.assignments[assignmentIndex];
      remapsActivePool = remapsActivePool || (assignment.field == RCCL_EXECUTION_OUTPUT_PATH &&
                                              assignment.value.type == RCCL_EXECUTION_VALUE_SIGNED &&
                                              assignment.value.signedValue == RCCL_P2P_PATH_SENDRECV);
    }
    // P2P rules that keep the normal Send/Recv path cap channels within the
    // communicator's existing pool and require no widening.
    if (p2pScope && !remapsActivePool) continue;
    for (size_t assignmentIndex = 0; assignmentIndex < ruleView.assignmentCount; assignmentIndex++) {
      const rcclExecutionPolicyAssignment& assignment = ruleView.assignments[assignmentIndex];
      if (assignment.field != RCCL_EXECUTION_OUTPUT_N_CHANNELS || assignment.value.type != RCCL_EXECUTION_VALUE_SIGNED)
        continue;
      int64_t value = assignment.value.signedValue;
      if (value > requiredChannels && value <= MAXCHANNELS) requiredChannels = static_cast<int>(value);
    }
  }
  return requiredChannels;
}
