/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_COLLECTIVE_EXECUTION_POLICY_H_
#define RCCL_COLLECTIVE_EXECUTION_POLICY_H_

#include "core.h"

#include <stddef.h>
#include <stdint.h>

struct ncclComm;

// P2P and regular collectives share a matcher but have different execution
// semantics. Scope prevents a rule for one planning path from matching the
// other accidentally.
enum rcclExecutionScope {
  RCCL_EXECUTION_SCOPE_P2P = 0,
  RCCL_EXECUTION_SCOPE_COLLECTIVE = 1,
};

enum rcclP2pExecutionPath {
  RCCL_P2P_PATH_AUTO = 0,
  RCCL_P2P_PATH_SENDRECV = 1,
};

enum rcclP2pTransferMode {
  RCCL_P2P_TRANSFER_AUTO = 0,
  RCCL_P2P_TRANSFER_WRITE = 1,
  RCCL_P2P_TRANSFER_READ = 2,
};

// Concrete facts supplied by RCCL adapters. Add future dimensions here and in
// the adapter without changing the generic evaluator or rule representation.
struct rcclCollectivePolicyInput {
  enum rcclExecutionScope scope;
  const char* gfxArch;
  ncclFunc_t collType;
  size_t dataSize; // Aggregate payload bytes per rank.
  int nNodes;
  int nRanks;
  int nChannels;
  bool p2pEnabled;
  bool inPlace;            // Active send/receive buffer ranges overlap
};

// Sparse execution overrides. AUTO/-1 fields preserve RCCL's normal selection.
// nChannels is interpreted by the scope-specific consumer: an operation-local
// P2P pool for P2P scope and collective channels for collective scope.
struct rcclCollectiveExecutionPolicy {
  int nChannels;
  int algorithm; // NCCL_ALGO_*; -1 means AUTO.
  int protocol;  // NCCL_PROTO_*; -1 means AUTO.
  enum rcclP2pExecutionPath path;
  enum rcclP2pTransferMode transferMode;
};

enum rcclExecutionPolicyValueType {
  RCCL_EXECUTION_VALUE_SIGNED,
  RCCL_EXECUTION_VALUE_UNSIGNED,
  RCCL_EXECUTION_VALUE_STRING,
  RCCL_EXECUTION_VALUE_BOOL,
};

struct rcclExecutionPolicyValue {
  enum rcclExecutionPolicyValueType type;
  union {
    int64_t signedValue;
    uint64_t unsignedValue;
    const char* stringValue;
    bool boolValue;
  };

  constexpr rcclExecutionPolicyValue() : type(RCCL_EXECUTION_VALUE_SIGNED), signedValue(0) {}
  static constexpr rcclExecutionPolicyValue Signed(int64_t value) {
    return rcclExecutionPolicyValue(RCCL_EXECUTION_VALUE_SIGNED, value);
  }
  static constexpr rcclExecutionPolicyValue Unsigned(uint64_t value) {
    return rcclExecutionPolicyValue(RCCL_EXECUTION_VALUE_UNSIGNED, value);
  }
  static constexpr rcclExecutionPolicyValue String(const char* value) {
    return rcclExecutionPolicyValue(RCCL_EXECUTION_VALUE_STRING, value);
  }
  static constexpr rcclExecutionPolicyValue Boolean(bool value) {
    return rcclExecutionPolicyValue(RCCL_EXECUTION_VALUE_BOOL, value);
  }

private:
  constexpr rcclExecutionPolicyValue(enum rcclExecutionPolicyValueType valueType, int64_t value)
    : type(valueType), signedValue(value) {}
  constexpr rcclExecutionPolicyValue(enum rcclExecutionPolicyValueType valueType, uint64_t value)
    : type(valueType), unsignedValue(value) {}
  constexpr rcclExecutionPolicyValue(enum rcclExecutionPolicyValueType valueType, const char* value)
    : type(valueType), stringValue(value) {}
  constexpr rcclExecutionPolicyValue(enum rcclExecutionPolicyValueType valueType, bool value)
    : type(valueType), boolValue(value) {}
};

enum rcclExecutionPolicyInputField {
  RCCL_EXECUTION_INPUT_SCOPE,
  RCCL_EXECUTION_INPUT_GFX_ARCH,
  RCCL_EXECUTION_INPUT_COLL_TYPE,
  RCCL_EXECUTION_INPUT_DATA_SIZE,
  RCCL_EXECUTION_INPUT_N_NODES,
  RCCL_EXECUTION_INPUT_N_RANKS,
  RCCL_EXECUTION_INPUT_N_CHANNELS,
  RCCL_EXECUTION_INPUT_P2P_ENABLED,
  RCCL_EXECUTION_INPUT_IN_PLACE,
};

enum rcclExecutionPolicyOutputField {
  RCCL_EXECUTION_OUTPUT_N_CHANNELS,
  RCCL_EXECUTION_OUTPUT_ALGORITHM,
  RCCL_EXECUTION_OUTPUT_PROTOCOL,
  RCCL_EXECUTION_OUTPUT_PATH,
  RCCL_EXECUTION_OUTPUT_TRANSFER_MODE,
};

enum rcclExecutionPolicyCompareOp {
  RCCL_EXECUTION_COMPARE_EQ,
  RCCL_EXECUTION_COMPARE_NE,
  RCCL_EXECUTION_COMPARE_LT,
  RCCL_EXECUTION_COMPARE_LE,
  RCCL_EXECUTION_COMPARE_GT,
  RCCL_EXECUTION_COMPARE_GE,
  RCCL_EXECUTION_COMPARE_ARCH_MATCH,
};

struct rcclExecutionPolicyFact {
  enum rcclExecutionPolicyInputField field;
  struct rcclExecutionPolicyValue value;
};

// A closed input range is represented by two conditions on the same field,
// for example DATA_SIZE >= minDataSize and DATA_SIZE <= maxDataSize.
struct rcclExecutionPolicyCondition {
  enum rcclExecutionPolicyInputField field;
  enum rcclExecutionPolicyCompareOp op;
  struct rcclExecutionPolicyValue value;
};

struct rcclExecutionPolicyAssignment {
  enum rcclExecutionPolicyOutputField field;
  struct rcclExecutionPolicyValue value;
};

// A rule is an atomic execution profile. Assignments are applied to a fresh
// default policy and are never merged with assignments from another rule.
struct rcclCollectiveExecutionRule {
  const struct rcclExecutionPolicyCondition* conditions;
  size_t conditionCount;
  const struct rcclExecutionPolicyAssignment* assignments;
  size_t assignmentCount;
};

// The rule-array index is both the build-local identifier and the priority:
// lower indices have higher priority. An identical RCCL build therefore keeps
// the same index-to-rule mapping for the lifetime of the process.
using rcclExecutionPolicyRuleId = uint16_t;
constexpr rcclExecutionPolicyRuleId RCCL_EXECUTION_POLICY_NO_RULE = UINT16_MAX;

enum rcclExecutionPolicyResolutionStatus {
  RCCL_EXECUTION_POLICY_SELECTED,
  RCCL_EXECUTION_POLICY_NO_MATCH,
  RCCL_EXECUTION_POLICY_VALIDATION_FAILED,
  RCCL_EXECUTION_POLICY_INVALID_INPUT,
};

enum rcclExecutionPolicyValidationError {
  RCCL_EXECUTION_POLICY_VALIDATION_NONE,
  RCCL_EXECUTION_POLICY_VALIDATION_INVALID_ASSIGNMENT,
  RCCL_EXECUTION_POLICY_VALIDATION_CHANNELS_UNAVAILABLE,
  RCCL_EXECUTION_POLICY_VALIDATION_ALGO_PROTO_UNAVAILABLE,
  RCCL_EXECUTION_POLICY_VALIDATION_INVALID_SCOPE_OUTPUT,
  RCCL_EXECUTION_POLICY_VALIDATION_INVALID_PATH_TRANSFER,
  RCCL_EXECUTION_POLICY_VALIDATION_P2P_PROTOCOL_UNAVAILABLE,
  RCCL_EXECUTION_POLICY_VALIDATION_P2P_TRANSFER_UNAVAILABLE,
};

// Normalized per-resolution constraints. Callers translate environment,
// communicator, tuning, and connector state into this immutable snapshot; the
// policy framework never reads those mutable sources directly.
struct rcclExecutionPolicyValidationContext {
  int minChannels; // Inclusive; <= 0 leaves the lower bound unconstrained.
  int maxChannels; // Inclusive; <= 0 leaves the upper bound unconstrained.

  // Baseline shape selected before policy overrides. When shape validation is
  // enabled, sparse algorithm/protocol assignments are completed from it.
  int baselineAlgorithm;
  int baselineProtocol;
  bool validateAlgoProto;
  bool algoProtoAvailable[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  // P2P capability masks use enum values as bit positions. Checks are opt-in
  // because connector facts are unavailable during early routing/preconnect.
  bool validateP2pProtocol;
  uint32_t p2pProtocolMask;
  bool validateP2pTransfer;
  uint32_t p2pTransferMask;
};

struct rcclExecutionPolicyResolution {
  enum rcclExecutionPolicyResolutionStatus status;
  enum rcclExecutionPolicyValidationError validationError;
  rcclExecutionPolicyRuleId selectedRuleId;
  rcclExecutionPolicyRuleId firstRejectedRuleId;
  enum rcclExecutionPolicyValidationError firstRejectedError;
  struct rcclCollectiveExecutionPolicy policy;
};

void rcclDefaultCollectiveExecutionPolicy(struct rcclCollectiveExecutionPolicy* policy);
void rcclDefaultExecutionPolicyValidationContext(struct rcclExecutionPolicyValidationContext* context);
struct rcclExecutionPolicyResolution rcclEvaluateCollectiveExecutionRulesWithValidation(
  const struct rcclExecutionPolicyFact* facts, size_t factCount,
  const struct rcclCollectiveExecutionRule* rules, size_t ruleCount,
  const struct rcclExecutionPolicyValidationContext* validation);
struct rcclExecutionPolicyResolution rcclResolveCollectiveExecutionPolicyWithValidation(
  const struct rcclCollectivePolicyInput* input,
  const struct rcclExecutionPolicyValidationContext* validation);
bool rcclEvaluateCollectiveExecutionRules(const struct rcclExecutionPolicyFact* facts, size_t factCount,
                                           const struct rcclCollectiveExecutionRule* rules, size_t ruleCount,
                                           struct rcclCollectiveExecutionPolicy* policy,
                                           rcclExecutionPolicyRuleId* selectedRuleId);
bool rcclResolveCollectiveExecutionPolicy(const struct rcclCollectivePolicyInput* input,
                                          struct rcclCollectiveExecutionPolicy* policy,
                                          rcclExecutionPolicyRuleId* selectedRuleId);
bool rcclBuffersOverlap(const void* firstBuffer, size_t firstBytes, const void* secondBuffer, size_t secondBytes);

// Maximum channel count requested by any collective-scope rule matching the
// communicator's stable facts. Used during communicator setup so per-operation
// rules can select channels that were actually provisioned.
int rcclGetCollectiveExecutionPolicyRequiredChannels(struct ncclComm* comm, bool p2pDisabled);

// Convenience adapter from communicator state to generic rule-engine facts.
bool rcclGetCollectiveExecutionPolicy(const struct ncclComm* comm, enum rcclExecutionScope scope,
                                      ncclFunc_t collType,
                                      size_t dataSize, bool p2pDisabled, bool inPlace,
                                      struct rcclCollectiveExecutionPolicy* policy);
inline bool rcclGetCollectiveExecutionPolicy(const struct ncclComm* comm, enum rcclExecutionScope scope,
                                             ncclFunc_t collType, size_t dataSize, bool p2pDisabled,
                                             struct rcclCollectiveExecutionPolicy* policy) {
  return rcclGetCollectiveExecutionPolicy(comm, scope, collType, dataSize, p2pDisabled, /*inPlace=*/false, policy);
}

inline bool rcclEvaluateCollectiveExecutionRules(const struct rcclExecutionPolicyFact* facts, size_t factCount,
                                                 const struct rcclCollectiveExecutionRule* rules, size_t ruleCount,
                                                 struct rcclCollectiveExecutionPolicy* policy) {
  return rcclEvaluateCollectiveExecutionRules(facts, factCount, rules, ruleCount, policy, nullptr);
}

inline bool rcclResolveCollectiveExecutionPolicy(const struct rcclCollectivePolicyInput* input,
                                                 struct rcclCollectiveExecutionPolicy* policy) {
  return rcclResolveCollectiveExecutionPolicy(input, policy, nullptr);
}

#endif // RCCL_COLLECTIVE_EXECUTION_POLICY_H_
