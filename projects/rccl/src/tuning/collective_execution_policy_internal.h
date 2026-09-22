/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_COLLECTIVE_EXECUTION_POLICY_INTERNAL_H_
#define RCCL_COLLECTIVE_EXECUTION_POLICY_INTERNAL_H_

#include "collective_execution_policy.h"
#include "device.h"

#include <array>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <type_traits>

namespace rcclExecutionPolicyRules {

struct ByteRange {
  size_t min;
  size_t max;
};

struct IntRange {
  int min;
  int max;
};

enum class BoolConstraint {
  Any,
  False,
  True,
};

struct TransportConstraint {
  bool any;
  rcclExecutionTransport value;
};

enum class ChannelCount : int {};
enum class Algorithm : int {};
enum class Protocol : int {};

template<typename T>
struct Override {
  bool isSet;
  T value;

  static constexpr Override Keep() { return {false, T{}}; }
  static constexpr Override Set(T value) { return {true, value}; }
};

using ChannelOverride = Override<ChannelCount>;
using AlgorithmOverride = Override<Algorithm>;
using ProtocolOverride = Override<Protocol>;
using PathOverride = Override<rcclP2pExecutionPath>;
using TransferOverride = Override<rcclP2pTransferMode>;
using TransportOverride = Override<rcclExecutionTransport>;

// KEEP is convertible only to an output override. AUTO is intentionally
// limited to fields whose automatic value is -1.
struct KeepTag {
  template<typename T>
  constexpr operator Override<T>() const {
    return Override<T>::Keep();
  }
};

struct AutoTag {
  constexpr operator ChannelOverride() const {
    return ChannelOverride::Set(static_cast<ChannelCount>(-1));
  }
  constexpr operator AlgorithmOverride() const {
    return AlgorithmOverride::Set(static_cast<Algorithm>(-1));
  }
  constexpr operator ProtocolOverride() const {
    return ProtocolOverride::Set(static_cast<Protocol>(-1));
  }
};

constexpr KeepTag KEEP{};
constexpr AutoTag AUTO{};

static_assert(!std::is_convertible<int, ChannelOverride>::value,
              "Channel overrides must use CHANNELS or AUTO");
static_assert(!std::is_convertible<AlgorithmOverride, ProtocolOverride>::value,
              "Algorithm and protocol overrides must remain distinct");
static_assert(!std::is_convertible<TransferOverride, ProtocolOverride>::value,
              "Transfer mode and protocol overrides must remain distinct");
static_assert(!std::is_convertible<AutoTag, PathOverride>::value,
              "Path AUTO must use the typed AUTO_PATH value");
static_assert(!std::is_convertible<bool, BoolConstraint>::value,
              "Boolean matches must use an explicit BoolConstraint");

struct RuleMatch {
  rcclExecutionScope scope;
  const char* gfxArch;
  ncclFunc_t collType;
  ByteRange dataSize;
  IntRange nNodes;
  IntRange nRanks;
  IntRange nChannels;
  TransportConstraint transport;
  BoolConstraint inPlace;
};

struct RuleProfile {
  ChannelOverride nChannels;
  AlgorithmOverride algorithm;
  ProtocolOverride protocol;
  PathOverride path;
  TransferOverride transferMode;
  TransportOverride transport;
};

struct RuleSpec {
  RuleMatch match;
  RuleProfile profile;
};

struct RuleRow {
  rcclExecutionPolicyCondition conditions[13];
  rcclExecutionPolicyAssignment assignments[6];
  size_t conditionCount;
  size_t assignmentCount;
};

struct RuleTableView {
  const RuleSpec* specs;
  const RuleRow* rows;
  size_t count;
};

constexpr ByteRange kAnyBytes = {0, SIZE_MAX};
constexpr IntRange kAnyInt = {INT_MIN, INT_MAX};
constexpr IntRange EQ(int value) { return {value, value}; }
constexpr IntRange GE(int value) { return {value, INT_MAX}; }

// Compact rule-table vocabulary. Public API names remain explicit.
constexpr rcclExecutionScope P2P = RCCL_EXECUTION_SCOPE_P2P;
constexpr rcclExecutionScope COLL = RCCL_EXECUTION_SCOPE_COLLECTIVE;
constexpr ncclFunc_t ALL_TO_ALL = ncclFuncAlltoAll;
constexpr ncclFunc_t ALL_TO_ALL_V = ncclFuncAlltoAllv;
constexpr ncclFunc_t ALL_GATHER = ncclFuncAllGather;
constexpr ncclFunc_t ALL_REDUCE = ncclFuncAllReduce;
constexpr ncclFunc_t BROADCAST = ncclFuncBroadcast;
constexpr ncclFunc_t REDUCE = ncclFuncReduce;
constexpr ncclFunc_t REDUCE_SCATTER = ncclFuncReduceScatter;
constexpr ncclFunc_t GATHER = ncclFuncGather;
constexpr ncclFunc_t SCATTER = ncclFuncScatter;
constexpr TransportConstraint ANY_TRANSPORT = {
  true, RCCL_EXECUTION_TRANSPORT_UNKNOWN};
constexpr TransportConstraint IPC = {
  false, RCCL_EXECUTION_TRANSPORT_IPC};
constexpr TransportConstraint SHM = {
  false, RCCL_EXECUTION_TRANSPORT_SHM};
constexpr TransportConstraint NET = {
  false, RCCL_EXECUTION_TRANSPORT_NET};
constexpr TransportConstraint COLLNET = {
  false, RCCL_EXECUTION_TRANSPORT_COLLNET};
constexpr TransportConstraint MIXED = {
  false, RCCL_EXECUTION_TRANSPORT_MIXED};
constexpr BoolConstraint ANY_BOOL = BoolConstraint::Any;
constexpr BoolConstraint BOOL_FALSE = BoolConstraint::False;
constexpr BoolConstraint BOOL_TRUE = BoolConstraint::True;
constexpr ChannelOverride CHANNELS(int value) {
  return ChannelOverride::Set(static_cast<ChannelCount>(value));
}
constexpr AlgorithmOverride ALGORITHM(int value) {
  return AlgorithmOverride::Set(static_cast<Algorithm>(value));
}
constexpr ProtocolOverride PROTOCOL(int value) {
  return ProtocolOverride::Set(static_cast<Protocol>(value));
}
constexpr AlgorithmOverride RING = ALGORITHM(NCCL_ALGO_RING);
constexpr ProtocolOverride LL = PROTOCOL(NCCL_PROTO_LL);
constexpr ProtocolOverride SIMPLE = PROTOCOL(NCCL_PROTO_SIMPLE);
constexpr PathOverride AUTO_PATH = PathOverride::Set(RCCL_P2P_PATH_AUTO);
constexpr PathOverride SENDRECV = PathOverride::Set(RCCL_P2P_PATH_SENDRECV);
constexpr TransferOverride AUTO_XFER = TransferOverride::Set(RCCL_P2P_TRANSFER_AUTO);
constexpr TransferOverride WRITE = TransferOverride::Set(RCCL_P2P_TRANSFER_WRITE);
constexpr TransferOverride READ = TransferOverride::Set(RCCL_P2P_TRANSFER_READ);
constexpr TransportOverride USE_IPC =
  TransportOverride::Set(RCCL_EXECUTION_TRANSPORT_IPC);
constexpr TransportOverride USE_SHM =
  TransportOverride::Set(RCCL_EXECUTION_TRANSPORT_SHM);

constexpr bool isAny(const IntRange& range) {
  return range.min == INT_MIN && range.max == INT_MAX;
}

constexpr bool isValidCountRange(const IntRange& range, int minimum) {
  if (isAny(range)) return true;
  if (range.min > range.max) return false;
  if (range.min != INT_MIN && range.min < minimum) return false;
  if (range.max != INT_MAX && range.max < minimum) return false;
  return true;
}

constexpr bool isValidBoolConstraint(BoolConstraint value) {
  return value == BoolConstraint::Any || value == BoolConstraint::False ||
         value == BoolConstraint::True;
}

constexpr bool isValidTransportConstraint(const TransportConstraint& constraint) {
  return constraint.any ||
         (constraint.value > RCCL_EXECUTION_TRANSPORT_UNKNOWN &&
          constraint.value < RCCL_EXECUTION_TRANSPORT_COUNT);
}

template<typename T>
constexpr int overrideValue(const Override<T>& value) {
  return static_cast<int>(value.value);
}

template<typename T>
constexpr bool isValidOverride(const Override<T>& value, int minimum, int maximum) {
  return !value.isSet || (overrideValue(value) >= minimum && overrideValue(value) <= maximum);
}

template<typename T>
constexpr bool isSetTo(const Override<T>& value, T expected) {
  return value.isSet && value.value == expected;
}

constexpr bool hasAssignment(const RuleProfile& profile) {
  return profile.nChannels.isSet || profile.algorithm.isSet || profile.protocol.isSet ||
         profile.path.isSet || profile.transferMode.isSet ||
         profile.transport.isSet;
}

// Checks invariants that depend only on build-time rule data. Runtime
// availability checks remain part of match -> validate -> select.
constexpr bool isRuleSpecValid(const RuleSpec& rule) {
  const RuleMatch& match = rule.match;
  const RuleProfile& profile = rule.profile;

  if (match.scope != P2P && match.scope != COLL) return false;
  if (match.gfxArch == nullptr || match.gfxArch[0] == '\0') return false;
  if (static_cast<int>(match.collType) < 0 || static_cast<int>(match.collType) >= ncclNumFuncs) return false;
  if (match.dataSize.min > match.dataSize.max) return false;
  if (!isValidCountRange(match.nNodes, /*minimum=*/1) ||
      !isValidCountRange(match.nRanks, /*minimum=*/1) ||
      !isValidCountRange(match.nChannels, /*minimum=*/0))
    return false;
  if (!isValidTransportConstraint(match.transport) ||
      !isValidBoolConstraint(match.inPlace))
    return false;

  if (profile.nChannels.isSet && overrideValue(profile.nChannels) != -1 &&
      (overrideValue(profile.nChannels) < 1 || overrideValue(profile.nChannels) > MAXCHANNELS))
    return false;
  if (!isValidOverride(profile.algorithm, -1, NCCL_NUM_ALGORITHMS - 1) ||
      !isValidOverride(profile.protocol, -1, NCCL_NUM_PROTOCOLS - 1) ||
      !isValidOverride(profile.path, RCCL_P2P_PATH_AUTO, RCCL_P2P_PATH_SENDRECV) ||
      !isValidOverride(profile.transferMode, RCCL_P2P_TRANSFER_AUTO, RCCL_P2P_TRANSFER_READ) ||
      !isValidOverride(profile.transport, RCCL_EXECUTION_TRANSPORT_IPC,
                       RCCL_EXECUTION_TRANSPORT_COUNT - 1))
    return false;
  if (!hasAssignment(profile)) return false;

  // P2P-only outputs cannot be attached to a regular collective profile.
  if (match.scope == COLL &&
      ((profile.path.isSet &&
        !isSetTo(profile.path, RCCL_P2P_PATH_AUTO)) ||
       (profile.transferMode.isSet &&
        !isSetTo(profile.transferMode, RCCL_P2P_TRANSFER_AUTO))))
    return false;
  if (isSetTo(profile.path, RCCL_P2P_PATH_SENDRECV) && match.scope != P2P) return false;
  if ((isSetTo(profile.transferMode, RCCL_P2P_TRANSFER_READ) ||
       isSetTo(profile.transferMode, RCCL_P2P_TRANSFER_WRITE)) &&
      !isSetTo(profile.path, RCCL_P2P_PATH_SENDRECV))
    return false;
  return true;
}

template<size_t RuleCount>
constexpr size_t firstInvalidRule(const RuleSpec (&rules)[RuleCount]) {
  for (size_t ruleId = 0; ruleId < RuleCount; ruleId++) {
    if (!isRuleSpecValid(rules[ruleId])) return ruleId;
  }
  return RuleCount;
}

inline void appendCondition(RuleRow* row, rcclExecutionPolicyInputField field,
                            rcclExecutionPolicyCompareOp op, rcclExecutionPolicyValue value) {
  row->conditions[row->conditionCount++] = {field, op, value};
}

inline void appendIntRange(RuleRow* row, rcclExecutionPolicyInputField field, const IntRange& range) {
  if (range.min == INT_MIN && range.max == INT_MAX) return;
  if (range.min == range.max) {
    appendCondition(row, field, RCCL_EXECUTION_COMPARE_EQ, rcclExecutionPolicyValue::Signed(range.min));
    return;
  }
  if (range.min != INT_MIN)
    appendCondition(row, field, RCCL_EXECUTION_COMPARE_GE, rcclExecutionPolicyValue::Signed(range.min));
  if (range.max != INT_MAX)
    appendCondition(row, field, RCCL_EXECUTION_COMPARE_LE, rcclExecutionPolicyValue::Signed(range.max));
}

inline void appendByteRange(RuleRow* row, const ByteRange& range) {
  if (range.min == 0 && range.max == SIZE_MAX) return;
  if (range.min == range.max) {
    appendCondition(row, RCCL_EXECUTION_INPUT_DATA_SIZE, RCCL_EXECUTION_COMPARE_EQ,
                    rcclExecutionPolicyValue::Unsigned(range.min));
    return;
  }
  if (range.min != 0)
    appendCondition(row, RCCL_EXECUTION_INPUT_DATA_SIZE, RCCL_EXECUTION_COMPARE_GE,
                    rcclExecutionPolicyValue::Unsigned(range.min));
  if (range.max != SIZE_MAX)
    appendCondition(row, RCCL_EXECUTION_INPUT_DATA_SIZE, RCCL_EXECUTION_COMPARE_LE,
                    rcclExecutionPolicyValue::Unsigned(range.max));
}

template<typename T>
inline void appendAssignment(RuleRow* row, rcclExecutionPolicyOutputField field,
                             const Override<T>& value) {
  if (!value.isSet) return;
  row->assignments[row->assignmentCount++] = {
    field, rcclExecutionPolicyValue::Signed(static_cast<int>(value.value))};
}

inline RuleRow expandRule(const RuleSpec& spec) {
  RuleRow row{};
  appendCondition(&row, RCCL_EXECUTION_INPUT_SCOPE, RCCL_EXECUTION_COMPARE_EQ,
                  rcclExecutionPolicyValue::Signed(spec.match.scope));
  appendCondition(&row, RCCL_EXECUTION_INPUT_GFX_ARCH, RCCL_EXECUTION_COMPARE_ARCH_MATCH,
                  rcclExecutionPolicyValue::String(spec.match.gfxArch));
  appendCondition(&row, RCCL_EXECUTION_INPUT_COLL_TYPE, RCCL_EXECUTION_COMPARE_EQ,
                  rcclExecutionPolicyValue::Signed(static_cast<int64_t>(spec.match.collType)));
  appendByteRange(&row, spec.match.dataSize);
  appendIntRange(&row, RCCL_EXECUTION_INPUT_N_NODES, spec.match.nNodes);
  appendIntRange(&row, RCCL_EXECUTION_INPUT_N_RANKS, spec.match.nRanks);
  appendIntRange(&row, RCCL_EXECUTION_INPUT_N_CHANNELS, spec.match.nChannels);
  if (!spec.match.transport.any)
    appendCondition(&row, RCCL_EXECUTION_INPUT_TRANSPORT,
                    RCCL_EXECUTION_COMPARE_EQ,
                    rcclExecutionPolicyValue::Signed(spec.match.transport.value));
  if (spec.match.inPlace != BoolConstraint::Any)
    appendCondition(&row, RCCL_EXECUTION_INPUT_IN_PLACE, RCCL_EXECUTION_COMPARE_EQ,
                    rcclExecutionPolicyValue::Boolean(spec.match.inPlace == BoolConstraint::True));

  appendAssignment(&row, RCCL_EXECUTION_OUTPUT_N_CHANNELS, spec.profile.nChannels);
  appendAssignment(&row, RCCL_EXECUTION_OUTPUT_ALGORITHM, spec.profile.algorithm);
  appendAssignment(&row, RCCL_EXECUTION_OUTPUT_PROTOCOL, spec.profile.protocol);
  appendAssignment(&row, RCCL_EXECUTION_OUTPUT_PATH, spec.profile.path);
  appendAssignment(&row, RCCL_EXECUTION_OUTPUT_TRANSFER_MODE, spec.profile.transferMode);
  appendAssignment(&row, RCCL_EXECUTION_OUTPUT_TRANSPORT, spec.profile.transport);
  return row;
}

template<size_t RuleCount>
inline std::array<RuleRow, RuleCount> expandRules(const RuleSpec (&specs)[RuleCount]) {
  std::array<RuleRow, RuleCount> rows{};
  for (size_t i = 0; i < RuleCount; i++) rows[i] = expandRule(specs[i]);
  return rows;
}

// Defined in collective_execution_policy_rules.cc. Array index is rule ID and
// priority; lower indices have higher priority.
extern const RuleTableView kCollectiveExecutionRuleTable;

} // namespace rcclExecutionPolicyRules

#endif // RCCL_COLLECTIVE_EXECUTION_POLICY_INTERNAL_H_
