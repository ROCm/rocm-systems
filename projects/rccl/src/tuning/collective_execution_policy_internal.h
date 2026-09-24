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

// One concrete transport candidate, or ANY_TRANSPORT for a transport-neutral
// rule. Concrete candidates are filtered against the input's eligible mask and
// materialize that transport in the selected profile.
struct TransportCandidate {
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
  TransportCandidate candidateTransport;
  BoolConstraint inPlace;
};

struct RuleProfile {
  ChannelOverride nChannels;
  AlgorithmOverride algorithm;
  ProtocolOverride protocol;
  PathOverride path;
  TransferOverride transferMode;
};

// Offline-measured efficiency of a rule's profile over its byte range: peak
// measured bus bandwidth divided by the platform's theoretical peak. Every
// transport on a platform shares one denominator, so a higher score always
// means more throughput for the same input. Scores are checked at build time
// against table order and are not used at runtime.
struct RuleScore {
  bool isSet;
  float efficiency;
};

struct RuleSpec {
  RuleMatch match;
  RuleProfile profile;
  RuleScore score;
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
constexpr TransportCandidate ANY_TRANSPORT = {
  true, RCCL_EXECUTION_TRANSPORT_UNKNOWN};
constexpr TransportCandidate IPC = {
  false, RCCL_EXECUTION_TRANSPORT_IPC};
constexpr TransportCandidate SHM = {
  false, RCCL_EXECUTION_TRANSPORT_SHM};
constexpr TransportCandidate NET = {
  false, RCCL_EXECUTION_TRANSPORT_NET};
constexpr TransportCandidate COLLNET = {
  false, RCCL_EXECUTION_TRANSPORT_COLLNET};
constexpr TransportCandidate MIXED = {
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
// A rule the offline sweep could not measure (for example a fallback shadowed
// by a higher-priority rule of the same transport). It wins only when no
// measured candidate is valid.
constexpr RuleScore UNMEASURED = {true, 0.0f};
constexpr RuleScore EFFICIENCY(double measured, double peak) {
  return {true, static_cast<float>(measured / peak)};
}

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

constexpr bool isValidTransportCandidate(const TransportCandidate& candidate) {
  return candidate.any ||
         (candidate.value > RCCL_EXECUTION_TRANSPORT_UNKNOWN &&
          candidate.value < RCCL_EXECUTION_TRANSPORT_COUNT);
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
         profile.path.isSet || profile.transferMode.isSet;
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
  if (!isValidTransportCandidate(match.candidateTransport) ||
      !isValidBoolConstraint(match.inPlace))
    return false;

  if (profile.nChannels.isSet && overrideValue(profile.nChannels) != -1 &&
      (overrideValue(profile.nChannels) < 1 || overrideValue(profile.nChannels) > MAXCHANNELS))
    return false;
  if (!isValidOverride(profile.algorithm, -1, NCCL_NUM_ALGORITHMS - 1) ||
      !isValidOverride(profile.protocol, -1, NCCL_NUM_PROTOCOLS - 1) ||
      !isValidOverride(profile.path, RCCL_P2P_PATH_AUTO, RCCL_P2P_PATH_SENDRECV) ||
      !isValidOverride(profile.transferMode, RCCL_P2P_TRANSFER_AUTO, RCCL_P2P_TRANSFER_READ))
    return false;
  if (!hasAssignment(profile) && match.candidateTransport.any) return false;
  if (!rule.score.isSet || !(rule.score.efficiency >= 0.0f) || rule.score.efficiency > 1.0f) return false;

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

template<typename T>
constexpr bool sameOverride(const Override<T>& first, const Override<T>& second) {
  return first.isSet == second.isSet &&
         (!first.isSet || first.value == second.value);
}

constexpr bool sameP2pWorkShape(const RuleProfile& first, const RuleProfile& second) {
  return sameOverride(first.nChannels, second.nChannels) &&
         sameOverride(first.algorithm, second.algorithm) &&
         sameOverride(first.protocol, second.protocol) &&
         sameOverride(first.path, second.path) &&
         sameOverride(first.transferMode, second.transferMode);
}

// A concrete P2P transport candidate must fix every field that defines the
// shared work/channel shape. This prevents a high-priority candidate from
// accidentally depending on a lower-priority profile or mutable defaults.
template<size_t RuleCount>
constexpr size_t firstIncompleteP2pTransportShape(const RuleSpec (&rules)[RuleCount]) {
  for (size_t ruleId = 0; ruleId < RuleCount; ruleId++) {
    const RuleSpec& rule = rules[ruleId];
    if (rule.match.scope == P2P && !rule.match.candidateTransport.any &&
        (!rule.profile.nChannels.isSet || !rule.profile.protocol.isSet ||
         !rule.profile.path.isSet || !rule.profile.transferMode.isSet))
      return ruleId;
  }
  return RuleCount;
}

// AllToAllV can place differently sized send and receive tasks in one P2P work
// item. Keep every size/placement transport rule on one shared non-transport
// shape so direction-local matches cannot select incompatible channel pools.
template<size_t RuleCount>
constexpr size_t firstMismatchedAllToAllVShape(const RuleSpec (&rules)[RuleCount]) {
  const RuleProfile* expected = nullptr;
  for (size_t ruleId = 0; ruleId < RuleCount; ruleId++) {
    const RuleSpec& rule = rules[ruleId];
    if (rule.match.scope != P2P || rule.match.collType != ALL_TO_ALL_V) continue;
    if (expected == nullptr)
      expected = &rule.profile;
    else if (!sameP2pWorkShape(*expected, rule.profile))
      return ruleId;
  }
  return RuleCount;
}

constexpr bool archPatternsOverlap(const char* first, const char* second) {
  // IsArchMatch is a prefix match, so two patterns select a common arch only
  // when one is a prefix of the other.
  for (size_t i = 0; first[i] != '\0' && second[i] != '\0'; i++)
    if (first[i] != second[i]) return false;
  return true;
}

constexpr bool intRangesOverlap(const IntRange& first, const IntRange& second) {
  return first.min <= second.max && second.min <= first.max;
}

constexpr bool rulesCanMatchSameInput(const RuleMatch& first, const RuleMatch& second) {
  return first.scope == second.scope && first.collType == second.collType &&
         first.dataSize.min <= second.dataSize.max && second.dataSize.min <= first.dataSize.max &&
         intRangesOverlap(first.nNodes, second.nNodes) && intRangesOverlap(first.nRanks, second.nRanks) &&
         intRangesOverlap(first.nChannels, second.nChannels) &&
         (first.inPlace == BoolConstraint::Any || second.inPlace == BoolConstraint::Any ||
          first.inPlace == second.inPlace) &&
         archPatternsOverlap(first.gfxArch, second.gfxArch);
}

// Scores are peaks over the inputs a rule covers, so they only rank candidates
// that cover the same inputs. Any two rules that can match one input must have
// identical byte ranges and placement constraints;
// tools/scripts/execution_policy_scores.py splits rules to satisfy this.
template<size_t RuleCount>
constexpr size_t firstMisalignedRule(const RuleSpec (&rules)[RuleCount]) {
  for (size_t second = 1; second < RuleCount; second++) {
    for (size_t first = 0; first < second; first++) {
      const RuleMatch& a = rules[first].match;
      const RuleMatch& b = rules[second].match;
      if (a.collType != b.collType || a.scope != b.scope) continue;
      if (a.dataSize.min == b.dataSize.min && a.dataSize.max == b.dataSize.max &&
          a.inPlace == b.inPlace)
        continue;
      if (rulesCanMatchSameInput(a, b)) return second;
    }
  }
  return RuleCount;
}

// Selection is first-match, so among rules that can match one input the
// earlier rule must not be measurably slower. Scores within the margin are
// measurement noise; unmeasured rules keep the position their author chose.
constexpr float kScoreOrderMargin = 1.03f;

template<size_t RuleCount>
constexpr size_t firstMisorderedRule(const RuleSpec (&rules)[RuleCount]) {
  for (size_t second = 1; second < RuleCount; second++) {
    const float later = rules[second].score.efficiency;
    if (later <= 0.0f) continue;
    for (size_t first = 0; first < second; first++) {
      const float earlier = rules[first].score.efficiency;
      if (earlier <= 0.0f || later <= earlier * kScoreOrderMargin) continue;
      if (rulesCanMatchSameInput(rules[first].match, rules[second].match)) return second;
    }
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
  if (!spec.match.candidateTransport.any) {
    appendCondition(
      &row, RCCL_EXECUTION_INPUT_ELIGIBLE_TRANSPORTS,
      RCCL_EXECUTION_COMPARE_MASK_CONTAINS,
      rcclExecutionPolicyValue::Unsigned(
        rcclExecutionTransportBit(spec.match.candidateTransport.value)));
  }
  if (spec.match.inPlace != BoolConstraint::Any)
    appendCondition(&row, RCCL_EXECUTION_INPUT_IN_PLACE, RCCL_EXECUTION_COMPARE_EQ,
                    rcclExecutionPolicyValue::Boolean(spec.match.inPlace == BoolConstraint::True));

  appendAssignment(&row, RCCL_EXECUTION_OUTPUT_N_CHANNELS, spec.profile.nChannels);
  appendAssignment(&row, RCCL_EXECUTION_OUTPUT_ALGORITHM, spec.profile.algorithm);
  appendAssignment(&row, RCCL_EXECUTION_OUTPUT_PROTOCOL, spec.profile.protocol);
  appendAssignment(&row, RCCL_EXECUTION_OUTPUT_PATH, spec.profile.path);
  appendAssignment(&row, RCCL_EXECUTION_OUTPUT_TRANSFER_MODE, spec.profile.transferMode);
  if (!spec.match.candidateTransport.any) {
    row.assignments[row.assignmentCount++] = {
      RCCL_EXECUTION_OUTPUT_TRANSPORT,
      rcclExecutionPolicyValue::Signed(spec.match.candidateTransport.value)};
  }
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
