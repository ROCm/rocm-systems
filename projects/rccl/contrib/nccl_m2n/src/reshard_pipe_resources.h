/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 ************************************************************************/

#ifndef NCCLM2N_RESHARD_PIPE_RESOURCES_H_
#define NCCLM2N_RESHARD_PIPE_RESOURCES_H_

#include <cstdint>
#include <limits>

inline constexpr int RESHARD_PIPE_MAX_PARENT_NAMESPACES = 64;
inline constexpr int RESHARD_PIPE_DEFAULT_PARENT_CAPACITY = 8;

/* Local communicator identities are lookup keys, never cross-rank identities.
 * Propose without mutation; callers collectively verify the proposed ID before
 * committing it. IDs cannot be recycled while their FIFO histories are live. */
struct ReshardPipeNamespacePool {
  uintptr_t parents[RESHARD_PIPE_MAX_PARENT_NAMESPACES];
  int count;

  int propose(uintptr_t parent, int capacity) const {
    if (parent == 0 || capacity <= 0 || capacity > RESHARD_PIPE_MAX_PARENT_NAMESPACES) return -1;
    for (int i = 0; i < count; i++) {
      if (parents[i] == parent) return i < capacity ? i : -1;
    }
    return count < capacity ? count : -1;
  }

  bool commit(uintptr_t parent, int capacity, int id) {
    if (id < 0 || propose(parent, capacity) != id) return false;
    if (id == count) parents[count++] = parent;
    return true;
  }
};

/* Persistent signal/counter capacity and execution concurrency are independent.
 * PACK can supply its legacy lane-sized totals to the same resource helper. */
struct ReshardCommBResourcePlan {
  int signals;
  int counters;
  int contexts;
  int barriers;

  bool operator==(const ReshardCommBResourcePlan& other) const {
    return signals == other.signals && counters == other.counters && contexts == other.contexts &&
           barriers == other.barriers;
  }
};

/* Namespace agreement already establishes this mapping on all participants.
 * Unrelated local parent history must not influence execution-lane choice. */
inline int reshardPipeResourceLane(int namespaceId, int lanes) {
  return namespaceId >= 0 && lanes > 0 ? namespaceId % lanes : -1;
}

inline bool reshardCheckedResourceProduct(int a, int b, int* out) {
  if (out == nullptr || a < 0 || b < 0 || (a != 0 && b > std::numeric_limits<int>::max() / a)) return false;
  *out = a * b;
  return true;
}

inline bool reshardBuildPipeCommBResourcePlan(int parentCapacity, int lanes, int controlSlots,
                                              int signalsPerControl, int countersPerControl, int contextsPerLane,
                                              ReshardCommBResourcePlan* out) {
  if (out == nullptr || parentCapacity <= 0 || parentCapacity > RESHARD_PIPE_MAX_PARENT_NAMESPACES ||
      lanes <= 0 || controlSlots <= 0 || signalsPerControl <= 0 || countersPerControl <= 0 ||
      contextsPerLane <= 0) return false;
  int controls = 0;
  ReshardCommBResourcePlan plan{};
  if (!reshardCheckedResourceProduct(parentCapacity, controlSlots, &controls) ||
      !reshardCheckedResourceProduct(controls, signalsPerControl, &plan.signals) ||
      !reshardCheckedResourceProduct(controls, countersPerControl, &plan.counters) ||
      !reshardCheckedResourceProduct(lanes, contextsPerLane, &plan.contexts)) return false;
  *out = plan;
  return true;
}

#endif
