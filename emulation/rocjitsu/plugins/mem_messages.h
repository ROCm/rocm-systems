// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mem_messages.h
/// @brief Inter-component messages for the async cycle-model memory path: a CU's
/// CuCycleModel sends MemReqMsg for its L1-miss/bypass traffic to the shared
/// MemSysCycleModel, which replies MemCompletionMsg. Payloads are in CU-cycle units
/// (each SharedReq already carries its arrive_cyc); the simdojo Link carries the
/// message in engine ticks for delivery ordering only.

#pragma once

#include "simdojo/sim/component.h"   // ComponentID
#include "simdojo/sim/message.h"

#include "cycle_model/CacheModels.h"        // SharedReq
#include "cycle_model/MemReqStateMachine.h" // MemReqId

#include <cstdint>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

/// One CU's shared-hierarchy request for a single memory instruction.
class MemReqMsg : public simdojo::Message {
public:
  MemReqMsg(cycle_model::MemReqId rid, std::vector<cycle_model::SharedReq> shared,
            simdojo::ComponentID requester)
      : rid(rid), shared(std::move(shared)), requester_id(requester) {}
  cycle_model::MemReqId rid;
  std::vector<cycle_model::SharedReq> shared;
  simdojo::ComponentID requester_id;     // globally-unique sender (assert/debug)
};

/// The shared model's completion reply, echoing the CU-local rid.
class MemCompletionMsg : public simdojo::Message {
public:
  MemCompletionMsg(cycle_model::MemReqId rid, uint64_t completion_cyc)
      : rid(rid), completion_cyc(completion_cyc) {}
  cycle_model::MemReqId rid;
  uint64_t completion_cyc;               // in CU-cycle units
};

}  // namespace amdgpu
}  // namespace rocjitsu
