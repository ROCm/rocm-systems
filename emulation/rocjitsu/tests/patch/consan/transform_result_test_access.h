// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_legacy_lowering.h"
#include "rocjitsu/code/patch/consan/consan_pipeline.h"

#include <utility>

namespace rocjitsu {

/// Test-only access to the raw-mechanism publication seam.
///
/// Production transforms never accept or return `ConSanResult` at their hook
/// boundary. Mechanism and HSA-hook unit tests still need synthetic lowering
/// artifacts while the prototype lowerer is decomposed, so this adapter alone
/// may publish such a fixture through the exact production validation and
/// stage-record construction path. Keeping the privilege in the test tree
/// prevents raw compatibility state from becoming a production API again.
struct TransformResultTestAccess {
  [[nodiscard]] static TransformResult
  publish(std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
          const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
          const ConSanDebugOverrides &debug, const MutationRequest &mutation,
          const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
          ConSanResult mechanism_result) {
    return TransformResult::publish_optional(code_object_bytes, request, transform_policy,
                                             runtime_policy, debug, mutation, capabilities,
                                             resources, std::move(mechanism_result));
  }
};

} // namespace rocjitsu
