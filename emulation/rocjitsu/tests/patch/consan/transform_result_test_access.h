// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_pipeline.h"

#include <utility>

namespace rocjitsu {

/// Test-only access to static-artifact publication.
///
/// HSA-hook and pipeline unit tests need synthetic lowering artifacts without
/// constructing private working candidates or mutable lowering options. This
/// adapter publishes only the same shared artifact value returned by the
/// production boundary through exact stage-record construction and validation.
struct TransformResultTestAccess {
  [[nodiscard]] static TransformResult
  publish(std::span<const uint8_t> code_object_bytes, const ConSanRequest &request,
          const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
          const ConSanDebugOverrides &debug, const MutationRequest &mutation,
          const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
          ConSanTransformArtifacts artifacts) {
    return TransformResult::publish_optional(code_object_bytes, request, transform_policy,
                                             runtime_policy, debug, mutation, capabilities,
                                             resources, std::move(artifacts));
  }
};

} // namespace rocjitsu
