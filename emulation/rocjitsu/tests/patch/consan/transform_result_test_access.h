// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_pipeline.h"
#include "rocjitsu/code/patch/consan/consan_transform_diagnostics.h"

#include <utility>

namespace rocjitsu::consan {

/// Test-only access to static-artifact publication.
///
/// HSA-hook and pipeline unit tests need synthetic lowering artifacts without
/// constructing private working candidates or mutable lowering options. This
/// adapter publishes only the same shared artifact value returned by the
/// production boundary through exact stage-record construction and validation.
struct TransformResultTestAccess {
  [[nodiscard]] static TransformDiagnosticReport diagnostic_report(const TransformResult &result) {
    return transform_diagnostic_report(result);
  }

  [[nodiscard]] static TransformResult
  publish(std::span<const uint8_t> code_object_bytes, const Request &request,
          const TransformPolicy &transform_policy, const RuntimePolicy &runtime_policy,
          const DebugOverrides &debug, const MutationRequest &mutation,
          const RuntimeCapabilities &capabilities, const BoundRuntimeResources &resources,
          TransformArtifacts artifacts) {
    return TransformResult::execute_test_transaction(code_object_bytes, request, transform_policy,
                                                     runtime_policy, debug, mutation, capabilities,
                                                     resources, std::move(artifacts));
  }
};

} // namespace rocjitsu::consan
