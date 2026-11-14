/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

namespace graph_fence_test {

inline constexpr char kChildExecutable[] = "hipGraphInstantiateNullStreamFence_exe";
inline constexpr char kScenarioGraph[] = "graph";
inline constexpr char kScenarioNoGraph[] = "nograph";
inline constexpr char kScenarioCapturedSystemScope[] = "captured_system_scope";
inline constexpr char kFenceRegion[] = "fence";
inline constexpr char kCapturedSystemScopeRegion[] = "captured_system_scope";

}  // namespace graph_fence_test
