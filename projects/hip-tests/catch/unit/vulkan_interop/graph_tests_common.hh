/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stddef.h>

#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>

template <typename F> void GraphAddNodeCommonNegativeTests(F f, hipGraph_t graph) {
  hipGraphNode_t node = nullptr;
  SECTION("graph == nullptr") {
    HIP_CHECK_ERROR(f(&node, nullptr, nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(f(nullptr, graph, nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("dependencies == nullptr with size != 0") {
    HIP_CHECK_ERROR(f(&node, graph, nullptr, 1), hipErrorInvalidValue);
  }

// Disabled on AMD due to defect - EXSWHTEC-202
#if HT_NVIDIA
  SECTION("Node in dependency is from different graph") {
    hipGraph_t other_graph = nullptr;
    HIP_CHECK(hipGraphCreate(&other_graph, 0));
    hipGraphNode_t other_node = nullptr;
    HIP_CHECK(hipGraphAddEmptyNode(&other_node, other_graph, nullptr, 0));
    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));
    HIP_CHECK_ERROR(f(&node, graph, &other_node, 1), hipErrorInvalidValue);
    HIP_CHECK(hipGraphDestroy(other_graph));
  }
#endif

// Disabled on AMD due to defect - EXSWHTEC-201
#if HT_NVIDIA
  SECTION("Duplicate node in dependencies") {
    hipGraphNode_t dep_node = nullptr;
    // Need to create two nodes to avoid overlap with Invalid numNodes case
    // First one is left dangling as the graph will be destroyed after the section anyway
    HIP_CHECK(hipGraphAddEmptyNode(&dep_node, graph, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&dep_node, graph, nullptr, 0));
    hipGraphNode_t deps[] = {dep_node, dep_node};
    HIP_CHECK_ERROR(f(&node, graph, deps, 2), hipErrorInvalidValue);
  }
#endif
}