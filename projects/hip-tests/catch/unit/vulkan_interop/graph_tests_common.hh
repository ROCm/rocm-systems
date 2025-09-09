/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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

  SECTION("Invalid numNodes") {
    hipGraphNode_t dep_node = nullptr;
    HIP_CHECK(hipGraphAddEmptyNode(&dep_node, graph, nullptr, 0));
    HIP_CHECK_ERROR(f(&node, graph, &dep_node, 2), hipErrorInvalidValue);
  }

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