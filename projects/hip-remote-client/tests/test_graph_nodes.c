/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file test_graph_nodes.c
 * @brief Tests for Graph Node APIs
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hip_remote/hip_remote_client.h"

#define TEST_CHECK(name, condition) do { \
    if (!(condition)) { \
        printf("FAIL: %s\n", name); \
        printf("  %s:%d\n", __FILE__, __LINE__); \
        return 1; \
    } \
    printf("  PASS: %s\n", name); \
} while (0)

static int test_graph_add_empty_node(void) {
    printf("\nTest: hipGraphAddEmptyNode\n");

    /* Create a graph */
    hipGraph_t graph = NULL;
    hipError_t err = hipGraphCreate(&graph, 0);
    TEST_CHECK("hipGraphCreate succeeds", err == hipSuccess && graph != NULL);

    /* Add an empty node */
    hipGraphNode_t emptyNode = NULL;
    err = hipGraphAddEmptyNode(&emptyNode, graph, NULL, 0);
    TEST_CHECK("hipGraphAddEmptyNode succeeds", err == hipSuccess);
    TEST_CHECK("empty node is not NULL", emptyNode != NULL);
    printf("  Created empty node: %p\n", emptyNode);

    /* Get node type */
    hipGraphNodeType type = hipGraphNodeTypeKernel;  /* Set to wrong value */
    err = hipGraphNodeGetType(emptyNode, &type);
    TEST_CHECK("hipGraphNodeGetType succeeds", err == hipSuccess);
    TEST_CHECK("node type is empty", type == hipGraphNodeTypeEmpty);
    printf("  Node type: %d (expected %d)\n", type, hipGraphNodeTypeEmpty);

    /* Get graph nodes */
    size_t numNodes = 0;
    err = hipGraphGetNodes(graph, NULL, &numNodes);
    TEST_CHECK("hipGraphGetNodes count succeeds", err == hipSuccess);
    TEST_CHECK("graph has 1 node", numNodes == 1);

    /* Get root nodes */
    size_t numRoots = 0;
    err = hipGraphGetRootNodes(graph, NULL, &numRoots);
    TEST_CHECK("hipGraphGetRootNodes count succeeds", err == hipSuccess);
    TEST_CHECK("graph has 1 root", numRoots == 1);

    /* Clean up */
    hipGraphDestroy(graph);
    return 0;
}

static int test_graph_add_dependencies(void) {
    printf("\nTest: hipGraphAddDependencies\n");

    hipGraph_t graph = NULL;
    hipError_t err = hipGraphCreate(&graph, 0);
    TEST_CHECK("create graph", err == hipSuccess);

    /* Add two empty nodes */
    hipGraphNode_t node1 = NULL, node2 = NULL;
    err = hipGraphAddEmptyNode(&node1, graph, NULL, 0);
    TEST_CHECK("add node1", err == hipSuccess);
    err = hipGraphAddEmptyNode(&node2, graph, NULL, 0);
    TEST_CHECK("add node2", err == hipSuccess);

    /* Initially both should be root nodes */
    size_t numRoots = 0;
    err = hipGraphGetRootNodes(graph, NULL, &numRoots);
    TEST_CHECK("get root count", err == hipSuccess);
    TEST_CHECK("2 roots initially", numRoots == 2);

    /* Add dependency: node1 -> node2 */
    hipGraphNode_t from[] = { node1 };
    hipGraphNode_t to[] = { node2 };
    err = hipGraphAddDependencies(graph, from, to, 1);
    TEST_CHECK("hipGraphAddDependencies succeeds", err == hipSuccess);

    /* Now only node1 should be a root */
    err = hipGraphGetRootNodes(graph, NULL, &numRoots);
    TEST_CHECK("get root count after deps", err == hipSuccess);
    TEST_CHECK("1 root after dependency", numRoots == 1);

    hipGraphDestroy(graph);
    return 0;
}

static int test_graph_add_memcpy_node_1d(void) {
    printf("\nTest: hipGraphAddMemcpyNode1D\n");

    hipGraph_t graph = NULL;
    hipError_t err = hipGraphCreate(&graph, 0);
    TEST_CHECK("create graph", err == hipSuccess);

    /* Allocate device memory */
    void* d_src = NULL;
    void* d_dst = NULL;
    err = hipMalloc(&d_src, 1024);
    TEST_CHECK("malloc src", err == hipSuccess);
    err = hipMalloc(&d_dst, 1024);
    TEST_CHECK("malloc dst", err == hipSuccess);

    /* Add memcpy node */
    hipGraphNode_t memcpyNode = NULL;
    err = hipGraphAddMemcpyNode1D(&memcpyNode, graph, NULL, 0,
                                   d_dst, d_src, 1024, hipMemcpyDeviceToDevice);
    TEST_CHECK("hipGraphAddMemcpyNode1D succeeds", err == hipSuccess);
    TEST_CHECK("memcpy node is not NULL", memcpyNode != NULL);
    printf("  Created memcpy node: %p\n", memcpyNode);

    /* Check node type */
    hipGraphNodeType type;
    err = hipGraphNodeGetType(memcpyNode, &type);
    TEST_CHECK("get node type", err == hipSuccess);
    TEST_CHECK("node type is memcpy", type == hipGraphNodeTypeMemcpy);

    /* Instantiate and launch */
    hipGraphExec_t exec = NULL;
    err = hipGraphInstantiate(&exec, graph, NULL, NULL, 0);
    TEST_CHECK("instantiate graph", err == hipSuccess);

    err = hipGraphLaunch(exec, NULL);
    TEST_CHECK("launch graph", err == hipSuccess);

    err = hipDeviceSynchronize();
    TEST_CHECK("sync after launch", err == hipSuccess);

    /* Clean up */
    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);
    hipFree(d_src);
    hipFree(d_dst);
    return 0;
}

static int test_graph_add_memset_node(void) {
    printf("\nTest: hipGraphAddMemsetNode\n");

    hipGraph_t graph = NULL;
    hipError_t err = hipGraphCreate(&graph, 0);
    TEST_CHECK("create graph", err == hipSuccess);

    /* Allocate device memory */
    void* d_ptr = NULL;
    err = hipMalloc(&d_ptr, 1024);
    TEST_CHECK("malloc", err == hipSuccess);

    /* Add memset node */
    hipMemsetParams params = {
        .dst = d_ptr,
        .pitch = 0,
        .value = 0x42,
        .elementSize = 1,
        .width = 1024,
        .height = 1
    };

    hipGraphNode_t memsetNode = NULL;
    err = hipGraphAddMemsetNode(&memsetNode, graph, NULL, 0, &params);
    TEST_CHECK("hipGraphAddMemsetNode succeeds", err == hipSuccess);
    TEST_CHECK("memset node is not NULL", memsetNode != NULL);
    printf("  Created memset node: %p\n", memsetNode);

    /* Check node type */
    hipGraphNodeType type;
    err = hipGraphNodeGetType(memsetNode, &type);
    TEST_CHECK("get node type", err == hipSuccess);
    TEST_CHECK("node type is memset", type == hipGraphNodeTypeMemset);

    /* Instantiate and launch */
    hipGraphExec_t exec = NULL;
    err = hipGraphInstantiate(&exec, graph, NULL, NULL, 0);
    TEST_CHECK("instantiate graph", err == hipSuccess);

    err = hipGraphLaunch(exec, NULL);
    TEST_CHECK("launch graph", err == hipSuccess);

    err = hipDeviceSynchronize();
    TEST_CHECK("sync after launch", err == hipSuccess);

    /* Verify memset worked */
    unsigned char host_data[16];
    err = hipMemcpy(host_data, d_ptr, 16, hipMemcpyDeviceToHost);
    TEST_CHECK("copy back to verify", err == hipSuccess);
    TEST_CHECK("memset value correct", host_data[0] == 0x42 && host_data[15] == 0x42);

    /* Clean up */
    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);
    hipFree(d_ptr);
    return 0;
}

static int test_graph_with_dependencies(void) {
    printf("\nTest: Graph with chained nodes\n");

    hipGraph_t graph = NULL;
    hipError_t err = hipGraphCreate(&graph, 0);
    TEST_CHECK("create graph", err == hipSuccess);

    /* Allocate memory */
    void* d_a = NULL;
    void* d_b = NULL;
    err = hipMalloc(&d_a, 1024);
    TEST_CHECK("malloc a", err == hipSuccess);
    err = hipMalloc(&d_b, 1024);
    TEST_CHECK("malloc b", err == hipSuccess);

    /* Node 1: memset d_a to 0xAA */
    hipMemsetParams params1 = {
        .dst = d_a, .pitch = 0, .value = 0xAA,
        .elementSize = 1, .width = 1024, .height = 1
    };
    hipGraphNode_t node1 = NULL;
    err = hipGraphAddMemsetNode(&node1, graph, NULL, 0, &params1);
    TEST_CHECK("add memset node 1", err == hipSuccess);

    /* Node 2: copy d_a to d_b (depends on node1) */
    hipGraphNode_t deps[] = { node1 };
    hipGraphNode_t node2 = NULL;
    err = hipGraphAddMemcpyNode1D(&node2, graph, deps, 1,
                                   d_b, d_a, 1024, hipMemcpyDeviceToDevice);
    TEST_CHECK("add memcpy node 2", err == hipSuccess);

    /* Node 3: memset d_a to 0xBB (depends on node2 to ensure ordering) */
    hipMemsetParams params3 = {
        .dst = d_a, .pitch = 0, .value = 0xBB,
        .elementSize = 1, .width = 1024, .height = 1
    };
    hipGraphNode_t deps2[] = { node2 };
    hipGraphNode_t node3 = NULL;
    err = hipGraphAddMemsetNode(&node3, graph, deps2, 1, &params3);
    TEST_CHECK("add memset node 3", err == hipSuccess);

    /* Verify structure */
    size_t numNodes = 0;
    err = hipGraphGetNodes(graph, NULL, &numNodes);
    TEST_CHECK("get node count", err == hipSuccess);
    TEST_CHECK("graph has 3 nodes", numNodes == 3);

    size_t numRoots = 0;
    err = hipGraphGetRootNodes(graph, NULL, &numRoots);
    TEST_CHECK("get root count", err == hipSuccess);
    TEST_CHECK("graph has 1 root", numRoots == 1);

    /* Instantiate and launch */
    hipGraphExec_t exec = NULL;
    err = hipGraphInstantiate(&exec, graph, NULL, NULL, 0);
    TEST_CHECK("instantiate graph", err == hipSuccess);

    err = hipGraphLaunch(exec, NULL);
    TEST_CHECK("launch graph", err == hipSuccess);

    err = hipDeviceSynchronize();
    TEST_CHECK("sync", err == hipSuccess);

    /* Verify: d_a should be 0xBB, d_b should be 0xAA */
    unsigned char host_a[16], host_b[16];
    err = hipMemcpy(host_a, d_a, 16, hipMemcpyDeviceToHost);
    TEST_CHECK("copy a", err == hipSuccess);
    err = hipMemcpy(host_b, d_b, 16, hipMemcpyDeviceToHost);
    TEST_CHECK("copy b", err == hipSuccess);

    TEST_CHECK("d_a has 0xBB", host_a[0] == 0xBB && host_a[15] == 0xBB);
    TEST_CHECK("d_b has 0xAA", host_b[0] == 0xAA && host_b[15] == 0xAA);

    /* Clean up */
    hipGraphExecDestroy(exec);
    hipGraphDestroy(graph);
    hipFree(d_a);
    hipFree(d_b);
    return 0;
}

static int test_null_checks(void) {
    printf("\nTest: Graph Node NULL parameter checks\n");

    hipError_t err;
    hipGraph_t graph = NULL;
    hipGraphCreate(&graph, 0);

    hipGraphNode_t node = NULL;

    /* hipGraphAddEmptyNode NULL checks */
    err = hipGraphAddEmptyNode(NULL, graph, NULL, 0);
    TEST_CHECK("hipGraphAddEmptyNode(NULL, ...) fails", err == hipErrorInvalidValue);

    err = hipGraphAddEmptyNode(&node, NULL, NULL, 0);
    TEST_CHECK("hipGraphAddEmptyNode(..., NULL, ...) fails", err == hipErrorInvalidValue);

    /* hipGraphAddMemcpyNode1D NULL checks */
    err = hipGraphAddMemcpyNode1D(NULL, graph, NULL, 0, NULL, NULL, 0, hipMemcpyDefault);
    TEST_CHECK("hipGraphAddMemcpyNode1D(NULL, ...) fails", err == hipErrorInvalidValue);

    /* hipGraphAddMemsetNode NULL checks */
    err = hipGraphAddMemsetNode(NULL, graph, NULL, 0, NULL);
    TEST_CHECK("hipGraphAddMemsetNode(NULL, ...) fails", err == hipErrorInvalidValue);

    /* hipGraphNodeGetType NULL checks */
    hipGraphNodeType type;
    err = hipGraphNodeGetType(NULL, &type);
    TEST_CHECK("hipGraphNodeGetType(NULL, ...) fails", err == hipErrorInvalidValue);

    /* hipGraphDestroyNode NULL check */
    err = hipGraphDestroyNode(NULL);
    TEST_CHECK("hipGraphDestroyNode(NULL) fails", err == hipErrorInvalidValue);

    hipGraphDestroy(graph);
    return 0;
}

int main(void) {
    printf("=== Remote HIP Graph Node Tests ===\n");

    const char* host = getenv("TF_WORKER_HOST");
    if (!host || strlen(host) == 0) {
        printf("\nNOTE: TF_WORKER_HOST not set, using localhost\n");
    }

    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count == 0) {
        printf("SKIP: No GPUs available (err=%d, count=%d)\n", err, count);
        return 77;
    }
    printf("Found %d GPU(s)\n", count);

    int failures = 0;
    failures += test_null_checks();
    failures += test_graph_add_empty_node();
    failures += test_graph_add_dependencies();
    failures += test_graph_add_memcpy_node_1d();
    failures += test_graph_add_memset_node();
    failures += test_graph_with_dependencies();

    printf("\n=== Results ===\n");
    if (failures > 0) {
        printf("%d graph node test(s) FAILED\n", failures);
        return 1;
    }
    printf("All graph node tests PASSED\n");
    return 0;
}
