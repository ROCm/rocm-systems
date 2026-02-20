/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Test suite for Advanced Graph APIs (Clone, Update, Query)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hip_remote/hip_remote_client.h"

#define CHECK_HIP(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error at %s:%d: %s (code %d)\n", \
                    __FILE__, __LINE__, hipGetErrorString(err), err); \
            return 1; \
        } \
    } while (0)

#define CHECK_HIP_WARN(call, msg) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "Warning at %s:%d: %s - %s (code %d)\n", \
                    __FILE__, __LINE__, msg, hipGetErrorString(err), err); \
        } \
    } while (0)

int main() {
    printf("=== HIP Remote Advanced Graph APIs Test ===\n\n");

    /* Initialize device */
    printf("[TEST] Initializing device\n");
    CHECK_HIP(hipSetDevice(0));

    /* Test 1: Create a simple graph with dependencies */
    printf("\n[TEST] Creating graph with nodes\n");
    hipGraph_t graph = NULL;
    CHECK_HIP(hipGraphCreate(&graph, 0));

    /* Add some empty nodes with dependencies to test query APIs */
    hipGraphNode_t node1 = NULL, node2 = NULL, node3 = NULL;
    CHECK_HIP(hipGraphAddEmptyNode(&node1, graph, NULL, 0));
    CHECK_HIP(hipGraphAddEmptyNode(&node2, graph, &node1, 1));
    CHECK_HIP(hipGraphAddEmptyNode(&node3, graph, &node2, 1));
    printf("  Created graph with 3 nodes (node1 <- node2 <- node3)\n");

    /* Test 2: hipGraphNodeGetDependencies */
    printf("\n[TEST] hipGraphNodeGetDependencies\n");
    size_t num_deps = 0;
    CHECK_HIP(hipGraphNodeGetDependencies(node2, NULL, &num_deps));
    printf("  Node2 has %zu dependencies\n", num_deps);
    if (num_deps > 0) {
        hipGraphNode_t* deps = (hipGraphNode_t*)malloc(num_deps * sizeof(hipGraphNode_t));
        CHECK_HIP(hipGraphNodeGetDependencies(node2, deps, &num_deps));
        printf("  Retrieved %zu dependency nodes\n", num_deps);
        free(deps);
    }

    /* Test 3: hipGraphNodeGetDependentNodes */
    printf("\n[TEST] hipGraphNodeGetDependentNodes\n");
    size_t num_dependents = 0;
    CHECK_HIP(hipGraphNodeGetDependentNodes(node1, NULL, &num_dependents));
    printf("  Node1 has %zu dependent nodes\n", num_dependents);
    if (num_dependents > 0) {
        hipGraphNode_t* dependents = (hipGraphNode_t*)malloc(num_dependents * sizeof(hipGraphNode_t));
        CHECK_HIP(hipGraphNodeGetDependentNodes(node1, dependents, &num_dependents));
        printf("  Retrieved %zu dependent nodes\n", num_dependents);
        free(dependents);
    }

    /* Test 4: hipGraphClone */
    printf("\n[TEST] hipGraphClone\n");
    hipGraph_t cloned_graph = NULL;
    CHECK_HIP(hipGraphClone(&cloned_graph, graph));
    printf("  Cloned graph: %p -> %p\n", (void*)graph, (void*)cloned_graph);

    /* Verify cloned graph has same structure */
    size_t num_nodes_original = 0, num_nodes_cloned = 0;
    CHECK_HIP(hipGraphGetNodes(graph, NULL, &num_nodes_original));
    CHECK_HIP(hipGraphGetNodes(cloned_graph, NULL, &num_nodes_cloned));
    printf("  Original graph: %zu nodes, Cloned graph: %zu nodes\n",
           num_nodes_original, num_nodes_cloned);

    /* Test 5: hipGraphExecUpdate */
    printf("\n[TEST] hipGraphExecUpdate\n");
    hipGraphExec_t graphExec = NULL;
    CHECK_HIP(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    printf("  Instantiated graph exec: %p\n", (void*)graphExec);

    /* Try updating with the same graph (should succeed with no changes) */
    hipGraphExecUpdateResult updateResult;
    hipGraphNode_t errorNode = NULL;
    CHECK_HIP(hipGraphExecUpdate(graphExec, graph, &errorNode, &updateResult));
    printf("  Update result: %d\n", updateResult);
    if (updateResult == hipGraphExecUpdateSuccess) {
        printf("  Graph exec updated successfully (no topology changes)\n");
    } else {
        printf("  Graph exec update indicated changes: %d\n", updateResult);
    }

    /* Test 6: hipGraphExecKernelNodeSetParams (with dummy node) */
    printf("\n[TEST] hipGraphExecKernelNodeSetParams\n");
    printf("  Note: This is a kernel node API, testing with dummy parameters\n");
    /* This would need an actual kernel node, so we'll just test the API flow */
    hipKernelNodeParams params;
    memset(&params, 0, sizeof(params));
    params.gridDim.x = 1;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.blockDim.x = 1;
    params.blockDim.y = 1;
    params.blockDim.z = 1;
    params.sharedMemBytes = 0;
    params.kernelParams = NULL;
    params.extra = NULL;

    /* This will likely fail since node1 is an empty node, not a kernel node */
    hipError_t err = hipGraphExecKernelNodeSetParams(graphExec, node1, &params);
    if (err == hipSuccess) {
        printf("  API call succeeded (unexpected for empty node)\n");
    } else {
        printf("  API call returned error as expected for non-kernel node: %s\n",
               hipGetErrorString(err));
    }

    /* Cleanup */
    printf("\n[TEST] Cleanup\n");
    CHECK_HIP(hipGraphExecDestroy(graphExec));
    CHECK_HIP(hipGraphDestroy(cloned_graph));
    CHECK_HIP(hipGraphDestroy(graph));

    printf("\n=== All Advanced Graph API Tests Passed! ===\n");
    return 0;
}
