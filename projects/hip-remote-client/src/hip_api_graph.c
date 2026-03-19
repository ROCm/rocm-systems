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
 * @file hip_api_graph.c
 * @brief Graph Node API implementations for remote HIP
 */

#include "hip_remote/hip_remote_internal.h"
#include "hip_remote/hip_remote_protocol.h"

#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Graph Node APIs
 * ============================================================================ */

hipError_t hipGraphAddMemcpyNode1D(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                    const hipGraphNode_t* pDependencies, size_t numDependencies,
                                    void* dst, const void* src, size_t count, hipMemcpyKind kind) {
    if (!pGraphNode || !graph) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > 0 && !pDependencies) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > HIP_REMOTE_MAX_GRAPH_DEPENDENCIES) {
        return hipErrorInvalidValue;
    }

    /* Build request with dependencies */
    size_t deps_size = numDependencies * sizeof(uint64_t);
    size_t req_size = sizeof(HipRemoteGraphAddMemcpyNode1DRequest) + deps_size;
    uint8_t* buffer = (uint8_t*)malloc(req_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    HipRemoteGraphAddMemcpyNode1DRequest* req = (HipRemoteGraphAddMemcpyNode1DRequest*)buffer;
    req->graph = (uint64_t)(uintptr_t)graph;
    req->num_deps = (uint32_t)numDependencies;
    req->reserved = 0;
    req->dst = (uint64_t)(uintptr_t)dst;
    req->src = (uint64_t)(uintptr_t)src;
    req->count = count;
    req->kind = (int32_t)kind;

    /* Copy dependencies - caller guarantees pDependencies is valid for numDependencies elements */
    uint64_t* deps = (uint64_t*)(buffer + sizeof(HipRemoteGraphAddMemcpyNode1DRequest));
    for (size_t i = 0; i < numDependencies; i++) {
        deps[i] = (uint64_t)(uintptr_t)pDependencies[i];
    }

    HipRemoteGraphAddNodeResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_ADD_MEMCPY_NODE_1D,
        buffer, req_size,
        &resp, sizeof(resp)
    );

    free(buffer);

    if (err == hipSuccess) {
        *pGraphNode = (hipGraphNode_t)(uintptr_t)resp.node;
    } else {
        *pGraphNode = NULL;
    }
    return err;
}

hipError_t hipGraphAddMemcpyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipMemcpy3DParms* pCopyParams) {
    if (!pGraphNode || !graph || !pCopyParams) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > 0 && !pDependencies) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > HIP_REMOTE_MAX_GRAPH_DEPENDENCIES) {
        return hipErrorInvalidValue;
    }

    /* Build request with dependencies */
    size_t deps_size = numDependencies * sizeof(uint64_t);
    size_t req_size = sizeof(HipRemoteGraphAddMemcpyNodeRequest) + deps_size;
    uint8_t* buffer = (uint8_t*)malloc(req_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    HipRemoteGraphAddMemcpyNodeRequest* req = (HipRemoteGraphAddMemcpyNodeRequest*)buffer;
    req->graph = (uint64_t)(uintptr_t)graph;
    req->num_deps = (uint32_t)numDependencies;
    req->reserved = 0;
    req->dst = (uint64_t)(uintptr_t)pCopyParams->dstPtr.ptr;
    req->src = (uint64_t)(uintptr_t)pCopyParams->srcPtr.ptr;
    req->width = pCopyParams->extent.width;
    req->height = pCopyParams->extent.height;
    req->dst_pitch = pCopyParams->dstPtr.pitch;
    req->src_pitch = pCopyParams->srcPtr.pitch;
    req->kind = (int32_t)pCopyParams->kind;
    req->reserved2 = 0;

    /* Copy dependencies */
    uint64_t* deps = (uint64_t*)(buffer + sizeof(HipRemoteGraphAddMemcpyNodeRequest));
    for (size_t i = 0; i < numDependencies; i++) {
        deps[i] = (uint64_t)(uintptr_t)pDependencies[i];
    }

    HipRemoteGraphAddNodeResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_ADD_MEMCPY_NODE,
        buffer, req_size,
        &resp, sizeof(resp)
    );

    free(buffer);

    if (err == hipSuccess) {
        *pGraphNode = (hipGraphNode_t)(uintptr_t)resp.node;
    } else {
        *pGraphNode = NULL;
    }
    return err;
}

hipError_t hipGraphAddMemsetNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipMemsetParams* pMemsetParams) {
    if (!pGraphNode || !graph || !pMemsetParams) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > 0 && !pDependencies) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > HIP_REMOTE_MAX_GRAPH_DEPENDENCIES) {
        return hipErrorInvalidValue;
    }

    /* Build request with dependencies */
    size_t deps_size = numDependencies * sizeof(uint64_t);
    size_t req_size = sizeof(HipRemoteGraphAddMemsetNodeRequest) + deps_size;
    uint8_t* buffer = (uint8_t*)malloc(req_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    HipRemoteGraphAddMemsetNodeRequest* req = (HipRemoteGraphAddMemsetNodeRequest*)buffer;
    req->graph = (uint64_t)(uintptr_t)graph;
    req->num_deps = (uint32_t)numDependencies;
    req->reserved = 0;
    req->dst = (uint64_t)(uintptr_t)pMemsetParams->dst;
    req->pitch = pMemsetParams->pitch;
    req->value = (int32_t)pMemsetParams->value;
    req->element_size = pMemsetParams->elementSize;
    req->width = pMemsetParams->width;
    req->height = pMemsetParams->height;

    /* Copy dependencies */
    uint64_t* deps = (uint64_t*)(buffer + sizeof(HipRemoteGraphAddMemsetNodeRequest));
    for (size_t i = 0; i < numDependencies; i++) {
        deps[i] = (uint64_t)(uintptr_t)pDependencies[i];
    }

    HipRemoteGraphAddNodeResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_ADD_MEMSET_NODE,
        buffer, req_size,
        &resp, sizeof(resp)
    );

    free(buffer);

    if (err == hipSuccess) {
        *pGraphNode = (hipGraphNode_t)(uintptr_t)resp.node;
    } else {
        *pGraphNode = NULL;
    }
    return err;
}

hipError_t hipGraphAddKernelNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                  const hipGraphNode_t* pDependencies, size_t numDependencies,
                                  const hipKernelNodeParams* pNodeParams) {
    if (!pGraphNode || !graph || !pNodeParams) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > 0 && !pDependencies) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > HIP_REMOTE_MAX_GRAPH_DEPENDENCIES) {
        return hipErrorInvalidValue;
    }

    /* Count kernel arguments - NULL terminated or use typical max */
    uint32_t num_args = 0;
    if (pNodeParams->kernelParams) {
        while (pNodeParams->kernelParams[num_args] != NULL && num_args < 256) {
            num_args++;
        }
    }

    /* Build request with dependencies and args */
    size_t deps_size = numDependencies * sizeof(uint64_t);
    size_t args_size = num_args * sizeof(uint64_t);
    size_t req_size = sizeof(HipRemoteGraphAddKernelNodeRequest) + deps_size + args_size;
    uint8_t* buffer = (uint8_t*)malloc(req_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    HipRemoteGraphAddKernelNodeRequest* req = (HipRemoteGraphAddKernelNodeRequest*)buffer;
    req->graph = (uint64_t)(uintptr_t)graph;
    req->num_deps = (uint32_t)numDependencies;
    req->reserved = 0;
    req->func = (uint64_t)(uintptr_t)pNodeParams->func;
    req->grid_dim_x = pNodeParams->gridDim.x;
    req->grid_dim_y = pNodeParams->gridDim.y;
    req->grid_dim_z = pNodeParams->gridDim.z;
    req->block_dim_x = pNodeParams->blockDim.x;
    req->block_dim_y = pNodeParams->blockDim.y;
    req->block_dim_z = pNodeParams->blockDim.z;
    req->shared_mem_bytes = pNodeParams->sharedMemBytes;
    req->num_args = num_args;

    /* Copy dependencies */
    uint64_t* deps = (uint64_t*)(buffer + sizeof(HipRemoteGraphAddKernelNodeRequest));
    for (size_t i = 0; i < numDependencies; i++) {
        deps[i] = (uint64_t)(uintptr_t)pDependencies[i];
    }

    /* Copy kernel arguments (assume pointer-sized) */
    uint64_t* args = deps + numDependencies;
    for (uint32_t i = 0; i < num_args; i++) {
        args[i] = *(uint64_t*)pNodeParams->kernelParams[i];
    }

    HipRemoteGraphAddNodeResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_ADD_KERNEL_NODE,
        buffer, req_size,
        &resp, sizeof(resp)
    );

    free(buffer);

    if (err == hipSuccess) {
        *pGraphNode = (hipGraphNode_t)(uintptr_t)resp.node;
    } else {
        *pGraphNode = NULL;
    }
    return err;
}

hipError_t hipGraphAddDependencies(hipGraph_t graph,
                                    const hipGraphNode_t* from,
                                    const hipGraphNode_t* to,
                                    size_t numDependencies) {
    if (!graph) {
        return hipErrorInvalidValue;
    }
    if (numDependencies == 0) {
        return hipSuccess;
    }
    if (!from || !to) {
        return hipErrorInvalidValue;
    }

    /* Build request with dependency pairs */
    size_t pairs_size = numDependencies * 2 * sizeof(uint64_t);
    size_t req_size = sizeof(HipRemoteGraphAddDependenciesRequest) + pairs_size;
    uint8_t* buffer = (uint8_t*)malloc(req_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    HipRemoteGraphAddDependenciesRequest* req = (HipRemoteGraphAddDependenciesRequest*)buffer;
    req->graph = (uint64_t)(uintptr_t)graph;
    req->num_deps = (uint32_t)numDependencies;

    /* Copy from/to pairs */
    uint64_t* pairs = (uint64_t*)(buffer + sizeof(HipRemoteGraphAddDependenciesRequest));
    for (size_t i = 0; i < numDependencies; i++) {
        pairs[i * 2] = (uint64_t)(uintptr_t)from[i];
        pairs[i * 2 + 1] = (uint64_t)(uintptr_t)to[i];
    }

    HipRemoteResponseHeader resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_ADD_DEPENDENCIES,
        buffer, req_size,
        &resp, sizeof(resp)
    );

    free(buffer);
    return err;
}

hipError_t hipGraphAddEmptyNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                 const hipGraphNode_t* pDependencies, size_t numDependencies) {
    if (!pGraphNode || !graph) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > 0 && !pDependencies) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > HIP_REMOTE_MAX_GRAPH_DEPENDENCIES) {
        return hipErrorInvalidValue;
    }

    /* Build request with dependencies */
    size_t deps_size = numDependencies * sizeof(uint64_t);
    size_t req_size = sizeof(HipRemoteGraphAddEmptyNodeRequest) + deps_size;
    uint8_t* buffer = (uint8_t*)malloc(req_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    HipRemoteGraphAddEmptyNodeRequest* req = (HipRemoteGraphAddEmptyNodeRequest*)buffer;
    req->graph = (uint64_t)(uintptr_t)graph;
    req->num_deps = (uint32_t)numDependencies;

    /* Copy dependencies */
    uint64_t* deps = (uint64_t*)(buffer + sizeof(HipRemoteGraphAddEmptyNodeRequest));
    for (size_t i = 0; i < numDependencies; i++) {
        deps[i] = (uint64_t)(uintptr_t)pDependencies[i];
    }

    HipRemoteGraphAddNodeResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_ADD_EMPTY_NODE,
        buffer, req_size,
        &resp, sizeof(resp)
    );

    free(buffer);

    if (err == hipSuccess) {
        *pGraphNode = (hipGraphNode_t)(uintptr_t)resp.node;
    } else {
        *pGraphNode = NULL;
    }
    return err;
}

hipError_t hipGraphGetNodes(hipGraph_t graph, hipGraphNode_t* nodes, size_t* numNodes) {
    if (!graph || !numNodes) {
        return hipErrorInvalidValue;
    }

    uint32_t max_nodes = nodes ? (uint32_t)*numNodes : 0;

    HipRemoteGraphGetNodesRequest req = {
        .graph = (uint64_t)(uintptr_t)graph,
        .max_nodes = max_nodes
    };

    /* Allocate response buffer for variable-length response */
    size_t resp_size = sizeof(HipRemoteGraphGetNodesResponse) + max_nodes * sizeof(uint64_t);
    uint8_t* buffer = (uint8_t*)malloc(resp_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_GET_NODES,
        &req, sizeof(req),
        buffer, resp_size
    );

    if (err == hipSuccess) {
        HipRemoteGraphGetNodesResponse* resp = (HipRemoteGraphGetNodesResponse*)buffer;
        *numNodes = resp->num_nodes;

        if (nodes && max_nodes > 0) {
            uint64_t* node_handles = (uint64_t*)(buffer + sizeof(HipRemoteGraphGetNodesResponse));
            uint32_t copy_count = (resp->num_nodes < max_nodes) ? resp->num_nodes : max_nodes;
            for (uint32_t i = 0; i < copy_count; i++) {
                nodes[i] = (hipGraphNode_t)(uintptr_t)node_handles[i];
            }
        }
    }

    free(buffer);
    return err;
}

hipError_t hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t* pRootNodes, size_t* pNumRootNodes) {
    if (!graph || !pNumRootNodes) {
        return hipErrorInvalidValue;
    }

    uint32_t max_nodes = pRootNodes ? (uint32_t)*pNumRootNodes : 0;

    HipRemoteGraphGetNodesRequest req = {
        .graph = (uint64_t)(uintptr_t)graph,
        .max_nodes = max_nodes
    };

    size_t resp_size = sizeof(HipRemoteGraphGetNodesResponse) + max_nodes * sizeof(uint64_t);
    uint8_t* buffer = (uint8_t*)malloc(resp_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_GET_ROOT_NODES,
        &req, sizeof(req),
        buffer, resp_size
    );

    if (err == hipSuccess) {
        HipRemoteGraphGetNodesResponse* resp = (HipRemoteGraphGetNodesResponse*)buffer;
        *pNumRootNodes = resp->num_nodes;

        if (pRootNodes && max_nodes > 0) {
            uint64_t* node_handles = (uint64_t*)(buffer + sizeof(HipRemoteGraphGetNodesResponse));
            uint32_t copy_count = (resp->num_nodes < max_nodes) ? resp->num_nodes : max_nodes;
            for (uint32_t i = 0; i < copy_count; i++) {
                pRootNodes[i] = (hipGraphNode_t)(uintptr_t)node_handles[i];
            }
        }
    }

    free(buffer);
    return err;
}

hipError_t hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t* from, hipGraphNode_t* to, size_t* numEdges) {
    if (!graph || !numEdges) {
        return hipErrorInvalidValue;
    }

    uint32_t max_edges = (from && to) ? (uint32_t)*numEdges : 0;

    HipRemoteGraphGetEdgesRequest req = {
        .graph = (uint64_t)(uintptr_t)graph,
        .max_edges = max_edges
    };

    size_t resp_size = sizeof(HipRemoteGraphGetEdgesResponse) + max_edges * 2 * sizeof(uint64_t);
    uint8_t* buffer = (uint8_t*)malloc(resp_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_GET_EDGES,
        &req, sizeof(req),
        buffer, resp_size
    );

    if (err == hipSuccess) {
        HipRemoteGraphGetEdgesResponse* resp = (HipRemoteGraphGetEdgesResponse*)buffer;
        *numEdges = resp->num_edges;

        if (from && to && max_edges > 0) {
            uint64_t* pairs = (uint64_t*)(buffer + sizeof(HipRemoteGraphGetEdgesResponse));
            /* Validate response doesn't exceed buffer bounds */
            size_t max_safe_edges = (resp_size - sizeof(HipRemoteGraphGetEdgesResponse)) / (2 * sizeof(uint64_t));
            uint32_t safe_num_edges = (resp->num_edges < max_safe_edges) ? resp->num_edges : (uint32_t)max_safe_edges;
            uint32_t copy_count = (safe_num_edges < max_edges) ? safe_num_edges : max_edges;
            for (uint32_t i = 0; i < copy_count; i++) {
                from[i] = (hipGraphNode_t)(uintptr_t)pairs[i * 2];
                to[i] = (hipGraphNode_t)(uintptr_t)pairs[i * 2 + 1];
            }
        }
    }

    free(buffer);
    return err;
}

hipError_t hipGraphNodeGetType(hipGraphNode_t node, hipGraphNodeType* pType) {
    if (!node || !pType) {
        return hipErrorInvalidValue;
    }

    HipRemoteGraphNodeGetTypeRequest req = {
        .node = (uint64_t)(uintptr_t)node
    };
    HipRemoteGraphNodeGetTypeResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_NODE_GET_TYPE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *pType = (hipGraphNodeType)resp.type;
    }
    return err;
}

hipError_t hipGraphDestroyNode(hipGraphNode_t node) {
    if (!node) {
        return hipErrorInvalidValue;
    }

    HipRemoteGraphDestroyNodeRequest req = {
        .node = (uint64_t)(uintptr_t)node
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_GRAPH_DESTROY_NODE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipGraphAddEventRecordNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                       const hipGraphNode_t* pDependencies, size_t numDependencies,
                                       hipEvent_t event) {
    if (!pGraphNode || !graph || !event) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > 0 && !pDependencies) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > HIP_REMOTE_MAX_GRAPH_DEPENDENCIES) {
        return hipErrorInvalidValue;
    }

    size_t deps_size = numDependencies * sizeof(uint64_t);
    size_t req_size = sizeof(HipRemoteGraphAddEventNodeRequest) + deps_size;
    uint8_t* buffer = (uint8_t*)malloc(req_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    HipRemoteGraphAddEventNodeRequest* req = (HipRemoteGraphAddEventNodeRequest*)buffer;
    req->graph = (uint64_t)(uintptr_t)graph;
    req->num_deps = (uint32_t)numDependencies;
    req->reserved = 0;
    req->event = (uint64_t)(uintptr_t)event;

    uint64_t* deps = (uint64_t*)(buffer + sizeof(HipRemoteGraphAddEventNodeRequest));
    for (size_t i = 0; i < numDependencies; i++) {
        deps[i] = (uint64_t)(uintptr_t)pDependencies[i];
    }

    HipRemoteGraphAddNodeResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_ADD_EVENT_RECORD_NODE,
        buffer, req_size,
        &resp, sizeof(resp)
    );

    free(buffer);

    if (err == hipSuccess) {
        *pGraphNode = (hipGraphNode_t)(uintptr_t)resp.node;
    } else {
        *pGraphNode = NULL;
    }
    return err;
}

hipError_t hipGraphAddEventWaitNode(hipGraphNode_t* pGraphNode, hipGraph_t graph,
                                     const hipGraphNode_t* pDependencies, size_t numDependencies,
                                     hipEvent_t event) {
    if (!pGraphNode || !graph || !event) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > 0 && !pDependencies) {
        return hipErrorInvalidValue;
    }
    if (numDependencies > HIP_REMOTE_MAX_GRAPH_DEPENDENCIES) {
        return hipErrorInvalidValue;
    }

    size_t deps_size = numDependencies * sizeof(uint64_t);
    size_t req_size = sizeof(HipRemoteGraphAddEventNodeRequest) + deps_size;
    uint8_t* buffer = (uint8_t*)malloc(req_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    HipRemoteGraphAddEventNodeRequest* req = (HipRemoteGraphAddEventNodeRequest*)buffer;
    req->graph = (uint64_t)(uintptr_t)graph;
    req->num_deps = (uint32_t)numDependencies;
    req->reserved = 0;
    req->event = (uint64_t)(uintptr_t)event;

    uint64_t* deps = (uint64_t*)(buffer + sizeof(HipRemoteGraphAddEventNodeRequest));
    for (size_t i = 0; i < numDependencies; i++) {
        deps[i] = (uint64_t)(uintptr_t)pDependencies[i];
    }

    HipRemoteGraphAddNodeResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_ADD_EVENT_WAIT_NODE,
        buffer, req_size,
        &resp, sizeof(resp)
    );

    free(buffer);

    if (err == hipSuccess) {
        *pGraphNode = (hipGraphNode_t)(uintptr_t)resp.node;
    } else {
        *pGraphNode = NULL;
    }
    return err;
}

/* ============================================================================
 * Graph Clone and Query APIs
 * ============================================================================ */

hipError_t hipGraphClone(hipGraph_t* pGraphClone, hipGraph_t originalGraph) {
    if (!pGraphClone || !originalGraph) {
        return hipErrorInvalidValue;
    }

    HipRemoteGraphCloneRequest req = {
        .original_graph = (uint64_t)(uintptr_t)originalGraph
    };

    HipRemoteGraphCloneResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_CLONE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *pGraphClone = (hipGraph_t)(uintptr_t)resp.cloned_graph;
    } else {
        *pGraphClone = NULL;
    }
    return err;
}

hipError_t hipGraphNodeGetDependencies(hipGraphNode_t node, hipGraphNode_t* pDependencies,
                                        size_t* pNumDependencies) {
    if (!node || !pNumDependencies) {
        return hipErrorInvalidValue;
    }

    HipRemoteGraphNodeGetDependenciesRequest req = {
        .node = (uint64_t)(uintptr_t)node,
        .max_nodes = pDependencies ? (uint32_t)*pNumDependencies : 0
    };

    /* Allocate buffer for response + node array */
    size_t max_nodes = pDependencies ? *pNumDependencies : 0;
    size_t resp_size = sizeof(HipRemoteGraphNodeGetDependenciesResponse) +
                       max_nodes * sizeof(uint64_t);
    uint8_t* buffer = (uint8_t*)malloc(resp_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_NODE_GET_DEPENDENCIES,
        &req, sizeof(req),
        buffer, resp_size
    );

    if (err == hipSuccess) {
        HipRemoteGraphNodeGetDependenciesResponse* resp =
            (HipRemoteGraphNodeGetDependenciesResponse*)buffer;
        *pNumDependencies = resp->num_nodes;

        if (pDependencies && resp->num_nodes > 0) {
            uint64_t* node_handles = (uint64_t*)(buffer + sizeof(*resp));
            for (size_t i = 0; i < resp->num_nodes && i < max_nodes; i++) {
                pDependencies[i] = (hipGraphNode_t)(uintptr_t)node_handles[i];
            }
        }
    }

    free(buffer);
    return err;
}

hipError_t hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t* pDependentNodes,
                                          size_t* pNumDependentNodes) {
    if (!node || !pNumDependentNodes) {
        return hipErrorInvalidValue;
    }

    HipRemoteGraphNodeGetDependenciesRequest req = {
        .node = (uint64_t)(uintptr_t)node,
        .max_nodes = pDependentNodes ? (uint32_t)*pNumDependentNodes : 0
    };

    /* Allocate buffer for response + node array */
    size_t max_nodes = pDependentNodes ? *pNumDependentNodes : 0;
    size_t resp_size = sizeof(HipRemoteGraphNodeGetDependenciesResponse) +
                       max_nodes * sizeof(uint64_t);
    uint8_t* buffer = (uint8_t*)malloc(resp_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_NODE_GET_DEPENDENT_NODES,
        &req, sizeof(req),
        buffer, resp_size
    );

    if (err == hipSuccess) {
        HipRemoteGraphNodeGetDependenciesResponse* resp =
            (HipRemoteGraphNodeGetDependenciesResponse*)buffer;
        *pNumDependentNodes = resp->num_nodes;

        if (pDependentNodes && resp->num_nodes > 0) {
            uint64_t* node_handles = (uint64_t*)(buffer + sizeof(*resp));
            for (size_t i = 0; i < resp->num_nodes && i < max_nodes; i++) {
                pDependentNodes[i] = (hipGraphNode_t)(uintptr_t)node_handles[i];
            }
        }
    }

    free(buffer);
    return err;
}

/* ============================================================================
 * Graph Execution Update APIs
 * ============================================================================ */

hipError_t hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
                               hipGraphNode_t* hErrorNode_out,
                               hipGraphExecUpdateResult* updateResult_out) {
    if (!hGraphExec || !hGraph) {
        return hipErrorInvalidValue;
    }

    HipRemoteGraphExecUpdateRequest req = {
        .graph_exec = (uint64_t)(uintptr_t)hGraphExec,
        .graph = (uint64_t)(uintptr_t)hGraph
    };

    HipRemoteGraphExecUpdateResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_EXEC_UPDATE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (updateResult_out) {
        *updateResult_out = (hipGraphExecUpdateResult)resp.update_result;
    }

    /* Error node is not returned in current implementation */
    if (hErrorNode_out) {
        *hErrorNode_out = NULL;
    }

    return err;
}

hipError_t hipGraphExecKernelNodeSetParams(hipGraphExec_t hGraphExec, hipGraphNode_t node,
                                            const hipKernelNodeParams* pNodeParams) {
    if (!hGraphExec || !node || !pNodeParams) {
        return hipErrorInvalidValue;
    }

    /* Allocate buffer for request + kernel parameters */
    size_t num_params = 0;
    if (pNodeParams->kernelParams) {
        /* Count parameters - terminated by NULL */
        while (pNodeParams->kernelParams[num_params] != NULL) {
            num_params++;
        }
    }

    size_t params_size = num_params * sizeof(uint64_t);
    size_t req_size = sizeof(HipRemoteGraphExecKernelNodeSetParamsRequest) + params_size;
    uint8_t* buffer = (uint8_t*)malloc(req_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

    HipRemoteGraphExecKernelNodeSetParamsRequest* req =
        (HipRemoteGraphExecKernelNodeSetParamsRequest*)buffer;
    req->graph_exec = (uint64_t)(uintptr_t)hGraphExec;
    req->node = (uint64_t)(uintptr_t)node;
    req->func = (uint64_t)(uintptr_t)pNodeParams->func;
    req->grid_dim_x = pNodeParams->gridDim.x;
    req->grid_dim_y = pNodeParams->gridDim.y;
    req->grid_dim_z = pNodeParams->gridDim.z;
    req->block_dim_x = pNodeParams->blockDim.x;
    req->block_dim_y = pNodeParams->blockDim.y;
    req->block_dim_z = pNodeParams->blockDim.z;
    req->shared_mem = pNodeParams->sharedMemBytes;
    req->num_params = (uint32_t)num_params;

    /* Copy kernel parameter pointers */
    if (num_params > 0) {
        uint64_t* params = (uint64_t*)(buffer + sizeof(*req));
        for (size_t i = 0; i < num_params; i++) {
            params[i] = (uint64_t)(uintptr_t)pNodeParams->kernelParams[i];
        }
    }

    HipRemoteResponseHeader resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS,
        buffer, req_size,
        &resp, sizeof(resp)
    );

    free(buffer);
    return err;
}
